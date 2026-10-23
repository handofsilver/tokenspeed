# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU checks of the real Ascend GemmaRMSNorm arithmetic and residual contract."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch


@pytest.fixture
def norm_class():
    # Accelerator imports cannot load on CPU. Compile the real class, replacing
    # only unavailable device metadata and its GPU-only NVFP4 capability query.
    source = (
        Path(__file__).resolve().parents[3]
        / "python/tokenspeed/runtime/layers/layernorm.py"
    )
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "GemmaRMSNorm"
    )
    namespace = dict(
        torch=torch,
        nn=torch.nn,
        ClassVar=ClassVar,
        _platform=SimpleNamespace(is_npu=True),
        _is_amd=False,
        nvfp4_copy_supported=lambda width: False,
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace["GemmaRMSNorm"]


def _inputs(norm_class, dtype, rows):
    torch.manual_seed(23)
    norm = norm_class(24, eps=1e-5).to(dtype=dtype)
    # Includes values for which (1 + weight) in BF16 loses the weight offset.
    weight = torch.linspace(-0.3, 0.3, 24).to(dtype)
    norm._weight_loader(norm.weight, weight)
    return norm, torch.randn(rows, 24).to(dtype), torch.randn(rows, 24).to(dtype)


def _reference(norm, x):
    return torch.nn.functional.rms_norm(
        x.float(), (x.shape[-1],), 1.0 + norm.weight.float(), norm.variance_epsilon
    ).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [0, 1, 5])
@pytest.mark.parametrize("use_residual", [False, True])
@torch.no_grad()
def test_ascend_gemma_norm_and_residual(norm_class, dtype, rows, use_residual):
    norm, x, residual = _inputs(norm_class, dtype, rows)
    expected_input = x + residual if use_residual else x
    result = norm(x, residual if use_residual else None)
    if use_residual:
        actual, actual_residual = result
        torch.testing.assert_close(actual_residual, expected_input, rtol=0, atol=0)
    else:
        actual = result
    torch.testing.assert_close(actual, _reference(norm, expected_input))
    assert actual.dtype == dtype


@pytest.mark.parametrize("method", ["add_norm_with_fp8", "add_norm_with_fp4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [0, 1, 5])
@torch.no_grad()
def test_ascend_unquantized_add_norm(norm_class, method, dtype, rows):
    norm, x, residual = _inputs(norm_class, dtype, rows)
    expected_residual = x + residual
    actual, quantized, actual_residual = getattr(norm, method)(x, residual, None)
    assert quantized is None
    torch.testing.assert_close(actual, _reference(norm, expected_residual))
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["add_norm_with_fp8", "add_norm_with_fp4"])
@torch.no_grad()
def test_ascend_add_norm_rejects_unsupported_quantized_output(norm_class, method):
    norm, x, residual = _inputs(norm_class, torch.bfloat16, 2)
    with pytest.raises(NotImplementedError, match="quantized output copies"):
        getattr(norm, method)(x, residual, torch.tensor([0.1]))


@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_TEST_NPU") != "1",
    reason="NPU execution is opt-in after selecting an available device",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "method", ["forward", "add_norm_with_fp8", "add_norm_with_fp4"]
)
@torch.no_grad()
def test_gemma_norm_on_actual_npu(dtype, method):
    pytest.importorskip("torch_npu")
    from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

    if not torch.npu.is_available():
        pytest.skip("no visible NPU")
    torch.manual_seed(29)
    weight = torch.randn(1024).to(dtype) * 0.3
    x_cpu = torch.randn(5, 1024).to(dtype)
    residual_cpu = torch.randn_like(x_cpu)
    expected_residual = x_cpu + residual_cpu
    expected = torch.nn.functional.rms_norm(
        expected_residual.float(), (1024,), 1.0 + weight.float(), 1e-6
    ).to(dtype)

    norm = GemmaRMSNorm(1024, eps=1e-6).to(device="npu", dtype=dtype)
    norm._weight_loader(norm.weight, weight.to("npu"))
    x, residual = x_cpu.to("npu"), residual_cpu.to("npu")
    if method == "forward":
        actual, actual_residual = norm(x, residual)
    else:
        actual, quantized, actual_residual = getattr(norm, method)(x, residual, None)
        assert quantized is None

    assert actual.device == x.device
    assert actual_residual.device == x.device
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(actual_residual.cpu(), expected_residual, rtol=0, atol=0)
