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

"""Ascend Q/K Gemma normalization against the installed HF Qwen3.5 module."""

import os

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.ascend import qk_rmsnorm

DEVICES = ["cpu"] + (["npu"] if os.getenv("TOKENSPEED_TEST_NPU") == "1" else [])


def _hf_reference(x, weight, eps):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    norm = Qwen3_5RMSNorm(weight.numel(), eps=eps).to(weight.dtype)
    norm.weight.data.copy_(weight.cpu())
    with torch.no_grad():
        return norm(x.cpu().reshape(-1, weight.numel())).reshape(x.shape)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["flat", "strided", "heads", "empty"])
def test_qk_gemma_norm_matches_hf(device, dtype, layout):
    torch.manual_seed(420)
    tokens, head_dim = (0 if layout == "empty" else 17), 256
    packed = torch.randn(tokens, 10 * head_dim + 7, dtype=dtype).to(device)
    q = packed[:, : 8 * head_dim]
    k = packed[:, 8 * head_dim : 10 * head_dim]
    if layout != "strided":
        q, k = q.contiguous(), k.contiguous()
    else:
        assert not q.is_contiguous() and not k.is_contiguous()
    if layout == "heads":
        q, k = q.reshape(tokens, 8, head_dim), k.reshape(tokens, 2, head_dim)
    # Small Gemma weights make the loss from rounding (1 + weight) observable.
    q_weight = (torch.randn(head_dim) * 0.1).to(dtype)
    k_weight = (torch.randn(head_dim) * 0.1).to(dtype)
    q_weight[0], k_weight[0] = 2**-9, 2**-9
    expected_q = _hf_reference(q, q_weight, 1e-6)
    expected_k = _hf_reference(k, k_weight, 1e-6)
    q, k, q_weight, k_weight = (x.to(device) for x in (q, k, q_weight, k_weight))
    original_q, original_k = q.clone(), k.clone()
    actual_q, actual_k = qk_rmsnorm(q, k, q_weight, k_weight, 1e-6, weight_offset=1.0)
    for actual, expected in ((actual_q, expected_q), (actual_k, expected_k)):
        assert actual.device.type == device
        assert actual.dtype == dtype and actual.is_contiguous()
        if device == "cpu":
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            torch.testing.assert_close(
                actual.cpu(),
                expected,
                rtol=8e-3 if dtype == torch.bfloat16 else 2e-6,
                atol=2e-3 if dtype == torch.bfloat16 else 2e-6,
            )
    torch.testing.assert_close(q, original_q, rtol=0, atol=0)
    torch.testing.assert_close(k, original_k, rtol=0, atol=0)
