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

"""Independent numerical and state-ownership checks for Ascend GDN.

The same tensor implementation can be tested on CPU; NPU parametrizations
also exercise the public dispatch and verify that outputs remain on device.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step
from tokenspeed_kernel.ops.attention.gdn.ascend import (
    ascend_causal_conv1d_prefill,
    ascend_causal_conv1d_update,
    ascend_gdn_chunk_prefill,
    ascend_gdn_decode_step,
    ascend_gdn_gating,
)

DEVICES = ["cpu"] + (
    ["npu"] if hasattr(torch, "npu") and torch.npu.is_available() else []
)


def reference(q, k, v, g, beta, state, bounds, scale, normalize):
    q, k, v, g, beta, state = [
        t.detach().cpu().double() for t in (q, k, v, g, beta, state)
    ]
    if normalize:
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q.repeat_interleave(v.shape[2] // q.shape[2], dim=2)
    k = k.repeat_interleave(v.shape[2] // k.shape[2], dim=2)
    output = torch.zeros_like(v)
    final = []
    for row, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        # Independent orientation: [H,K,V], unlike production [H,V,K].
        current = state[row].transpose(-2, -1).clone()
        for token in range(begin, end):
            qt, kt, vt = q[0, token], k[0, token], v[0, token]
            current = current * g[0, token].exp()[:, None, None]
            prediction = torch.einsum("hk,hkv->hv", kt, current)
            update = beta[0, token, :, None] * (vt - prediction)
            current = current + torch.einsum("hk,hv->hkv", kt, update)
            output[0, token] = torch.einsum("hk,hkv->hv", qt, current) * scale
        final.append(current.transpose(-2, -1))
    return output, torch.stack(final)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dims", [(16, 12), (128, 128)])
def test_chunk_scan_matches_sequential_reference(device, dtype, dims):
    torch.manual_seed(17)
    key_dim, value_dim = dims
    bounds = [0, 1, 4, 71, 201]
    q = torch.randn(1, 201, 2, key_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    # Epsilon is additive, not clamp_min. Tiny vectors reveal that distinction.
    q[:, :3] *= 1e-4
    k[:, :3] *= 1e-4
    q[:, 3] = 0
    k[:, 3] = 0
    v = torch.randn(1, 201, 4, value_dim, dtype=dtype, device=device)
    g = -torch.rand(1, 201, 4, device=device) * 2
    # Include underflowing decay; it must not introduce an inf from an upper triangle.
    g[:, 69] = -100
    beta = torch.rand(1, 201, 4, device=device).to(dtype)
    state = torch.randn(4, 4, value_dim, key_dim, device=device) * 0.05
    original = state.clone()
    prefill = gdn_chunk_prefill if device == "npu" else ascend_gdn_chunk_prefill
    result = prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=key_dim**-0.5,
        initial_state=state,
        cu_seqlens=torch.tensor(bounds, dtype=torch.int32, device=device),
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
    )
    expected, final = reference(q, k, v, g, beta, state, bounds, key_dim**-0.5, True)
    assert result.out.device.type == device
    torch.testing.assert_close(
        result.out.cpu().float(),
        expected.to(dtype).float(),
        rtol=8e-3 if dtype == torch.bfloat16 else 1e-4,
        atol=2e-3 if dtype == torch.bfloat16 else 2e-5,
    )
    torch.testing.assert_close(
        result.final_state.cpu().double(), final, rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(state, original, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_split_prefill_matches_whole_sequence(device):
    torch.manual_seed(27)
    q = torch.randn(1, 139, 2, 16, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, 139, 4, 12, dtype=torch.bfloat16, device=device)
    g = -torch.rand(1, 139, 4, device=device)
    beta = torch.rand(1, 139, 4, device=device).to(torch.bfloat16)
    initial = torch.zeros(1, 4, 12, 16, device=device)

    def run(begin, end, state):
        return ascend_gdn_chunk_prefill(
            q[:, begin:end],
            k[:, begin:end],
            v[:, begin:end],
            g[:, begin:end],
            beta[:, begin:end],
            scale=0.25,
            initial_state=state,
            cu_seqlens=torch.tensor([0, end - begin], device=device),
            qk_l2norm=True,
            output_final_state=True,
            output_h=False,
        )

    whole = run(0, 139, initial)
    first = run(0, 3, initial)
    middle = run(3, 70, first.final_state)
    last = run(70, 139, middle.final_state)
    torch.testing.assert_close(
        torch.cat((first.out, middle.out, last.out), dim=1),
        whole.out,
        rtol=8e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        last.final_state, whole.final_state, rtol=2e-4, atol=2e-5
    )


@pytest.mark.parametrize("device", DEVICES)
def test_decode_keeps_snapshots_null_and_padding_isolated(device):
    torch.manual_seed(32)
    packed = torch.randn(
        4, 1, 2 * 2 * 16 + 4 * 12 + 7, device=device, dtype=torch.bfloat16
    )
    qflat, kflat, vflat, _ = packed.split([32, 32, 48, 7], dim=-1)
    q, k, v = qflat.view(4, 1, 2, 16), kflat.view(4, 1, 2, 16), vflat.view(4, 1, 4, 12)
    pool = torch.randn(7, 4, 12, 16, device=device)
    pool[0] = 0
    original = pool.clone()
    reads = torch.tensor([4, 1, -1, 0], dtype=torch.int32, device=device)
    writes = torch.tensor([2, 5, -1, 6], dtype=torch.int32, device=device)
    a = torch.randn(4, 1, 4, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(4, device=device)
    dt_bias = torch.randn(4, device=device)
    decode = gdn_decode_step if device == "npu" else ascend_gdn_decode_step
    actual = decode(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool,
        initial_state_indices=reads,
        output_state_indices=writes,
        scale=0.25,
        use_qk_l2norm=True,
    )
    states = original[reads.clamp_min(0).long()].clone()
    states[reads <= 0] = 0
    expected, final = reference(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        ascend_gdn_gating(A_log, a[:, 0], dt_bias)[None],
        b[:, 0].cpu().float().sigmoid().to(b.dtype)[None],
        states,
        [0, 1, 2, 3, 4],
        0.25,
        True,
    )
    live = torch.tensor([0, 1, 3])
    torch.testing.assert_close(
        actual.cpu()[live].float(),
        expected.transpose(0, 1)[live].to(torch.bfloat16).float(),
        rtol=8e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        pool.cpu()[[2, 5, 6]].double(), final[live], rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(
        pool[[0, 1, 3, 4]], original[[0, 1, 3, 4]], equal_nan=True, rtol=0, atol=0
    )
    assert actual[2].count_nonzero().item() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_continuous_decode_matches_sequential_recurrence(device):
    torch.manual_seed(73)
    steps = 19
    q = torch.randn(2, steps, 2, 16, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(2, steps, 4, 12, dtype=torch.bfloat16, device=device)
    a = torch.randn(2, steps, 4, dtype=torch.bfloat16, device=device)
    b = torch.randn_like(a)
    A_log = torch.randn(4, device=device)
    dt_bias = torch.randn(4, device=device)
    pool = torch.zeros(5, 4, 12, 16, device=device)
    initial = torch.randn(2, 4, 12, 16, device=device) * 0.1
    pool[1:3] = initial
    reads = torch.tensor([1, 2], device=device)
    writes = torch.tensor([3, 4], device=device)
    values = []
    for token in range(steps):
        values.append(
            ascend_gdn_decode_step(
                q[:, token : token + 1],
                k[:, token : token + 1],
                v[:, token : token + 1],
                A_log=A_log,
                a=a[:, token : token + 1],
                dt_bias=dt_bias,
                b=b[:, token : token + 1],
                initial_state=pool,
                initial_state_indices=reads,
                output_state_indices=writes,
                scale=0.25,
                use_qk_l2norm=True,
            )
        )
        reads, writes = writes, reads
    packed = [t.reshape(1, 2 * steps, *t.shape[2:]) for t in (q, k, v)]
    g = ascend_gdn_gating(A_log, a, dt_bias).reshape(1, 2 * steps, 4)
    beta = b.cpu().float().sigmoid().to(b.dtype).reshape(1, 2 * steps, 4)
    expected, final = reference(
        *packed, g, beta, initial, [0, steps, 2 * steps], 0.25, True
    )
    actual = torch.cat(values, dim=1).reshape_as(expected)
    torch.testing.assert_close(
        actual.cpu().float(), expected.to(torch.bfloat16).float(), rtol=8e-3, atol=2e-3
    )
    torch.testing.assert_close(pool[reads].cpu().double(), final, rtol=2e-4, atol=2e-5)
    assert pool[0].count_nonzero().item() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_prefill_and_continuous_decode_use_identical_gates(device):
    torch.manual_seed(109)
    steps = 73
    q = torch.randn(1, steps, 2, 16, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, steps, 4, 12, dtype=torch.bfloat16, device=device)
    a = torch.randn(1, steps, 4, dtype=torch.bfloat16, device=device)
    b = (
        torch.linspace(-1, 1, steps * 4, device=device)
        .reshape(1, steps, 4)
        .to(torch.bfloat16)
    )
    A_log = torch.randn(4, device=device)
    dt_bias = torch.randn(4, device=device)
    initial = torch.randn(1, 4, 12, 16, device=device) * 0.1
    g = ascend_gdn_gating(A_log, a, dt_bias)
    beta = b.sigmoid()
    prefill = ascend_gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=0.25,
        initial_state=initial,
        cu_seqlens=torch.tensor([0, steps], device=device),
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
    )
    # Start from a genuine prefill, then cross the chunk boundary one token at
    # a time. The same raw b must have identical rounding on the two paths.
    prefix = 7
    first = ascend_gdn_chunk_prefill(
        q[:, :prefix],
        k[:, :prefix],
        v[:, :prefix],
        g[:, :prefix],
        beta[:, :prefix],
        scale=0.25,
        initial_state=initial,
        cu_seqlens=torch.tensor([0, prefix], device=device),
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
    )
    pool = torch.zeros(2, 4, 12, 16, device=device)
    pool[1:2] = first.final_state
    page = torch.tensor([1], device=device)
    outputs = [first.out]
    for token in range(prefix, steps):
        outputs.append(
            ascend_gdn_decode_step(
                q[:, token : token + 1],
                k[:, token : token + 1],
                v[:, token : token + 1],
                A_log=A_log,
                a=a[:, token : token + 1],
                dt_bias=dt_bias,
                b=b[:, token : token + 1],
                initial_state=pool,
                initial_state_indices=page,
                output_state_indices=None,
                scale=0.25,
                use_qk_l2norm=True,
            )
        )
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), prefill.out, rtol=8e-3, atol=2e-3
    )
    torch.testing.assert_close(pool[1:2], prefill.final_state, rtol=2e-4, atol=2e-5)
    _, final = reference(
        q,
        k,
        v,
        g,
        b.cpu().float().sigmoid().to(b.dtype),
        initial,
        [0, steps],
        0.25,
        True,
    )
    torch.testing.assert_close(pool[1:2].cpu().double(), final, rtol=2e-4, atol=2e-5)
    assert pool[0].count_nonzero().item() == 0


@dataclass(frozen=True)
class ConvMetadata:
    cu_seqlens_cpu: torch.Tensor


@pytest.mark.parametrize("device", DEVICES)
def test_causal_convolution_short_prefill_decode_and_page_ownership(device):
    torch.manual_seed(18)
    channels, width = 8, 4
    weights = torch.randn(channels, width, dtype=torch.bfloat16, device=device)
    bias = torch.randn(channels, dtype=torch.bfloat16, device=device)
    pool = torch.randn(7, channels, width - 1, dtype=torch.bfloat16, device=device)
    pool[0] = 0
    initial = pool.clone()
    x = torch.randn(channels, 8, dtype=torch.bfloat16, device=device)
    bounds = [0, 1, 6]
    output = ascend_causal_conv1d_prefill(
        x,
        weights,
        bias,
        pool,
        torch.tensor(bounds, device=device),
        torch.tensor([2, 3], device=device),
        torch.tensor([True, False], device=device),
        "silu",
        -1,
        True,
        prefill_metadata=ConvMetadata(torch.tensor(bounds)),
    )
    for row, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        history = initial[2] if row == 0 else torch.zeros_like(initial[3])
        joined = torch.cat((history, x[:, begin:end]), dim=-1)
        ref = F.silu(
            F.conv1d(
                joined.cpu().float()[None],
                weights.cpu().float()[:, None],
                bias.cpu().float(),
                groups=channels,
            ).to(torch.bfloat16)
        )[0]
        torch.testing.assert_close(
            output[:, begin:end].cpu().float(),
            ref.to(torch.bfloat16).float(),
            rtol=8e-3,
            atol=2e-3,
        )
        torch.testing.assert_close(
            pool[row + 2], joined[:, -(width - 1) :], rtol=0, atol=0
        )
    assert output[:, 6:].count_nonzero().item() == 0
    saved = pool.clone()
    token = torch.randn(4, channels, dtype=torch.bfloat16, device=device)
    actual = ascend_causal_conv1d_update(
        token,
        pool,
        weights,
        bias,
        "silu",
        None,
        torch.tensor([2, 3, -1, 0], device=device),
        None,
        None,
        torch.tensor([[4], [5], [-1], [6]], device=device),
        -1,
        True,
        parent_indices=None,
    )
    for row, (source, target) in enumerate(((2, 4), (3, 5), (-1, -1), (0, 6))):
        if source < 0:
            assert actual[row].count_nonzero().item() == 0
            continue
        history = saved[source] if source else torch.zeros_like(saved[0])
        joined = torch.cat((history, token[row, :, None]), dim=-1)
        ref = F.silu(
            F.conv1d(
                joined.cpu().float()[None],
                weights.cpu().float()[:, None],
                bias.cpu().float(),
                groups=channels,
            ).to(torch.bfloat16)
        )[0, :, 0]
        torch.testing.assert_close(
            actual[row].cpu().float(),
            ref.to(torch.bfloat16).float(),
            rtol=8e-3,
            atol=2e-3,
        )
        torch.testing.assert_close(pool[target], joined[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(pool[:4], saved[:4], equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_causal_convolution_rounds_before_silu(device):
    torch.manual_seed(39)
    channels, tokens = 32, 17
    weight = torch.randn(channels, 4, dtype=torch.bfloat16, device=device)
    x = torch.randn(channels, tokens, dtype=torch.bfloat16, device=device)
    pool = torch.randn(3, channels, 3, dtype=torch.bfloat16, device=device)
    initial = pool.clone()
    joined = torch.cat((initial[1], x), dim=-1)
    raw_reference = F.conv1d(
        joined.cpu().float()[None], weight.cpu().float()[:, None], groups=channels
    )[0]
    # Use the same device activation, while the independent grouped Conv1d
    # reference exposes the precise dtype boundary under test.
    expected = F.silu(raw_reference.to(torch.bfloat16).to(device))
    fused = F.silu(raw_reference.to(device)).to(torch.bfloat16)
    assert torch.any(
        expected != fused
    ).item(), "fixture must expose intermediate rounding"
    actual = ascend_causal_conv1d_prefill(
        x,
        weight,
        None,
        pool,
        torch.tensor([0, tokens], device=device),
        torch.tensor([1], device=device),
        torch.tensor([True], device=device),
        "silu",
        -1,
        True,
        prefill_metadata=ConvMetadata(torch.tensor([0, tokens])),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The single-token update has exactly the same convolution output dtype.
    for token in range(tokens):
        step = ascend_causal_conv1d_update(
            x[:, token][None],
            initial,
            weight,
            None,
            "silu",
            None,
            torch.tensor([1], device=device),
            None,
            None,
            torch.tensor([[1]], device=device),
            -1,
            True,
            parent_indices=None,
        )
        torch.testing.assert_close(step[0], expected[:, token], rtol=0, atol=0)
    torch.testing.assert_close(initial[1], pool[1], rtol=0, atol=0)


@pytest.mark.skipif("npu" not in DEVICES, reason="Ascend NPU unavailable")
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("sigmoid_gate", [True, False])
def test_grouped_gated_norm(norm_before_gate, sigmoid_gate):
    from tokenspeed_kernel.ops.layernorm.ascend import gated_rmsnorm

    torch.manual_seed(51)
    packed = torch.randn(7, 48, dtype=torch.bfloat16, device="npu")
    x, z = packed[:, :24], packed[:, 24:]
    weight = torch.randn(8, dtype=torch.bfloat16, device="npu")
    actual = gated_rmsnorm(
        x,
        weight,
        z=z,
        eps=1e-6,
        group_size=8,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        fp8_scale=None,
    )
    values, gates = x.cpu().double().reshape(7, 3, 8), z.cpu().double().reshape(7, 3, 8)
    gates = gates.sigmoid() if sigmoid_gate else F.silu(gates)
    if not norm_before_gate:
        values = values * gates
    expected = F.rms_norm(values, (8,), None, 1e-6).to(torch.bfloat16)
    expected = expected * weight.cpu()
    if norm_before_gate:
        expected = expected.double() * gates
    torch.testing.assert_close(
        actual.cpu().float(),
        expected.reshape(7, 24).to(torch.bfloat16).float(),
        rtol=8e-3,
        atol=2e-3,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_gated_norm_matches_hf_bf16_rounding(device):
    from tokenspeed_kernel.ops.layernorm.ascend import gated_rmsnorm
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

    torch.manual_seed(43)
    rows, heads, dim = 7, 4, 128
    packed = torch.randn(rows, 2 * heads * dim, dtype=torch.bfloat16).to(device)
    x, z = packed[:, : heads * dim], packed[:, heads * dim :]
    weight = (torch.randn(dim) * 0.5 + 1).to(dtype=torch.bfloat16, device=device)
    reference_norm = Qwen3_5RMSNormGated(dim, eps=1e-6).to(
        device=device, dtype=torch.bfloat16
    )
    reference_norm.weight.data.copy_(weight)
    with torch.no_grad():
        expected = reference_norm(x.reshape(-1, dim), z.reshape(-1, dim)).reshape_as(x)
    fused = F.rms_norm(
        x.float().reshape(rows, heads, dim), (dim,), weight.float(), 1e-6
    )
    fused = (fused.reshape_as(x) * F.silu(z.float())).to(torch.bfloat16)
    assert torch.any(
        expected != fused
    ).item(), "fixture must expose HF's intermediate rounding"
    actual = gated_rmsnorm(
        x,
        weight,
        z=z,
        eps=1e-6,
        group_size=dim,
        norm_before_gate=True,
        sigmoid_gate=False,
        fp8_scale=None,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("key_noise", [0.0, 0.001])
@pytest.mark.parametrize("beta_value", [0.5, 0.9, 1.0])
@pytest.mark.parametrize("log_decay", [0.0, -0.01])
def test_correlated_keys_do_not_amplify_triangular_solve_error(
    device, key_noise, beta_value, log_decay
):
    """Matrix-power inversion diverged on these otherwise stable recurrences."""
    torch.manual_seed(425)
    tokens, dim = 64, 128
    base = torch.randn(1, 1, 1, dim)
    k = base.expand(1, tokens, 1, dim) + key_noise * torch.randn(1, tokens, 1, dim)
    q = k.clone()
    v = torch.randn(1, tokens, 1, dim)
    g = torch.full((1, tokens, 1), log_decay)
    beta = torch.full((1, tokens, 1), beta_value)
    state = torch.randn(1, 1, dim, dim) * 0.02
    expected, final = reference(q, k, v, g, beta, state, [0, tokens], dim**-0.5, True)
    q, k, v, g, beta, state = (x.to(device) for x in (q, k, v, g, beta, state))
    prefill = gdn_chunk_prefill if device == "npu" else ascend_gdn_chunk_prefill
    actual = prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=dim**-0.5,
        initial_state=state,
        cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
    )
    assert actual.out.device.type == actual.final_state.device.type == device
    assert actual.out.isfinite().all() and actual.final_state.isfinite().all()
    torch.testing.assert_close(
        actual.out.cpu().double(), expected, rtol=1e-4, atol=2e-5
    )
    torch.testing.assert_close(
        actual.final_state.cpu().double(), final, rtol=2e-4, atol=2e-5
    )


@pytest.mark.parametrize("device", DEVICES)
def test_decode_page_interleaved_state_preserves_other_fields(device):
    torch.manual_seed(432)
    pages, heads, dim = 7, 16, 128
    backing = torch.randn(pages, 3, heads, dim, dim, device=device) * 0.02
    pool = backing[:, 1]
    pool[0].zero_()
    assert not pool.is_contiguous()
    assert pool.stride() == (3 * heads * dim * dim, dim * dim, dim, 1)
    original = backing.cpu().clone()
    q = torch.randn(3, 1, heads, dim, dtype=torch.bfloat16, device=device)
    k, v = torch.randn_like(q), torch.randn_like(q)
    a = torch.randn(3, 1, heads, dtype=torch.bfloat16, device=device)
    b = torch.randn_like(a)
    A_log = torch.randn(heads, device=device)
    dt_bias = torch.randn(heads, device=device)
    reads = torch.tensor([2, 1, -1], dtype=torch.int32, device=device)
    writes = torch.tensor([4, 5, -1], dtype=torch.int32, device=device)
    initial = original[[2, 1, 0], 1].clone()
    log_decay = ascend_gdn_gating(A_log, a[:, 0], dt_bias)
    expected, final = reference(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        log_decay[None],
        b[:, 0].sigmoid()[None],
        initial,
        [0, 1, 2, 3],
        dim**-0.5,
        True,
    )
    decode = gdn_decode_step if device == "npu" else ascend_gdn_decode_step
    actual = decode(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool,
        initial_state_indices=reads,
        output_state_indices=writes,
        scale=dim**-0.5,
        use_qk_l2norm=True,
    )
    assert actual.device.type == device
    torch.testing.assert_close(
        actual[:2].cpu().float(),
        expected.transpose(0, 1)[:2].to(torch.bfloat16).float(),
        rtol=8e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        pool[[4, 5]].cpu().double(), final[:2], rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(
        backing[:, [0, 2]].cpu(), original[:, [0, 2]], rtol=0, atol=0
    )
    torch.testing.assert_close(
        pool[[0, 1, 2, 3, 6]].cpu(), original[[0, 1, 2, 3, 6], 1], rtol=0, atol=0
    )
    assert actual[2].count_nonzero().item() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_conv_update_page_interleaved_state_preserves_other_fields(device):
    torch.manual_seed(433)
    channels = 6144
    backing = torch.randn(7, 5, channels, 3, dtype=torch.bfloat16, device=device)
    pool = backing[:, 2]
    pool[0].zero_()
    assert not pool.is_contiguous()
    assert pool.stride() == (5 * channels * 3, 3, 1)
    original = backing.cpu().clone()
    x = torch.randn(3, channels, dtype=torch.bfloat16, device=device)
    weight = torch.randn(channels, 4, dtype=torch.bfloat16, device=device)
    reads = torch.tensor([2, 1, -1], dtype=torch.int32, device=device)
    writes = torch.tensor([4, 5, -1], dtype=torch.int32, device=device)
    history = original[[2, 1, 0], 2]
    joined = torch.cat((history, x.cpu().unsqueeze(-1)), dim=-1)
    conv = F.conv1d(joined.float(), weight.cpu().float()[:, None], groups=channels)
    expected = F.silu(conv[..., 0].to(torch.bfloat16))
    actual = ascend_causal_conv1d_update(
        x,
        pool,
        weight,
        None,
        "silu",
        None,
        reads,
        None,
        None,
        writes[:, None],
        -1,
        True,
        parent_indices=None,
    )
    assert actual.device.type == device
    torch.testing.assert_close(actual[:2].cpu(), expected[:2], rtol=8e-3, atol=2e-3)
    torch.testing.assert_close(pool[[4, 5]].cpu(), joined[:2, :, 1:], rtol=0, atol=0)
    untouched = [0, 1, 3, 4]
    torch.testing.assert_close(
        backing[:, untouched].cpu(), original[:, untouched], rtol=0, atol=0
    )
    torch.testing.assert_close(
        pool[[0, 1, 2, 3, 6]].cpu(), original[[0, 1, 2, 3, 6], 2], rtol=0, atol=0
    )
    assert actual[2].count_nonzero().item() == 0
