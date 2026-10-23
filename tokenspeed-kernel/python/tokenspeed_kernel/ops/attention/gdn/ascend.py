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

"""Eager Ascend Gated DeltaNet and short convolution kernels.

All tensor arithmetic stays on the input device. The chunk scan solves its
unit lower triangular delta-rule system directly, avoiding unstable powers
and explicit inversion when neighboring keys are highly correlated.
Only sequence bounds and page indices are read on the host. Graph capture,
intermediate checkpoints, and speculative/tree updates are rejected.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_ASCEND = CapabilityRequirement(vendors=frozenset({"ascend"}))
_DTYPES = {torch.bfloat16, torch.float16, torch.float32}
_CHUNK_SIZE = 64


def _check_eager(x: torch.Tensor) -> None:
    if x.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError("Ascend GDN requires eager execution (--enforce-eager)")


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def _expand_heads(x: torch.Tensor, value_heads: int) -> torch.Tensor:
    if value_heads % x.shape[-2]:
        raise ValueError("GDN value heads must be divisible by query/key heads")
    return x.repeat_interleave(value_heads // x.shape[-2], dim=-2)


def ascend_gdn_gating(A_log, a, dt_bias):
    """Form log decay in FP32 without narrowing the learned decay parameters."""
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())


def _chunk_scan(q, k, v, g, beta, state, scale):
    """One sequence, [T,H,D] inputs and an FP32 K-last [H,V,K] state."""
    outputs = []
    for begin in range(0, q.shape[0], _CHUNK_SIZE):
        end = min(begin + _CHUNK_SIZE, q.shape[0])
        qc = q[begin:end].transpose(0, 1) * scale
        kc = k[begin:end].transpose(0, 1)
        vc = v[begin:end].transpose(0, 1)
        bc = beta[begin:end].transpose(0, 1).unsqueeze(-1)
        gc = g[begin:end].transpose(0, 1).cumsum(-1)
        length = end - begin
        lower = torch.ones((length, length), dtype=torch.bool, device=q.device).tril()
        differences = gc.unsqueeze(-1) - gc.unsqueeze(-2)
        decay = differences.masked_fill(~lower, -torch.inf).exp()
        eye = torch.eye(length, device=q.device, dtype=torch.float32)
        lower_system = (bc * (kc @ kc.transpose(-2, -1)) * decay).tril(-1)
        initial_projection = kc @ state.transpose(-2, -1)
        rhs = bc * (vc - gc.exp().unsqueeze(-1) * initial_projection)
        # Nilpotent matrix powers can still grow enormously and cancel in
        # FP32. A direct triangular solve avoids constructing that inverse.
        delta = torch.linalg.solve_triangular(
            eye + lower_system, rhs, upper=False, unitriangular=True
        )
        output = gc.exp().unsqueeze(-1) * (qc @ state.transpose(-2, -1))
        output = output + ((qc @ kc.transpose(-2, -1)) * decay) @ delta
        outputs.append(output.transpose(0, 1))
        tail_decay = (gc[:, -1:] - gc).exp()
        state = gc[:, -1].exp()[:, None, None] * state
        state = state + delta.transpose(-2, -1) @ (tail_decay.unsqueeze(-1) * kc)
    if not outputs:
        return v.new_empty(v.shape), state
    return torch.cat(outputs, dim=0), state


@register_kernel(
    "attention",
    "gdn_chunk_prefill",
    name="ascend_gdn_chunk_prefill",
    solution="torch_npu",
    capability=_ASCEND,
    signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
    priority=Priority.PORTABLE,
    traits={"output_h": frozenset({False}), "qk_l2norm": frozenset({False, True})},
)
def ascend_gdn_chunk_prefill(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale,
    initial_state,
    cu_seqlens,
    qk_l2norm,
    output_final_state,
    output_h,
):
    """GDN prefill with FP32 accumulation; see the public GDN op contract."""
    _check_eager(q)
    if output_h:
        raise NotImplementedError("Ascend GDN does not expose intermediate checkpoints")
    if q.shape[0] != 1 or k.shape[:2] != q.shape[:2] or v.shape[:2] != q.shape[:2]:
        raise ValueError("Ascend GDN expects varlen-packed [1,T,H,D] inputs")
    bounds = cu_seqlens.detach().cpu().tolist()
    if len(bounds) != initial_state.shape[0] + 1 or bounds[0] != 0:
        raise ValueError("GDN bounds must match the recurrent state batch")
    if bounds[-1] > q.shape[1] or any(b < a for a, b in zip(bounds, bounds[1:])):
        raise ValueError("GDN bounds must be ordered within the input extent")
    qf = _l2norm(q) if qk_l2norm else q.float()
    kf = _l2norm(k) if qk_l2norm else k.float()
    qf = _expand_heads(qf, v.shape[2])[0]
    kf = _expand_heads(kf, v.shape[2])[0]
    vf, gf, bf = v[0].float(), g[0].float(), beta[0].float()
    out = torch.zeros_like(v)
    final = []
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    for row, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        values, state = _chunk_scan(
            qf[begin:end],
            kf[begin:end],
            vf[begin:end],
            gf[begin:end],
            bf[begin:end],
            initial_state[row].float(),
            scale,
        )
        out[0, begin:end] = values.to(v.dtype)
        final.append(state)
    final_state = (
        torch.stack(final).to(initial_state.dtype) if output_final_state else None
    )
    return GdnChunkPrefillResult(out=out, final_state=final_state)


@register_kernel(
    "attention",
    "gdn_decode_step",
    name="ascend_gdn_decode_step",
    solution="torch_npu",
    capability=_ASCEND,
    signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
    priority=Priority.PORTABLE,
    traits={"strided_qkv": frozenset({True})},
)
def ascend_gdn_decode_step(
    q,
    k,
    v,
    *,
    A_log,
    a,
    dt_bias,
    b,
    initial_state,
    initial_state_indices,
    scale,
    output_state_indices,
    use_qk_l2norm,
):
    """One decode step, reading snapshots before writing separate live pages.

    Negative ids are padding and may not be written. The runtime owns its
    reserved zero page and supplies distinct positive destination pages.
    """
    _check_eager(q)
    if q.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1:
        raise ValueError("Ascend GDN decode supports one token per request")
    reads = initial_state_indices.long()
    writes = reads if output_state_indices is None else output_state_indices.long()
    if reads.shape != (q.shape[0],) or writes.shape != reads.shape:
        raise ValueError("GDN decode needs one read/write index per request")
    qf = _l2norm(q) if use_qk_l2norm else q.float()
    kf = _l2norm(k) if use_qk_l2norm else k.float()
    qf = _expand_heads(qf, v.shape[2])[:, 0]
    kf = _expand_heads(kf, v.shape[2])[:, 0]
    state = initial_state.index_select(0, reads.clamp_min(0)).float()
    state = state.masked_fill((reads < 0)[:, None, None, None], 0)
    log_decay = -A_log.float().exp() * F.softplus(a[:, 0].float() + dt_bias.float())
    state = state * log_decay.exp()[..., None, None]
    prediction = (state * kf.unsqueeze(-2)).sum(-1)
    # Match prefill and the checkpoint reference: beta is rounded to the
    # projection dtype before the recurrence accumulates it in FP32.
    delta = b[:, 0].sigmoid().float().unsqueeze(-1) * (v[:, 0].float() - prediction)
    state = state + delta.unsqueeze(-1) * kf.unsqueeze(-2)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    output = (state * qf.unsqueeze(-2)).sum(-1) * scale
    output = output.masked_fill((reads < 0)[:, None, None], 0)
    live = (reads >= 0) & (writes >= 0)
    initial_state.index_copy_(0, writes[live], state[live].to(initial_state.dtype))
    return output[:, None].to(q.dtype)


def _conv_activation(x, activation):
    if activation in ("silu", "swish", True):
        return F.silu(x)
    if activation in (None, False):
        return x
    raise ValueError(f"Unsupported causal-convolution activation: {activation}")


def _channel_major(state, weight):
    if state.shape[1:] == (weight.shape[0], weight.shape[1] - 1):
        return state
    if state.shape[1:] == (weight.shape[1] - 1, weight.shape[0]):
        return state.transpose(1, 2)
    raise ValueError("Causal convolution state does not match channels and width")


@register_kernel(
    "attention",
    "causal_conv1d_prefill",
    name="ascend_causal_conv1d_prefill",
    solution="torch_npu",
    capability=_ASCEND,
    signatures=format_signatures("x", "dense", _DTYPES),
    priority=Priority.PORTABLE + 1,
)
def ascend_causal_conv1d_prefill(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    activation,
    pad_slot_id,
    validate_data,
    *,
    prefill_metadata,
):
    """Depthwise causal convolution over packed [C,T] input; update live windows."""
    _check_eager(x)
    state = _channel_major(conv_states, weight)
    bounds = prefill_metadata.cu_seqlens_cpu
    bounds = (query_start_loc if bounds is None else bounds).detach().cpu().tolist()
    count = len(bounds) - 1
    indices = (
        torch.arange(count, device=x.device) if cache_indices is None else cache_indices
    ).long()
    if indices.shape != (count,) or bounds[0] != 0 or bounds[-1] > x.shape[-1]:
        raise ValueError("Convolution bounds and indices do not match packed input")
    index_values = indices.detach().cpu().tolist()
    if validate_data and any(
        i != pad_slot_id and not 0 <= i < state.shape[0] for i in index_values
    ):
        raise ValueError("Convolution cache index is out of bounds")
    history = state.index_select(0, indices.clamp_min(0)).clone()
    flags = (
        torch.zeros(count, dtype=torch.bool, device=x.device)
        if has_initial_state is None
        else has_initial_state
    )
    history.masked_fill_(~flags[:, None, None], 0)
    out = torch.zeros_like(x)
    new_windows = history.clone()
    width = weight.shape[1]
    if width < 2:
        raise ValueError("Ascend causal convolution requires width >= 2")
    for row, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        if end < begin:
            raise ValueError("Convolution sequence bounds must be ordered")
        if end == begin or index_values[row] == pad_slot_id:
            continue
        joined = torch.cat((history[row], x[:, begin:end]), dim=-1)
        values = torch.zeros(
            (x.shape[0], end - begin), dtype=torch.float32, device=x.device
        )
        for offset in range(width):
            values += (
                joined[:, offset : offset + end - begin].float()
                * weight[:, offset : offset + 1].float()
            )
        if bias is not None:
            values += bias.float().unsqueeze(-1)
        # HF's BF16 Conv1d rounds its output before the separate SiLU op.
        out[:, begin:end] = _conv_activation(values.to(x.dtype), activation).to(x.dtype)
        new_windows[row] = joined[:, -(width - 1) :]
    live = (indices != pad_slot_id) & (indices >= 0)
    state.index_copy_(0, indices[live], new_windows[live])
    return out


@register_kernel(
    "attention",
    "causal_conv1d_update",
    name="ascend_causal_conv1d_update",
    solution="torch_npu",
    capability=_ASCEND,
    signatures=format_signatures("x", "dense", _DTYPES),
    priority=Priority.PORTABLE + 1,
    traits={"tree": frozenset({False})},
)
def ascend_causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias,
    activation,
    cache_seqlens,
    conv_state_indices,
    num_accepted_tokens,
    intermediate_conv_window,
    output_state_indices,
    pad_slot_id,
    validate_data,
    *,
    parent_indices,
):
    """One-token depthwise convolution preserving the input snapshot and null page."""
    _check_eager(x)
    if any(
        value is not None
        for value in (
            cache_seqlens,
            num_accepted_tokens,
            intermediate_conv_window,
            parent_indices,
        )
    ):
        raise NotImplementedError(
            "Ascend causal convolution supports ordinary eager decode only"
        )
    if x.ndim != 2:
        raise NotImplementedError(
            "Ascend causal convolution requires one token per request"
        )
    state = _channel_major(conv_state, weight)
    reads = (
        torch.arange(x.shape[0], device=x.device)
        if conv_state_indices is None
        else conv_state_indices
    ).long()
    writes = (
        reads
        if output_state_indices is None
        else output_state_indices.reshape(-1).long()
    )
    if reads.shape != (x.shape[0],) or writes.shape != reads.shape:
        raise ValueError("Convolution decode needs one read/write index per request")
    if validate_data:
        for indices in (reads, writes):
            if any(
                i != pad_slot_id and not 0 <= i < state.shape[0]
                for i in indices.detach().cpu().tolist()
            ):
                raise ValueError("Convolution cache index is out of bounds")
    history = state.index_select(0, reads.clamp_min(0)).clone()
    history.masked_fill_((reads == pad_slot_id)[:, None, None], 0)
    joined = torch.cat((history, x.unsqueeze(-1)), dim=-1)
    values = (joined.float() * weight.float().unsqueeze(0)).sum(-1)
    if bias is not None:
        values += bias.float()
    values = _conv_activation(values.to(x.dtype), activation).to(x.dtype)
    live = (reads != pad_slot_id) & (writes != pad_slot_id) & (writes >= 0)
    state.index_copy_(0, writes[live], joined[live, :, 1:])
    return values.masked_fill((reads == pad_slot_id)[:, None], 0)
