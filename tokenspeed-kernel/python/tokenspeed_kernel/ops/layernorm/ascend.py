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

"""Ascend layernorm kernels."""

import torch
from tokenspeed_kernel_npu.ops.layernorm import qk_rmsnorm as _npu_qk_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import rmsnorm

__all__ = ["gated_rmsnorm", "qk_rmsnorm", "rmsnorm"]


def gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    z: torch.Tensor | None,
    eps: float,
    group_size: int | None,
    norm_before_gate: bool,
    sigmoid_gate: bool,
    fp8_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Grouped gated RMSNorm with Qwen3.5's activation-dtype boundaries.

    A group-sized weight is shared across heads. Static FP8 outputs are not
    part of the dense Ascend path and are rejected rather than discarded.
    Statistics are FP32; normalized values are cast before multiplying the
    weight, matching HF's BF16 norm and weight-product rounding.
    """
    if fp8_scale is not None:
        raise NotImplementedError("Ascend gated RMSNorm does not produce FP8")
    width = x.shape[-1]
    group_size = width if group_size is None else group_size
    if width % group_size or weight.shape not in ((width,), (group_size,)):
        raise ValueError("Gated RMSNorm group and weight dimensions do not match")
    if z is not None and z.shape != x.shape:
        raise ValueError("Gated RMSNorm gate must have the same shape as the input")
    values = x.float()
    gate = None
    if z is not None:
        gate = (
            z.float().sigmoid() if sigmoid_gate else torch.nn.functional.silu(z.float())
        )
        if not norm_before_gate:
            values = values * gate
    grouped = values.reshape(*values.shape[:-1], width // group_size, group_size)
    normed = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    # HF Qwen3_5RMSNormGated casts the normalized activation before its
    # weight multiplication. For BF16 weights, that product rounds again.
    normed = normed.to(x.dtype)
    if weight.numel() == group_size:
        normed = normed * weight
    else:
        normed = normed * weight.reshape(width // group_size, group_size)
    output = normed.reshape_as(values)
    if gate is not None and norm_before_gate:
        output = output * gate
    return output.to(x.dtype)


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    weight_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head norm; Gemma-style offset weights keep normalization in FP32."""
    if weight_offset:
        # Qwen3.5 rounds only after FP32 normalization and weight scaling.
        # Passing FP32 activations also prevents the lower-level helper from
        # rounding its reciprocal RMS and normalized activations to BF16.
        q_out, k_out = _npu_qk_rmsnorm(
            q.float(),
            k.float(),
            q_weight.float() + weight_offset,
            k_weight.float() + weight_offset,
            eps,
        )
        return q_out.to(q.dtype), k_out.to(k.dtype)
    return _npu_qk_rmsnorm(q, k, q_weight, k_weight, eps)
