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

"""Ascend multi-head attention kernels."""

from __future__ import annotations

import math

import torch
import torch_npu

_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _causal_mask(device: torch.device) -> torch.Tensor:
    mask = _CAUSAL_MASKS.get(device)
    if mask is None:
        mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1
        )
        _CAUSAL_MASKS[device] = mask
    return mask


def _scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def _check_options(
    *,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
) -> None:
    if window_left >= 0:
        raise NotImplementedError("Ascend MHA does not support sliding windows")
    if logit_cap:
        raise NotImplementedError("Ascend MHA does not support logit caps")
    if sinks is not None:
        raise NotImplementedError("Ascend MHA does not support attention sinks")
    if return_lse:
        raise NotImplementedError("Ascend MHA does not return LSE")


def mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run causal variable-length MHA over uncached K/V."""
    del cu_seqlens, max_seqlen
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if q.shape[-1] == 256:
        # CANN 9's TND path only accepts head dimensions 64/128/192.
        # BSH supports 256; dispatch each unpadded request separately.
        output = torch.empty_like(q)
        for start, end in zip(cu_seqlens_cpu, cu_seqlens_cpu[1:]):
            length = end - start
            if length == 0:
                continue
            # A single query has no future K/V positions. CANN dispatches it
            # through decode tiling, which rejects the compressed causal mask.
            use_causal_mask = length > 1
            request_output, _ = torch_npu.npu_fused_infer_attention_score(
                q[start:end].reshape(1, length, -1).contiguous(),
                k[start:end].reshape(1, length, -1).contiguous(),
                v[start:end].reshape(1, length, -1).contiguous(),
                atten_mask=_causal_mask(q.device) if use_causal_mask else None,
                actual_seq_lengths=[length],
                actual_seq_lengths_kv=[length],
                num_heads=q.shape[1],
                num_key_value_heads=k.shape[1],
                scale=_scale(q, softmax_scale),
                input_layout="BSH",
                sparse_mode=2 if use_causal_mask else 0,
            )
            output[start:end] = request_output.reshape_as(q[start:end])
        return output

    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        atten_mask=_causal_mask(q.device),
        actual_seq_lengths=cu_seqlens_cpu[1:],
        actual_seq_lengths_kv=cu_seqlens_cpu[1:],
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=_scale(q, softmax_scale),
        input_layout="TND",
        sparse_mode=2,
    )
    return output


def mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run variable-length MHA over a paged K/V cache."""
    del cu_seqlens_kv, max_seqlen_q, max_seqlen_k, enable_pdl
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    if q.shape[-1] == 256:
        if torch.npu.is_current_stream_capturing():
            raise NotImplementedError(
                "Ascend head_dim=256 paged extend requires eager execution"
            )
        # BSH uses per-request lengths, unlike TND's cumulative Q offsets.
        # Only this small metadata crosses to the CPU; Q/K/V stay on the NPU.
        offsets = cu_seqlens_q.cpu().tolist()
        lengths = cache_seqlens.cpu().tolist()
        output = torch.empty_like(q)
        for batch, (start, end) in enumerate(zip(offsets, offsets[1:])):
            length = end - start
            if length == 0:
                continue
            # Queries align to the cache suffix, so the one-query case can
            # attend to every visible key and requires no causal mask.
            use_causal_mask = is_causal and length > 1
            request_output, _ = torch_npu.npu_fused_infer_attention_score(
                q[start:end].reshape(1, length, -1).contiguous(),
                k_cache.flatten(2),
                v_cache.flatten(2),
                atten_mask=_causal_mask(q.device) if use_causal_mask else None,
                actual_seq_lengths=[length],
                actual_seq_lengths_kv=[lengths[batch]],
                block_table=page_table[batch : batch + 1],
                num_heads=q.shape[1],
                num_key_value_heads=k_cache.shape[2],
                scale=_scale(q, softmax_scale),
                input_layout="BSH",
                sparse_mode=3 if use_causal_mask else 0,
                block_size=k_cache.shape[1],
            )
            output[start:end] = request_output.reshape_as(q[start:end])
        return output

    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k_cache.flatten(2),
        v_cache.flatten(2),
        atten_mask=_causal_mask(q.device) if is_causal else None,
        actual_seq_lengths=cu_seqlens_q[1:],
        actual_seq_lengths_kv=cache_seqlens,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="TND",
        sparse_mode=3 if is_causal else 0,
        block_size=k_cache.shape[1],
    )
    return output


def mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run fixed-shape, graph-capturable decode over a paged K/V cache."""
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if max_seqlen_q != 1:
        raise NotImplementedError("Ascend MHA decode supports one query per request")
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    del max_seqlen_k, enable_pdl
    batch_size = cache_seqlens.shape[0]
    actual_seq_lengths_kv = (
        [1] * batch_size if torch.npu.is_current_stream_capturing() else cache_seqlens
    )
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q.reshape(batch_size, 1, -1),
        k_cache.flatten(2),
        v_cache.flatten(2),
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="BSH",
        block_size=k_cache.shape[1],
    )
    return output.reshape_as(q)


__all__ = [
    "mha_decode_with_kvcache",
    "mha_extend_with_kvcache",
    "mha_prefill",
]
