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

"""Qwen3.5's 8-Q/2-KV, head-dimension-256 MHA on actual Ascend hardware."""

import math

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend MHA tests require an NPU"
)


def _reference(q, k, v, *, causal, scale=None):
    """Independent CPU FP64 GQA with suffix-aligned causal positions."""
    q, k, v = (x.cpu().double() for x in (q, k, v))
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k)
    scores *= 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    if causal:
        mask = (
            torch.arange(k.shape[0])[None, :]
            > (k.shape[0] - q.shape[0] + torch.arange(q.shape[0]))[:, None]
        )
        scores.masked_fill_(mask, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).float()


def _live_rows(cache, table, batch, length):
    page_count = (length + cache.shape[1] - 1) // cache.shape[1]
    pages = table.cpu()[batch, :page_count].long()
    return cache.cpu()[pages].reshape(-1, 2, 256)[:length]


def _assert_close(output, expected):
    assert output.device.type == "npu"
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output.cpu().float(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("scale", [None, 0.11])
def test_head256_varlen_prefill(scale):
    torch.manual_seed(410)
    offsets = [0, 3, 3, 132]
    # Strided inputs exercise the model's split projection/gate layout.
    q = torch.randn(132, 8, 512, dtype=torch.bfloat16, device="npu")[..., ::2]
    k = torch.randn(132, 2, 512, dtype=torch.bfloat16, device="npu")[..., ::2]
    v = torch.randn_like(k)
    cumulative = torch.tensor(offsets, dtype=torch.int32, device="npu")
    output = mha_prefill(q, k, v, cumulative, offsets, 129, softmax_scale=scale)
    expected = torch.cat(
        [
            _reference(q[a:b], k[a:b], v[a:b], causal=True, scale=scale)
            for a, b in zip(offsets, offsets[1:])
            if b > a
        ]
    )
    _assert_close(output, expected)


def test_head256_empty_prefill():
    q = torch.empty(0, 8, 256, dtype=torch.bfloat16, device="npu")
    k = torch.empty(0, 2, 256, dtype=torch.bfloat16, device="npu")
    cumulative = torch.zeros(2, dtype=torch.int32, device="npu")
    output = mha_prefill(q, k, k, cumulative, [0, 0], 0)
    assert output.shape == q.shape
    assert output.device == q.device


def test_head256_single_query_prefill():
    q = torch.randn(1, 8, 256, dtype=torch.bfloat16, device="npu")
    k = torch.randn(1, 2, 256, dtype=torch.bfloat16, device="npu")
    v = torch.randn_like(k)
    cumulative = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    output = mha_prefill(q, k, v, cumulative, [0, 1], 1)
    _assert_close(output, _reference(q, k, v, causal=True))


@pytest.mark.parametrize("q_length", [1, 2])
def test_head256_paged_extend_long_cache_short_query(q_length):
    # A 512-token chunk followed by one token enters CANN's decode tiler.
    # Also test two queries to ensure their right-aligned mask is retained.
    torch.manual_seed(414)
    kv_length = 513
    q = torch.randn(q_length, 8, 256, dtype=torch.bfloat16, device="npu")
    q[0].zero_()
    k_cache = torch.randn(8, 128, 2, 256, dtype=torch.bfloat16, device="npu")
    v_cache = torch.randn_like(k_cache)
    page_ids = torch.randperm(8)[:5].tolist()
    # Amplify the last token so dropping the two-query causal mask is detected.
    v_cache[page_ids[-1], 0].fill_(100)
    table = torch.tensor([page_ids + [2**30] * 3], dtype=torch.int32, device="npu")
    output = mha_extend_with_kvcache(
        q,
        torch.tensor([0, q_length], dtype=torch.int32, device="npu"),
        torch.tensor([0, kv_length], dtype=torch.int32, device="npu"),
        k_cache,
        v_cache,
        table,
        torch.tensor([kv_length], dtype=torch.int32, device="npu"),
        q_length,
        1024,
        is_causal=True,
    )
    expected = _reference(
        q,
        _live_rows(k_cache, table, 0, kv_length),
        _live_rows(v_cache, table, 0, kv_length),
        causal=True,
    )
    _assert_close(output, expected)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale", [None, 0.11])
def test_head256_paged_extend(causal, scale):
    torch.manual_seed(411)
    offsets = [0, 2, 2, 131]
    lengths = [5, 0, 259]
    q = torch.randn(131, 8, 256, dtype=torch.bfloat16, device="npu")
    k_cache = torch.randn(6, 128, 2, 256, dtype=torch.bfloat16, device="npu")
    v_cache = torch.randn_like(k_cache)
    original_k, original_v = k_cache.cpu().clone(), v_cache.cpu().clone()
    invalid = 2**30
    table = torch.tensor(
        [[4, invalid, invalid], [invalid, invalid, invalid], [3, 1, 0]],
        dtype=torch.int32,
        device="npu",
    )
    output = mha_extend_with_kvcache(
        q,
        torch.tensor(offsets, dtype=torch.int32, device="npu"),
        torch.tensor([0, 5, 5, 264], dtype=torch.int32, device="npu"),
        k_cache,
        v_cache,
        table,
        torch.tensor(lengths, dtype=torch.int32, device="npu"),
        129,
        259,
        is_causal=causal,
        softmax_scale=scale,
    )
    expected = torch.cat(
        [
            _reference(
                q[a:b],
                _live_rows(k_cache, table, batch, lengths[batch]),
                _live_rows(v_cache, table, batch, lengths[batch]),
                causal=causal,
                scale=scale,
            )
            for batch, (a, b) in enumerate(zip(offsets, offsets[1:]))
            if b > a
        ]
    )
    _assert_close(output, expected)
    torch.testing.assert_close(k_cache.cpu(), original_k, atol=0, rtol=0)
    torch.testing.assert_close(v_cache.cpu(), original_v, atol=0, rtol=0)


@pytest.mark.parametrize("scale", [None, 0.11])
def test_head256_paged_decode(scale):
    torch.manual_seed(412)
    q = torch.randn(3, 8, 256, dtype=torch.bfloat16, device="npu")
    k_cache = torch.randn(6, 128, 2, 256, dtype=torch.bfloat16, device="npu")
    v_cache = torch.randn_like(k_cache)
    invalid = 2**30
    table = torch.tensor(
        [[5, invalid, invalid], [3, 1, invalid], [4, 0, 2]],
        dtype=torch.int32,
        device="npu",
    )
    lengths = [1, 139, 259]
    output = mha_decode_with_kvcache(
        q,
        k_cache,
        v_cache,
        table,
        torch.tensor(lengths, dtype=torch.int32, device="npu"),
        259,
        softmax_scale=scale,
    )
    expected = torch.cat(
        [
            _reference(
                q[b : b + 1],
                _live_rows(k_cache, table, b, length),
                _live_rows(v_cache, table, b, length),
                causal=False,
                scale=scale,
            )
            for b, length in enumerate(lengths)
        ]
    )
    _assert_close(output, expected)


def test_head256_prefill_extend_decode_agree():
    torch.manual_seed(413)
    total, prefix, chunk_end = 143, 7, 141
    q = torch.randn(total, 8, 256, dtype=torch.bfloat16, device="npu")
    k = torch.randn(total, 2, 256, dtype=torch.bfloat16, device="npu")
    v = torch.randn_like(k)
    full = mha_prefill(
        q, k, v, torch.tensor([0, total], device="npu"), [0, total], total
    )
    parts = [
        mha_prefill(
            q[:prefix],
            k[:prefix],
            v[:prefix],
            torch.tensor([0, prefix], device="npu"),
            [0, prefix],
            prefix,
        )
    ]
    k_cache = torch.zeros(3, 128, 2, 256, dtype=torch.bfloat16, device="npu")
    v_cache = torch.zeros_like(k_cache)
    k_cache[2], v_cache[2] = k[:128], v[:128]
    k_cache[0, : total - 128], v_cache[0, : total - 128] = k[128:], v[128:]
    table = torch.tensor([[2, 0, 2**30]], dtype=torch.int32, device="npu")
    parts.append(
        mha_extend_with_kvcache(
            q[prefix:chunk_end],
            torch.tensor([0, chunk_end - prefix], dtype=torch.int32, device="npu"),
            torch.tensor([0, chunk_end], dtype=torch.int32, device="npu"),
            k_cache,
            v_cache,
            table,
            torch.tensor([chunk_end], dtype=torch.int32, device="npu"),
            chunk_end - prefix,
            chunk_end,
            is_causal=True,
        )
    )
    for position in range(chunk_end, total):
        parts.append(
            mha_decode_with_kvcache(
                q[position : position + 1],
                k_cache,
                v_cache,
                table,
                torch.tensor([position + 1], dtype=torch.int32, device="npu"),
                total,
            )
        )
    combined = torch.cat(parts)
    _assert_close(combined, _reference(q, k, v, causal=True))
    torch.testing.assert_close(combined, full, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "option", [{"window_left": 8}, {"logit_cap": 1.0}, {"return_lse": True}]
)
def test_head256_rejects_unsupported_options(option):
    q = torch.empty(1, 8, 256, dtype=torch.bfloat16, device="npu")
    k = torch.empty(1, 2, 256, dtype=torch.bfloat16, device="npu")
    with pytest.raises(NotImplementedError):
        mha_prefill(q, k, k, torch.tensor([0, 1], device="npu"), [0, 1], 1, **option)


@pytest.mark.parametrize("mode", ["decode", "extend"])
@pytest.mark.parametrize("cache_length", [129, 144, 513])
def test_head256_page_interleaved_kv_cache(mode, cache_length):
    torch.manual_seed(430)
    # CacheArena uses an explicit page stride over interleaved layer fields.
    # Both K and V are non-contiguous, with distinct storage offsets.
    backing = torch.randn(8, 5, 128, 2, 256, dtype=torch.bfloat16, device="npu")
    k_cache, v_cache = backing[:, 1], backing[:, 3]
    assert not k_cache.is_contiguous() and not v_cache.is_contiguous()
    assert k_cache.stride() == v_cache.stride() == (5 * 128 * 512, 512, 256, 1)
    assert k_cache.flatten(2).stride(0) == 5 * 128 * 512
    original = backing.cpu().clone()
    live_pages = (cache_length + 127) // 128
    indices = torch.randperm(8)[:live_pages].tolist()
    table = torch.tensor(
        [indices + [2**30] * (8 - live_pages)], dtype=torch.int32, device="npu"
    )
    q_length = 1 if mode == "decode" else 3
    q = torch.randn(q_length, 8, 256, dtype=torch.bfloat16, device="npu")
    lengths = torch.tensor([cache_length], dtype=torch.int32, device="npu")

    def run(k, v):
        if mode == "decode":
            return mha_decode_with_kvcache(q, k, v, table, lengths, 1024)
        return mha_extend_with_kvcache(
            q,
            torch.tensor([0, q_length], dtype=torch.int32, device="npu"),
            torch.tensor([0, cache_length], dtype=torch.int32, device="npu"),
            k,
            v,
            table,
            lengths,
            q_length,
            1024,
            is_causal=True,
        )

    actual = run(k_cache, v_cache)
    expected = _reference(
        q,
        _live_rows(k_cache, table, 0, cache_length),
        _live_rows(v_cache, table, 0, cache_length),
        causal=True,
    )
    _assert_close(actual, expected)
    torch.testing.assert_close(actual, run(k_cache.contiguous(), v_cache.contiguous()))
    torch.testing.assert_close(backing.cpu(), original, rtol=0, atol=0)
