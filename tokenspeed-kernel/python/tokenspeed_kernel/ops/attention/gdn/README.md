# GDN kernels

## Ascend eager execution

The `torch_npu` solution keeps the existing Qwen3.5 model, hybrid attention
backend, scheduler, and cache pages. Tensor arithmetic runs on the NPU with
FP32 recurrent state and accumulation. Prefill processes 64-token chunks,
solving each chunk's unit lower triangular delta-rule system directly with
`torch.linalg.solve_triangular`, without constructing an inverse. This avoids
the cancellation errors of matrix-power inverses for highly correlated keys.
Decode updates one token per request. Both use
the public K-last state layout `[pages, value_heads, value_dim, key_dim]`.

The same registration provides short depthwise causal convolution. Prefill
handles packed variable-length inputs and short resumed windows; decode
reads the input page before writing the designated output page. Padding
indices do not write cache state. The scheduler continues to own the null
page, live pages, and shared prefix snapshots.

This baseline requires `--enforce-eager --disable-prefill-graph --disable-pdl`
and ordinary autoregressive generation. It rejects speculative decoding,
graph capture, and intermediate GDN checkpoints. Sequence bounds and page
indices may synchronize to the host for eager planning; model tensors and
recurrent arithmetic are never moved to the CPU. Native CANN recurrent GDN
is not used because the available API requires BF16 recurrent state.
On the tested CANN version, triangular solve runs on the NPU's device-side
AI CPU, with AI Core auxiliary kernels; it is not a host CPU fallback.

`tokenspeed-kernel/test/ascend/test_gdn.py` compares the kernels with an
independent FP64 sequential recurrence and grouped convolution reference.
It covers variable lengths across chunk boundaries, additive normalization
epsilon, extreme decay, highly correlated keys, resumed prefill, strided decode, page remapping,
padding, and grouped gated RMSNorm. Run it on an explicitly selected NPU.

## FlashInfer PDL adapters

The FlashInfer adapter retains the upstream launch geometry and runtime ABI,
while wrapping device bodies with PDL synchronization. Adapted functions and
in-memory compilation caches live in private namespaces.

Decode and prefill explicitly select FlashInfer's CuTe backend. FlashInfer
0.7.0's automatic backend selection can use Cake GDN for supported shapes,
which bypasses the PDL-wrapped device bodies. Regression tests cover both
power-of-two and other head groupings and reject calls into that alternate path.

FlashInfer 0.7.0 also persists CuTe-DSL kernels to disk. PDL artifacts use a
separate `tokenspeed_pdl_` module namespace so an ordinary kernel cannot satisfy
a PDL cache lookup, or vice versa. Cache invalidation includes all upstream
source files plus the local PDL and GDN adapter sources. Installed FlashInfer
modules and the process-wide cache configuration remain untouched.

GPU regression tests cover persistent cache hits after clearing the in-memory
cache, invalidation after source changes, and both launch orders. Runtime GDN
tests additionally check CUDA Graph dependency edges and exact replay results
for decode, MTP, BF16 state, and prefill.

## Chain verification

ReplaySSM chains use chunked verification on Blackwell for 13–16 tokens and
at least 24 (request, value-head) states. Other shapes and platforms retain
step-by-step verification. `test/runtime/test_gdn_verify_policy.py` checks this
selection and its boundaries without launching device kernels.

## Variable prefill lengths

The Triton L2-normalization fallback uses a runtime token count. The launch
grid and masked block pointers cover the exact rows, while head width and tile
size stay compile-time parameters. Consecutive agentic turns therefore reuse
one compiled kernel across different token counts instead of compiling a new
variant for each length. Tests compare irregular lengths around the usual
prefill graph buckets against an independent PyTorch result.

Compilation guards warm these kernels, then sweep token counts (and PLE
request counts) without allowing new Triton specializations. PLE covers
transitions between single-request, uniform multi-request and ragged indexing
within one compiled kernel. Those selections use runtime values so startup
warmup also covers batch layouts first seen during serving.

## Accepted-prefix replay

`triton_gdn_replay_commit` commits accepted prefixes for all layers in one
portable Gluon launch. Each CTA owns a layer, request, value head and V tile.
Keys and gates are prepared in fixed chunks, with K/V staged in shared memory.

State updates stay in FP32 until the final write, and rejected tokens are
masked during loading. Aligned state accesses are vectorized, with fallbacks
for unaligned rows and arbitrary row strides. Batch size, window length and
payload layer stride remain runtime scalars, so serving shape changes reuse
the compiled kernel.
