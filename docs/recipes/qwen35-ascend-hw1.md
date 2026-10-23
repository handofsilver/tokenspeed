# Qwen3.5-0.8B 的 Ascend 910C 适配

本实验基于 TokenSpeed `cf898936350f46ac6d0443b154739beaf51fe8a0`，将
`Qwen/Qwen3.5-0.8B` 的文本推理接入单芯片 Ascend 910C。模型 revision 为
`2fc06364715b967f1860aea9cf38778875588b17`。运行入口和环境安装分别见
[实验脚本](../../examples/aics_qwen35/README.md)与
[环境说明](../../examples/aics_qwen35/environment-notes.md)。

## 模型配置

文本网络共 24 层，其中 18 层为 Gated DeltaNet（GDN），每四层的最后一层为
full attention。hidden size 为 1024，词表大小为 248320，dense MLP 中间维度为
3584。本次使用未量化 BF16 权重、FP32 GDN recurrent state，开启
`language_model_only=True`，采用 eager、greedy、batch 1 推理。

| 部分 | 单芯片配置 |
| --- | --- |
| GDN | Q/K/V 各 16 头，每头 128 维；卷积宽度 4、通道数 6144 |
| Full attention | 8 个 Q 头、2 个 KV 头，每头 256 维，带输出门控 |
| RoPE | 每头前 64 维旋转，theta 为 10000000 |
| GDN 状态池 | FP32 `[page,16,128,128]`，最后两维依次为 value、key |
| 卷积状态池 | BF16 `[page,6144,3]` |

图捕获、量化、speculative decoding 和多芯片运行不在本次适配范围内。

## 实现

模型结构、checkpoint loader、C++ scheduler 和 KV/state cache 沿用 TokenSpeed。
新增算子通过 kernel registry 分派，GDN 沿原有 hybrid attention 路径执行：

```text
Qwen3_5GatedDeltaNet
  → QKV/Z/b/a projection
  → GdnAttnBackend / MambaAttnBackend
  → causal_conv1d → gdn_chunk_prefill 或 gdn_decode_step
  → gated RMSNorm → output projection
```

| 部分 | 本次处理 |
| --- | --- |
| Embedding、线性投影、MLP、LM head、greedy sampling | 复用原有 NPU/Torch 路径 |
| Causal convolution | 新增 Ascend prefill/update，FP32 累加后转 BF16，再执行 SiLU |
| GDN | 新增 chunk prefill 与单步 decode，归一化、decay 和状态更新使用 FP32 |
| Gated RMSNorm | 新增 NPU 实现，保留参考实现的 BF16 中间舍入 |
| Gemma RMSNorm、Q/K norm | 用 FP32 计算 `1 + weight` 与归一化，最后恢复激活 dtype |
| 256 维头 attention | 为 prefill/extend 增加 BSH 路径，调用 CANN fused attention |
| 模型初始化 | 辅助 CUDA stream 仅在 NVIDIA 上创建 |
| Prefill 采样 offset | 在独立副本上更新 offset，避免修改缓存长度 |

主要实现位于 [GDN Ascend 算子](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/ascend.py)、
[Ascend layernorm](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/layernorm/ascend.py)、
[NPU MHA](../../tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/mha.py) 和
[model executor](../../python/tokenspeed/runtime/execution/model_executor.py)。

### GDN 与状态管理

以下用数学状态 $S_t\in\mathbb R^{K\times V}$ 表示一个 head；代码状态池使用其转置。
Q/K 先按 `sqrt(sum(x²) + 1e-6)` 归一化，随后计算：

$$
\bar S_t=e^{g_t}S_{t-1},\qquad
\Delta_t=\beta_t(v_t-k_t^\top\bar S_t),\qquad
S_t=\bar S_t+k_t\Delta_t,\qquad
o_t=q_t^\top S_t/\sqrt K.
$$

Decode 直接执行这一递推。Prefill 以 64 token 分块，将 delta rule 写为下三角
系统，通过 `torch.linalg.solve_triangular` 求解，再用矩阵乘得到输出与最终状态。
相比有限矩阵多项式展开，直接求解在高度相关的 K 上更稳定。该求解在设备侧 AiCPU
执行。beta 保留 `b.sigmoid().float()` 的顺序，与 BF16 参考路径一致。

状态由现有 cache recipe 和 LCM cache 管理。Decode 先读取所有输入页，再写目标页，
以保留共享输入快照；负页号表示 padding，不写回状态。页 0 是只读的零状态。
当前 Qwen recipe 的 KV/SSM 使用紧密页步长，卷积窗口允许 page-interleaved
arena 中的非连续 view。相关测试同时检查状态值和相邻未写字段。

### 两个兼容性问题

CANN 9 的 TND attention 路径不支持该模型的 256 维头，因此 prefill/extend 按
请求改用 BSH。Prefill 使用左上对齐因果 mask，extend 使用右下对齐 mask；单 query
没有未来 key，使用 `None` mask。KV cache 和页表直接传给融合算子。变长 metadata
仍需同步到主机，逐请求调用也带来额外开销。

另一个问题出在 torch_npu 2.9.0 的 `index_copy`：设备测试中，非原地调用返回了
新 tensor，却也修改了源 tensor。临时采样 offset 因此污染了缓存长度，使 129-token
输入的首个 decode 位置从 129 变成 257。修复为先 `clone()`，再在副本上执行
`index_copy_`。修复后该案例最大 logprob 误差从 0.494367 降至 0.018241。

## 正确性结果

2026-10-15 的正式测试使用真实权重和 Transformers 5.17.0 参考实现。两边均设置
`ACL_PRECISION_MODE=must_keep_origin_dtype` 并关闭 matmul/conv HF32。
输入不加 chat template，greedy 生成 16 token，忽略 EOS。

| 请求顺序 | 输入 / 输出 token | 一致的生成 ID | 最大 selected-token logprob 误差 |
| --- | --- | --- | --- |
| A-before | 17 / 16 | 16/16 | 0.071408 |
| B | 129 / 16 | 16/16 | 0.018241 |
| A-after | 17 / 16 | 16/16 | 0.071408 |
| C | 513 / 16 | 16/16 | 0.090854 |

64 个生成 ID 全部一致，logprob 误差均小于预定阈值 0.10。A→B→A 在同一请求槽
内连续执行，中间不清缓存，A 前后的 ID 和 logprob 完全相同。513-token 输入跨越
512-token prefill 分块边界。阈值校准使用独立的 HF BF16/NPU cached 与 FP32/CPU
完整前缀重算；A17 的 16 个 ID 一致，最大 logprob 差为 0.07467115。

### 算子回归

| 测试 | 通过数 | 主要覆盖 |
| --- | --- | --- |
| GDN | 54 | recurrence、chunk 边界、连续 decode、卷积、gated norm、状态页 |
| Gemma RMSNorm | 32 | residual、FP32/BF16、空 batch、实际 NPU |
| Q/K Gemma norm 与 dispatch | 34 | weight offset、strided 输入、CPU/NPU |
| 256 维头 MHA | 22 | prefill/extend/decode、分页、单 query、非连续 cache |
| Prefill sample offset | 13 | 源缓存长度不变、跨页和后续 decode 位置 |

以上为 155 项定向测试。完整 `test_zero_output_prefill.py` 为 52 passed、5 failed；
5 项失败来自已有测试替身缺少属性，在原始 base commit 上也能复现。

在准备好的服务器环境中运行以下命令。每个测试组使用独立进程，显式开启 NPU 测试：

```bash
export TOKENSPEED_TEST_NPU=1
npu_test() {
  python -c 'import sys; sys.path.insert(0, "examples/aics_qwen35"); from precision import configure_npu_precision; configure_npu_precision(); import pytest; raise SystemExit(pytest.main(sys.argv[1:]))' "$@"
}
npu_test -q tokenspeed-kernel/test/ascend/test_gdn.py
npu_test -q test/runtime/layers/test_ascend_gemma_rmsnorm.py
npu_test -q tokenspeed-kernel/test/ascend/test_qk_gemma_norm.py \
  tokenspeed-kernel/test/ops/test_layernorm_dispatch.py
npu_test -q tokenspeed-kernel-npu/test/test_mha_head256.py
npu_test -q test/runtime/test_zero_output_prefill.py -k prefill_sample_offset
```

## 性能结果

配置为单芯片、batch 1、eager、BF16 激活和 FP32 GDN 状态，关闭前缀缓存、调度
重叠及输出 logprob。上下文长度为 1024，cache 容量为 16384 token，prefill 分块
为 512。每组预热一次、测量三次，实际输出长度均达到要求。

TTFT 从主机发起请求计至首个非空流式帧；TPOT 为首末帧交付时间差除以输出数减一。
它们包含调度与 IPC，不含模型加载和 Engine 启动。下表给出中位数与范围：

| 输入 / 输出 token | TTFT，ms | TPOT，ms/token |
| --- | --- | --- |
| 128 / 16 | 165.286 [160.788, 170.609] | 75.663 [75.183, 77.767] |
| 128 / 64 | 167.538 [155.004, 167.566] | 71.146 [70.461, 72.493] |
| 512 / 16 | 385.258 [384.446, 392.543] | 71.492 [71.335, 75.817] |
| 512 / 64 | 395.756 [392.004, 397.199] | 76.040 [76.040, 77.994] |

HBM 每 200 ms 采样一次，取三次测量请求区间内的最大值，单位 MiB：

| 输入 / 输出 token | Worker allocated | Worker reserved | 整芯片 driver-used |
| --- | --- | --- | --- |
| 128 / 16 | 1985.68 | 2234.00 | 2520.05 |
| 128 / 64 | 1982.26 | 2234.00 | 2520.05 |
| 512 / 16 | 2013.77 | 2354.00 | 2640.54 |
| 512 / 64 | 2010.73 | 2354.00 | 2640.54 |

allocated/reserved 来自 worker 的 Torch allocator；driver-used 还包含设备上下文
与其他进程。200 ms 采样不能捕捉所有瞬时峰值。

### Profiler 观察

17 和 512 token 输入各单独预热并采集一次，生成 4 token，包含一次 prefill 与
三次 decode。下表为阶段内设备 kernel 时长之和，不是请求墙钟时间。

| 输入 / 阶段 | Kernel 数 | 时长和，ms | 主要 kernel |
| --- | --- | --- | --- |
| 17 / Prefill | 4080 | 32.226 | 三角求解 23.01%，Mul 10.40%，Cast 5.89% |
| 17 / 三次 Decode | 8034 | 32.521 | MatMul 15.24%，Mul 14.48%，ReduceSum 7.58% |
| 512 / Prefill | 11105 | 332.751 | 三角求解 73.37%，Mul 3.81%，Matmul 辅助 Transpose 2.39% |
| 512 / 三次 Decode | 8034 | 33.171 | MatMul 14.94%，Mul 14.27%，AiCPU Index 8.29% |

512-token prefill 中 144 次三角求解耗时 244.130 ms，是最主要的设备开销；全部
transpose 合计 19.807 ms，占 5.95%。Decode 的 index/gather/scatter 合计
7.173 ms，占 21.62%，可优先考虑状态索引与更新的融合，减少小算子提交。

阶段归因沿 `torch_to_npu` flow 的 CPU 起点进行，两次采集未归因 kernel 的时长
占比为 0.374% 和 0.078%。CPU recorder 出现重复范围，所以未用 CPU self-time
做定量结论。Profiler 的同步会改变时序，TTFT/TPOT 只取未启用 profiler 的测量。
当前数据支持先优化三角求解与状态操作，尚不足以判断 HBM 带宽饱和或预计加速比例。
