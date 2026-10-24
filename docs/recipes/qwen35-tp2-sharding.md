# Qwen3.5 的双 chip 张量并行

本配置在两个 rank 上运行同一个模型，attention TP、dense TP 和 LM-head TP
均为 2，attention DP 为 1。两个 rank 接收相同的 token 和位置，分别计算自己的
权重分片，再通过 HCCL 合并结果。以下形状适用于 BF16、纯文本、eager 执行，
不启用量化、推测解码、KV parallelism 或 all-reduce fusion。

## 模型几何

参数来自各模型的官方 `config.json`。四个模型的词表均为 248320，MHA head
dimension 为 256，GDN 的 key/value head dimension 均为 128，卷积宽度为 4。
每四层中前三层为 GDN，第四层为 full attention。
固定版本和完整配置见 [`model-manifest.json`](../../examples/aics_qwen35_tp2/model-manifest.json)。

| 模型 | hidden `H` | MLP `I` | 层数 | MHA Q/KV heads | GDN key/value heads | 共享 embedding 与 LM head |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.8B | 1024 | 3584 | 24 | 8 / 2 | 16 / 16 | 是 |
| 4B | 2560 | 9216 | 32 | 16 / 4 | 16 / 32 | 是 |
| 9B | 4096 | 12288 | 32 | 16 / 4 | 16 / 32 | 否 |
| 27B | 5120 | 17408 | 64 | 24 / 4 | 16 / 48 | 否 |

下文 `T` 为本次 forward 的 token 行数；prefill 中是请求的输入行数之和，
decode 中通常是活跃请求数。权重按 PyTorch 的 `[out_features, in_features]`
记。两 rank 的形状相同，保存的通道范围不同。

## Gated DeltaNet 层

设全模型 GDN 的 key/value 宽度为 `K = 16 × 128 = 2048` 和
`V = value_heads × 128`。每个 rank 的输入都是完整的 `[T,H]`。
`in_proj_qkvzba` 对 q、k、v、z、b、a 六个成员分别沿输出通道切分：

| 张量或权重 | 0.8B，每 rank | 4B / 9B，每 rank | 27B，每 rank |
| --- | --- | --- | --- |
| `in_proj_qkvzba.weight` | `[4112,1024]` | `[6176,H]` | `[8240,5120]` |
| q、k，各自 | `[T,8,128]` | `[T,8,128]` | `[T,8,128]` |
| v、z，各自 | `[T,8,128]` | `[T,16,128]` | `[T,24,128]` |
| b、a，各自 | `[T,8]` | `[T,16]` | `[T,24]` |
| `conv1d.weight` | `[3072,1,4]` | `[4096,1,4]` | `[5120,1,4]` |
| `A_log`、`dt_bias`，各自 | `[8]` | `[16]` | `[24]` |
| `out_proj.weight` | `[1024,1024]` | `[H,2048]` | `[5120,3072]` |
| 输出投影后的局部和 | `[T,1024]` | `[T,H]` | `[T,5120]` |

rank 0 持有每个成员的前半通道，rank 1 持有后半通道。例如 4B/9B 上，
rank 0 的 key heads 为 0–7，value heads 为 0–15；rank 1 分别为 8–15
和 16–31。卷积权重也分别切 q、k、v 三段，不能把整个拼接矩阵直接从中间切开。

GDN 的 value heads 可以多于 key heads。Ascend 算子使用
`repeat_interleave(value_heads / key_heads)` 在本地展开 q/k。4B/9B 上 key 0
对应 value 0、1；27B 上 key 0 对应 value 0、1、2。27B 的 rank 1 从 key 8
和 value 24 开始，分组边界仍然一致。展开顺序是相邻复制同一个 key head，
不是将整组 key heads 重复两次或三次。

卷积、GDN 扫描和每个 value head 内的 gated RMSNorm 均在本地完成，不需要
跨 rank 交换状态。gated norm 的 128 个权重复制到两个 rank。`out_proj` 沿输入
通道切分，各 rank 产生 `[T,H]` 的部分和；`CommManager.post_attn_comm` 随后
执行 SUM all-reduce，得到每个 rank 都可使用的完整 attention 输出。

相关实现：[`Qwen3_5GatedDeltaNet`](../../python/tokenspeed/runtime/models/qwen3_5.py)、
[`MergedColumnParallelLinear`](../../python/tokenspeed/runtime/layers/linear.py)、
[`mamba_v2_sharded_weight_loader`](../../python/tokenspeed/runtime/model_loader/weight_utils.py)、
[`_expand_heads`](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/ascend.py)。

## Full attention 层

Q 和输出 gate 按 head 成对排列，q/gate、k、v 的权重分别切分。归一化和 RoPE
仅操作本 rank 的 heads；Q/K norm 的 256 个权重在两 rank 上复制。

| 张量或权重 | 0.8B，每 rank | 4B / 9B，每 rank | 27B，每 rank |
| --- | --- | --- | --- |
| 输入 | `[T,1024]` | `[T,H]` | `[T,5120]` |
| `qkv_proj.weight`，含 Q gate | `[2560,1024]` | `[5120,H]` | `[7168,5120]` |
| Q、gate，各自 | `[T,4,256]` | `[T,8,256]` | `[T,12,256]` |
| K、V，各自 | `[T,1,256]` | `[T,2,256]` | `[T,2,256]` |
| attention 输出 | `[T,4,256]` | `[T,8,256]` | `[T,12,256]` |
| `o_proj.weight` | `[1024,1024]` | `[H,2048]` | `[5120,3072]` |
| 输出投影后的局部和 | `[T,1024]` | `[T,H]` | `[T,5120]` |

例如 27B 上，rank 0 计算 Q heads 0–11 和 KV heads 0–1，rank 1 计算
Q heads 12–23 和 KV heads 2–3。每个 rank 在本地执行自己的 grouped-query
attention；`o_proj` 后执行一次 `[T,H]` SUM all-reduce。Ascend MHA 的
`num_heads` 和 `num_key_value_heads` 来自本地 Q/KV 张量，因此 TP 不改变
单 head 的 256 维约束。prefill 和 paged extend 使用 BSH 路径。

相关实现：[`Qwen3_5AttentionDecoderLayer`](../../python/tokenspeed/runtime/models/qwen3_5.py)、
[`QKVParallelLinear`](../../python/tokenspeed/runtime/layers/linear.py)、
[`Ascend MHA`](../../tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/mha.py)。

## MLP、embedding 与输出

每层的 gate/up projection 沿输出通道切分，两个 rank 各得到
`[T,I/2]` 的 gate 和 up 激活，局部执行 SiLU 与逐元素乘法。
`down_proj.weight` 沿输入通道切分，各自产生 `[T,H]` 部分和，再进行一次
SUM all-reduce。

| 权重或激活 | rank 0 | rank 1 |
| --- | --- | --- |
| 输入 | `[T,H]`，复制 | `[T,H]`，复制 |
| `gate_up_proj.weight` | `[I,H]`，gate/up 各取前半行 | `[I,H]`，gate/up 各取后半行 |
| SiLU × up | `[T,I/2]`，前半通道 | `[T,I/2]`，后半通道 |
| `down_proj.weight` | `[H,I/2]`，前半输入通道 | `[H,I/2]`，后半输入通道 |
| all-reduce 后输出 | `[T,H]`，完整 | `[T,H]`，完整 |

embedding 和 LM head 按词表切分，每 rank 的权重形状均为 `[124160,H]`。
embedding 查询后用 SUM all-reduce 合并非零结果。模型末尾，每个 rank 计算
`[R,124160]` 的局部 logits，HCCL all-gather 后按 rank 顺序拼为
`[R,248320]`，再进行 greedy 采样或 logprob 计算；`R` 是需要打分的行数，
普通 decode 通常等于当前请求数。

在上述配置下，一次 forward 有 `2 × 层数 + 1` 次模型主体 SUM all-reduce，
即 0.8B 为 49 次，4B/9B 为 65 次，27B 为 129 次，另有 LM-head logits
all-gather。greedy 采样后还从 TP rank 0 广播一次 `[R]` INT32 token IDs，
使两个 rank 后续使用相同的生成序列；`R` 是采样张量的行数，可能包含 padding。
因此 0.8B 的普通 forward 共 51 次 HCCL collective：49 次 all-reduce、
1 次 all-gather 和 1 次 broadcast。采样广播直接调用 `torch.distributed.broadcast`，
仅包装 `HcclBackend` 方法的主机侧记录不会包含它，CANN trace 会包含它。
每次 `[T,H]` BF16 all-reduce 的逻辑张量大小为 `2TH` 字节；这不是
链路传输字节数，实际通信算法和分段应以 HCCL trace 为准。调度器控制消息和
测量工具也可能产生额外通信，不能将 trace 中的全部通信都算作模型层通信。

## 推理状态

两个 rank 保存相同的 token、位置、请求槽和页表等调度元数据，缓存数据则按
head 切分。设 `P` 为状态页数，`B` 为 KV page 的 token 数：

| 每层的本地状态 | 0.8B | 4B / 9B | 27B |
| --- | --- | --- | --- |
| GDN conv，BF16 | `[P,3072,3]` | `[P,4096,3]` | `[P,5120,3]` |
| GDN SSM，FP32 | `[P,8,128,128]` | `[P,16,128,128]` | `[P,24,128,128]` |
| full attention K/V，各自 BF16 | `[P,B,1,256]` | `[P,B,2,256]` | `[P,B,2,256]` |

SSM 的最后两维是 `[value_dim,key_dim]`。表中是逻辑形状，conv 分量可以使用
页内交错的物理 stride。这里采用按 head 的 TP，两个 rank 都保存请求的完整
时间范围；没有沿序列长度切页，也没有每 token 交换 SSM 或 KV 的步骤。

状态形状来自 [`LinearAttnConfig`](../../python/tokenspeed/runtime/layers/attention/configs/linear_attn.py)
和 [`Qwen35CacheRecipe`](../../python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py)。
状态更新沿用 scheduler 分配的输入/输出页，不另建 rank 私有的请求状态表。

`tokenspeed-kernel/test/ascend/test_gdn.py` 中的
`test_tp2_head_shards_match_full_reference_through_decode` 覆盖 1、2、3 倍
value heads：将两份本地 prefill 输出和 SSM 沿 head 维拼回，随后连续 decode
并交换状态页，与完整 head 的独立逐 token 参考比较。四个完整模型还进行了
TP1/TP2 连续推理对照，覆盖权重加载、层间 all-reduce、KV cache 和采样路径；
比较结果及 4B、9B 的数值差异见[运行说明](../../examples/aics_qwen35_tp2/README.md#正确性对照)。
