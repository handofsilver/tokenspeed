# Qwen3.5-0.8B 单芯片实验

这组脚本在 Ascend 910C 上运行 TokenSpeed 原生 `Engine`，并用 Transformers
比较生成结果。模型使用 BF16 权重、FP32 GDN 状态，采用 eager 和 greedy decode，
仅处理文本。适配改动和已有结果见[实验记录](../../docs/recipes/qwen35-ascend-hw1.md)，
仓库来源见[首页](../../README.md)。

## 准备

先按[环境说明](environment-notes.md)建立服务器上的虚拟环境。以下命令均在仓库
根目录执行；模型和 NPU 测试在服务器运行。选择一块获准使用且空闲的物理芯片：

```bash
export ASCEND_RT_VISIBLE_DEVICES=7  # 按实际分配修改
export AICS_MODEL_PATH="$PWD/models/Qwen3.5-0.8B"
mkdir -p outputs/aics
```

模型文件需要预先下载。运行脚本前可用下列命令取得本次使用的版本：

```bash
hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir "$AICS_MODEL_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

各脚本的 `--model` 可覆盖 `AICS_MODEL_PATH`；未设置时使用
`models/Qwen3.5-0.8B`。脚本使用逻辑设备 `npu:0`，并检查 Engine 来自当前源码目录。
`precision.py` 在参考进程和实际推理 worker 内设置
`ACL_PRECISION_MODE=must_keep_origin_dtype`，关闭 matmul/conv HF32，随后读取选项
确认设置生效。

## 正确性比较

```bash
python examples/aics_qwen35/run.py prepare \
  --output outputs/aics/cases.json

python examples/aics_qwen35/run.py reference \
  --cases outputs/aics/cases.json --suite validation \
  --device npu --dtype bfloat16 --cache-mode cached \
  --output outputs/aics/hf-bf16.json

python examples/aics_qwen35/run.py native \
  --cases outputs/aics/cases.json --suite validation \
  --attention-backend mha --sampling-backend greedy \
  --output outputs/aics/native-validation.json

python examples/aics_qwen35/run.py compare \
  --native outputs/aics/native-validation.json \
  --reference outputs/aics/hf-bf16.json --logprob-atol 0.10 \
  --output outputs/aics/comparison.json
```

输入分别为 17、129、513 token，每次生成 16 token，不加 chat template 或隐式
special token，并忽略 EOS。请求按 A→B→A→C 顺序在同一单槽 Engine 中执行，
关闭前缀缓存，中途不清空缓存。重复 A 用于检查请求间状态隔离，C 跨越 512-token
prefill 分块边界。

比较同时要求生成 ID 完全一致、selected-token logprob 误差不超过 0.10，
以及重复 A 的 ID 和 logprob 完全一致。这个阈值来自独立参考校准：A17 的
Transformers BF16/NPU cached 推理与 FP32/CPU 完整前缀重算生成 ID 一致，
最大 logprob 差为 0.07467115。`compare` 严格通过时退出码为 0，数值比较失败
为 2，执行错误为 1。

`reference` 还支持 `--case-ids A17`、`--device cpu --dtype float32
--cache-mode recompute`。CPU 校准也应在服务器运行。若两边生成序列发生分岔，
可加 `--native-trajectories outputs/aics/native-validation.json`，让参考模型沿
原生输出轨迹评分，区分条件历史变化与数值误差。

## 性能测量

```bash
python examples/aics_qwen35/run.py native \
  --cases outputs/aics/cases.json --suite benchmark \
  --attention-backend mha --sampling-backend greedy \
  --warmup 1 --repeats 3 \
  --output outputs/aics/native-benchmark.json
```

测试 128/512 输入 × 16/64 输出四组长度，每组预热一次、测量三次，性能测试关闭
输出 logprob。默认配置为 batch 1、上下文 1024、cache 容量 16384 token、prefill
分块 512，并关闭图捕获与调度重叠。

- TTFT：主机发起请求至首个非空流式帧的时间，包含调度、IPC 和交付开销。
- TPOT：首末非空帧的交付时间差，除以实际生成 token 数减一。若首帧合并了多个
  token，脚本不计算 TPOT。
- HBM：worker 每 200 ms 采样一次，分别记录 Torch allocated/reserved 和整芯片
  driver-used。前两者属于 worker，后者还包括设备上下文及其他进程；采样最大值
  可能漏掉短暂峰值。

JSON 保存参数、版本、源码与输入哈希、实际 token 数、逐次时间及内存记录。
预热不计入汇总。`status=completed` 表示执行结束，数值比较结果在
`comparison.json` 的 `strict_pass` 中。

## Profiler

```bash
python examples/aics_qwen35/profile_native.py \
  --cases outputs/aics/cases.json --case-ids A17 --max-new-tokens 4 \
  --attention-backend mha --sampling-backend greedy \
  --output outputs/aics/profile-A17.json

python examples/aics_qwen35/profile_native.py \
  --cases outputs/aics/cases.json --case-ids bench-512-16 --max-new-tokens 4 \
  --attention-backend mha --sampling-backend greedy \
  --output outputs/aics/profile-512.json
```

每次先预热，再记录一次 prefill 和三次 decode，导出 Chrome trace 与阶段汇总。
设备 kernel 通过 `torch_to_npu` flow 的 CPU 起点归到 prefill/decode，未关联的
事件单列。Profiler 会同步设备，测得的请求延迟不能替代上面的 benchmark。
当前版本还存在 CPU recorder 范围重复问题，因此结果分析使用设备 kernel 事件，
不使用 CPU 算子次数或 self-time 判断占比。

## 脚本测试

以下测试不加载模型，可在本地运行：

```bash
python -m unittest discover -s examples/aics_qwen35 -p 'test_*.py' -v
```

它们检查流式计数、时间汇总、生成轨迹比较、精度配置与内存采样数据处理。
算子回归需要服务器 NPU，命令见[实验记录](../../docs/recipes/qwen35-ascend-hw1.md#算子回归)。
