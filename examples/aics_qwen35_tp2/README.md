# Qwen3.5 双 chip 实验

在 TokenSpeed 的原生 Engine 上运行 Qwen3.5-0.8B、4B、9B 和 27B，比较 TP1 与 TP2 的正确性、延迟、吞吐和每个 rank 的 HBM 占用。模型分别加载，权重版本见 `model-manifest.json`。

以下命令在配置好 `torch_npu`、CANN 和 TokenSpeed 依赖的 Linux 服务器上，从仓库根目录执行。`ASCEND_RT_VISIBLE_DEVICES` 填当前分配的物理 chip 编号；TP2 应选择同一张卡上的两个 chip。脚本在进程内将它们映射为 logical device 0、1。

依赖环境沿用[第一次作业的安装说明](../aics_qwen35/environment-notes.md)。运行前加载 CANN，并让 Python 使用当前 checkout：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export TOKENSPEED_MAMBA_SSM_DTYPE=float32
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

## 准备模型和输入

```bash
python -m examples.aics_qwen35_tp2.download_models \
  --mode download --endpoint https://modelscope.cn \
  --models Qwen3.5-0.8B Qwen3.5-4B Qwen3.5-9B Qwen3.5-27B \
  --destination /path/to/models

python -m examples.aics_qwen35_tp2.run prepare \
  --model /path/to/models/Qwen3.5-0.8B \
  --input-lengths 128,512 --output-tokens 64 \
  --output /path/to/results/0.8b-cases.json
```

ModelScope 下载入口仅采用清单中与固定 Hugging Face 版本校验值相同的文件。

每个模型单独生成输入文件。验证输入为 17、129、513 token，输出 16 token；性能输入为指定的两档长度，输出默认 64 token。TP1、TP2 和参考实现使用同一份输入文件。

## 运行

先用 `--suite smoke --warmup 0 --repeats 1` 做 17 输入、4 输出的短测试。连续推理验证使用以下配置：

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 \
python -m examples.aics_qwen35_tp2.run native \
  --model /path/to/models/Qwen3.5-0.8B \
  --cases /path/to/results/0.8b-cases.json \
  --suite validation --tp-size 2 --concurrency 1 \
  --attention-backend mha --sampling-backend greedy \
  --overlap-schedule off --record-collectives on \
  --prefill-chunk-tokens 512 --memory-utilization 0.15 \
  --max-total-tokens 16384 \
  --output /path/to/results/0.8b-tp2-validation.json
```

验证顺序为 A17 → B129 → A17 → C513，中间不刷新缓存。结果保存生成 ID、对应 logprob、重复 A 的结果和双 rank 的执行记录。TP1 将可见设备改为一个 chip，`--tp-size` 改为 `1`，其余任务配置保持一致。

四个模型的验证使用以下预算，TP1 与 TP2 保持相同的缓存容量上限：

| 模型 | `--memory-utilization` | `--max-total-tokens` |
| --- | ---: | ---: |
| Qwen3.5-0.8B | 0.15 | 16384 |
| Qwen3.5-4B | 0.30 | 16384 |
| Qwen3.5-9B | 0.50 | 16384 |
| Qwen3.5-27B | 0.96 | 8192 |

运行其他模型时，替换命令中的模型目录、输入文件和输出文件，再使用表中对应的两个参数。例如，27B 的 TP2 验证为：

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 \
python -m examples.aics_qwen35_tp2.run native \
  --model /path/to/models/Qwen3.5-27B \
  --cases /path/to/results/27b-cases.json \
  --suite validation --tp-size 2 --concurrency 1 \
  --attention-backend mha --sampling-backend greedy \
  --overlap-schedule off --record-collectives on \
  --prefill-chunk-tokens 512 --memory-utilization 0.96 \
  --max-total-tokens 8192 \
  --output /path/to/results/27b-tp2-validation.json
```

27B 纯文本参数占用约 50.098 GiB。TP1 启动前，应确认所选 chip 的空闲 HBM 不低于 **52.9822 GiB**；该预算包括已知 buffer、缓存、0.25 GiB 执行 workspace 预算和额外 2 GiB 余量。本次 TP1 在物理 chip 1 上运行成功，启动前空闲 HBM 为 58.1355 GiB。`max_total_tokens=8192` 实际分配 6 个物理缓存页（含 null 页），启动时得到 `max_single_request_tokens=1536`，可以容纳最长的 513 输入加 16 输出。

性能实验把 `--suite` 改为 `benchmark`，`--record-collectives` 改为 `off`。在同一模型上分别设置 TP1/TP2 和 `--concurrency 1/4`，每组可用 `--warmup 2 --repeats 5`。使用上表中的预算，并在运行前重新检查空闲 HBM。

`--max-total-tokens` 默认 16384，控制缓存页规划的总容量，区别于当前请求实际活跃的上下文长度。混合 KV、卷积和 GDN 状态的缓存组需要满足共同页粒度；设得过小会导致启动时没有可用的单请求容量。本次配置下，0.8B 设置为 4096 时无法启动，因此采用 16384。`--memory-utilization` 则决定加载权重后为缓存规划保留多少预算：加载后空闲 HBM 减去总 HBM 的 `1 - utilization` 部分。高比例不等于实际占满显存，缓存分配仍受 token 上限约束。

并发请求通过同一个 Engine 发出。每请求保留 TTFT、TPOT 和完成时间；整组吞吐用总输出 token 数除以从共同释放请求到最后请求完成的时间。计时包含调度、IPC 和流式交付，正式计时不启用 profiler。

双 chip 记录写入输出 JSON 旁的 `.workers-*` 目录，包含：

- rank、PID、物理 chip 与 logical device 映射；
- 推理前后的实际精度选项；
- 加载后参数的形状、类型和设备；
- 完成的 prefill/decode、每次 forward 的 token 数与 batch size；
- 对应设备的 allocator 样本，以及单独列出的整 chip driver 占用。

`--record-collectives on` 另存 HCCL 调用次数、输入形状和逻辑 payload 字节数，用于通信分析。这会增加主机记录开销；正式性能组关闭，通信统计组单独采集。逻辑 payload 不等于链路传输字节数，主机调用时长也不等于设备通信时长。

## 正确性对照

```bash
python -m examples.aics_qwen35_tp2.compare \
  --native /path/to/results/0.8b-tp2-validation.json \
  --reference /path/to/results/0.8b-tp1-validation.json \
  --logprob-atol 0.10 \
  --output /path/to/results/0.8b-tp1-tp2-comparison.json
```

TP1 与 TP2 使用相同权重、输入、精度和采样设置。若生成路径出现分歧，使用相同历史的 teacher-forced logprob 比较进一步定位；历史改变后的 logprob 不直接比较。

4B、9B 的独立参考使用 Transformers BF16，在服务器 CPU 上打分。以下命令复用第一次作业的参考入口，按原生 TP2 已生成的历史逐步计算参考 logits：

```bash
python examples/aics_qwen35/run.py reference \
  --model /path/to/models/Qwen3.5-4B \
  --cases /path/to/results/4b-cases.json --suite validation \
  --device cpu --dtype bfloat16 --cache-mode cached --cpu-threads 16 \
  --native-trajectories /path/to/results/4b-tp2-validation.json \
  --output /path/to/results/4b-reference.json

python -m examples.aics_qwen35_tp2.compare \
  --native /path/to/results/4b-tp2-validation.json \
  --reference /path/to/results/4b-reference.json --logprob-atol 0.10 \
  --output /path/to/results/4b-reference-comparison.json
```

实测中，0.8B、27B 的 TP1/TP2 严格对照通过；4B 有一条请求发生生成分岔，9B 的最大 logprob 差为 0.104253。4B TP2、9B TP1/TP2 分别对独立参考均通过原 0.10 阈值。4B 首次分岔位置的两名候选另用 CPU BF16/FP32 复查：两种精度均将 TP2 选择的 561 排在第一、TP1 选择的 198 排在第二，logit 间隔分别为 0.125 和 0.018093。该检查沿用共同的前两个输出，只计算到第三步。原始比较、补查和各配置的性能数据见 [`results.json`](results.json)。

## 时间线

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 \
python -m examples.aics_qwen35_tp2.profile_native \
  --model /path/to/models/Qwen3.5-0.8B \
  --cases /path/to/results/0.8b-cases.json \
  --case-ids bench-512-64 --suite benchmark --tp-size 2 --concurrency 4 \
  --attention-backend mha --sampling-backend greedy \
  --overlap-schedule off --record-collectives on \
  --prefill-chunk-tokens 512 --memory-utilization 0.15 \
  --max-total-tokens 16384 --profile-output-tokens 16 \
  --output /path/to/results/0.8b-tp2-c4-profile.json
```

每个 rank 分别导出 CPU+NPU trace，标出 prefill、decode 和 HCCL 调用。时间线按设备区间的并集和交集计算，保留未识别任务，避免把异步工作简单相加。当前 `torch_npu` 的子线程 profiler 接口在每次 forward 结束时同步设备，采集结果用于算子与通信诊断，不能替代普通性能测试或证明跨 forward 的重叠收益。

采集显式启用 Level1，以记录 HCCL 数据；依据为[昇腾 ProfilerLevel 文档](https://www.hiascend.com/doc_center/source/en/canncommercial/700/modeldevpt/ptmigr/ptaoplist_000536.html)。原始 profiler 文件一并保留。

## 本地纯数据测试

```bash
python -m pytest -q examples/aics_qwen35_tp2
```

测试不加载模型、不初始化 Torch 或 NPU。模型验证与性能测试均在服务器执行。

算子和运行时测试在服务器分别执行，避免两个 `test` 目录的 pytest 包名冲突：

```bash
python -m pytest -q test/runtime/test_logits_processor.py \
  test/runtime/distributed/test_hccl_backend.py
python -m pytest -q tokenspeed-kernel/test/ascend/test_gdn.py
```

上述定向测试共 220 项通过，1 项 CUDA 用例跳过。
