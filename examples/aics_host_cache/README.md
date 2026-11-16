# Ascend Host cache 实验

每轮生成结束后释放设备上的前缀缓存，再让同一会话继续，或换一个新会话。开启 Host 缓存时，后续请求可以从 Host 恢复 KV、递归和卷积状态。关闭时，后续请求从头做 prefill。两组都经过原生 Engine 和 C++ 调度器。

## 环境

使用 Linux、Ascend 910C、CANN 9.0.0、Python 3.11、PyTorch 2.9.0 和 torch_npu 2.9.0。先加载 CANN 和 Python 环境，按[双 chip 实验说明](../aics_qwen35_tp2/README.md)设置当前源码的 `PYTHONPATH`、FP32 递归状态和线程数。模型沿用[同一份权重清单](../aics_qwen35_tp2/model-manifest.json)，本次只用到 0.8B 与 9B。

在仓库根目录执行以下命令。调度器增加了 Host 占用和淘汰计数，需要从当前分支编译扩展。示例使用 Python 基础环境提供的 OpenSSL 头文件和库，如使用系统 OpenSSL，可省略 `OPENSSL_ROOT_DIR`。

```bash
work="$PWD/../cache-experiment"
mkdir -p "$work"
python -m pip install --target "$work/builddeps" nanobind==3.1.0 tokenspeed-spdlog==1.15.1
export PYTHONPATH="$work/builddeps:$PYTHONPATH"
cmake -S tokenspeed-scheduler -B "$work/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DOPENSSL_ROOT_DIR="$(python -c 'import sys; print(sys.base_prefix)')"
cmake --build "$work/build" -j3
cmake --install "$work/build" --prefix "$work/overlay"
cp tokenspeed-scheduler/python/tokenspeed_scheduler/__init__.py "$work/overlay/tokenspeed_scheduler/"
export PYTHONPATH="$work/overlay:$PYTHONPATH"
```

## 运行对照

将 `MODELS` 设为模型所在目录，`PAIR` 设为同一张卡上可用的两个物理 chip，`SINGLE` 设为一个可用 chip。每组使用新的输出目录。以下命令依次运行报告中的九组实验。

```bash
export AICS_CACHE_RECORDINGS="$work/runs"
mkdir -p "$AICS_CACHE_RECORDINGS"
trial() {
  ASCEND_RT_VISIBLE_DEVICES="$1" python -m examples.aics_host_cache.run \
    --run-dir "$AICS_CACHE_RECORDINGS/$2" --model "$MODELS/Qwen3.5-$3" \
    --tp "$4" --pattern "$5" --offload "$6" --budget-ratio "$7" --num-requests "$8"
}

trial "$SINGLE" s1-high-off   0.8B 1 high off 2     10
trial "$SINGLE" s1-high-wide  0.8B 1 high on  2     10
trial "$PAIR"   s2-high-off   0.8B 2 high off 2     10
trial "$PAIR"   s2-high-wide  0.8B 2 high on  2     10
trial "$PAIR"   s2-high-tight 0.8B 2 high on  0.375 10
trial "$PAIR"   s2-low-off    0.8B 2 low  off 2     10
trial "$PAIR"   s2-low-on     0.8B 2 low  on  2     10
trial "$PAIR"   m9-high-off   9B   2 high off 2      6
trial "$PAIR"   m9-high-on    9B   2 high on  2      6

python -m examples.aics_host_cache.analyze \
  --run "$AICS_CACHE_RECORDINGS/s2-high-wide" \
  --reference "$AICS_CACHE_RECORDINGS/s2-high-off" \
  --logprob-atol 0.10 --output "$work/comparison.json"
python -m pytest -q examples/aics_host_cache/test_recordings.py
```

`requests.json` 记录输入、生成结果、首 token 延迟及整轮耗时。各 rank 的 `rank-*.jsonl` 包括块表、前缀位置、传输和内存采样。测试会检查每次命中的三个状态组是否都实际读回，回收后设备块和 Host 引用是否释放，以及紧预算下的淘汰和重算。

[汇总结果](results.json)中每组对应一次独立启动，预热不计。吞吐以输出 token 总数除以各轮总耗时，包含请求结束到设备块可重新分配的时间。搬运的字节数和流上事件耗时按 rank 单列；不同流上的时间可能重叠。HBM 为轮末采样，Host 内存为预分配的页锁定缓冲区；另列每轮实际占用的父块数。
