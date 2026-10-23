# Ascend 实验环境

本次实验使用课程服务器预装的 Python/CANN 环境，并在项目虚拟环境中补充依赖。
以下版本记录于 2026-10-15；重建时需要相同的基础环境。

| 组件 | 版本 |
| --- | --- |
| 架构 / Python | Linux aarch64 / 3.11.16 |
| Ascend 驱动 / CANN | 25.5.2 / 9.0.0 |
| PyTorch / torch_npu | 2.9.0+cpu / 2.9.0 |
| Transformers / NumPy | 5.17.0 / 2.4.6 |
| Triton-Ascend | distribution 3.2.1；`triton.__version__` 为 3.2.0 |
| apache-tvm-ffi / tokenspeed-scheduler | 0.1.13.post3 / 0.1.37 |
| compressed-tensors / OpenAI SDK | 0.15.0 / 2.25.0 |
| pytest / pip / setuptools | 8.4.2 / 24.0 / 79.0.1 |

PyTorch 的 NPU 支持由 `torch_npu` 提供。`+cpu` 是该 PyTorch 包的版本后缀，
不代表模型在主机 CPU 上推理。

## 安装与启动

在服务器的仓库根目录执行。这里继承 `/opt/miniconda3/envs/aics` 中的课程基础包，
只向项目 `.venv` 安装覆盖包，不修改共享环境：

```bash
/opt/miniconda3/envs/aics/bin/python -m venv --system-site-packages .venv
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source .venv/bin/activate

python -m pip install --no-deps 'pip==24.0' 'setuptools==79.0.1'
python -m pip install --no-deps \
  -r examples/aics_qwen35/requirements-overlay.txt
python -m pip install --no-deps \
  --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  'triton-ascend==3.2.1'

export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export TOKENSPEED_MAMBA_SSM_DTYPE=float32
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

`requirements-overlay.txt` 固定了 43 项覆盖包。`--no-deps` 用于恢复这次实验的版本
组合；依赖基础环境提供的包不会重新安装。CANN 的 `set_env.sh` 会设置 Python 路径，
所以追加源码路径时保留原有 `PYTHONPATH`。TokenSpeed 和两个 kernel 包直接从源码
导入，不使用上游 `pyproject.toml` 的完整默认安装组合。

每次重新登录后，仍需加载 CANN、激活 `.venv` 并设置上述环境变量。模型下载、
芯片选择与运行命令见[运行说明](README.md)。

## 已知依赖问题

该环境的 `pip check` 返回 1，报告 Triton-Ascend 3.2.1 的八项声明依赖不满足：

| 声明依赖 | 实际状态 |
| --- | --- |
| triton==3.5.0 | 无独立 `triton` distribution |
| pytest-xdist==3.6.1 | 未安装 |
| attrs==24.2.0 | 26.1.0 |
| decorator==5.1.1 | 5.3.1 |
| numpy==1.26.4 | 2.4.6 |
| psutil==6.0.0 | 7.2.2 |
| pytest==8.3.2 | 8.4.2 |
| scipy==1.13.1（Python < 3.13） | 1.17.1 |

Ascend wheel 自带 `triton/` 和 Ascend 编译器，但元数据仍声明依赖独立
`triton==3.5.0`。安装普通 Triton 会与它覆盖同一目录，因此没有用这种方式消除
报错。测试也未使用 pytest-xdist 的并行执行功能。

模型比较和定向算子测试在上述组合下完成；其余依赖功能未全面验证。共享基础环境
若发生变化，应重新保存版本清单、运行 `pip check`，再做算子和模型回归。
这份覆盖包清单不是独立操作系统或 CANN 环境的锁文件。

本次使用的 [Triton-Ascend wheel](https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl)
大小为 188451952 bytes，SHA-256 为：

```text
1467b81425dc239f495a460b5c3d6ff1b035c117185791ec41c0bf3d7bb1fd7a
```
