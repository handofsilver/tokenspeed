# Qwen3.5 on Ascend 910C：双 chip 张量并行

AI Computing Systems 课程第二次作业。在同一张 Ascend 910C 卡的两个 chip 上，分别运行 Qwen3.5-0.8B、4B、9B 和 27B，使用 TokenSpeed 的原生 Engine 和 HCCL 实现 TP=2。

实验比较 TP1 与 TP2 的推理结果、延迟、吞吐和显存占用，并分析各层的权重切分、计算与通信时间线。

- [环境、模型准备与运行命令](examples/aics_qwen35_tp2/README.md)
- [模型分片与通信位置](docs/recipes/qwen35-tp2-sharding.md)
- [实验结果](examples/aics_qwen35_tp2/results.json)
- [第一次作业：单 chip 适配](https://github.com/handofsilver/tokenspeed/tree/hw1-qwen35-ascend)

## 上游项目

代码基于 [lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed) 的 `cf898936350f46ac6d0443b154739beaf51fe8a0` 版本，沿用上游的 [MIT 许可证](LICENSE)。

- [TokenSpeed 文档](https://lightseek.org/tokenspeed/)
- [安装与入门](https://lightseek.org/tokenspeed/guides/getting-started)
- [启动服务](https://lightseek.org/tokenspeed/guides/launching)
- [并行配置](https://lightseek.org/tokenspeed/serving/parallelism)
