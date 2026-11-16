# Qwen3.5 在 Ascend 910C 上的分层缓存

AI Computing Systems 第三次作业。在 TokenSpeed 的 Host 缓存上适配了 Ascend 的状态恢复，并用 Qwen3.5-0.8B 和 9B 比较两种做法：释放设备缓存后重新计算前缀，或从 Host 读回前缀再继续推理。

- [实验设置与复现命令](examples/aics_host_cache/README.md)
- [各组测量结果](examples/aics_host_cache/results.json)
- [第一次作业 · 单 chip 适配](https://github.com/handofsilver/tokenspeed/tree/hw1-qwen35-ascend)
- [第二次作业 · 双 chip 张量并行](https://github.com/handofsilver/tokenspeed/tree/hw2-qwen35-tp2)

上游项目为 [lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed)，沿用其 [MIT 许可证](LICENSE)。
