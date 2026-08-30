# Changelog

本项目遵循语义化版本。这里只记录对学习者可见的课程、实验和复现方式变化。

## [Unreleased]

## [0.2.0] - 2026-08-30

### Changed

- 实验 06 现在生成后续训练共用的真实预训练 checkpoint，并冻结其中的模型配置与字符 Tokenizer；预训练词表会覆盖 train/dev/test 指令字符。
- 实验 10 改为从同一个 base 独立比较 pretrained base、random-init Full SFT、pretrained Full SFT 与 pretrained LoRA-SFT，并用可配置 CLI、随机小批次和默认三个 seed 运行。
- 实验 08 明确定位为小型 Linear 上的 LoRA 原理演示；真实 TinyGPT LoRA-SFT 由实验 10 完成。

### Added

- 增加按 intent family 划分的 train/dev/test 指令集以及 seen/paraphrase/new-intent 标签；同时报告 assistant loss、严格任务成功率、关键词/格式准确率、原语料保持度代理、可训练参数量和耗时。
- 增加版本化 checkpoint/LoRA artifact 加载 API；实验 07 可在新进程中直接使用 base checkpoint 与 adapter 生成，并严格验证格式、配置、Tokenizer、base/data SHA-256 和 LoRA 超参数。
- 增加版本化 `sft_comparison.json` 运行清单与长格式 CSV，记录环境、Git 状态、数据和产物摘要、逐步 loss、逐样本结果、生成失败案例及多 seed 均值/总体标准差；CI 会上传这两项产物。
- Full SFT 保存完整模型权重与 metadata（不含 optimizer 状态）；LoRA-SFT 保存 adapter-only 产物，并记录 base SHA-256、冻结 Tokenizer、模型配置、目标层、rank 与 alpha，且验证恢复后输出一致。
- base 与微调产物记录数据 SHA-256、步数、学习率、优化器配置和最终指标；实验 10 会拒绝旧 checkpoint 与新版语料混用。
- 增加 GitHub Pages 中文在线学习站，支持章节导航、全文搜索、数学公式、代码复制与深浅色主题。
- 增加 tag 驱动的 Release 工作流，校验版本与 CHANGELOG，从对应 tag 构建 PDF 并发布 SHA-256 校验和。
- 增加 Ubuntu wheel 构建、非编辑安装和快速训练链路验证，并上传 wheel 作为 CI 产物。
- PDF 嵌入源文件摘要，CI 会在重建前验证已跟踪 PDF 的结构与来源一致性。

## [0.1.0] - 2026-08-27

### Added

- 从数学、Tokenizer、Attention 到 TinyGPT 的十阶段中文课程。
- CPU 可运行的预训练、生成、LoRA、SFT、RAG 与本地 Agent 实验。
- 自动测试、Windows CI、隔离环境引导和 PDF 构建验证。
- 完整学习指导 PDF、结课项目评分量表与 Attention 消融示例报告。
- 代码/文档双许可证、Happy-LLM 署名说明和八周 GitHub 里程碑。

### Safety and scope

- 必做实验不调用在线模型 API，不需要 API Key 或 GPU。
- 所有模型与应用均为教学玩具，用于验证机制，不代表生产能力或安全性。

[Unreleased]: https://github.com/yuweiyang9611/LearnLLM/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yuweiyang9611/LearnLLM/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yuweiyang9611/LearnLLM/releases/tag/v0.1.0
