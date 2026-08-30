# Changelog

本项目遵循语义化版本。这里只记录对学习者可见的课程、实验和复现方式变化。

## [Unreleased]

### Changed

- 实验 06 现在生成后续训练共用的真实预训练 checkpoint，并冻结其中的模型配置与字符 Tokenizer；预训练词表会覆盖训练及 heldout 指令字符。
- 实验 10 改为从同一个 base 独立比较 pretrained base、random-init Full SFT、pretrained Full SFT 与 pretrained LoRA-SFT，不再重新创建随机小模型冒充微调。
- 实验 08 明确定位为小型 Linear 上的 LoRA 原理演示；真实 TinyGPT LoRA-SFT 由实验 10 完成。

### Added

- 增加未参与 SFT 的 heldout 指令集（用于观察指令格式迁移，不作为未知知识评测），并在对比中同时报告训练/SFT-heldout assistant loss、原语料保持度/干扰代理、可训练参数量、耗时和生成样例。
- Full SFT 保存完整模型权重与 metadata（不含 optimizer 状态）；LoRA-SFT 保存 adapter-only 产物，并记录 base SHA-256、冻结 Tokenizer、模型配置、目标层、rank 与 alpha，且验证恢复后输出一致。
- base 与微调产物记录数据 SHA-256、步数、学习率、优化器配置和最终指标；实验 10 会拒绝旧 checkpoint 与新版语料混用。

## [0.1.0] - 2026-07-16

### Added

- 从数学、Tokenizer、Attention 到 TinyGPT 的十阶段中文课程。
- CPU 可运行的预训练、生成、LoRA、SFT、RAG 与本地 Agent 实验。
- 自动测试、Windows CI、隔离环境引导和 PDF 构建验证。
- 完整学习指导 PDF、结课项目评分量表与 Attention 消融示例报告。
- 代码/文档双许可证、Happy-LLM 署名说明和八周 GitHub 里程碑。

### Safety and scope

- 必做实验不调用在线模型 API，不需要 API Key 或 GPU。
- 所有模型与应用均为教学玩具，用于验证机制，不代表生产能力或安全性。
