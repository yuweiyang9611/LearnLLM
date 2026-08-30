# LearnLLM v0.2.0

v0.2.0 把课程的训练主线从相互独立的演示脚本，升级为可追溯的 `Pretrain -> Full SFT / LoRA-SFT` 对照流水线。Full SFT 与 LoRA-SFT 现在从同一个预训练 base 独立分叉，共用冻结的模型配置和字符 Tokenizer。

## 获取方式

- [在线阅读 LearnLLM](https://yuweiyang9611.github.io/LearnLLM/)
- [下载 v0.2.0 中文学习指导 PDF](https://github.com/yuweiyang9611/LearnLLM/releases/download/v0.2.0/LearnLLM-v0.2.0-Chinese-Study-Guide.pdf)
- [查看 v0.2.0 源码](https://github.com/yuweiyang9611/LearnLLM/tree/v0.2.0)

## 训练与教学改进

- 实验 06 生成后续训练共用的真实预训练 checkpoint，其中包含模型 config、冻结 Tokenizer、数据 SHA-256 和训练 metadata。
- 实验 10 对比 pretrained base、random-init Full SFT、pretrained Full SFT 与 pretrained LoRA-SFT。
- 指令数据按 intent family 划分为 train/dev/test，并标注 seen、paraphrase 与 new-intent；test 只在训练结束后评测。
- 默认运行三个 seed，报告 assistant loss、严格任务成功率、关键词/格式准确率及均值与总体标准差，并保留原语料保持度代理。
- 每次运行写出版本化 JSON manifest 与长格式 CSV，包含环境、数据/产物 SHA-256、训练曲线、逐样本结果和生成失败案例；CI 会上传这些结果。
- Full SFT 保存完整模型产物；LoRA-SFT 保存 adapter-only 产物，并验证重载后输出一致。
- 新增严格 artifact API；实验 07 可从完整 checkpoint，或从匹配的 base + LoRA adapter 在独立进程中直接生成。
- 实验 08 保留为小型 Linear 上的 LoRA 原理演示，并复用正式 SFT 序列构建逻辑说明 assistant-only mask 的右移边界。

## 在线学习与可复现性

- 新增 GitHub Pages 课程站，包含中文全文搜索、数学公式、代码复制、响应式导航和深浅色主题。
- 学习指导 PDF 现在嵌入源文件摘要，CI 会发现已跟踪 PDF 与 Markdown/构建脚本不一致的情况。
- CI 在 Windows Python 3.11/3.12 上运行完整验证，并在 Ubuntu 上构建 wheel、执行非编辑安装与快速训练链路测试。

## 升级注意

旧版 `checkpoints/tiny_gpt.pt` 不包含 v0.2.0 要求的完整 metadata。请先重新运行实验 06，再运行实验 10：

```powershell
.\.venv\Scripts\python.exe .\experiments\06_train_tiny_gpt.py --quick
.\.venv\Scripts\python.exe .\experiments\10_sft_tiny_gpt.py --quick
```

独立加载实验 10 生成的 LoRA adapter：

```powershell
.\.venv\Scripts\python.exe .\experiments\07_generate.py `
  --base-checkpoint .\checkpoints\tiny_gpt.pt `
  --adapter .\checkpoints\tiny_gpt_lora_adapter.pt `
  --prompt "为什么需要因果掩码？"
```

这是用于理解机制的 TinyGPT，严格任务成功率可能为 0%；该结果应作为失败案例保留，而不是通过放宽指标隐藏。

完整变更见 [CHANGELOG.md](https://github.com/yuweiyang9611/LearnLLM/blob/v0.2.0/CHANGELOG.md)。课程文档与 PDF 按 CC BY-NC-SA 4.0 许可；原创代码按 MIT 许可。
