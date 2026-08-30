# LearnLLM：从零理解并实现大语言模型

这是一套面向初学者、可以在 Windows + CPU 上完成的 LLM 原理课程。内容以 `Happy-LLM-v1.0.pdf` 的知识主线为参考，但补齐了原书默认读者已经掌握的 Python、数学、PyTorch 与实验方法，并把大规模训练改造成可验证的小实验。

本项目的目标不是调用一次现成 API，而是让你能解释并亲手验证这条完整数据流：

```text
文本 -> token -> token id -> embedding + position
     -> 多层因果 Transformer -> logits -> Softmax
     -> 下一个 token -> 自回归生成
```

## 在线学习

课程站点已发布到 [LearnLLM GitHub Pages](https://yuweiyang9611.github.io/LearnLLM/)，支持中文全文搜索、章节导航、深浅主题、代码复制和数学公式渲染。网页适合阅读与复习；需要运行实验时，再按下方步骤克隆仓库并创建隔离环境。

## 从新克隆开始

准备 Windows、Git 和 Python 3.11 或 3.12。下面是新电脑从克隆到验收的完整路径：

```powershell
git clone https://github.com/yuweiyang9611/LearnLLM.git
Set-Location .\LearnLLM

# 创建项目专用 .venv，并把依赖安装到其中
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Python python

# 运行正式验证：自动测试 + 环境隔离 + P1/P2 快速实验
powershell -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

`bootstrap.ps1` 不会向系统 Python 安装包；创建环境之后，所有命令都明确使用 `.\.venv\Scripts\python.exe`。如果 `python` 不是目标解释器，可把 `-Python python` 替换为 Python 3.11/3.12 的 `python.exe` 完整路径。重复运行 bootstrap 是安全的，它会复用当前项目的 `.venv`。

环境准备好后，可以从前六个原理实验开始：

```powershell
.\.venv\Scripts\python.exe .\experiments\00_environment_check.py
.\.venv\Scripts\python.exe .\experiments\01_math_foundations.py
.\.venv\Scripts\python.exe .\experiments\02_tokenization.py
.\.venv\Scripts\python.exe .\experiments\03_bigram_language_model.py
.\.venv\Scripts\python.exe .\experiments\04_attention.py
.\.venv\Scripts\python.exe .\experiments\05_transformer_block.py
```

进入训练阶段后，实验存在明确的上游关系：先运行实验 06 生成真实预训练 checkpoint，并把其中的模型配置与字符 Tokenizer 作为后续阶段不可随意重建的基线；再运行实验 10，让 Full SFT 与 LoRA-SFT 从同一个 base 独立分叉，并加入 random-init SFT 作为额外基线。

```powershell
# 先生成 checkpoints/tiny_gpt.pt
.\.venv\Scripts\python.exe .\experiments\06_train_tiny_gpt.py --quick

# 使用同一份 base config、权重和冻结 Tokenizer 比较四组结果；默认 3 个 seed
.\.venv\Scripts\python.exe .\experiments\10_sft_tiny_gpt.py --quick
```

实验 10 使用可机器读取的 4/4/4 切分：train 四条是 `seen`，dev 四条是相同意图族的 `paraphrase`，test 四条包含两条 `paraphrase` 和两条 `new_intent`。只用 dev 选择步数、学习率或 LoRA 配置；每个分支和 seed 的 test 在训练结束后才评一次。默认运行 seed 42/43/44，并对 assistant loss、严格任务成功率、关键词准确率、格式准确率和原语料验证 loss（保持度/干扰代理）报告均值与总体标准差，同时逐 seed 记录可训练/总参数量与耗时。`new_intent` 只是数据切分标签，不等于预训练时未知的知识，不能据此宣称未知知识泛化；Tiny 字符模型得到 0% 严格任务成功率也可能是诚实结果。

每次运行还会在唯一的 `outputs/sft/<时间>-seeds-.../` 目录写入 `sft_comparison.json` manifest 与长格式 `sft_comparison.csv` 训练历史。路径、seed、步数、三种学习率、数据文件、LoRA rank/alpha/dropout、生成长度、batch size、设备以及完整模型/adapter 输出都可通过 CLI 配置；用 `--help` 查看全部选项。实验 06/10 用数据 SHA-256 拒绝旧 checkpoint 与新语料混用。严格 artifact loader 会校验版本、config、冻结 Tokenizer、base/data SHA-256、LoRA targets 与 tensor 键/形状，再恢复模型。实验 08 仍是独立的小型 Linear 原理演示；真实 adapter 由实验 10 训练并保存。

实验 10 完成后，可以让实验 07 在新的进程中独立加载 base + adapter，而不是依赖内存中的训练模型：

```powershell
.\.venv\Scripts\python.exe .\experiments\07_generate.py `
  --base-checkpoint .\checkpoints\tiny_gpt.pt `
  --adapter .\checkpoints\tiny_gpt_lora_adapter.pt `
  --prompt "为什么需要因果掩码？" --tokens 80
```

## 推荐阅读顺序

1. [总学习指导](LEARNING_GUIDE.md)：先理解目标、节奏和验收方式。
2. [学习路线与环境](docs/00_学习路线与环境.md)：完成环境、Python 和张量预备。
3. [数学与 PyTorch 基础](docs/01_数学与PyTorch基础.md)：Softmax、交叉熵、梯度。
4. [Tokenizer 与语言模型](docs/02_Tokenizer与语言模型.md)：从字符、BPE 到 Bigram。
5. [注意力与 Transformer](docs/03_注意力与Transformer.md)：Q/K/V、因果掩码、多头与残差。
6. [模型家族与训练生命周期](docs/04_模型家族与训练生命周期.md)：BERT、T5、GPT，Pretrain、SFT、偏好对齐。
7. [迷你 GPT：预训练与生成](docs/05_迷你GPT预训练与生成.md)：训练并保存后续微调共用的 Decoder-only base。
8. [SFT、LoRA 与对齐](docs/06_SFT_LoRA与对齐.md)：从同一 base 对比 Full SFT 与 LoRA-SFT，理解“训练什么参数”和“训练什么目标”。
9. [评测、RAG 与 Agent](docs/07_评测_RAG与Agent.md)：从模型到可评测的应用系统。
10. [结课项目](docs/08_结课项目.md)：完成一份可复现的实验报告。

PDF 页码、课程映射以及原资料中需要谨慎处理的代码问题，见 [PDF 章节对照与勘误](docs/PDF_章节对照与勘误.md)。

## 学习指导 PDF

仓库内包含已经生成的 [《从零学习 LLM：大语言模型原理、实验与项目实践》](output/pdf/LearnLLM-从零学习大语言模型.pdf)。需要从 Markdown 讲义重建时，只把额外依赖安装到项目 `.venv`：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Python python -WithDocs
.\.venv\Scripts\python.exe .\scripts\build_learning_guide_pdf.py
.\.venv\Scripts\python.exe .\scripts\verify_pdf.py
```

PDF 依赖固定在 `requirements-docs.txt`，不属于基础实验的必装项。构建脚本在 Windows 上默认使用微软雅黑和宋体；如果字体位于别处，可以通过命令行覆盖：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_learning_guide_pdf.py `
  --font-regular "D:\Fonts\regular.ttf" `
  --font-bold "D:\Fonts\bold.ttf" `
  --font-code "D:\Fonts\code.ttf"
```

也可以设置 `LEARNLLM_FONT_REGULAR`、`LEARNLLM_FONT_BOLD`、`LEARNLLM_FONT_CODE` 和可选的 `LEARNLLM_FONT_MATH` 环境变量。缺少必需字体时，脚本会列出缺失路径和覆盖方法。

## 自动化验证

GitHub Actions 在 Windows 上分别使用 Python 3.11 和 3.12 执行 `scripts/verify.ps1`，覆盖至少 46 项单测、venv 隔离、RAG、SFT、Agent 与结课示例；独立的 Ubuntu + Python 3.12 任务会构建 wheel、非 editable 安装该 wheel，并运行单测和实验 06→10 快速链路。Windows 的另一项任务会从讲义重建 PDF、运行结构和文本检查并上传产物。工作流定义见 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)。

## 实验地图

| 实验 | 预计时间 | 核心问题 | 通过标准 |
|---|---:|---|---|
| 00 环境检查 | 2 分钟 | 我是否真的在项目 venv 中？ | 解释器位于 `.venv`，随机数可复现 |
| 01 数学基础 | 20-40 分钟 | logits 如何变成训练信号？ | 稳定 Softmax；手写 CE 与 PyTorch 一致 |
| 02 Tokenization | 30-60 分钟 | 模型看到的是字、词还是别的？ | 字符编码可往返；观察 BPE 合并 |
| 03 Bigram LM | 30-60 分钟 | “预测下一个 token”究竟是什么？ | 困惑度低于均匀随机基线 |
| 04 Attention | 45-90 分钟 | Q/K/V 如何混合信息？ | 权重行和为 1；未来权重为 0 |
| 05 Transformer | 60-120 分钟 | 一个 Decoder Block 如何工作？ | 形状正确；无未来泄漏 |
| 06 训练 TinyGPT | 10-30 分钟运行 | 训练循环如何让 loss 下降？ | loss 明显下降；保存真实 base checkpoint 与冻结 Tokenizer |
| 07 生成与采样 | 30-60 分钟 | temperature/top-k 改变什么？ | 固定种子可复现并比较策略 |
| 08 LoRA 原理 | 45-90 分钟 | 低秩旁路为什么能只训练少量参数？ | 小型 Linear 原权重冻结；A/B 有梯度；合并等价 |
| 09 Tiny RAG | 45-90 分钟 | 如何让检索结果变成可核对的回答？ | 正确召回；带来源；资料外问题拒答 |
| 10 TinyGPT SFT 对比 | 45-90 分钟 | 预训练与参数高效微调各带来什么？ | 四分支共用 config/Tokenizer；隔离 train/dev/test；记录任务/关键词/格式指标、保持度代理和参数量；adapter 可独立恢复 |
| 11 本地 Agent | 45-90 分钟 | 工具、状态和停止条件如何组成控制循环？ | 白名单、错误状态与最大步数均通过 |

时间是“边读边做”的估计；脚本本身通常只运行数秒，TinyGPT CPU 训练视步数而定。

## 学习规则

- 运行前先写下预测：shape 是什么、loss 会升还是降、mask 后哪一格应为 0。
- 一次只改一个变量：例如只去掉 `sqrt(d_k)`，不要同时改维度和随机种子。
- 每次实验记录“现象、解释、失败案例”，不要只保存成功输出。可复制 [实验记录模板](docs/实验记录模板.md)。

## 重要边界

- 这里的 TinyGPT 用于证明原理，不具备生产级模型的知识、语言能力或安全性。
- Softmax 概率不是事实正确率；注意力权重也不是完整、可靠的推理解释。
- 必做实验不下载预训练模型、不需要 API Key、不需要 GPU。
- FlashAttention、DeepSpeed、完整 RLHF、海量预训练和在线大模型 API 被列为进阶方向，不是入门成功的前提。

## 许可与署名

本仓库采用双许可：原创软件代码使用 [MIT License](LICENSE-CODE)，课程文档、教学数据与生成的 PDF 使用 [CC BY-NC-SA 4.0](LICENSE-DOCS)。根目录 [LICENSE](LICENSE) 说明适用范围。

课程参考 [Datawhale Happy-LLM](https://github.com/datawhalechina/happy-llm) 的知识主线，但已面向 Windows + CPU 初学者重新组织，讲解、实验、数据、测试和构建系统均为本项目重新设计。完整来源、上游协议、修改说明与无背书声明见 [ATTRIBUTION.md](ATTRIBUTION.md)。传播文档或 PDF 时请保留署名、标明修改，并遵守非商业和相同方式共享条件。
