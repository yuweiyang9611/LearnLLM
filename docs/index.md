---
title: LearnLLM
description: 从 token、Attention 与 Transformer 出发，在 Windows 和 CPU 上亲手实现并验证一个微型大语言模型。
hide:
  - toc
---

<section class="llm-hero" markdown>

<span class="llm-eyebrow">从原理到可复现实验</span>

# 从 token 到 Transformer，亲手走完 LLM 的完整数据流

LearnLLM 是一套面向初学者的中文课程。你不需要 GPU，也不需要 API Key；只需 Python、Windows 和一颗愿意追问“为什么”的脑袋。

[开始学习](总学习指导.md){ .md-button .md-button--primary }
[搭建实验环境](00_学习路线与环境.md){ .md-button }
[下载中文 PDF](https://github.com/yuweiyang9611/LearnLLM/releases/download/v0.1.0/LearnLLM-v0.1.0-Chinese-Study-Guide.pdf){ .md-button }

<div class="llm-stats" markdown>
  <div><strong>10</strong><span>学习阶段</span></div>
  <div><strong>12</strong><span>可运行实验</span></div>
  <div><strong>20</strong><span>自动化测试</span></div>
  <div><strong>CPU</strong><span>即可完成</span></div>
</div>

</section>

## 你会建立怎样的理解

<div class="grid cards llm-concepts" markdown>

-   **01 · 看懂数据流**

    文本如何变成 token、embedding、Attention、logits，最后生成下一个 token。

-   **02 · 手算关键公式**

    用极小例子掌握 Softmax、交叉熵、缩放点积与因果掩码，而不是只背名词。

-   **03 · 亲手实现模型**

    从 Bigram 到多头注意力，再到能训练、保存和生成文本的 TinyGPT。

-   **04 · 扩展到真实系统**

    分清预训练、SFT、LoRA、RAG 与 Agent，并用指标和失败案例评价它们。

</div>

## 推荐学习路径

<div class="llm-path" markdown>

1. **建立实验室** · 配好隔离环境，学会预测、运行、对照和记录。
2. **掌握最小数学** · 把 logits、概率、loss 和梯度逐行对应到代码。
3. **拆开 Transformer** · 理解 Q/K/V、mask、多头、残差、Norm 与 MLP。
4. **训练 TinyGPT** · 跑通预训练、生成、SFT 与 LoRA 的最小闭环。
5. **构建可评测系统** · 完成 Tiny RAG、本地 Agent 和可复现结课项目。

</div>

## 章节与配套实验

| 在线章节 | 配套实验 | 你要验证的核心问题 |
|---|---|---|
| 00–01 环境与数学 | 实验 00–01 | 环境是否隔离？Softmax 与交叉熵是否算对？ |
| 02 Tokenizer 与语言模型 | 实验 02–03 | 文本怎样编码？Bigram 能学到什么？ |
| 03 注意力与 Transformer | 实验 04–05 | Q/K/V、缩放和因果 mask 怎样协作？ |
| 04–05 模型与预训练 | 实验 06–07 | 训练怎样降低 loss？采样策略改变了什么？ |
| 06 SFT、LoRA 与对齐 | 实验 08、10 | 哪些参数在训练？哪些 token 贡献 loss？ |
| 07 评测、RAG 与 Agent | 实验 09、11 | 系统怎样引用证据、拒答并安全调用工具？ |
| 08 结课项目 | Attention 消融示例 | 怎样把问题、对照、指标和边界写成可复现实验？ |

!!! tip "最有效的学习方式"

    每次运行前先写下预测：tensor 的 shape 是什么、loss 会升还是降、mask 后哪一格必须为 0。代码跑通只是开始，能解释结果才算学会。

!!! info "网页负责阅读，实验需要在本地运行"

    在线站点提供全文搜索、公式渲染和章节导航。需要改参数、训练模型或保存结果时，请克隆 GitHub 仓库并在项目 `.venv` 中运行实验。

## 五分钟启动

```powershell
git clone https://github.com/yuweiyang9611/LearnLLM.git
Set-Location .\LearnLLM
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Python python
powershell -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

[进入总学习指导 →](总学习指导.md){ .llm-next }
