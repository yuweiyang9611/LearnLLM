---
title: LearnLLM
description: 从 token、Attention 与 Transformer 出发，在 Windows 和 CPU 上亲手实现并验证一个微型大语言模型。
hide:
  - navigation
  - toc
  - footer
---

<div class="llm-masthead" markdown>
<div class="llm-masthead-copy" markdown>

# 从零学习大语言模型 { #token-transformer-llm }

九章配十二个本地实验，在 Windows 和 CPU 上亲手实现并验证一个微型大语言模型。每章五步：读、看代码、跑、自测、记一笔；做到哪一步记在这台浏览器里，可以[备份成 JSON](#llm-backup)。
{ .llm-masthead-note }

</div>
<div class="llm-stats" role="group" aria-label="课程概况">
<p class="nb-stat" data-hue="green"><b>9</b><span>章</span></p>
<p class="nb-stat" data-hue="cyan"><b>12</b><span>个本地实验</span></p>
<p class="nb-stat" data-hue="blue"><b>15</b><span>个网页</span></p>
</div>
</div>

<section class="llm-start nb-banner" aria-labelledby="llm-start-label" markdown>
<div class="llm-start-copy" markdown>
<p class="nb-banner-eyebrow" id="llm-start-label">从这里开始</p>
<p class="nb-banner-title">00 学习路线与环境</p>
<p class="nb-banner-note" markdown="span">建一间只属于本项目的 `.venv`，养成先预测、再运行、再对照的习惯。</p>
<p class="llm-start-pdf" markdown="span">想离线读，[下载中文 PDF](https://github.com/yuweiyang9611/LearnLLM/raw/refs/heads/main/output/pdf/LearnLLM-%E4%BB%8E%E9%9B%B6%E5%AD%A6%E4%B9%A0%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.pdf)：由网页的同一份 Markdown 生成，是 main 分支的最新版。</p>
</div>
<p class="llm-start-go" markdown="span">[从 00 开始](00_学习路线与环境.md){ .nb-button .nb-button--solid .llm-continue }</p>
</section>

<div class="llm-run" markdown>

## 五分钟启动 { #_4 .llm-run-label }

```powershell
git clone https://github.com/yuweiyang9611/LearnLLM.git
Set-Location .\LearnLLM
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Python python
powershell -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

在 Windows 的 PowerShell 里运行，需要 Python 3.11 或 3.12；不需要 GPU，也不需要 API Key。想先看全貌，读[总学习指导](总学习指导.md)。
{ .llm-note }

</div>

<section class="llm-section" markdown>

## 章节与配套实验 { #_3 }

按编号读，每章都指向要在本地运行的实验。四个部分各用一种颜色：开始学习绿，核心原理青，训练与对齐蓝，应用与项目紫。
{ .llm-section-note }

<ol class="nb-grid llm-cards" id="_2" markdown>
<li class="nb-card nb-entry" data-part="start" markdown>
<div class="nb-card-head" markdown>
<div markdown>
<p class="nb-card-eyebrow">开始学习 · 第 00 章</p>
<h3 markdown="span">[学习路线与环境](00_学习路线与环境.md)</h3>
</div>
</div>
<div class="nb-card-body" markdown>
<p markdown="span">建一个只属于本项目的 `.venv`，养成先预测、再运行、再对照的习惯。</p>
<p class="nb-card-meta"><span>实验 00</span></p>
</div>
</li>
<li class="nb-card nb-entry" data-part="core" markdown>
<div class="nb-card-head" markdown>
<div markdown>
<p class="nb-card-eyebrow">核心原理 · 第 01 章</p>
<h3 markdown="span">[数学与 PyTorch 基础](01_数学与PyTorch基础.md)</h3>
</div>
</div>
<div class="nb-card-body" markdown>
<p>把 logits、Softmax、交叉熵和一次梯度更新逐行对应到代码。</p>
<p class="nb-card-meta"><span>实验 01</span></p>
</div>
</li>
<li class="nb-card nb-entry" data-part="core" markdown>
<div class="nb-card-head" markdown>
<div markdown>
<p class="nb-card-eyebrow">核心原理 · 第 02 章</p>
<h3 markdown="span">[Tokenizer 与语言模型](02_Tokenizer与语言模型.md)</h3>
</div>
</div>
<div class="nb-card-body" markdown>
<p>文本怎样编码成 token？Bigram 能学到什么，又学不到什么？</p>
<p class="nb-card-meta"><span>实验 02–03</span></p>
</div>
</li>
<li class="nb-card nb-entry" data-part="core" markdown>
<div class="nb-card-head" markdown>
<div markdown>
<p class="nb-card-eyebrow">核心原理 · 第 03 章</p>
<h3 markdown="span">[注意力与 Transformer](03_注意力与Transformer.md)</h3>
</div>
</div>
<div class="nb-card-body" markdown>
<p>Q/K/V、缩放和因果掩码怎样协作，怎样证明模型没有偷看未来。</p>
<p class="nb-card-meta"><span>实验 04–05</span></p>
</div>
</li>
<li class="nb-card nb-entry" data-part="train" markdown>
<div class="nb-card-head" markdown>
<div markdown>
<p class="nb-card-eyebrow">训练与对齐 · 第 04 章</p>
<h3 markdown="span">[模型家族与训练生命周期](04_模型家族与训练生命周期.md)</h3>
</div>
</div>
<div class="nb-card-body" markdown>
<p>从可见信息、输出形式和训练目标，推出三类架构为什么不同。</p>
<p class="nb-card-meta"><span>纸笔实验</span></p>
</div>
</li>
<li class="nb-card nb-entry" data-part="train" markdown>
<div class="nb-card-head" markdown>
<div markdown>
<p class="nb-card-eyebrow">训练与对齐 · 第 05 章</p>
<h3 markdown="span">[迷你 GPT 预训练与生成](05_迷你GPT预训练与生成.md)</h3>
</div>
</div>
<div class="nb-card-body" markdown>
<p>训练怎样降低 loss？checkpoint 为什么必须冻结 config 与 Tokenizer？</p>
<p class="nb-card-meta"><span>实验 06–07</span></p>
</div>
</li>
<li class="nb-card nb-entry" data-part="train" markdown>
<div class="nb-card-head" markdown>
<div markdown>
<p class="nb-card-eyebrow">训练与对齐 · 第 06 章</p>
<h3 markdown="span">[SFT、LoRA 与对齐](06_SFT_LoRA与对齐.md)</h3>
</div>
</div>
<div class="nb-card-body" markdown>
<p>在同一个 base 上比较随机初始化、Full SFT 与 LoRA-SFT 的任务指标和参数成本。</p>
<p class="nb-card-meta"><span>实验 08、10</span></p>
</div>
</li>
<li class="nb-card nb-entry" data-part="apply" markdown>
<div class="nb-card-head" markdown>
<div markdown>
<p class="nb-card-eyebrow">应用与项目 · 第 07 章</p>
<h3 markdown="span">[评测、RAG 与 Agent](07_评测_RAG与Agent.md)</h3>
</div>
</div>
<div class="nb-card-body" markdown>
<p>系统怎样引用证据、在证据不足时拒答，并安全地调用工具。</p>
<p class="nb-card-meta"><span>实验 09、11</span></p>
</div>
</li>
<li class="nb-card nb-entry" data-part="apply" markdown>
<div class="nb-card-head" markdown>
<div markdown>
<p class="nb-card-eyebrow">应用与项目 · 第 08 章</p>
<h3 markdown="span">[结课项目](08_结课项目.md)</h3>
</div>
</div>
<div class="nb-card-body" markdown>
<p>把问题、对照、指标和局限写成一份别人能复现的实验报告。</p>
<p class="nb-card-meta"><span>Attention 消融示例</span></p>
</div>
</li>
</ol>

</section>

<section class="llm-section" markdown>

## 配套资料 { #llm-materials }

<ul class="llm-rows" markdown>
<li class="llm-row" markdown>
<p class="llm-key">导读</p>
<h3 class="llm-title" markdown="span">[总学习指导](总学习指导.md)</h3>
<p class="llm-note">学完应该会什么、十个阶段的路线、八周节奏，以及结课时怎样自检。</p>
<p class="llm-state">十个阶段</p>
</li>
<li class="llm-row" markdown>
<p class="llm-key">模板</p>
<h3 class="llm-title" markdown="span">[实验记录模板](实验记录模板.md)</h3>
<p class="llm-note">每次实验照着填：问题、运行前预测、控制变量、结果、解释和结论边界。</p>
<p class="llm-state"></p>
</li>
<li class="llm-row" markdown>
<p class="llm-key">对照</p>
<h3 class="llm-title" markdown="span">[PDF 章节对照与勘误](PDF_章节对照与勘误.md)</h3>
<p class="llm-note">《Happy-LLM v1.0》的物理页对照、公式索引和原书示例勘误。</p>
<p class="llm-state">163 个物理页</p>
</li>
<li class="llm-row" markdown>
<p class="llm-key">版本</p>
<h3 class="llm-title" markdown="span">[v0.2.0 发布说明](RELEASE_v0.2.0.md) · [v0.1.0](RELEASE_v0.1.0.md)</h3>
<p class="llm-note">每个版本改了什么，升级时要注意什么。</p>
<p class="llm-state"></p>
</li>
</ul>

</section>

<section class="llm-section" markdown>

## 运行前要知道 { #_1 }

<ul class="llm-notes" markdown>
<li markdown>**先写预测，再按 Enter。**写下 tensor 的 shape、loss 会升还是降、mask 后哪一格必须为 0。代码跑通只是开始，能解释结果才算学会。</li>
<li markdown><strong class="llm-warn">注意：</strong>实验 10 依赖实验 06。先运行实验 06 生成 `checkpoints/tiny_gpt.pt`，实验 10 会复用其中的预训练权重、config 和冻结的 Tokenizer。分支怎样比较、指标怎样读，见[第 06 章](06_SFT_LoRA与对齐.md)。</li>
<li markdown>**网页上能跑的只有三个实验。**第 02 章的实验 03、第 07 章的实验 09 和 11 不用 torch，面板里的「运行」把它们交给浏览器里的 Python 运行时；其余实验要 PyTorch，面板写着「需要本机」。训练模型或保存结果，请克隆仓库，在项目的 `.venv` 中运行。</li>
</ul>

</section>

<section class="llm-section" aria-labelledby="llm-backup-title" markdown>

## 学习记录 { #llm-backup }

<div class="llm-backup" id="llm-backup-panel" markdown>
<p class="llm-note" id="llm-backup-title" markdown="span">每章的五步做到哪里、自测和笔记，都只存在这台浏览器里。换电脑或清缓存前先备份；恢复会覆盖这里的记录。</p>
<p class="llm-backup-actions"><button type="button" class="nb-button" id="llm-export" disabled>备份 JSON</button><label class="nb-button" for="llm-import" aria-disabled="true">恢复<input type="file" id="llm-import" accept="application/json,.json" disabled></label><button type="button" class="nb-button" id="llm-reset" disabled>清空记录</button></p>
<p class="llm-backup-status" role="status" aria-live="polite"></p>
</div>

</section>
