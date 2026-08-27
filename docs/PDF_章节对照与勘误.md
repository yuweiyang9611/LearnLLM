# 《Happy-LLM v1.0》章节对照、公式索引与勘误

> 本文中的“物理页”指 PDF 文件从 1 开始计数的实际页序号，不是正文中可能出现的章节页码。PDF 共 163 个物理页。
>
> 本项目是参考《Happy-LLM v1.0》知识脉络重新设计的原创教学改编。项目中的讲解、代码、实验、测试和验收标准均为重新组织与实现，不是对 PDF 正文或源码的逐字复制。PDF 页码仅用于回查主题；当 PDF 示例与本项目验证结果不一致时，以本项目中通过测试的实现和本文勘误为准。

## 1. 全书物理页对照

| PDF 物理页 | 原书内容 | 在本课程中的作用 |
| --- | --- | --- |
| 1-6 | 项目介绍、前言、学习建议 | 了解原书目标；本项目另行补足零基础数学、环境隔离和可验证实验方法 |
| 7-14 | 第 1 章：NLP 基础概念 | 建立 token、词表、语言模型和文本表示的直觉 |
| 15-31 | 第 2 章：Transformer 架构 | 课程 04-05 的主要理论来源：注意力、掩码、多头、归一化、残差和位置编码 |
| 32-57 | 第 3 章：预训练语言模型 | 比较 Encoder-only、Encoder-Decoder、Decoder-only 及 BERT、T5、GPT、LLaMA、GLM |
| 58-73 | 第 4 章：大语言模型 | 理解 Pretrain、SFT、RLHF 的目标差异，以及数据、算力和对齐问题 |
| 74-116 | 第 5 章：动手搭建大模型 | 从 RMSNorm、GQA、RoPE、MLP 到 Tokenizer、Dataset、预训练和 SFT 的综合参考 |
| 117-142 | 第 6 章：大模型训练流程实践 | Transformers、Trainer、DeepSpeed、Adapter、Prefix Tuning、LoRA |
| 143-163 | 第 7 章：大模型应用 | 评测、RAG 和 Agent；适合作为完成核心实验后的扩展项目 |

### 1.1 与核心实验最相关的细分页码

| PDF 物理页 | 主题 | 建议回查时关注什么 |
| --- | --- | --- |
| 12-13 | 语言模型、Word2Vec | “根据上下文预测 token”和向量点积的最早直觉 |
| 15-18 | Q、K、V 与缩放点积注意力 | 从相似度、Softmax 到 Value 加权和 |
| 18-19 | 自注意力与因果掩码 | 为什么未来位置必须被遮挡，为什么用 `-inf` |
| 19-21 | 多头注意力 | 每个头的投影、并行计算、拼接和输出投影 |
| 21-25 | Encoder-Decoder、FFN、LayerNorm、残差 | Transformer Block 的组成部件 |
| 25-30 | Embedding、位置编码、完整 Transformer | token id 如何进入网络，以及位置信息如何注入 |
| 58-61 | LLM 的定义、能力和局限 | 上下文学习、指令遵循、推理、长文本、幻觉 |
| 62-65 | Pretrain、并行训练、数据清洗 | CLM 目标、数据/模型并行、过滤与去重 |
| 66-69 | SFT | 指令数据、多轮对话，以及只在 assistant 回复上计算损失 |
| 69-72 | RLHF | Reward Model、PPO 的四模型结构、KL 约束和 DPO 思路 |
| 74-87 | 手写 LLaMA2 | RMSNorm、GQA、RoPE、SwiGLU、DecoderLayer、采样 |
| 87-103 | Tokenizer 与 Dataset | BPE、特殊 token、X/Y 右移、预训练和 SFT loss mask |
| 103-112 | 预训练与 SFT 循环 | 学习率、梯度累积、裁剪、AMP、checkpoint |
| 135-142 | 高效微调与 LoRA | 低秩旁路、冻结基座、目标层、PEFT 配置 |
| 148-154 | Tiny-RAG | 切分、向量化、余弦相似度、top-k 检索和上下文生成 |

## 2. 课程实验相对前一阶段增加了什么

课程刻意把原书中的大模型拆成六个可以在本机验证的小台阶。每个实验只引入少量新概念，并保留上一实验已经验证过的能力。

| 实验 | 相对前一阶段新增的能力 | 主要不变量或验收点 | PDF 物理页 |
| --- | --- | --- | --- |
| `experiments/00_environment_check.py` | 建立项目内 `.venv`，确认解释器、依赖、设备和随机种子可复现 | Python 位于 `.venv`；NumPy/PyTorch 可导入；CPU 也通过；同种子结果一致 | 7-30 的实验前置 |
| `experiments/01_math_foundations.py` | 从普通数值进入张量计算：点积、矩阵乘法、数值稳定 Softmax、交叉熵 | shape 正确；Softmax 每行和为 1；极大 logits 不产生 `nan`；正确类别概率增大时 CE 下降 | 12-13、16-20 |
| `experiments/02_tokenization.py` | 把字符串变成 token/id，并能从 id 还原；显式处理词表和未知字符 | 编码结果为整数序列；已知文本往返一致；未知字符行为有定义 | 7-13；扩展阅读 87-93 |
| `experiments/03_bigram_language_model.py` | 从静态 token 映射进入最小“预测下一个 token”模型；增加 NLL、PPL 与采样 | 概率归一；NLL/PPL 可解释；固定种子采样可复现；模型仅使用前一个 token | 12-13 |
| `experiments/04_attention.py` | 用 Q、K、V 替代 bigram 查表；增加缩放点积注意力和因果掩码 | 权重和为 1；输出 shape 正确；被遮挡权重为 0；修改未来 token 不改变过去位置 | 15-20 |
| `experiments/05_transformer_block.py` | 在单头注意力上增加多头拆分/合并、残差、归一化和 FFN，组成 Decoder Block | `D % H == 0`；输入输出均为 `(B,T,D)`；因果性不被多头和 FFN 破坏；反向传播有有限梯度 | 19-30 |

这里选择 Decoder-only Block，是为了直接连接现代生成式 LLM。它吸收了原书第 2 章的基础组件，但没有照搬原书的完整机器翻译 Encoder-Decoder 示例。

## 3. 公式索引：保留原书符号，再说明课程中的含义

### 3.1 点积、Softmax 与注意力（物理页 17）

原书先用字典权重给出：

\[
value=0.6\times10+0.4\times5+0\times2=8
\]

向量点积写为：

\[
v\cdot w=\sum_i v_iw_i
\]

单个查询与全部 Key 的相似度：

\[
x=qK^T
\]

Softmax：

\[
\operatorname{softmax}(x)_i=\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

从单查询推广到矩阵，并加入缩放后，核心公式为：

\[
\operatorname{attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

课程 01 验证点积和稳定 Softmax，课程 04 才把它们组合成完整注意力。`\sqrt{d_k}` 用于控制点积随维度增大而增长，减轻 Softmax 过度饱和。

### 3.2 多头注意力（物理页 20）

\[
\operatorname{MultiHead}(Q,K,V)
=\operatorname{Concat}(head_1,\ldots,head_h)W^O
\]

\[
head_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
\]

课程 05 重点不是背公式，而是追踪形状：

```text
(B,T,D) -> (B,H,T,D_h) -> (B,H,T,T)
        -> (B,H,T,D_h) -> (B,T,D)
```

其中 `D_h = D / H`。

### 3.3 归一化（物理页 23）

原书先以批归一化的符号说明均值、方差和标准化：

\[
\mu_j=\frac{1}{m}\sum_{i=1}^{m}Z_j^i
\]

\[
\sigma^2=\frac{1}{m}\sum_{i=1}^{m}(Z_j^i-\mu_j)^2
\]

\[
\widetilde{Z_j}=\frac{Z_j-\mu_j}{\sqrt{\sigma^2+\epsilon}}
\]

LayerNorm 使用同类计算，但统计轴不是 batch 轴，而是每个 token 的特征轴。课程 05 应检查归一化后的均值和方差近似目标值，不应声称数据因此服从正态分布。

### 3.4 正余弦位置编码（物理页 26）

\[
PE(pos,2i)=\sin\left(pos/10000^{2i/d_{model}}\right)
\]

\[
PE(pos,2i+1)=\cos\left(pos/10000^{2i/d_{model}}\right)
\]

这是原始 Transformer 的绝对位置编码。课程 05 的最小 Decoder Block 可以使用更简单的位置处理，但阅读时要区分它与 LLaMA 使用的 RoPE。

### 3.5 训练计算量的历史近似（物理页 63）

原书列出：

\[
C\sim6ND
\]

其中 `C` 为计算量、`N` 为参数量、`D` 为训练 token 数。它适合帮助理解参数、数据和算力之间的关系，不是任何时代、模型和硬件上都固定成立的配方。

### 3.6 RMSNorm（物理页 75）

\[
\operatorname{RMSNorm}(x)
=\frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2+\epsilon}}\cdot\gamma
\]

它不减均值，只按均方根缩放，再乘可学习参数 `\gamma`。这是第 5 章 LLaMA 风格模型与第 2 章 LayerNorm 的重要区别。

### 3.7 LoRA（物理页 137-138）

原权重冻结，更新量写成低秩乘积：

\[
W_0+\Delta W=W_0+BA,
\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k}
\]

前向传播：

\[
h=W_0x+\Delta Wx=W_0x+BAx
\]

原书在方阵近似下给出的可训练参数量为：

\[
\Theta=2\times L_{LoRA}\times d_{model}\times r
\]

更一般地，一个 `d_out x d_in` 线性层增加 `r(d_in+d_out)` 个权重参数。`r` 越小，参数越少，但表达能力也可能受限。

### 3.8 RAG 的余弦相似度（物理页 150）

原书以代码实现点积除以两个向量的模，等价于：

\[
\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}
\]

余弦相似度只衡量向量方向接近程度。RAG 是否给出可靠答案，还取决于切分、Embedding、top-k、提示模板和生成模型，不能只看这个分数。

## 4. 不要盲目照抄：原书示例勘误与课程处理

| PDF 物理页 | 原书示例的风险 | 本项目中的处理 |
| --- | --- | --- |
| 17 | “语义相似的向量点积应大于 0、不相似应小于 0”不是一般数学保证；点积还受向量长度和训练方式影响 | 只把点积解释为当前表示空间中的相似度打分；课程 01 同时观察数值、方向和 Softmax 结果 |
| 18 | `attention(x,x,x)` 只表达 Q/K/V 来自同一序列，省略了可学习的 `W_Q/W_K/W_V` | 课程 04 先讲最小核心；课程 05 显式加入独立投影层 |
| 23 | “归一化后变成标准正态分布”过强；零均值、单位方差不等于服从正态分布 | 验收均值、方差和有限数值，不作分布形状保证 |
| 24 | `EncoderLayer` 示例先用归一化结果覆盖 `x`，随后再将该值当残差，容易丢失真正的原始残差分支 | 保存原输入；采用清晰的 Pre-Norm 结构 `x = x + sublayer(norm(x))` |
| 29-30 | 示例把同一 `x` 同时作为 Encoder 和 Decoder 输入，没有展示右移后的目标序列；`get_num_params(non_embedding=True)` 还访问不存在的 `wpe.weight` | 核心课程不把该片段当作可训练机器翻译实现；课程 05 构造经过单测的 Decoder-only Block |
| 75 | 配置示例默认 `dim=768`，RMSNorm 测试输出却写成末维 `288` | 输出末维必须等于实际输入 `D`；测试直接断言 `output.shape == input.shape` |
| 92、99 | 文中称为正确输出，但实际显示 `Decoded text matches original: False`、`Special tokens preserved: False` | 课程 02 对已知字符要求严格往返；特殊 token 与空格策略必须通过显式测试 |
| 100 | 配置把 `<|im_end|>` 设为 pad token，Dataset 却硬编码 `padding=0`，并依赖 `ignore_index=0`；语义不一致 | 始终读取 `tokenizer.pad_token_id`，并用独立 loss mask 或统一的 `ignore_index` |
| 101 | assistant 起始序列被硬编码为 `[3,1074,537,500,203]`，更换 tokenizer 后会失效 | 由当前 tokenizer 动态编码 assistant 边界，或在结构化消息转换时直接生成 mask |
| 103 以后 | 默认约 215M 模型、多 GPU、DataParallel、CUDA AMP、SwanLab/API key，不适合作为零基础本地首跑；整份大语料 `readlines()` 也可能耗尽内存 | 核心实验默认 CPU、小张量、小语料、无外部登录；先做单 batch 过拟合，再按硬件逐步放大；大数据使用流式读取 |
| 154 | `VectorStore.query` 的定义需要 `EmbeddingModel`，Demo 却传 `model='zhipu'`；本地模型还硬编码 `.cuda()` | 接口统一传入 Embedding 实例；设备通过参数选择，CPU 路径必须可运行 |
| 多处 | 小节编号错位或重复，例如物理页 87 的 `5.3.1`、物理页 112 重复 `5.3.4`，部分标题存在缺字 | 以本文物理页和主题名定位，不依赖原书小节编号的连续性 |

## 5. 如何使用这份对照

1. 先运行当前编号的课程实验，再带着输出和问题回查 PDF 对应物理页。
2. 阅读公式时，在纸上标出每个张量的 shape；无法写出 shape 时，不进入下一公式。
3. PDF 中的大模型代码先看结构，不直接运行默认配置；先在课程的小模型上验证同一原理。
4. 遇到本文勘误中的片段，先解释为什么存在风险，再对比本项目实现和测试。
5. 完成课程 05 后，再按个人硬件选择第 5-7 章的 Tokenizer、Tiny-LLaMA、LoRA、RAG 或 Agent 扩展；这些扩展不是核心六实验的前置条件。

学习本项目的目标不是复述书中的代码，而是能够从公式推出 shape，从 shape 写出实现，再用测试证明实现符合因果性、概率归一和数值稳定等不变量。
