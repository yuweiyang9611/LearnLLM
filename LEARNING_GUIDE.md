# 从零学习 LLM：总学习指导

## 1. 学完之后你应该真正会什么

“会用聊天机器人”和“理解 LLM”是两件不同的事。本课程把最终能力写成可以检查的行为。完成后，你应该能：

1. 把一句文本完整追踪为 token id、embedding、注意力输出、logits 和下一个 token。
2. 手算小规模 Softmax、交叉熵和缩放点积注意力，并逐行对应到代码。
3. 解释 Encoder-only、Encoder-Decoder、Decoder-only 的信息流和训练目标。
4. 实现带因果掩码的多头自注意力和 Decoder Block，并证明没有未来泄漏。
5. 在 CPU 上训练一个微型 Decoder-only Transformer，保存、加载并生成文本。
6. 区分预训练、SFT、LoRA、量化、偏好对齐；知道它们分别改变数据、目标、参数或数值表示中的哪一部分。
7. 用损失、困惑度、Recall@k、延迟与失败案例评价模型/系统，而不只凭“看起来像人话”。
8. 搭建本地 Tiny RAG，返回检索证据、支持性判定和明确拒答，并能判断问题出在检索还是回答层。
9. 实现带工具白名单、状态轨迹、错误处理和最大步数的本地 Agent 控制循环。

## 2. 先建立一个正确的总图

Decoder-only LLM 学习的核心概率是：

\[
P(x_1, x_2, \ldots, x_T)
=\prod_{t=1}^{T}P(x_t\mid x_{<t})
\]

含义不是“模型从数据库找到唯一答案”，而是：给定已经出现的 token，模型为下一个 token 给出一个条件概率分布。训练时，正确的下一个 token 是监督信号；推理时，程序从这个分布中选择或采样下一个 token，再把它接回输入。

一次前向传播可以画成：

```text
token ids [B,T]
   │
   ├─ token embedding [B,T,C]
   └─ position embedding [T,C]
              │ 相加
              ▼
      Transformer Block × L
   ┌────────────────────────────┐
   │ Norm -> causal attention   │
   │       -> residual add      │
   │ Norm -> MLP -> residual    │
   └────────────────────────────┘
              │
              ▼
         logits [B,T,V]
              │
       Cross-Entropy / Sampling
```

符号第一次出现时都要读成中文：

- `B`：batch size，一批有多少条序列。
- `T`：sequence length，每条序列有多少个 token。
- `C`：model dimension，每个位置的隐藏向量维度。
- `H`：attention heads，注意力头数。
- `D`：head dimension，通常 `D = C / H`。
- `V`：vocabulary size，词表大小。
- `L`：Transformer Block 层数。

## 3. 十个阶段的学习路线

### 阶段 0：环境和实验方法

要回答的问题：怎样确保“我的代码能复现”，而不是“昨天碰巧运行过”？

- 学会明确调用 `.venv\Scripts\python.exe`。
- 理解解释器、包目录、CPU/GPU、随机种子。
- 运行 `experiments/00_environment_check.py`。
- 验收：`sys.prefix != sys.base_prefix`，路径指向项目 `.venv`。

### 阶段 1：最小数学与自动求导

要回答的问题：模型怎样把“猜错了多少”变成每个参数应该往哪边调整？

- 向量点积和矩阵乘法。
- logits、稳定 Softmax、负对数似然、交叉熵。
- 梯度、链式法则、`loss.backward()`。
- 实验：`01_math_foundations.py`。
- 验收：极大 logits 不溢出；手写交叉熵与 PyTorch 一致。

### 阶段 2：Token 与统计语言模型

要回答的问题：神经网络为什么不能直接读取字符串？

- 字符、词、子词 Tokenizer 的取舍。
- `encode`、`decode`、词表、未知 token、特殊 token。
- Bigram 条件概率、加一平滑、困惑度。
- 实验：`02_tokenization.py`、`03_bigram_language_model.py`。
- 验收：往返编码无损；Bigram 困惑度低于均匀随机基线。

### 阶段 3：Attention

要回答的问题：一个位置怎样有选择地读取其他位置？

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

- Q/K/V 是输入经过三个可学习线性投影后的向量；它们不是字面数据库字段。
- `QK^T` 得到每个查询对每个键的分数。
- 除以 `sqrt(d_k)` 避免高维点积让 Softmax 过早饱和。
- 因果 mask 把未来位置的分数变为负无穷，使其 Softmax 权重为 0。
- 实验：`04_attention.py`。
- 验收：每行权重和为 1；上三角未来权重全为 0。

### 阶段 4：Transformer Decoder

要回答的问题：Attention 为什么还不等于完整 Transformer？

- 多头拆分和拼接。
- token/position embedding。
- 残差连接、Norm、逐位置 MLP。
- 多层堆叠和语言模型输出头。
- 实验：`05_transformer_block.py`。
- 验收：输出 `[B,T,V]`；改变未来 token 不影响过去位置 logits。

### 阶段 5：模型家族

要回答的问题：BERT、T5、GPT 到底差在哪里？

| 家族 | 可见上下文 | 典型训练目标 | 典型用途 |
|---|---|---|---|
| Encoder-only | 双向 | MLM | 理解、分类、抽取 |
| Encoder-Decoder | Encoder 双向；Decoder 因果 | Seq2Seq 去噪/生成 | 翻译、摘要、条件生成 |
| Decoder-only | 因果单向 | CLM | 通用文本生成、LLM |

验收不是背表，而是能为一个新任务画出输入、mask、输出和 loss 位置。

### 阶段 6：预训练 TinyGPT

要回答的问题：一次标准训练迭代具体发生什么？

```text
抽取 [B,T+1] token
  -> x = 前 T 个，y = 后 T 个
  -> logits = model(x)
  -> CE(logits, y)
  -> zero_grad -> backward -> clip -> optimizer.step
  -> 定期 validation / checkpoint
```

- 实验：`06_train_tiny_gpt.py`。
- 先用 `--quick` 检查管线，再增加步数。
- 验收：loss 比初始值明显下降；保存包含配置、权重和冻结 Tokenizer 的真实 base checkpoint；加载后输出一致。
- 后续实验 07 和 10 都以 `checkpoints/tiny_gpt.pt` 为源，不得在微调阶段重建词表或位置嵌入。checkpoint 记录预训练语料、train/dev/test 指令文件及兼容数据视图的 SHA-256；修改数据、指令字符或窗口上限后，应回到实验 06 重新预训练。
- 重要限制：在小语料上过拟合只证明训练管线工作，不证明模型理解世界。

### 阶段 7：自回归生成与评测

要回答的问题：同一组模型权重为什么能生成不同风格文本？

- Greedy：总选概率最大 token，稳定但容易重复。
- Temperature：用 `logits / temperature` 改变分布尖锐程度。
- Top-k：只在概率最高的 k 个候选中采样。
- 实验：`07_generate.py`。
- 验收：固定随机种子可复现；能解释采样策略没有增加模型知识。
- 实验 10 生成 adapter 后，实验 07 也可用 `--base-checkpoint checkpoints/tiny_gpt.pt --adapter checkpoints/tiny_gpt_lora_adapter.pt` 在独立进程中严格恢复 LoRA 模型；loader 会先校验 artifact 版本、config、Tokenizer、SHA-256、targets 和 tensor 结构。

### 阶段 8：SFT、LoRA 和对齐

要回答的问题：预训练之后，怎样让模型按指令回答？怎样少训练参数？

- SFT 改变训练样本格式和 loss mask，通常只对 assistant 回复计算 loss。
- LoRA 冻结原参数，以 `BA` 表示低秩增量：

\[
W = W_0 + \Delta W = W_0 + BA
\]

- 量化改变参数的数值表示；LoRA 改变可训练参数集合；两者不是同义词。
- 偏好对齐使用 chosen/rejected 或奖励信号改变输出偏好；不是简单的事实注入。
- 实验：先运行 `06_train_tiny_gpt.py` 生成 base，再运行 `10_sft_tiny_gpt.py`；`08_lora.py` 是独立的 Linear 原理演示。
- 实验 10 从同一 base 独立比较 pretrained base、random-init Full SFT、pretrained Full SFT 和 pretrained LoRA-SFT，而不是把 LoRA 接在 Full SFT 后面。各机制的学习率会单独打印并写入产物 metadata；这是起点与评测设置受控的机制对照，不是所有超参数相同的单变量实验。
- 数据按 4/4/4 隔离：train 是四条 `seen`，dev 是四条 `paraphrase`，test 包含两条 `paraphrase` 与两条 `new_intent`。dev 用于选择训练配置；每个分支和 seed 的 test 只在训练完成后评一次，不能反复看 test 再调参。
- 验收 SFT：prompt 与 padding label 都是 `-100`；训练 loss 下降；Full SFT 参数确实改变并保存完整模型权重 checkpoint（不含续训所需的 optimizer 状态）。
- 验收 LoRA：基础权重逐元素不变；只有 A/B 可训练；注入前后输出一致；adapter-only 保存与恢复一致。
- 比较时同时记录 train/dev/test assistant loss、任务成功率、关键词准确率、格式准确率、原预训练语料 validation loss 的保持度/干扰代理、可训练/总参数量、耗时与生成案例。默认 seed 42/43/44；loss 与任务指标跨 seed 报告均值和总体标准差，参数量和耗时按 seed 保留。JSON manifest 保存逐样本、逐类别和运行环境，CSV 保存长格式训练历史。
- 严格任务成功要求一条生成同时命中全部必要关键词并满足全部格式规则。这个 Tiny 字符模型得到 0% 严格成功率也可能是正确、应保留的失败结果；`new_intent` 不代表知识在预训练语料中从未出现，不能宣称未知知识泛化。

### 阶段 9：RAG、Agent 和系统评测

要回答的问题：模型外部的资料、工具和控制代码如何参与回答？

- RAG：索引 -> 检索 -> 基于证据回答或拒答 -> 返回来源 -> 检查支持性。
- Agent：规划器 + 工具 + 状态 + 有界控制循环，而不是“产生意识”。生产系统可由 LLM 充当规划器，但权限与终止条件仍必须由程序保证。
- 实验：`09_tiny_rag.py`、`11_local_agent.py`。
- 验收 RAG：分别报告 Retrieval Recall、拒答准确率、回答支持率和端到端成功率；资料外问题不得伪造引用。
- 验收 Agent：算术与检索工具走白名单；未知工具、工具异常和最大步数分别进入可观察的停止状态。

## 4. 建议的八周节奏

| 周 | 内容 | 最低交付物 |
|---|---|---|
| 1 | 环境、Python、张量、矩阵 | 实验 00/01 记录；手算一个 Softmax |
| 2 | Tokenizer、Bigram、困惑度 | 实验 02/03；解释 token 不等于词 |
| 3 | Q/K/V、缩放、mask | 实验 04；手画 3×3 causal mask |
| 4 | 多头、残差、Norm、MLP | 实验 05；完成无未来泄漏测试 |
| 5 | BERT/T5/GPT 与训练生命周期 | 模型家族对比图；CLM/SFT loss mask |
| 6 | TinyGPT 预训练和生成 | 真实 base checkpoint、冻结 Tokenizer、loss 记录、三种采样结果 |
| 7 | SFT、LoRA、评测 | 四分支对比；4/4/4 train/dev/test；三类任务指标与保持度代理；多 seed JSON/CSV；完整模型权重与 LoRA adapter |
| 8 | RAG、Agent 与结课项目 | 检索/拒答/支持指标；Agent 轨迹；消融与失败案例 |

每天 45-90 分钟即可。每周最后一次不学新概念，只做复述、测试和修复。

## 5. 每个实验都按同一套方法做

1. 写问题：今天要解释哪个机制？
2. 写预测：运行前预计 shape、数值关系或趋势是什么？
3. 手算极小例子：2-3 个 token 足够。
4. 运行最小代码：先不改参数。
5. 对照预期：不是只看“没有报错”。
6. 单变量实验：只改一个设置。
7. 加自动断言：把理解写成可重复检查。
8. 记录失败：错误信息、根因、修复和仍未解释的点。
9. 用自己的话复述：尽量不看讲义。

## 6. 必须能识别的十四个误区

1. LLM 不是知识数据库，而是条件概率模型。
2. token 不固定等于一个汉字、一个英文单词或一个语义单位。
3. embedding 相近是训练后产生的表示性质，不是人为给每个维度命名。
4. Q/K/V 是学习投影；字典查询只是直觉比喻。
5. Attention 权重不是模型完整推理过程，也不是可靠解释。
6. 因果 mask 只限制当前前向传播读取未来位置，不会删除模型参数中的知识。
7. 上下文窗口不是长期记忆。
8. 训练 loss 下降不等于理解；验证 loss 上升通常是过拟合信号。
9. 不同 Tokenizer/词表下的困惑度不能直接机械比较。
10. SFT、LoRA、量化、偏好对齐解决的问题不同。
11. assistant-only loss 仍要按“预测下一个 token”右移标签；只把 prompt 原位替换为 `-100` 可能错一位。
12. RAG 通常不修改模型参数，也不能保证答案必然正确；资料不足时拒答是能力而不是失败。
13. Agent 的工具权限、参数校验和最大步数不能只依赖模型自觉遵守。
14. 玩具模型验证的是机制和工程管线，不能外推为生产 LLM 能力。

## 7. 结课时如何判断自己真的掌握了

不看代码回答以下问题，并用实验验证其中至少三个：

- 为什么稳定 Softmax 要先减最大 logits？
- 为什么语言模型训练的标签相对输入右移一位？
- `QK^T` 的 shape 为什么是 `[B,H,T,T]`？
- 不除以 `sqrt(d_k)` 可能发生什么？
- 怎样用测试证明模型没有看到未来 token？
- 残差连接和归一化分别解决什么训练问题？
- 为什么初始交叉熵常在 `ln(V)` 附近？
- Temperature 为 0.5 和 1.5 各会怎样改变分布？
- LoRA 的 rank 变大时，可训练参数如何变化？
- assistant-only SFT 中，为什么第一个回答 token 的 label 对齐在最后一个 prompt token 的输入位置？
- 检索正确但回答错误、检索错误但回答看似正确，应分别怎样诊断？
- 怎样用测试证明 Agent 遇到未知工具、工具异常或无限规划时一定会停？

如果某一题只能背定义，就回到相应实验，构造一个最小反例。
