# 02 Tokenizer 与语言模型：从一串文字到“预测下一个 token”

> PDF 对照：《Happy-LLM v1.0》物理页 7-11 概览 NLP 任务并讨论中文分词、子词切分等问题；物理页 12-13 转向文本表示、语言模型、Word2Vec 与 ELMo。本章不照抄书中代码，而是用字符 Tokenizer 和 Bigram 模型重建一条最小可运行链路。本文的措辞、例子、推导与实验均为本项目原创改写，页码只作主题索引。

## 本节问题

上一章中的模型只看到了数字。真实文本怎样进入这些公式？

我们要回答：

1. token 是字、词还是子词？
2. Tokenizer 与模型分别负责什么？
3. `encode`、`decode` 和词表是什么关系？
4. 未登录 token（OOV）为什么麻烦？
5. 语言模型到底在估计哪个概率？
6. 最简单的 Bigram 模型能教会我们哪些 LLM 原理？
7. 负对数似然、交叉熵和困惑度之间是什么关系？

对应实验：

```text
experiments/02_tokenization.py
experiments/03_bigram_language_model.py
```

## 第一部分：Tokenizer

### 直觉：Tokenizer 是可逆的编号协议

计算机训练时不能直接把“语言模型”四个字交给矩阵乘法。Tokenizer 先约定一个词表：

```text
id 0 -> <unk>
id 1 -> <bos>
id 2 -> <eos>
id 3 -> 模
id 4 -> 型
id 5 -> 语
id 6 -> 言
```

于是：

```text
"语言模型" -> [5, 6, 3, 4]
```

模型只处理右边的整数。生成后，Tokenizer 再把整数还原成文字。

Tokenizer 不是语言模型：

- Tokenizer 决定如何切分、怎样编号；
- 语言模型学习 token 序列的概率；
- Tokenizer 的 id 本身没有大小语义。`id=100` 不比 `id=3` “更重要”。

### 定义

给定词表 `Vocab`，Tokenizer 至少提供：

```text
encode: text -> list[int]
decode: list[int] -> text
vocab_size: int
```

理想的课程级检查是：

```text
decode(encode(text)) == text
```

真实工业 Tokenizer 可能会规范化空格或 Unicode，因此“逐字节完全相等”不是普遍规律；但本项目的字符 Tokenizer 应尽量保持往返一致，方便初学者定位问题。

### 字符、词与子词

| 粒度 | 优点 | 缺点 | 本项目用途 |
| --- | --- | --- | --- |
| 字符 | 实现简单，中文直观，几乎不会遇到整词 OOV | 序列较长，英文语义片段被拆散 | 核心实验 |
| 词 | 序列较短，单位直观 | 中文需分词；生僻词和新词会 OOV；词表很大 | 对比思考 |
| 子词 | 在词表大小与序列长度之间折中，可组合新词 | 训练与解码更复杂 | 后续 BPE 扩展 |

PDF 物理页 8 介绍子词切分。本阶段先用字符级方法，是为了把“切分算法的复杂度”与“语言模型的复杂度”分离。等 Bigram 和 Transformer 流程跑通，再换 BPE，观察变化会更清楚。

### 特殊 token

常见特殊 token：

- `<bos>`：序列开始；
- `<eos>`：序列结束；
- `<pad>`：把不同长度序列补齐；
- `<unk>`：遇到词表外 token 时的回退符号。

它们必须拥有不同 id。是否需要全部使用，取决于实验。字符 Tokenizer 可以仅用语料实际字符，也可以显式加入特殊 token；关键是训练、生成与解码采用同一约定。

### 形状

一条文本：

```text
"模型"
```

编码为：

```text
[id_模, id_型]       shape: (T,) = (2,)
```

两条等长文本组成 batch：

```text
[[...], [...]]       shape: (B,T)
```

进入 embedding 后：

```text
token ids (B,T) -> embeddings (B,T,D)
```

Embedding 表可以看作一个矩阵：

```text
embedding weight: (V,D)
```

用 token id 查表，得到该 token 的 `D` 维向量。

## 运行 Tokenizer 实验

在 PowerShell 中：

```powershell
Set-Location 'D:\path\to\LearnLLM'
.\.venv\Scripts\python.exe .\experiments\02_tokenization.py
```

也可以直接检查课程核心 API：

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from learn_llm.tokenizer import CharTokenizer; text=Path('data/tiny_corpus.txt').read_text(encoding='utf-8'); t=CharTokenizer.from_text(text); ids=t.encode('语言模型'); print('V=', t.vocab_size, 'ids=', ids, 'decoded=', t.decode(ids))"
```

### 运行前先预测

1. 同一字符每次出现时，id 应相同还是随机变化？
2. `encode` 输出的长度与字符数有什么关系？
3. 把语料中从未出现的字符传给只从该语料建词表的 Tokenizer，可能发生什么？
4. `decode(encode(text))` 在本实验中应得到什么？
5. 词表顺序改变但模型权重不变，模型含义会保持不变吗？

第 5 题答案是否定的。模型参数是围绕某套 id 协议训练的；更换 id 映射等于把输入符号的含义打乱。

### 预期输出与验收

脚本先做字符往返，再打印 BPE 合并过程；输出模式应类似：

```text
原文: 语言模型预测下一个 token。
token ids: [...]
还原: 语言模型预测下一个 token。
词表大小: <正整数>
merge 01: (...) (出现 ... 次)
...
PASS: 字符级往返无损，BPE 重复合并高频相邻符号。
```

验收：

- [ ] `vocab_size` 等于词表中的唯一 token 数；
- [ ] 相同字符始终映射到相同 id；
- [ ] 所有编码 id 都满足 `0 <= id < vocab_size`；
- [ ] 对语料内示例，`decode(encode(text)) == text`；
- [ ] 扩展练习：未知字符不会被静默映射成一个普通字符；
- [ ] 同样语料重复构建词表时，id 映射稳定。

## 第二部分：语言模型

### 直觉：Bigram 是只有一步记忆的语言模型

Bigram 模型只看当前 token `x_t`，预测下一个 token `x_{t+1}`：

$$
P(x_{t+1}\mid x_t)
$$

例如语料中“语”后面经常出现“言”，那么 `P(言 | 语)` 应高于一个从未跟在“语”后面的字符。

Bigram 明显不够聪明。看到“苹果公司发布”与“我吃了一个苹果”，当前 token 相同时，它无法利用更早上下文区分含义。但它已经包含现代 LLM 的训练骨架：

1. 把文本编码为 token ids；
2. 用前文预测下一个 token；
3. 得到 logits；
4. 用交叉熵评价；
5. 调整参数或统计量；
6. 按概率逐步生成新 token。

Transformer 主要改变的是“怎样利用更长上下文产生 logits”，而不是把训练目标换成完全不同的东西。

### 从序列构造训练对

若 token 序列为：

```text
[3, 8, 2, 5]
```

输入与目标错开一位：

```text
inputs  = [3, 8, 2]
targets = [8, 2, 5]
```

也就是：

```text
3 -> 8
8 -> 2
2 -> 5
```

这是后续 Transformer 预训练仍会使用的“shifted targets”。

### 计数模型

建立计数矩阵：

```text
counts: (V,V)
```

`counts[i,j]` 表示 token `i` 后面出现 token `j` 的次数。第 `i` 行归一化后得到条件分布 `P(next=j | current=i)`。

最直接的最大似然估计：

$$
P(j\mid i)=\frac{C_{ij}}{\sum_k C_{ik}}
$$

如果某一行总计数为 0，分母会为 0；如果某个合法转移从未出现，概率会为 0，负对数似然会变成无穷。可用加法平滑：

$$
P(j\mid i)=\frac{C_{ij}+\alpha}{\sum_k(C_{ik}+\alpha)}
$$

其中 `α > 0`。`α` 越大，分布越接近均匀；越小，越相信观察到的计数。

### NLL、平均交叉熵与困惑度

给定真实序列，平均负对数似然：

$$
\operatorname{NLL}
=-\frac{1}{N}\sum_{t=1}^{N}\log P(x_{t+1}\mid x_t)
$$

困惑度：

$$
\operatorname{PPL}=e^{\operatorname{NLL}}
$$

直觉上，PPL 可以粗略理解为模型在每一步面对的“有效候选数”。若模型始终在 `V` 个 token 上均匀预测：

$$
\operatorname{NLL}=\ln(V),\quad \operatorname{PPL}=V
$$

因此，在训练语料上，一个学到局部规律的 Bigram 模型通常应比均匀基线 `V` 更低。但训练 PPL 低不代表泛化好，更不代表理解语义。

### 采样生成

从起始 token `x_0` 开始：

1. 读取 `P(next | x_0)`；
2. 按概率抽取 `x_1`；
3. 再读取 `P(next | x_1)`；
4. 重复到指定长度或 `<eos>`。

采样含随机性，所以要固定随机种子。生成文本只需展示学到的二元相邻模式，不应期待它写出连贯文章。

### 形状总览

```text
encoded corpus       (N,)
inputs               (N-1,)
targets              (N-1,)
transition counts    (V,V)
transition probs     (V,V)
one step probs       (V,)
generated ids        (T_generated,)
```

若使用可训练的 embedding 形式，当前 token id 查到的整行 logits 也可写成：

```text
logit table          (V,V)
inputs               (B,T)
selected logits      (B,T,V)
```

## 运行 Bigram 实验

```powershell
Set-Location 'D:\path\to\LearnLLM'
.\.venv\Scripts\python.exe .\experiments\03_bigram_language_model.py
```

相关单元测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

### 运行前先预测

1. `inputs` 和 `targets` 的长度是否相同？
2. 概率矩阵的每一行之和应是多少？
3. 没有平滑时，未见转移的 NLL 可能是什么？
4. 均匀模型的 PPL 与词表大小有什么关系？
5. 固定同一种子采样两次，文本是否应一致？
6. Bigram 能否区分“银行利率上升”和“河流两岸”中更远距离的上下文？

### 预期输出与验收

脚本使用稀疏计数而不是显式打印整个 `(V,V)` 矩阵；输出模式应包含：

```text
词表大小: <V>
在训练语料中，'语' 后最常见字符: [...]
Bigram 训练语料困惑度: <通常小于 V>
均匀随机基线困惑度: <V>
采样文本: <固定长度文本>
PASS: 利用一个 token 的上下文，模型优于均匀随机猜测。
```

关键验收：

- [ ] `inputs[1:]` 与 `targets[:-1]` 表达同一段中间序列；
- [ ] 能解释若把稀疏计数展开，`counts`、`probs` 会是 `(V,V)`；
- [ ] 能用公式说明每个上下文条件分布的概率和为 1；
- [ ] 启用平滑后没有 `NaN`、`inf` 或 `log(0)`；
- [ ] 训练语料 PPL 通常低于均匀基线 `V`；
- [ ] 扩展练习：重置相同随机种子会产生相同样例；
- [ ] 生成 id 全在合法词表范围内，且能被 decode。

## 常见故障与排查

### 1. 中文文件读出乱码

显式使用 UTF-8：

```python
text = path.read_text(encoding="utf-8")
```

PowerShell 显示异常时可先执行：

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

### 2. `KeyError` 或未知字符错误

确认待编码文本是否包含训练语料从未出现的字符。课程实现应明确报错或使用 `<unk>`，不要悄悄丢掉字符。若选择扩充词表，模型也必须用新词表重新初始化或训练。

### 3. decode 后字符顺序不对

检查 `stoi`（token 到 id）和 `itos`（id 到 token）是否真正互为逆映射。不要分别用两种不同排序构建它们。

### 4. 概率矩阵出现 `NaN`

检查是否有整行计数为 0，并发生 `0/0`。可以只为已出现的上下文建模，或添加 `α > 0` 的平滑。

### 5. PPL 是无穷大

真实目标的预测概率中出现了 0。检查平滑，并在取 log 前确认概率为正。不要只用一个很小的 epsilon 掩盖上游计数错误。

### 6. 生成结果不通顺

这通常不是 bug。Bigram 只记住一个 token 的上下文。先检查相邻字符模式、长度、合法 id 和可复现性，不要用文章质量作为这个实验的验收标准。

### 7. 训练 PPL 很低，所以认为模型理解了文本

训练指标只说明模型适合这批数据。可以保留一段验证文本，比较 train PPL 与 validation PPL；差距较大表示过拟合或分布差异。

## 练习

### Tokenizer 基础练习

1. 分别对中文、英文、带 emoji 的字符串做字符编码，比较 token 数。
2. 打印词表前 20 个映射，说明 id 的排序规则。
3. 给 Tokenizer 增加 `<bos>` 和 `<eos>`，确保两者 id 不同且解码规则清晰。
4. 构造一个语料外字符，记录实现是报错还是映射 `<unk>`，评价两种策略的利弊。

### Bigram 基础练习

1. 手工为文本 `ABABA` 写出全部训练对和 `(A,B)`、`(B,A)` 计数。
2. 比较 `α=0.01`、`0.1`、`1.0` 时同一行概率的变化。
3. 分别从同一个 token 生成 50 个字符，改变种子并比较文本。
4. 计算均匀分布在 `V=10`、`V=100` 时的 NLL 与 PPL。

### 进阶练习

1. 把数据划分成 train/validation，分别计算 PPL，并解释差距。
2. 将 Bigram 扩展为 Trigram：用最近两个 token 预测下一个 token。估算计数表从 `V^2` 增长到多大。
3. 实现 temperature：用 `logits / temperature` 再 Softmax。比较 `0.5`、`1.0`、`2.0` 的生成多样性。
4. 保持语料不变，比较字符词表与简单按空格分词的词表大小、序列长度和 OOV 行为。
5. 解释为什么“Tokenizer 词表更大”不一定总是更好。提示：考虑 embedding 参数量、序列长度和稀有 token。

## 掌握检查

- [ ] 我能区分 Tokenizer 与语言模型。
- [ ] 我能解释字符、词、子词三种粒度的权衡。
- [ ] 我能验证 `encode/decode` 的往返关系与 id 范围。
- [ ] 我能从长度为 `N` 的序列构造 `N-1` 个 next-token 训练对。
- [ ] 我能解释 Bigram 估计的是 `P(x_{t+1}|x_t)`。
- [ ] 我能写出计数归一化与加法平滑公式。
- [ ] 我能从平均 NLL 得到 PPL，并说明均匀基线为何是 `V`。
- [ ] 我知道 Bigram 的生成不流畅并不自动表示代码错误。
- [ ] 我知道训练 PPL 下降不等于真正理解或泛化。

完成后进入 [03 注意力与 Transformer](./03_注意力与Transformer.md)：模型将不再只看前一个 token，而是学习如何从整个可见上下文中聚合信息。
