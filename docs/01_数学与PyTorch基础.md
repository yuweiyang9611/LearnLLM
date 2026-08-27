# 01 数学与 PyTorch 基础：把“模型猜测”变成可以训练的数

> PDF 对照：《Happy-LLM v1.0》物理页 12-13 从文本表示与语言模型引出“用数表示语言”；物理页 16-20 的注意力公式大量使用点积、矩阵乘法、缩放和 Softmax。本章将这些数学工具提前拆成 CPU 小实验。本文的措辞、推导、示例和代码说明均为本项目原创改写，PDF 页码只用于主题索引。

## 本节问题

一个 Decoder-only 语言模型在位置 `t` 做的核心事情，可以压缩成一句话：

> 根据位置 `0..t` 已经出现的 token，给词表中每个“下一个 token”打分，再让正确答案的概率变大。

这句话隐藏了五个问题：

1. 为什么文本最后会变成向量和矩阵？
2. 模型输出的 `logits` 是概率吗？
3. Softmax 为什么能把任意实数变成概率？
4. 为什么直接计算 `exp(1000)` 会出错？
5. 交叉熵怎样把一次预测变成一个可优化的标量 loss？

对应实验：

```text
experiments/01_math_foundations.py
```

## 直觉：logits 是选票，Softmax 是换算规则

假设词表只有三个 token：

```text
["猫", "狗", "。"]
```

模型输出：

```text
[2.0, 1.0, -1.0]
```

这三个数称为 logits。它们只是相对分数：`2.0` 比 `1.0` 更支持“猫”，但不能说“猫的概率是 200%”。Softmax 像一套换算规则，把所有选票转成非负且总和为 1 的概率。

如果正确答案是“狗”，交叉熵只关心模型分给“狗”的概率。概率越小，惩罚越大；概率越接近 1，惩罚越接近 0。

## 1. 标量、向量、矩阵和张量

### 定义

- 标量：一个数，例如 loss `2.31`。
- 向量：一列或一行数，例如一个 token 的 embedding，形状为 `(D,)`。
- 矩阵：二维数表，例如一条序列的 embeddings，形状为 `(T,D)`。
- 张量：任意维度的规则数组。批量序列通常是三维张量 `(B,T,D)`。

在 PyTorch 中，它们都由 `torch.Tensor` 表示。

### 形状

假设：

```text
B = 2   # 两条句子
T = 4   # 每条句子四个 token
D = 8   # 每个 token 用八个数表示
V = 6   # 词表有六个 token
```

典型数据流是：

```text
token ids       (B,T)   = (2,4)
embeddings      (B,T,D) = (2,4,8)
hidden states   (B,T,D) = (2,4,8)
logits          (B,T,V) = (2,4,6)
target ids      (B,T)   = (2,4)
loss            ()      = 标量
```

`()` 表示零维张量，也就是一个标量。

## 2. 点积与矩阵乘法

### 定义与公式

两个长度都为 `D` 的向量 `x`、`y` 的点积：

$$
x \cdot y = \sum_{i=1}^{D} x_i y_i
$$

点积把两条向量压成一个数。在注意力中，它被用作 Query 与 Key 的匹配分数。

矩阵乘法把许多次点积并行完成。若：

```text
X: (T,D)
W: (D,V)
```

则：

```text
X @ W: (T,V)
```

可以把它理解为：序列的每个位置都通过同一个线性映射，得到对词表中 `V` 个 token 的分数。

### 形状检查法

矩阵乘法 `(a,b) @ (b,c) -> (a,c)`。中间两个 `b` 必须相等，并在结果中消失。

例如：

```text
(B,T,D) @ (D,V) -> (B,T,V)
```

PyTorch 会把前面的 batch 维保留下来。

## 3. Softmax

### 定义与公式

对一组 logits `z`：

$$
\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

它保证：

1. 每个概率都大于 0；
2. 所有概率之和为 1；
3. 较大的 logit 对应较大的概率；
4. 所有 logits 同时加上同一个常数，概率不变。

第 4 点非常重要。因为：

$$
\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}
=
\frac{e^{z_i}/e^c}{\sum_j e^{z_j}/e^c}
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

所以可以选择 `c = max(z)`，先减去最大值再求指数：

$$
\operatorname{softmax}(z_i)
=
\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}
$$

这称为稳定 Softmax。最大指数变成 `exp(0)=1`，可避免 `exp(1000)` 溢出。

### Softmax 沿哪一维？

语言模型 logits 的形状通常是 `(B,T,V)`。我们要在每个 batch、每个时间位置上，对整个词表分配概率，因此沿最后一维 `V` 做 Softmax：

```python
probs = softmax(logits, dim=-1)
```

结果仍是 `(B,T,V)`，并满足：

```text
probs.sum(dim=-1) -> (B,T)，每个元素约等于 1
```

## 4. 交叉熵

### 定义与直觉

如果正确 token 的 id 是 `y`，模型给它的概率是 `p_y`，单个位置的负对数似然为：

$$
L=-\log(p_y)
$$

几个直观数值：

| 正确答案概率 `p_y` | loss `-ln(p_y)` | 含义 |
| ---: | ---: | --- |
| `1.0` | `0` | 完全确信且答对 |
| `0.5` | 约 `0.693` | 只有一半把握 |
| `0.1` | 约 `2.303` | 给正确答案的概率很低 |
| `0.01` | 约 `4.605` | 非常自信地忽略了正确答案 |

一批序列的交叉熵，就是对有效位置的 `-log(p_y)` 求平均。实际实现通常直接接收 logits，并在内部用数值稳定的 `log_softmax`，不要先手工 Softmax 再取 log。

### 形状展开

语言模型训练时：

```text
logits:  (B,T,V)
targets: (B,T)
```

交叉熵实现常把前两维合并：

```text
logits.reshape(B*T, V): (B*T,V)
targets.reshape(B*T):   (B*T,)
loss:                   ()
```

目标中每个整数都必须落在 `[0, V-1]`。

## 5. 梯度和一次优化更新

loss 只是评价一次预测有多差。训练还需要计算：每个参数改变一点，loss 会怎样变化。这就是梯度。

一次最简更新可以写成：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

- `θ`：模型参数；
- `∇θL`：loss 对参数的梯度；
- `η`：学习率。

PyTorch 的基本顺序：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

先清梯度是因为 PyTorch 默认累加梯度。忘记 `zero_grad()` 会让更新混入前几步的梯度。

## Windows 运行命令

进入项目根目录：

```powershell
Set-Location 'D:\path\to\LearnLLM'
```

先确认核心包可导入：

```powershell
.\.venv\Scripts\python.exe -c "from learn_llm.math_utils import stable_softmax, cross_entropy_from_logits; print('math utilities ready')"
```

运行本章实验：

```powershell
.\.venv\Scripts\python.exe .\experiments\01_math_foundations.py
```

如果想同时运行相关单元测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

## 运行前先预测

不要先看输出。对下面问题写下答案：

1. `(2,3) @ (3,4)` 的结果是什么形状？
2. logits `[2,1,-1]` 中哪个 token 概率最大？
3. logits `[2,1,-1]` 和 `[1002,1001,999]` 的 Softmax 是否应相同？
4. 对 `[1000,1001]` 直接求指数可能发生什么？稳定 Softmax 呢？
5. 正确 token 概率从 `0.1` 增加到 `0.8`，交叉熵应上升还是下降？

答案：结果为 `(2,4)`；第一个；应相同；直接指数可能溢出而稳定版本应有限；交叉熵下降。

## 预期输出与验收

数值可能因打印精度略有不同，脚本的输出结构类似：

```text
logits shape: (2, 3) = [batch, classes]
probabilities: <两行概率>
row sums: tensor([1., 1.])
manual cross entropy:  <数值>
PyTorch cross entropy: <相同或极接近的数值>
absolute error:        <接近 0>
gradient dL/dlogits:   <有限梯度>
PASS: Softmax 稳定，交叉熵和自动求导结果正确。
```

验收不变量：

- [ ] 矩阵乘法输出形状符合 `(a,b) @ (b,c) -> (a,c)`；
- [ ] Softmax 每行之和与 1 的误差小于约 `1e-6`；
- [ ] 大 logits 的 Softmax 结果没有 `NaN` 或 `inf`；
- [ ] 扩展练习：给所有 logits 加同一个常数，概率基本不变；
- [ ] 手工交叉熵和 PyTorch 结果误差小于约 `1e-6`；
- [ ] 扩展练习：提高正确类别的 logit 后，loss 下降。

不要执着于“概率必须打印成某几个固定小数”。浮点数显示位数可能不同，数学不变量才是验收标准。

## 常见故障与排查

### 1. `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

把两个输入形状写在纸上。检查左矩阵最后一维是否等于右矩阵倒数第二维：

```text
(..., D) @ (D, ...)
```

不要靠反复添加 `transpose` 碰运气。先说清楚每一维的语义。

### 2. Softmax 概率和不是 1

最常见原因是 `dim` 选错。对 `(B,T,V)` logits，应沿词表维 `V`，即 `dim=-1`。

调试时打印：

```python
print(logits.shape)
print(probs.sum(dim=-1).shape)
print(probs.sum(dim=-1))
```

### 3. Softmax 出现 `NaN`

检查是否直接对原 logits 求 `exp`。应先减去该行最大值。还要检查输入本身是否已经包含 `NaN` 或 `inf`。

### 4. 交叉熵报目标越界

若词表大小为 `V`，目标 id 必须满足：

```text
0 <= target < V
```

打印 `targets.min()`、`targets.max()` 和 `V`。

### 5. 手工交叉熵与 PyTorch 不一致

依次检查：

1. 两者是否使用自然对数 `ln`；
2. 是否对同一批位置求平均；
3. 是否先错误地把概率四舍五入；
4. 是否把 logits 当成概率；
5. 是否沿正确的词表维做归一化。

### 6. 第二次反向传播报计算图已释放

默认情况下，`.backward()` 后计算图会释放。下一步训练应重新执行 forward 得到新 loss，而不是对旧 loss 重复反向传播。

## 练习

### 基础练习

1. 手算 `[0,0,0]` 的 Softmax。为什么答案一定是均匀分布？
2. 证明给所有 logits 加常数不改变 Softmax。
3. 构造 `(B,T,D)=(2,5,8)` 和 `(D,V)=(8,11)` 的张量，预测并验证结果形状。
4. 分别计算正确答案概率为 `0.9` 与 `0.01` 时的负对数似然。
5. 把正确 token 的 logit 增加 `1.0`，确认 loss 下降；再把错误 token 的 logit增加 `1.0`，观察 loss。

### 进阶练习

1. 故意实现一个不减最大值的 Softmax，用 `[1000,1001,1002]` 观察失败，再解释稳定版本为什么成功。
2. 为 `stable_softmax` 写一个测试：随机 logits 加上 `12345` 后，概率应在浮点误差内保持一致。
3. 不调用 `torch.nn.functional.cross_entropy`，只用 `logsumexp` 和索引实现交叉熵。
4. 假设模型对 `V` 个 token 给出完全均匀的概率，推导单位置交叉熵为什么等于 `ln(V)`。这个结果将在训练 Mini-LLM 时成为“随机基线”。
5. 比较把 batch 中所有位置求平均与只对非 padding 位置求平均的差异。

## 掌握检查

- [ ] 我能区分 logits、概率与 loss。
- [ ] 我能根据矩阵乘法两端形状推导输出形状。
- [ ] 我能解释为什么 `(B,T,V)` 的 Softmax 沿 `V` 维计算。
- [ ] 我能写出稳定 Softmax，并解释“减最大值”不改变结果。
- [ ] 我能从正确 token 的概率写出 `-ln(p)`。
- [ ] 我知道语言模型交叉熵为什么把 `(B,T,V)` 与 `(B,T)` 变成一个标量。
- [ ] 我知道初始均匀预测的 loss 约为 `ln(V)`。
- [ ] 我能解释 `zero_grad -> backward -> step` 的顺序。

完成后进入 [02 Tokenizer 与语言模型](./02_Tokenizer与语言模型.md)，把本章的概率工具放到真正的文本序列上。
