# 03 注意力与 Transformer：让每个位置只读取它应该看到的上下文

> PDF 对照：《Happy-LLM v1.0》物理页 15-20 依次讨论注意力、缩放点积、自注意力、因果掩码与多头注意力；物理页 21-24 讨论 Encoder-Decoder、前馈网络、归一化和残差；物理页 25-30 讨论 embedding、位置编码和完整 Transformer 的组装。本章以 Decoder-only 路线重写为两个 CPU 实验。所有措辞、推导、例子和代码说明均为本项目原创改写，页码只用于定位 PDF 主题。

## 本节问题

Bigram 只能看前一个 token。现在我们希望当前位置能读取更长的上下文，但不能在训练时偷看未来答案：

1. Query、Key、Value 各自做什么？
2. 为什么相似度用 `QK^T`，结果是什么形状？
3. 为什么要除以 `sqrt(D_h)`？
4. 因果掩码究竟遮住矩阵的哪一半？
5. 多头注意力为什么不等于简单重复同一个头？
6. token 位置信息从哪里来？
7. 残差、归一化、前馈网络怎样与注意力组成一个 Block？
8. 怎样用测试证明模型没有读取未来 token？

对应实验：

```text
experiments/04_attention.py
experiments/05_transformer_block.py
```

## 第一部分：缩放点积注意力

### 直觉：检索、匹配、取回

可以把注意力看成一次可微分检索：

- Query：当前位置在寻找什么；
- Key：每个候选位置用什么标签参与匹配；
- Value：匹配到该位置后，实际取回什么内容。

Query 与 Key 的点积产生匹配分数。Softmax 把分数变成权重，再对 Value 加权求和。

“注意力权重”描述这一次信息混合的比例，但它不自动等于完整、可靠的因果解释。不要把某个权重较大直接翻译为“模型之所以回答，是因为这个词”。

### 定义与公式

缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}+M\right)V
$$

其中：

- `Q`、`K` 的最后一维都是 `D_h`；
- `QK^T` 给出每个 Query 对每个 Key 的分数；
- `sqrt(D_h)` 是缩放因子；
- `M` 是可选掩码；
- Softmax 沿 Key 位置维计算；
- 最后乘 `V` 得到聚合后的内容。

### 为什么需要缩放？

若 `q_i`、`k_i` 是大致独立、均值 0、方差 1 的分量，长度 `D_h` 的点积：

$$
q\cdot k=\sum_{i=1}^{D_h}q_i k_i
$$

其波动规模会随 `D_h` 增大。过大的分数进入 Softmax 后，分布容易过度尖锐：一个位置接近 1，其余接近 0，梯度也可能变得不利于训练。除以 `sqrt(D_h)` 把分数拉回较稳定的尺度。

这不是为了让结果“更小看起来漂亮”，而是为了控制统计尺度。

## 形状推导：单头注意力

先忽略 batch：

```text
Q:       (T_q,D_h)
K:       (T_k,D_h)
V:       (T_k,D_v)
K^T:     (D_h,T_k)
scores:  (T_q,T_k)
weights: (T_q,T_k)
output:  (T_q,D_v)
```

矩阵乘法：

```text
(T_q,D_h) @ (D_h,T_k) -> (T_q,T_k)
(T_q,T_k) @ (T_k,D_v) -> (T_q,D_v)
```

自注意力中，Q、K、V 来自同一序列，因此通常 `T_q = T_k = T`：

```text
scores/weights: (T,T)
```

加入 batch 后：

```text
Q,K,V:  (B,T,D_h)
scores: (B,T,T)
output: (B,T,D_v)
```

每一行权重都应满足：

$$
\sum_{j=1}^{T} A_{ij}=1
$$

也就是说，每个 Query 对所有可见 Key 的权重之和为 1。

## 因果掩码

### 直觉：并行训练，但不泄露答案

训练序列：

```text
<bos> 我 喜欢 学习
```

模型需要同时学习：

```text
<bos>       -> 我
<bos> 我    -> 喜欢
<bos> 我 喜欢 -> 学习
```

为了利用矩阵并行计算，我们一次输入完整序列；为了不偷看答案，位置 `i` 只能读取 `j <= i` 的 token。

### 掩码定义

采用加法掩码时：

$$
M_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

`j > i` 是当前 Query 右侧的未来 Key。加上 `-inf` 后，Softmax 将其概率变成 0。

长度 `T=4` 时，可见性矩阵是下三角：

```text
query 0: [可见, 屏蔽, 屏蔽, 屏蔽]
query 1: [可见, 可见, 屏蔽, 屏蔽]
query 2: [可见, 可见, 可见, 屏蔽]
query 3: [可见, 可见, 可见, 可见]
```

如果代码用布尔 mask，要先确认 `True` 代表“保留”还是“屏蔽”。不同 API 的约定可能相反。

### 掩码形状

单头可用：

```text
mask: (T,T) 或 (1,T,T)
```

多头 batch 通常使用可广播形状：

```text
mask: (1,1,T,T)
```

它会广播到 `(B,H,T,T)`，同一条因果规则供所有 batch 和 head 使用。

## 运行注意力实验

PowerShell：

```powershell
Set-Location 'D:\path\to\LearnLLM'
.\.venv\Scripts\python.exe .\experiments\04_attention.py
```

### 运行前先预测

1. 若 `Q,K,V` 都是 `(2,4,8)`，单头输出是什么形状？权重是什么形状？
2. 权重最后一维的和应是多少？
3. 因果模式下，`weights[:,0,1:]` 应接近什么？
4. 第三个位置能否读取第一个位置？能否读取第四个位置？
5. 把 `V` 全部设为同一个向量，所有位置的输出会有什么特点？
6. 去掉 `sqrt(D_h)` 后，较大维度的注意力通常会更平坦还是更尖锐？

### 预期输出与验收

脚本打印完整的 3×3 教学矩阵；输出模式类似：

```text
Q/K/V shape: (1, 1, 3, 2) ...
attention weights (no mask): <3×3 矩阵>
row sums: tensor([[[1., 1., 1.]]])
attention output: <3×2 矩阵>
causal weights: <上三角为 0 的 3×3 矩阵>
scaled mean entropy: ...
unscaled mean entropy: ...
PASS: 权重逐行归一化，因果掩码完全阻断未来位置。
```

不变量：

- [ ] 输出的 Query 长度与 `Q` 相同；
- [ ] 权重形状最后两维是 `(T_q,T_k)`；
- [ ] 每一行可见位置的概率和约为 1；
- [ ] 因果模式中所有 `j > i` 的权重接近 0；
- [ ] 输出等于权重对 `V` 的加权和；
- [ ] 输出、权重都没有 `NaN` 或 `inf`；
- [ ] 关闭因果模式时，未来位置允许获得非零权重。

## 第二部分：多头注意力

### 直觉

单头只能在一个投影空间中计算相似度。多头注意力为不同头使用不同的可学习投影，使它们能够形成不同的匹配方式。不能预先规定某个头一定学习语法、另一个头一定学习指代；这是训练可能出现的行为，不是结构保证。

### 定义

对输入 `X`：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

将模型维度 `D` 分成 `H` 个头：

$$
D_h=D/H
$$

每个头独立计算：

$$
head_h=Attention(Q_h,K_h,V_h)
$$

拼接并投影：

$$
MHA(X)=Concat(head_1,\ldots,head_H)W_O
$$

必须满足 `D % H == 0`。

### 多头形状追踪

```text
input X                    (B,T,D)
Q/K/V projection           (B,T,D)
reshape                     (B,T,H,D_h)
transpose for attention     (B,H,T,D_h)
scores                      (B,H,T,T)
weights                     (B,H,T,T)
per-head output             (B,H,T,D_h)
transpose back              (B,T,H,D_h)
concatenate                 (B,T,D)
output projection           (B,T,D)
```

例：`B=2,T=5,D=16,H=4`，则 `D_h=4`，注意力分数形状为 `(2,4,5,5)`。

## 第三部分：位置、残差、归一化与前馈网络

### 为什么需要位置信息？

不额外提供位置时，自注意力主要根据内容相似度混合信息，本身没有稳定的绝对或相对顺序标记。Transformer 会加入或注入位置信息。

PDF 物理页 26-28 介绍经典正弦/余弦位置编码。一种常见形式为：

$$
PE(pos,2i)=\sin\left(pos/10000^{2i/D}\right)
$$

$$
PE(pos,2i+1)=\cos\left(pos/10000^{2i/D}\right)
$$

它的形状是 `(T,D)`，可广播并加到 `(B,T,D)` 的 token embeddings 上。现代 Decoder-only 模型也常使用 RoPE 等相对位置方案；本阶段先掌握“内容表示必须与位置信息结合”这一结构要求。

### 残差连接

残差连接：

$$
y=x+f(x)
$$

它为信息与梯度提供直接路径。相加要求两个张量形状相同，所以注意力子层与 FFN 子层都通常保持最后维度 `D` 不变。

### 归一化

LayerNorm 对每个 token 的特征维进行归一化。输入和输出形状相同：

```text
(B,T,D) -> (B,T,D)
```

本项目使用的具体归一化以实现为准。关键检查是：归一化不改变 batch、序列长度和模型维度，并保持输出有限。

### 前馈网络

FFN 对每个位置独立应用同一组非线性变换：

$$
FFN(x)=W_2\,\sigma(W_1x+b_1)+b_2
$$

形状通常是：

```text
(B,T,D)
 -> Linear: (B,T,D_ff)
 -> activation: (B,T,D_ff)
 -> Linear: (B,T,D)
```

注意力在位置之间混合信息；FFN 在每个位置的特征维上进行更丰富的非线性变换。

## 一个 Decoder Transformer Block

常见的 Pre-Norm 结构可写为：

$$
x_1=x+MHA(Norm_1(x))
$$

$$
y=x_1+FFN(Norm_2(x_1))
$$

数据流：

```text
x (B,T,D)
  -> Norm -> causal MHA -> (B,T,D) -> residual add
  -> Norm -> FFN        -> (B,T,D) -> residual add
y (B,T,D)
```

Block 不负责直接输出词表概率。完整语言模型还需要：

```text
token ids (B,T)
 -> token embedding + position information (B,T,D)
 -> N x Transformer Block (B,T,D)
 -> final norm (B,T,D)
 -> language-model head (B,T,V)
```

## Encoder、Decoder 与本项目选择

PDF 物理页 21-24 介绍 Encoder-Decoder：

- Encoder 的自注意力通常可以双向读取整条输入；
- Decoder 的自注意力需要因果掩码；
- Encoder-Decoder 模型还使用 cross-attention，让 Decoder 的 Query 读取 Encoder 产生的 Key/Value。

本项目先实现 Decoder-only Block，因为 next-token 生成只需要一条因果序列：

```text
过去 token -> 因果 Decoder -> 下一个 token logits
```

这不是说 Encoder 或 Encoder-Decoder 不重要，而是先用最短路径理解现代生成式 LLM 的核心训练目标。

## 运行 Transformer Block 实验

```powershell
Set-Location 'D:\path\to\LearnLLM'
.\.venv\Scripts\python.exe .\experiments\05_transformer_block.py
```

相关测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

### 运行前先预测

设 `B=2,T=6,D=16,H=4,D_ff=64`：

1. `D_h` 是多少？
2. 每头 scores 是什么形状？全部头合在一起呢？
3. Block 输入与输出形状是否相同？
4. FFN 中间张量是什么形状？
5. 将第 5 个 token 改掉，位置 0-4 的输出是否应变化？位置 5 呢？
6. 若 `D=15,H=4`，模型应正常运行还是尽早报错？

答案：`D_h=4`；单头 `(2,6,6)`、合并头 `(2,4,6,6)`；相同；中间为 `(2,6,64)`；因果且关闭 dropout 时位置 0-4 不应变化、位置 5 可以变化；应尽早因不可整除而报错。

### 预期输出与验收

脚本直接从整个 TinyGPT 的 logits 检查因果性；输出模式类似：

```text
input shape:  (1, 6) = [B, T]
logits shape: (1, 6, 32) = [B, T, vocab]
parameters:   <参数量>
未来 token 改变后，过去 logits 最大差异: 0.000e+00
PASS: 模型形状正确，并且没有未来信息泄漏。
```

验收：

- [ ] `D % H == 0`，并正确计算 `D_h`；
- [ ] 拆头、换轴、合头后，元素数量保持一致；
- [ ] Block 输入输出都是 `(B,T,D)`；
- [ ] FFN 中间维扩展后再回到 `D`；
- [ ] 注意力和 FFN 两条残差相加的形状完全一致；
- [ ] 扩展练习：forward 输出全为有限数；
- [ ] 扩展练习：对输出求标量 loss 后，参数梯度存在且有限；
- [ ] `eval()` 模式下修改未来 token，不改变过去位置输出；
- [ ] 修改过去 token 可以影响其后位置，说明上下文路径确实存在。

## 最重要的无泄漏测试

只检查 mask 图案还不够。更强的行为测试是：

1. 创建两条完全相同的 token 序列；
2. 只修改第二条序列最后一个 token；
3. 在 `model.eval()` 与 `torch.no_grad()` 下分别前向传播；
4. 比较最后位置之前的输出或 logits。

如果模型是严格因果的，前缀输出应在浮点误差内相同：

```text
allclose(output_a[:, :-1], output_b[:, :-1]) == True
```

最后位置可以不同，因为它能看到被修改的 token。若前缀也变化，可能存在：

- mask 三角方向反了；
- mask 没有广播到全部 head；
- 位置或 batch 维换错；
- 测试仍在 train 模式，dropout 引入随机差异；
- 比较的并不是同一模型参数下的两次 forward。

## 常见故障与排查

### 1. `D` 不能被 `H` 整除

`D_h=D/H` 必须是整数。初始化时立即断言：

```text
D % H == 0
```

不要在 reshape 时报一个难懂的元素数量错误后才发现。

### 2. scores 的形状不是 `(B,H,T,T)`

检查 `Q,K` 是否先变为 `(B,H,T,D_h)`，并只转置 `K` 的最后两维：

```text
Q @ K.transpose(-2,-1)
```

不要误把 batch 或 head 维转走。

### 3. 因果 mask 方向反了

手工打印 `T=4` 的 mask。第 0 行只能看到第 0 列；最后一行应能看到全部列。若相反，三角方向或布尔语义错误。

### 4. 第一行注意力全是 `NaN`

这通常表示第一行所有位置都被屏蔽，Softmax 正在处理一整行 `-inf`。因果 self-attention 必须允许每个位置读取自己，即保留对角线。

### 5. `view` 报张量不连续

`transpose` 后内存布局可能不连续。可以使用 `reshape`，或在 `view` 前调用 `contiguous()`。但先确认换轴逻辑正确，不要只为消除报错而随意 contiguous。

### 6. 合并多头后 token 顺序混乱

正确路线是：

```text
(B,H,T,D_h) -> transpose -> (B,T,H,D_h) -> reshape -> (B,T,D)
```

不能直接把 `(B,H,T,D_h)` reshape 成 `(B,T,D)`，否则 head 与 token 的内存次序会混合。

### 7. 无泄漏测试偶尔失败

先调用 `model.eval()` 关闭 dropout，并在同一个模型上比较。再检查绝对/相对容差，而不是要求浮点逐位相等。

### 8. 梯度为 `None`

确认参数参与了 forward，loss 由输出计算，且中间没有 `.detach()`、转换成 NumPy 或包在 `torch.no_grad()` 中。

### 9. 梯度出现 `NaN` 或特别大

逐层检查第一个非有限值；确认注意力有缩放、mask 不产生全 `-inf` 行、Softmax 使用稳定实现。不要先用梯度裁剪掩盖前向传播错误。

## 练习

### 注意力基础练习

1. 手工设置一个 Query，使它与某个 Key 完全相同、与其他 Key 正交，预测注意力最大位置。
2. 令所有 logits 相同，推导未屏蔽位置上的注意力分布。
3. 对 `T=5` 写出因果可见性矩阵，数出一共有多少个可见元素。
4. 比较 `causal=False` 与 `causal=True` 的权重矩阵，说明哪些元素必须变化。
5. 把 Value 的每一行设为对应位置编号，解释输出为何是“被关注位置编号的加权平均”。

### 多头与 Block 练习

1. 对 `D=24`，列出 `H=1,2,3,4,6,8` 时的 `D_h`。
2. 在保持 `D` 不变时改变头数，比较参数量是否一定按头数线性增长。提示：考虑组合的 Q/K/V 投影矩阵。
3. 暂时去掉输出投影 `W_O`，记录形状并说明表达能力上少了什么混合步骤。
4. 分别画出 Pre-Norm 与 Post-Norm 的数据流，不要求立刻断言哪一个“永远更好”。
5. 将 FFN 扩展倍率从 `4D` 改为 `2D`，计算参数量变化。

### 关键挑战

1. 自己实现并通过“未来 token 修改不影响过去输出”的行为测试。
2. 去掉因果 mask，证明无泄漏测试会失败；再恢复 mask，证明测试通过。
3. 去掉缩放因子，比较不同 `D_h` 下注意力权重熵。记录现象，不只描述结论。
4. 画出正弦位置编码前四个通道随位置变化的曲线，观察不同频率。
5. 堆叠两个 Block，验证形状保持 `(B,T,D)`，并检查所有层都获得梯度。
6. 估算 self-attention 的 scores 张量元素数 `B*H*T*T`。当 `T` 翻倍时，元素数变为多少倍？这解释了长上下文的主要成本之一。

## 掌握检查

- [ ] 我能用“检索、匹配、取回”解释 Q、K、V，同时知道这只是直觉类比。
- [ ] 我能从 `Q`、`K` 形状推导 `QK^T` 的形状。
- [ ] 我能写出缩放点积注意力完整公式。
- [ ] 我能解释缩放因子为什么是 `sqrt(D_h)` 量级。
- [ ] 我能画出因果 mask，并确认第 0 行只看自己、最后一行看全部过去。
- [ ] 我能追踪 `(B,T,D)` 拆成 `(B,H,T,D_h)` 再合回去的每一步。
- [ ] 我能区分注意力负责“位置间混合”、FFN 负责“每个位置的特征变换”。
- [ ] 我能解释残差相加为什么要求形状不变。
- [ ] 我知道位置编码解决什么问题。
- [ ] 我能用行为测试证明没有未来信息泄漏，而不只目测 mask。
- [ ] 我能说明 Encoder 双向注意力、Decoder 因果注意力和 cross-attention 的差别。

完成本章后，你已经从 token id 走到了一个可运行的 Decoder Transformer Block。下一阶段只需加入数据切片、语言模型头、优化循环与生成策略，就能组成一个真正可训练的 Mini-LLM。
