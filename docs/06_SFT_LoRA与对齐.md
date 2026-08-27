# 06 SFT、LoRA 与对齐

对应 PDF：SFT 物理页 66-69、130-134；LoRA 物理页 135-142；偏好对齐物理页 69-72。

## 一句话先区分

- SFT：改变训练数据与监督位置，让模型学习指令-回答行为。
- LoRA：改变哪些参数参与训练，用低秩增量降低成本。
- 量化：用更低位宽表示权重/激活，降低内存或计算成本。
- 偏好对齐：改变优化目标，让 chosen 相对 rejected 更受偏好。

它们可以组合，但不是同义词。

## 1. SFT 的 label mask

执行：

```powershell
.\.venv\Scripts\python.exe .\experiments\10_sft_tiny_gpt.py --quick
```

实验读取 `data/tiny_instructions.jsonl`，把每条样本格式化为 prompt 与 assistant 回复。Decoder-only 模型要预测“下一个 token”，所以标签不仅要 mask，还要右移对齐：

```python
sequence = prompt_ids + response_ids
input_ids = sequence[:-1]
labels = [-100] * (len(prompt_ids) - 1) + response_ids
```

`input_ids` 与 `labels` 等长。最后一个 prompt token 负责预测第一个 response token，因此第一个有效 label 位于它的输入位置。交叉熵设置 `ignore_index=-100` 后，prompt 仍是模型可见上下文，但 prompt 目标与右侧 padding 都不贡献监督 loss。

为什么不能硬编码“assistant 开始标记”的 token id？因为换一个 Tokenizer 后，标记可能被切成不同数量和不同编号的 token。应动态编码模板或在构造数据时明确边界。

实验会打印训练前后逐样本 assistant loss、参数 L2 变化、greedy 输出，并把 checkpoint 写到 `output/checkpoints/tiny_gpt_sft.pt`。验收不是只看平均 loss：四条样本都要下降，参数必须真的改变，checkpoint 必须能够保存。

## 2. LoRA 原理

执行：

```powershell
.\.venv\Scripts\python.exe .\experiments\08_lora.py
```

原线性层：

\[
h=W_0x
\]

LoRA 冻结 `W0`，学习低秩增量：

\[
h=W_0x+\frac{\alpha}{r}BAx
\]

若 `W0` 形状为 `[d_out,d_in]`：

- `A` 形状 `[r,d_in]`。
- `B` 形状 `[d_out,r]`。
- 可训练参数量 `r(d_in+d_out)`，而完整权重是 `d_in*d_out`。

当 `r` 远小于输入/输出维度时，可训练参数显著减少。

## 3. 为什么 B 初始化为 0

本项目 A 随机初始化、B 零初始化。因此训练开始时 `BA=0`，LoRA 层输出与原层完全一致，不会一包裹就破坏预训练函数。

第一步时，B 可以先获得梯度；A 的梯度可能因 B=0 而为 0。B 更新后，后续 A 也能获得梯度。这是正常现象，不要用“第一步 A 梯度必须非零”作为错误断言。

## 4. 合并权重

推理前可以计算：

\[
W_{merged}=W_0+\frac{\alpha}{r}BA
\]

这样部署时仍是一层普通 Linear，不必额外执行两次低秩乘法。本项目实验验证合并前后输出在浮点误差内一致。

## 5. LoRA 运行后应该看到

```text
初始 LoRA 增量最大值: 0...
loss: 较大值 -> 很小值
base.requires_grad: False
A/B gradient present: True True
PASS: 原权重冻结，低秩参数完成适配，合并前后输出一致。
```

验收要看四件事：

1. 包裹前后初始输出一致。
2. base 权重训练前后逐元素相同。
3. A/B 是可训练参数并最终有梯度。
4. 合并权重后的输出和 LoRA 分支输出一致。

## 6. rank 与 alpha

- rank `r`：低秩通道容量；越大，可训练参数越多，表达能力通常更强，但不是必然更好。
- alpha：更新缩放；实现常用 `alpha/r`，使不同 rank 的更新尺度更易控制。
- dropout：只作用于 LoRA 分支，是正则化手段。

单变量练习：把 rank 改为 1、2、4，记录参数量、最终 loss 和收敛步数。不要同时改学习率。

## 7. LoRA 与 QLoRA

QLoRA 通常是“量化冻结的基础模型 + 可训练 LoRA 适配器”。

- 量化减少基础权重内存。
- LoRA 减少需要更新和保存的参数。
- 量化误差、计算内核和设备兼容性是额外变量。

本项目必做实验不引入量化库，因为低秩原理可以在普通 Linear 上清楚验证。

## 8. 偏好对齐的边界

假设有 `(prompt, chosen, rejected)`：

```text
prompt: 请解释因果掩码
chosen: 给出定义、用途和限制
rejected: 只说“它让模型更聪明”
```

偏好目标让模型更倾向 chosen，但 chosen 标注本身也可能有偏差。对齐不是事实验证器；仍需数据质量、离线评测、安全测试和线上监控。

## 9. 常见错误

- base 参数仍在优化器中：统计 `requires_grad`，并检查优化器参数组。
- 初始输出不同：检查 B 是否零初始化、缩放和 dropout。
- 训练后 base 改变：可能没有冻结，或错误地把合并权重写回后继续比较。
- SFT loss 包含 prompt：检查 labels 的 `-100` 区域。
- SFT label 边界错一位：先构造完整序列，再同时检查 `input_ids=sequence[:-1]` 和右移后的 labels。
- padding 参与 loss：batch 中所有补齐位置的 label 也必须是 `-100`。
- 把 LoRA 文件当完整模型：适配器通常依赖正确版本的基础模型。

## 10. 出门测

1. `d_in=d_out=4096,r=8` 时，完整权重与 LoRA 可训练参数各是多少？
2. 为什么 LoRA 可以减少训练成本，却不保证推理模型本体很小？
3. SFT 和 LoRA 能否同时使用？分别描述数据目标与参数集合。
4. 为什么低秩适配成功不等于解决灾难性遗忘、事实错误或安全问题？
