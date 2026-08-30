# 06 SFT、LoRA 与对齐

对应 PDF：SFT 物理页 66-69、130-134；LoRA 物理页 135-142；偏好对齐物理页 69-72。

## 一句话先区分

- SFT：改变训练数据与监督位置，让模型学习指令-回答行为。
- LoRA：改变哪些参数参与训练，用低秩增量降低成本。
- 量化：用更低位宽表示权重/激活，降低内存或计算成本。
- 偏好对齐：改变优化目标，让 chosen 相对 rejected 更受偏好。

它们可以组合，但不是同义词。

## 0. 先确认实验关系

实验 10 是真实的 checkpoint-based 微调对比，必须先有实验 06 生成的 base：

```powershell
.\.venv\Scripts\python.exe .\experiments\06_train_tiny_gpt.py --quick
.\.venv\Scripts\python.exe .\experiments\10_sft_tiny_gpt.py --quick
```

实验 10 从 `checkpoints/tiny_gpt.pt` 恢复并冻结同一套 Tokenizer 与模型配置，然后从这个 base 独立分叉比较：

| 分支 | 起点 | 更新内容 | 主要用途 |
|---|---|---|---|
| pretrained base | 实验 06 checkpoint | 不训练 | 微调前基线 |
| random-init Full SFT | 相同 config 与 Tokenizer、随机权重 | 全部模型参数 | 观察“只靠四条指令从头训练” |
| pretrained Full SFT | 同一预训练 base | 全部模型参数 | 观察完整微调与原语料干扰迹象 |
| pretrained LoRA-SFT | 同一预训练 base | Attention LoRA A/B | 观察参数效率与冻结约束 |

Full SFT 和 LoRA-SFT 是从同一 base 出发的两种微调方式，不是先做 Full SFT、再给它套 LoRA。这样才能公平比较预训练、初始化和可训练参数集合各自的影响。

这里的“公平”指 config、Tokenizer、训练数据、评测数据以及两个预训练分支的起点一致。为了让 30 步 CPU 快速实验都能产生可观察信号，默认学习率是 Full SFT `3e-3`、random-init/LoRA-SFT `1e-2`；脚本会明确打印并写入产物 metadata。它不是“所有超参数完全相同”的单变量消融，不能仅凭最终 loss 或耗时断言某种算法普遍更优。

实验 08 则是独立的原理演示：它用一个小型 Linear 展示 LoRA 的初始化、梯度、冻结与权重合并，不读取 TinyGPT checkpoint，也不产出实际适配器。真实 TinyGPT LoRA-SFT 和 adapter-only 保存都在实验 10 中完成。

## 1. SFT 的 label mask

实验读取 `data/tiny_instructions.jsonl`，把每条样本格式化为 prompt 与 assistant 回复。Decoder-only 模型要预测“下一个 token”，所以标签不仅要 mask，还要右移对齐：

```python
sequence = prompt_ids + response_ids
input_ids = sequence[:-1]
labels = [-100] * (len(prompt_ids) - 1) + response_ids
```

`input_ids` 与 `labels` 等长。最后一个 prompt token 负责预测第一个 response token，因此第一个有效 label 位于它的输入位置。交叉熵设置 `ignore_index=-100` 后，prompt 仍是模型可见上下文，但 prompt 目标与右侧 padding 都不贡献监督 loss。

为什么不能硬编码“assistant 开始标记”的 token id？因为换一个 Tokenizer 后，标记可能被切成不同数量和不同编号的 token。应动态编码模板或在构造数据时明确边界。

实验 10 会检查所有序列都不超过 base checkpoint 的 `block_size`，并使用 checkpoint 中的冻结 Tokenizer；它不会依据 SFT 数据重建词表。Full SFT 的完整模型权重与 metadata 保存到 `checkpoints/tiny_gpt_sft.pt`，但不含 optimizer 状态，不能直接据此无缝续训。

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

实验 10 的真实 LoRA-SFT 默认把每个 Transformer Block 中的 `attention.qkv` 和 `attention.output` 包装为 LoRA，使用 `rank=4`、`alpha=8`。其余 base 参数全部冻结；与 token embedding 共享权重的 `lm_head` 不作为 LoRA target，以免破坏权重绑定。

## 5. LoRA 运行后应该看到

运行实验 08 时，预期看到单层原理验证：

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

运行实验 10 时，还要看到 LoRA 注入前后的 TinyGPT logits 一致、所有冻结参数训练前后逐元素不变，以及 adapter-only 文件恢复后的 logits 与训练完成模型完全一致。

## 6. 怎样读四分支指标

实验 10 同时报告：

- `train loss`：只在四条训练指令的 assistant token 上计算，用来观察拟合。
- `heldout loss`：在 `data/tiny_instructions_eval.jsonl` 的未参与 SFT 的指令上计算，用来观察指令格式迁移；其概念和答案字符可能已在预训练语料出现，所以它不是未知知识或独立泛化评测，也不保证随训练下降。
- `corpus val`：回到原预训练语料后 10% 的固定种子采样窗口计算，作为保持度/干扰代理；这个微型指标只能提示遗忘迹象，不能单独证明灾难性遗忘。
- `trainable/total` 与耗时：区分 Full SFT 和 LoRA-SFT 的参数成本。
- seen/heldout greedy completion：只作直观案例，不能替代 loss 与失败分析。

随机初始化 SFT 即使把训练 loss 压低，也不代表学到了通用语言能力。预训练 Full SFT 可能在指令 loss 上下降更快，却使原语料验证 loss 上升；LoRA-SFT 的参数更少，也不保证 heldout 一定优于 Full SFT。实验的目标是把这些差异测出来，而不是预设某一分支必胜。

实验会生成两个用途不同的产物：

- `checkpoints/tiny_gpt_sft.pt`：预训练 Full SFT 的完整模型权重与 metadata（相对于 adapter-only；不含 optimizer 状态）。metadata 含 base/data SHA-256、步数、学习率、优化器配置与最终指标。
- `checkpoints/tiny_gpt_lora_adapter.pt`：只包含 LoRA A/B；同时记录 base/data SHA-256、Tokenizer、config、targets、rank、alpha、步数、学习率、优化器配置与最终指标。

adapter 不是完整模型。恢复时必须先加载 SHA-256 匹配的 base，再注入相同 targets，最后加载 A/B。

## 7. rank 与 alpha

- rank `r`：低秩通道容量；越大，可训练参数越多，表达能力通常更强，但不是必然更好。
- alpha：更新缩放；实现常用 `alpha/r`，使不同 rank 的更新尺度更易控制。
- dropout：只作用于 LoRA 分支，是正则化手段。

单变量练习：把 rank 改为 1、2、4，记录参数量、最终 loss 和收敛步数。不要同时改学习率。

## 8. LoRA 与 QLoRA

QLoRA 通常是“量化冻结的基础模型 + 可训练 LoRA 适配器”。

- 量化减少基础权重内存。
- LoRA 减少需要更新和保存的参数。
- 量化误差、计算内核和设备兼容性是额外变量。

本项目必做实验不引入量化库，因为低秩原理可以在普通 Linear 上清楚验证。

## 9. 偏好对齐的边界

假设有 `(prompt, chosen, rejected)`：

```text
prompt: 请解释因果掩码
chosen: 给出定义、用途和限制
rejected: 只说“它让模型更聪明”
```

偏好目标让模型更倾向 chosen，但 chosen 标注本身也可能有偏差。对齐不是事实验证器；仍需数据质量、离线评测、安全测试和线上监控。

## 10. 常见错误

- base 参数仍在优化器中：统计 `requires_grad`，并检查优化器参数组。
- 初始输出不同：检查 B 是否零初始化、缩放和 dropout。
- 训练后 base 改变：可能没有冻结，或错误地把合并权重写回后继续比较。
- SFT loss 包含 prompt：检查 labels 的 `-100` 区域。
- SFT label 边界错一位：先构造完整序列，再同时检查 `input_ids=sequence[:-1]` 和右移后的 labels。
- padding 参与 loss：batch 中所有补齐位置的 label 也必须是 `-100`。
- 把 LoRA 文件当完整模型：适配器通常依赖正确版本的基础模型。
- 先做 Full SFT 再比较 LoRA：两个分支起点不同，参数效率与原语料保持度对比失去意义。
- 用指令数据重建 Tokenizer：相同 token id 可能改变含义；必须使用 base checkpoint 内的冻结状态。
- 复用数据版本不匹配的旧 base：实验 10 会校验数据 SHA-256，并要求重新运行实验 06。
- 只看训练 loss：同时检查 heldout、原语料验证 loss、冻结参数和 adapter 往返。

## 11. 出门测

1. `d_in=d_out=4096,r=8` 时，完整权重与 LoRA 可训练参数各是多少？
2. 为什么 LoRA 可以减少训练成本，却不保证推理模型本体很小？
3. SFT 和 LoRA 能否同时使用？分别描述数据目标与参数集合。
4. 为什么低秩适配成功不等于解决灾难性遗忘、事实错误或安全问题？
5. 为什么 Full SFT 与 LoRA-SFT 必须各自从同一个 base checkpoint 分叉？
