# 05 迷你 GPT：预训练与生成

对应 PDF：物理页 74-116。原资料直接进入 LLaMA2 组件和较大数据，本项目把配置缩到约几十万参数，并使用仓库内置语料，使你能在 CPU 上观察完整训练闭环。

## 本章成果

- 训练 `TinyGPT`，看到训练 loss 下降。
- 理解 x/y 右移、batch、反向传播、梯度裁剪和 checkpoint。
- 得到实验 10 共用的真实预训练 base，并理解为什么模型配置与 Tokenizer 必须随 checkpoint 冻结。
- 用 greedy、temperature、top-k 生成文本。
- 解释玩具模型为何会“像语料”，却没有通用语言能力。

## 1. 先认识配置

默认教学配置：

| 参数 | 值 | 意义 |
|---|---:|---|
| `block_size` | 不超过 48 | 模型一次可读的最长 token 数 |
| `n_layer` | 2 | Decoder Block 数 |
| `n_head` | 4 | 注意力头数 |
| `n_embd` | 64 | 每个位置的隐藏维度 |
| `vocab_size` | 由本地预训练语料决定 | 输出分类数量；语料前 90% 覆盖后续 SFT train/dev/test 所需字符 |
| `dropout` | 0 | 小实验先消除随机影响 |

参数量远小于真实 LLM。缩小不会改变 causal attention、残差、MLP、交叉熵和生成循环的逻辑。

## 2. 训练前先做三项预测

1. 初始模型接近均匀猜测时，loss 大约是 `ln(vocab_size)`。
2. 训练 loss 应下降；验证 loss 不保证一直下降。
3. 小语料上的生成会重复、背诵和断句异常。

把预测写进实验记录，再运行：

```powershell
# 约 40 步，只验证管线
.\.venv\Scripts\python.exe .\experiments\06_train_tiny_gpt.py --quick

# 默认 200 步，观察更清楚的趋势
.\.venv\Scripts\python.exe .\experiments\06_train_tiny_gpt.py
```

预期输出模式：

```text
参数量: ...
均匀随机理论 loss ln(V): ...
初始 train/val loss: ... / ...
step    1 | train ... | val ...
...
PASS: loss 明显下降，checkpoint 与训练指标已保存。
```

具体浮点数会因 PyTorch 版本和硬件略有差异；趋势和断言才是验收标准。

这一步不是可跳过的“热身”。它会生成 `checkpoints/tiny_gpt.pt`，实验 07 和实验 10 都从这里恢复同一套模型配置、预训练权重与字符 Tokenizer。若还没有该文件，实验 10 会要求先运行实验 06，而不会悄悄退回随机模型。

## 3. 一步训练逐行解释

```python
x, y = sample_language_model_batch(...)
optimizer.zero_grad(set_to_none=True)
logits, loss = model(x, y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

### `x, y`

从连续 token 中取长度 `T+1` 的窗口。`x` 是前 `T` 个，`y` 是后 `T` 个，所以 `y[t]` 正是 `x[t]` 的下一个 token。

### `zero_grad`

PyTorch 默认累加梯度。不清零就会把多个 step 的梯度叠在一起；梯度累积是可用技术，但必须有意控制。

### `forward + loss`

模型输出 `[B,T,V]`，每个位置有 V 个 logits。交叉熵选择目标 token 的负对数概率并求平均。

### `backward`

自动求导沿计算图计算每个可训练参数的梯度。

### `clip_grad_norm_`

当总梯度范数过大时按比例缩小，降低训练突然发散的风险。它不是修复所有训练问题的万能开关。

### `optimizer.step`

AdamW 根据梯度、动量统计和学习率更新参数。只有到这里权重才真正改变。

## 4. 训练集与验证集

本项目把语料前 90% 用于训练、后 10% 用于验证。

- train loss 下降：模型越来越适合见过的训练窗口。
- val loss 下降：对留出的同分布文本也有改善。
- train 降而 val 升：过拟合信号。

由于语料非常小且不是随机打散的独立样本，验证数值只能教学使用，不能当作严谨模型排行榜。

## 5. Checkpoint 保存了什么

`checkpoints/tiny_gpt.pt` 包含：

- 模型参数 `model_state`。
- 优化器状态，便于续训。
- 当前 step。
- 模型配置。
- 字符 Tokenizer 状态。
- 随机种子、预训练语料相对路径、后续指令词表来源，以及预训练语料、train/dev/test 与兼容数据视图的 SHA-256。

只保存权重却不保存配置和 Tokenizer，往往无法正确重建模型。Tokenizer id 映射一旦变化，即使 tensor shape 相同，语义也完全错位。

本项目在实验 06 生成词表后就把 Tokenizer 冻结。实验 10 不会根据指令数据重新建立词表，而会验证 train/dev/test 中的字符都已被预训练词表覆盖，并核对数据 SHA-256。这样即使 `checkpoints/` 被 Git 忽略，本地遗留的旧 base 也不能悄悄配合新版数据运行。仓库在语料前 90% 放置了一条自然的词表桥接文本，使这些字符不仅拥有 id，也实际参加预训练；修改指令模板或数据后若出现新字符，应更新预训练语料并重新运行实验 06，而不是在微调阶段扩展词表。

实验 07 和 10 使用严格 artifact loader：除了模型 state dict，它还校验格式版本、config、冻结 Tokenizer、数据指纹、tensor 键与形状。加载 LoRA 时还会验证 base SHA-256、targets、rank/alpha/dropout 和 adapter tensor；校验失败就停止，而不是部分加载一个表面上能运行的模型。

位置嵌入同样属于 checkpoint。当前 SFT 序列右移后的最大长度必须不超过 base 的 `block_size`；实验 10 会按 checkpoint 中的上限检查，不会为了容纳更长样本创建另一套位置嵌入。

## 6. 生成实验

先训练，再运行：

```powershell
.\.venv\Scripts\python.exe .\experiments\07_generate.py --prompt "语言模型" --tokens 80
```

不传 artifact 参数时，上述命令严格加载默认的 `checkpoints/tiny_gpt.pt`。实验 10 训练出 adapter 后，可在独立进程中显式组合 base + adapter：

```powershell
.\.venv\Scripts\python.exe .\experiments\07_generate.py `
  --base-checkpoint .\checkpoints\tiny_gpt.pt `
  --adapter .\checkpoints\tiny_gpt_lora_adapter.pt `
  --prompt "为什么需要因果掩码？" --tokens 80
```

`--base-checkpoint` 与 `--adapter` 必须成对出现，也不能和 `--checkpoint` 混用。adapter 不是完整模型文件；它只能与元数据所绑定的 base 一起恢复。

脚本比较三种策略：

### Greedy

\[
x_{t+1}=\arg\max_i p_i
\]

优点是确定；缺点是容易陷入高概率重复，且无法探索次高概率的合理延续。

### Temperature

\[
p_i=\operatorname{softmax}(z_i/\tau)
\]

- `τ < 1`：分布更尖锐，更保守。
- `τ > 1`：分布更平，更随机。
- Temperature 不修改模型权重，也不让事实更可靠。

### Top-k

只保留 k 个最大 logits，再归一化采样。它减少长尾低概率 token 造成的离题，但 k 太小会损失多样性。

## 7. 把 base 交给指令微调

完成实验 06 后，可运行同一 checkpoint 上的对照实验：

```powershell
.\.venv\Scripts\python.exe .\experiments\10_sft_tiny_gpt.py --quick
```

实验 10 不会把 Full SFT 的结果再作为 LoRA 的起点。两条预训练微调分支都从 `tiny_gpt.pt` 独立复制，因此 loss、原语料保持度代理、参数量和耗时才可比较。指令数据使用 4/4/4 的 train/dev/test 隔离：dev 用于调参，test 在训练结束后才评一次。详细的四分支设计、任务指标、JSON/CSV 记录与 adapter 文件见下一章。

## 8. 单变量探索

依次做，不要同时改：

1. 固定 `top_k=20`，比较 temperature 0.5、1.0、1.5。
2. 固定 temperature 1.0，比较 top-k 1、5、30。
3. 固定所有参数，只换随机种子。
4. 把 `block_size` 从 48 改 16，重新训练，观察局部长距离模式。
5. 增大 `n_embd`，记录参数量和每步耗时，而不假设一定改善验证 loss。

## 9. 常见故障

| 现象 | 优先检查 |
|---|---|
| loss 不降 | y 是否右移；优化器是否包含参数；学习率；是否调用 `step()` |
| loss 变 NaN | 学习率过大；输入/梯度是否有限；是否忘记缩放/裁剪 |
| 生成提示 OOV | 提示字符是否出现在训练语料；Tokenizer 是否同 checkpoint |
| 加载时报 shape mismatch | 配置与 checkpoint 是否一致 |
| 实验 10 找不到 base | 先运行实验 06；不要让微调脚本随机创建替代 checkpoint |
| 指令字符不在词表 | 更新预训练语料前 90% 的自然桥接文本，并重新运行实验 06 |
| SFT 序列超过窗口 | 缩短样本，或用更大 `block_size` 重新预训练；不能只扩大微调模型 |
| 输出乱码/重复 | 小语料和字符模型的预期局限；采样是否过热/过冷 |
| 训练快但验证差 | 过拟合、切分偏差、语料太小 |

## 10. 出门测

1. 为什么 `logits[:, -1, :]` 可以用于生成下一个 token？
2. `V=100` 时均匀预测的交叉熵约是多少？
3. 为什么保存模型时必须一并保存 Tokenizer？
4. 为什么“训练集 loss 很低”不能证明模型理解语言？
5. 为什么实验 10 必须复用实验 06 的 Tokenizer 和 `block_size`？
