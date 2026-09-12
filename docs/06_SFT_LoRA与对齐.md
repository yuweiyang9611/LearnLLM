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
$SftRun = ".\outputs\sft\lesson-$(Get-Date -Format yyyyMMdd-HHmmssfff)"
.\.venv\Scripts\python.exe .\experiments\10_sft_tiny_gpt.py --quick --output-dir $SftRun
```

`--quick` 把每个训练分支固定为 30 步，但仍默认运行 seed 42/43/44。所有关键输入和输出都可配置；例如：

```powershell
.\.venv\Scripts\python.exe .\experiments\10_sft_tiny_gpt.py `
  --steps 120 --seeds 7 11 19 `
  --train-data .\data\tiny_instructions.jsonl `
  --dev-data .\data\tiny_instructions_dev.jsonl `
  --test-data .\data\tiny_instructions_test.jsonl `
  --output-dir .\outputs\sft\my-comparison
```

`--help` 还列出 base/full/adapter 路径、三种学习率、LoRA rank/alpha/dropout、生成长度、train batch size 和设备。自定义数据必须保留相应的 `split` 字段，并满足冻结 Tokenizer 与 base `block_size` 的约束。

实验 10 默认经 `outputs/pretrain/latest.json` 找到 `base.pt` 并恢复并冻结同一套 Tokenizer 与模型配置，然后从这个 base 独立分叉比较：

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

实验读取结构化 JSONL，把每条样本格式化为 prompt 与 assistant 回复。每行都有稳定 `id`、`split`、`intent_family`、`evaluation_category`、必要关键词和格式规则。Decoder-only 模型要预测“下一个 token”，所以标签不仅要 mask，还要右移对齐：

```python
response_ids = response_ids + [tokenizer.eos_id]
sequence = prompt_ids + response_ids
input_ids = sequence[:-1]
labels = [-100] * (len(prompt_ids) - 1) + response_ids
```

`input_ids` 与 `labels` 等长。最后一个 prompt token 负责预测第一个 response token，因此第一个有效 label 位于它的输入位置。EOS 是最后一个有效 label，生成时用它决定停止。交叉熵设置 `ignore_index=-100` 后，prompt 仍是模型可见上下文，但 prompt 目标与右侧 padding 都不贡献监督 loss。

为什么不能硬编码“assistant 开始标记”的 token id？因为换一个 Tokenizer 后，标记可能被切成不同数量和不同编号的 token。应动态编码模板或在构造数据时明确边界。

实验 10 会检查所有序列都不超过 base checkpoint 的 `block_size`，并使用 checkpoint 中的冻结 Tokenizer；它不会依据 SFT 数据重建词表。Full SFT 的完整模型权重与 metadata 保存到 `outputs/sft/<run-id>/seed-<seed>/full/model.pt`，但不含 optimizer 状态，不能直接据此无缝续训。

### 1.1 Train/dev/test 协议

| 文件 | 数量 | 类别 | 用途 |
|---|---:|---|---|
| `data/tiny_instructions.jsonl` | 4 | `seen` | 唯一参与梯度更新的数据 |
| `data/tiny_instructions_dev.jsonl` | 4 | `paraphrase` | 调步数、学习率、rank 等配置 |
| `data/tiny_instructions_test.jsonl` | 4 | 2 条 `paraphrase` + 2 条 `new_intent` | 配置确定后的一次最终评测 |

train 与 dev 共享四个意图族，dev 改写提问；test 的两条 `new_intent` 使用未在 train/dev 出现的意图族。实验会拒绝跨切分重复 id。每个训练分支和 seed 都先完成训练，才在同一个最终评测阶段对 test 运行一次 teacher-forced loss 与 greedy 任务评测。学习者只能根据 dev 调参，不能反复查看 test 结果再选择配置。

这里的 `new_intent` 只表示“意图族未出现在 SFT train/dev”，不表示相关字符、概念或知识从未出现在预训练语料。它可用于观察有限的意图迁移失败，却不能支持“模型学会未知知识”或“具有独立分布泛化能力”的结论。

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

实验 10 对 train/dev/test 都记录 assistant-only loss 和 greedy 生成，并计算三类透明指标：

- `task_success_rate`：一条回答必须命中全部概念组、未命中声明的矛盾规则并满足全部格式规则才记 1，否则记 0；这是最严格的主指标。
- `keyword_accuracy`：逐条计算已命中必要关键词的比例，再对样本取平均。
- `format_accuracy`：是否满足句末、最大句数和禁止前缀等全部规则，再对样本取平均。

JSON 中既保留每条 prediction、命中关键词和分数，也按 `seen`、`paraphrase`、`new_intent` 类别汇总。规则是确定性的教学启发式，不是语义等价或人工质量评审。Tiny 字符模型生成不稳定，严格任务成功率为 0% 完全可能是诚实结果；不要改宽规则或偷看 test 只为得到非零数字。

还要一起阅读：

- `corpus_validation_loss`：回到原预训练语料后 10% 的固定种子采样窗口计算，作为保持度/干扰代理；它只能提示遗忘迹象，不能单独证明灾难性遗忘。
- `trainable_parameters`、`total_parameters` 与 `duration_seconds`：区分 Full SFT 和 LoRA-SFT 的参数与时间成本。
- train/dev/test 生成案例：用于定位失败，不能替代量化指标。

默认运行 seed 42/43/44。JSON 对每个可汇总 loss 和任务指标报告 `mean` 与 population `std`（总体标准差），并保留逐 seed 原始结果；参数量和耗时也按 seed 保留。随机初始化 SFT 即使把 train loss 压低，也不代表学到了通用语言能力。预训练 Full SFT 可能使原语料验证 loss 上升；LoRA-SFT 参数更少，也不保证 dev/test 更好。实验的目标是测出差异，而不是预设某一分支必胜。

每次运行会生成四类产物：

- `outputs/sft/<run-id>/sft_comparison.json`：带 schema 版本的 manifest，包含 CLI 配置、git/运行环境、数据与 artifact SHA-256、逐 seed/分支/样本结果、跨 seed 汇总、不变量和限制。
- 同目录 `sft_comparison.csv`：长格式逐 step 训练 loss 与各最终 loss，便于画曲线或导入表格。
- `outputs/sft/<run-id>/seed-<seed>/full/model.pt`：每个 seed 的预训练 Full SFT 完整模型权重与 metadata（相对于 adapter-only；不含 optimizer 状态）。
- `outputs/sft/<run-id>/seed-<seed>/lora/adapter.pt`：每个 seed 的 LoRA A/B 与 metadata，包含 base/data SHA-256、Tokenizer、config、targets、rank、alpha、dropout 和最终指标。

每个 seed 的 random 分支也保存到 `seed-<seed>/random/model.pt`。run 根目录携带独立 base 与语料副本、数据快照和 SHA-256；整个目录可搬迁。可选 `--full-checkpoint`、`--adapter-output` 仅额外导出首 seed，已有目标拒绝覆盖。非空 run 目录也拒绝复用。

### 6.1 严格恢复与独立推理

公共 loader 会在构造可用模型前校验 artifact 格式版本、config、Tokenizer、data/base SHA-256、state 键/形状/类型与有限值；LoRA 还会核对 targets、rank、alpha 和 dropout。任何一项不一致都会报错，不会静默 `strict=False` 或部分加载。

adapter 不是完整模型。实验 10 在保存后已经用严格 loader 独立恢复一次并要求 logits 完全一致；也可以直接让实验 07 在新进程中复现这条路径：

```powershell
.\.venv\Scripts\python.exe .\experiments\07_generate.py `
  --run-dir $SftRun --branch lora --seed 42 `
  --instruction "LoRA 为什么节省参数？" --tokens 64
```

`--base-checkpoint` 和 `--adapter` 必须成对传入，且不能与完整模型的 `--checkpoint` 同时使用。

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
- 使用版本 1 的旧 base：请重跑实验 06→10；新评测不要求修改 base，但字符覆盖与上下文上限仍须满足。
- 用 dev 之外的数据调参：test 只在训练完成后评一次，不能反复查看后再选择配置。
- 只看训练 loss：同时检查 dev/test 任务指标、原语料验证 loss、冻结参数和 adapter 往返。

## 11. 出门测

1. `d_in=d_out=4096,r=8` 时，完整权重与 LoRA 可训练参数各是多少？
2. 为什么 LoRA 可以减少训练成本，却不保证推理模型本体很小？
3. SFT 和 LoRA 能否同时使用？分别描述数据目标与参数集合。
4. 为什么低秩适配成功不等于解决灾难性遗忘、事实错误或安全问题？
5. 为什么 Full SFT 与 LoRA-SFT 必须各自从同一个 base checkpoint 分叉？

## 版本 2：失败记录和扩展评测

训练过程逐步保存历史，各分支及 seed 单独保存产物。`status` 区分 running/completed/failed/interrupted；`quality_status` 与逐项 `checks` 记录下降幅度、阈值、实际值和原因。效果不佳不再触发中途退出，所有 seed 继续完成；`--strict-checks` 在写完记录后返回 2。模型执行错误和冻结约束破坏属于失败，返回 1；用户中断返回 130。部分分支保留历史，但不会进入已完成分支的均值统计。`completed_seeds` 给出完整完成的 seed，单个指标的 `runs` 给出实际分母。

默认演示集仍为 4/4/4。`--eval-suite extended` 使用原有 4 条训练样本，新增 20 条 dev 和 40 条 test；测试专属意图不进入 dev。数据采用现有字符词表，模型知识没有因扩展评测而增加。每次报告分别记录数据 SHA-256 和评分实现 SHA-256。

```powershell
.\.venv\Scripts\python.exe .\experiments\10_sft_tiny_gpt.py --quick --eval-suite extended --seed 42
```

每条样本的 `required_concepts` 是概念组列表：组内表达任选一个，组间必须全部满足；未声明时沿用原关键词。`contradictions` 声明已知矛盾表达。任务成功要求概念完整、未命中矛盾、格式正确；原始关键词准确率仍独立报告。逐样本结果列出 `missing_concepts`、`matched_contradictions`、`format_failures` 与生成结束原因 `stop_reason`（eos/length）。

例如“验证集损失帮助发现过度拟合。”可通过同义规则，而“验证损失不能帮助发现过拟合。”会命中否定反例。规则只覆盖已声明的表达，不能识别任意语义矛盾，不能当作人工理解的替代。固定评分器反例位于 `data/evaluation_cases.jsonl`。

每次运行自动生成 PNG/SVG 原始 loss 曲线、可训练参数量与训练耗时图。耗时包含训练循环和记录开销，不能当作纯算子基准；不同 seed 汇总为均值与总体标准差。重新绘图无需模型：

```powershell
.\.venv\Scripts\python.exe .\scripts\plot_experiments.py $SftRun --output-dir .\outputs\plots\lesson
```
