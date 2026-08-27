# 示例结课报告：缩放如何影响 Attention 分布

## 研究问题

当 head dimension `d_k` 增大时，除以 `sqrt(d_k)` 是否能防止 Attention 概率分布越来越尖锐？

## 运行前假设

若 Q、K 各维近似独立且方差相近，未缩放点积的方差会随 `d_k` 增长。Softmax 将因此更容易饱和：平均熵下降，最大权重上升。缩放后，这两个指标应在不同 `d_k` 下相对稳定。

## 方法

- 配置见 [`config/experiment.json`](config/experiment.json)。
- 对每个 `d_k` 生成相同数量的随机 Q、K、V。
- 对照组直接对 `QK^T` 做 Softmax；实验组先除以 `sqrt(d_k)`。
- 指标：每行 Attention 的平均熵、每行最大权重的平均值。
- 因果 mask 不是本实验变量；脚本另用断言确认未来权重严格为 0。

## 复现

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe .\examples\capstone_attention\run.py
```

脚本会重写 [`results.csv`](results.csv)，并检查三个宽松趋势：未缩放熵明显下降、缩放后熵变化较小、高维时未缩放最大权重更高。

## 结果与结论

原始汇总值见 [`results.csv`](results.csv)。结果支持假设：未缩放 Attention 随 `d_k` 增大显著变尖，缩放后的熵和峰值稳定得多。因此 `sqrt(d_k)` 的作用不是改变 tensor shape，而是控制送入 Softmax 的数值尺度。

本结论只说明初始化分布下的数值机制，不证明真实模型训练后每个 head 都具有相同熵，也不说明 Attention 权重是完整的推理解释。失败案例与外推边界见 [`examples.md`](examples.md) 和 [`limitations.md`](limitations.md)。
