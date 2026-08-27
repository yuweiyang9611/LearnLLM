"""实验 01：从 logits、Softmax、交叉熵走到梯度下降。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from _common import seed_everything
from learn_llm.math_utils import cross_entropy_from_logits, stable_softmax


def main() -> None:
    seed_everything()

    # 极大的 logits 专门用于验证“先减最大值”的数值稳定技巧。
    logits = torch.tensor(
        [[1000.0, 1001.0, 999.0], [2.0, 0.0, -1.0]],
        requires_grad=True,
    )
    targets = torch.tensor([1, 0])

    probabilities = stable_softmax(logits, dim=-1)
    manual_loss = cross_entropy_from_logits(logits, targets)
    reference_loss = F.cross_entropy(logits, targets)
    manual_loss.backward()

    print("logits shape:", tuple(logits.shape), "= [batch, classes]")
    print("probabilities:\n", probabilities.detach())
    print("row sums:", probabilities.sum(dim=-1).detach())
    print(f"manual cross entropy:    {manual_loss.item():.8f}")
    print(f"PyTorch cross entropy:   {reference_loss.item():.8f}")
    print(f"absolute error:          {(manual_loss-reference_loss).abs().item():.3e}")
    print("gradient dL/dlogits:\n", logits.grad)

    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    assert torch.allclose(manual_loss, reference_loss, atol=1e-7)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print("PASS: Softmax 稳定，交叉熵和自动求导结果正确。")


if __name__ == "__main__":
    main()

