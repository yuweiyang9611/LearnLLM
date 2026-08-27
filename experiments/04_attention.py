"""实验 04：逐步观察缩放点积注意力和因果掩码。"""

from __future__ import annotations

import math

import torch

from _common import seed_everything
from learn_llm.attention import scaled_dot_product_attention


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


def main() -> None:
    seed_everything()
    # [B, H, T, D] = [批大小, 头数, 序列长度, 每头维度]
    q = torch.tensor([[[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]]])
    k = q.clone()
    v = torch.tensor([[[[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]]])

    output, weights = scaled_dot_product_attention(q, k, v, causal=False)
    causal_output, causal_weights = scaled_dot_product_attention(q, k, v, causal=True)

    raw_scores = q @ k.transpose(-2, -1)
    unscaled_weights = torch.softmax(raw_scores, dim=-1)

    print("Q/K/V shape:", tuple(q.shape), tuple(k.shape), tuple(v.shape))
    print("attention weights (no mask):\n", weights[0, 0])
    print("row sums:", weights.sum(dim=-1))
    print("attention output:\n", output[0, 0])
    print("causal weights:\n", causal_weights[0, 0])
    print(f"scaled mean entropy:   {entropy(weights).mean().item():.4f}")
    print(f"unscaled mean entropy: {entropy(unscaled_weights).mean().item():.4f}")
    print(f"scale used: sqrt(d_k) = {math.sqrt(q.size(-1)):.4f}")

    upper_triangle = torch.triu(causal_weights[0, 0], diagonal=1)
    assert output.shape == q.shape
    assert causal_output.shape == q.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    assert torch.count_nonzero(upper_triangle) == 0
    print("PASS: 权重逐行归一化，因果掩码完全阻断未来位置。")


if __name__ == "__main__":
    main()

