"""实验 05：把 token 送入一个极小 Decoder-only Transformer。"""

from __future__ import annotations

import torch

from _common import seed_everything
from learn_llm.model import TinyGPT, TinyGPTConfig


def main() -> None:
    seed_everything()
    config = TinyGPTConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
    )
    model = TinyGPT(config).eval()
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    with torch.no_grad():
        logits, _ = model(token_ids)

        # 只改最后一个（未来）token；前五个位置的 logits 应完全不变。
        changed = token_ids.clone()
        changed[0, -1] = 7
        changed_logits, _ = model(changed)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    max_past_difference = (logits[:, :-1] - changed_logits[:, :-1]).abs().max().item()

    print("input shape: ", tuple(token_ids.shape), "= [B, T]")
    print("logits shape:", tuple(logits.shape), "= [B, T, vocab]")
    print("parameters:  ", f"{parameter_count:,}")
    print("未来 token 改变后，过去 logits 最大差异:", f"{max_past_difference:.3e}")

    assert logits.shape == (1, 6, config.vocab_size)
    assert max_past_difference < 1e-6
    print("PASS: 模型形状正确，并且没有未来信息泄漏。")


if __name__ == "__main__":
    main()
