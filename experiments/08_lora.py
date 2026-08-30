"""实验 08：先观察 SFT loss mask，再手写并训练一个 LoRA 线性层。"""

from __future__ import annotations

import json

import torch
from torch import nn

from _common import DATA_DIR, seed_everything
from learn_llm.lora import LoRALinear
from learn_llm.sft import (
    IGNORE_INDEX,
    build_sft_sequence,
    format_instruction_prompt,
)
from learn_llm.tokenizer import CharTokenizer


def show_sft_mask() -> None:
    first_line = (DATA_DIR / "tiny_instructions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    instruction = record["instruction"]
    prompt = format_instruction_prompt(instruction)
    response = record["response"]
    tokenizer = CharTokenizer.from_text(prompt + response)
    sequence = build_sft_sequence(tokenizer, instruction, response)
    prompt_token_count = sequence.prompt_token_count
    first_assistant_label_index = next(
        index for index, label in enumerate(sequence.labels) if label != IGNORE_INDEX
    )

    print("=== SFT loss mask ===")
    print(prompt + response)
    print(f"prompt token 数: {prompt_token_count}")
    print(f"首个 assistant label 索引: {first_assistant_label_index}")
    print(f"answer token 数: {sequence.response_token_count}，loss 中计入")
    assert first_assistant_label_index == prompt_token_count - 1
    assert first_assistant_label_index == sequence.first_assistant_label_index
    assert sequence.labels[:first_assistant_label_index] == (
        IGNORE_INDEX,
    ) * first_assistant_label_index
    assert all(label != IGNORE_INDEX for label in sequence.labels[first_assistant_label_index:])


def main() -> None:
    seed_everything()
    show_sft_mask()

    print("\n=== LoRA 低秩适配 ===")
    base = nn.Linear(4, 2, bias=False)
    frozen_weight_before = base.weight.detach().clone()
    layer = LoRALinear.from_linear(base, rank=2, alpha=4.0)
    x = torch.randn(64, 4)
    target_weight = torch.tensor([[1.2, -0.5, 0.3, 0.7], [-0.8, 0.2, 1.1, -0.4]])
    target = x @ target_weight.T

    with torch.no_grad():
        initial_base_output = base(x)
        initial_lora_output = layer(x)
    initial_delta = (initial_base_output - initial_lora_output).abs().max().item()

    trainable = [parameter for parameter in layer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.05)
    loss_history = []
    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((layer(x) - target) ** 2)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    merged_weight = layer.merged_weight()
    merged_output = x @ merged_weight.T
    lora_output = layer(x)

    total_parameters = sum(parameter.numel() for parameter in layer.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in layer.parameters() if parameter.requires_grad)
    print(f"初始 LoRA 增量最大值: {initial_delta:.3e}（B 从 0 初始化）")
    print(f"loss: {loss_history[0]:.6f} -> {loss_history[-1]:.6f}")
    print(f"总参数 / 可训练参数: {total_parameters} / {trainable_parameters}")
    print("base.requires_grad:", layer.base.weight.requires_grad)
    print("A/B gradient present:", layer.lora_A.grad is not None, layer.lora_B.grad is not None)

    assert initial_delta < 1e-7
    assert torch.equal(layer.base.weight, frozen_weight_before)
    assert layer.lora_A.grad is not None and layer.lora_B.grad is not None
    assert loss_history[-1] < loss_history[0] * 0.05
    assert torch.allclose(merged_output, lora_output, atol=1e-5)
    print("PASS: 原权重冻结，低秩参数完成适配，合并前后输出一致。")


if __name__ == "__main__":
    main()
