"""实验 10：用 assistant-only 标签真正微调 TinyGPT 的全部参数。"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from _common import CHECKPOINT_DIR, DATA_DIR, seed_everything
from learn_llm.model import TinyGPT, TinyGPTConfig
from learn_llm.sft import (
    IGNORE_INDEX,
    SFTSequence,
    build_sft_sequence,
    collate_sft_batch,
    evaluate_assistant_loss,
    format_instruction_prompt,
    train_sft_steps,
)
from learn_llm.tokenizer import CharTokenizer
from learn_llm.training import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=180, help="SFT 优化步数")
    parser.add_argument("--quick", action="store_true", help="固定使用 30 步快速验收")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_instruction_pairs(path: Path) -> list[tuple[str, str]]:
    """读取小型 JSONL；字段错误会在训练前明确失败。"""

    pairs: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        instruction = record.get("instruction")
        response = record.get("response")
        if not isinstance(instruction, str) or not isinstance(response, str):
            raise ValueError(f"第 {line_number} 行缺少字符串 instruction/response")
        pairs.append((instruction, response))
    if not pairs:
        raise ValueError("指令数据集不能为空")
    return pairs


def per_example_losses(
    model: TinyGPT,
    sequences: list[SFTSequence],
    *,
    pad_token_id: int,
) -> list[float]:
    """逐条评估，帮助观察平均值是否掩盖了某一条样本。"""

    return [
        evaluate_assistant_loss(
            model, collate_sft_batch([sequence], pad_token_id=pad_token_id)
        )
        for sequence in sequences
    ]


@torch.no_grad()
def greedy_completion(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    instruction: str,
    *,
    max_new_tokens: int,
) -> str:
    """从 ``助手：`` 后开始 greedy 解码，仅用于直观对比。"""

    prompt_ids = tokenizer.encode(
        format_instruction_prompt(instruction), return_tensor=True
    )
    assert isinstance(prompt_ids, torch.Tensor)
    generated = model.generate(
        prompt_ids.unsqueeze(0),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(generated[0, prompt_ids.numel() :])


def parameter_l2_change(
    model: TinyGPT, before: dict[str, torch.Tensor]
) -> float:
    """量化参数更新，避免把“只算了 loss”误称为完成 SFT。"""

    squared_change = 0.0
    for name, parameter in model.named_parameters():
        difference = parameter.detach() - before[name]
        squared_change += float(torch.sum(difference * difference).item())
    return math.sqrt(squared_change)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps 必须为正数")
    steps = 30 if args.quick else args.steps
    seed_everything(args.seed)

    pairs = load_instruction_pairs(DATA_DIR / "tiny_instructions.jsonl")
    # 先建立覆盖所有 prompt/response 的教学字符词表。mask 边界稍后仍由
    # encode 后的 token ID 长度决定，而不是拿 Python 字符数代替 token 数。
    training_text = "".join(
        format_instruction_prompt(instruction) + response
        for instruction, response in pairs
    )
    tokenizer = CharTokenizer.from_text(training_text)
    sequences = [
        build_sft_sequence(tokenizer, instruction, response)
        for instruction, response in pairs
    ]
    pad_token_id = 0  # padding 标签恒为 -100，因此可复用任一合法词表 ID。
    batch = collate_sft_batch(sequences, pad_token_id=pad_token_id)

    config = TinyGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=batch.input_ids.shape[1],
        n_layer=1,
        n_head=4,
        n_embd=32,
        dropout=0.0,
    )
    model = TinyGPT(config)
    parameter_snapshot = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    initial_loss = evaluate_assistant_loss(model, batch)
    initial_per_example = per_example_losses(
        model, sequences, pad_token_id=pad_token_id
    )
    first_instruction, first_response = pairs[0]
    before_text = greedy_completion(
        model,
        tokenizer,
        first_instruction,
        max_new_tokens=min(len(first_response), 24),
    )

    print("=== TinyGPT assistant-only SFT ===")
    print(asdict(config))
    print(f"样本数 / batch 序列长度: {len(sequences)} / {batch.input_ids.shape[1]}")
    print(f"参与 loss 的答案 token 数: {batch.supervised_token_count}")
    print(f"被忽略的 prompt/padding 标签数: {(batch.labels == IGNORE_INDEX).sum().item()}")
    print(f"SFT 前平均 assistant loss: {initial_loss:.4f}")

    started = time.perf_counter()
    history = train_sft_steps(
        model,
        batch,
        steps=steps,
        learning_rate=1e-2,
    )
    duration = time.perf_counter() - started
    final_loss = evaluate_assistant_loss(model, batch)
    final_per_example = per_example_losses(
        model, sequences, pad_token_id=pad_token_id
    )
    after_text = greedy_completion(
        model,
        tokenizer,
        first_instruction,
        max_new_tokens=min(len(first_response), 24),
    )
    parameter_change = parameter_l2_change(model, parameter_snapshot)

    print(f"训练首步/末步 loss: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"SFT 后平均 assistant loss: {final_loss:.4f}")
    print("逐样本 assistant loss:")
    for index, (before, after) in enumerate(
        zip(initial_per_example, final_per_example, strict=True), 1
    ):
        print(f"  {index}: {before:.4f} -> {after:.4f}")
    print(f"模型参数 L2 变化: {parameter_change:.4f}")
    print(f"示例 greedy 输出（训练前）: {before_text!r}")
    print(f"示例 greedy 输出（训练后）: {after_text!r}")
    print(f"耗时: {duration:.2f}s")

    checkpoint_path = CHECKPOINT_DIR / "tiny_gpt_sft.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        step=steps,
        metadata={
            "config": asdict(config),
            "tokenizer": tokenizer.state_dict(),
            "dataset": "data/tiny_instructions.jsonl",
            "seed": args.seed,
            "objective": "assistant-only shifted next-token loss",
        },
    )
    print(f"checkpoint: {checkpoint_path}")

    assert final_loss < initial_loss * 0.30, (
        "assistant loss 下降不足；检查 shifted labels、学习率与随机种子"
    )
    assert all(
        after < before * 0.55
        for before, after in zip(initial_per_example, final_per_example, strict=True)
    ), "至少一条指令没有被明显学到"
    assert parameter_change > 1e-3, "参数没有发生足够变化，不能称为真实 SFT"
    print("PASS: TinyGPT 参数已更新，且每条指令的 assistant loss 都明显下降。")


if __name__ == "__main__":
    main()
