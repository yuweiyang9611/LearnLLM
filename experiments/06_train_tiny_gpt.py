"""实验 06：在本地小语料上预训练一个 CPU 版 TinyGPT。"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict

import torch

from _common import CHECKPOINT_DIR, DATA_DIR, PROJECT_ROOT, seed_everything
from learn_llm.model import TinyGPT, TinyGPTConfig
from learn_llm.tokenizer import CharTokenizer
from learn_llm.training import sample_language_model_batch, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200, help="优化步数")
    parser.add_argument("--quick", action="store_true", help="使用 40 步快速检查")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    token_ids: torch.Tensor,
    *,
    block_size: int,
    batches: int = 8,
) -> float:
    model.eval()
    losses = []
    generator = torch.Generator().manual_seed(2026)
    for _ in range(batches):
        x, y = sample_language_model_batch(
            token_ids,
            block_size=block_size,
            batch_size=8,
            generator=generator,
        )
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    steps = 40 if args.quick else args.steps
    seed_everything(args.seed)

    corpus = (DATA_DIR / "tiny_corpus.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(corpus)
    all_ids = tokenizer.encode(corpus, return_tensor=True)
    assert isinstance(all_ids, torch.Tensor)

    split = int(0.9 * len(all_ids))
    train_ids = all_ids[:split]
    validation_ids = all_ids[split:]
    block_size = min(48, len(validation_ids) - 1)
    config = TinyGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
    )
    model = TinyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    batch_generator = torch.Generator().manual_seed(args.seed)

    random_baseline = math.log(tokenizer.vocab_size)
    initial_train_loss = estimate_loss(model, train_ids, block_size=block_size)
    initial_validation_loss = estimate_loss(model, validation_ids, block_size=block_size)
    history: list[tuple[int, float, float]] = []
    started = time.perf_counter()

    print("=== TinyGPT 训练配置 ===")
    print(asdict(config))
    print(f"参数量: {model.parameter_count():,}")
    print(f"均匀随机理论 loss ln(V): {random_baseline:.4f}")
    print(f"初始 train/val loss: {initial_train_loss:.4f} / {initial_validation_loss:.4f}")

    model.train()
    report_every = max(1, steps // 5)
    for step in range(1, steps + 1):
        x, y = sample_language_model_batch(
            train_ids,
            block_size=block_size,
            batch_size=16,
            generator=batch_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % report_every == 0 or step == steps:
            train_loss = estimate_loss(model, train_ids, block_size=block_size, batches=4)
            val_loss = estimate_loss(model, validation_ids, block_size=block_size, batches=4)
            history.append((step, train_loss, val_loss))
            print(f"step {step:4d} | train {train_loss:.4f} | val {val_loss:.4f}")
            model.train()

    final_train_loss = history[-1][1]
    duration = time.perf_counter() - started
    checkpoint_path = CHECKPOINT_DIR / "tiny_gpt.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        step=steps,
        metadata={
            "config": asdict(config),
            "tokenizer": tokenizer.state_dict(),
            "corpus": "data/tiny_corpus.txt",
            "seed": args.seed,
        },
    )

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    metrics_path = output_dir / "tiny_gpt_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "train_loss", "validation_loss"])
        writer.writerows(history)

    print(f"耗时: {duration:.2f}s")
    print(f"checkpoint: {checkpoint_path}")
    print(f"metrics:    {metrics_path}")
    assert final_train_loss < initial_train_loss * 0.92, (
        "loss 下降不足；先检查数据右移、学习率和梯度，再增加训练步数"
    )
    print("PASS: loss 明显下降，checkpoint 与训练指标已保存。")


if __name__ == "__main__":
    main()

