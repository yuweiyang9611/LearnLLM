"""实验 06：在本地小语料上预训练一个 CPU 版 TinyGPT。"""

from __future__ import annotations

import argparse
import json
import math
import time
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from _common import DATA_DIR, PROJECT_ROOT, seed_everything
from learn_llm.experiment_tracking import BranchResult, privacy_safe_path, sha256_file
from learn_llm.run_management import RunRecorder, atomic_json, unique_run_dir
from learn_llm.model import TinyGPT, TinyGPTConfig
from learn_llm.sft import format_instruction_prompt
from learn_llm.tokenizer import CharTokenizer
from learn_llm.training import (
    load_checkpoint,
    sample_language_model_batch,
    save_checkpoint,
)


INSTRUCTION_DATA_FILENAMES = (
    "tiny_instructions.jsonl",
    "tiny_instructions_dev.jsonl",
    "tiny_instructions_test.jsonl",
    # Compatibility view containing dev + test for learners on older commands.
    "tiny_instructions_eval.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200, help="优化步数")
    parser.add_argument("--quick", action="store_true", help="使用 40 步快速检查")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "tiny_corpus.txt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict-checks", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    token_ids: torch.Tensor,
    *,
    block_size: int,
    batches: int = 8,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    generator = torch.Generator().manual_seed(2026)
    for _ in range(batches):
        x, y = sample_language_model_batch(
            token_ids,
            block_size=block_size,
            batch_size=8,
            generator=generator,
            device=next(model.parameters()).device,
        )
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


def required_instruction_block_size() -> int:
    """Return the longest shifted SFT sequence used after pretraining."""

    lengths: list[int] = []
    for filename in INSTRUCTION_DATA_FILENAMES:
        for line in (DATA_DIR / filename).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            instruction = record.get("instruction")
            response = record.get("response")
            if not isinstance(instruction, str) or not isinstance(response, str):
                raise ValueError(f"{filename} contains an invalid instruction row")
            lengths.append(len(format_instruction_prompt(instruction) + response))
    if not lengths:
        raise ValueError("instruction datasets cannot be empty")
    return max(lengths)


@torch.no_grad()
def verify_checkpoint_round_trip(
    checkpoint_path: Path,
    source_model: TinyGPT,
    probe: torch.Tensor,
) -> None:
    """Rebuild config/tokenizer from metadata and require identical logits."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    config_state = metadata.get("config")
    tokenizer_state = metadata.get("tokenizer")
    if not isinstance(config_state, dict) or not isinstance(tokenizer_state, dict):
        raise ValueError("checkpoint 缺少 config/tokenizer metadata")
    restored_config = TinyGPTConfig(**config_state)
    restored_tokenizer = CharTokenizer.from_state_dict(tokenizer_state)
    if restored_config.vocab_size != restored_tokenizer.vocab_size:
        raise ValueError("checkpoint config 与 tokenizer 词表大小不一致")

    restored_model = TinyGPT(restored_config).eval()
    diagnostics = load_checkpoint(checkpoint_path, restored_model, strict=True)
    if diagnostics["missing_keys"] or diagnostics["unexpected_keys"]:
        raise ValueError(f"checkpoint strict load 失败: {diagnostics}")
    source_model.eval()
    expected_logits, _ = source_model(probe)
    actual_logits, _ = restored_model(probe)
    if not torch.equal(expected_logits, actual_logits):
        raise AssertionError("checkpoint 往返后的 logits 不一致")
    if (
        restored_model.lm_head.weight.data_ptr()
        != restored_model.token_embedding.weight.data_ptr()
    ):
        raise AssertionError("checkpoint 往返后权重绑定已断开")


def run_training(args, recorder):
    steps = 40 if args.quick else args.steps
    if args.steps <= 0 or args.batch_size <= 0 or args.seed < 0:
        raise ValueError("steps/batch-size must be positive and seed non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive and finite")
    seed_everything(args.seed)
    corpus = args.corpus.read_text(encoding="utf-8")
    corpus_copy = recorder.directory / "corpus.txt"
    shutil.copyfile(args.corpus, corpus_copy)
    tokenizer = CharTokenizer.from_text(corpus, add_eos=True)
    all_ids = tokenizer.encode(corpus, return_tensor=True)
    split = int(.9 * len(all_ids))
    train_ids, validation_ids = all_ids[:split], all_ids[split:]
    block_size = args.block_size if args.block_size is not None else max(48, required_instruction_block_size())
    if min(len(train_ids), len(validation_ids)) <= block_size:
        raise ValueError("train/validation corpus must be longer than block-size")
    config = TinyGPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size,
                          n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = TinyGPT(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    recorder.manifest["config"].update({"model": asdict(config), "steps": steps})
    recorder.manifest["data_sha256"] = {"corpus.txt": sha256_file(corpus_copy)}
    recorder.artifact("corpus", corpus_copy)
    result = BranchResult(name="base", seed=args.seed, learning_rate=args.learning_rate,
                          trainable_parameters=model.parameter_count(), total_parameters=model.parameter_count(),
                          duration_seconds=0., status="running")
    recorder.branch(result)
    initial_loss = estimate_loss(model, train_ids, block_size=block_size)
    generator = torch.Generator().manual_seed(args.seed)
    started = time.perf_counter()
    report_every = max(1, steps // 5)
    for step in range(1, steps + 1):
        model.train()
        x, y = sample_language_model_batch(train_ids, block_size=block_size, batch_size=args.batch_size,
                                           generator=generator, device=args.device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("training loss is not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        recorder.step(result, step, float(loss.detach()), time.perf_counter() - started)
        if step == 1 or step % report_every == 0 or step == steps:
            train_loss = estimate_loss(model, train_ids, block_size=block_size, batches=4)
            val_loss = estimate_loss(model, validation_ids, block_size=block_size, batches=4)
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise RuntimeError("evaluation loss is not finite")
            result.validation_history.append({"step": step, "loss": val_loss})
            result.losses.update(train_loss=train_loss, validation_loss=val_loss)
            recorder.flush()
            print(f"step {step:4d} | train {train_loss:.4f} | val {val_loss:.4f}")
    path = recorder.directory / "base.pt"
    save_checkpoint(path, model, optimizer=optimizer, step=steps, metadata={
        "config": asdict(config), "tokenizer": tokenizer.state_dict(),
        "data_sha256": recorder.manifest["data_sha256"], "seed": args.seed, "corpus": "corpus.txt",
    })
    verify_checkpoint_round_trip(path, model.cpu(), validation_ids[:block_size].unsqueeze(0))
    recorder.artifact("base", path, seed=args.seed, branch="base", model_config=asdict(config))
    result.status = "completed"
    recorder.check("train_loss_decline", result.losses["train_loss"], initial_loss * .92)
    recorder.finish()
    if recorder.exit_code(args.strict_checks) == 0:
        root = PROJECT_ROOT / "outputs" / "pretrain"
        # Only default-root runs update the convenience pointer; custom runs are explicit.
        if recorder.directory.parent == root.resolve():
            atomic_json(root / "latest.json", {"run": recorder.directory.name})
    print(f"Run: {recorder.directory}")
    print(f'Inference: python experiments/07_generate.py --run-dir "{recorder.directory}"')
    return recorder.exit_code(args.strict_checks)


def main():
    args = parse_args()
    directory = args.output_dir or unique_run_dir(PROJECT_ROOT / "outputs" / "pretrain")
    config = {key: privacy_safe_path(value, project_root=PROJECT_ROOT) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    recorder = RunRecorder(directory, kind="pretrain", config=config, project_root=PROJECT_ROOT)
    try:
        return run_training(args, recorder)
    except (Exception, KeyboardInterrupt) as error:
        recorder.finish(error)
        print(f"{type(error).__name__}: {error}; records: {directory}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
