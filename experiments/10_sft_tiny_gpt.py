"""实验 10：从同一预训练 checkpoint 比较随机训练、Full SFT 与 LoRA-SFT。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from _common import CHECKPOINT_DIR, DATA_DIR, PROJECT_ROOT, seed_everything
from learn_llm.lora import (
    inject_lora,
    load_lora_adapter_state_dict,
    lora_adapter_state_dict,
    lora_parameter_names,
)
from learn_llm.model import TinyGPT, TinyGPTConfig
from learn_llm.sft import (
    IGNORE_INDEX,
    SFTBatch,
    SFTSequence,
    build_sft_sequence,
    collate_sft_batch,
    evaluate_assistant_loss,
    format_instruction_prompt,
    train_sft_steps,
)
from learn_llm.tokenizer import CharTokenizer
from learn_llm.training import (
    load_checkpoint,
    sample_language_model_batch,
    save_checkpoint,
)


RANDOM_LEARNING_RATE = 1e-2
FULL_LEARNING_RATE = 3e-3
LORA_LEARNING_RATE = 1e-2
DATA_FILENAMES = (
    "tiny_corpus.txt",
    "tiny_instructions.jsonl",
    "tiny_instructions_eval.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=180, help="每个分支的优化步数")
    parser.add_argument("--quick", action="store_true", help="每个分支固定训练 30 步")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "tiny_gpt.pt",
        help="实验 06 生成的预训练 checkpoint",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    return parser.parse_args()


def load_instruction_pairs(path: Path) -> list[tuple[str, str]]:
    """Read a small JSONL file and fail before training on malformed rows."""

    pairs: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        instruction = record.get("instruction")
        response = record.get("response")
        if not isinstance(instruction, str) or not isinstance(response, str):
            raise ValueError(f"{path.name}:{line_number} 缺少字符串 instruction/response")
        pairs.append((instruction, response))
    if not pairs:
        raise ValueError(f"指令数据集不能为空: {path}")
    return pairs


def require_tokenizer_coverage(
    tokenizer: CharTokenizer,
    pairs: list[tuple[str, str]],
    *,
    dataset_name: str,
) -> None:
    """Reject silent UNK mapping or token-ID changes during fine-tuning."""

    text = "".join(
        format_instruction_prompt(instruction) + response
        for instruction, response in pairs
    )
    missing = sorted(set(text) - set(tokenizer.stoi))
    if missing:
        raise ValueError(
            f"预训练 tokenizer 未覆盖 {dataset_name} 字符: {''.join(missing)!r}"
        )


def current_data_sha256() -> dict[str, str]:
    return {
        f"data/{filename}": hashlib.sha256(
            (DATA_DIR / filename).read_bytes()
        ).hexdigest()
        for filename in DATA_FILENAMES
    }


def require_matching_data_sha256(metadata: dict[str, object]) -> dict[str, str]:
    """Reject an ignored old base checkpoint paired with newer repository data."""

    recorded = metadata.get("data_sha256")
    current = current_data_sha256()
    if not isinstance(recorded, dict):
        raise ValueError(
            "base checkpoint 未记录数据 SHA-256；请重新运行实验 06"
        )
    mismatches = [
        path
        for path, digest in current.items()
        if recorded.get(path) != digest
    ]
    if mismatches:
        raise ValueError(
            "base checkpoint 与当前数据版本不匹配："
            f"{', '.join(mismatches)}；请重新运行实验 06"
        )
    return current


def build_instruction_batch(
    tokenizer: CharTokenizer,
    pairs: list[tuple[str, str]],
    *,
    block_size: int,
) -> tuple[list[SFTSequence], SFTBatch]:
    sequences = [
        build_sft_sequence(
            tokenizer,
            instruction,
            response,
            max_sequence_length=block_size,
        )
        for instruction, response in pairs
    ]
    return sequences, collate_sft_batch(sequences, pad_token_id=0)


def load_pretrained_base(
    checkpoint_path: Path,
) -> tuple[TinyGPT, TinyGPTConfig, CharTokenizer, dict[str, object]]:
    """Recreate the exact experiment-06 model and frozen tokenizer."""

    if not checkpoint_path.is_file():
        command = (
            r".\.venv\Scripts\python.exe "
            r".\experiments\06_train_tiny_gpt.py --quick"
        )
        raise FileNotFoundError(
            f"缺少预训练 checkpoint: {checkpoint_path}\n请先运行: {command}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("不支持的预训练 checkpoint 格式")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint 缺少 metadata")
    config_state = metadata.get("config")
    tokenizer_state = metadata.get("tokenizer")
    if not isinstance(config_state, dict) or not isinstance(tokenizer_state, dict):
        raise ValueError("checkpoint 缺少 config/tokenizer metadata")

    config = TinyGPTConfig(**config_state)
    tokenizer = CharTokenizer.from_state_dict(tokenizer_state)
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "checkpoint config.vocab_size 与 tokenizer.vocab_size 不一致"
        )
    model = TinyGPT(config)
    diagnostics = load_checkpoint(checkpoint_path, model, strict=True)
    if diagnostics["missing_keys"] or diagnostics["unexpected_keys"]:
        raise ValueError(f"checkpoint strict load 失败: {diagnostics}")
    if model.lm_head.weight.data_ptr() != model.token_embedding.weight.data_ptr():
        raise AssertionError("checkpoint 加载后 embedding/lm_head 权重绑定已断开")
    return model, config, tokenizer, metadata


def per_example_losses(
    model: TinyGPT,
    sequences: list[SFTSequence],
) -> list[float]:
    return [
        evaluate_assistant_loss(
            model,
            collate_sft_batch([sequence], pad_token_id=0),
        )
        for sequence in sequences
    ]


@torch.no_grad()
def greedy_completion(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    instruction: str,
    *,
    max_new_tokens: int = 24,
) -> str:
    prompt_ids = tokenizer.encode(
        format_instruction_prompt(instruction),
        return_tensor=True,
    )
    assert isinstance(prompt_ids, torch.Tensor)
    generated = model.generate(
        prompt_ids.unsqueeze(0),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(generated[0, prompt_ids.numel() :])


@torch.no_grad()
def evaluate_corpus_validation_loss(
    model: TinyGPT,
    tokenizer: CharTokenizer,
) -> float:
    """Measure a retention/interference proxy on fixed original-data windows."""

    corpus = (DATA_DIR / "tiny_corpus.txt").read_text(encoding="utf-8")
    token_ids = tokenizer.encode(corpus, return_tensor=True)
    assert isinstance(token_ids, torch.Tensor)
    validation_ids = token_ids[int(0.9 * len(token_ids)) :]
    was_training = model.training
    model.eval()
    losses: list[float] = []
    generator = torch.Generator().manual_seed(2026)
    try:
        for _ in range(8):
            x, y = sample_language_model_batch(
                validation_ids,
                block_size=model.config.block_size,
                batch_size=8,
                generator=generator,
            )
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.item()))
    finally:
        model.train(was_training)
    result = sum(losses) / len(losses)
    if not math.isfinite(result):
        raise RuntimeError("corpus validation loss is not finite")
    return result


def attention_lora_targets(config: TinyGPTConfig) -> tuple[str, ...]:
    return tuple(
        f"blocks.{layer_index}.attention.{name}"
        for layer_index in range(config.n_layer)
        for name in ("qkv", "output")
    )


def parameter_l2_change(
    model: TinyGPT,
    before: dict[str, torch.Tensor],
) -> float:
    squared_change = 0.0
    for name, parameter in model.named_parameters():
        difference = parameter.detach() - before[name]
        squared_change += float(torch.sum(difference * difference).item())
    return math.sqrt(squared_change)


def count_parameters(model: TinyGPT, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def train_branch(
    model: TinyGPT,
    batch: SFTBatch,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[list[float], float]:
    started = time.perf_counter()
    history = train_sft_steps(
        model,
        batch,
        steps=steps,
        learning_rate=learning_rate,
    )
    return history, time.perf_counter() - started


def relative_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps 必须为正数")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank/alpha 必须为正数")
    steps = 30 if args.quick else args.steps
    seed_everything(args.seed)
    base_checkpoint = args.base_checkpoint.resolve()

    base_model, config, tokenizer, base_metadata = load_pretrained_base(
        base_checkpoint
    )
    data_sha256 = require_matching_data_sha256(base_metadata)
    train_pairs = load_instruction_pairs(DATA_DIR / "tiny_instructions.jsonl")
    heldout_pairs = load_instruction_pairs(
        DATA_DIR / "tiny_instructions_eval.jsonl"
    )
    require_tokenizer_coverage(
        tokenizer,
        train_pairs,
        dataset_name="tiny_instructions.jsonl",
    )
    require_tokenizer_coverage(
        tokenizer,
        heldout_pairs,
        dataset_name="tiny_instructions_eval.jsonl",
    )
    train_sequences, train_batch = build_instruction_batch(
        tokenizer,
        train_pairs,
        block_size=config.block_size,
    )
    heldout_sequences, heldout_batch = build_instruction_batch(
        tokenizer,
        heldout_pairs,
        block_size=config.block_size,
    )

    full_model = copy.deepcopy(base_model)
    lora_model = copy.deepcopy(base_model)
    base_model.requires_grad_(False)
    seed_everything(args.seed + 1)
    random_model = TinyGPT(config)

    base_train_loss = evaluate_assistant_loss(base_model, train_batch)
    base_heldout_loss = evaluate_assistant_loss(base_model, heldout_batch)
    base_corpus_loss = evaluate_corpus_validation_loss(base_model, tokenizer)
    random_initial_loss = evaluate_assistant_loss(random_model, train_batch)

    full_snapshot = {
        name: parameter.detach().clone()
        for name, parameter in full_model.named_parameters()
    }
    full_history, full_duration = train_branch(
        full_model,
        train_batch,
        steps=steps,
        learning_rate=FULL_LEARNING_RATE,
    )
    random_history, random_duration = train_branch(
        random_model,
        train_batch,
        steps=steps,
        learning_rate=RANDOM_LEARNING_RATE,
    )

    lora_targets = attention_lora_targets(config)
    seed_everything(args.seed + 2)
    inject_lora(
        lora_model,
        lora_targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    trainable_names = lora_parameter_names(lora_model)
    if not trainable_names or any(
        not (name.endswith(".lora_A") or name.endswith(".lora_B"))
        for name in trainable_names
    ):
        raise AssertionError(f"LoRA 可训练参数集合错误: {trainable_names}")
    frozen_snapshot = {
        name: parameter.detach().clone()
        for name, parameter in lora_model.named_parameters()
        if not parameter.requires_grad
    }
    adapter_before = lora_adapter_state_dict(lora_model)
    with torch.no_grad():
        base_logits, _ = base_model(train_batch.input_ids)
        initial_lora_logits, _ = lora_model(train_batch.input_ids)
    initial_lora_delta = float(
        (base_logits - initial_lora_logits).abs().max().item()
    )
    if initial_lora_delta > 1e-7:
        raise AssertionError(
            f"LoRA 注入改变了初始输出: max delta={initial_lora_delta:.3e}"
        )
    lora_history, lora_duration = train_branch(
        lora_model,
        train_batch,
        steps=steps,
        learning_rate=LORA_LEARNING_RATE,
    )

    random_train_loss = evaluate_assistant_loss(random_model, train_batch)
    random_heldout_loss = evaluate_assistant_loss(random_model, heldout_batch)
    random_corpus_loss = evaluate_corpus_validation_loss(random_model, tokenizer)
    full_train_loss = evaluate_assistant_loss(full_model, train_batch)
    full_heldout_loss = evaluate_assistant_loss(full_model, heldout_batch)
    full_corpus_loss = evaluate_corpus_validation_loss(full_model, tokenizer)
    lora_train_loss = evaluate_assistant_loss(lora_model, train_batch)
    lora_heldout_loss = evaluate_assistant_loss(lora_model, heldout_batch)
    lora_corpus_loss = evaluate_corpus_validation_loss(lora_model, tokenizer)
    per_example = {
        "base": per_example_losses(base_model, train_sequences),
        "random": per_example_losses(random_model, train_sequences),
        "full": per_example_losses(full_model, train_sequences),
        "lora": per_example_losses(lora_model, train_sequences),
    }
    heldout_per_example = {
        "base": per_example_losses(base_model, heldout_sequences),
        "random": per_example_losses(random_model, heldout_sequences),
        "full": per_example_losses(full_model, heldout_sequences),
        "lora": per_example_losses(lora_model, heldout_sequences),
    }

    changed_frozen = [
        name
        for name, parameter in lora_model.named_parameters()
        if not parameter.requires_grad
        and not torch.equal(parameter.detach(), frozen_snapshot[name])
    ]
    if changed_frozen:
        raise AssertionError(f"LoRA 训练修改了冻结参数: {changed_frozen}")
    adapter_after = lora_adapter_state_dict(lora_model)
    if not all(
        not torch.equal(adapter_before[name], adapter_after[name])
        for name in adapter_before
    ):
        raise AssertionError("至少一个 LoRA A/B 参数没有更新")

    print("=== 真实 TinyGPT 指令微调对比 ===")
    print(f"base checkpoint: {relative_project_path(base_checkpoint)}")
    print(f"配置: {asdict(config)}")
    print(
        f"训练/留出样本: {len(train_pairs)} / {len(heldout_pairs)}；"
        f"最大序列: {train_batch.input_ids.shape[1]} / {config.block_size}"
    )
    print(
        f"assistant token / ignored token: {train_batch.supervised_token_count} / "
        f"{int((train_batch.labels == IGNORE_INDEX).sum().item())}"
    )
    print(f"LoRA targets: {', '.join(lora_targets)}")
    print(f"LoRA 初始输出最大差: {initial_lora_delta:.3e}")
    print(
        "学习率（各机制独立调节，非严格单变量对照）: "
        f"random={RANDOM_LEARNING_RATE:g}, full={FULL_LEARNING_RATE:g}, "
        f"LoRA={LORA_LEARNING_RATE:g}"
    )
    print()
    print("分支                 train loss   heldout loss   corpus val   trainable/total   秒")
    rows = (
        (
            "pretrained base",
            base_train_loss,
            base_heldout_loss,
            base_corpus_loss,
            count_parameters(base_model, trainable_only=True),
            count_parameters(base_model),
            0.0,
        ),
        (
            "random-init SFT",
            random_train_loss,
            random_heldout_loss,
            random_corpus_loss,
            count_parameters(random_model, trainable_only=True),
            count_parameters(random_model),
            random_duration,
        ),
        (
            "pretrained Full SFT",
            full_train_loss,
            full_heldout_loss,
            full_corpus_loss,
            count_parameters(full_model, trainable_only=True),
            count_parameters(full_model),
            full_duration,
        ),
        (
            "pretrained LoRA-SFT",
            lora_train_loss,
            lora_heldout_loss,
            lora_corpus_loss,
            count_parameters(lora_model, trainable_only=True),
            count_parameters(lora_model),
            lora_duration,
        ),
    )
    for name, train_loss, heldout_loss, corpus_loss, trainable, total, duration in rows:
        print(
            f"{name:<22} {train_loss:>10.4f} {heldout_loss:>14.4f} "
            f"{corpus_loss:>12.4f} {trainable:>8,}/{total:<8,} {duration:>6.2f}"
        )
    print()
    print(
        f"训练首/末步 loss | random {random_history[0]:.4f}->{random_history[-1]:.4f} "
        f"| full {full_history[0]:.4f}->{full_history[-1]:.4f} "
        f"| LoRA {lora_history[0]:.4f}->{lora_history[-1]:.4f}"
    )
    print("训练集逐样本 assistant loss (base -> random / full / LoRA):")
    for index, (instruction, _) in enumerate(train_pairs):
        print(
            f"  {instruction!r}: {per_example['base'][index]:.4f} -> "
            f"{per_example['random'][index]:.4f} / "
            f"{per_example['full'][index]:.4f} / "
            f"{per_example['lora'][index]:.4f}"
        )
    print("留出集逐样本 assistant loss (base -> random / full / LoRA):")
    for index, (instruction, _) in enumerate(heldout_pairs):
        print(
            f"  {instruction!r}: {heldout_per_example['base'][index]:.4f} -> "
            f"{heldout_per_example['random'][index]:.4f} / "
            f"{heldout_per_example['full'][index]:.4f} / "
            f"{heldout_per_example['lora'][index]:.4f}"
        )
    print(f"Full SFT 参数 L2 变化: {parameter_l2_change(full_model, full_snapshot):.4f}")
    print(
        f"原语料干扰代理 Δ corpus val | Full "
        f"{full_corpus_loss - base_corpus_loss:+.4f} "
        f"| LoRA {lora_corpus_loss - base_corpus_loss:+.4f}"
    )
    seen_instruction = train_pairs[0][0]
    heldout_instruction = heldout_pairs[0][0]
    for name, model in (
        ("base", base_model),
        ("random", random_model),
        ("full", full_model),
        ("lora", lora_model),
    ):
        print(
            f"{name:>6} seen={greedy_completion(model, tokenizer, seen_instruction)!r} "
            f"heldout={greedy_completion(model, tokenizer, heldout_instruction)!r}"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    full_checkpoint_path = CHECKPOINT_DIR / "tiny_gpt_sft.pt"
    base_sha256 = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    save_checkpoint(
        full_checkpoint_path,
        full_model,
        step=steps,
        metadata={
            "config": asdict(config),
            "tokenizer": tokenizer.state_dict(),
            "base_checkpoint": relative_project_path(base_checkpoint),
            "base_sha256": base_sha256,
            "train_dataset": "data/tiny_instructions.jsonl",
            "heldout_dataset": "data/tiny_instructions_eval.jsonl",
            "data_sha256": data_sha256,
            "seed": args.seed,
            "objective": "pretrained full-parameter assistant-only SFT",
            "training": {
                "steps": steps,
                "learning_rate": FULL_LEARNING_RATE,
                "optimizer": "AdamW",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            },
            "metrics": {
                "train_assistant_loss": full_train_loss,
                "heldout_assistant_loss": full_heldout_loss,
                "corpus_validation_loss": full_corpus_loss,
            },
        },
    )

    adapter_path = CHECKPOINT_DIR / "tiny_gpt_lora_adapter.pt"
    adapter_payload = {
        "format_version": 1,
        "adapter_state": adapter_after,
        "metadata": {
            "config": asdict(config),
            "tokenizer": tokenizer.state_dict(),
            "base_checkpoint": relative_project_path(base_checkpoint),
            "base_sha256": base_sha256,
            "targets": list(lora_targets),
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": 0.0,
            "train_dataset": "data/tiny_instructions.jsonl",
            "heldout_dataset": "data/tiny_instructions_eval.jsonl",
            "data_sha256": data_sha256,
            "seed": args.seed,
            "objective": "pretrained LoRA-only assistant SFT",
            "training": {
                "steps": steps,
                "learning_rate": LORA_LEARNING_RATE,
                "optimizer": "AdamW",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            },
            "metrics": {
                "train_assistant_loss": lora_train_loss,
                "heldout_assistant_loss": lora_heldout_loss,
                "corpus_validation_loss": lora_corpus_loss,
            },
        },
    }
    torch.save(adapter_payload, adapter_path)

    restored_lora = copy.deepcopy(base_model)
    inject_lora(
        restored_lora,
        lora_targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    restored_payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    restored_metadata = restored_payload.get("metadata", {})
    if restored_metadata.get("base_sha256") != base_sha256:
        raise ValueError("LoRA adapter 指向的 base checkpoint SHA-256 不匹配")
    load_lora_adapter_state_dict(
        restored_lora,
        restored_payload["adapter_state"],
    )
    with torch.no_grad():
        expected_logits, _ = lora_model(heldout_batch.input_ids)
        restored_logits, _ = restored_lora(heldout_batch.input_ids)
    if not torch.equal(expected_logits, restored_logits):
        raise AssertionError("adapter-only 保存/恢复后的 logits 不一致")

    if random_train_loss >= random_initial_loss * 0.80:
        raise AssertionError("随机初始化监督训练 loss 下降不足")
    if full_train_loss >= base_train_loss * 0.45:
        raise AssertionError("预训练 Full SFT loss 下降不足")
    if lora_train_loss >= base_train_loss * 0.75:
        raise AssertionError("预训练 LoRA-SFT loss 下降不足")
    if parameter_l2_change(full_model, full_snapshot) <= 1e-3:
        raise AssertionError("Full SFT 没有真正更新预训练参数")
    if any(
        after >= before
        for before, after in zip(per_example["base"], per_example["full"])
    ):
        raise AssertionError("Full SFT 并未改善每一条训练样本")
    if any(
        after >= before
        for before, after in zip(per_example["base"], per_example["lora"])
    ):
        raise AssertionError("LoRA-SFT 并未改善每一条训练样本")

    print(f"Full checkpoint: {relative_project_path(full_checkpoint_path)}")
    print(f"LoRA adapter:    {relative_project_path(adapter_path)}")
    print(
        "PASS: 三个训练分支共享 tokenizer/config；预训练 Full 与 LoRA "
        "从同一 base 独立分叉，且 adapter-only 往返一致。"
    )


if __name__ == "__main__":
    main()
