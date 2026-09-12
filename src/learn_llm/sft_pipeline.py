"""Reusable model-side operations for the TinyGPT SFT comparison runner."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import torch

from .evaluation import (
    EvaluationExample,
    evaluate_predictions,
    summarize_by_category,
)
from .experiment_tracking import BranchResult, privacy_safe_path
from .model import TinyGPT, TinyGPTConfig
from .sft import (
    SFTBatch,
    SFTSequence,
    build_sft_sequence,
    collate_sft_batch,
    evaluate_assistant_loss,
    format_instruction_prompt,
    train_sft_steps,
)
from .tokenizer import CharTokenizer
from .training import sample_language_model_batch


InstructionPair = tuple[str, str]


def data_file_sha256(
    paths: Iterable[Path],
    *,
    project_root: Path,
) -> dict[str, str]:
    """Hash every input file using stable project-relative display paths."""

    result: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        name = privacy_safe_path(resolved, project_root=project_root)
        result[name] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def require_recorded_data(
    metadata: Mapping[str, object],
    current: Mapping[str, str],
) -> None:
    """Reject an old checkpoint when any tracked repository data changed."""

    recorded = metadata.get("data_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("base checkpoint does not contain data_sha256 metadata")
    mismatches = [
        path for path, digest in current.items() if recorded.get(path) != digest
    ]
    if mismatches:
        raise ValueError(
            "base checkpoint data fingerprints do not match: "
            + ", ".join(mismatches)
        )


def require_tokenizer_coverage(
    tokenizer: CharTokenizer,
    pairs: Sequence[InstructionPair],
    *,
    dataset_name: str,
) -> None:
    text = "".join(
        format_instruction_prompt(instruction) + response
        for instruction, response in pairs
    )
    missing = sorted(set(text) - set(tokenizer.stoi))
    if missing:
        raise ValueError(
            f"pretrained tokenizer does not cover {dataset_name}: "
            f"{''.join(missing)!r}"
        )


def build_instruction_batch(
    tokenizer: CharTokenizer,
    pairs: Sequence[InstructionPair],
    *,
    block_size: int,
    device: torch.device | str = "cpu",
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
    return sequences, collate_sft_batch(
        sequences,
        pad_token_id=0,
        device=device,
    )


def per_example_losses(
    model: TinyGPT,
    sequences: Sequence[SFTSequence],
    *,
    device: torch.device | str = "cpu",
) -> list[float]:
    return [
        evaluate_assistant_loss(
            model,
            collate_sft_batch([sequence], pad_token_id=0, device=device),
        )
        for sequence in sequences
    ]


@torch.no_grad()
def greedy_completion(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    instruction: str,
    *,
    max_new_tokens: int = 64,
    device: torch.device | str = "cpu",
    return_details: bool = False,
) -> str | tuple[str, str]:
    prompt_ids = tokenizer.encode(
        format_instruction_prompt(instruction),
        return_tensor=True,
    )
    assert isinstance(prompt_ids, torch.Tensor)
    prompt_ids = prompt_ids.to(device)
    generated = model.generate(
        prompt_ids.unsqueeze(0),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_id,
    )
    completion = generated[0, prompt_ids.numel():].cpu()
    text = tokenizer.decode(completion, skip_special_tokens=True)
    reason = "eos" if tokenizer.eos_id is not None and (completion == tokenizer.eos_id).any() else "length"
    return (text, reason) if return_details else text


@torch.no_grad()
def evaluate_corpus_validation_loss(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    corpus_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Measure retention on deterministic windows from the original corpus."""

    corpus = corpus_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(corpus, return_tensor=True)
    assert isinstance(token_ids, torch.Tensor)
    validation_ids = token_ids[int(0.9 * len(token_ids)) :]
    was_training = model.training
    model.eval()
    losses: list[float] = []
    generator = torch.Generator().manual_seed(2026)
    try:
        for _ in range(8):
            inputs, targets = sample_language_model_batch(
                validation_ids,
                block_size=model.config.block_size,
                batch_size=8,
                generator=generator,
                device=device,
            )
            _, loss = model(inputs, targets)
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
    before: Mapping[str, torch.Tensor],
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
    sequences: Sequence[SFTSequence] | None = None,
    batch_size: int | None = None,
    sampling_seed: int | None = None,
    device: torch.device | str = "cpu",
    on_step: Callable[[int, float, float], None] | None = None,
) -> tuple[list[float], float]:
    """Train on a fixed batch or a reproducible stochastic sequence stream."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    started = time.perf_counter()
    if sequences is None:
        history = train_sft_steps(
            model,
            batch,
            steps=steps,
            learning_rate=learning_rate,
            on_step=(lambda step, loss: on_step(step, loss, time.perf_counter() - started)) if on_step else None,
        )
    else:
        if not sequences:
            raise ValueError("sequences cannot be empty")
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be positive for stochastic training")
        if sampling_seed is None:
            raise ValueError("sampling_seed is required for stochastic training")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=0.0,
        )
        generator = torch.Generator().manual_seed(sampling_seed)
        history = []
        for step in range(steps):
            indices = torch.randint(
                len(sequences),
                (batch_size,),
                generator=generator,
            ).tolist()
            sampled = collate_sft_batch(
                [sequences[index] for index in indices],
                pad_token_id=0,
                device=device,
            )
            history.extend(
                train_sft_steps(
                    model,
                    sampled,
                    steps=1,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                )
            )
            if on_step is not None:
                on_step(step + 1, history[-1], time.perf_counter() - started)
    return history, time.perf_counter() - started


def _summary_metrics(summary: object) -> dict[str, float]:
    return {
        "task_success_rate": float(getattr(summary, "task_success_rate")),
        "keyword_accuracy": float(getattr(summary, "keyword_accuracy")),
        "format_accuracy": float(getattr(summary, "format_accuracy")),
    }


def evaluate_branch(
    *,
    name: str,
    seed: int,
    model: TinyGPT,
    tokenizer: CharTokenizer,
    examples_by_split: Mapping[str, Sequence[EvaluationExample]],
    sequences_by_split: Mapping[str, Sequence[SFTSequence]],
    batches_by_split: Mapping[str, SFTBatch],
    corpus_path: Path,
    learning_rate: float | None,
    history: list[float],
    duration_seconds: float,
    max_new_tokens: int,
    device: torch.device | str,
) -> BranchResult:
    """Collect teacher-forced and generated-answer metrics for one branch."""

    losses = {
        f"{split}_assistant_loss": evaluate_assistant_loss(model, batch)
        for split, batch in batches_by_split.items()
    }
    losses["corpus_validation_loss"] = evaluate_corpus_validation_loss(
        model,
        tokenizer,
        corpus_path,
        device=device,
    )
    per_example = {
        split: per_example_losses(model, sequences, device=device)
        for split, sequences in sequences_by_split.items()
    }

    generations: dict[str, list[dict[str, object]]] = {}
    task_metrics: dict[str, dict[str, float]] = {}
    for split, examples in examples_by_split.items():
        completions = {
            example.example_id: greedy_completion(
                model,
                tokenizer,
                example.instruction,
                max_new_tokens=max_new_tokens,
                device=device,
                return_details=True,
            )
            for example in examples
        }
        predictions = {key: value[0] for key, value in completions.items()}
        report = evaluate_predictions(examples, predictions)
        result_by_id = {result.example_id: result for result in report.results}
        generations[split] = [
            {
                "id": example.example_id,
                "intent_family": example.intent_family,
                "evaluation_category": example.evaluation_category,
                "instruction": example.instruction,
                "reference_response": example.reference_response,
                "prediction": predictions[example.example_id],
                "stop_reason": completions[example.example_id][1],
                "missing_concepts": result_by_id[example.example_id].missing_concepts,
                "matched_contradictions": result_by_id[example.example_id].matched_contradictions,
                "format_failures": result_by_id[example.example_id].format_failures,
                "required_keywords": list(example.required_keywords),
                "matched_keywords": list(
                    result_by_id[example.example_id].matched_keywords
                ),
                "task_success": result_by_id[example.example_id].task_success,
                "keyword_accuracy": result_by_id[
                    example.example_id
                ].keyword_accuracy,
                "format_accuracy": result_by_id[
                    example.example_id
                ].format_accuracy,
            }
            for example in examples
        ]
        task_metrics[split] = _summary_metrics(report.summary)
        for category, summary in summarize_by_category(report.results).items():
            task_metrics[f"{split}.{category}"] = _summary_metrics(summary)

    return BranchResult(
        name=name,
        seed=seed,
        learning_rate=learning_rate,
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
        duration_seconds=duration_seconds,
        history=history,
        losses=losses,
        per_example_losses=per_example,
        generations=generations,
        task_metrics=task_metrics,
    )
