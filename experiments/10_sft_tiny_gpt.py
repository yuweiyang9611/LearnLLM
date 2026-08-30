"""实验 10：可信地比较随机训练、Full SFT 与 LoRA-SFT。"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import CHECKPOINT_DIR, DATA_DIR, PROJECT_ROOT, seed_everything
from learn_llm.artifacts import (
    LORA_ADAPTER_FORMAT_VERSION,
    TinyGPTBundle,
    load_lora_model,
    load_tiny_gpt_checkpoint,
)
from learn_llm.evaluation import EvaluationExample, load_evaluation_jsonl
from learn_llm.experiment_tracking import (
    BranchResult,
    SFTExperimentConfig,
    build_run_manifest,
    privacy_safe_path,
    sha256_file,
    write_run_artifacts,
)
from learn_llm.lora import (
    inject_lora,
    lora_adapter_state_dict,
    lora_parameter_names,
)
from learn_llm.model import TinyGPT
from learn_llm.sft import evaluate_assistant_loss
from learn_llm.sft_pipeline import (
    attention_lora_targets,
    build_instruction_batch,
    data_file_sha256,
    evaluate_branch,
    parameter_l2_change,
    require_tokenizer_coverage,
    train_branch,
)
from learn_llm.training import save_checkpoint


DEFAULT_SEEDS = (42, 43, 44)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=180, help="每个训练分支的优化步数")
    parser.add_argument("--quick", action="store_true", help="每个分支固定训练 30 步")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, help="兼容旧命令：只运行一个 seed")
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="用于均值/标准差汇总的 seed；默认 42 43 44",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "tiny_gpt.pt",
        help="实验 06 生成的预训练 checkpoint",
    )
    parser.add_argument(
        "--train-data", type=Path, default=DATA_DIR / "tiny_instructions.jsonl"
    )
    parser.add_argument(
        "--dev-data", type=Path, default=DATA_DIR / "tiny_instructions_dev.jsonl"
    )
    parser.add_argument(
        "--test-data", type=Path, default=DATA_DIR / "tiny_instructions_test.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="运行结果目录；默认在 outputs/sft 下创建唯一目录",
    )
    parser.add_argument(
        "--full-checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "tiny_gpt_sft.pt",
    )
    parser.add_argument(
        "--adapter-output",
        type=Path,
        default=CHECKPOINT_DIR / "tiny_gpt_lora_adapter.pt",
    )
    parser.add_argument("--random-learning-rate", type=float, default=1e-2)
    parser.add_argument("--full-learning-rate", type=float, default=3e-3)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-2)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 cuda:N")
    return parser.parse_args(argv)


def _default_output_dir(seeds: tuple[int, ...]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    seed_label = "-".join(str(seed) for seed in seeds)
    return PROJECT_ROOT / "outputs" / "sft" / f"{timestamp}-seeds-{seed_label}"


def build_config(args: argparse.Namespace) -> SFTExperimentConfig:
    seeds = (
        tuple(args.seeds)
        if args.seeds is not None
        else (args.seed,)
        if args.seed is not None
        else DEFAULT_SEEDS
    )
    config = SFTExperimentConfig(
        base_checkpoint=args.base_checkpoint.resolve(),
        train_data=args.train_data.resolve(),
        dev_data=args.dev_data.resolve(),
        test_data=args.test_data.resolve(),
        output_dir=(args.output_dir or _default_output_dir(seeds)).resolve(),
        full_checkpoint=args.full_checkpoint.resolve(),
        adapter_output=args.adapter_output.resolve(),
        seeds=seeds,
        steps=30 if args.quick else args.steps,
        train_batch_size=args.train_batch_size,
        random_learning_rate=args.random_learning_rate,
        full_learning_rate=args.full_learning_rate,
        lora_learning_rate=args.lora_learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    config.validate()
    return config


def load_split(path: Path, expected_split: str) -> list[EvaluationExample]:
    examples = list(load_evaluation_jsonl(path))
    wrong = [example.example_id for example in examples if example.split != expected_split]
    if wrong:
        raise ValueError(
            f"{path.name} contains rows outside split {expected_split!r}: {wrong}"
        )
    return examples


def _pairs(examples: list[EvaluationExample]) -> list[tuple[str, str]]:
    return [(item.instruction, item.reference_response) for item in examples]


def _project_path(path: Path) -> str:
    return privacy_safe_path(path, project_root=PROJECT_ROOT)


def _verify_recorded_repository_data(bundle: TinyGPTBundle) -> None:
    current: dict[str, str] = {}
    for name in bundle.data_sha256:
        path = PROJECT_ROOT / name
        if name.startswith("external/") or not path.is_file():
            raise ValueError(f"cannot verify base checkpoint data file: {name}")
        current[name] = sha256_file(path)
    if current != bundle.data_sha256:
        changed = sorted(
            name
            for name in set(current) | set(bundle.data_sha256)
            if current.get(name) != bundle.data_sha256.get(name)
        )
        raise ValueError(
            "base checkpoint does not match current repository data: "
            + ", ".join(changed)
            + "; rerun experiment 06"
        )


def _checkpoint_metadata(
    *,
    bundle: TinyGPTBundle,
    config: SFTExperimentConfig,
    result: BranchResult,
    objective: str,
    run_data_sha256: dict[str, str],
) -> dict[str, Any]:
    return {
        "config": asdict(bundle.config),
        "tokenizer": bundle.tokenizer.state_dict(),
        "base_checkpoint": _project_path(bundle.checkpoint_path),
        "base_sha256": bundle.checkpoint_sha256,
        "data_sha256": bundle.data_sha256,
        "training_data_sha256": run_data_sha256,
        "train_dataset": _project_path(config.train_data),
        "dev_dataset": _project_path(config.dev_data),
        "test_dataset": _project_path(config.test_data),
        "run_output": _project_path(config.output_dir),
        "seed": result.seed,
        "objective": objective,
        "training": {
            "steps": config.steps,
            "learning_rate": result.learning_rate,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
        "metrics": {**result.losses, "task": result.task_metrics},
    }


def _save_primary_artifacts(
    *,
    bundle: TinyGPTBundle,
    config: SFTExperimentConfig,
    full_model: TinyGPT,
    lora_model: TinyGPT,
    full_result: BranchResult,
    lora_result: BranchResult,
    lora_targets: tuple[str, ...],
    run_data_sha256: dict[str, str],
    test_input_ids: torch.Tensor,
) -> None:
    save_checkpoint(
        config.full_checkpoint,
        full_model,
        step=config.steps,
        metadata=_checkpoint_metadata(
            bundle=bundle,
            config=config,
            result=full_result,
            objective="pretrained full-parameter assistant-only SFT",
            run_data_sha256=run_data_sha256,
        ),
    )

    adapter_metadata = _checkpoint_metadata(
        bundle=bundle,
        config=config,
        result=lora_result,
        objective="pretrained LoRA-only assistant SFT",
        run_data_sha256=run_data_sha256,
    )
    adapter_metadata.update(
        {
            "targets": list(lora_targets),
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
        }
    )
    config.adapter_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": LORA_ADAPTER_FORMAT_VERSION,
            "adapter_state": lora_adapter_state_dict(lora_model),
            "metadata": adapter_metadata,
        },
        config.adapter_output,
    )

    restored = load_lora_model(
        config.base_checkpoint,
        config.adapter_output,
        map_location="cpu",
        expected_data_sha256=bundle.data_sha256,
    )
    source = copy.deepcopy(lora_model).to("cpu").eval()
    with torch.no_grad():
        expected_logits, _ = source(test_input_ids.cpu())
        actual_logits, _ = restored.model(test_input_ids.cpu())
    if not torch.equal(expected_logits, actual_logits):
        raise AssertionError("independent LoRA reload did not reproduce logits")


def _print_summary(
    config: SFTExperimentConfig,
    seed_runs: list[dict[str, BranchResult]],
    manifest: dict[str, Any],
) -> None:
    print("=== TinyGPT train/dev/test SFT comparison ===")
    print(f"seeds: {', '.join(map(str, config.seeds))}; steps: {config.steps}")
    print("seed branch       train loss  dev loss  test loss  test success  seconds")
    for branches in seed_runs:
        for name in ("base", "random", "full", "lora"):
            result = branches[name]
            print(
                f"{result.seed:>4} {name:<10} "
                f"{result.losses['train_assistant_loss']:>10.4f} "
                f"{result.losses['dev_assistant_loss']:>9.4f} "
                f"{result.losses['test_assistant_loss']:>10.4f} "
                f"{result.task_metrics['test']['task_success_rate']:>13.1%} "
                f"{result.duration_seconds:>8.2f}"
            )
    print("\n跨 seed 汇总（mean ± population std）:")
    for name in ("random", "full", "lora"):
        metrics = manifest["aggregate"][name]
        loss = metrics["test_assistant_loss"]
        success = metrics["test.task_success_rate"]
        print(
            f"  {name:<6} test loss {loss['mean']:.4f} ± {loss['std']:.4f}; "
            f"task success {success['mean']:.1%} ± {success['std']:.1%}"
        )


def main() -> None:
    config = build_config(parse_args())
    examples_by_split = {
        "train": load_split(config.train_data, "train"),
        "dev": load_split(config.dev_data, "dev"),
        "test": load_split(config.test_data, "test"),
    }
    all_ids = [item.example_id for rows in examples_by_split.values() for item in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("example IDs must be unique across train/dev/test")

    bundle = load_tiny_gpt_checkpoint(
        config.base_checkpoint,
        map_location=config.device,
    )
    _verify_recorded_repository_data(bundle)
    pairs_by_split = {name: _pairs(rows) for name, rows in examples_by_split.items()}
    for split, pairs in pairs_by_split.items():
        require_tokenizer_coverage(
            bundle.tokenizer,
            pairs,
            dataset_name=_project_path(getattr(config, f"{split}_data")),
        )
    prepared = {
        split: build_instruction_batch(
            bundle.tokenizer,
            pairs,
            block_size=bundle.config.block_size,
            device=config.device,
        )
        for split, pairs in pairs_by_split.items()
    }
    sequences_by_split = {name: value[0] for name, value in prepared.items()}
    batches_by_split = {name: value[1] for name, value in prepared.items()}
    run_data_sha256 = data_file_sha256(
        (
            DATA_DIR / "tiny_corpus.txt",
            config.train_data,
            config.dev_data,
            config.test_data,
        ),
        project_root=PROJECT_ROOT,
    )

    seed_runs: list[dict[str, BranchResult]] = []
    invariant_runs: list[dict[str, Any]] = []
    primary_saved = False
    for seed in config.seeds:
        seed_everything(seed)
        base_model = copy.deepcopy(bundle.model).eval()
        base_model.requires_grad_(False)
        full_model = copy.deepcopy(bundle.model)
        full_model.requires_grad_(True)
        lora_model = copy.deepcopy(bundle.model)
        seed_everything(seed + 1)
        random_model = TinyGPT(bundle.config).to(config.device)
        train_batch = batches_by_split["train"]
        base_train_loss = evaluate_assistant_loss(base_model, train_batch)
        random_initial_loss = evaluate_assistant_loss(random_model, train_batch)

        full_snapshot = {
            name: parameter.detach().clone()
            for name, parameter in full_model.named_parameters()
        }
        full_history, full_duration = train_branch(
            full_model,
            train_batch,
            steps=config.steps,
            learning_rate=config.full_learning_rate,
            sequences=sequences_by_split["train"],
            batch_size=config.train_batch_size,
            sampling_seed=seed + 100,
            device=config.device,
        )
        random_history, random_duration = train_branch(
            random_model,
            train_batch,
            steps=config.steps,
            learning_rate=config.random_learning_rate,
            sequences=sequences_by_split["train"],
            batch_size=config.train_batch_size,
            sampling_seed=seed + 100,
            device=config.device,
        )

        lora_targets = attention_lora_targets(bundle.config)
        seed_everything(seed + 2)
        inject_lora(
            lora_model,
            lora_targets,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        trainable_names = lora_parameter_names(lora_model)
        if not trainable_names or any(
            not (name.endswith(".lora_A") or name.endswith(".lora_B"))
            for name in trainable_names
        ):
            raise AssertionError(f"invalid LoRA trainable set: {trainable_names}")
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
                f"LoRA injection changed initial logits: {initial_lora_delta:.3e}"
            )
        lora_history, lora_duration = train_branch(
            lora_model,
            train_batch,
            steps=config.steps,
            learning_rate=config.lora_learning_rate,
            sequences=sequences_by_split["train"],
            batch_size=config.train_batch_size,
            sampling_seed=seed + 100,
            device=config.device,
        )
        adapter_after = lora_adapter_state_dict(lora_model)
        changed_frozen = [
            name
            for name, parameter in lora_model.named_parameters()
            if not parameter.requires_grad
            and not torch.equal(parameter.detach(), frozen_snapshot[name])
        ]
        if changed_frozen:
            raise AssertionError(f"LoRA training changed frozen parameters: {changed_frozen}")
        if not any(
            not torch.equal(adapter_before[name], adapter_after[name])
            for name in adapter_before
        ):
            raise AssertionError("LoRA optimizer did not update adapter parameters")

        common_evaluation = {
            "tokenizer": bundle.tokenizer,
            "examples_by_split": examples_by_split,
            "sequences_by_split": sequences_by_split,
            "batches_by_split": batches_by_split,
            "corpus_path": DATA_DIR / "tiny_corpus.txt",
            "max_new_tokens": config.max_new_tokens,
            "device": config.device,
        }
        branches = {
            "base": evaluate_branch(
                name="base", seed=seed, model=base_model, learning_rate=None,
                history=[], duration_seconds=0.0, **common_evaluation,
            ),
            "random": evaluate_branch(
                name="random", seed=seed, model=random_model,
                learning_rate=config.random_learning_rate,
                history=random_history, duration_seconds=random_duration,
                **common_evaluation,
            ),
            "full": evaluate_branch(
                name="full", seed=seed, model=full_model,
                learning_rate=config.full_learning_rate,
                history=full_history, duration_seconds=full_duration,
                **common_evaluation,
            ),
            "lora": evaluate_branch(
                name="lora", seed=seed, model=lora_model,
                learning_rate=config.lora_learning_rate,
                history=lora_history, duration_seconds=lora_duration,
                **common_evaluation,
            ),
        }
        full_change = parameter_l2_change(full_model, full_snapshot)
        if branches["random"].losses["train_assistant_loss"] >= random_initial_loss * 0.85:
            raise AssertionError("random-init SFT loss did not decline enough")
        if branches["full"].losses["train_assistant_loss"] >= base_train_loss * 0.55:
            raise AssertionError("Full SFT loss did not decline enough")
        if branches["lora"].losses["train_assistant_loss"] >= base_train_loss * 0.90:
            raise AssertionError("LoRA-SFT loss did not decline enough")
        if full_change <= 1e-3:
            raise AssertionError("Full SFT did not update pretrained parameters")
        for branch_name in ("full", "lora"):
            if any(
                after >= before
                for before, after in zip(
                    branches["base"].per_example_losses["train"],
                    branches[branch_name].per_example_losses["train"],
                )
            ):
                raise AssertionError(
                    f"{branch_name} SFT failed to improve every training example"
                )

        invariant_runs.append(
            {
                "seed": seed,
                "lora_initial_max_logit_delta": initial_lora_delta,
                "lora_frozen_parameters_unchanged": True,
                "full_parameter_l2_change": full_change,
                "test_evaluated_after_training_only": True,
            }
        )
        if not primary_saved:
            _save_primary_artifacts(
                bundle=bundle,
                config=config,
                full_model=full_model,
                lora_model=lora_model,
                full_result=branches["full"],
                lora_result=branches["lora"],
                lora_targets=lora_targets,
                run_data_sha256=run_data_sha256,
                test_input_ids=batches_by_split["test"].input_ids,
            )
            invariant_runs[-1]["independent_lora_reload_exact"] = True
            primary_saved = True
        seed_runs.append(branches)

    manifest = build_run_manifest(
        project_root=PROJECT_ROOT,
        config=config,
        data_sha256=run_data_sha256,
        base_sha256=bundle.checkpoint_sha256,
        seed_runs=seed_runs,
        invariants={"per_seed": invariant_runs},
    )
    manifest["artifacts"].update(
        {
            "full_checkpoint": {
                "path": _project_path(config.full_checkpoint),
                "sha256": sha256_file(config.full_checkpoint),
                "seed": config.seeds[0],
            },
            "lora_adapter": {
                "path": _project_path(config.adapter_output),
                "sha256": sha256_file(config.adapter_output),
                "seed": config.seeds[0],
            },
        }
    )
    json_path, csv_path = write_run_artifacts(config.output_dir, manifest)
    _print_summary(config, seed_runs, manifest)
    print(f"\nFull checkpoint: {_project_path(config.full_checkpoint)}")
    print(f"LoRA adapter:    {_project_path(config.adapter_output)}")
    print(f"JSON manifest:   {_project_path(json_path)}")
    print(f"CSV history:     {_project_path(csv_path)}")
    print(
        "PASS: train/dev/test are isolated; all seeds completed; the primary "
        "adapter was reloaded independently with exact logits."
    )


if __name__ == "__main__":
    main()
