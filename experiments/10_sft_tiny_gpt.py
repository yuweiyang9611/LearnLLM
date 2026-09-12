"""实验 10：可信地比较随机训练、Full SFT 与 LoRA-SFT。"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import DATA_DIR, PROJECT_ROOT, seed_everything
from learn_llm.artifacts import (
    LORA_ADAPTER_FORMAT_VERSION,
    TinyGPTBundle,
    load_lora_model,
    load_tiny_gpt_checkpoint,
)
from learn_llm.evaluation import EVALUATION_RULES_VERSION, EvaluationExample, load_evaluation_jsonl, normalize_text
from learn_llm.experiment_tracking import (
    BranchResult,
    SFTExperimentConfig,
    privacy_safe_path,
    sha256_file,
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
from learn_llm.run_management import RunRecorder, latest_base, unique_run_dir


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
        default=None,
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
        default=None,
    )
    parser.add_argument(
        "--adapter-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--random-learning-rate", type=float, default=1e-2)
    parser.add_argument("--full-learning-rate", type=float, default=3e-3)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-2)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 cuda:N")
    parser.add_argument("--strict-checks", action="store_true")
    parser.add_argument("--eval-suite", choices=("demo", "extended"), default="demo")
    args = parser.parse_args(argv)
    if args.eval_suite == "extended":
        if args.dev_data == DATA_DIR / "tiny_instructions_dev.jsonl":
            args.dev_data = DATA_DIR / "extended_instructions_dev.jsonl"
        if args.test_data == DATA_DIR / "tiny_instructions_test.jsonl":
            args.test_data = DATA_DIR / "extended_instructions_test.jsonl"
    return args


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
        base_checkpoint=args.base_checkpoint.resolve() if args.base_checkpoint else None,
        train_data=args.train_data.resolve(),
        dev_data=args.dev_data.resolve(),
        test_data=args.test_data.resolve(),
        output_dir=(args.output_dir or _default_output_dir(seeds)).resolve(),
        full_checkpoint=args.full_checkpoint.resolve() if args.full_checkpoint else None,
        adapter_output=args.adapter_output.resolve() if args.adapter_output else None,
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
        strict_checks=args.strict_checks,
        eval_suite=args.eval_suite,
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
        path = (bundle.checkpoint_path.parent / name).resolve()
        if not path.is_relative_to(bundle.checkpoint_path.parent.resolve()):
            raise ValueError("base data path escapes its directory")
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


def run_comparison(config, recorder):
    for destination in (config.full_checkpoint, config.adapter_output):
        if destination is not None and (destination.exists() or destination.is_relative_to(recorder.directory)):
            raise FileExistsError(f"export must be a new path outside the run: {destination}")
    base_path = config.base_checkpoint or latest_base(PROJECT_ROOT / "outputs" / "pretrain")
    bundle = load_tiny_gpt_checkpoint(base_path, map_location=config.device)
    _verify_recorded_repository_data(bundle)
    # Carry all immutable inputs required to reproduce this run alongside the base.
    for name in bundle.data_sha256:
        source = (bundle.checkpoint_path.parent / name).resolve()
        if not source.is_relative_to(bundle.checkpoint_path.parent.resolve()):
            raise ValueError("base data path escapes its directory")
        target = recorder.directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    copied_base = recorder.directory / "base.pt"
    shutil.copyfile(base_path, copied_base)
    config = replace(config, base_checkpoint=copied_base)
    bundle = load_tiny_gpt_checkpoint(copied_base, map_location=config.device)
    recorder.manifest["config"] = config.to_manifest(project_root=PROJECT_ROOT)
    recorder.manifest["config"]["model"] = asdict(bundle.config)
    recorder.artifact("base", copied_base, branch="base", model_config=asdict(bundle.config))
    examples = {split: load_split(getattr(config, f"{split}_data"), split) for split in ("train", "dev", "test")}
    ids = [row.example_id for rows in examples.values() for row in rows]
    prompts = [normalize_text(row.instruction) for rows in examples.values() for row in rows]
    if len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts):
        raise ValueError("IDs and normalized questions must be unique across train/dev/test")
    pairs = {split: _pairs(rows) for split, rows in examples.items()}
    prepared = {}
    for split in examples:
        require_tokenizer_coverage(bundle.tokenizer, pairs[split], dataset_name=split)
        prepared[split] = build_instruction_batch(bundle.tokenizer, pairs[split], block_size=bundle.config.block_size, device=config.device)
    sequences = {split: value[0] for split, value in prepared.items()}
    batches = {split: value[1] for split, value in prepared.items()}
    data_hashes = data_file_sha256([config.train_data, config.dev_data, config.test_data], project_root=PROJECT_ROOT)
    recorder.manifest["data_sha256"] = data_hashes
    recorder.manifest["evaluation_rules_version"] = EVALUATION_RULES_VERSION
    recorder.manifest["evaluation_rules_sha256"] = sha256_file(PROJECT_ROOT / "src/learn_llm/evaluation.py")
    for split in examples:
        copied_data = recorder.directory / "data" / f"{split}.jsonl"
        copied_data.parent.mkdir(exist_ok=True)
        shutil.copyfile(getattr(config, f"{split}_data"), copied_data)
        recorder.artifact(f"data.{split}", copied_data)
    corpus_path = recorder.directory / bundle.metadata["corpus"] if "corpus" in bundle.metadata else recorder.directory / "corpus.txt"
    common = dict(tokenizer=bundle.tokenizer, examples_by_split=examples, sequences_by_split=sequences,
                  batches_by_split=batches, corpus_path=corpus_path, max_new_tokens=config.max_new_tokens, device=config.device)
    for seed in config.seeds:
        seed_everything(seed)
        base_model = copy.deepcopy(bundle.model).eval()
        base_model.requires_grad_(False)
        base_result = evaluate_branch(name="base", seed=seed, model=base_model, learning_rate=None,
                                      history=[], duration_seconds=0., **common)
        recorder.branch(base_result)
        for name in ("random", "full", "lora"):
            seed_everything(seed + (1 if name == "random" else 2 if name == "lora" else 0))
            model = TinyGPT(bundle.config).to(config.device) if name == "random" else copy.deepcopy(bundle.model)
            model.requires_grad_(True)
            targets = attention_lora_targets(bundle.config)
            if name == "lora":
                inject_lora(model, targets, rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout)
                if not lora_parameter_names(model):
                    raise RuntimeError("empty LoRA trainable set")
                model.eval()
                with torch.no_grad():
                    before, _ = base_model(batches["train"].input_ids)
                    after, _ = model(batches["train"].input_ids)
                if not torch.allclose(before, after, atol=1e-7, rtol=0):
                    raise RuntimeError("LoRA injection changed initial logits")
            snapshot = {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
            initial_loss = evaluate_assistant_loss(model, batches["train"])
            rate = getattr(config, f"{name}_learning_rate")
            progress = BranchResult(name=name, seed=seed, learning_rate=rate, status="running",
                                    trainable_parameters=model.parameter_count(trainable_only=True),
                                    total_parameters=model.parameter_count(), duration_seconds=0.)
            recorder.branch(progress)
            history, duration = train_branch(model, batches["train"], steps=config.steps, learning_rate=rate,
                sequences=sequences["train"], batch_size=config.train_batch_size, sampling_seed=seed + 100,
                device=config.device, on_step=lambda step, loss, elapsed: recorder.step(progress, step, loss, elapsed))
            if name == "lora" and any(not torch.equal(p.detach(), snapshot[k]) for k, p in model.named_parameters() if not p.requires_grad):
                raise RuntimeError("LoRA training changed frozen parameters")
            result = evaluate_branch(name=name, seed=seed, model=model, learning_rate=rate,
                                     history=history, duration_seconds=duration, **common)
            result.status = "running"
            recorder.branch(result)
            factor = {"random": .85, "full": .55, "lora": .90}[name]
            recorder.check("train_loss_decline", result.losses["train_assistant_loss"], initial_loss * factor, seed=seed, branch=name)
            recorder.check("parameter_update", parameter_l2_change(model, snapshot), 0., relation="gt", seed=seed, branch=name)
            if name in ("full", "lora"):
                for index, (old, new) in enumerate(zip(base_result.per_example_losses["train"], result.per_example_losses["train"])):
                    recorder.check(f"training_example_{index}", new, old, seed=seed, branch=name)
            destination = recorder.directory / f"seed-{seed}" / name / ("adapter.pt" if name == "lora" else "model.pt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            metadata = _checkpoint_metadata(bundle=bundle, config=config, result=result, objective=f"{name} assistant-only SFT with EOS", run_data_sha256=data_hashes)
            if name == "lora":
                metadata.update(targets=list(targets), rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout)
                torch.save({"format_version": LORA_ADAPTER_FORMAT_VERSION, "adapter_state": lora_adapter_state_dict(model), "metadata": metadata}, destination)
                restored = load_lora_model(copied_base, destination).model
            else:
                save_checkpoint(destination, model, step=config.steps, metadata=metadata)
                restored = load_tiny_gpt_checkpoint(destination).model
            source = copy.deepcopy(model).cpu().eval()
            with torch.no_grad():
                expected, _ = source(batches["train"].input_ids.cpu())
                actual, _ = restored(batches["train"].input_ids.cpu())
            if not torch.equal(expected, actual):
                raise RuntimeError("independent reload does not reproduce logits")
            recorder.artifact(f"{seed}.{name}", destination, seed=seed, branch=name, model_config=asdict(bundle.config))
            result.status = "completed"
            recorder.flush()
            export = config.full_checkpoint if name == "full" else config.adapter_output if name == "lora" else None
            if seed == config.seeds[0] and export is not None:
                export.parent.mkdir(parents=True, exist_ok=True)
                with export.open("xb") as output, destination.open("rb") as source_file:
                    shutil.copyfileobj(source_file, output)
    recorder.finish()
    _print_summary(config, list(recorder.branches.values()), recorder.manifest)
    print(f"Run: {recorder.directory}")
    print(f'Inference: python experiments/07_generate.py --run-dir "{recorder.directory}" --branch lora --instruction "为什么需要因果掩码？"')
    return recorder.exit_code(config.strict_checks)


def main():
    args = parse_args()
    args.output_dir = args.output_dir or unique_run_dir(PROJECT_ROOT / "outputs/sft")
    raw_config = {key: privacy_safe_path(value, project_root=PROJECT_ROOT) if isinstance(value, Path) else value for key, value in vars(args).items()}
    recorder = RunRecorder(args.output_dir, kind="sft-comparison", config=raw_config, project_root=PROJECT_ROOT)
    try:
        return run_comparison(build_config(args), recorder)
    except (Exception, KeyboardInterrupt) as error:
        recorder.finish(error)
        print(f"{type(error).__name__}: {error}; records: {recorder.directory}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
