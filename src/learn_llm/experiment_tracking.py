"""Versioned, machine-readable records for the teaching experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

import numpy
import torch


RUN_MANIFEST_VERSION = 1


def privacy_safe_path(path: Path, *, project_root: Path) -> str:
    """Return a relative path or a non-identifying external path token."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        path_token = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
        return f"external/{path_token}/{resolved.name}"


@dataclass(frozen=True, slots=True)
class SFTExperimentConfig:
    """All user-controlled settings for one SFT comparison run."""

    base_checkpoint: Path
    train_data: Path
    dev_data: Path
    test_data: Path
    output_dir: Path
    full_checkpoint: Path
    adapter_output: Path
    seeds: tuple[int, ...] = (42, 43, 44)
    steps: int = 180
    train_batch_size: int = 2
    random_learning_rate: float = 1e-2
    full_learning_rate: float = 3e-3
    lora_learning_rate: float = 1e-2
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    max_new_tokens: int = 24
    device: str = "cpu"

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        learning_rates = (
            self.random_learning_rate,
            self.full_learning_rate,
            self.lora_learning_rate,
        )
        if any(rate <= 0 for rate in learning_rates):
            raise ValueError("learning rates must be positive")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        artifact_paths = {
            self.base_checkpoint.resolve(),
            self.full_checkpoint.resolve(),
            self.adapter_output.resolve(),
        }
        if len(artifact_paths) != 3:
            raise ValueError(
                "base_checkpoint, full_checkpoint, and adapter_output must be distinct"
            )
        data_paths = {
            self.train_data.resolve(),
            self.dev_data.resolve(),
            self.test_data.resolve(),
        }
        if len(data_paths) != 3:
            raise ValueError("train_data, dev_data, and test_data must be distinct")

    def to_manifest(self, *, project_root: Path) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "base_checkpoint",
            "train_data",
            "dev_data",
            "test_data",
            "output_dir",
            "full_checkpoint",
            "adapter_output",
        ):
            result[key] = privacy_safe_path(result[key], project_root=project_root)
        result["seeds"] = list(self.seeds)
        return result


@dataclass(slots=True)
class BranchResult:
    """Metrics and generated cases for one branch and random seed."""

    name: str
    seed: int
    learning_rate: float | None
    trainable_parameters: int
    total_parameters: int
    duration_seconds: float
    history: list[float] = field(default_factory=list)
    losses: dict[str, float] = field(default_factory=dict)
    per_example_losses: dict[str, list[float]] = field(default_factory=dict)
    generations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    task_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(project_root: Path) -> dict[str, Any]:
    """Return reproducibility metadata without making git a hard dependency."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def runtime_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "executable": Path(sys.executable).name,
    }


def aggregate_branch_metrics(
    runs: Iterable[Mapping[str, BranchResult]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute population mean/std for each scalar loss across seeds."""

    grouped: dict[str, dict[str, list[float]]] = {}
    for branches in runs:
        for branch_name, branch in branches.items():
            destination = grouped.setdefault(branch_name, {})
            for metric_name, value in branch.losses.items():
                destination.setdefault(metric_name, []).append(float(value))
            for split, metrics in branch.task_metrics.items():
                for metric_name, value in metrics.items():
                    destination.setdefault(
                        f"{split}.{metric_name}", []
                    ).append(float(value))

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for branch_name, metrics in grouped.items():
        aggregate[branch_name] = {
            metric_name: {
                "mean": fmean(values),
                "std": pstdev(values),
                "runs": float(len(values)),
            }
            for metric_name, values in metrics.items()
        }
    return aggregate


def build_run_manifest(
    *,
    project_root: Path,
    config: SFTExperimentConfig,
    data_sha256: Mapping[str, str],
    base_sha256: str,
    seed_runs: list[Mapping[str, BranchResult]],
    invariants: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "learnllm.sft-comparison-run",
        "schema_version": RUN_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(project_root),
        "environment": runtime_environment(),
        "config": config.to_manifest(project_root=project_root),
        "artifacts": {
            "base_checkpoint_sha256": base_sha256,
            "data_sha256": dict(sorted(data_sha256.items())),
        },
        "runs": [
            {
                "seed": next(iter(branches.values())).seed,
                "branches": {
                    name: result.to_manifest()
                    for name, result in sorted(branches.items())
                },
            }
            for branches in seed_runs
        ],
        "aggregate": aggregate_branch_metrics(seed_runs),
        "invariants": dict(invariants),
        "limitations": [
            "This tiny teaching model is not a model-quality benchmark.",
            "Task metrics are deterministic heuristics, not human evaluation.",
            "The final test split is evaluated after training and is not used for tuning.",
        ],
    }


def write_run_artifacts(
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Atomically write the JSON manifest and long-format history CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sft_comparison.json"
    csv_path = output_dir / "sft_comparison.csv"
    json_temporary = json_path.with_suffix(".json.tmp")
    csv_temporary = csv_path.with_suffix(".csv.tmp")

    json_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    json_temporary.replace(json_path)

    fieldnames = (
        "seed",
        "step",
        "branch",
        "split",
        "loss",
        "learning_rate",
        "trainable_parameters",
        "total_parameters",
        "duration_seconds",
    )
    with csv_temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for run in manifest["runs"]:
            seed = run["seed"]
            for branch_name, branch in sorted(run["branches"].items()):
                common = {
                    "seed": seed,
                    "branch": branch_name,
                    "learning_rate": branch["learning_rate"],
                    "trainable_parameters": branch["trainable_parameters"],
                    "total_parameters": branch["total_parameters"],
                    "duration_seconds": branch["duration_seconds"],
                }
                for step, loss in enumerate(branch["history"], start=1):
                    writer.writerow(
                        {**common, "step": step, "split": "train_step", "loss": loss}
                    )
                for split, loss in sorted(branch["losses"].items()):
                    writer.writerow(
                        {**common, "step": "final", "split": split, "loss": loss}
                    )
    csv_temporary.replace(csv_path)
    return json_path, csv_path
