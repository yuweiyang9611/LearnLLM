"""Durable, relocatable experiment records; model quality is separate from execution."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment_tracking import (
    RUN_MANIFEST_VERSION, BranchResult, aggregate_branch_metrics, git_state, runtime_environment,
    sha256_file, write_run_artifacts,
)


def unique_run_dir(root: Path) -> Path:
    return root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def read_manifest(directory: Path) -> dict[str, Any]:
    files = [directory / name for name in ("pretraining.json", "sft_comparison.json") if (directory / name).is_file()]
    if len(files) != 1:
        raise ValueError(f"expected exactly one run manifest in {directory}")
    value = json.loads(files[0].read_text(encoding="utf-8"))
    if value.get("schema_version") != 2:
        raise ValueError("run requires version 2; rerun experiments 06 then 10")
    return value


def checked_artifact(directory: Path, entry: dict[str, Any]) -> Path:
    path = (directory / entry["path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("artifact path escapes the run directory")
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"artifact SHA-256 mismatch: {entry['path']}")
    return path


def latest_base(root: Path) -> Path:
    pointer = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    directory = (root / pointer["run"]).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("latest run path escapes pretraining root")
    manifest = read_manifest(directory)
    if manifest["status"] != "completed":
        raise ValueError("latest pretraining run is not completed")
    return checked_artifact(directory, manifest["artifacts"]["base"])


class RunRecorder:
    def __init__(self, directory: Path, *, kind: str, config: dict[str, Any], project_root: Path):
        self.directory = directory.resolve()
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"run directory is not empty: {directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        # Preserve invalid numeric CLI inputs as text so validation can report them
        # in a standards-compliant failure manifest. Runtime metrics remain strict.
        config = {key: str(value) if isinstance(value, float) and not math.isfinite(value) else value for key, value in config.items()}
        self.branches: dict[int, dict[str, BranchResult]] = {}
        self.manifest: dict[str, Any] = {
            "schema": f"learnllm.{kind}-run", "schema_version": RUN_MANIFEST_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running", "config": config,
            "git": git_state(project_root), "environment": runtime_environment(),
            "checks": [], "artifacts": {}, "runs": [], "aggregate": {},
            "limitations": ["Rule-based evaluation is not general semantic correctness.",
                            "Seed variation here does not include independent pretraining."],
        }
        self.flush()

    def flush(self) -> None:
        self.manifest["runs"] = [
            {"seed": seed, "branches": {name: asdict(result) for name, result in branches.items()}}
            for seed, branches in self.branches.items()
        ]
        self.manifest["aggregate"] = aggregate_branch_metrics(self.branches.values())
        self.manifest["completed_seeds"] = [
            seed for seed, branches in self.branches.items()
            if set(branches) == ({"base"} if self.manifest["schema"] == "learnllm.pretrain-run" else {"base", "random", "full", "lora"})
            and all(result.status == "completed" for result in branches.values())
        ]
        self.manifest["quality_status"] = "failed" if any(c["status"] == "failed" for c in self.manifest["checks"]) else "passed" if self.manifest["checks"] else "not_evaluated"
        write_run_artifacts(self.directory, self.manifest)

    def branch(self, result: BranchResult) -> None:
        self.branches.setdefault(result.seed, {})[result.name] = result
        self.flush()

    def step(self, result: BranchResult, step: int, loss: float, elapsed: float) -> None:
        if not math.isfinite(loss):
            raise RuntimeError("training loss is not finite")
        result.history.append(loss)
        result.duration_seconds = elapsed
        if len(result.history) != step:
            raise ValueError("training steps are not sequential")
        self.flush()

    def artifact(self, key: str, path: Path, **metadata: Any) -> None:
        self.manifest["artifacts"][key] = {
            "path": path.resolve().relative_to(self.directory).as_posix(),
            "sha256": sha256_file(path), **metadata,
        }
        self.flush()

    def check(self, name: str, actual: float, threshold: float, *, relation: str = "lt", **metadata: Any) -> None:
        passed = actual < threshold if relation == "lt" else actual > threshold
        self.manifest["checks"].append({
            "name": name, "actual": actual, "threshold": threshold, "relation": relation,
            "status": "passed" if passed else "failed",
            "reason": f"expected actual {relation} threshold", **metadata,
        })
        self.flush()

    def finish(self, error: BaseException | None = None) -> None:
        self.manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed" if error is not None else "completed"
        if error is not None:
            self.manifest["error"] = {"type": type(error).__name__, "message": str(error)}
            for branches in self.branches.values():
                for result in branches.values():
                    if result.status == "running":
                        result.status = self.manifest["status"]
        self.flush()
        try:
            from .plotting import plot_runs
            self.manifest["plots"] = [p.name for p in plot_runs([self.directory], self.directory / "plots")]
        except Exception as plot_error:
            self.manifest["plot_error"] = {"type": type(plot_error).__name__, "message": str(plot_error)}
        self.flush()

    def exit_code(self, strict: bool) -> int:
        return 2 if strict and self.manifest["quality_status"] == "failed" else 0
