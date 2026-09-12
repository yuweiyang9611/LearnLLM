"""Headless plots built from saved CSV values, never from a training model."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

from .run_management import read_manifest


def plot_runs(directories: list[Path], output: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "learnllm-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    summaries = []
    run_labels = []
    for index, directory in enumerate(directories, 1):
        manifest = read_manifest(directory)
        config = manifest["config"]
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        data_id = hashlib.sha256(json.dumps(manifest.get("data_sha256", {}), sort_keys=True).encode()).hexdigest()[:8]
        label = f"run {index} cfg:{fingerprint} data:{data_id} {config.get('eval_suite', 'pretrain')} [{manifest['status']}]"
        run_labels.append(label)
        stem = "pretraining" if manifest["schema"] == "learnllm.pretrain-run" else "sft_comparison"
        with (directory / f"{stem}.csv").open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        series = defaultdict(list)
        for row in rows:
            if row["step"] != "final":
                series[(row["seed"], row["branch"], row["split"])].append((int(row["step"]), float(row["loss"])))
        for (seed, branch, split), values in series.items():
            ax.plot(*zip(*values), label=f"run {index} {branch}/{split} s{seed}", alpha=.8)
        groups = defaultdict(list)
        for run in manifest["runs"]:
            for name, branch in run["branches"].items():
                if branch.get("status", "completed") == "completed":
                    groups[name].append(branch)
        for name, branches in groups.items():
            summaries.append((f"run {index}\n{name}\nn={len(branches)}", branches))
    ax.set(xlabel="Optimization step", ylabel="Loss")
    ax.set_title("Raw training and validation losses\n" + "\n".join(run_labels), fontsize=10)
    if ax.lines:
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    else:
        ax.text(.5, .5, "No completed training steps", ha="center", transform=ax.transAxes)

    def save(figure, name):
        try:
            for suffix in ("png", "svg"):
                destination = output / f"{name}.{suffix}"
                figure.savefig(destination, dpi=150)
                saved.append(destination)
        finally:
            plt.close(figure)

    save(fig, "loss")
    for metric, title in (("trainable_parameters", "Trainable parameters"), ("duration_seconds", "Training duration (seconds)")):
        fig, ax = plt.subplots(figsize=(max(8, len(summaries) * 1.5), 5), layout="constrained")
        values = [[b[metric] for b in branches] for _, branches in summaries]
        ax.bar(range(len(values)), [fmean(v) for v in values], yerr=[pstdev(v) for v in values], capsize=4)
        ax.set_xticks(range(len(values)), [label for label, _ in summaries], fontsize=9)
        ax.set(ylabel=title)
        ax.set_title(f"{title}: mean and population standard deviation\n" + "\n".join(run_labels), fontsize=9)
        save(fig, metric)
    # A readable sidecar maps short plot labels to complete configurations and datasets.
    (output / "runs.json").write_text(json.dumps([
        {"label": index, "directory": str(path), "config": read_manifest(path)["config"],
         "data_sha256": read_manifest(path).get("data_sha256", {})}
        for index, path in enumerate(directories, 1)
    ], ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return saved
