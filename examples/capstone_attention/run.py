"""Reproduce the completed attention-scaling capstone example.

The experiment asks whether dividing attention logits by ``sqrt(d_k)`` keeps
the attention distribution from becoming increasingly sharp as the head
dimension grows.  It writes raw aggregate metrics to ``results.csv`` and uses
assertions only for broad, causal trends rather than exact floating values.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch


EXAMPLE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learn_llm.attention import scaled_dot_product_attention


def distribution_metrics(weights: torch.Tensor) -> tuple[float, float]:
    """Return mean row entropy and mean largest probability."""

    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = -(safe * safe.log()).sum(dim=-1).mean()
    peak = weights.max(dim=-1).values.mean()
    return float(entropy), float(peak)


def run_condition(
    *, head_dimension: int, scaled: bool, trials: int, tokens: int, seed: int
) -> dict[str, float | int | bool]:
    generator = torch.Generator().manual_seed(seed + head_dimension)
    query = torch.randn(trials, 1, tokens, head_dimension, generator=generator)
    key = torch.randn(trials, 1, tokens, head_dimension, generator=generator)
    value = torch.randn(trials, 1, tokens, head_dimension, generator=generator)

    if scaled:
        _, weights = scaled_dot_product_attention(query, key, value)
    else:
        scores = query @ key.transpose(-2, -1)
        weights = torch.softmax(scores, dim=-1)

    entropy, peak = distribution_metrics(weights)
    return {
        "d_k": head_dimension,
        "scaled": scaled,
        "mean_entropy": entropy,
        "mean_peak_weight": peak,
    }


def verify_causal_mask(seed: int) -> None:
    """Keep leakage prevention as a correctness guard, not an experiment variable."""

    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(2, 1, 6, 8, generator=generator)
    key = torch.randn(2, 1, 6, 8, generator=generator)
    value = torch.randn(2, 1, 6, 8, generator=generator)
    _, weights = scaled_dot_product_attention(query, key, value, causal=True)
    future = torch.triu(weights, diagonal=1)
    assert torch.count_nonzero(future) == 0


def main() -> None:
    config = json.loads(
        (EXAMPLE_ROOT / "config" / "experiment.json").read_text(encoding="utf-8")
    )
    dimensions = [int(item) for item in config["head_dimensions"]]
    trials = int(config["trials"])
    tokens = int(config["tokens"])
    seed = int(config["seed"])

    rows = [
        run_condition(
            head_dimension=dimension,
            scaled=scaled,
            trials=trials,
            tokens=tokens,
            seed=seed,
        )
        for dimension in dimensions
        for scaled in (False, True)
    ]

    output = EXAMPLE_ROOT / "results.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    verify_causal_mask(seed)
    by_condition = {(int(row["d_k"]), bool(row["scaled"])): row for row in rows}
    smallest = dimensions[0]
    largest = dimensions[-1]
    unscaled_entropy_drop = float(by_condition[(smallest, False)]["mean_entropy"]) - float(
        by_condition[(largest, False)]["mean_entropy"]
    )
    scaled_entropy_change = abs(
        float(by_condition[(smallest, True)]["mean_entropy"])
        - float(by_condition[(largest, True)]["mean_entropy"])
    )

    assert unscaled_entropy_drop > 0.45
    assert scaled_entropy_change < 0.20
    assert float(by_condition[(largest, False)]["mean_peak_weight"]) > float(
        by_condition[(largest, True)]["mean_peak_weight"]
    )

    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Unscaled entropy drop: {unscaled_entropy_drop:.4f}")
    print(f"Scaled entropy change: {scaled_entropy_change:.4f}")
    print("PASS: scaling stabilizes sharpness as d_k grows; causal mask has zero leakage.")


if __name__ == "__main__":
    main()
