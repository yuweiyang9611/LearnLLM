from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learn_llm.experiment_tracking import (  # noqa: E402
    BranchResult,
    SFTExperimentConfig,
    aggregate_branch_metrics,
    privacy_safe_path,
    write_run_artifacts,
)


def branch(seed: int, final_loss: float) -> BranchResult:
    return BranchResult(
        name="full",
        seed=seed,
        learning_rate=0.003,
        trainable_parameters=10,
        total_parameters=10,
        duration_seconds=0.25,
        history=[2.0, final_loss],
        losses={"train": final_loss, "test": final_loss + 0.1},
        task_metrics={"test": {"task_success": 0.5 + 0.1 * (seed - 1)}},
    )


class ExperimentTrackingTests(unittest.TestCase):
    def test_external_paths_are_private_and_collision_resistant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            first = Path(directory) / "one" / "data.jsonl"
            second = Path(directory) / "two" / "data.jsonl"
            first_display = privacy_safe_path(first, project_root=root)
            second_display = privacy_safe_path(second, project_root=root)
            self.assertTrue(first_display.startswith("external/"))
            self.assertTrue(first_display.endswith("/data.jsonl"))
            self.assertNotEqual(first_display, second_display)
            self.assertNotIn(str(Path(directory)), first_display)

    def test_config_validation_rejects_ambiguous_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "base_checkpoint": root / "base.pt",
                "train_data": root / "train.jsonl",
                "dev_data": root / "dev.jsonl",
                "test_data": root / "test.jsonl",
                "output_dir": root / "out",
                "full_checkpoint": root / "full.pt",
                "adapter_output": root / "adapter.pt",
            }
            SFTExperimentConfig(**common).validate()
            with self.assertRaisesRegex(ValueError, "unique"):
                SFTExperimentConfig(**common, seeds=(1, 1)).validate()
            with self.assertRaisesRegex(ValueError, "learning rates"):
                SFTExperimentConfig(**common, full_learning_rate=0.0).validate()
            overlapping = dict(common)
            overlapping["adapter_output"] = overlapping["full_checkpoint"]
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                SFTExperimentConfig(**overlapping).validate()

            duplicate_data = dict(common)
            duplicate_data["test_data"] = duplicate_data["dev_data"]
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                SFTExperimentConfig(**duplicate_data).validate()

    def test_aggregate_reports_mean_and_population_std(self) -> None:
        aggregate = aggregate_branch_metrics(
            [{"full": branch(1, 1.0)}, {"full": branch(2, 3.0)}]
        )
        self.assertEqual(aggregate["full"]["train"]["mean"], 2.0)
        self.assertEqual(aggregate["full"]["train"]["std"], 1.0)
        self.assertEqual(aggregate["full"]["train"]["runs"], 2.0)
        self.assertAlmostEqual(
            aggregate["full"]["test.task_success"]["mean"], 0.55
        )

    def test_writer_creates_versioned_json_and_long_csv(self) -> None:
        manifest = {
            "schema": "learnllm.sft-comparison-run",
            "schema_version": 1,
            "runs": [
                {
                    "seed": 1,
                    "branches": {"full": branch(1, 1.0).to_manifest()},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path = write_run_artifacts(Path(directory), manifest)
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema_version"], 1)
            with csv_path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["branch"], "full")
            self.assertEqual(rows[-1]["split"], "train")


if __name__ == "__main__":
    unittest.main()
