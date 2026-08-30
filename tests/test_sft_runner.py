from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(SRC_DIR))


def load_experiment() -> object:
    path = EXPERIMENTS_DIR / "10_sft_tiny_gpt.py"
    spec = importlib.util.spec_from_file_location("learn_llm_experiment_10", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SFTRunnerCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = load_experiment()

    def test_default_run_uses_three_seeds_and_configurable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = self.experiment.parse_args(
                ["--quick", "--output-dir", str(output), "--train-batch-size", "3"]
            )
            config = self.experiment.build_config(args)
        self.assertEqual(config.seeds, (42, 43, 44))
        self.assertEqual(config.steps, 30)
        self.assertEqual(config.train_batch_size, 3)
        self.assertEqual(config.output_dir, output.resolve())

    def test_single_seed_compatibility_and_multi_seed_cli_are_exclusive(self) -> None:
        config = self.experiment.build_config(
            self.experiment.parse_args(["--seed", "7"])
        )
        self.assertEqual(config.seeds, (7,))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.experiment.parse_args(["--seed", "7", "--seeds", "7", "8"])
        self.assertEqual(raised.exception.code, 2)

    def test_invalid_numeric_options_fail_before_training(self) -> None:
        cases = (
            ["--steps", "0"],
            ["--train-batch-size", "0"],
            ["--lora-rank", "0"],
            ["--seeds", "1", "1"],
            ["--full-checkpoint", "same.pt", "--adapter-output", "same.pt"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                args = self.experiment.parse_args(arguments)
                with self.assertRaises(ValueError):
                    self.experiment.build_config(args)


if __name__ == "__main__":
    unittest.main()
