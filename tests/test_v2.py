"""Behavioral regressions for experiment durability, EOS and offline evaluation."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from learn_llm.artifacts import ArtifactValidationError, load_lora_model, load_tiny_gpt_checkpoint
from learn_llm.evaluation import EvaluationExample, evaluate_prediction, load_evaluation_jsonl, normalize_text
from learn_llm.experiment_tracking import BranchResult
from learn_llm.model import TinyGPT, TinyGPTConfig
from learn_llm.run_management import RunRecorder, checked_artifact, read_manifest
from learn_llm.sft import build_sft_sequence, collate_sft_batch, train_sft_steps
from learn_llm.sft_pipeline import train_branch
from learn_llm.tokenizer import CharTokenizer


class EOSAndEvaluationTests(unittest.TestCase):
    def test_eos_is_distinct_roundtrips_and_is_supervised(self):
        tokenizer = CharTokenizer.from_text("用户：问\n助手：答。", add_eos=True)
        restored = CharTokenizer.from_state_dict(tokenizer.state_dict())
        self.assertEqual(restored.eos_id, tokenizer.eos_id)
        sequence = build_sft_sequence(tokenizer, "问", "答。")
        self.assertEqual(sequence.labels[-1], tokenizer.eos_id)
        self.assertEqual(sequence.response_token_count, 3)
        self.assertEqual(sequence.labels[sequence.first_assistant_label_index], tokenizer.encode("答")[0])
        shorter = build_sft_sequence(tokenizer, "问", "答")
        batch = collate_sft_batch([sequence, shorter], pad_token_id=0)
        self.assertEqual(batch.labels[1, -1].item(), -100)
        self.assertEqual(restored.decode([tokenizer.eos_id], skip_special_tokens=True), "")

    def test_batch_stops_independently_and_restores_training_mode(self):
        model = TinyGPT(TinyGPTConfig(vocab_size=3, block_size=8, n_layer=1, n_head=1, n_embd=4))
        model.train()
        calls = []
        def forward(ids):
            calls.append(ids.shape[1])
            logits = torch.full((*ids.shape, 3), -100.)
            logits[0, -1, 2 if len(calls) == 1 else 0] = 100.
            logits[1, -1, 1 if len(calls) == 1 else 2] = 100.
            return logits, None
        with patch.object(model, "forward", side_effect=forward):
            generated = model.generate(torch.tensor([[0], [0]]), 8, do_sample=False, eos_token_id=2)
        self.assertEqual(generated.tolist(), [[0, 2, 2], [0, 1, 2]])
        self.assertEqual(len(calls), 2)
        self.assertTrue(model.training)
        with patch.object(model, "forward", side_effect=RuntimeError("broken")):
            with self.assertRaises(RuntimeError):
                model.generate(torch.tensor([[0]]), 1, eos_token_id=2)
        self.assertTrue(model.training)

    def test_zero_length_and_length_limit_without_eos(self):
        model = TinyGPT(TinyGPTConfig(vocab_size=3, block_size=4, n_layer=1, n_head=1, n_embd=4))
        ids = torch.tensor([[0]])
        self.assertTrue(torch.equal(ids, model.generate(ids, 0, eos_token_id=2)))
        def forward(ids):
            logits = torch.zeros((*ids.shape, 3)); logits[..., 0] = 100.
            return logits, None
        with patch.object(model, "forward", side_effect=forward):
            self.assertEqual(model.generate(ids, 5, do_sample=False, eos_token_id=2).shape[1], 6)
            self.assertEqual(model.generate(ids, 5, do_sample=False).shape[1], 6)

    def test_all_extended_rows_are_unique_covered_and_within_context(self):
        demo = sum((list(load_evaluation_jsonl(ROOT / "data" / f"tiny_instructions{suffix}.jsonl")) for suffix in ("", "_dev", "_test")), [])
        dev = load_evaluation_jsonl(ROOT / "data/extended_instructions_dev.jsonl")
        test = load_evaluation_jsonl(ROOT / "data/extended_instructions_test.jsonl")
        self.assertEqual((len(dev), len(test)), (20, 40))
        rows = demo + list(dev) + list(test)
        self.assertEqual(len({r.example_id for r in rows}), 72)
        self.assertEqual(len({normalize_text(r.instruction) for r in rows}), 72)
        vocab = CharTokenizer.from_text((ROOT / "data/tiny_corpus.txt").read_text(encoding="utf-8"), add_eos=True)
        train_families = {r.intent_family for r in demo if r.split == "train"}
        self.assertTrue({r.intent_family for r in dev} <= train_families)
        self.assertTrue({r.intent_family for r in test if r.evaluation_category == "new_intent"}.isdisjoint(train_families))
        for row in rows:
            with self.subTest(row=row.example_id):
                build_sft_sequence(vocab, row.instruction, row.reference_response, max_sequence_length=49)
                self.assertEqual(evaluate_prediction(row, row.reference_response).task_success, 1.)

    def test_rule_fixtures_include_negation_synonyms_and_reasons(self):
        examples = {r.example_id: r for r in load_evaluation_jsonl(ROOT / "data/tiny_instructions_test.jsonl")}
        for line in (ROOT / "data/evaluation_cases.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            result = evaluate_prediction(examples[row["example_id"]], row["prediction"])
            self.assertEqual(result.task_success, row["expected"], row["kind"])
            if row["kind"] in ("negation", "contradiction"):
                self.assertTrue(result.matched_contradictions)
            if row["kind"] == "format":
                self.assertIn("must_end_with", result.format_failures)


class DurableRunTests(unittest.TestCase):
    def test_interruption_preserves_steps_and_does_not_aggregate_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder(Path(directory), kind="sft-comparison", config={}, project_root=ROOT)
            branch = BranchResult("full", 1, .01, 1, 1, 0., status="running")
            recorder.branch(branch)
            recorder.step(branch, 1, 2., .1)
            with patch("learn_llm.plotting.plot_runs", side_effect=RuntimeError("plot unavailable")):
                recorder.finish(KeyboardInterrupt())
            result = read_manifest(Path(directory))
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["runs"][0]["branches"]["full"]["history"], [2.])
            self.assertEqual(result["aggregate"], {})
            self.assertEqual(result["completed_seeds"], [])
            self.assertIn("plot_error", result)
            before = (Path(directory) / "sft_comparison.json").read_bytes()
            recorder.manifest["bad"] = float("nan")
            with self.assertRaises(ValueError):
                recorder.flush()
            self.assertEqual((Path(directory) / "sft_comparison.json").read_bytes(), before)
            with self.assertRaises(FileExistsError):
                RunRecorder(Path(directory), kind="pretrain", config={}, project_root=ROOT)

    def test_training_callback_preserves_completed_step_on_tool_failure(self):
        tokenizer = CharTokenizer.from_text("用户：问\n助手：答", add_eos=True)
        sequence = build_sft_sequence(tokenizer, "问", "答")
        batch = collate_sft_batch([sequence], pad_token_id=0)
        model = TinyGPT(TinyGPTConfig(vocab_size=len(tokenizer), block_size=20, n_layer=1, n_head=1, n_embd=4))
        recorded = []
        def callback(step, loss, elapsed):
            recorded.append((step, loss))
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            train_branch(model, batch, steps=3, learning_rate=.01, sequences=[sequence], batch_size=1,
                         sampling_seed=1, on_step=callback)
        self.assertEqual(len(recorded), 1)
        recorded.clear()
        with self.assertRaises(KeyboardInterrupt):
            train_branch(model, batch, steps=3, learning_rate=.01, on_step=callback)
        self.assertEqual(len(recorded), 1)
        with patch.object(model, "forward", return_value=(None, torch.tensor(float("nan"), requires_grad=True))):
            with self.assertRaisesRegex(RuntimeError, "not finite"):
                train_sft_steps(model, batch, steps=1)


class RunIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.base = cls.root / "pretrain"
        cls.sft = cls.root / "sft"
        cls.run_cli("06_train_tiny_gpt.py", "--steps", "1", "--n-layer", "1", "--n-head", "1", "--n-embd", "8", "--output-dir", str(cls.base))
        cls.run_cli("10_sft_tiny_gpt.py", "--steps", "1", "--seeds", "1", "2", "3", "--full-learning-rate", "1e-12", "--max-new-tokens", "2", "--base-checkpoint", str(cls.base / "base.pt"), "--output-dir", str(cls.sft))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def run_cli(cls, script, *args, expected=0):
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLCONFIGDIR=str(cls.root / "mpl-cache"))
        completed = subprocess.run([sys.executable, str(ROOT / "experiments" / script), *args], cwd=ROOT,
                                   env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if completed.returncode != expected:
            raise AssertionError(f"exit {completed.returncode}, expected {expected}\n{completed.stdout}\n{completed.stderr}")
        return completed

    def test_all_seeds_survive_failed_quality_checks_and_strict_exit_is_two(self):
        manifest = read_manifest(self.sft)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["quality_status"], "failed")
        self.assertEqual(manifest["completed_seeds"], [1, 2, 3])
        self.assertEqual(manifest["aggregate"]["full"]["train_assistant_loss"]["runs"], 3)
        strict = self.root / "strict"
        self.run_cli("10_sft_tiny_gpt.py", "--steps", "1", "--seed", "1", "--full-learning-rate", "1e-12", "--max-new-tokens", "2", "--strict-checks", "--base-checkpoint", str(self.base / "base.pt"), "--output-dir", str(strict), expected=2)
        self.assertTrue((strict / "sft_comparison.csv").is_file())
        self.assertTrue((strict / "seed-1/lora/adapter.pt").is_file())

    def test_every_model_loads_after_relocation_and_hash_tampering_fails(self):
        moved = self.root / "moved"
        shutil.copytree(self.sft, moved)
        manifest = read_manifest(moved)
        base = checked_artifact(moved, manifest["artifacts"]["base"])
        for seed in (1, 2, 3):
            for branch in ("random", "full", "lora"):
                path = checked_artifact(moved, manifest["artifacts"][f"{seed}.{branch}"])
                bundle = load_lora_model(base, path) if branch == "lora" else load_tiny_gpt_checkpoint(path)
                self.assertIsNotNone(bundle.tokenizer.eos_id)
        self.run_cli("07_generate.py", "--run-dir", str(moved), "--branch", "lora", "--seed", "2", "--instruction", "为什么需要因果掩码？", "--tokens", "2")
        base.write_bytes(base.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            checked_artifact(moved, manifest["artifacts"]["base"])

    def test_old_format_is_rejected_with_retraining_instructions(self):
        payload = torch.load(self.base / "base.pt", weights_only=True)
        payload["format_version"] = 1
        path = self.root / "old.pt"
        torch.save(payload, path)
        with self.assertRaisesRegex(ArtifactValidationError, "06_train_tiny_gpt"):
            load_tiny_gpt_checkpoint(path)

    def test_invalid_config_and_short_context_leave_failure_manifest(self):
        path = self.root / "invalid"
        self.run_cli("06_train_tiny_gpt.py", "--steps", "0", "--output-dir", str(path), expected=1)
        self.assertEqual(read_manifest(path)["status"], "failed")
        nonfinite = self.root / "nonfinite"
        self.run_cli("06_train_tiny_gpt.py", "--learning-rate", "nan", "--output-dir", str(nonfinite), expected=1)
        self.assertEqual(read_manifest(nonfinite)["status"], "failed")
        self.assertEqual(read_manifest(nonfinite)["config"]["learning_rate"], "nan")
        short = self.root / "short"
        self.run_cli("06_train_tiny_gpt.py", "--steps", "1", "--block-size", "4", "--n-layer", "1", "--n-head", "1", "--n-embd", "8", "--output-dir", str(short))
        out = self.root / "short-sft"
        self.run_cli("10_sft_tiny_gpt.py", "--base-checkpoint", str(short / "base.pt"), "--output-dir", str(out), expected=1)
        self.assertIn("exceeds", read_manifest(out)["error"]["message"])

    def test_automatic_and_multi_run_plots_are_readable(self):
        from PIL import Image
        from learn_llm.plotting import plot_runs
        for directory in (self.base, self.sft):
            manifest = read_manifest(directory)
            self.assertNotIn("plot_error", manifest)
            with Image.open(directory / "plots/loss.png") as image:
                image.verify()
        paths = plot_runs([self.base, self.sft], self.root / "comparison")
        self.assertEqual(len(paths), 6)
        self.assertIn("<svg", (self.root / "comparison/loss.svg").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
