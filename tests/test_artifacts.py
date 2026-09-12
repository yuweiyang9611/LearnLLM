from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learn_llm.artifacts import (  # noqa: E402
    LORA_ADAPTER_FORMAT_VERSION,
    TINY_GPT_CHECKPOINT_FORMAT_VERSION,
    ArtifactValidationError,
    load_lora_model,
    load_tiny_gpt_checkpoint,
)
from learn_llm.lora import LoRALinear, inject_lora, lora_adapter_state_dict  # noqa: E402
from learn_llm.model import TinyGPT, TinyGPTConfig  # noqa: E402
from learn_llm.tokenizer import CharTokenizer  # noqa: E402
from learn_llm.training import save_checkpoint  # noqa: E402


TARGET = "blocks.0.attention.qkv"


class ArtifactLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        torch.manual_seed(2026)
        self.tokenizer = CharTokenizer.from_text("abcde", add_eos=True)
        self.config = TinyGPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=5,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
        )
        self.data_sha256 = {
            "data/tiny_corpus.txt": hashlib.sha256(b"corpus-v1").hexdigest(),
            "data/tiny_instructions.jsonl": hashlib.sha256(
                b"instructions-v1"
            ).hexdigest(),
            "data/tiny_instructions_eval.jsonl": hashlib.sha256(
                b"heldout-v1"
            ).hexdigest(),
        }
        self.base_model = TinyGPT(self.config).eval()
        self.input_ids = torch.tensor([[0, 1, 2, 3]])
        with torch.no_grad():
            self.base_logits, _ = self.base_model(self.input_ids)

        self.base_path = self.directory / "tiny_gpt.pt"
        save_checkpoint(
            self.base_path,
            self.base_model,
            step=17,
            metadata={
                "config": asdict(self.config),
                "tokenizer": self.tokenizer.state_dict(),
                "data_sha256": self.data_sha256,
                "seed": 2026,
            },
        )
        self.base_sha256 = hashlib.sha256(self.base_path.read_bytes()).hexdigest()

        self.adapted_model = TinyGPT(self.config).eval()
        self.adapted_model.load_state_dict(self.base_model.state_dict(), strict=True)
        inject_lora(
            self.adapted_model,
            (TARGET,),
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )
        with torch.no_grad():
            layer = self.adapted_model.get_submodule(TARGET)
            assert isinstance(layer, LoRALinear)
            layer.lora_A.copy_(
                torch.arange(layer.lora_A.numel()).reshape_as(layer.lora_A) / 100.0
            )
            layer.lora_B.copy_(
                (torch.arange(layer.lora_B.numel()).reshape_as(layer.lora_B) + 1)
                / 200.0
            )
            self.adapter_logits, _ = self.adapted_model(self.input_ids)

        self.adapter_payload = {
            "format_version": 2,
            "adapter_state": lora_adapter_state_dict(self.adapted_model),
            "metadata": {
                "config": asdict(self.config),
                "tokenizer": self.tokenizer.state_dict(),
                "base_checkpoint": "checkpoints/tiny_gpt.pt",
                "base_sha256": self.base_sha256,
                "data_sha256": self.data_sha256,
                "targets": [TARGET],
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "training": {"steps": 3, "learning_rate": 0.01},
            },
        }
        self.adapter_path = self.directory / "tiny_gpt_lora_adapter.pt"
        torch.save(self.adapter_payload, self.adapter_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_checkpoint_payload(
        self,
        name: str,
        mutate: object,
    ) -> Path:
        payload = torch.load(self.base_path, map_location="cpu", weights_only=True)
        assert callable(mutate)
        mutate(payload)
        path = self.directory / name
        torch.save(payload, path)
        return path

    def _write_adapter_payload(
        self,
        name: str,
        mutate: object,
    ) -> Path:
        payload = copy.deepcopy(self.adapter_payload)
        assert callable(mutate)
        mutate(payload)
        path = self.directory / name
        torch.save(payload, path)
        return path

    def test_checkpoint_bundle_round_trip_and_expected_data_binding(self) -> None:
        bundle = load_tiny_gpt_checkpoint(
            self.base_path,
            expected_data_sha256=self.data_sha256,
        )
        with torch.no_grad():
            actual_logits, _ = bundle.model(self.input_ids)

        self.assertEqual(bundle.format_version, TINY_GPT_CHECKPOINT_FORMAT_VERSION)
        self.assertEqual(bundle.step, 17)
        self.assertEqual(bundle.config, self.config)
        self.assertEqual(bundle.tokenizer.state_dict(), self.tokenizer.state_dict())
        self.assertEqual(bundle.data_sha256, self.data_sha256)
        self.assertEqual(bundle.checkpoint_sha256, self.base_sha256)
        self.assertFalse(bundle.model.training)
        self.assertEqual(
            bundle.model.lm_head.weight.data_ptr(),
            bundle.model.token_embedding.weight.data_ptr(),
        )
        torch.testing.assert_close(actual_logits, self.base_logits, rtol=0.0, atol=0.0)

        changed = dict(self.data_sha256)
        changed["data/tiny_corpus.txt"] = "0" * 64
        with self.assertRaisesRegex(ArtifactValidationError, "fingerprints"):
            load_tiny_gpt_checkpoint(
                self.base_path,
                expected_data_sha256=changed,
            )

    def test_checkpoint_schema_rejects_corruption_before_returning_a_model(self) -> None:
        cases = {
            "format": lambda payload: payload.__setitem__("format_version", 99),
            "config": lambda payload: payload["metadata"]["config"].__setitem__(
                "unexpected", 1
            ),
            "tokenizer": lambda payload: payload["metadata"]["tokenizer"].__setitem__(
                "characters", ["b", "a", "c", "d", "e"]
            ),
            "data_sha": lambda payload: payload["metadata"]["data_sha256"].__setitem__(
                "data/tiny_corpus.txt", "not-a-digest"
            ),
            "state_keys": lambda payload: payload["model_state"].pop(
                "position_embedding.weight"
            ),
            "state_shape": lambda payload: payload["model_state"].__setitem__(
                "position_embedding.weight", torch.zeros(1)
            ),
            "state_dtype": lambda payload: payload["model_state"].__setitem__(
                "position_embedding.weight",
                payload["model_state"]["position_embedding.weight"].double(),
            ),
            "state_nan": lambda payload: payload["model_state"][
                "position_embedding.weight"
            ].fill_(float("nan")),
            "tied_state": lambda payload: payload["model_state"].__setitem__(
                "lm_head.weight",
                payload["model_state"]["lm_head.weight"].clone().add_(1.0),
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                path = self._write_checkpoint_payload(f"bad-checkpoint-{name}.pt", mutate)
                with self.assertRaises(ArtifactValidationError):
                    load_tiny_gpt_checkpoint(path)

    def test_lora_bundle_reproduces_adapter_logits(self) -> None:
        bundle = load_lora_model(self.base_path, self.adapter_path)
        with torch.no_grad():
            actual_logits, _ = bundle.model(self.input_ids)

        self.assertEqual(bundle.adapter_format_version, LORA_ADAPTER_FORMAT_VERSION)
        self.assertEqual(bundle.adapter_path, self.adapter_path.resolve())
        self.assertEqual(
            bundle.adapter_sha256,
            hashlib.sha256(self.adapter_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(bundle.adapter_metadata["rank"], 2)
        self.assertIsInstance(bundle.model.get_submodule(TARGET), LoRALinear)
        self.assertFalse(bundle.model.training)
        torch.testing.assert_close(actual_logits, self.adapter_logits, rtol=0.0, atol=0.0)

    def test_lora_metadata_and_state_are_strict_and_fail_atomically(self) -> None:
        sentinel = load_tiny_gpt_checkpoint(self.base_path)
        sentinel_state = {
            name: value.detach().clone()
            for name, value in sentinel.model.state_dict().items()
        }

        def change_data(payload: dict[str, object]) -> None:
            metadata = payload["metadata"]
            assert isinstance(metadata, dict)
            data_sha256 = metadata["data_sha256"]
            assert isinstance(data_sha256, dict)
            data_sha256["data/tiny_corpus.txt"] = "0" * 64

        def remove_state_key(payload: dict[str, object]) -> None:
            state = payload["adapter_state"]
            assert isinstance(state, dict)
            state.pop(f"{TARGET}.lora_A")

        def change_state_shape(payload: dict[str, object]) -> None:
            state = payload["adapter_state"]
            assert isinstance(state, dict)
            state[f"{TARGET}.lora_A"] = torch.zeros(1)

        def change_state_dtype(payload: dict[str, object]) -> None:
            state = payload["adapter_state"]
            assert isinstance(state, dict)
            value = state[f"{TARGET}.lora_A"]
            assert isinstance(value, torch.Tensor)
            state[f"{TARGET}.lora_A"] = value.double()

        cases = {
            "format": lambda payload: payload.__setitem__("format_version", 1),
            "base_sha": lambda payload: payload["metadata"].__setitem__(
                "base_sha256", "0" * 64
            ),
            "data_sha": change_data,
            "config": lambda payload: payload["metadata"]["config"].__setitem__(
                "block_size", 4
            ),
            "tokenizer": lambda payload: payload["metadata"]["tokenizer"].__setitem__(
                "add_unk", True
            ),
            "targets": lambda payload: payload["metadata"].__setitem__(
                "targets", ["blocks.0.attention.missing"]
            ),
            "duplicate_targets": lambda payload: payload["metadata"].__setitem__(
                "targets", [TARGET, TARGET]
            ),
            "rank": lambda payload: payload["metadata"].__setitem__("rank", 0),
            "alpha": lambda payload: payload["metadata"].__setitem__(
                "alpha", float("nan")
            ),
            "dropout": lambda payload: payload["metadata"].__setitem__(
                "dropout", 1.0
            ),
            "state_keys": remove_state_key,
            "state_shape": change_state_shape,
            "state_dtype": change_state_dtype,
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                path = self._write_adapter_payload(f"bad-adapter-{name}.pt", mutate)
                with self.assertRaises(ArtifactValidationError):
                    load_lora_model(self.base_path, path)

                # The API never accepts a caller-owned model, so a failed load
                # cannot freeze or inject into this already returned bundle.
                self.assertIsInstance(sentinel.model.get_submodule(TARGET), nn.Linear)
                for parameter_name, expected in sentinel_state.items():
                    self.assertTrue(
                        torch.equal(
                            sentinel.model.state_dict()[parameter_name],
                            expected,
                        )
                    )


class GenerateArtifactCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        experiments_directory = PROJECT_ROOT / "experiments"
        sys.path.insert(0, str(experiments_directory))
        path = experiments_directory / "07_generate.py"
        spec = importlib.util.spec_from_file_location("learn_llm_experiment_07", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.experiment = module

    def test_default_checkpoint_and_explicit_adapter_modes(self) -> None:
        default_args = self.experiment.parse_args([])
        self.assertIsNone(default_args.checkpoint)
        self.assertIsNone(default_args.adapter)

        adapter_args = self.experiment.parse_args(
            [
                "--base-checkpoint",
                "base.pt",
                "--adapter",
                "adapter.pt",
            ]
        )
        self.assertIsNone(adapter_args.checkpoint)
        self.assertEqual(adapter_args.base_checkpoint, Path("base.pt"))
        self.assertEqual(adapter_args.adapter, Path("adapter.pt"))

    def test_cli_rejects_ambiguous_or_incomplete_artifact_modes(self) -> None:
        invalid = (
            ["--checkpoint", "full.pt", "--base-checkpoint", "base.pt", "--adapter", "a.pt"],
            ["--base-checkpoint", "base.pt"],
            ["--adapter", "adapter.pt"],
            ["--tokens", "-1"],
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self.experiment.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
