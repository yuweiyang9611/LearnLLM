from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learn_llm.lora import (  # noqa: E402
    LoRALinear,
    inject_lora,
    load_lora_adapter_state_dict,
    lora_adapter_state_dict,
    lora_parameter_names,
)
from learn_llm.model import TinyGPT, TinyGPTConfig  # noqa: E402


TARGETS = (
    "blocks.0.attention.qkv",
    "blocks.0.mlp.network.0",
)


def build_model(*, dtype: torch.dtype = torch.float32) -> TinyGPT:
    torch.manual_seed(2026)
    model = TinyGPT(
        TinyGPTConfig(
            vocab_size=13,
            block_size=6,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
        )
    )
    return model.to(dtype=dtype)


class LoRAInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_injection_preserves_logits_and_only_adapters_are_trainable(self) -> None:
        model = build_model().eval()
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        with torch.no_grad():
            expected_logits, _ = model(input_ids)

        injected = inject_lora(
            model,
            TARGETS,
            rank=2,
            alpha=4.0,
            dropout=0.25,
        )
        with torch.no_grad():
            actual_logits, _ = model(input_ids)

        self.assertEqual(injected, TARGETS)
        for target in TARGETS:
            self.assertIsInstance(model.get_submodule(target), LoRALinear)
            self.assertFalse(model.get_submodule(target).training)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=0.0)

        expected_trainable = {
            f"{target}.lora_A" for target in TARGETS
        } | {
            f"{target}.lora_B" for target in TARGETS
        }
        self.assertEqual(set(lora_parameter_names(model)), expected_trainable)
        self.assertEqual(
            {
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
            expected_trainable,
        )

    def test_injected_parameters_inherit_base_dtype_and_device(self) -> None:
        model = build_model(dtype=torch.float64)
        inject_lora(model, (TARGETS[0],), rank=3, alpha=6.0)
        layer = model.get_submodule(TARGETS[0])

        self.assertIsInstance(layer, LoRALinear)
        assert isinstance(layer, LoRALinear)
        self.assertEqual(layer.lora_A.dtype, layer.base.weight.dtype)
        self.assertEqual(layer.lora_B.dtype, layer.base.weight.dtype)
        self.assertEqual(layer.lora_A.device, layer.base.weight.device)
        self.assertEqual(layer.lora_B.device, layer.base.weight.device)

        input_ids = torch.tensor([[1, 2, 3]])
        logits, _ = model(input_ids)
        self.assertEqual(logits.dtype, torch.float64)

    def test_adapter_training_never_changes_base_parameters(self) -> None:
        model = build_model()
        inject_lora(model, TARGETS, rank=2, alpha=4.0)
        base_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if not (name.endswith(".lora_A") or name.endswith(".lora_B"))
        }
        adapter_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=0.05,
            weight_decay=0.0,
        )
        input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        targets = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 4]])

        model.train()
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, targets)
            self.assertIsNotNone(loss)
            assert loss is not None
            loss.backward()
            optimizer.step()

        parameters = dict(model.named_parameters())
        for name, expected in base_before.items():
            self.assertTrue(
                torch.equal(parameters[name], expected),
                f"frozen base parameter changed: {name}",
            )
        self.assertTrue(
            any(
                not torch.equal(parameters[name], expected)
                for name, expected in adapter_before.items()
            ),
            "optimizer did not update any LoRA adapter parameter",
        )

    def test_invalid_targets_fail_atomically(self) -> None:
        cases = (
            ((TARGETS[0], "blocks.0.attention.missing"), ValueError),
            ((TARGETS[0], "blocks.0.attention_norm"), TypeError),
            ((TARGETS[0], TARGETS[0]), ValueError),
            ((), ValueError),
        )
        for targets, exception in cases:
            with self.subTest(targets=targets):
                model = build_model()
                parameters_before = {
                    name: parameter.requires_grad
                    for name, parameter in model.named_parameters()
                }
                with self.assertRaises(exception):
                    inject_lora(model, targets, rank=2, alpha=4.0)

                self.assertIsInstance(model.get_submodule(TARGETS[0]), nn.Linear)
                self.assertEqual(
                    {
                        name: parameter.requires_grad
                        for name, parameter in model.named_parameters()
                    },
                    parameters_before,
                )

        for invalid_options in (
            {"rank": 0, "alpha": 4.0},
            {"rank": 2, "alpha": 0.0},
            {"rank": 2, "alpha": 4.0, "dropout": 1.0},
        ):
            with self.subTest(options=invalid_options):
                model = build_model()
                parameters_before = {
                    name: parameter.requires_grad
                    for name, parameter in model.named_parameters()
                }
                with self.assertRaises(ValueError):
                    inject_lora(model, (TARGETS[0],), **invalid_options)
                self.assertIsInstance(model.get_submodule(TARGETS[0]), nn.Linear)
                self.assertEqual(
                    {
                        name: parameter.requires_grad
                        for name, parameter in model.named_parameters()
                    },
                    parameters_before,
                )


class LoRAAdapterStateTests(unittest.TestCase):
    def test_adapter_state_round_trip_reproduces_logits(self) -> None:
        source = build_model().eval()
        destination = build_model().eval()
        inject_lora(source, TARGETS, rank=2, alpha=4.0)
        inject_lora(destination, TARGETS, rank=2, alpha=4.0)

        with torch.no_grad():
            for index, parameter in enumerate(
                (
                    parameter
                    for parameter in source.parameters()
                    if parameter.requires_grad
                ),
                start=1,
            ):
                values = torch.arange(
                    parameter.numel(),
                    dtype=parameter.dtype,
                    device=parameter.device,
                ).reshape_as(parameter)
                parameter.copy_((values + index) / 100.0)

        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            expected_logits, _ = source(input_ids)
        adapter_state = lora_adapter_state_dict(source)
        self.assertTrue(adapter_state)
        self.assertTrue(
            all(value.device.type == "cpu" for value in adapter_state.values())
        )

        # The exported state must own its storage rather than aliasing the model.
        with torch.no_grad():
            for parameter in source.parameters():
                if parameter.requires_grad:
                    parameter.zero_()

        load_lora_adapter_state_dict(destination, adapter_state)
        with torch.no_grad():
            actual_logits, _ = destination(input_ids)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=0.0)

        destination_parameters = dict(destination.named_parameters())
        for name, expected in adapter_state.items():
            torch.testing.assert_close(
                destination_parameters[name], expected, rtol=0.0, atol=0.0
            )

    def test_adapter_loader_rejects_wrong_keys_shapes_and_values(self) -> None:
        model = build_model()
        inject_lora(model, (TARGETS[0],), rank=2, alpha=4.0)
        adapter_state = lora_adapter_state_dict(model)
        first_name = next(iter(adapter_state))

        missing = dict(adapter_state)
        missing.pop(first_name)
        with self.assertRaisesRegex(ValueError, "missing="):
            load_lora_adapter_state_dict(model, missing)

        unexpected = dict(adapter_state)
        unexpected["unexpected.lora_A"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unexpected="):
            load_lora_adapter_state_dict(model, unexpected)

        wrong_shape = dict(adapter_state)
        wrong_shape[first_name] = torch.zeros(1)
        snapshot = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            load_lora_adapter_state_dict(model, wrong_shape)
        for name, expected in snapshot.items():
            self.assertTrue(torch.equal(dict(model.named_parameters())[name], expected))

        wrong_type = dict(adapter_state)
        wrong_type[first_name] = "not a tensor"  # type: ignore[assignment]
        with self.assertRaisesRegex(TypeError, "not a Tensor"):
            load_lora_adapter_state_dict(model, wrong_type)


if __name__ == "__main__":
    unittest.main()
