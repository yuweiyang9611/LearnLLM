"""Versioned, strict loaders for TinyGPT checkpoints and LoRA adapters.

The low-level helpers in :mod:`learn_llm.training` intentionally accept an
already constructed model.  This module is the artifact boundary: it rebuilds
the recorded config and tokenizer, validates every field needed for inference,
and only then returns a newly constructed model.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .lora import inject_lora, load_lora_adapter_state_dict
from .model import TinyGPT, TinyGPTConfig
from .tokenizer import CharTokenizer


TINY_GPT_CHECKPOINT_FORMAT_VERSION = 2
LORA_ADAPTER_FORMAT_VERSION = 2

_CONFIG_KEYS = {
    "vocab_size",
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "bias",
}
_TOKENIZER_KEYS = {"characters", "add_unk", "add_eos"}
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


class ArtifactValidationError(ValueError):
    """Raised when an artifact is readable but violates its versioned schema."""


@dataclass(frozen=True, slots=True)
class TinyGPTBundle:
    """A model together with the recorded information needed to use it safely."""

    model: TinyGPT
    config: TinyGPTConfig
    tokenizer: CharTokenizer
    checkpoint_path: Path
    checkpoint_sha256: str
    format_version: int
    step: int
    metadata: dict[str, object]
    data_sha256: dict[str, str]
    adapter_path: Path | None = None
    adapter_sha256: str | None = None
    adapter_format_version: int | None = None
    adapter_metadata: dict[str, object] | None = None


def _read_payload(
    path: str | Path,
    *,
    map_location: torch.device | str,
    artifact_name: str,
) -> tuple[Path, Mapping[str, Any], str]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"{artifact_name} does not exist: {source}")
    source = source.resolve()
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    try:
        payload = torch.load(
            io.BytesIO(content),
            map_location=map_location,
            weights_only=True,
        )
    except Exception as error:
        raise ArtifactValidationError(
            f"cannot read {artifact_name} as a weights-only PyTorch artifact: {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError(f"{artifact_name} payload must be a mapping")
    return source, payload, digest


def _require_format_version(
    payload: Mapping[str, Any],
    *,
    expected: int,
    artifact_name: str,
) -> int:
    version = payload.get("format_version")
    if type(version) is not int or version != expected:
        raise ArtifactValidationError(
            f"unsupported {artifact_name} format_version: {version!r}; "
            f"expected {expected}. 重新训练: python experiments/06_train_tiny_gpt.py --quick; "
            "python experiments/10_sft_tiny_gpt.py --quick"
        )
    return version


def _require_metadata(
    payload: Mapping[str, Any], *, artifact_name: str
) -> dict[str, object]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or not all(
        isinstance(key, str) for key in metadata
    ):
        raise ArtifactValidationError(f"{artifact_name} metadata must be a string-keyed mapping")
    return dict(metadata)


def _require_config(
    metadata: Mapping[str, object], *, artifact_name: str
) -> TinyGPTConfig:
    raw = metadata.get("config")
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ArtifactValidationError(f"{artifact_name} metadata.config must be a mapping")
    supplied_keys = set(raw)
    if supplied_keys != _CONFIG_KEYS:
        missing = sorted(_CONFIG_KEYS - supplied_keys)
        unexpected = sorted(supplied_keys - _CONFIG_KEYS)
        raise ArtifactValidationError(
            f"{artifact_name} config keys do not match version 2; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
        if type(raw[name]) is not int:
            raise ArtifactValidationError(f"{artifact_name} config.{name} must be an integer")
    dropout = raw["dropout"]
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(float(dropout))
    ):
        raise ArtifactValidationError(f"{artifact_name} config.dropout must be finite")
    if type(raw["bias"]) is not bool:
        raise ArtifactValidationError(f"{artifact_name} config.bias must be a boolean")
    try:
        config = TinyGPTConfig(**dict(raw))
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(f"invalid {artifact_name} TinyGPT config") from error
    return config


def _require_tokenizer(
    metadata: Mapping[str, object], *, artifact_name: str
) -> CharTokenizer:
    raw = metadata.get("tokenizer")
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ArtifactValidationError(f"{artifact_name} metadata.tokenizer must be a mapping")
    supplied_keys = set(raw)
    if supplied_keys != _TOKENIZER_KEYS:
        missing = sorted(_TOKENIZER_KEYS - supplied_keys)
        unexpected = sorted(supplied_keys - _TOKENIZER_KEYS)
        raise ArtifactValidationError(
            f"{artifact_name} tokenizer keys do not match version 2; "
            f"missing={missing}, unexpected={unexpected}"
        )
    characters = raw["characters"]
    if not isinstance(characters, list) or not all(
        isinstance(character, str) and len(character) == 1
        for character in characters
    ):
        raise ArtifactValidationError(
            f"{artifact_name} tokenizer.characters must be a list of characters"
        )
    if type(raw["add_unk"]) is not bool:
        raise ArtifactValidationError(f"{artifact_name} tokenizer.add_unk must be a boolean")
    if raw["add_eos"] is not True:
        raise ArtifactValidationError("version 2 requires tokenizer.add_eos=true; rerun experiments 06 then 10")
    try:
        tokenizer = CharTokenizer.from_state_dict(dict(raw))
    except ValueError as error:
        raise ArtifactValidationError(f"invalid {artifact_name} tokenizer state") from error
    if tokenizer.state_dict() != dict(raw):
        raise ArtifactValidationError(
            f"{artifact_name} tokenizer state is not canonical or contains duplicates"
        )
    return tokenizer


def _normalize_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ArtifactValidationError(f"{field_name} must be a 64-character SHA-256")
    return value.lower()


def _require_data_sha256(
    metadata_or_fingerprints: Mapping[str, object] | Mapping[str, str],
    *,
    artifact_name: str,
    nested: bool = True,
) -> dict[str, str]:
    raw: object = (
        metadata_or_fingerprints.get("data_sha256")
        if nested
        else metadata_or_fingerprints
    )
    if not isinstance(raw, Mapping) or not raw:
        raise ArtifactValidationError(
            f"{artifact_name} data_sha256 must be a non-empty mapping"
        )
    normalized: dict[str, str] = {}
    for name, digest in raw.items():
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError(
                f"{artifact_name} data_sha256 keys must be non-empty strings"
            )
        normalized[name] = _normalize_sha256(
            digest,
            field_name=f"{artifact_name} data_sha256[{name!r}]",
        )
    return normalized


def _require_matching_expected_data(
    actual: Mapping[str, str],
    expected: Mapping[str, str] | None,
    *,
    artifact_name: str,
) -> None:
    if expected is None:
        return
    normalized_expected = _require_data_sha256(
        expected,
        artifact_name="expected",
        nested=False,
    )
    if dict(actual) != normalized_expected:
        missing = sorted(set(normalized_expected) - set(actual))
        unexpected = sorted(set(actual) - set(normalized_expected))
        changed = sorted(
            name
            for name in set(actual) & set(normalized_expected)
            if actual[name] != normalized_expected[name]
        )
        raise ArtifactValidationError(
            f"{artifact_name} data fingerprints do not match expected values; "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )


def _require_step(payload: Mapping[str, Any], *, artifact_name: str) -> int:
    step = payload.get("step", 0)
    if type(step) is not int or step < 0:
        raise ArtifactValidationError(f"{artifact_name} step must be a non-negative integer")
    return step


def _require_tensor_state(
    payload: Mapping[str, Any],
    *,
    key: str,
    artifact_name: str,
) -> dict[str, Tensor]:
    raw = payload.get(key)
    if not isinstance(raw, Mapping) or not raw:
        raise ArtifactValidationError(f"{artifact_name} {key} must be a non-empty mapping")
    if not all(isinstance(name, str) for name in raw):
        raise ArtifactValidationError(f"{artifact_name} {key} keys must be strings")
    state: dict[str, Tensor] = {}
    for name, value in raw.items():
        if not isinstance(value, Tensor):
            raise ArtifactValidationError(f"{artifact_name} {key}[{name!r}] must be a Tensor")
        state[name] = value
    return state


def _validate_model_state(model: TinyGPT, state: Mapping[str, Tensor]) -> None:
    expected = model.state_dict()
    expected_keys = set(expected)
    supplied_keys = set(state)
    if expected_keys != supplied_keys:
        missing = sorted(expected_keys - supplied_keys)
        unexpected = sorted(supplied_keys - expected_keys)
        raise ArtifactValidationError(
            f"checkpoint model_state keys do not match TinyGPT; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(expected_keys):
        value = state[name]
        target = expected[name]
        if value.shape != target.shape:
            raise ArtifactValidationError(
                f"checkpoint tensor shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(target.shape)}"
            )
        if value.dtype != target.dtype:
            raise ArtifactValidationError(
                f"checkpoint tensor dtype mismatch for {name}: "
                f"{value.dtype} != {target.dtype}"
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ArtifactValidationError(f"checkpoint tensor contains NaN/Inf: {name}")
    if not torch.equal(
        state["token_embedding.weight"],
        state["lm_head.weight"],
    ):
        raise ArtifactValidationError(
            "checkpoint tied token_embedding/lm_head tensors disagree"
        )


def load_tiny_gpt_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
    expected_data_sha256: Mapping[str, str] | None = None,
) -> TinyGPTBundle:
    """Load and strictly validate a version-1 TinyGPT checkpoint.

    The returned model is newly constructed and in evaluation mode.  Therefore
    a validation failure cannot partially mutate a caller-owned model.
    """

    source, payload, digest = _read_payload(
        path,
        map_location=map_location,
        artifact_name="TinyGPT checkpoint",
    )
    version = _require_format_version(
        payload,
        expected=TINY_GPT_CHECKPOINT_FORMAT_VERSION,
        artifact_name="TinyGPT checkpoint",
    )
    metadata = _require_metadata(payload, artifact_name="TinyGPT checkpoint")
    config = _require_config(metadata, artifact_name="TinyGPT checkpoint")
    tokenizer = _require_tokenizer(metadata, artifact_name="TinyGPT checkpoint")
    if config.vocab_size != tokenizer.vocab_size:
        raise ArtifactValidationError(
            "TinyGPT checkpoint config.vocab_size does not match tokenizer.vocab_size"
        )
    data_sha256 = _require_data_sha256(
        metadata,
        artifact_name="TinyGPT checkpoint",
    )
    _require_matching_expected_data(
        data_sha256,
        expected_data_sha256,
        artifact_name="TinyGPT checkpoint",
    )
    step = _require_step(payload, artifact_name="TinyGPT checkpoint")
    model_state = _require_tensor_state(
        payload,
        key="model_state",
        artifact_name="TinyGPT checkpoint",
    )

    device = torch.device(map_location)
    model = TinyGPT(config).to(device)
    _validate_model_state(model, model_state)
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise ArtifactValidationError("TinyGPT checkpoint strict load failed") from error
    if model.lm_head.weight.data_ptr() != model.token_embedding.weight.data_ptr():
        raise ArtifactValidationError(
            "TinyGPT checkpoint load broke embedding/lm_head weight tying"
        )
    model.eval()
    return TinyGPTBundle(
        model=model,
        config=config,
        tokenizer=tokenizer,
        checkpoint_path=source,
        checkpoint_sha256=digest,
        format_version=version,
        step=step,
        metadata=metadata,
        data_sha256=data_sha256,
    )


def _require_finite_number(
    value: object,
    *,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ArtifactValidationError(f"{field_name} must be a finite number")
    return float(value)


def _require_adapter_metadata(
    metadata: Mapping[str, object],
    *,
    base: TinyGPTBundle,
) -> tuple[tuple[str, ...], int, float, float, dict[str, str]]:
    adapter_config = _require_config(metadata, artifact_name="LoRA adapter")
    if adapter_config != base.config:
        raise ArtifactValidationError("LoRA adapter config does not match base checkpoint")
    adapter_tokenizer = _require_tokenizer(metadata, artifact_name="LoRA adapter")
    if adapter_tokenizer.state_dict() != base.tokenizer.state_dict():
        raise ArtifactValidationError("LoRA adapter tokenizer does not match base checkpoint")

    recorded_base_sha256 = _normalize_sha256(
        metadata.get("base_sha256"),
        field_name="LoRA adapter base_sha256",
    )
    if recorded_base_sha256 != base.checkpoint_sha256:
        raise ArtifactValidationError("LoRA adapter base SHA-256 does not match checkpoint")

    data_sha256 = _require_data_sha256(metadata, artifact_name="LoRA adapter")
    if data_sha256 != base.data_sha256:
        raise ArtifactValidationError(
            "LoRA adapter data fingerprints do not match base checkpoint"
        )

    raw_targets = metadata.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets or not all(
        isinstance(name, str) and name for name in raw_targets
    ):
        raise ArtifactValidationError(
            "LoRA adapter targets must be a non-empty list of module names"
        )
    targets = tuple(raw_targets)
    if len(set(targets)) != len(targets):
        raise ArtifactValidationError("LoRA adapter targets must be unique")
    for name in targets:
        try:
            target = base.model.get_submodule(name)
        except (AttributeError, KeyError) as error:
            raise ArtifactValidationError(f"unknown LoRA adapter target: {name}") from error
        if not isinstance(target, nn.Linear):
            raise ArtifactValidationError(f"LoRA adapter target is not Linear: {name}")

    rank = metadata.get("rank")
    if type(rank) is not int or rank <= 0:
        raise ArtifactValidationError("LoRA adapter rank must be a positive integer")
    alpha = _require_finite_number(
        metadata.get("alpha"),
        field_name="LoRA adapter alpha",
    )
    if alpha <= 0:
        raise ArtifactValidationError("LoRA adapter alpha must be positive")
    dropout = _require_finite_number(
        metadata.get("dropout"),
        field_name="LoRA adapter dropout",
    )
    if not 0.0 <= dropout < 1.0:
        raise ArtifactValidationError("LoRA adapter dropout must be in [0, 1)")
    return targets, rank, alpha, dropout, data_sha256


def _validate_adapter_state(
    base: TinyGPT,
    state: Mapping[str, Tensor],
    *,
    targets: tuple[str, ...],
    rank: int,
) -> None:
    expected: dict[str, tuple[torch.Size, torch.dtype, torch.device]] = {}
    for name in targets:
        layer = base.get_submodule(name)
        assert isinstance(layer, nn.Linear)
        expected[f"{name}.lora_A"] = (
            torch.Size((rank, layer.in_features)),
            layer.weight.dtype,
            layer.weight.device,
        )
        expected[f"{name}.lora_B"] = (
            torch.Size((layer.out_features, rank)),
            layer.weight.dtype,
            layer.weight.device,
        )
    expected_keys = set(expected)
    supplied_keys = set(state)
    if expected_keys != supplied_keys:
        missing = sorted(expected_keys - supplied_keys)
        unexpected = sorted(supplied_keys - expected_keys)
        raise ArtifactValidationError(
            f"LoRA adapter state keys do not match metadata; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(expected_keys):
        shape, dtype, device = expected[name]
        value = state[name]
        if value.shape != shape:
            raise ArtifactValidationError(
                f"LoRA adapter tensor shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(shape)}"
            )
        if value.dtype != dtype:
            raise ArtifactValidationError(
                f"LoRA adapter tensor dtype mismatch for {name}: "
                f"{value.dtype} != {dtype}"
            )
        if value.device != device:
            raise ArtifactValidationError(
                f"LoRA adapter tensor device mismatch for {name}: "
                f"{value.device} != {device}"
            )
        if not value.is_floating_point():
            raise ArtifactValidationError(f"LoRA adapter tensor must be floating point: {name}")
        if not bool(torch.isfinite(value).all().item()):
            raise ArtifactValidationError(f"LoRA adapter tensor contains NaN/Inf: {name}")


def load_lora_model(
    base_checkpoint: str | Path,
    adapter: str | Path,
    *,
    map_location: torch.device | str = "cpu",
    expected_data_sha256: Mapping[str, str] | None = None,
) -> TinyGPTBundle:
    """Load a base checkpoint and a compatible version-1 LoRA adapter.

    All adapter metadata and tensors are validated before the newly loaded base
    model is adapted.  No caller-owned model is accepted, which makes failures
    atomic from the caller's perspective.
    """

    base = load_tiny_gpt_checkpoint(
        base_checkpoint,
        map_location=map_location,
        expected_data_sha256=expected_data_sha256,
    )
    adapter_path, payload, adapter_sha256 = _read_payload(
        adapter,
        map_location=map_location,
        artifact_name="LoRA adapter",
    )
    adapter_version = _require_format_version(
        payload,
        expected=LORA_ADAPTER_FORMAT_VERSION,
        artifact_name="LoRA adapter",
    )
    adapter_metadata = _require_metadata(payload, artifact_name="LoRA adapter")
    targets, rank, alpha, dropout, _ = _require_adapter_metadata(
        adapter_metadata,
        base=base,
    )
    adapter_state = _require_tensor_state(
        payload,
        key="adapter_state",
        artifact_name="LoRA adapter",
    )
    _validate_adapter_state(
        base.model,
        adapter_state,
        targets=targets,
        rank=rank,
    )

    inject_lora(
        base.model,
        targets,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    load_lora_adapter_state_dict(base.model, adapter_state)
    base.model.eval()
    return TinyGPTBundle(
        model=base.model,
        config=base.config,
        tokenizer=base.tokenizer,
        checkpoint_path=base.checkpoint_path,
        checkpoint_sha256=base.checkpoint_sha256,
        format_version=base.format_version,
        step=base.step,
        metadata=base.metadata,
        data_sha256=base.data_sha256,
        adapter_path=adapter_path,
        adapter_sha256=adapter_sha256,
        adapter_format_version=adapter_version,
        adapter_metadata=adapter_metadata,
    )
