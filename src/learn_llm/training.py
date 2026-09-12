"""Tiny training-loop and checkpoint helpers shared by the experiments."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch for repeatable educational experiments."""

    random.seed(seed)
    torch.manual_seed(seed)


def sample_language_model_batch(
    token_ids: Tensor,
    *,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample input/next-token windows from a one-dimensional token stream."""

    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    if block_size <= 0 or batch_size <= 0:
        raise ValueError("block_size and batch_size must be positive")
    if token_ids.numel() <= block_size:
        raise ValueError("token stream must contain more than block_size tokens")
    starts = torch.randint(
        token_ids.numel() - block_size,
        (batch_size,),
        generator=generator,
    )
    inputs = torch.stack([token_ids[start : start + block_size] for start in starts])
    targets = torch.stack(
        [token_ids[start + 1 : start + block_size + 1] for start in starts]
    )
    if device is not None:
        inputs = inputs.to(device)
        targets = targets.to(device)
    return inputs, targets


def train_steps(
    model: nn.Module,
    input_ids: Tensor,
    targets: Tensor,
    *,
    steps: int = 50,
    learning_rate: float = 3e-3,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = 1.0,
) -> list[float]:
    """Optimize a fixed tiny batch and return one scalar loss per step."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("the model did not return a loss when targets were supplied")
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save model/optimizer state in a portable state-dictionary checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "metadata": dict(metadata or {}),
    }
    torch.save(checkpoint, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint and return its step, metadata, and key diagnostics."""

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=strict)
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return {
        "step": int(checkpoint.get("step", 0)),
        "metadata": checkpoint.get("metadata", {}),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
