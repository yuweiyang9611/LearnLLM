"""A minimal Low-Rank Adaptation (LoRA) linear layer."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable rank-``r`` update.

    The update is ``(x A^T) B^T * alpha/r``.  ``B`` starts at zero so wrapping a
    pretrained layer leaves its output exactly unchanged before fine-tuning.
    """

    def __init__(
        self,
        in_features: int | nn.Linear,
        out_features: int | None = None,
        *,
        rank: int = 4,
        alpha: float = 1.0,
        bias: bool = True,
        dropout: float = 0.0,
        base_layer: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        if isinstance(in_features, nn.Linear):
            if out_features is not None or base_layer is not None:
                raise ValueError(
                    "when the first argument is a Linear layer, do not also "
                    "supply out_features or base_layer"
                )
            base_layer = in_features
            input_size = base_layer.in_features
            output_size = base_layer.out_features
        else:
            if out_features is None:
                raise ValueError("out_features is required when constructing a new base layer")
            input_size = in_features
            output_size = out_features

        if base_layer is None:
            base_layer = nn.Linear(input_size, output_size, bias=bias)
        elif (
            base_layer.in_features != input_size
            or base_layer.out_features != output_size
        ):
            raise ValueError("base_layer dimensions do not match in/out_features")
        self.base = base_layer
        self.base.requires_grad_(False)

        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(
            self.base.weight.new_empty((rank, input_size))
        )
        self.lora_B = nn.Parameter(
            self.base.weight.new_zeros((output_size, rank))
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        return cls(
            layer,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base(x)
        low_rank = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_output + low_rank * self.scaling

    def merged_weight(self) -> Tensor:
        """Return the effective weight after folding in the low-rank update."""

        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def inject_lora(
    module: nn.Module,
    target_names: Iterable[str],
    *,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> tuple[str, ...]:
    """Freeze ``module`` and replace exact Linear targets with LoRA layers.

    Targets use ``named_modules()`` paths such as
    ``blocks.0.attention.qkv``.  Every target is validated before the model is
    changed so a typo cannot leave a partially adapted model behind.
    """

    names = tuple(target_names)
    if not names:
        raise ValueError("at least one LoRA target is required")
    if len(set(names)) != len(names):
        raise ValueError("LoRA target names must be unique")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    resolved: list[tuple[nn.Module, str, nn.Linear]] = []
    for full_name in names:
        if not isinstance(full_name, str):
            raise TypeError("LoRA target names must be strings")
        parent_name, separator, child_name = full_name.rpartition(".")
        if not child_name:
            raise ValueError(f"invalid LoRA target name: {full_name!r}")
        try:
            parent = module.get_submodule(parent_name) if separator else module
            child = getattr(parent, child_name)
        except (AttributeError, KeyError) as error:
            raise ValueError(f"unknown LoRA target: {full_name}") from error
        if not isinstance(child, nn.Linear):
            raise TypeError(f"LoRA target is not Linear: {full_name}")
        resolved.append((parent, child_name, child))

    module.requires_grad_(False)
    for parent, child_name, child in resolved:
        wrapper = LoRALinear.from_linear(
            child,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        wrapper.train(child.training)
        setattr(
            parent,
            child_name,
            wrapper,
        )
    return names


def lora_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Return the trainable adapter parameter names in deterministic order."""

    return tuple(
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )


def lora_adapter_state_dict(module: nn.Module) -> dict[str, Tensor]:
    """Return a CPU-only state dictionary containing LoRA A/B tensors only."""

    adapter_names = lora_parameter_names(module)
    if not adapter_names:
        raise ValueError("model does not contain trainable LoRA parameters")
    invalid = [
        name
        for name in adapter_names
        if not (name.endswith(".lora_A") or name.endswith(".lora_B"))
    ]
    if invalid:
        raise ValueError(f"non-LoRA parameters are trainable: {invalid}")
    parameters = dict(module.named_parameters())
    return {
        name: parameters[name].detach().cpu().clone()
        for name in adapter_names
    }


def load_lora_adapter_state_dict(
    module: nn.Module,
    adapter_state: Mapping[str, Tensor],
) -> None:
    """Strictly restore adapter tensors into an already adapted model."""

    expected = set(lora_parameter_names(module))
    supplied = set(adapter_state)
    if expected != supplied:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(
            f"LoRA adapter keys do not match; missing={missing}, "
            f"unexpected={unexpected}"
        )
    parameters = dict(module.named_parameters())
    for name in sorted(expected):
        value = adapter_state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"adapter value is not a Tensor: {name}")
        if value.shape != parameters[name].shape:
            raise ValueError(
                f"adapter shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(parameters[name].shape)}"
            )
    with torch.no_grad():
        for name in sorted(expected):
            value = adapter_state[name]
            parameters[name].copy_(value)
