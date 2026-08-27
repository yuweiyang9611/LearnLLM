"""A minimal Low-Rank Adaptation (LoRA) linear layer."""

from __future__ import annotations

import math

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
        self.lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.lora_B = nn.Parameter(torch.zeros(output_size, rank))
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
