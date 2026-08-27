"""A compact decoder-only Transformer (a tiny GPT) for CPU experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import scaled_dot_product_attention
from .math_utils import cross_entropy_from_logits


@dataclass(frozen=True, slots=True, init=False)
class TinyGPTConfig:
    """Architecture settings kept intentionally close to common GPT notation."""

    vocab_size: int
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True

    def __init__(
        self,
        vocab_size: int,
        block_size: int | None = None,
        n_layer: int | None = None,
        n_head: int | None = None,
        n_embd: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        *,
        max_seq_len: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        d_model: int | None = None,
    ) -> None:
        """Accept both compact GPT names and descriptive teaching aliases."""

        def resolve(
            compact_name: str,
            compact_value: int | None,
            descriptive_name: str,
            descriptive_value: int | None,
            default: int,
        ) -> int:
            if (
                compact_value is not None
                and descriptive_value is not None
                and compact_value != descriptive_value
            ):
                raise ValueError(
                    f"{compact_name} and {descriptive_name} describe the same "
                    "setting and cannot disagree"
                )
            return (
                compact_value
                if compact_value is not None
                else descriptive_value
                if descriptive_value is not None
                else default
            )

        object.__setattr__(self, "vocab_size", vocab_size)
        object.__setattr__(
            self,
            "block_size",
            resolve("block_size", block_size, "max_seq_len", max_seq_len, 128),
        )
        object.__setattr__(
            self,
            "n_layer",
            resolve("n_layer", n_layer, "num_layers", num_layers, 2),
        )
        object.__setattr__(
            self,
            "n_head",
            resolve("n_head", n_head, "num_heads", num_heads, 2),
        )
        object.__setattr__(
            self,
            "n_embd",
            resolve("n_embd", n_embd, "d_model", d_model, 64),
        )
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "bias", bias)

        if self.vocab_size <= 0 or self.block_size <= 0:
            raise ValueError("vocab_size and block_size must be positive")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def max_seq_len(self) -> int:
        return self.block_size

    @property
    def num_layers(self) -> int:
        return self.n_layer

    @property
    def num_heads(self) -> int:
        return self.n_head

    @property
    def d_model(self) -> int:
        return self.n_embd


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attention_dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, embedding_size = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size, sequence_length, self.n_head, self.head_size
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        attended, _ = scaled_dot_product_attention(
            query,
            key,
            value,
            causal=True,
            dropout_p=self.attention_dropout,
            training=self.training,
        )
        joined = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_size
        )
        return self.residual_dropout(self.output(joined))


class FeedForward(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """Pre-normalization Transformer block with two residual pathways."""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TinyGPT(nn.Module):
    """A decoder-only language model small enough to train in a unit test."""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.final_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        # Weight tying both reduces parameters and mirrors many full-sized LLMs.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return token logits and, when targets are supplied, next-token loss."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block_size "
                f"{self.config.block_size}"
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")

        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = cross_entropy_from_logits(logits, targets, ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively append tokens using sampling or greedy decoding."""

        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty (batch, sequence) tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        was_training = self.training
        self.eval()
        generated = input_ids
        try:
            for _ in range(max_new_tokens):
                context = generated[:, -self.config.block_size :]
                logits, _ = self(context)
                next_logits = logits[:, -1, :]

                if not do_sample or temperature == 0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    if top_k is not None:
                        k = min(max(int(top_k), 1), next_logits.shape[-1])
                        threshold = torch.topk(next_logits, k).values[:, [-1]]
                        next_logits = next_logits.masked_fill(
                            next_logits < threshold, float("-inf")
                        )
                    probabilities = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(
                        probabilities, num_samples=1, generator=generator
                    )
                generated = torch.cat((generated, next_token), dim=1)
        finally:
            self.train(was_training)
        return generated

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        parameters = (
            parameter for parameter in self.parameters() if not trainable_only or parameter.requires_grad
        )
        return sum(parameter.numel() for parameter in parameters)
