"""Small supervised-fine-tuning helpers for the TinyGPT lessons.

The important detail in SFT is not merely concatenating a prompt and an
answer.  A causal language model predicts token ``t + 1`` from position ``t``;
therefore the labels must be shifted, and only predictions whose target is an
assistant token should contribute to the loss.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from .tokenizer import CharTokenizer


IGNORE_INDEX = -100


class CausalLanguageModel(Protocol):
    """The small part of ``TinyGPT`` used by these helpers."""

    training: bool

    def __call__(
        self, input_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]: ...

    def train(self, mode: bool = True) -> CausalLanguageModel: ...

    def eval(self) -> CausalLanguageModel: ...


@dataclass(frozen=True, slots=True)
class SFTSequence:
    """One tokenized, shifted instruction/answer training sequence.

    ``prompt_token_count`` and ``response_token_count`` are token counts, not
    source-text character counts.  For the teaching ``CharTokenizer`` they are
    often numerically equal before adding EOS; response_token_count includes
    the EOS target when enabled. The implementation deliberately derives them
    from ``encode`` so the masking rule transfers to subword tokenizers.
    """

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_token_count: int
    response_token_count: int

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    @property
    def first_assistant_label_index(self) -> int:
        """Index whose target is the first assistant token after shifting."""

        return self.prompt_token_count - 1


@dataclass(frozen=True, slots=True)
class SFTBatch:
    """A right-padded batch accepted directly by ``TinyGPT.forward``."""

    input_ids: Tensor
    labels: Tensor
    lengths: Tensor

    @property
    def supervised_token_count(self) -> int:
        return int((self.labels != IGNORE_INDEX).sum().item())


def format_instruction_prompt(instruction: str) -> str:
    """Format one instruction while leaving the assistant answer open."""

    cleaned = instruction.strip()
    if not cleaned:
        raise ValueError("instruction cannot be empty")
    return f"用户：{cleaned}\n助手："


def build_sft_sequence(
    tokenizer: CharTokenizer,
    instruction: str,
    response: str,
    *,
    max_sequence_length: int | None = None,
) -> SFTSequence:
    """Encode one pair and build assistant-only next-token labels.

    With EOS enabled the response includes a final EOS target. For the complete
    token sequence ``prompt + response``, TinyGPT receives
    all tokens except the last as inputs and all tokens except the first as
    next-token targets.  Targets before the first response token are replaced
    by ``IGNORE_INDEX``.  No silent truncation is performed because cutting an
    answer would make this first-principles experiment harder to reason about.
    """

    if not response:
        raise ValueError("response cannot be empty")
    if max_sequence_length is not None and max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")

    prompt_ids = tokenizer.encode(format_instruction_prompt(instruction))
    response_ids = tokenizer.encode(response)
    if not isinstance(prompt_ids, list) or not isinstance(response_ids, list):
        raise TypeError("tokenizer.encode must return token ID lists")

    if tokenizer.eos_id is not None:
        response_ids.append(tokenizer.eos_id)
    full_ids = prompt_ids + response_ids
    input_ids = full_ids[:-1]
    # labels[i] is the token predicted from input_ids[i].  The first answer
    # token is therefore supervised at index len(prompt_ids) - 1.
    labels = [IGNORE_INDEX] * (len(prompt_ids) - 1) + response_ids
    if len(input_ids) != len(labels):
        raise AssertionError("shifted inputs and labels must have equal length")
    if max_sequence_length is not None and len(input_ids) > max_sequence_length:
        raise ValueError(
            f"encoded sequence length {len(input_ids)} exceeds "
            f"max_sequence_length {max_sequence_length}"
        )

    return SFTSequence(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        prompt_token_count=len(prompt_ids),
        response_token_count=len(response_ids),
    )


def collate_sft_batch(
    sequences: list[SFTSequence] | tuple[SFTSequence, ...],
    *,
    pad_token_id: int,
    device: torch.device | str | None = None,
) -> SFTBatch:
    """Right-pad variable-length examples without supervising padding.

    TinyGPT does not yet expose an attention-padding mask.  Right padding is
    still correct here: causal attention prevents real tokens from seeing
    later pad positions, and every padded target stays at ``IGNORE_INDEX``.
    """

    if not sequences:
        raise ValueError("at least one SFT sequence is required")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative vocabulary ID")
    if any(sequence.sequence_length == 0 for sequence in sequences):
        raise ValueError("SFT sequences cannot be empty")

    max_length = max(sequence.sequence_length for sequence in sequences)
    inputs = torch.full(
        (len(sequences), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    labels = torch.full(
        (len(sequences), max_length),
        IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )
    lengths = torch.empty(len(sequences), dtype=torch.long, device=device)

    for row, sequence in enumerate(sequences):
        length = sequence.sequence_length
        inputs[row, :length] = torch.tensor(
            sequence.input_ids, dtype=torch.long, device=device
        )
        labels[row, :length] = torch.tensor(
            sequence.labels, dtype=torch.long, device=device
        )
        lengths[row] = length

    return SFTBatch(input_ids=inputs, labels=labels, lengths=lengths)


@torch.no_grad()
def evaluate_assistant_loss(model: CausalLanguageModel, batch: SFTBatch) -> float:
    """Return mean cross entropy over assistant targets only."""

    if batch.supervised_token_count == 0:
        raise ValueError("the batch contains no supervised assistant tokens")
    was_training = model.training
    model.eval()
    try:
        _, loss = model(batch.input_ids, batch.labels)
    finally:
        model.train(was_training)
    if loss is None:
        raise RuntimeError("the language model did not return a training loss")
    if not torch.isfinite(loss):
        raise RuntimeError("assistant loss is not finite")
    return float(loss.item())


def train_sft_steps(
    model: nn.Module,
    batch: SFTBatch,
    *,
    steps: int,
    learning_rate: float = 3e-3,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = 1.0,
    on_step: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Fine-tune model parameters on one small padded instruction batch."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if batch.supervised_token_count == 0:
        raise ValueError("the batch contains no supervised assistant tokens")
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=0.0,
        )

    model.train()
    losses: list[float] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(batch.input_ids, batch.labels)
        if loss is None:
            raise RuntimeError("the language model did not return a training loss")
        if not torch.isfinite(loss):
            raise RuntimeError("training loss is not finite")
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if on_step is not None:
            on_step(step + 1, losses[-1])
    return losses
