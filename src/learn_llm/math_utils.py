"""Numerically stable versions of the first equations used in language models."""

from __future__ import annotations

import torch
from torch import Tensor


def stable_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Compute softmax without overflowing on large positive logits.

    Subtracting the maximum does not change the result: every numerator and the
    denominator are multiplied by the same constant.  It does, however, make
    the largest exponent exactly ``exp(0) == 1``.
    """

    if not logits.is_floating_point():
        raise TypeError("stable_softmax expects a floating-point tensor")
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def cross_entropy_from_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    reduction: str = "mean",
    ignore_index: int | None = None,
) -> Tensor:
    """Cross entropy for class logits stored in the final dimension.

    ``targets`` must have shape ``logits.shape[:-1]``.  The implementation uses
    log-sum-exp instead of taking the logarithm of an already rounded softmax.
    This is the same stable identity used by production deep-learning libraries.
    """

    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    if tuple(targets.shape) != tuple(logits.shape[:-1]):
        raise ValueError(
            "targets must match every logits dimension except the final class "
            f"dimension; got logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
        )
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if targets.dtype != torch.long:
        targets = targets.long()

    # Center first rather than evaluating ``logits - logsumexp(logits)``
    # directly.  Both are algebraically equal, but the latter subtracts two
    # numbers near 10,000 in a common stress test and loses float32 precision.
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_probabilities = shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)

    if ignore_index is None:
        losses = -log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        valid = None
    else:
        valid = targets != ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        losses = -log_probabilities.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        losses = losses.masked_fill(~valid, 0.0)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if valid is None:
        return losses.mean()
    # Matching PyTorch's behavior, an all-ignored batch has an undefined mean.
    return losses.sum() / valid.sum()


# A short alias is convenient in model code and remains explicit at call sites.
cross_entropy = cross_entropy_from_logits
