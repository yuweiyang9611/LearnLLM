"""Readable scaled dot-product attention with explicit masking."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    causal: bool = False,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return ``(context, attention_weights)`` for tensors shaped ``(..., T, D)``.

    A boolean ``attn_mask`` uses ``True`` for permitted query/key pairs.  A
    floating-point mask is added to the scores, which is useful for precomputed
    ``0 / -inf`` masks.  Leading batch/head dimensions follow broadcasting rules.
    """

    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must each have at least two dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")

    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(query.shape[-1])

    if causal:
        query_length, key_length = query.shape[-2], key.shape[-2]
        # The offset also makes the function correct for a short query attending
        # to a longer cached key/value sequence during autoregressive decoding.
        query_positions = torch.arange(query_length, device=query.device)
        query_positions = query_positions + max(key_length - query_length, 0)
        key_positions = torch.arange(key_length, device=query.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    if attn_mask is not None:
        mask = attn_mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.to(dtype=scores.dtype)

    weights = torch.softmax(scores, dim=-1)
    # An intentionally all-masked row has no probability distribution. Returning
    # zeros is more useful for experiments than allowing NaNs to spread.
    weights = torch.nan_to_num(weights, nan=0.0)
    if dropout_p:
        weights = F.dropout(weights, p=dropout_p, training=training)
    context = weights @ value
    return context, weights
