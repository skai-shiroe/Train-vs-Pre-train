"""Scaled dot product attention.

The single formula the whole architecture rests on, from section 3.2.1 of the
original paper:

```text
Attention(Q, K, V) = softmax(Q Kt / sqrt(d_k)) V
```

Reading the terms:

* **Q**, the queries. One vector per output position: what this position is
  looking for.
* **K**, the keys. One vector per input position: what this position offers.
* **V**, the values. One vector per input position: what is actually read when
  the position is selected.
* **d_k**, the dimension of a single head, that is ``d_model / num_heads``.

``Q Kt`` gives every query a similarity score against every key. The softmax
turns each row of scores into a probability distribution, and the product with
``V`` returns a weighted average of the values.

**Why divide by sqrt(d_k).** If the components of ``Q`` and ``K`` are
independent with zero mean and unit variance, their dot product over ``d_k``
dimensions has variance ``d_k``. As ``d_k`` grows the scores spread out, the
softmax saturates, and its gradient vanishes. Dividing by ``sqrt(d_k)`` brings
the variance back to one and keeps the gradient usable.

**Masking.** Forbidden positions receive the smallest finite value of the dtype
before the softmax, so their weight after the softmax is zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.scratch.masks import masked_fill_value


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: torch.nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``softmax(Q Kt / sqrt(d_k)) V``.

    Args:
        query: Tensor of shape ``(batch, heads, query_len, d_head)``.
        key: Tensor of shape ``(batch, heads, key_len, d_head)``.
        value: Tensor of shape ``(batch, heads, key_len, d_head)``.
        mask: Optional boolean tensor broadcastable to
            ``(batch, heads, query_len, key_len)``. ``True`` keeps a position,
            ``False`` forbids it.
        dropout: Optional dropout applied to the attention weights, as in the
            original paper.

    Returns:
        A pair ``(context, weights)`` where ``context`` has shape
        ``(batch, heads, query_len, d_head)`` and ``weights`` has shape
        ``(batch, heads, query_len, key_len)``.

    Raises:
        ValueError: If the key and the value hold a different number of positions.
    """
    if key.size(-2) != value.size(-2):
        raise ValueError(
            f"Key and value must share their length, got {key.size(-2)} and {value.size(-2)}."
        )

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, masked_fill_value(scores.dtype))

    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    return torch.matmul(weights, value), weights
