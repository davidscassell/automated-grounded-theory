"""Token-packed batches. Composition depends only on passage order and the budget.

Tokens are estimated as characters / 4 (the same approximation the evaluator
uses). A caller-supplied ``batch_size`` is still a hard cap, so tests that
set the batch size keep that many passages per call when the passages are short.
"""

from __future__ import annotations

from typing import Sequence


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def pack_indices(
    indices: Sequence[int],
    texts: Sequence[str],
    *,
    batch_size: int,
    token_budget: int,
) -> list[list[int]]:
    """Split ``indices`` into batches in the order given. Never reorders them."""
    width = max(1, int(batch_size))
    budget = max(1, int(token_budget))
    batches: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index in indices:
        cost = estimate_tokens(texts[index] if index < len(texts) else "")
        if current and (len(current) >= width or used + cost > budget):
            batches.append(current)
            current = []
            used = 0
        current.append(index)
        used += cost
        if len(current) >= width:
            batches.append(current)
            current = []
            used = 0
    if current:
        batches.append(current)
    return batches
