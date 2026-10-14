"""Token-packed batches. A batch fits ``token_budget`` only when the assembled
request plus reserved output fits.

``token_budget`` is the whole call: instructions, retrieved definitions,
examples, neighbors, metadata, schema, the passages, and room for the model's
answer. When a tokenizer with ``encode`` is available it is used. Otherwise
tokens are estimated as characters / 3, which is deliberately higher than the
old characters / 4 average so JSON keys and instructions are not under-counted.

A single passage that cannot fit is split on whitespace into pieces that do.
If no piece fits, that passage is sent alone and named in a warning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from langchain_core.messages import HumanMessage

from automated_grounded_theory.prompts import message_text

# Models that raised from get_num_tokens are not asked again.
_BROKEN_TOKENIZERS: set[int] = set()


def estimate_tokens(text: str) -> int:
    """Conservative character estimate. About three characters per token.

    Empty text is zero. Any other text is at least one token.
    """
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def resolve_tokenizer(model):
    """A tokenizer that can encode text, or None.

    LangChain's ``get_num_tokens`` falls back to a GPT-2 tokenizer that needs
    the ``transformers`` package. That fallback is not used. Pass a model with
    ``tokenizer.encode`` or ``encode`` when a real count is available.
    """
    if model is None:
        return None
    for attr in ("tokenizer", "_tokenizer"):
        tokenizer = getattr(model, attr, None)
        if tokenizer is not None and callable(getattr(tokenizer, "encode", None)):
            return tokenizer
    if callable(getattr(model, "encode", None)):
        return model
    return None


def count_text_tokens(text: str, tokenizer=None) -> int:
    """Token count for ``text``. Falls back to :func:`estimate_tokens`."""
    if not text:
        return 0
    if tokenizer is None or id(tokenizer) in _BROKEN_TOKENIZERS:
        return estimate_tokens(text)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return len(encode(text))
        except Exception:
            _BROKEN_TOKENIZERS.add(id(tokenizer))
            return estimate_tokens(text)
    counter = getattr(tokenizer, "get_num_tokens", None)
    if callable(counter):
        try:
            return max(0, int(counter(text)))
        except Exception:
            _BROKEN_TOKENIZERS.add(id(tokenizer))
            return estimate_tokens(text)
    return estimate_tokens(text)


_SCHEMA_TOKEN_CACHE: dict[tuple[int, int | None], int] = {}


def schema_token_count(schema, tokenizer=None) -> int:
    """Tokens in the JSON schema sent with a structured request.

    Message text does not include it. A budget that ignores it under-counts
    every call. The schema does not change between span checks, so the count
    is cached. Rebuilding it for every slice of a long passage dominated the
    100,000-character split.
    """
    if schema is None:
        return 0
    model = getattr(schema, "model_json_schema", None)
    if not callable(model):
        return 0
    key = (id(schema), None if tokenizer is None else id(tokenizer))
    cached = _SCHEMA_TOKEN_CACHE.get(key)
    if cached is not None:
        return cached
    raw = json.dumps(model(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    count = count_text_tokens(raw, tokenizer)
    _SCHEMA_TOKEN_CACHE[key] = count
    return count


def repair_note(schema) -> str:
    """The short repair instruction reserved inside ``token_budget``."""
    name = getattr(schema, "__name__", "output")
    return (
        "REPAIR THIS OUTPUT\n"
        f"The previous output was not valid {name}. Return only the structured object."
    )


def repair_allowance_tokens(schema, tokenizer=None) -> int:
    """Tokens a single repair note adds on top of the original messages."""
    return count_message_tokens([HumanMessage(content=repair_note(schema))], tokenizer)


def request_input_tokens(messages: Sequence, schema, tokenizer=None) -> int:
    """Input tokens for messages plus the structured-output schema."""
    return count_message_tokens(messages, tokenizer) + schema_token_count(schema, tokenizer)


def shrink_repair_messages(messages: Sequence, schema, input_budget: int, tokenizer=None) -> list:
    """Shorten a repair note until the assembled input fits ``input_budget``.

    The original messages stay intact. Only the trailing repair note is cut,
    down to :func:`repair_note`.
    """
    fitted = list(messages)
    budget = max(1, int(input_budget))
    short = repair_note(schema)
    for _ in range(24):
        if request_input_tokens(fitted, schema, tokenizer) <= budget:
            return fitted
        text = message_text(fitted[-1])
        if len(text) <= len(short):
            fitted[-1] = HumanMessage(content=short)
            return fitted
        fitted[-1] = HumanMessage(content=text[: max(len(short), len(text) // 2)])
    fitted[-1] = HumanMessage(content=short)
    return fitted


def count_message_tokens(messages: Sequence, tokenizer=None) -> int:
    """Tokens in assembled chat messages, including a small per-message frame."""
    if (
        tokenizer is not None
        and id(tokenizer) not in _BROKEN_TOKENIZERS
        and hasattr(tokenizer, "get_num_tokens_from_messages")
    ):
        try:
            return max(0, int(tokenizer.get_num_tokens_from_messages(list(messages))))
        except Exception:
            _BROKEN_TOKENIZERS.add(id(tokenizer))
    total = 0
    for message in messages:
        total += count_text_tokens(message_text(message), tokenizer) + 4
    return total


def reserved_output_tokens(n_passages: int, *, max_codes_per_passage: int = 6) -> int:
    """Room left in ``token_budget`` for the model's answer.

    One passage reserves a short memo plus about 48 tokens per code decision.
    """
    per_code = 48
    per_passage = 36 + per_code * max(1, int(max_codes_per_passage))
    return per_passage * max(1, int(n_passages))


@dataclass
class PackedBatch:
    """One model call. ``slices`` are ``(start, end)`` spans inside a single passage."""

    indices: list[int]
    slices: list[tuple[int, int]] | None = None
    warning: str = ""


@dataclass
class _PackResult:
    batches: list[PackedBatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def pack_indices(
    indices: Sequence[int],
    texts: Sequence[str],
    *,
    batch_size: int,
    token_budget: int,
    preamble: str = "",
    extras: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
    tokenizer=None,
    reserved_output: int = 0,
    measure: Callable[[Sequence[int], Mapping[int, str] | None], int] | None = None,
) -> tuple[list[PackedBatch], list[str]]:
    """Split ``indices`` into batches in the order given. Never reorders them.

    ``measure(indices, overrides)`` returns input tokens for the assembled
    request. ``overrides`` maps a passage index to replacement text. When
    ``measure`` is omitted, the estimate is ``preamble`` plus each passage,
    its ``extras`` entry (neighbors, metadata, retrieved codes), and schema
    overhead. ``reserved_output`` is added per passage and must fit inside
    ``token_budget`` together with that request.

    A passage that does not fit alone is split on whitespace. If no slice
    fits, it is returned as its own batch and listed in the warnings.
    """
    width = max(1, int(batch_size))
    budget = max(1, int(token_budget))
    reserve = max(0, int(reserved_output))

    def cost(batch: Sequence[int], overrides: Mapping[int, str] | None = None) -> int:
        incoming = _incoming_tokens(
            batch,
            texts,
            overrides=overrides,
            preamble=preamble,
            extras=extras,
            tokenizer=tokenizer,
            measure=measure,
        )
        return incoming + reserve * max(1, len(batch))

    packed = _PackResult()
    current: list[int] = []

    def flush() -> None:
        nonlocal current
        if current:
            packed.batches.append(PackedBatch(indices=list(current)))
            current = []

    for index in indices:
        text = texts[index] if 0 <= index < len(texts) else ""
        if len(current) >= width:
            flush()
        if current and cost([*current, index]) > budget:
            flush()
        if not current and cost([index]) > budget:
            _emit_oversized(
                packed,
                index=index,
                text=text,
                budget=budget,
                name=_name(names, index),
                fits=lambda piece, index=index: cost([index], {index: piece}) <= budget,
            )
            continue
        current.append(index)
        if len(current) >= width:
            flush()
    flush()
    return packed.batches, packed.warnings


def _name(names: Sequence[str] | None, index: int) -> str:
    if names is not None and 0 <= index < len(names) and str(names[index]).strip():
        return str(names[index])
    return f"index {index}"


def _incoming_tokens(
    batch: Sequence[int],
    texts: Sequence[str],
    *,
    overrides: Mapping[int, str] | None,
    preamble: str,
    extras: Sequence[str] | None,
    tokenizer,
    measure: Callable[[Sequence[int], Mapping[int, str] | None], int] | None,
) -> int:
    if measure is not None:
        return max(0, int(measure(list(batch), overrides)))
    parts = [preamble]
    for index in batch:
        if overrides and index in overrides:
            parts.append(overrides[index])
        elif 0 <= index < len(texts):
            parts.append(texts[index])
        if extras is not None and 0 <= index < len(extras):
            parts.append(extras[index])
    # JSON keys, roles, and list framing beyond the raw strings.
    framing = 32 * max(1, len(batch))
    return count_text_tokens("\n".join(parts), tokenizer) + framing


def _emit_oversized(packed: _PackResult, *, index: int, text: str, budget: int, name: str, fits) -> None:
    spans = split_text_spans(text, fits)
    if spans and len(spans) > 1:
        warning = (
            f"Passage {name} is larger than token_budget ({budget} tokens) "
            f"and was split into {len(spans)} requests."
        )
        packed.warnings.append(warning)
        packed.batches.append(PackedBatch(indices=[index], slices=spans, warning=warning))
        return
    warning = (
        f"Passage {name} is larger than token_budget ({budget} tokens). "
        "It could not be split, so it was sent alone."
    )
    packed.warnings.append(warning)
    packed.batches.append(PackedBatch(indices=[index], warning=warning))


def split_text_spans(text: str, fits: Callable[[str], bool]) -> list[tuple[int, int]] | None:
    """Whitespace spans of ``text`` that each satisfy ``fits``.

    Returns None when the text already fits (the caller keeps it whole) or
    when even the smallest piece does not fit.
    """
    if fits(text):
        return None
    points = [0]
    for match in re.finditer(r"\s+", text):
        points.append(match.end())
    points.append(len(text))
    points = sorted(set(points))
    if len(points) <= 2:
        return None
    spans: list[tuple[int, int]] = []
    start_at = 0
    while start_at < len(points) - 1:
        lo = start_at + 1
        hi = len(points) - 1
        best: int | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            piece = text[points[start_at] : points[mid]]
            if piece.strip() and fits(piece):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            return None
        spans.append((points[start_at], points[best]))
        start_at = best
    if len(spans) <= 1:
        return None
    return spans
