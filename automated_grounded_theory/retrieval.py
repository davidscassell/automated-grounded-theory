"""Code retrieval for the prompt, and merge candidates for sweeps.

The default retrievers are lexical overlap and, when the caller supplies
embeddings, cosine similarity. Pass any object with ``rank()`` as
``code_retriever`` to route passages with your own scorer.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Protocol, Sequence

from automated_grounded_theory.validation import code_key


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.casefold()))


def jaccard(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_left * norm_right)


def code_text(entry: Mapping) -> str:
    examples = entry.get("examples") or []
    quote = ""
    if examples:
        quote = str(examples[0].get("quote") or "")
    return "\n".join(
        part
        for part in (
            str(entry.get("code") or ""),
            str(entry.get("definition") or ""),
            quote,
        )
        if part
    )


class CodeRetriever(Protocol):
    """Rank codebook entries for a passage. Most relevant label first."""

    def rank(self, passage: str, codebook: Sequence[Mapping], *, k: int) -> list[str]:
        """Return up to ``k`` code labels."""


class LexicalCodeRetriever:
    """Rank codes by token overlap with the passage. No model and no network."""

    name = "lexical"

    def rank(self, passage: str, codebook: Sequence[Mapping], *, k: int) -> list[str]:
        if k <= 0 or not codebook:
            return []
        scored = [(jaccard(passage, code_text(entry)), str(entry.get("code") or "")) for entry in codebook]
        scored = [item for item in scored if item[0] > 0 and item[1]]
        scored.sort(key=lambda item: (-item[0], item[1].casefold()))
        return [label for _, label in scored[:k]]


class EmbeddingCodeRetriever:
    """Rank codes by cosine similarity of caller-supplied embeddings."""

    name = "embedding"

    def __init__(self, embeddings) -> None:
        if embeddings is None or not hasattr(embeddings, "embed_documents"):
            raise TypeError("EmbeddingCodeRetriever requires a LangChain Embeddings instance.")
        self.embeddings = embeddings

    def rank(self, passage: str, codebook: Sequence[Mapping], *, k: int) -> list[str]:
        if k <= 0 or not codebook:
            return []
        labels = [str(entry.get("code") or "") for entry in codebook]
        vectors = self.embeddings.embed_documents([code_text(entry) for entry in codebook])
        query = self.embeddings.embed_query(passage)
        scored = [(cosine(query, vector), label) for vector, label in zip(vectors, labels) if label]
        scored.sort(key=lambda item: (-item[0], item[1].casefold()))
        return [label for _, label in scored[:k]]


def merge_candidates(
    codebook: Sequence[Mapping],
    *,
    embeddings=None,
    embedding_threshold: float = 0.82,
    lexical_threshold: float = 0.5,
    limit: int = 30,
) -> list[dict]:
    """Pair codes that are close enough for a sweep model to confirm or reject.

    Embeddings are used when ``embeddings`` is provided. Otherwise pairs are
    token-Jaccard matches. The sweep prompt is only allowed to merge pairs
    returned here.
    """
    entries = [entry for entry in codebook if entry.get("code")]
    if len(entries) < 2:
        return []
    method = "embedding" if embeddings is not None else "lexical"
    if embeddings is not None:
        vectors = embeddings.embed_documents([code_text(entry) for entry in entries])
        scores = [
            (cosine(vectors[i], vectors[j]), i, j)
            for i in range(len(entries))
            for j in range(i + 1, len(entries))
        ]
        threshold = embedding_threshold
    else:
        # Labels only. A sharpened definition should not hide two codes
        # that still name the same incident.
        scores = [
            (jaccard(str(entries[i].get("code") or ""), str(entries[j].get("code") or "")), i, j)
            for i in range(len(entries))
            for j in range(i + 1, len(entries))
        ]
        threshold = lexical_threshold

    pairs = []
    for score, i, j in scores:
        if score < threshold:
            continue
        left = str(entries[i]["code"])
        right = str(entries[j]["code"])
        if code_key(left) == code_key(right):
            continue
        # Prefer the more frequent label as the default target. The model may swap them.
        left_count = int(entries[i].get("frequency") or 0)
        right_count = int(entries[j].get("frequency") or 0)
        if right_count > left_count or (right_count == left_count and right.casefold() < left.casefold()):
            source, target = left, right
        else:
            source, target = right, left
        pairs.append(
            {
                "source": source,
                "target": target,
                "score": float(score),
                "method": method,
            }
        )
    pairs.sort(key=lambda item: (-item["score"], item["source"].casefold(), item["target"].casefold()))
    return pairs[:limit]


def select_prompt_codes(
    codebook: Sequence[Mapping],
    ranked_labels: Sequence[str],
    *,
    frequent_limit: int,
) -> tuple[list[dict], int]:
    """Union of retrieved codes and the most frequent codes, plus an omitted count."""
    by_key = {code_key(str(entry.get("code") or "")): dict(entry) for entry in codebook}
    ordered: list[dict] = []
    seen: set[str] = set()
    for label in ranked_labels:
        key = code_key(label)
        if key in by_key and key not in seen:
            ordered.append(by_key[key])
            seen.add(key)
    frequent = sorted(
        codebook,
        key=lambda entry: (-int(entry.get("frequency") or 0), str(entry.get("code") or "").casefold()),
    )
    for entry in frequent[:frequent_limit]:
        key = code_key(str(entry.get("code") or ""))
        if key and key not in seen:
            ordered.append(dict(entry))
            seen.add(key)
    omitted = max(0, len(by_key) - len(ordered))
    return ordered, omitted
