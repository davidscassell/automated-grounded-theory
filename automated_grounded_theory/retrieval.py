"""Code retrieval for the prompt, and merge candidates for sweeps.

The default retrievers are lexical overlap and, when the caller supplies
embeddings, cosine similarity. Pass any object with ``rank()`` as
``code_retriever`` to route passages with your own scorer.
"""

from __future__ import annotations

import asyncio
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
        return _rank_vectors(labels, vectors, query, k)

    async def arank_many(
        self,
        passages: Sequence[str],
        codebook: Sequence[Mapping],
        *,
        k: int,
        limiter,
        batch_size: int = 64,
    ) -> list[list[str]]:
        """Rank every passage. The codebook is embedded once, in batches."""
        if k <= 0 or not codebook or not passages:
            return [[] for _ in passages]
        labels = [str(entry.get("code") or "") for entry in codebook]
        vectors = await aembed_texts(
            self.embeddings,
            [code_text(entry) for entry in codebook],
            batch_size=batch_size,
            limiter=limiter,
        )
        queries = await aembed_texts(
            self.embeddings,
            list(passages),
            batch_size=batch_size,
            limiter=limiter,
        )
        return [_rank_vectors(labels, vectors, query, k) for query in queries]


def _rank_vectors(labels: Sequence[str], vectors: Sequence[Sequence[float]], query: Sequence[float], k: int) -> list[str]:
    scored = [(cosine(query, vector), label) for vector, label in zip(vectors, labels) if label]
    scored.sort(key=lambda item: (-item[0], item[1].casefold()))
    return [label for _, label in scored[:k]]


async def aembed_texts(embeddings, texts: Sequence[str], *, batch_size: int, limiter) -> list[list[float]]:
    """Embed ``texts`` in batches. Uses ``aembed_documents`` when the object has it.

    Timeouts, connection errors, and rate limits retry with the limiter's backoff.
    """
    if not texts:
        return []
    size = max(1, int(batch_size))
    batches = [list(texts[start : start + size]) for start in range(0, len(texts), size)]

    async def one(batch: list[str]) -> list[list[float]]:
        async with limiter.embed_semaphore:
            for attempt in range(limiter.max_retries + 1):
                try:
                    embed = getattr(embeddings, "aembed_documents", None)

                    async def _embed():
                        if embed is None:
                            return await asyncio.to_thread(embeddings.embed_documents, batch)
                        return await embed(batch)

                    if limiter.request_timeout and limiter.request_timeout > 0:
                        return await asyncio.wait_for(_embed(), limiter.request_timeout)
                    return await _embed()
                except Exception as exc:
                    if attempt >= limiter.max_retries or not _transient(exc):
                        raise
                    if _rate_limit(exc):
                        limiter.rate_limit_retries += 1
                    else:
                        limiter.timeout_retries += 1
                    await asyncio.sleep(limiter._wait(attempt, exc))
        raise RuntimeError("embedding retry loop exited without a result")

    groups = await asyncio.gather(*(one(batch) for batch in batches))
    return [vector for group in groups for vector in group]


def _rate_limit(exc: BaseException) -> bool:
    from automated_grounded_theory.async_support import is_rate_limit

    return is_rate_limit(exc)


def _transient(exc: BaseException) -> bool:
    from automated_grounded_theory.async_support import is_transient

    return is_transient(exc)


def comparison_order(rows: Sequence[Mapping], *, label: str = "code") -> list:
    """Sort rows alphabetically by label, then by definition.

    Sweeps and duplicate-theme review use this order so near-duplicate wording
    lands in the same chunk. The sort does not depend on frequency or on embeddings.
    """
    ordered = [row for row in rows]
    ordered.sort(
        key=lambda row: (
            str(row.get(label) or "").casefold(),
            str(row.get("definition") or "").casefold(),
        )
    )
    return ordered


def merge_candidates(
    codebook: Sequence[Mapping],
    *,
    embeddings=None,
    embedding_threshold: float = 0.65,
    lexical_threshold: float = 0.5,
    limit: int = 30,
) -> list[dict]:
    """Optional similarity hints. The sweep model may merge pairs that are not here.

    Token overlap is always considered. Embeddings add pairs when ``embeddings``
    is provided. They never replace the lexical hints.
    """
    entries = [entry for entry in codebook if entry.get("code")]
    vectors = None
    if embeddings is not None and entries:
        vectors = embeddings.embed_documents([code_text(entry) for entry in entries])
    return hint_pairs(
        entries,
        vectors,
        embedding_threshold=embedding_threshold,
        lexical_threshold=lexical_threshold,
        limit=limit,
    )


def hint_pairs(
    entries: Sequence[Mapping],
    vectors: Sequence[Sequence[float]] | None = None,
    *,
    embedding_threshold: float = 0.65,
    lexical_threshold: float = 0.5,
    limit: int | None = 30,
) -> list[dict]:
    """Union of lexical hints and, when vectors are present, embedding hints.

    An empty list does not mean there is nothing to merge. The model still
    reads the codebook. ``limit`` caps how many hints are shown, highest score
    first.
    """
    if len(entries) < 2:
        return []
    groups = [
        score_pairs(
            entries,
            None,
            embedding_threshold=embedding_threshold,
            lexical_threshold=lexical_threshold,
            limit=None,
        )
    ]
    if vectors is not None:
        groups.append(
            score_pairs(
                entries,
                vectors,
                embedding_threshold=embedding_threshold,
                lexical_threshold=lexical_threshold,
                limit=None,
            )
        )
    paired = union_candidates(groups)
    if limit is not None:
        return paired[: max(0, int(limit))]
    return paired


def score_pairs(
    entries: Sequence[Mapping],
    vectors: Sequence[Sequence[float]] | None,
    *,
    embedding_threshold: float = 0.65,
    lexical_threshold: float = 0.5,
    limit: int | None = 30,
    chunk_indexes: Sequence[int] | None = None,
) -> list[dict]:
    """Pair codes that are close enough for a sweep to confirm.

    When ``chunk_indexes`` is set, a pair is kept when at least one side is in
    that chunk. Labels are compared when ``vectors`` is missing, so a sharpened
    definition cannot hide two codes that still name the same incident.
    """
    if len(entries) < 2:
        return []
    method = "embedding" if vectors is not None else "lexical"
    threshold = embedding_threshold if vectors is not None else lexical_threshold
    allowed = None if chunk_indexes is None else set(chunk_indexes)
    scores = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if allowed is not None and i not in allowed and j not in allowed:
                continue
            if vectors is not None:
                score = cosine(vectors[i], vectors[j])
            else:
                score = jaccard(str(entries[i].get("code") or ""), str(entries[j].get("code") or ""))
            scores.append((score, i, j))
    pairs = []
    for score, i, j in scores:
        if score < threshold:
            continue
        left = str(entries[i].get("code") or "")
        right = str(entries[j].get("code") or "")
        if not left or not right or code_key(left) == code_key(right):
            continue
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
    if limit is None:
        return pairs
    return pairs[: max(0, int(limit))]


def union_candidates(groups: Sequence[Sequence[Mapping]]) -> list[dict]:
    """Dedupe candidate pairs from concurrent chunks. Highest score wins."""
    best: dict[frozenset[str], dict] = {}
    for group in groups:
        for item in group:
            source = code_key(str(item.get("source") or ""))
            target = code_key(str(item.get("target") or ""))
            if not source or not target or source == target:
                continue
            key = frozenset((source, target))
            current = best.get(key)
            if current is None or float(item.get("score") or 0) > float(current.get("score") or 0):
                best[key] = dict(item)
    pairs = list(best.values())
    pairs.sort(key=lambda item: (-float(item["score"]), str(item["source"]).casefold(), str(item["target"]).casefold()))
    return pairs


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
