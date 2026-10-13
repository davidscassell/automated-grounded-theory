"""Code retrieval for the prompt, and merge candidates for sweeps.

The default retrievers are lexical overlap and, when the caller supplies
embeddings, cosine similarity. Pass any object with ``rank()`` as
``code_retriever`` to route passages with your own scorer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import threading
from typing import Any, Mapping, Protocol, Sequence

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


# Output-affecting settings. Two clients with the same model name and different
# dimensions, endpoints, or deployments must not share vectors.
_FINGERPRINT_FIELDS = (
    "dimensions",
    "embedding_dimensions",
    "output_dimensionality",
    "azure_endpoint",
    "azure_deployment",
    "deployment",
    "deployment_name",
    "base_url",
    "openai_api_base",
    "api_base",
    "endpoint",
    "api_version",
    "openai_api_version",
    "model_kwargs",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def embedding_config_fingerprint(embeddings) -> str:
    """Stable hash of settings that change the vectors, not of the instance id."""
    payload = {}
    for attr in _FINGERPRINT_FIELDS:
        value = getattr(embeddings, attr, None)
        if value in (None, "", {}, []):
            continue
        payload[attr] = _json_safe(value)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def embedding_model_id(embeddings) -> str:
    """Identity of the embedding model and the settings that change its vectors.

    Two instances of the same named model share a cache when their dimensions,
    endpoint, and deployment match. A 3-dimension client and a 5-dimension
    client do not, even when ``model`` is the same string.
    """
    name = ""
    for attr in ("model", "model_name", "model_id"):
        value = getattr(embeddings, attr, None)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    ident = name or str(id(embeddings))
    return f"{type(embeddings).__qualname__}:{ident}:{embedding_config_fingerprint(embeddings)}"


def expected_dimensions(embeddings) -> int | None:
    """Configured output width, or None when the client does not declare one."""
    for attr in ("dimensions", "embedding_dimensions", "output_dimensionality"):
        found = _positive_int(getattr(embeddings, attr, None))
        if found is not None:
            return found
    kwargs = getattr(embeddings, "model_kwargs", None)
    if isinstance(kwargs, dict):
        for attr in ("dimensions", "embedding_dimensions", "output_dimensionality"):
            found = _positive_int(kwargs.get(attr))
            if found is not None:
                return found
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit() and int(value.strip()) > 0:
        return int(value.strip())
    return None


def _check_vectors(embeddings, vectors: Sequence[Sequence[float]]) -> None:
    """Refuse to cache a batch whose width is mixed or not the configured width."""
    if not vectors:
        return
    lengths = {len(vector) for vector in vectors}
    if len(lengths) != 1:
        raise RuntimeError(
            f"Embedding model returned mixed vector lengths {sorted(lengths)}. Refusing to cache them."
        )
    got = next(iter(lengths))
    if got <= 0:
        raise RuntimeError("Embedding model returned an empty vector.")
    expected = expected_dimensions(embeddings)
    if expected is not None and got != expected:
        raise RuntimeError(
            f"Embedding model returned {got} dimensions but is configured for {expected}. "
            "Refusing to cache a vector that does not match the configured output size."
        )


def _content_key(model_id: str, text: str, role: str) -> tuple[str, str]:
    """Cache key: embedding model (and role) plus the sha256 of the exact text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (f"{model_id}:{role}", digest)


class _VectorCache:
    """Process-wide vectors and one codebook snapshot index.

    Concurrent callers that ask for the same text share one embedding call.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.vectors: dict[tuple[str, str], list[float]] = {}
        self.inflight: dict[tuple[str, str], asyncio.Future] = {}
        self.snapshots: dict[tuple[str, str], list[list[float]]] = {}
        self.snapshot_inflight: dict[tuple[str, str], asyncio.Future] = {}


_VECTOR_CACHE = _VectorCache()


def _require_vectors(result: Sequence[list[float] | None]) -> list[list[float]]:
    if any(vector is None for vector in result):
        raise RuntimeError("embedding cache missed a text it had claimed")
    return [list(vector) for vector in result]


def clear_embedding_cache() -> None:
    """Drop cached vectors. In-flight calls are left to finish."""
    with _VECTOR_CACHE.lock:
        _VECTOR_CACHE.vectors.clear()
        _VECTOR_CACHE.snapshots.clear()


def _store_vector(key: tuple[str, str], vector: Sequence[float]) -> list[float]:
    stored = [float(item) for item in vector]
    _VECTOR_CACHE.vectors[key] = stored
    return stored


def _abort_future(future: asyncio.Future | None, exc: BaseException) -> None:
    """Fail a shared future. Cancellation is propagated, not stored as a result."""
    if future is None or future.done():
        return
    if isinstance(exc, asyncio.CancelledError):
        future.cancel()
        return
    future.set_exception(exc)
    future.add_done_callback(_silence_future)


def _silence_future(future: asyncio.Future) -> None:
    if future.cancelled():
        return
    future.exception()


def _abort_inflight(keys: Sequence[tuple[str, str]], exc: BaseException) -> None:
    with _VECTOR_CACHE.lock:
        for key in keys:
            _abort_future(_VECTOR_CACHE.inflight.pop(key, None), exc)


def _release_inflight(keys: Sequence[tuple[str, str]], exc: BaseException) -> None:
    """Fail unfinished calls. A query that already cached its vector is kept."""
    with _VECTOR_CACHE.lock:
        for key in keys:
            future = _VECTOR_CACHE.inflight.get(key)
            if future is None or future.done():
                _VECTOR_CACHE.inflight.pop(key, None)
                continue
            cached = _VECTOR_CACHE.vectors.get(key)
            _VECTOR_CACHE.inflight.pop(key, None)
            if cached is not None:
                future.set_result(list(cached))
            else:
                _abort_future(future, exc)


def cached_embed_texts(
    embeddings,
    texts: Sequence[str],
    *,
    role: str = "document",
    embed=None,
) -> list[list[float]]:
    """Embed ``texts``, reusing vectors whose exact content is already cached.

    ``embed`` defaults to ``embeddings.embed_documents``. Pass a callable that
    maps a list of strings to vectors when the call is ``embed_query`` instead.
    """
    if not texts:
        return []
    model = embedding_model_id(embeddings)
    keys = [_content_key(model, text, role) for text in texts]
    result: list[list[float] | None] = [None] * len(texts)
    missing: list[int] = []
    seen: set[tuple[str, str]] = set()
    with _VECTOR_CACHE.lock:
        for index, key in enumerate(keys):
            hit = _VECTOR_CACHE.vectors.get(key)
            if hit is not None:
                result[index] = hit
            elif key not in seen:
                seen.add(key)
                missing.append(index)
    if missing:
        embed_fn = embed or embeddings.embed_documents
        fresh = embed_fn([texts[index] for index in missing])
        if len(fresh) != len(missing):
            raise RuntimeError("embedding model returned a different number of vectors than texts")
        _check_vectors(embeddings, fresh)
        with _VECTOR_CACHE.lock:
            for index, vector in zip(missing, fresh):
                result[index] = _store_vector(keys[index], vector)
    with _VECTOR_CACHE.lock:
        for index, key in enumerate(keys):
            if result[index] is None:
                result[index] = list(_VECTOR_CACHE.vectors[key])
    return _require_vectors(result)


async def cached_aembed_texts(
    embeddings,
    texts: Sequence[str],
    *,
    batch_size: int,
    limiter,
    role: str = "document",
) -> list[list[float]]:
    """Async form of :func:`cached_embed_texts`. Identical in-flight texts share one call."""
    if not texts:
        return []
    model = embedding_model_id(embeddings)
    keys = [_content_key(model, text, role) for text in texts]
    loop = asyncio.get_running_loop()
    result: list[list[float] | None] = [None] * len(texts)
    mine_keys: list[tuple[str, str]] = []
    mine_texts: list[str] = []
    waiting: list[tuple[int, asyncio.Future]] = []
    claimed: set[tuple[str, str]] = set()
    with _VECTOR_CACHE.lock:
        for index, key in enumerate(keys):
            hit = _VECTOR_CACHE.vectors.get(key)
            if hit is not None:
                result[index] = hit
                continue
            inflight = _VECTOR_CACHE.inflight.get(key)
            if inflight is not None:
                waiting.append((index, inflight))
                continue
            if key in claimed:
                continue
            claimed.add(key)
            future: asyncio.Future = loop.create_future()
            _VECTOR_CACHE.inflight[key] = future
            mine_keys.append(key)
            mine_texts.append(texts[index])
    if mine_texts:
        try:
            embedded = await aembed_texts(
                embeddings,
                mine_texts,
                batch_size=batch_size,
                limiter=limiter,
                role=role,
            )
            if len(embedded) != len(mine_texts):
                raise RuntimeError("embedding model returned a different number of vectors than texts")
            _check_vectors(embeddings, embedded)
            with _VECTOR_CACHE.lock:
                for key, vector in zip(mine_keys, embedded):
                    stored = _store_vector(key, vector)
                    future = _VECTOR_CACHE.inflight.pop(key, None)
                    if future is not None and not future.done():
                        future.set_result(stored)
        except BaseException as exc:
            _release_inflight(mine_keys, exc)
            raise
    for index, future in waiting:
        # shield: cancelling this waiter must not cancel the shared call.
        result[index] = list(await asyncio.shield(future))
    with _VECTOR_CACHE.lock:
        for index, key in enumerate(keys):
            if result[index] is None:
                result[index] = list(_VECTOR_CACHE.vectors[key])
    return _require_vectors(result)


def _snapshot_key(model_id: str, texts: Sequence[str]) -> tuple[str, str]:
    digest = hashlib.sha256("\0".join(texts).encode("utf-8")).hexdigest()
    return (model_id, digest)


async def shared_codebook_vectors(
    embeddings,
    entries: Sequence[Mapping],
    *,
    limiter,
    batch_size: int,
) -> list[list[float]]:
    """One vector list for this codebook snapshot, shared by concurrent callers.

    A later snapshot re-embeds only the texts whose content changed.
    """
    texts = [code_text(entry) for entry in entries]
    snap = _snapshot_key(embedding_model_id(embeddings), texts)
    loop = asyncio.get_running_loop()
    with _VECTOR_CACHE.lock:
        cached = _VECTOR_CACHE.snapshots.get(snap)
        if cached is not None:
            return [list(vector) for vector in cached]
        waiter = _VECTOR_CACHE.snapshot_inflight.get(snap)
        if waiter is not None:
            owner = False
        else:
            waiter = loop.create_future()
            _VECTOR_CACHE.snapshot_inflight[snap] = waiter
            owner = True
    if not owner:
        vectors = await asyncio.shield(waiter)
        return [list(vector) for vector in vectors]
    try:
        vectors = await cached_aembed_texts(
            embeddings,
            texts,
            batch_size=batch_size,
            limiter=limiter,
        )
    except BaseException as exc:
        with _VECTOR_CACHE.lock:
            _VECTOR_CACHE.snapshot_inflight.pop(snap, None)
        _abort_future(waiter, exc)
        raise
    with _VECTOR_CACHE.lock:
        _VECTOR_CACHE.snapshots[snap] = vectors
        _VECTOR_CACHE.snapshot_inflight.pop(snap, None)
    if not waiter.done():
        waiter.set_result(vectors)
    return [list(vector) for vector in vectors]


def review_fingerprint(entry: Mapping) -> str:
    """Revision token for a code. Label, definition, revision, and example quotes.

    Frequency does not change it. A sweep stores this on ``reviewed_revision``.
    """
    payload = {
        "revision": int(entry.get("revision") or 1),
        "label": str(entry.get("code") or entry.get("label") or ""),
        "definition": str(entry.get("definition") or ""),
        "examples": [str(item.get("quote") or "") for item in (entry.get("examples") or [])[:3]],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return f"{payload['revision']}:{digest}"


def code_needs_review(entry: Mapping) -> bool:
    """True when the code is new or its review fingerprint changed."""
    return str(entry.get("reviewed_revision") or "") != review_fingerprint(entry)


def full_audit_due(sweep_index: int, *, consolidation: bool, full_audit_every: int) -> bool:
    """Consolidation is always a full audit. Otherwise every ``full_audit_every`` sweeps.

    ``full_audit_every`` of 0 audits only on consolidation. Sweep 1 is a full
    audit when the interval is positive, because ``(1 - 1) % n == 0``.
    """
    if consolidation:
        return True
    every = int(full_audit_every)
    if every <= 0:
        return False
    return (int(sweep_index) - 1) % every == 0


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
        vectors = cached_embed_texts(self.embeddings, [code_text(entry) for entry in codebook])
        query = cached_embed_texts(
            self.embeddings,
            [passage],
            role="query",
            embed=lambda texts: [self.embeddings.embed_query(text) for text in texts],
        )[0]
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
        """Rank every passage. Unchanged codebook rows are not embedded again.

        Concurrent callers with the same snapshot share one vector list.
        """
        if k <= 0 or not codebook or not passages:
            return [[] for _ in passages]
        labels = [str(entry.get("code") or "") for entry in codebook]
        vectors = await shared_codebook_vectors(
            self.embeddings,
            codebook,
            limiter=limiter,
            batch_size=batch_size,
        )
        # Passages are queries, matching ``rank``. A query vector and a document
        # vector of the same text do not share a cache entry, so using one role
        # here and the other in ``rank`` embeds the passage twice.
        queries = await cached_aembed_texts(
            self.embeddings,
            list(passages),
            batch_size=batch_size,
            limiter=limiter,
            role="query",
        )
        return [_rank_vectors(labels, vectors, query, k) for query in queries]


def _rank_vectors(labels: Sequence[str], vectors: Sequence[Sequence[float]], query: Sequence[float], k: int) -> list[str]:
    scored = [(cosine(query, vector), label) for vector, label in zip(vectors, labels) if label]
    scored.sort(key=lambda item: (-item[0], item[1].casefold()))
    return [label for _, label in scored[:k]]


def _cached_vector(embeddings, text: str, role: str) -> list[float] | None:
    key = _content_key(embedding_model_id(embeddings), text, role)
    with _VECTOR_CACHE.lock:
        hit = _VECTOR_CACHE.vectors.get(key)
        return list(hit) if hit is not None else None


def _cache_vector(embeddings, text: str, role: str, vector: Sequence[float]) -> list[float]:
    """Store one vector before the rest of the batch finishes."""
    _check_vectors(embeddings, [vector])
    key = _content_key(embedding_model_id(embeddings), text, role)
    with _VECTOR_CACHE.lock:
        return list(_store_vector(key, vector))


def _batch_query_api(embeddings):
    """A provider call that embeds many queries at once, or None.

    ``aembed_queries`` / ``embed_queries`` take a list of strings and return
    one vector per string. They are not the document API: a query vector and
    a document vector of the same text stay distinct.
    """
    async_fn = getattr(embeddings, "aembed_queries", None)
    if callable(async_fn):
        return async_fn, False
    sync_fn = getattr(embeddings, "embed_queries", None)
    if callable(sync_fn):
        return sync_fn, True
    return None, False


def _has_single_query_api(embeddings) -> bool:
    return callable(getattr(embeddings, "aembed_query", None)) or callable(getattr(embeddings, "embed_query", None))


async def _invoke_one_query(embeddings, text: str) -> list[float]:
    aembed_query = getattr(embeddings, "aembed_query", None)
    if callable(aembed_query):
        return await aembed_query(text)
    embed_query = getattr(embeddings, "embed_query", None)
    if callable(embed_query):
        return await asyncio.to_thread(embed_query, text)
    raise TypeError("embeddings object has no query method")


async def _embed_documents_batch(embeddings, batch: list[str]) -> list[list[float]]:
    embed = getattr(embeddings, "aembed_documents", None)
    if embed is None:
        return await asyncio.to_thread(embeddings.embed_documents, batch)
    return await embed(batch)


async def _retry_embedding(limiter, embed):
    """Timeout and retries for one embedding call. ``embed`` starts a new attempt."""
    for attempt in range(limiter.max_retries + 1):
        try:
            if limiter.request_timeout and limiter.request_timeout > 0:
                return await asyncio.wait_for(embed(), limiter.request_timeout)
            return await embed()
        except Exception as exc:
            if attempt >= limiter.max_retries or not _transient(exc):
                raise
            if _rate_limit(exc):
                limiter.rate_limit_retries += 1
            else:
                limiter.timeout_retries += 1
            await asyncio.sleep(limiter._wait(attempt, exc))
    raise RuntimeError("embedding retry loop exited without a result")


def _uncached_unique(embeddings, texts: Sequence[str], role: str) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        if _cached_vector(embeddings, text, role) is None:
            missing.append(text)
    return missing


async def _embed_query_batch(embeddings, texts: Sequence[str], *, batch_size: int, limiter) -> list[list[float]]:
    """One provider call per chunk. Successes are cached before the next chunk."""
    fn, sync = _batch_query_api(embeddings)
    size = max(1, int(batch_size))
    pending = _uncached_unique(embeddings, texts, "query")
    chunks = [pending[start : start + size] for start in range(0, len(pending), size)]

    async def one(chunk: list[str]) -> None:
        async with limiter.embed_semaphore:
            # A previous chunk, or a concurrent caller, may have filled these.
            needed = [text for text in chunk if _cached_vector(embeddings, text, "query") is None]
            if not needed:
                return

            async def attempt() -> None:
                current = [text for text in needed if _cached_vector(embeddings, text, "query") is None]
                if not current:
                    return
                if sync:
                    vectors = await asyncio.to_thread(fn, current)
                else:
                    vectors = await fn(current)
                if len(vectors) != len(current):
                    raise RuntimeError("embedding model returned a different number of vectors than texts")
                _check_vectors(embeddings, vectors)
                for text, vector in zip(current, vectors):
                    _cache_vector(embeddings, text, "query", vector)

            await _retry_embedding(limiter, attempt)

    if chunks:
        await asyncio.gather(*(one(chunk) for chunk in chunks))
    cached = [_cached_vector(embeddings, text, "query") for text in texts]
    if any(vector is None for vector in cached):
        raise RuntimeError("embedding cache missed a query it had claimed")
    return [list(vector) for vector in cached if vector is not None]


async def _embed_queries_one_by_one(embeddings, texts: Sequence[str], limiter) -> list[list[float]]:
    """Each query takes one semaphore slot, one timeout, and its own retries.

    A query that returns is cached before the others finish, so a retry of a
    sibling does not embed it again.
    """
    unique = _uncached_unique(embeddings, texts, "query")

    async def one(text: str) -> None:
        if _cached_vector(embeddings, text, "query") is not None:
            return
        async with limiter.embed_semaphore:
            if _cached_vector(embeddings, text, "query") is not None:
                return

            async def attempt() -> None:
                if _cached_vector(embeddings, text, "query") is not None:
                    return
                vector = await _invoke_one_query(embeddings, text)
                _cache_vector(embeddings, text, "query", vector)

            await _retry_embedding(limiter, attempt)

    if unique:
        await asyncio.gather(*(one(text) for text in unique))
    cached = [_cached_vector(embeddings, text, "query") for text in texts]
    if any(vector is None for vector in cached):
        raise RuntimeError("embedding cache missed a query it had claimed")
    return [list(vector) for vector in cached if vector is not None]


async def aembed_texts(
    embeddings,
    texts: Sequence[str],
    *,
    batch_size: int,
    limiter,
    role: str = "document",
) -> list[list[float]]:
    """Embed ``texts`` under the shared embedding semaphore.

    ``role="query"`` uses the provider's batch query API when it has one
    (``aembed_queries`` or ``embed_queries``). Otherwise each query is its own
    call, with its own timeout and retries, and only as many run at once as
    the semaphore allows. A finished query is cached immediately. Document
    role uses ``aembed_documents`` for the whole batch. The same text is not
    embedded twice. Timeouts, connection errors, and rate limits retry with
    the limiter's backoff. The wait yields the event loop.
    """
    if not texts:
        return []
    if role == "query" and (_batch_query_api(embeddings)[0] is not None or _has_single_query_api(embeddings)):
        if _batch_query_api(embeddings)[0] is not None:
            return await _embed_query_batch(embeddings, list(texts), batch_size=batch_size, limiter=limiter)
        return await _embed_queries_one_by_one(embeddings, list(texts), limiter)
    size = max(1, int(batch_size))
    batches = [list(texts[start : start + size]) for start in range(0, len(texts), size)]

    async def one(batch: list[str]) -> list[list[float]]:
        async with limiter.embed_semaphore:

            async def embed():
                return await _embed_documents_batch(embeddings, batch)

            return await _retry_embedding(limiter, embed)

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
        vectors = cached_embed_texts(embeddings, [code_text(entry) for entry in entries])
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
    chunk_indexes: Sequence[int] | None = None,
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
            chunk_indexes=chunk_indexes,
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
                chunk_indexes=chunk_indexes,
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


def select_cross_chunk(
    codes: Sequence[Mapping],
    chunk: Sequence[Mapping],
    vectors: Sequence[Sequence[float]] | None = None,
    *,
    limit: int = 8,
    embedding_threshold: float = 0.65,
    lexical_threshold: float = 0.5,
) -> list[dict]:
    """Codes outside ``chunk`` that are similar to a code inside it.

    Embeddings are used when ``vectors`` lines up with ``codes``. Token overlap
    of the label, definition, and first example is always considered. The result
    is capped at ``limit``, highest score first.
    """
    cap = max(0, int(limit))
    if cap == 0 or not codes or not chunk:
        return []
    chunk_keys = {
        code_key(str(entry.get("code") or entry.get("label") or ""))
        for entry in chunk
    }
    chunk_keys.discard("")
    indexed = []
    for index, entry in enumerate(codes):
        key = code_key(str(entry.get("code") or entry.get("label") or ""))
        if key:
            indexed.append((index, key, entry))
    best: dict[str, tuple[float, dict]] = {}
    insiders = [(index, key, entry) for index, key, entry in indexed if key in chunk_keys]
    outsiders = [(index, key, entry) for index, key, entry in indexed if key not in chunk_keys]
    for left_index, _left_key, left in insiders:
        left_text = code_text(left)
        for right_index, right_key, right in outsiders:
            lexical = jaccard(left_text, code_text(right))
            embedding = 0.0
            if vectors is not None and left_index < len(vectors) and right_index < len(vectors):
                embedding = cosine(vectors[left_index], vectors[right_index])
            if lexical < lexical_threshold and embedding < embedding_threshold:
                continue
            score = max(lexical, embedding)
            current = best.get(right_key)
            if current is None or score > current[0]:
                best[right_key] = (score, right)
    ranked = sorted(best.values(), key=lambda item: (-item[0], str(item[1].get("code") or "").casefold()))
    return [entry for _score, entry in ranked[:cap]]


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
