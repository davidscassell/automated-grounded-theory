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
import sys
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Protocol, Sequence

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


class _Inflight:
    """One shared embedding call.

    ``waiters`` counts callers that attached to this future after the owner
    claimed it. ``task`` is the provider call. Releasing the cache entry while
    that task is still running lets a later request start a second call.
    """

    __slots__ = ("future", "waiters", "task")

    def __init__(self, future: asyncio.Future) -> None:
        self.future = future
        self.waiters = 0
        self.task: asyncio.Task | None = None


# How many codebook snapshots the process keeps. Each snapshot stores content
# hashes, not its own copy of every vector. 1..N growing codebooks used to
# retain 1+2+...+N vector lists.
DEFAULT_SNAPSHOT_LIMIT = 8

_RUN_SCOPE: ContextVar[str | None] = ContextVar("embedding_run_scope", default=None)
_BACKGROUND_TASKS: set[asyncio.Future] = set()


class _UnsetType:
    pass


_UNSET = _UnsetType()


class _VectorCache:
    """Process-wide canonical vectors and a bounded codebook snapshot index.

    Concurrent callers that ask for the same text share one embedding call.
    Snapshots point at those vectors by content hash. Snapshot retention is an
    LRU cap plus an optional run scope that drops its snapshots when the run
    ends. Canonical vectors are unbounded unless ``vector_limit`` or
    ``vector_bytes`` is set. Those caps are also LRU. A vector a retained
    snapshot still points at is not evicted.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.vectors: OrderedDict[tuple[str, str], tuple[float, ...]] = OrderedDict()
        self.vector_bytes_used = 0
        self.inflight: dict[tuple[str, str], _Inflight] = {}
        self.snapshots: OrderedDict[tuple[str, str], tuple[tuple[str, str], ...]] = OrderedDict()
        self.snapshot_owners: dict[tuple[str, str], set[str]] = {}
        self.snapshot_inflight: dict[tuple[str, str], asyncio.Future] = {}
        self.snapshot_limit = DEFAULT_SNAPSHOT_LIMIT
        self.vector_limit: int | None = None
        self.vector_bytes: int | None = None
        self.pinned: set[tuple[str, str]] = set()


_VECTOR_CACHE = _VectorCache()


def _vector_payload_bytes(vector: Sequence[float]) -> int:
    """Float payload of one vector. Eight bytes per dimension, not object overhead."""
    return 8 * len(vector)


def configure_embedding_cache(
    *,
    snapshot_limit: Any = _UNSET,
    vector_limit: Any = _UNSET,
    vector_bytes: Any = _UNSET,
) -> None:
    """Set snapshot and optional canonical-vector retention.

    Vectors stay canonical and are not copied into each snapshot.
    ``snapshot_limit`` is the snapshot LRU cap. ``vector_limit`` is how many
    canonical vectors to keep. ``vector_bytes`` is their float-payload budget
    (8 bytes per dimension). Either vector cap may be ``None`` to leave that
    bound open. A vector referenced by a retained snapshot is not evicted.
    A run can also call :func:`embedding_run_scope` to drop the snapshots it
    created when the run finishes. Vectors are not dropped with those snapshots.
    """
    with _VECTOR_CACHE.lock:
        if snapshot_limit is not _UNSET:
            if snapshot_limit is None or int(snapshot_limit) < 1:
                raise ValueError("snapshot_limit must be at least 1.")
            _VECTOR_CACHE.snapshot_limit = int(snapshot_limit)
            _evict_snapshots()
        if vector_limit is not _UNSET:
            if vector_limit is not None and int(vector_limit) < 1:
                raise ValueError("vector_limit must be at least 1, or None for no count cap.")
            _VECTOR_CACHE.vector_limit = None if vector_limit is None else int(vector_limit)
        if vector_bytes is not _UNSET:
            if vector_bytes is not None and int(vector_bytes) < 1:
                raise ValueError("vector_bytes must be at least 1, or None for no byte cap.")
            _VECTOR_CACHE.vector_bytes = None if vector_bytes is None else int(vector_bytes)
        _evict_vectors()


def embedding_cache_stats() -> dict[str, int | None]:
    """Canonical vector count and how many snapshots are retained."""
    with _VECTOR_CACHE.lock:
        return {
            "vectors": len(_VECTOR_CACHE.vectors),
            "snapshots": len(_VECTOR_CACHE.snapshots),
            "snapshot_limit": _VECTOR_CACHE.snapshot_limit,
            "vector_limit": _VECTOR_CACHE.vector_limit,
            "vector_bytes": _VECTOR_CACHE.vector_bytes,
            "vector_bytes_used": _VECTOR_CACHE.vector_bytes_used,
        }


@contextmanager
def embedding_run_scope(run_id: str | None = None) -> Iterator[str]:
    """Drop this run's codebook snapshots when the run ends. Vectors stay cached."""
    ident = run_id or f"run-{threading.get_ident()}-{id(object())}"
    token = _RUN_SCOPE.set(ident)
    try:
        yield ident
    finally:
        _RUN_SCOPE.reset(token)
        with _VECTOR_CACHE.lock:
            _drop_run_snapshots(ident)


def _drop_run_snapshots(run_id: str) -> None:
    stale: list[tuple[str, str]] = []
    for key, owners in list(_VECTOR_CACHE.snapshot_owners.items()):
        if run_id not in owners:
            continue
        owners.discard(run_id)
        if not owners:
            stale.append(key)
    for key in stale:
        _VECTOR_CACHE.snapshot_owners.pop(key, None)
        _VECTOR_CACHE.snapshots.pop(key, None)


def _evict_snapshots() -> None:
    """Drop the oldest snapshots past the cap. An in-flight snapshot stays."""
    limit = _VECTOR_CACHE.snapshot_limit
    while len(_VECTOR_CACHE.snapshots) > limit:
        removed = False
        for key in list(_VECTOR_CACHE.snapshots):
            if key in _VECTOR_CACHE.snapshot_inflight:
                continue
            _VECTOR_CACHE.snapshots.pop(key, None)
            _VECTOR_CACHE.snapshot_owners.pop(key, None)
            removed = True
            break
        if not removed:
            break


def _remember_snapshot(snap: tuple[str, str], refs: tuple[tuple[str, str], ...]) -> None:
    _VECTOR_CACHE.snapshots[snap] = refs
    _VECTOR_CACHE.snapshots.move_to_end(snap)
    scope = _RUN_SCOPE.get()
    if scope is not None:
        _VECTOR_CACHE.snapshot_owners.setdefault(snap, set()).add(scope)
    _evict_snapshots()


def _touch_snapshot(snap: tuple[str, str]) -> None:
    _VECTOR_CACHE.snapshots.move_to_end(snap)
    scope = _RUN_SCOPE.get()
    if scope is not None:
        _VECTOR_CACHE.snapshot_owners.setdefault(snap, set()).add(scope)


def _materialize(refs: Sequence[tuple[str, str]]) -> list[list[float]]:
    return [list(_lookup_vector(key) or _VECTOR_CACHE.vectors[key]) for key in refs]


def _require_vectors(result: Sequence[list[float] | None]) -> list[list[float]]:
    if any(vector is None for vector in result):
        raise RuntimeError("embedding cache missed a text it had claimed")
    return [list(vector) for vector in result]


def clear_embedding_cache() -> None:
    """Drop cached vectors. In-flight calls are left to finish."""
    with _VECTOR_CACHE.lock:
        _VECTOR_CACHE.vectors.clear()
        _VECTOR_CACHE.vector_bytes_used = 0
        _VECTOR_CACHE.pinned.clear()
        _VECTOR_CACHE.snapshots.clear()
        _VECTOR_CACHE.snapshot_owners.clear()


def _referenced_vector_keys() -> set[tuple[str, str]]:
    used: set[tuple[str, str]] = set()
    for refs in _VECTOR_CACHE.snapshots.values():
        used.update(refs)
    return used


def _evict_vectors() -> None:
    """Drop least-recently-used vectors past the optional count or byte cap.

    Vectors a retained snapshot still references stay. If every remaining
    vector is referenced, the cap is left overfull rather than breaking a snapshot.
    """
    limit = _VECTOR_CACHE.vector_limit
    byte_cap = _VECTOR_CACHE.vector_bytes
    if limit is None and byte_cap is None:
        return
    protected = _referenced_vector_keys() | set(_VECTOR_CACHE.pinned)

    def over() -> bool:
        if limit is not None and len(_VECTOR_CACHE.vectors) > limit:
            return True
        if byte_cap is not None and _VECTOR_CACHE.vector_bytes_used > byte_cap:
            return True
        return False

    while over():
        removed = False
        for key in list(_VECTOR_CACHE.vectors):
            if key in protected:
                continue
            vector = _VECTOR_CACHE.vectors.pop(key)
            _VECTOR_CACHE.vector_bytes_used -= _vector_payload_bytes(vector)
            removed = True
            break
        if not removed:
            break


def _touch_vector(key: tuple[str, str]) -> None:
    if key in _VECTOR_CACHE.vectors:
        _VECTOR_CACHE.vectors.move_to_end(key)


def _lookup_vector(key: tuple[str, str]) -> tuple[float, ...] | None:
    """Return a cached vector and mark it recently used. Caller holds the lock."""
    hit = _VECTOR_CACHE.vectors.get(key)
    if hit is None:
        return None
    _VECTOR_CACHE.vectors.move_to_end(key)
    return hit


def _store_vector(key: tuple[str, str], vector: Sequence[float]) -> list[float]:
    """Store one immutable vector. A second store of the same key keeps the first."""
    existing = _VECTOR_CACHE.vectors.get(key)
    if existing is None:
        existing = tuple(float(item) for item in vector)
        _VECTOR_CACHE.vectors[key] = existing
        _VECTOR_CACHE.vector_bytes_used += _vector_payload_bytes(existing)
        _evict_vectors()
    else:
        _VECTOR_CACHE.vectors.move_to_end(key)
        existing = _VECTOR_CACHE.vectors[key]
    return list(existing)


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
            flight = _VECTOR_CACHE.inflight.pop(key, None)
            if flight is not None:
                _abort_future(flight.future, exc)


def _release_inflight(keys: Sequence[tuple[str, str]], exc: BaseException) -> None:
    """Fail unfinished calls. A query that already cached its vector is kept.

    An entry whose provider task is still running stays. Clearing it while the
    task continues lets the next request start a second call. A sibling that
    another caller is waiting on is that case: the task keeps the entry and
    publishes the vector when it returns.
    """
    with _VECTOR_CACHE.lock:
        for key in keys:
            flight = _VECTOR_CACHE.inflight.get(key)
            if flight is None or flight.future.done():
                _VECTOR_CACHE.inflight.pop(key, None)
                continue
            if flight.task is not None and not flight.task.done():
                continue
            cached = _VECTOR_CACHE.vectors.get(key)
            _VECTOR_CACHE.inflight.pop(key, None)
            if cached is not None:
                flight.future.set_result(list(cached))
            else:
                _abort_future(flight.future, exc)


def _assign_inflight_task(keys: Sequence[tuple[str, str]], task: asyncio.Task) -> None:
    with _VECTOR_CACHE.lock:
        for key in keys:
            flight = _VECTOR_CACHE.inflight.get(key)
            if flight is not None and flight.task is None:
                flight.task = task


def _detach_waiter(key: tuple[str, str]) -> None:
    with _VECTOR_CACHE.lock:
        flight = _VECTOR_CACHE.inflight.get(key)
        if flight is not None and flight.waiters > 0:
            flight.waiters -= 1


def _shared_with_other_waiters(keys: Sequence[tuple[str, str]]) -> bool:
    """True when another caller is already waiting on one of these entries."""
    with _VECTOR_CACHE.lock:
        for key in keys:
            flight = _VECTOR_CACHE.inflight.get(key)
            if flight is not None and flight.waiters > 0:
                return True
    return False


def _finish_flight(key: tuple[str, str], exc: BaseException | None) -> None:
    """Publish the outcome of the task that owns ``key``, if it still does.

    Success is the cached vector. A failure unblocks waiters when this task
    still holds the entry, which is what happens when the group left a shared
    sibling running.
    """
    task = asyncio.current_task()
    with _VECTOR_CACHE.lock:
        flight = _VECTOR_CACHE.inflight.get(key)
        if flight is None:
            return
        if flight.task is not None and task is not None and flight.task is not task:
            return
        if flight.future.done():
            _VECTOR_CACHE.inflight.pop(key, None)
            return
        cached = _VECTOR_CACHE.vectors.get(key)
        if cached is not None:
            _VECTOR_CACHE.inflight.pop(key, None)
            flight.future.set_result(list(cached))
            return
        if exc is None:
            return
        _VECTOR_CACHE.inflight.pop(key, None)
        _abort_future(flight.future, exc)


def _consume_task(task: asyncio.Task) -> None:
    if not task.done() or task.cancelled():
        return
    task.exception()


def _retain(task: asyncio.Future) -> asyncio.Future:
    """Keep a shielded task alive so the loop does not garbage-collect it."""
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


def _first_task_error(pairs: Sequence[tuple[asyncio.Task, Sequence[tuple[str, str]]]]) -> BaseException | None:
    for task, _keys in pairs:
        if not task.done() or task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            return exc
    return None


async def _settle_query_siblings(
    pairs: Sequence[tuple[asyncio.Task, Sequence[tuple[str, str]]]],
    *,
    cancel_unshared: bool,
) -> None:
    """Wait out sibling provider calls before the group error is released.

    ``cancel_unshared`` is for the caller's own cancellation. Siblings nobody
    else is waiting on are cancelled and awaited, so they are not still running
    when the cancellation returns. A sibling another caller already shielded
    keeps running and keeps its entry. A provider failure does not cancel
    siblings: their calls are awaited, and a success stays in the vector cache.
    """
    pending: list[asyncio.Task] = []
    for task, keys in pairs:
        if task.done():
            _consume_task(task)
            continue
        if cancel_unshared and _shared_with_other_waiters(keys):
            task.add_done_callback(_consume_task)
            continue
        if cancel_unshared:
            task.cancel()
        pending.append(task)
    if pending:
        # Shield the gather and keep a reference. Cancelling the owner while
        # this wait is in progress must not cancel a shared provider task,
        # and dropping the gather future would cancel its children on collection.
        gather = _retain(asyncio.gather(*pending, return_exceptions=True))
        await asyncio.shield(gather)
    for task in pending:
        _consume_task(task)


async def _await_query_group(jobs: Sequence[tuple[Any, Sequence[tuple[str, str]]]]) -> None:
    """Run query tasks. A failure does not abandon the others.

    ``asyncio.gather`` propagates the first exception and leaves unfinished
    siblings running. The cache handler then removed their entries, so the next
    request for a sibling started a second provider call. Exclusive siblings
    are finished (awaited, or cancelled when this caller itself is cancelled)
    before that error propagates. Cache ownership is released only after those
    tasks are done.
    """
    if not jobs:
        return
    pairs: list[tuple[asyncio.Task, Sequence[tuple[str, str]]]] = []
    for coro, keys in jobs:
        task = asyncio.create_task(coro)
        _assign_inflight_task(keys, task)
        pairs.append((task, keys))
    try:
        try:
            await asyncio.wait([task for task, _keys in pairs], return_when=asyncio.FIRST_EXCEPTION)
        except asyncio.CancelledError:
            await _settle_query_siblings(pairs, cancel_unshared=True)
            raise
        error = _first_task_error(pairs)
        if error is None:
            leftover = [task for task, _keys in pairs if not task.done()]
            if leftover:
                try:
                    await asyncio.wait(leftover)
                except asyncio.CancelledError:
                    await _settle_query_siblings(pairs, cancel_unshared=True)
                    raise
            return
        try:
            await _settle_query_siblings(pairs, cancel_unshared=False)
        except asyncio.CancelledError:
            # Settlement was in progress. The shielded gather did not cancel
            # shared tasks. Drop only the ones nobody else is waiting on.
            await _settle_query_siblings(pairs, cancel_unshared=True)
            raise
        raise error
    except asyncio.CancelledError:
        await _settle_query_siblings(pairs, cancel_unshared=True)
        raise


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
            hit = _lookup_vector(key)
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
                result[index] = list(_lookup_vector(key) or ())
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
    indexes_for: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(keys):
        indexes_for.setdefault(key, []).append(index)
    loop = asyncio.get_running_loop()
    result: list[list[float] | None] = [None] * len(texts)
    mine_keys: list[tuple[str, str]] = []
    mine_texts: list[str] = []
    waiting: list[tuple[int, asyncio.Future, tuple[str, str]]] = []
    registered: list[tuple[str, str]] = []
    claimed: set[tuple[str, str]] = set()
    try:
        with _VECTOR_CACHE.lock:
            for index, key in enumerate(keys):
                hit = _lookup_vector(key)
                if hit is not None:
                    result[index] = hit
                    continue
                inflight = _VECTOR_CACHE.inflight.get(key)
                if inflight is not None:
                    inflight.waiters += 1
                    registered.append(key)
                    waiting.append((index, inflight.future, key))
                    continue
                if key in claimed:
                    continue
                claimed.add(key)
                future: asyncio.Future = loop.create_future()
                _VECTOR_CACHE.inflight[key] = _Inflight(future)
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
                        # Keep the copy on this request. Storing the next vector
                        # may evict this one; that only changes what a later
                        # request can reuse.
                        for index in indexes_for.get(key, []):
                            if result[index] is None:
                                result[index] = list(stored)
                        flight = _VECTOR_CACHE.inflight.pop(key, None)
                        if flight is not None and not flight.future.done():
                            flight.future.set_result(list(stored))
            except BaseException as exc:
                _release_inflight(mine_keys, exc)
                raise
        for index, future, _key in waiting:
            # shield: cancelling this waiter must not cancel the shared call.
            result[index] = list(await asyncio.shield(future))
        with _VECTOR_CACHE.lock:
            for index, key in enumerate(keys):
                if result[index] is None:
                    result[index] = list(_lookup_vector(key) or ())
        return _require_vectors(result)
    finally:
        # Including waiters whose futures were never awaited, such as a joined
        # "slow" flight when a new "bad" text fails first.
        for key in registered:
            _detach_waiter(key)


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
    model_id = embedding_model_id(embeddings)
    snap = _snapshot_key(model_id, texts)
    refs = tuple(_content_key(model_id, text, "document") for text in texts)
    loop = asyncio.get_running_loop()
    with _VECTOR_CACHE.lock:
        cached = _VECTOR_CACHE.snapshots.get(snap)
        if cached is not None:
            _touch_snapshot(snap)
            return _materialize(cached)
        waiter = _VECTOR_CACHE.snapshot_inflight.get(snap)
        if waiter is not None:
            owner = False
        else:
            waiter = loop.create_future()
            _VECTOR_CACHE.snapshot_inflight[snap] = waiter
            owner = True
    if not owner:
        shared = await asyncio.shield(waiter)
        return [list(vector) for vector in shared]
    with _VECTOR_CACHE.lock:
        # Keep this snapshot's vectors through the embed. The cap must not drop
        # one of them before the snapshot exists to protect it.
        _VECTOR_CACHE.pinned.update(refs)
    try:
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
            _remember_snapshot(snap, refs)
            _VECTOR_CACHE.snapshot_inflight.pop(snap, None)
            materialized = _materialize(refs)
        if not waiter.done():
            waiter.set_result(materialized)
        return materialized if vectors else []
    finally:
        with _VECTOR_CACHE.lock:
            _VECTOR_CACHE.pinned.difference_update(refs)
            _evict_vectors()


def assignment_ident(item: Mapping) -> str:
    """Stable id for one assignment. ``segment_id`` wins when the codebook stored one."""
    segment_id = str(item.get("segment_id") or "").strip()
    if segment_id:
        return segment_id
    start = "" if item.get("doc_start") is None else str(item.get("doc_start"))
    end = "" if item.get("doc_end") is None else str(item.get("doc_end"))
    return "|".join(
        (
            str(item.get("passage_id") or ""),
            str(item.get("doc_id") or ""),
            start,
            end,
            str(item.get("quote") or ""),
        )
    )


def evidence_rows(entry: Mapping) -> list[Mapping]:
    """Full evidence history. Assignments win over the example list."""
    source = entry.get("assignments")
    if not source:
        source = entry.get("examples") or []
    rows: list[Mapping] = []
    seen: set[str] = set()
    for item in source:
        if isinstance(item, str):
            item = {"quote": item}
        if not isinstance(item, Mapping):
            continue
        if not str(item.get("quote") or "").strip():
            continue
        ident = assignment_ident(item)
        if ident in seen:
            continue
        seen.add(ident)
        rows.append(item)
    return rows


def assignment_ids(entry: Mapping) -> list[str]:
    """Ids of every assignment currently stored on ``entry``, in order."""
    return [assignment_ident(item) for item in evidence_rows(entry)]


# How many examples a sweep or coding prompt shows for one code. Review
# bookkeeping uses the same bound, so an id is "reviewed" only when it was sent.
PROMPT_EXAMPLE_LIMIT = 2


def record_review(
    entry: dict,
    fingerprint: str,
    seen_ids: Sequence[str] | None = None,
    sent_ids: Sequence[str] | None = None,
) -> None:
    """Remember the fingerprint, the ids that were sent, and the ids that existed.

    ``seen_ids`` is every assignment available when the review was built.
    ``sent_ids`` is the subset that was actually put in the prompt. Only the
    sent ids count as reviewed. Assignments that were available but not sent
    stay pending, so a later review still includes them, in a bounded batch.
    ``reviewed_assignment_ids`` is the cumulative sent set. Each ``reviews_seen``
    row keeps that review's sent ids separate from the ids available then.
    """
    available = [str(item) for item in (seen_ids if seen_ids is not None else assignment_ids(entry))]
    if sent_ids is None:
        sent = [assignment_ident(item) for item in select_prompt_examples(entry, limit=PROMPT_EXAMPLE_LIMIT)]
    else:
        sent = [str(item) for item in sent_ids if str(item)]
    prior = [str(item) for item in (entry.get("reviewed_assignment_ids") or [])] if "reviewed_assignment_ids" in entry else []
    cumulative = list(dict.fromkeys([*prior, *sent]))
    entry["reviewed_revision"] = fingerprint
    entry["reviewed_assignment_ids"] = cumulative
    entry["available_assignment_ids"] = list(available)
    history = []
    for item in entry.get("reviews_seen") or []:
        if isinstance(item, Mapping):
            history.append(
                {
                    "fingerprint": str(item.get("fingerprint") or ""),
                    "assignment_ids": list(item.get("sent_assignment_ids") or item.get("assignment_ids") or []),
                    "sent_assignment_ids": list(item.get("sent_assignment_ids") or item.get("assignment_ids") or []),
                    "available_assignment_ids": list(item.get("available_assignment_ids") or []),
                }
            )
    history.append(
        {
            "fingerprint": fingerprint,
            "assignment_ids": list(sent),
            "sent_assignment_ids": list(sent),
            "available_assignment_ids": list(available),
        }
    )
    entry["reviews_seen"] = history


def _reviewed_ids(entry: Mapping) -> set[str] | None:
    """Ids the last review saw, or None when this code has not been reviewed."""
    if "reviewed_assignment_ids" in entry:
        return {str(item) for item in (entry.get("reviewed_assignment_ids") or [])}
    seen = entry.get("reviews_seen") or []
    if seen and isinstance(seen[-1], Mapping) and "assignment_ids" in seen[-1]:
        return {str(item) for item in (seen[-1].get("assignment_ids") or [])}
    return None


def assignment_digest(entry: Mapping) -> str:
    """Hash of every example and assignment. A fourth quote changes it.

    This is the evidence revision used for review. It is not the example set
    sent to a prompt. :func:`select_prompt_examples` chooses that bounded set.
    """
    rows = []
    for item in list(entry.get("examples") or []) + list(entry.get("assignments") or []):
        if isinstance(item, str):
            rows.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        rows.append(
            "|".join(
                (
                    str(item.get("segment_id") or ""),
                    str(item.get("passage_id") or ""),
                    str(item.get("doc_id") or ""),
                    "" if item.get("doc_start") is None else str(item.get("doc_start")),
                    "" if item.get("doc_end") is None else str(item.get("doc_end")),
                    str(item.get("quote") or ""),
                )
            )
        )
    payload = {
        "evidence_revision": str(entry.get("evidence_revision") or ""),
        "rows": rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _example_marker(item: Mapping) -> tuple[str, str, str]:
    start = "" if item.get("doc_start") is None else str(item.get("doc_start"))
    return (str(item.get("passage_id") or ""), str(item.get("quote") or ""), start)


def _diverse_examples(pool: Sequence[Mapping], room: int, chosen: Sequence[Mapping]) -> list[dict]:
    """Fill ``room`` slots: documents, then contrasting quotes, then passage order.

    Document diversity runs before a passage-order fill. Otherwise the first
    document's passages take every slot and a later document never appears.
    """
    if room <= 0:
        return []
    added: list[dict] = []
    seen = {_example_marker(item) for item in chosen}
    seen_docs = {str(item.get("doc_id") or "") for item in chosen if item.get("doc_id")}

    def take(item: Mapping) -> bool:
        if len(added) >= room:
            return False
        marker = _example_marker(item)
        if marker in seen:
            return False
        added.append(dict(item))
        seen.add(marker)
        doc = str(item.get("doc_id") or "")
        if doc:
            seen_docs.add(doc)
        return True

    for item in pool:
        if len(added) >= room:
            return added
        doc = str(item.get("doc_id") or "")
        if not doc or doc in seen_docs:
            continue
        take(item)
    if len(added) >= room:
        return added

    def contrast(item: Mapping) -> float:
        quote = str(item.get("quote") or "")
        bases = list(chosen) + added
        if not bases:
            return 0.0
        return min(jaccard(quote, str(other.get("quote") or "")) for other in bases)

    ranked = sorted(enumerate(pool), key=lambda pair: (contrast(pair[1]), pair[0]))
    for _index, item in ranked:
        if len(added) >= room:
            return added
        take(item)
    for item in pool:
        if len(added) >= room:
            break
        take(item)
    return added


def select_prompt_examples(entry: Mapping, *, limit: int = 3) -> list[dict]:
    """A bounded example set for prompts. Not the review digest.

    New evidence is sent in batches of at most ``limit``, newest and most
    contrasting first. It is not all copied into one prompt. The last review's
    sent ids are ``reviewed_assignment_ids``. Ids that existed but were not
    sent stay fresh, so the next review takes another bounded batch.
    Representative and contrasting quotes from the already-sent history fill
    any room left under ``limit``: one new document at a time, then quotes
    that overlap the chosen set the least, then passage order.
    """
    rows = evidence_rows(entry)
    cap = max(0, int(limit))
    if cap == 0 or not rows:
        return []
    reviewed = _reviewed_ids(entry)
    if reviewed is None:
        fresh = list(rows)
        pool: list[Mapping] = []
    else:
        fresh = [item for item in rows if assignment_ident(item) not in reviewed]
        pool = [item for item in rows if assignment_ident(item) in reviewed]
    if len(fresh) <= cap:
        chosen = [dict(item) for item in fresh]
    else:
        chosen = _diverse_examples(fresh, cap, [])
    if len(chosen) >= cap:
        return chosen[:cap]
    chosen.extend(_diverse_examples(pool, cap - len(chosen), chosen))
    return chosen[:cap]


def review_fingerprint(entry: Mapping) -> str:
    """Revision token for a code. Label, definition, revision, and all evidence.

    Frequency does not change it. The digest covers every example and
    assignment, plus ``evidence_revision`` when the codebook stored one.
    A sweep stores this on ``reviewed_revision``. Prompt examples are chosen
    separately and do not define this token.
    """
    payload = {
        "revision": int(entry.get("revision") or 1),
        "label": str(entry.get("code") or entry.get("label") or ""),
        "definition": str(entry.get("definition") or ""),
        "evidence": assignment_digest(entry),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return f"{payload['revision']}:{digest}"


def code_needs_review(entry: Mapping) -> bool:
    """True when the code is new, its fingerprint changed, or evidence was not sent.

    A review that showed two of ten assignments does not retire the other
    eight. Those ids stay pending and the code is reviewed again, still
    inside the example limit. A fingerprint with no per-assignment bookkeeping
    is unchanged: older rows recorded only the revision.
    """
    if str(entry.get("reviewed_revision") or "") != review_fingerprint(entry):
        return True
    sent = _reviewed_ids(entry)
    if sent is None:
        return False
    return any(assignment_ident(item) not in sent for item in evidence_rows(entry))


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
        hit = _lookup_vector(key)
        return list(hit) if hit is not None else None


def _cache_vector(embeddings, text: str, role: str, vector: Sequence[float]) -> list[float]:
    """Store one vector before the rest of the batch finishes."""
    _check_vectors(embeddings, [vector])
    key = _content_key(embedding_model_id(embeddings), text, role)
    with _VECTOR_CACHE.lock:
        return list(_store_vector(key, vector))


def _hold_cached(embeddings, texts: Sequence[str], role: str, held: dict[tuple[str, str], list[float]]) -> None:
    """Copy vectors this request already has. Later eviction cannot drop the copy."""
    model = embedding_model_id(embeddings)
    for text in texts:
        key = _content_key(model, text, role)
        if key in held:
            continue
        hit = _cached_vector(embeddings, text, role)
        if hit is not None:
            held[key] = list(hit)


def _hold_stored(
    embeddings,
    text: str,
    role: str,
    vector: Sequence[float],
    held: dict[tuple[str, str], list[float]],
) -> list[float]:
    """Cache ``vector`` and keep the request's copy. Eviction only affects reuse."""
    stored = _cache_vector(embeddings, text, role, vector)
    held[_content_key(embedding_model_id(embeddings), text, role)] = list(stored)
    return stored


def _vectors_from_hold(
    embeddings,
    texts: Sequence[str],
    role: str,
    held: dict[tuple[str, str], list[float]],
) -> list[list[float]] | None:
    """Return this request's vectors. None when one text was never held or cached."""
    model = embedding_model_id(embeddings)
    result: list[list[float]] = []
    for text in texts:
        key = _content_key(model, text, role)
        vector = held.get(key)
        if vector is None:
            vector = _cached_vector(embeddings, text, role)
        if vector is None:
            return None
        result.append(list(vector))
    return result


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
    """One provider call per chunk. Successes are cached before the next chunk.

    Chunks are siblings. If one chunk exhausts its retries, the others are
    awaited before the error propagates, so their cache entries are not dropped
    while those calls are still running.
    """
    fn, sync = _batch_query_api(embeddings)
    size = max(1, int(batch_size))
    held: dict[tuple[str, str], list[float]] = {}
    _hold_cached(embeddings, texts, "query", held)
    pending = _uncached_unique(embeddings, texts, "query")
    chunks = [pending[start : start + size] for start in range(0, len(pending), size)]
    model = embedding_model_id(embeddings)

    async def one(chunk: list[str], keys: list[tuple[str, str]]) -> None:
        try:
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
                        _hold_stored(embeddings, text, "query", vector, held)

                await _retry_embedding(limiter, attempt)
        finally:
            exc = sys.exception()
            for key in keys:
                _finish_flight(key, exc)

    if chunks:
        jobs = []
        for chunk in chunks:
            keys = [_content_key(model, text, "query") for text in chunk]
            jobs.append((one(chunk, keys), keys))
        await _await_query_group(jobs)
    collected = _vectors_from_hold(embeddings, texts, "query", held)
    if collected is None:
        raise RuntimeError("embedding cache missed a query it had claimed")
    return collected


async def _embed_queries_one_by_one(embeddings, texts: Sequence[str], limiter) -> list[list[float]]:
    """Each query takes one semaphore slot, one timeout, and its own retries.

    A query that returns is cached before the others finish, so a retry of a
    sibling does not embed it again. If one query exhausts its retries, the
    others are awaited before that error propagates. Their in-flight entries
    stay until those tasks finish, and a sibling another caller is waiting on
    is not cancelled when this caller is cancelled.
    """
    held: dict[tuple[str, str], list[float]] = {}
    _hold_cached(embeddings, texts, "query", held)
    unique = _uncached_unique(embeddings, texts, "query")
    model = embedding_model_id(embeddings)

    async def one(text: str, key: tuple[str, str]) -> None:
        try:
            if _cached_vector(embeddings, text, "query") is not None:
                return
            async with limiter.embed_semaphore:
                if _cached_vector(embeddings, text, "query") is not None:
                    return

                async def attempt() -> None:
                    if _cached_vector(embeddings, text, "query") is not None:
                        return
                    vector = await _invoke_one_query(embeddings, text)
                    _hold_stored(embeddings, text, "query", vector, held)

                await _retry_embedding(limiter, attempt)
        finally:
            _finish_flight(key, sys.exception())

    if unique:
        jobs = []
        for text in unique:
            key = _content_key(model, text, "query")
            jobs.append((one(text, key), [key]))
        await _await_query_group(jobs)
    collected = _vectors_from_hold(embeddings, texts, "query", held)
    if collected is None:
        raise RuntimeError("embedding cache missed a query it had claimed")
    return collected


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
    return await _embed_document_batches(
        embeddings,
        list(texts),
        batch_size=batch_size,
        limiter=limiter,
        role=role,
    )


async def _embed_document_batches(
    embeddings,
    texts: Sequence[str],
    *,
    batch_size: int,
    limiter,
    role: str,
) -> list[list[float]]:
    """One provider call per document chunk. Successes are cached before the next chunk.

    Same ownership as the query path: a failed chunk does not drop a sibling
    that is still embedding, and a later request for that sibling is a cache hit.
    """
    size = max(1, int(batch_size))
    held: dict[tuple[str, str], list[float]] = {}
    _hold_cached(embeddings, texts, role, held)
    pending = _uncached_unique(embeddings, texts, role)
    chunks = [pending[start : start + size] for start in range(0, len(pending), size)]
    model = embedding_model_id(embeddings)

    async def one(chunk: list[str], keys: list[tuple[str, str]]) -> None:
        try:
            async with limiter.embed_semaphore:
                needed = [text for text in chunk if _cached_vector(embeddings, text, role) is None]
                if not needed:
                    return

                async def attempt() -> None:
                    current = [text for text in needed if _cached_vector(embeddings, text, role) is None]
                    if not current:
                        return
                    vectors = await _embed_documents_batch(embeddings, current)
                    if len(vectors) != len(current):
                        raise RuntimeError("embedding model returned a different number of vectors than texts")
                    _check_vectors(embeddings, vectors)
                    for text, vector in zip(current, vectors):
                        _hold_stored(embeddings, text, role, vector, held)

                await _retry_embedding(limiter, attempt)
        finally:
            exc = sys.exception()
            for key in keys:
                _finish_flight(key, exc)

    if chunks:
        jobs = []
        for chunk in chunks:
            keys = [_content_key(model, text, role) for text in chunk]
            jobs.append((one(chunk, keys), keys))
        await _await_query_group(jobs)
    collected = _vectors_from_hold(embeddings, texts, role, held)
    if collected is None:
        raise RuntimeError("embedding cache missed a text it had claimed")
    return collected


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
