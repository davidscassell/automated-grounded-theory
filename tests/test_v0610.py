"""0.6.10: a failed query does not drop a sibling that is still embedding."""

import asyncio
import importlib.util
from pathlib import Path

import pytest

from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.retrieval import (
    _VECTOR_CACHE,
    _content_key,
    cached_aembed_texts,
    clear_embedding_cache,
    embedding_model_id,
)
from automated_grounded_theory.version import __version__


def test_version_is_0_6_10():
    assert __version__ == "0.9.22"
    assert POLICY_VERSION == "0.9.22"


def _limiter() -> CallLimiter:
    return CallLimiter(4, max_retries=1, backoff=0, max_backoff=0, request_timeout=0)


def _pending() -> list[asyncio.Task]:
    current = asyncio.current_task()
    return [task for task in asyncio.all_tasks() if task is not current and not task.done()]


def _flight(embeddings, text: str):
    key = _content_key(embedding_model_id(embeddings), text, "query")
    with _VECTOR_CACHE.lock:
        return _VECTOR_CACHE.inflight.get(key)


def _waiters(embeddings, text: str) -> int:
    flight = _flight(embeddings, text)
    return 0 if flight is None else flight.waiters


async def _until(predicate) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("timed out waiting for the embedding fixture")


class _Provider:
    """``bad`` exhausts retries. ``slow`` stays in the provider until released."""

    def __init__(self, *, batch: bool) -> None:
        self.batch = batch
        self.model = f"sibling-cleanup-{'batch' if batch else 'query'}-{id(self)}"
        self.dimensions = 2
        self.calls: list[str] = []
        self.in_flight = 0
        self.slow_started = asyncio.Event()
        self.release_bad = asyncio.Event()
        self.release_slow = asyncio.Event()

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def _one(self, text: str) -> list[float]:
        self.calls.append(text)
        self.in_flight += 1
        try:
            if text == "bad":
                await self.release_bad.wait()
                raise TimeoutError("bad")
            self.slow_started.set()
            await self.release_slow.wait()
            return [1.0, 0.0]
        finally:
            self.in_flight -= 1


class _QueryProvider(_Provider):
    def __init__(self) -> None:
        super().__init__(batch=False)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._one(text)


class _BatchProvider(_Provider):
    def __init__(self) -> None:
        super().__init__(batch=True)

    async def aembed_queries(self, texts: list[str]) -> list[list[float]]:
        return [await self._one(texts[0])]


def _provider(batch: bool) -> _Provider:
    return _BatchProvider() if batch else _QueryProvider()


async def _embed(embeddings, texts: list[str], limiter: CallLimiter) -> list[list[float]]:
    return await cached_aembed_texts(
        embeddings,
        texts,
        batch_size=1,
        limiter=limiter,
        role="query",
    )


async def _exhausted_failure_does_not_duplicate_the_sibling(batch: bool) -> None:
    clear_embedding_cache()
    embeddings = _provider(batch)
    limiter = _limiter()
    group = asyncio.create_task(_embed(embeddings, ["bad", "slow"], limiter))
    await embeddings.slow_started.wait()
    follow = asyncio.create_task(_embed(embeddings, ["slow"], limiter))
    await _until(lambda: _waiters(embeddings, "slow") >= 1 or follow.done())
    embeddings.release_bad.set()
    await _until(lambda: embeddings.calls.count("bad") >= 2)
    await asyncio.sleep(0)
    assert embeddings.calls.count("slow") == 1
    assert embeddings.in_flight >= 1
    assert _flight(embeddings, "slow") is not None
    embeddings.release_slow.set()
    with pytest.raises(TimeoutError):
        await group
    leaked = [task for task in _pending() if task is not follow]
    assert embeddings.in_flight == 0
    assert leaked == []
    assert await follow == [[1.0, 0.0]]
    assert embeddings.calls.count("bad") == 2
    assert embeddings.calls.count("slow") == 1
    assert embeddings.in_flight == 0
    assert _pending() == []
    clear_embedding_cache()


def test_exhausted_query_failure_does_not_duplicate_a_running_sibling():
    asyncio.run(_exhausted_failure_does_not_duplicate_the_sibling(False))


def test_exhausted_batch_failure_does_not_duplicate_a_running_sibling():
    asyncio.run(_exhausted_failure_does_not_duplicate_the_sibling(True))


async def _shared_sibling_keeps_running(batch: bool) -> None:
    clear_embedding_cache()
    embeddings = _provider(batch)
    limiter = _limiter()
    group = asyncio.create_task(_embed(embeddings, ["bad", "slow"], limiter))
    await embeddings.slow_started.wait()
    follower = asyncio.create_task(_embed(embeddings, ["slow"], limiter))
    await _until(lambda: _waiters(embeddings, "slow") >= 1)
    embeddings.release_bad.set()
    await _until(lambda: embeddings.calls.count("bad") >= 2)
    await asyncio.sleep(0)
    assert not group.done()
    assert embeddings.calls.count("slow") == 1
    assert embeddings.in_flight >= 1
    flight = _flight(embeddings, "slow")
    assert flight is not None and not flight.task.done()
    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower
    assert embeddings.calls.count("slow") == 1
    assert embeddings.in_flight >= 1
    flight = _flight(embeddings, "slow")
    assert flight is not None and flight.task is not None and not flight.task.done()
    embeddings.release_slow.set()
    with pytest.raises(TimeoutError):
        await group
    assert embeddings.calls.count("bad") == 2
    assert embeddings.calls.count("slow") == 1
    assert embeddings.in_flight == 0
    assert _pending() == []
    again = await _embed(embeddings, ["slow"], limiter)
    assert again == [[1.0, 0.0]]
    assert embeddings.calls.count("slow") == 1
    clear_embedding_cache()


def test_shared_query_sibling_keeps_running_after_the_other_waiter_is_cancelled():
    asyncio.run(_shared_sibling_keeps_running(False))


def test_shared_batch_sibling_keeps_running_after_the_other_waiter_is_cancelled():
    asyncio.run(_shared_sibling_keeps_running(True))


def _benchmark():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_review_modes.py"
    spec = importlib.util.spec_from_file_location("benchmark_review_modes_v0610", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unique_passages_are_not_four_reused_stems():
    benchmark = _benchmark()
    reused = [paragraph for row in benchmark.corpus() for paragraph in row["text"].split("\n\n")]
    unique = [paragraph for row in benchmark.corpus(unique=True) for paragraph in row["text"].split("\n\n")]
    assert len(reused) == 24
    assert len(set(reused)) == 4
    assert len(unique) == 24
    assert len(set(unique)) == 24
    assert len(set(unique) & set(reused)) == 0
