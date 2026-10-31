"""0.6.9: long passages split without restarting the packer, and query embeddings do not share one timeout."""

import asyncio
import importlib.util
import time
from pathlib import Path

from automated_grounded_theory import open_code
from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.retrieval import aembed_texts, clear_embedding_cache
from automated_grounded_theory.version import __version__
from tests.fake_llm import FakeCoder


def test_version_is_0_6_9():
    assert __version__ == "0.9.41"
    assert POLICY_VERSION == "0.9.41"


def _paragraph(chars: int) -> str:
    word = "alpha "
    return (word * ((chars // len(word)) + 1))[:chars]


def _code_paragraph(chars: int):
    coder = FakeCoder()
    result = open_code(
        [{"id": "d1", "name": "Long", "text": _paragraph(chars)}],
        "What is happening?",
        model=coder,
        unit="paragraph",
        token_budget=4000,
        seed=1,
    )
    assert len(result.passages) == 1
    assert any("split into" in warning for warning in result.warnings)
    assert coder.call_count >= 2
    return result


def test_long_paragraph_splits_instead_of_restarting_the_packer():
    started = time.perf_counter()
    _code_paragraph(11_000)
    assert time.perf_counter() - started < 15


def test_hundred_thousand_character_paragraph_splits_quickly():
    started = time.perf_counter()
    _code_paragraph(100_000)
    assert time.perf_counter() - started < 20


class _SlowQueries:
    def __init__(self) -> None:
        self.model = "slow-queries-v069"
        self.dimensions = 2
        self.calls: list[str] = []
        self.in_flight = 0
        self.peak = 0

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.calls.append(text)
        return [1.0, float(len(text))]

    async def aembed_query(self, text):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.calls.append(text)
        try:
            await asyncio.sleep(0.03)
            return [1.0, float(len(text))]
        finally:
            self.in_flight -= 1


def test_four_queries_finish_inside_a_batch_timeout_and_each_is_embedded_once():
    asyncio.run(_four_queries_finish_inside_a_batch_timeout_and_each_is_embedded_once())


async def _four_queries_finish_inside_a_batch_timeout_and_each_is_embedded_once():
    clear_embedding_cache()
    embeddings = _SlowQueries()
    limiter = CallLimiter(4, max_retries=3, backoff=0, max_backoff=0, request_timeout=0.075)
    texts = [f"query {index} text" for index in range(4)]
    started = time.perf_counter()
    vectors = await aembed_texts(embeddings, texts, batch_size=8, limiter=limiter, role="query")
    elapsed = time.perf_counter() - started
    assert vectors == [[1.0, float(len(text))] for text in texts]
    assert embeddings.calls == texts
    assert embeddings.peak == 4
    # Sequential calls would take 4 * 30 ms. Overlap finishes inside the batch timeout.
    assert elapsed < 0.1
    again = await aembed_texts(embeddings, texts, batch_size=8, limiter=limiter, role="query")
    assert again == vectors
    assert embeddings.calls == texts
    clear_embedding_cache()


class _FlakyQuery:
    """The third query misses its first deadline. The others must not be repeated."""

    def __init__(self) -> None:
        self.model = "flaky-query-v069"
        self.dimensions = 2
        self.calls: list[str] = []

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def aembed_query(self, text):
        self.calls.append(text)
        if text == "q3" and self.calls.count("q3") == 1:
            await asyncio.sleep(0.2)
        else:
            await asyncio.sleep(0.01)
        return [1.0, 0.0]


def test_a_query_retry_does_not_embed_queries_that_already_succeeded():
    asyncio.run(_a_query_retry_does_not_embed_queries_that_already_succeeded())


async def _a_query_retry_does_not_embed_queries_that_already_succeeded():
    clear_embedding_cache()
    embeddings = _FlakyQuery()
    limiter = CallLimiter(4, max_retries=2, backoff=0, max_backoff=0, request_timeout=0.075)
    texts = ["q0", "q1", "q2", "q3"]
    vectors = await aembed_texts(embeddings, texts, batch_size=8, limiter=limiter, role="query")
    assert vectors == [[1.0, 0.0] for _ in texts]
    assert embeddings.calls.count("q0") == 1
    assert embeddings.calls.count("q1") == 1
    assert embeddings.calls.count("q2") == 1
    assert embeddings.calls.count("q3") == 2
    clear_embedding_cache()


class _BatchQueries:
    def __init__(self) -> None:
        self.model = "batch-queries-v069"
        self.dimensions = 2
        self.batches: list[list[str]] = []
        self.singles: list[str] = []

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def aembed_query(self, text):
        self.singles.append(text)
        await asyncio.sleep(0.03)
        return [1.0, float(len(text))]

    async def aembed_queries(self, texts):
        self.batches.append(list(texts))
        await asyncio.sleep(0.03)
        return [[1.0, float(len(text))] for text in texts]


def test_batch_query_api_embeds_the_chunk_once():
    asyncio.run(_batch_query_api_embeds_the_chunk_once())


async def _batch_query_api_embeds_the_chunk_once():
    clear_embedding_cache()
    embeddings = _BatchQueries()
    limiter = CallLimiter(4, max_retries=3, backoff=0, max_backoff=0, request_timeout=0.075)
    texts = [f"batch query {index}" for index in range(4)]
    vectors = await aembed_texts(embeddings, texts, batch_size=8, limiter=limiter, role="query")
    assert vectors == [[1.0, float(len(text))] for text in texts]
    assert embeddings.batches == [texts]
    assert embeddings.singles == []
    clear_embedding_cache()


def _benchmark():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_review_modes.py"
    spec = importlib.util.spec_from_file_location("benchmark_review_modes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_embedding_latency_mode_embeds_each_passage_query_once():
    clear_embedding_cache()
    benchmark = _benchmark()
    embeddings = benchmark.SlowEmbeddings(0.03)
    docs = []
    for doc_index in range(2):
        paragraphs = [
            f"I was packing suitcase number {doc_index}-{offset} before the train."
            for offset in range(4)
        ]
        docs.append({"id": f"d{doc_index}", "name": f"Doc {doc_index}", "text": "\n\n".join(paragraphs)})
    started = time.perf_counter()
    row = benchmark.run_mode(
        "embedding",
        delay=0,
        embeddings=embeddings,
        rows=docs,
        review_mode="wave",
        pipeline=False,
        batch_size=4,
        concurrency=4,
        sweep_every=100,
        temperature=0,
        unit="paragraph",
    )
    assert row["mode"] == "embedding"
    assert row["embedding_queries"] == len(embeddings.queries)
    assert embeddings.peak >= 2
    assert len(embeddings.queries) == len(set(embeddings.queries))
    assert len(embeddings.queries) >= 2
    assert time.perf_counter() - started < 5
    clear_embedding_cache()
