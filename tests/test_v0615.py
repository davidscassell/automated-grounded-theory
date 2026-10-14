"""0.6.15: pending sweep evidence is not a silent finish, and a published vector survives eviction."""

import asyncio

import pytest

from automated_grounded_theory import open_code
from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.retrieval import (
    DEFAULT_MAX_FINAL_REVIEW_CALLS,
    DEFAULT_MAX_FINAL_REVIEW_TOKENS,
    _VECTOR_CACHE,
    _content_key,
    assignment_ids,
    cached_aembed_texts,
    clear_embedding_cache,
    code_needs_review,
    configure_embedding_cache,
    embedding_model_id,
)
from automated_grounded_theory.version import __version__
from tests.corpus import CONTEXT
from tests.fake_llm import FakeCoder


def test_version_is_0_6_15():
    assert __version__ == "0.7.7"
    assert POLICY_VERSION == "0.7.7"
    assert DEFAULT_MAX_FINAL_REVIEW_CALLS == 8
    assert DEFAULT_MAX_FINAL_REVIEW_TOKENS == 24_000


def _eleven_documents() -> list[dict]:
    """Ten packing-bags incidents plus one other code, so consolidation runs."""
    documents = []
    for index in range(10):
        documents.append(
            {
                "id": f"p{index}",
                "name": f"P{index}",
                "text": f"I spent the morning packing my suitcase. Note {index}.",
            }
        )
    documents.append(
        {
            "id": "c",
            "name": "C",
            "text": "I was crying at the station when the train doors closed.",
        }
    )
    return documents


def _run(**kwargs):
    settings = dict(
        model=FakeCoder(),
        seed=7,
        sweep_every=100,
        pipeline=False,
        batch_size=1,
        concurrency=1,
    )
    settings.update(kwargs)
    return open_code(_eleven_documents(), CONTEXT, **settings)


def _bags(result):
    return next(entry for entry in result.codebook if entry["code"] == "packing bags")


def test_final_review_sends_the_assignments_consolidation_left_pending():
    """The reviewer's run: 11 documents, 10 packing-bags assignments, two sweeps, four ids.

    The default allowance is enough to send the rest through the production graph.
    """
    result = _run()
    bags = _bags(result)
    ids = assignment_ids(bags)
    assert len(result.passages) == 11
    assert len(ids) == 10
    assert not code_needs_review(bags)
    assert set(bags["reviewed_assignment_ids"]) == set(ids)
    assert result.metadata["review_complete"] is True
    assert result.metadata["review_stop"] == "complete"
    assert result.metadata["consolidated"] is True
    assert result.metadata["assignments_available"] == 11
    assert result.metadata["assignments_sent"] == 11
    assert result.metadata["assignments_pending"] == 0
    assert result.metadata["pending_counts"] == {}
    assert result.metadata["pending_assignment_ids"] == {}
    assert result.metadata["review_by_code"]["packing bags"] == {"available": 10, "sent": 10, "pending": 0}
    assert 1 <= result.metadata["review_calls_used"] <= DEFAULT_MAX_FINAL_REVIEW_CALLS
    assert result.metadata["max_final_review_calls"] == DEFAULT_MAX_FINAL_REVIEW_CALLS
    assert result.metadata["max_final_review_tokens"] == DEFAULT_MAX_FINAL_REVIEW_TOKENS
    assert any(item.get("consolidation") for item in result.sweep_log)
    assert any(item.get("evidence_followup") for item in result.sweep_log)
    assert not any("not fully consolidated" in warning for warning in result.warnings)


def test_exhausted_final_review_allowance_is_an_incomplete_consolidation():
    """Same corpus. Zero follow-up calls leaves the four sent ids and says so."""
    result = _run(max_final_review_calls=0)
    bags = _bags(result)
    ids = assignment_ids(bags)
    sent = list(bags["reviewed_assignment_ids"])
    pending = result.metadata["pending_assignment_ids"]["packing bags"]
    assert len(ids) == 10
    assert len(sent) == 4
    assert code_needs_review(bags)
    assert set(pending) == set(ids) - set(sent)
    assert result.metadata["pending_counts"]["packing bags"] == 6
    assert result.metadata["review_complete"] is False
    assert result.metadata["consolidated"] is False
    assert result.metadata["review_stop"] == "call_allowance"
    assert result.metadata["review_calls_used"] == 0
    assert result.metadata["assignments_available"] == 11
    assert result.metadata["assignments_sent"] == 5
    assert result.metadata["assignments_pending"] == 6
    assert result.metadata["review_by_code"]["packing bags"]["sent"] == 4
    assert len(result.sweep_log) == 2
    assert result.sweep_log[-1]["consolidation"] is True
    warning = next(item for item in result.warnings if "not fully consolidated" in item)
    assert "review_complete is false" in warning
    assert "6 on packing bags" in warning
    for ident in pending:
        assert ident in warning

    one_call = _run(max_final_review_calls=1)
    one_bags = _bags(one_call)
    assert len(one_bags["reviewed_assignment_ids"]) == 6
    assert one_call.metadata["review_calls_used"] == 1
    assert one_call.metadata["review_stop"] == "call_allowance"
    assert one_call.metadata["review_complete"] is False
    assert one_call.metadata["pending_counts"]["packing bags"] == 4
    assert one_call.metadata["assignments_sent"] == 7
    assert one_call.metadata["assignments_pending"] == 4
    assert any(item.get("evidence_followup") for item in one_call.sweep_log)

    starved = _run(max_final_review_tokens=1)
    starved_bags = _bags(starved)
    assert len(starved_bags["reviewed_assignment_ids"]) == 4
    assert starved.metadata["review_calls_used"] == 0
    assert starved.metadata["review_stop"] == "token_allowance"
    assert starved.metadata["review_complete"] is False
    assert starved.metadata["consolidated"] is False
    assert starved.metadata["pending_counts"]["packing bags"] == 6
    assert any("token allowance" in warning and "not fully consolidated" in warning for warning in starved.warnings)

    with pytest.raises(ValueError):
        _run(max_final_review_calls=-1)


class _Gated:
    """``a`` and ``b`` succeed together. ``bad`` raises after they have returned."""

    dimensions = 2

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list = []
        self.started = {name: asyncio.Event() for name in ("a", "b", "bad")}
        self.release = {name: asyncio.Event() for name in ("a", "b", "bad")}
        self.returned = asyncio.Event()
        self.finished: list[tuple[str, ...]] = []

    async def _wait(self, texts: list[str]) -> None:
        self.calls.append(list(texts) if not isinstance(texts, str) else texts)
        names = [texts] if isinstance(texts, str) else list(texts)
        for name in names:
            self.started[name].set()
        for name in names:
            await self.release[name].wait()
        if "bad" in names:
            raise ConnectionError("bad sibling")
        self.finished.append(tuple(names))
        self.returned.set()


class _BatchQueries(_Gated):
    def __init__(self) -> None:
        super().__init__("review9-batch")

    async def aembed_queries(self, texts):
        await self._wait(list(texts))
        return [[float(ord(text[0])), 0.1] for text in texts]


class _SingleQuery(_Gated):
    def __init__(self) -> None:
        super().__init__("review9-single")

    async def aembed_query(self, text):
        await self._wait(text)
        if text == "bad":
            raise ConnectionError("bad sibling")
        self.returned.set()
        return [float(ord(text[0])), 0.2]


class _Documents(_Gated):
    def __init__(self) -> None:
        super().__init__("review9-docs")

    async def aembed_documents(self, texts):
        await self._wait(list(texts))
        return [[0.3, float(ord(text[0]))] for text in texts]


def _waiters(embeddings, text: str, role: str) -> int:
    key = _content_key(embedding_model_id(embeddings), text, role)
    with _VECTOR_CACHE.lock:
        flight = _VECTOR_CACHE.inflight.get(key)
    return 0 if flight is None else flight.waiters


async def _follower_keeps_the_vector(embeddings, role: str) -> None:
    clear_embedding_cache()
    configure_embedding_cache(vector_limit=1)
    limiter = CallLimiter(4, max_retries=0, backoff=0, max_backoff=0, request_timeout=0)
    owner = asyncio.create_task(
        cached_aembed_texts(embeddings, ["a", "b", "bad"], batch_size=2, limiter=limiter, role=role)
    )
    await embeddings.started["a"].wait()
    follower = asyncio.create_task(
        cached_aembed_texts(embeddings, ["a"], batch_size=2, limiter=limiter, role=role)
    )
    for _ in range(50):
        if _waiters(embeddings, "a", role) >= 1 or follower.done():
            break
        await asyncio.sleep(0)
    assert _waiters(embeddings, "a", role) >= 1 or follower.done()
    embeddings.release["a"].set()
    embeddings.release["b"].set()
    for _ in range(100):
        if any("a" in item for item in embeddings.finished):
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("a was not published")
    embeddings.release["bad"].set()
    with pytest.raises(ConnectionError, match="bad sibling"):
        await owner
    vector = await asyncio.wait_for(follower, timeout=1)
    if role == "document":
        assert vector == [[0.3, float(ord("a"))]]
    elif embeddings.model.endswith("batch"):
        assert vector == [[float(ord("a")), 0.1]]
    else:
        assert vector == [[float(ord("a")), 0.2]]
    # The follower must not embed ``a`` again after the sibling failure.
    flat = [name for call in embeddings.calls for name in (call if isinstance(call, list) else [call])]
    assert flat.count("a") == 1


def test_published_vector_survives_eviction_on_every_embed_path():
    """vector_limit=1 used to hand the follower ConnectionError: bad sibling."""

    async def run() -> None:
        batch = _BatchQueries()
        single = _SingleQuery()
        documents = _Documents()
        await _follower_keeps_the_vector(batch, "query")
        assert batch.calls[0] == ["a", "b"]
        await _follower_keeps_the_vector(single, "query")
        await _follower_keeps_the_vector(documents, "document")

    try:
        asyncio.run(run())
    finally:
        configure_embedding_cache(vector_limit=None, vector_bytes=None)
        clear_embedding_cache()
