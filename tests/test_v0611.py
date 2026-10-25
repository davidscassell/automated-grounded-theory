"""0.6.11 regression tests for the eight review findings."""

import asyncio
import copy
import warnings

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory.api import CheckpointMismatch, open_code
from automated_grounded_theory.benchmark import quote_and_coverage_rates, unit_grid_from_annotations
from automated_grounded_theory.codebook import (
    _apply_counts,
    commit_passage,
    empty_counters,
    interpret_coding,
    sync_counts,
)
from automated_grounded_theory.evaluate import (
    aligned_assignment_agreement,
    assignment_ari,
    nominal_krippendorff_alpha,
)
from automated_grounded_theory.evaluate import _score
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.retrieval import (
    DEFAULT_SNAPSHOT_LIMIT,
    _VECTOR_CACHE,
    _content_key,
    aembed_texts,
    assignment_digest,
    cached_aembed_texts,
    clear_embedding_cache,
    configure_embedding_cache,
    embedding_cache_stats,
    embedding_model_id,
    embedding_run_scope,
    review_fingerprint,
    select_prompt_examples,
    shared_codebook_vectors,
)
from automated_grounded_theory.version import __version__
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder
from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.schemas import CodeProposal, PassageCoding


def test_version_is_0_6_11():
    assert __version__ == "0.9.32"
    assert POLICY_VERSION == "0.9.32"


def _coding(quote: str, label: str) -> dict:
    return {
        "codes": [
            {
                "action": "new",
                "code": label,
                "definition": f"Incidents of {label}.",
                "quote": quote,
                "confidence": 0.9,
                "rationale": "The incident is in the participant's words.",
            }
        ],
        "memo": "",
    }


def _holiday_passages():
    previous = "I resigned yesterday."
    current = "I enjoyed my holiday."
    document = f"{previous} {current}"
    start = len(previous) + 1
    previous_passage = {
        "passage_id": "p0",
        "doc_id": "d1",
        "doc_name": "Diary",
        "text": previous,
        "start_offset": 0,
        "end_offset": len(previous),
        "prev_text": "",
        "next_text": current,
        "document_text": document,
    }
    current_passage = {
        "passage_id": "p1",
        "doc_id": "d1",
        "doc_name": "Diary",
        "text": current,
        "start_offset": start,
        "end_offset": len(document),
        "prev_text": previous,
        "next_text": "",
        "document_text": document,
    }
    return previous_passage, current_passage, [previous_passage, current_passage]


def _interpret(coding, passage, passages, codebook=None):
    return interpret_coding(
        coding,
        passage,
        codebook or {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
        passages=passages,
        document_text=str(passage.get("document_text") or ""),
    )


def test_exact_quote_from_another_passage_is_flagged_and_reattributed():
    previous, current, passages = _holiday_passages()
    interpreted = _interpret(
        _coding("I resigned yesterday.", "resigning from work"),
        current,
        passages,
    )
    assert interpreted["quote_retries"] == []
    assert [item["code"] for item in interpreted["accepted"]] == ["resigning from work"]
    kept = interpreted["accepted"][0]
    assert kept["quote_status"] == "exact"
    assert kept["quote_source"] == "neighbor"
    assert kept["needs_review"] is True
    assert "evidence_outside_passage" in kept["review_reasons"]
    assert kept["coded_passage_id"] == "p1"
    assert kept["attributed_passage_id"] == "p0"
    flagged = [item for item in interpreted["review"] if item["reason"] == "evidence_outside_passage"]
    assert flagged
    assert flagged[0]["ids"]["coded_passage_id"] == "p1"
    assert flagged[0]["ids"]["attributed_passage_id"] == "p0"

    committed = commit_passage({}, [], [], {}, empty_counters(), current, interpreted)
    segment = committed["segments"][0]
    assert segment["passage_id"] == "p0"
    assert segment["coded_passage_id"] == "p1"
    assert segment["attributed_passage_id"] == "p0"
    entry = next(iter(committed["codebook"].values()))
    assert entry["frequency"] == 1
    assert entry["examples"][0]["passage_id"] == "p0"

    run = {"segments": committed["segments"], "passages": passages, "codebook": [], "memos": [], "sweep_log": [], "prompt_log": [], "saturation": []}
    metrics = _score(run, "")
    assert metrics["coverage_pct"] == 50.0
    rates = quote_and_coverage_rates(passages, committed["segments"])
    assert rates["uncoded_passages"] == 1
    same = aligned_assignment_agreement(run, run)
    assert same["mean_jaccard"] == 1.0
    assert assignment_ari(run, run) == 1.0
    counted_on_coded = {
        "segments": [
            {
                **segment,
                "passage_id": "p1",
                "attributed_passage_id": "p1",
            }
        ]
    }
    assert aligned_assignment_agreement(run, counted_on_coded)["mean_jaccard"] == 0.0
    from automated_grounded_theory.evaluate import _passage_labels

    assert set(_passage_labels(run)) == {"p0"}
    assert set(_passage_labels(counted_on_coded)) == {"p1"}

    inside = _interpret(_coding("I enjoyed my holiday.", "enjoying a holiday"), current, passages)
    exact = inside["accepted"][0]
    assert exact["quote_status"] == "exact"
    assert exact["quote_source"] == "passage"
    assert exact["needs_review"] is False
    assert "evidence_outside_passage" not in exact["review_reasons"]
    assert exact["coded_passage_id"] == "p1"
    assert exact["attributed_passage_id"] == "p1"


class _NeighborQuote(FakeCoder):
    """The holiday sentence is coded with a quote that sits in the previous sentence."""

    def code(self, text: str) -> PassageCoding:
        from automated_grounded_theory.prompts import extract_payload

        passages = extract_payload(text).get("passages") or []
        passage = str(passages[0].get("text") or "") if passages else ""
        if "holiday" not in passage.casefold():
            return PassageCoding(codes=[], memo="")
        return PassageCoding(
            codes=[
                CodeProposal(
                    action="new",
                    code="resigning from work",
                    definition="Leaving a job.",
                    quote="I resigned yesterday.",
                    rationale="The resignation is in the previous sentence.",
                    confidence=0.9,
                )
            ],
            memo="",
        )


def test_open_code_attributes_outside_evidence_and_keeps_the_code():
    result = open_code(
        [{"id": "diary", "name": "Diary", "text": "I resigned yesterday. I enjoyed my holiday."}],
        CONTEXT,
        model=_NeighborQuote(),
        seed=1,
        unit="sentence",
        sweep_every=25,
        validation_retries=0,
    )
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert segment["code"] == "resigning from work"
    assert segment["quote_status"] == "exact"
    assert segment["quote_source"] == "neighbor"
    assert segment["needs_review"] is True
    assert "evidence_outside_passage" in segment["review_reasons"]
    assert segment["coded_passage_id"] != segment["attributed_passage_id"]
    by_text = {item["text"]: item for item in result.passages}
    assert segment["attributed_passage_id"] == by_text["I resigned yesterday."]["passage_id"]
    assert segment["coded_passage_id"] == by_text["I enjoyed my holiday."]["passage_id"]
    assert by_text["I resigned yesterday."]["coverage_status"] == "coded"
    assert by_text["I enjoyed my holiday."]["coverage_status"] == "context_only"
    assert result.codebook[0]["examples"][0]["passage_id"] == segment["attributed_passage_id"]


def test_commit_passage_shares_structure_and_matches_full_counts():
    previous, current, passages = _holiday_passages()
    first = _interpret(_coding("I resigned yesterday.", "resigning from work"), previous, passages)
    committed = commit_passage({}, [], [], {}, empty_counters(), previous, first)
    book = committed["codebook"]
    segments = committed["segments"]
    before_book = copy.deepcopy(book)
    before_segments = copy.deepcopy(segments)
    shared_segment = segments[0]
    shared_event = next(iter(book.values()))["history"][0]
    second = _interpret(_coding("I enjoyed my holiday.", "enjoying a holiday"), current, passages, book)
    again = commit_passage(
        book,
        segments,
        committed["memos"],
        committed["highlight_ids"],
        committed["counters"],
        current,
        second,
    )
    assert book == before_book
    assert segments == before_segments
    assert again["segments"][0] is shared_segment
    assert next(iter(again["codebook"].values()))["history"][0] is shared_event
    synced = sync_counts(again["codebook"], again["segments"])
    for key, entry in again["codebook"].items():
        assert synced[key]["frequency"] == entry["frequency"]
        assert synced[key]["evidence_revision"] == entry["evidence_revision"]
        assert synced[key]["examples"] == entry["examples"]


def test_resume_rejects_a_different_question_unless_continuation_is_explicit():
    saver = InMemorySaver()
    kwargs = dict(seed=3, sweep_every=2, checkpointer=saver, thread_id="manifest-1", interrupt_before_sweep=True)
    paused = open_code(CORPUS, CONTEXT, model=FakeCoder(), **kwargs)
    assert paused.interrupted
    assert paused.metadata["checkpoint_manifest"]["context"]["research_question"] == CONTEXT["research_question"]
    changed = {**CONTEXT, "research_question": "What changed after the move?"}
    with pytest.raises(CheckpointMismatch) as raised:
        open_code(CORPUS, changed, model=FakeCoder(), resume=True, **kwargs)
    assert "context.research_question" in raised.value.differences
    assert raised.value.differences["context.research_question"]["old"] == CONTEXT["research_question"]
    assert raised.value.differences["context.research_question"]["new"] == changed["research_question"]

    with pytest.warns(UserWarning, match="changed checkpoint manifest"):
        continued = open_code(CORPUS, changed, model=FakeCoder(), resume=True, allow_changes=True, **kwargs)
    assert continued.metadata["context"]["research_question"] == changed["research_question"]
    assert continued.metadata["checkpoint_manifest"]["context"]["research_question"] == changed["research_question"]
    event = continued.metadata["continuations"][-1]
    assert event["event"] == "continuation"
    diff = event["differences"]["context.research_question"]
    assert diff["old"] == CONTEXT["research_question"]
    assert diff["new"] == changed["research_question"]
    assert any("changed checkpoint manifest" in note for note in continued.warnings)
    assert changed["research_question"] in __import__("json").dumps(continued.prompt_log)

    other = InMemorySaver()
    other_kwargs = dict(seed=3, sweep_every=2, checkpointer=other, thread_id="manifest-2", interrupt_before_sweep=True)
    open_code(CORPUS, CONTEXT, model=FakeCoder(), **other_kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finished = open_code(
            CORPUS,
            changed,
            model=FakeCoder(),
            resume=True,
            continuation=True,
            **other_kwargs,
        )
    assert any(issubclass(item.category, UserWarning) for item in caught)
    assert finished.metadata["context"]["research_question"] == changed["research_question"]
    assert finished.metadata["continuations"][-1]["event"] == "continuation"


class _GrowingEmbeddings:
    model = "snapshot-growth"
    dimensions = 4

    def embed_documents(self, texts):
        return [[float(len(text)), 0.0, 0.0, 1.0] for text in texts]

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


def _retained_vector_lists() -> int:
    """Canonical vectors plus any vector lists still stored inside snapshots."""
    with _VECTOR_CACHE.lock:
        count = len(_VECTOR_CACHE.vectors)
        for stored in _VECTOR_CACHE.snapshots.values():
            if not isinstance(stored, (list, tuple)) or not stored:
                continue
            first = stored[0]
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], float):
                count += len(stored)
        return count


def test_codebook_snapshots_share_canonical_vectors_and_are_bounded():
    clear_embedding_cache()
    configure_embedding_cache(snapshot_limit=4)
    embeddings = _GrowingEmbeddings()
    limiter = CallLimiter(4, max_retries=0, backoff=0, max_backoff=0, request_timeout=0)

    async def grow():
        with embedding_run_scope("snapshot-repro"):
            for count in range(1, 51):
                entries = [
                    {"code": f"code {index}", "definition": f"definition {index}", "examples": []}
                    for index in range(count)
                ]
                vectors = await shared_codebook_vectors(
                    embeddings,
                    entries,
                    limiter=limiter,
                    batch_size=16,
                )
                assert len(vectors) == count
            stats = embedding_cache_stats()
            assert stats["vectors"] == 50
            assert stats["snapshots"] == 4
            assert _retained_vector_lists() == 50
        after = embedding_cache_stats()
        assert after["vectors"] == 50
        assert after["snapshots"] == 0

    try:
        asyncio.run(grow())
    finally:
        configure_embedding_cache(snapshot_limit=DEFAULT_SNAPSHOT_LIMIT)
        clear_embedding_cache()


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
    def __init__(self) -> None:
        self.model = f"owner-lifecycle-{id(self)}"
        self.dimensions = 2
        self.calls: list[str] = []
        self.in_flight = 0
        self.slow_started = asyncio.Event()
        self.release_bad = asyncio.Event()
        self.release_slow = asyncio.Event()

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
    async def aembed_query(self, text: str) -> list[float]:
        return await self._one(text)


class _BatchProvider(_Provider):
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


async def _cancel_owner_during_settlement(batch: bool) -> None:
    clear_embedding_cache()
    embeddings = _provider(batch)
    limiter = _limiter()
    owner = asyncio.create_task(_embed(embeddings, ["bad", "slow"], limiter))
    await embeddings.slow_started.wait()
    follower = asyncio.create_task(_embed(embeddings, ["slow"], limiter))
    await _until(lambda: _waiters(embeddings, "slow") >= 1)
    embeddings.release_bad.set()
    await _until(lambda: embeddings.calls.count("bad") >= 2)
    await asyncio.sleep(0)
    assert not owner.done()
    assert embeddings.in_flight >= 1
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert embeddings.calls.count("slow") == 1
    assert embeddings.in_flight >= 1
    embeddings.release_slow.set()
    assert await follower == [[1.0, 0.0]]
    assert embeddings.calls.count("slow") == 1
    assert embeddings.in_flight == 0
    assert _pending() == []
    clear_embedding_cache()


def test_cancelling_the_query_owner_during_settlement_keeps_the_shared_sibling():
    asyncio.run(_cancel_owner_during_settlement(False))


def test_cancelling_the_batch_owner_during_settlement_keeps_the_shared_sibling():
    asyncio.run(_cancel_owner_during_settlement(True))


async def _stale_waiter_detaches_before_owner_cancel(batch: bool) -> None:
    clear_embedding_cache()
    embeddings = _provider(batch)
    limiter = _limiter()
    owner = asyncio.create_task(_embed(embeddings, ["slow"], limiter))
    await embeddings.slow_started.wait()
    joiner = asyncio.create_task(_embed(embeddings, ["slow", "bad"], limiter))
    await _until(lambda: _waiters(embeddings, "slow") >= 1)
    embeddings.release_bad.set()
    with pytest.raises(TimeoutError):
        await joiner
    assert _waiters(embeddings, "slow") == 0
    assert embeddings.in_flight >= 1
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    await _until(lambda: embeddings.in_flight == 0 and _pending() == [])
    assert embeddings.calls.count("slow") == 1
    assert embeddings.in_flight == 0
    assert _pending() == []
    clear_embedding_cache()


def test_query_joiner_detaches_a_slow_waiter_when_a_new_text_fails():
    asyncio.run(_stale_waiter_detaches_before_owner_cancel(False))


def test_batch_joiner_detaches_a_slow_waiter_when_a_new_text_fails():
    asyncio.run(_stale_waiter_detaches_before_owner_cancel(True))


class _DocumentProvider:
    def __init__(self) -> None:
        self.model = f"document-batches-{id(self)}"
        self.dimensions = 2
        self.calls: list[list[str]] = []
        self.slow_started = asyncio.Event()
        self.release_slow = asyncio.Event()

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def aembed_documents(self, texts):
        self.calls.append(list(texts))
        if texts == ["bad"]:
            raise TimeoutError("bad")
        if "slow" in texts:
            self.slow_started.set()
            await self.release_slow.wait()
        return [[1.0, 0.0] for _ in texts]


def test_document_batches_keep_a_sibling_that_succeeded():
    async def run():
        clear_embedding_cache()
        embeddings = _DocumentProvider()
        limiter = CallLimiter(4, max_retries=0, backoff=0, max_backoff=0, request_timeout=0)
        group = asyncio.create_task(
            aembed_texts(embeddings, ["bad", "slow"], batch_size=1, limiter=limiter, role="document")
        )
        await embeddings.slow_started.wait()
        embeddings.release_slow.set()
        with pytest.raises(TimeoutError):
            await group
        again = await aembed_texts(embeddings, ["slow"], batch_size=1, limiter=limiter, role="document")
        assert again == [[1.0, 0.0]]
        flat = [text for batch in embeddings.calls for text in batch]
        assert flat.count("slow") == 1
        assert flat.count("bad") == 1
        assert flat != ["bad", "slow", "slow"]
        clear_embedding_cache()

    asyncio.run(run())


def test_nominal_alpha_is_not_one_when_it_is_undefined_or_disagrees():
    disagreed = nominal_krippendorff_alpha([[1, 0]])
    assert disagreed.alpha == 0.0
    assert disagreed.undefined is False
    empty = nominal_krippendorff_alpha([])
    assert empty.alpha is None
    assert empty.undefined is True
    assert empty.reason == "no_paired_ratings"
    single = nominal_krippendorff_alpha([[1]])
    assert single.alpha is None
    assert single.undefined is True
    coded = {"segments": [{"passage_id": "p1", "code": "resigning from work", "attributed_passage_id": "p1"}]}
    absent = {"segments": []}
    compared = aligned_assignment_agreement(coded, absent)
    assert compared["mean_jaccard"] == 0.0
    assert compared["krippendorff_alpha"] == 0.0
    assert compared["krippendorff_alpha"] != 1
    assert compared["krippendorff_alpha_undefined"] is False
    same = {
        "segments": [
            {"passage_id": "p1", "code": "leaving"},
            {"passage_id": "p2", "code": "staying"},
        ]
    }
    agreed = aligned_assignment_agreement(same, same)
    assert agreed["krippendorff_alpha"] == 1.0
    assert agreed["krippendorff_alpha_undefined"] is False
    doc = aligned_assignment_agreement.__doc__ or ""
    assert "omitted" in doc
    assert "joint" in doc.casefold()


def test_muse_unit_grid_uses_human_spans_and_the_reviewed_mapping():
    passages = [
        {
            "passage_id": "p1",
            "doc_id": "d1",
            "text": "I resigned yesterday.",
            "start_offset": 0,
            "end_offset": len("I resigned yesterday."),
        }
    ]
    documents = [{"id": "d1", "text": "I resigned yesterday. I enjoyed my holiday."}]
    tags = [{"document": "d1", "excerpt": "I resigned yesterday.", "label": "resigning"}]
    segments = [{"passage_id": "p1", "code": "leaving a job", "doc_id": "d1"}]
    mappings = [
        {"human_code": "resigning", "llm_code": "leaving a job", "grade": "full", "confidence": 0.9}
    ]
    grid = unit_grid_from_annotations(passages, segments, tags, mappings, documents)
    assert grid["available"] is True
    assert grid["macro_f1"] == 1.0
    assert grid["codes"] == 1

    missing_map = unit_grid_from_annotations(passages, segments, tags, [], documents)
    assert missing_map["available"] is False
    assert missing_map["reason"] == "alignment_unavailable"
    assert missing_map["macro_f1"] is None

    weak = unit_grid_from_annotations(
        passages,
        segments,
        tags,
        [{"human_code": "resigning", "llm_code": "leaving a job", "grade": "full", "confidence": 0.2}],
        documents,
    )
    assert weak["available"] is False
    assert weak["reason"] == "alignment_unavailable"

    unlocated = unit_grid_from_annotations(
        passages,
        segments,
        [{"document": "d1", "excerpt": "not in the document", "label": "resigning"}],
        mappings,
        documents,
    )
    assert unlocated["available"] is False
    assert unlocated["reason"] == "span_alignment_unavailable"
    assert unlocated["macro_f1"] is None


def test_review_fingerprint_covers_every_example_and_prompts_stay_bounded():
    examples = [
        {"quote": "one", "passage_id": "p1", "doc_id": "d1"},
        {"quote": "two", "passage_id": "p1", "doc_id": "d1"},
        {"quote": "three", "passage_id": "p1", "doc_id": "d1"},
    ]
    entry = {
        "code": "packing bags",
        "definition": "putting clothes in a bag",
        "revision": 1,
        "frequency": 3,
        "examples": examples,
    }
    fingerprint = review_fingerprint(entry)
    fourth = {
        **entry,
        "examples": examples + [{"quote": "four", "passage_id": "p2", "doc_id": "d2"}],
    }
    assert review_fingerprint(fourth) != fingerprint
    assert assignment_digest(fourth) != assignment_digest(entry)
    assert review_fingerprint({**fourth, "frequency": 99}) == review_fingerprint(fourth)
    chosen = select_prompt_examples(fourth, limit=3)
    assert len(chosen) == 3
    assert any(item["passage_id"] == "p2" for item in chosen)
    assert [item["quote"] for item in select_prompt_examples(entry, limit=3)] == ["one", "two", "three"]

    applications = [
        {
            "segment_id": str(index),
            "passage_id": f"p{index}",
            "doc_id": "d",
            "quote": f"quote {index}",
            "code_key": "k",
            "doc_start": index,
            "doc_end": index + 1,
        }
        for index in range(8)
    ]
    eight = _apply_counts({"code": "k", "key": "k"}, applications, "k")
    nine = _apply_counts(
        {"code": "k", "key": "k"},
        applications
        + [
            {
                "segment_id": "8",
                "passage_id": "p8",
                "doc_id": "d",
                "quote": "quote 8",
                "code_key": "k",
                "doc_start": 8,
                "doc_end": 9,
            }
        ],
        "k",
    )
    assert len(eight["assignments"]) == 8
    assert len(nine["assignments"]) == 9
    assert nine["assignments"][-1]["quote"] == "quote 8"
    assert eight["evidence_revision"] != nine["evidence_revision"]
    assert review_fingerprint(eight) != review_fingerprint(nine)
