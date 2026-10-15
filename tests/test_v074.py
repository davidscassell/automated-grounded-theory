"""0.7.4 focused-coding fixes: claim search, reallocation wording, timeouts, alternatives."""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import __version__, focused_code
from automated_grounded_theory.async_support import CallLimiter, await_with_wall_timeout, run_sync
from automated_grounded_theory.focused import _claim_compete_items, _review_node
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.schemas import (
    FocusedAlternativeReading,
    FocusedCodeVerdict,
    FocusedCompeteBatch,
    FocusedCompetePassage,
)

from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _entry, _passages, _source


def test_version_is_074():
    assert __version__ == "0.7.8"
    assert POLICY_VERSION == "0.7.8"
    assert FocusedAlternativeReading.model_fields["alternative_fit"].is_required()


def _passage(passage_id: str, text: str, **extra) -> dict:
    row = {
        "passage_id": passage_id,
        "doc_id": passage_id,
        "text": text,
        "start_offset": 0,
        "end_offset": len(text),
        "metadata": {"respondent_id": passage_id},
    }
    row.update(extra)
    return row


def _segment(segment_id: str, code_id: str, passage_id: str, *, confidence: float = 0.9, reasons=None) -> dict:
    return {
        "segment_id": segment_id,
        "code_id": code_id,
        "code": code_id,
        "passage_id": passage_id,
        "doc_id": passage_id,
        "quote": passage_id,
        "confidence": confidence,
        "review_reasons": list(reasons or []),
        "needs_review": bool(reasons),
    }


def _state(passages, segments, claims, *, reread=None, codebook=None, seed: int = 1) -> dict:
    return {
        "passages": passages,
        "segments": segments,
        "properties": [item for item in claims if item.get("kind") != "dimension"],
        "dimensions": [item for item in claims if item.get("kind") == "dimension"],
        "codebook": {str(item["code_id"]): item for item in (codebook or [])},
        "focused_reread": list(reread or []),
        "metadata": {"procedure": {"seed": seed, "focused_review_below": 0.5}},
    }


def test_claim_search_pairs_a_claim_with_a_relevant_passage_only():
    passages = [
        _passage("p1", "alpha bravo"),
        _passage("p2", "charlie delta"),
        _passage("p3", "echo foxtrot suitcase"),
        _passage("p4", "golf hotel"),
        _passage("p5", "india juliet", embedding=[1.0, 0.0]),
        _passage("p6", "kilo lima", embedding=[0.0, 1.0]),
    ]
    claims = [
        {
            "entry_id": "prop-1",
            "category_id": "c1",
            "name": "timing",
            "text": "suitcase morning",
            "pole": "zzzz-absent",
            "passage_id": "p1",
            "embedding": [1.0, 0.0],
        }
    ]
    segments = [
        _segment("s1", "c1", "p1"),
        _segment("s2", "c1", "p2"),
        _segment("s3", "c-stay", "p4", confidence=0.2),
    ]
    book = [
        _entry("c1", "leaving home", 2, ["p1", "p2"]),
        _entry("c-stay", "staying home", 1, ["p4"], dimensions=[{"name": "timing", "value": "later"}]),
    ]
    state = _state(passages, segments, claims, reread=["p1"], codebook=book)
    found = _claim_compete_items(state, [], slots=4)
    sources = {item["passage_id"]: item["source"] for item in found}
    assert "p2" not in sources
    assert sources["p4"] == "claim_opposite_pole"
    assert sources["p3"] == "claim_similarity"
    assert sources["p5"] == "claim_similarity"
    assert "p6" not in sources
    assert "p1" not in sources

    flagged = _claim_compete_items(
        _state(
            [_passage("p1", "alpha"), _passage("p2", "bravo")],
            [_segment("s2", "c-other", "p2", confidence=0.95, reasons=["needs_review"])],
            [{"category_id": "c1", "name": "timing", "text": "qqqq", "pole": "zzzz", "passage_id": "p1"}],
            reread=["p1"],
            codebook=[_entry("c1", "leaving home", 1, ["p1"])],
        ),
        [],
        slots=1,
    )
    assert flagged == []

    low = _claim_compete_items(
        _state(
            [_passage("p1", "alpha"), _passage("p2", "bravo")],
            [_segment("s2", "c-other", "p2", confidence=0.1)],
            [{"category_id": "c1", "name": "timing", "text": "qqqq", "pole": "zzzz", "passage_id": "p1"}],
            reread=["p1"],
        ),
        [],
        slots=1,
    )
    assert low == []

    guaranteed = _claim_compete_items(
        _state(
            [_passage("p1", "alpha"), _passage("p2", "bravo"), _passage("p3", "charlie")],
            [_segment("s1", "c1", "p1", confidence=0.95)],
            [{"category_id": "c1", "name": "timing", "text": "qqqq", "pole": "zzzz", "passage_id": "p1"}],
            reread=["p1"],
            seed=1,
        ),
        [],
        slots=2,
    )
    assert guaranteed == []

    empty = _claim_compete_items(
        _state(
            [_passage("p1", "alpha")],
            [_segment("s1", "c1", "p1")],
            [{"category_id": "c1", "name": "timing", "text": "qqqq", "pole": "zzzz", "passage_id": "p1"}],
            reread=["p1"],
        ),
        [],
        slots=2,
    )
    assert empty == []


def test_claim_round_rereads_another_assignment_when_the_pole_is_absent():
    passages = [
        _passage("p1", "alpha bravo"),
        _passage("p2", "alpha charlie"),
        _passage("p3", "echo foxtrot"),
    ]
    segments = [
        _segment("s1", "c1", "p1", confidence=0.95),
        _segment("s2", "c1", "p2", confidence=0.95),
    ]
    result = focused_code(
        _source(
            passages=passages,
            codes=[_entry("c1", "leaving home", 2, ["p1", "p2"])],
            segments=segments,
        ),
        model=FakeFocusedCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0.5,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0.5,
        max_focused_review_calls=2,
        interrupt=False,
    )
    claim = result.purpose_coverage["claim_compete"]
    assert claim["calls"] >= 1
    assert claim["passage_ids"]
    assert set(claim["passage_ids"]) <= {"p1", "p2"}
    assert "p3" not in claim["passage_ids"]


def test_unspent_claim_calls_are_sent_and_described_as_reallocated():
    passages = [_passage("p1", "alpha bravo")]
    result = focused_code(
        _source(
            passages=passages,
            codes=[_entry("c1", "leaving home", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=FakeFocusedCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0.5,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0.5,
        max_focused_review_calls=2,
        interrupt=False,
    )
    assert result.purpose_coverage["claim_compete"]["calls"] == 0
    assert result.purpose_coverage["claim_compete"]["reserved_calls"] == 1
    assert result.purpose_coverage["outside"]["calls"] >= 1
    assert result.purpose_coverage["claim_compete"]["reallocated_away"] is True
    assert result.purpose_coverage["reallocation"]["reallocated"] is True
    note = next(item for item in result.warnings if "claim_compete" in item and "of 2" in item)
    assert "reallocated to outside" in note
    assert "unused" not in note


def _review_state(*, transfers=None, reallocated=False, purposes=None):
    return {
        "passages": [_passage("p1", "alpha")],
        "recode_queue": [{"passage_id": "p1", "purpose": "develop", "source": "assignment"}],
        "focused_reread": ["p1"],
        "focused_not_reread": [],
        "focused_calls_used": 18,
        "purpose_calls": {"develop": 18, "compete": 0, "outside": 0, "claim_compete": 0},
        "purpose_tokens": {},
        "purpose_provider_tokens": {},
        "warnings": [],
        "recode_stop": "complete",
        "metadata": {
            "procedure": {
                "max_focused_review_calls": 24,
                "focused_share_develop": 0.25,
                "focused_share_compete": 0.25,
                "focused_share_outside": 0.25,
                "focused_share_claim_compete": 0.25,
            },
            "reread_call_targets": {"develop": 6, "compete": 6, "outside": 6, "claim_compete": 6},
            "reread_shares_applied": {
                "develop": 0.25,
                "compete": 0.25,
                "outside": 0.25,
                "claim_compete": 0.25,
            },
            "reread_reallocation": {
                "reallocated": reallocated,
                "purposes": list(purposes or []),
                "note": "",
            },
            "call_transfers": list(transfers or []),
        },
    }


def test_reallocation_warning_uses_the_allowance_and_matches_the_flag():
    unused = _review_node(_review_state())
    coverage = unused["purpose_coverage"]
    assert coverage["reallocation"]["reallocated"] is False
    assert coverage["claim_compete"]["reallocated_away"] is False
    assert coverage["compete"]["reallocated_away"] is False
    claim_note = next(item for item in unused["warnings"] if item.startswith("Focused reread purpose claim_compete"))
    assert "reserved 6 of 24" in claim_note
    assert "6 calls were unused" in claim_note
    assert "reallocated" not in claim_note

    moved = _review_node(
        _review_state(transfers=[{"from": "claim_compete", "to": "outside", "calls": 6}])
    )
    moved_coverage = moved["purpose_coverage"]
    assert moved_coverage["reallocation"]["reallocated"] is True
    assert moved_coverage["claim_compete"]["reallocated_away"] is True
    assert moved_coverage["compete"]["reallocated_away"] is False
    moved_note = next(item for item in moved["warnings"] if item.startswith("Focused reread purpose claim_compete"))
    assert "reserved 6 of 24" in moved_note
    assert "reallocated to outside" in moved_note
    assert "unused" not in moved_note
    compete_note = next(item for item in moved["warnings"] if item.startswith("Focused reread purpose compete"))
    assert "were unused" in compete_note
    assert moved_coverage["reallocation"]["purposes"] == ["claim_compete"]


class _SwallowRunnable:
    def __init__(self):
        self.cancelled = 0

    async def ainvoke(self, payload):
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1


class SwallowingFocusedCoder(BaseChatModel):
    call_count: int = 0
    cancelled: int = 0

    @property
    def _llm_type(self) -> str:
        return "swallowing-focused"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        parent = self

        async def _call(messages):
            parent.call_count += 1
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    parent.cancelled += 1

        return RunnableLambda(_call)


def test_wall_clock_timeout_cancels_a_model_that_swallows_cancellation():
    runnable = _SwallowRunnable()
    limiter = CallLimiter(1, max_retries=4, backoff=0.5, max_backoff=8, request_timeout=0.25, wall_timeout=0.25)
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_sync(limiter.ainvoke(runnable, {"call": "hang"}), timeout=2.0)
    assert time.perf_counter() - started < 0.8
    assert runnable.cancelled >= 1

    async def _swallow():
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                continue

    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_sync(_swallow(), timeout=0.2)
    assert time.perf_counter() - started < 1.0

    async def _inner_swallow():
        async def _hang():
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    continue

        return await await_with_wall_timeout(_hang(), 0.2)

    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_sync(_inner_swallow(), timeout=1.0)
    assert time.perf_counter() - started < 1.0

    coder = SwallowingFocusedCoder()
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        focused_code(
            _source(
                passages=_passages()[:1],
                codes=[_entry("c-freq", "packing bags", 1, ["d1"])],
                segments=[
                    {
                        "segment_id": "s1",
                        "code_id": "c-freq",
                        "code": "packing bags",
                        "passage_id": "p1",
                        "doc_id": "d1",
                        "quote": _passages()[0]["text"],
                        "review_reasons": [],
                    }
                ],
            ),
            model=coder,
            seed=1,
            focused_min_score=0,
            focused_neighbor_limit=0,
            focused_outside_limit=0,
            focused_negative_case_limit=0,
            focused_request_timeout=0.25,
            focused_request_retries=4,
            max_focused_review_calls=4,
            interrupt=False,
        )
    assert time.perf_counter() - started < 1.0
    assert coder.cancelled >= 1


def _compete_result(alternative: FocusedAlternativeReading, *, memo: str = ""):
    passages = _passages()[:1]
    text = passages[0]["text"]

    class _Coder(FakeFocusedCoder):
        def compete_batch(self, payload):
            passage = payload["passages"][0]
            return FocusedCompeteBatch(
                passages=[
                    FocusedCompetePassage(
                        passage_id=passage["passage_id"],
                        verdicts=[
                            FocusedCodeVerdict(
                                code_id="c-freq",
                                code="packing bags",
                                verdict="keep",
                                rationale="Kept after considering a disconfirming reading.",
                                quote=passage["text"],
                            )
                        ],
                        strongest_alternative=alternative,
                        memo=memo,
                    )
                ]
            )

    return focused_code(
        _source(
            passages=passages,
            codes=[
                _entry("c-freq", "packing bags", 1, ["d1"]),
                _entry("c-rare", "writing letters", 1, ["d1"]),
                _entry("c-hour", "marking the hour", 1, ["d9"]),
            ],
            segments=[
                {
                    "segment_id": "s1",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": text,
                    "review_reasons": [],
                },
                {
                    "segment_id": "s2",
                    "code_id": "c-rare",
                    "code": "writing letters",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": text,
                    "review_reasons": [],
                },
            ],
        ),
        model=_Coder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0,
        focused_share_compete=1,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        max_focused_review_calls=1,
        interrupt=False,
    )


def test_self_reference_is_invalid_and_only_a_stronger_fit_is_flagged():
    self_ref = _compete_result(
        FocusedAlternativeReading(
            code="packing bags",
            code_id="c-freq",
            rationale="This names the code that was kept.",
            alternative_fit="stronger",
            quote="packing",
        )
    )
    segment = next(item for item in self_ref.segments if item["segment_id"] == "s1")
    assert "alternative_is_kept_code" in segment["review_reasons"]
    assert segment["alternative_is_kept_code"]["invalid"] is True
    assert segment["alternative_is_kept_code"]["applied"] is False
    assert segment["alternative_is_kept_code"]["code_id"] == "c-freq"
    assert "alternative_named_not_chosen" not in segment["review_reasons"]

    weaker = _compete_result(
        FocusedAlternativeReading(
            code="marking the hour",
            code_id="c-hour",
            rationale="Marking the hour was considered and fits less well.",
            alternative_fit="weaker",
            quote="letters",
        ),
        memo="writing letters fits this passage better than packing bags.",
    )
    weaker_segment = next(item for item in weaker.segments if item["segment_id"] == "s1")
    assert "alternative_named_not_chosen" not in weaker_segment["review_reasons"]
    assert "alternative_is_kept_code" not in weaker_segment["review_reasons"]

    comparable = _compete_result(
        FocusedAlternativeReading(
            code="marking the hour",
            code_id="c-hour",
            rationale="The two readings are comparable.",
            alternative_fit="comparable",
            quote="letters",
        )
    )
    comparable_segment = next(item for item in comparable.segments if item["segment_id"] == "s1")
    assert "alternative_named_not_chosen" not in comparable_segment["review_reasons"]

    stronger = _compete_result(
        FocusedAlternativeReading(
            code="marking the hour",
            code_id="c-hour",
            rationale="Marking the hour fits better than the kept code.",
            alternative_fit="stronger",
            quote="letters",
        )
    )
    stronger_segment = next(item for item in stronger.segments if item["segment_id"] == "s1")
    assert "alternative_named_not_chosen" in stronger_segment["review_reasons"]
    assert stronger_segment["alternative_named_not_chosen"]["alternatives"][0]["alternative_fit"] == "stronger"
