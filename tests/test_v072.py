"""0.7.2 focused-coding fixes: shares, coverage, compete, duplicates, spans, timeout."""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import __version__, focused_code
from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.focused import FOCUSED_SYSTEM, cite_note, plan_recode_queue
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.schemas import (
    FocusedCodeDecision,
    FocusedDimensionNote,
    FocusedPassageRecode,
    FocusedPropertyNote,
    FocusedRecodeBatch,
)

from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _book, _entry, _passages, _source


def test_version_is_072():
    assert __version__ == "0.9.30"
    assert POLICY_VERSION == "0.9.30"


def _passages_for_shares():
    rows = []
    texts = {
        "p1": "I was leaving despite the plan.",
        "p2": "I was staying despite the plan.",
        "p3": "The bags were already in the hall.",
        "p4": "Another bag waited by the stairs.",
        "p5": "The payroll office closed early.",
        "p6": "A receipt sat in the drawer.",
        "p7": "The hallway light stayed on.",
    }
    for index, (passage_id, text) in enumerate(texts.items(), start=1):
        rows.append(
            {
                "passage_id": passage_id,
                "doc_id": f"d{index}",
                "text": text,
                "start_offset": 0,
                "end_offset": len(text),
                "metadata": {"respondent_id": f"d{index}"},
            }
        )
    return rows


def _share_segments(passages):
    return [
        {
            "segment_id": "s1",
            "code_id": "c1",
            "passage_id": "p1",
            "doc_id": "d1",
            "quote": passages[0]["text"],
            "consequence": "leaving",
            "confidence": 0.9,
        },
        {
            "segment_id": "s2",
            "code_id": "c1",
            "passage_id": "p2",
            "doc_id": "d2",
            "quote": passages[1]["text"],
            "consequence": "staying",
            "confidence": 0.9,
        },
        {
            "segment_id": "s3",
            "code_id": "c1",
            "passage_id": "p3",
            "doc_id": "d3",
            "quote": passages[2]["text"],
            "confidence": 0.9,
        },
        {
            "segment_id": "s4",
            "code_id": "c1",
            "passage_id": "p4",
            "doc_id": "d4",
            "quote": passages[3]["text"],
            "confidence": 0.2,
        },
    ]


def test_shares_size_the_queue_and_limits_are_caps():
    passages = _passages_for_shares()
    segments = _share_segments(passages)
    codes = [_entry("c1", "packing bags", 4, ["d1", "d2", "d3", "d4"])]
    thirds = {"develop": 1 / 3, "compete": 1 / 3, "outside": 1 / 3}
    plan = plan_recode_queue(
        passages,
        segments,
        ["c1"],
        codes,
        seed=1,
        neighbor_limit=None,
        outside_limit=None,
        negative_case_limit=None,
        shares=thirds,
        max_calls=6,
    )
    assert plan["call_targets"]["develop"] == 2
    assert plan["call_targets"]["compete"] == 2
    assert plan["call_targets"]["outside"] == 2
    assert plan["call_targets"]["claim_compete"] == 0
    assert plan["purpose_counts"]["compete"] == 2
    assert plan["purpose_counts"]["outside"] == 2
    assert plan["purpose_counts"]["develop"] == 2
    compete_sources = {item["source"] for item in plan["queue"] if item["purpose"] == "compete"}
    assert "differing_consequence" in compete_sources
    capped = plan_recode_queue(
        passages,
        segments,
        ["c1"],
        codes,
        seed=1,
        neighbor_limit=0,
        outside_limit=1,
        negative_case_limit=0,
        shares=thirds,
        max_calls=6,
    )
    assert capped["purpose_counts"]["outside"] == 1
    assert all(item["source"] != "neighbor" for item in capped["queue"])


def test_sampling_seed_changes_membership_only_when_the_queue_is_cut():
    passages = _passages_for_shares()
    segments = [
        {
            "segment_id": f"s{index}",
            "code_id": "c1",
            "passage_id": passage["passage_id"],
            "quote": passage["text"],
            "confidence": 0.9,
        }
        for index, passage in enumerate(passages, start=1)
    ]
    codes = [_entry("c1", "packing bags", 7, ["d1"])]
    shares = {"develop": 1.0, "compete": 0.0, "outside": 0.0}
    fitting = [
        plan_recode_queue(
            passages[:2],
            segments[:2],
            ["c1"],
            codes,
            seed=3,
            neighbor_limit=0,
            outside_limit=0,
            negative_case_limit=0,
            shares=shares,
            max_calls=2,
            sampling_seed=sampling,
        )["queue"]
        for sampling in (1, 99)
    ]
    assert [item["passage_id"] for item in fitting[0]] == [item["passage_id"] for item in fitting[1]]
    cut = [
        {
            item["passage_id"]
            for item in plan_recode_queue(
                passages,
                segments,
                ["c1"],
                codes,
                seed=3,
                neighbor_limit=0,
                outside_limit=0,
                negative_case_limit=0,
                shares=shares,
                max_calls=2,
                sampling_seed=sampling,
            )["queue"]
        }
        for sampling in (1, 99)
    ]
    assert cut[0] != cut[1]


def test_applied_share_marks_a_purpose_that_received_fewer_calls():
    passages = _passages()[:3]
    segments = [
        {
            "segment_id": f"s{index}",
            "code_id": "c-freq",
            "code": "packing bags",
            "passage_id": passage["passage_id"],
            "doc_id": passage["doc_id"],
            "quote": passage["text"],
            "confidence": 0.9,
            "review_reasons": [],
        }
        for index, passage in enumerate(passages, start=1)
    ]
    result = focused_code(
        _source(passages=passages, codes=[_book()[0]], segments=segments),
        model=FakeFocusedCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=3,
        interrupt=False,
    )
    compete = result.purpose_coverage["compete"]
    assert compete["applied_share"] == 0
    assert compete["reserved_calls"] >= 1
    assert compete["reallocated_away"] is True
    assert any("reallocated" in note for note in result.warnings)
    assert result.metadata["status"] == "complete"
    assert result.focused_not_queued == []


def test_compete_seeds_before_claims_and_a_second_round_searches_them():
    passages = _passages_for_shares()[:3]
    segments = _share_segments(_passages_for_shares())[:2]
    plan = plan_recode_queue(
        passages,
        segments,
        ["c1"],
        [_entry("c1", "leaving home", 2, ["d1", "d2"])],
        seed=1,
        neighbor_limit=0,
        outside_limit=0,
        negative_case_limit=0,
        max_calls=3,
        claims=[],
    )
    compete = [item for item in plan["queue"] if item["purpose"] == "compete"]
    assert compete
    assert compete[0]["source"] == "differing_consequence"

    class _PoleCoder(FakeFocusedCoder):
        def batch(self, payload):
            rows = []
            for passage in payload.get("passages") or []:
                quote = str(passage.get("text") or "")
                code = (payload.get("raised_codes") or [{"code_id": "c1", "code": "leaving home"}])[0]
                rows.append(
                    FocusedPassageRecode(
                        passage_id=str(passage.get("passage_id") or ""),
                        codes=[
                            FocusedCodeDecision(
                                action="keep",
                                code=str(code.get("code") or ""),
                                code_id=str(code.get("code_id") or ""),
                                quote=quote,
                            )
                        ],
                        dimensions=[
                            FocusedDimensionNote(
                                category_id=str(code.get("code_id") or ""),
                                name="leaving",
                                pole="staying",
                                text="The other pole is staying.",
                                document_id=str(passage.get("doc_id") or ""),
                                passage_id=str(passage.get("passage_id") or ""),
                                quote=quote,
                            )
                        ],
                    )
                )
            return FocusedRecodeBatch(passages=rows)

    coder = _PoleCoder()
    result = focused_code(
        _source(
            passages=passages,
            codes=[_entry("c1", "leaving home", 1, ["d1"])],
            segments=segments[:1],
        ),
        model=coder,
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=4,
        interrupt=False,
    )
    assert coder.call_count >= 2
    claim_ids = result.purpose_coverage["claim_compete"]["passage_ids"]
    assert "p2" in claim_ids
    assert result.purpose_coverage["claim_compete"]["calls"] >= 1


def test_prompt_shows_the_full_codebook_and_flags_near_duplicates():
    assert "disconfirming" in FOCUSED_SYSTEM
    assert "drop" in FOCUSED_SYSTEM
    assert "Prefer a code already in the codebook" in FOCUSED_SYSTEM

    class _DuplicateCoder(FakeFocusedCoder):
        def batch(self, payload):
            assert payload.get("other_codes")
            assert payload.get("raised_codes")
            passage = payload["passages"][0]
            return FocusedRecodeBatch(
                passages=[
                    FocusedPassageRecode(
                        passage_id=passage["passage_id"],
                        codes=[
                            FocusedCodeDecision(
                                action="add",
                                code="packing the bags",
                                definition="Incidents of packing the bags.",
                                quote=passage["text"],
                            )
                        ],
                    )
                ]
            )

    passages = _passages()[:1]
    result = focused_code(
        _source(
            passages=passages,
            codes=_book()[:2],
            segments=[
                {
                    "segment_id": "s1",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": passages[0]["text"],
                    "review_reasons": [],
                }
            ],
        ),
        model=_DuplicateCoder(),
        seed=1,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=2,
        interrupt=False,
    )
    duplicate = next(entry for entry in result.codebook if entry["code"] == "packing the bags")
    assert duplicate["possible_duplicate"] is True
    assert duplicate["similar_code_id"] == "c-freq"
    assert duplicate["needs_review"] is True


def test_revise_and_drop_are_flagged_and_not_applied():
    class _ProposalCoder(FakeFocusedCoder):
        def batch(self, payload):
            passage = payload["passages"][0]
            code = payload["raised_codes"][0]
            return FocusedRecodeBatch(
                passages=[
                    FocusedPassageRecode(
                        passage_id=passage["passage_id"],
                        codes=[
                            FocusedCodeDecision(
                                action="drop",
                                code=code["code"],
                                code_id=code["code_id"],
                                quote=passage["text"],
                                rationale="This reading does not hold.",
                            ),
                            FocusedCodeDecision(
                                action="revise",
                                code=code["code"],
                                code_id=code["code_id"],
                                definition="A proposed definition that must wait for review.",
                                quote=passage["text"],
                            ),
                        ],
                    )
                ]
            )

    passages = _passages()[:1]
    text = passages[0]["text"]
    result = focused_code(
        _source(
            passages=passages,
            codes=[_book()[0]],
            segments=[
                {
                    "segment_id": "s1",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": text,
                    "definition": "Incidents of packing bags.",
                    "review_reasons": [],
                }
            ],
        ),
        model=_ProposalCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=2,
        interrupt=False,
    )
    segment = next(item for item in result.segments if item["segment_id"] == "s1")
    assert segment["quote"] == text
    assert "focused_drop_proposal" in segment["review_reasons"]
    assert "focused_revise_proposal" in segment["review_reasons"]
    assert segment["focused_drop_proposal"]["applied"] is False
    assert segment["focused_revise_proposal"]["applied"] is False
    assert "focused_disagreement" not in segment["review_reasons"]
    entry = next(item for item in result.codebook if item["code_id"] == "c-freq")
    assert entry["definition"] == "Incidents of packing bags."


def test_property_category_mismatch_is_separate_from_anchoring():
    passages = _passages()[:1]
    segments = [{"segment_id": "s1", "code_id": "c-freq", "passage_id": "p1"}]
    note = cite_note(
        {
            "category_id": "c-other",
            "name": "timing",
            "text": "When the incident sits.",
            "document_id": "d1",
            "passage_id": "p1",
            "quote": passages[0]["text"],
        },
        passages,
        kind="property",
        entry_id="prop-0001",
        coded_passage_id="p1",
        segments=segments,
    )
    assert note["supported"] is True
    assert note["quote_status"] == "exact"
    assert note["evidence_condition"] == "anchored"
    assert "property_category_mismatch" in note["flags"]
    assert note["needs_review"] is True
    result = focused_code(
        _source(
            passages=passages,
            codes=[
                _book()[0],
                _entry("c-other", "writing letters", 1, ["d9"]),
            ],
            segments=[
                {
                    "segment_id": "s1",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": passages[0]["text"],
                    "review_reasons": [],
                }
            ],
        ),
        model=_MismatchCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=2,
        interrupt=False,
    )
    stored = result.category_development
    assert stored["category_match"]["property_category_mismatch"] >= 1
    assert stored["category_match"]["anchored_but_mismatched"] >= 1
    assert "not category fit" in stored["category_match"]["note"]
    assert "not claim support" in stored["category_match"]["note"]
    mismatched = next(item for item in result.properties if "property_category_mismatch" in item["flags"])
    assert mismatched["supported"] is True


class _MismatchCoder(FakeFocusedCoder):
    def batch(self, payload):
        passage = payload["passages"][0]
        return FocusedRecodeBatch(
            passages=[
                FocusedPassageRecode(
                    passage_id=passage["passage_id"],
                    codes=[
                        FocusedCodeDecision(
                            action="keep",
                            code="packing bags",
                            code_id="c-freq",
                            quote=passage["text"],
                        )
                    ],
                    properties=[
                        FocusedPropertyNote(
                            category_id="c-other",
                            name="timing",
                            text="The property is when it happens.",
                            document_id=passage["doc_id"],
                            passage_id=passage["passage_id"],
                            quote=passage["text"],
                        )
                    ],
                )
            ]
        )


class _HangRunnable:
    async def ainvoke(self, payload):
        await asyncio.Event().wait()


class HangingFocusedCoder(BaseChatModel):
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "hanging-focused"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        async def _call(messages):
            self.call_count += 1
            await asyncio.Event().wait()

        return RunnableLambda(_call)


def test_wall_clock_timeout_stops_a_hanging_model():
    limiter = CallLimiter(1, max_retries=5, backoff=0.5, max_backoff=8, request_timeout=5, wall_timeout=0.25)
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        asyncio.run(limiter.ainvoke(_HangRunnable(), {"call": "hang"}))
    assert time.perf_counter() - started < 1.5

    passages = _passages()[:1]
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        focused_code(
            _source(
                passages=passages,
                codes=[_book()[0]],
                segments=[
                    {
                        "segment_id": "s1",
                        "code_id": "c-freq",
                        "code": "packing bags",
                        "passage_id": "p1",
                        "doc_id": "d1",
                        "quote": passages[0]["text"],
                        "review_reasons": [],
                    }
                ],
            ),
            model=HangingFocusedCoder(),
            seed=1,
            focused_min_score=0,
            focused_neighbor_limit=0,
            focused_outside_limit=0,
            focused_negative_case_limit=0,
            focused_request_timeout=0.25,
            focused_request_retries=4,
            max_focused_review_calls=2,
            interrupt=False,
            raise_on_timeout=True,
        )
    assert time.perf_counter() - started < 2
