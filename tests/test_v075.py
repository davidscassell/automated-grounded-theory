"""0.7.5 focused-coding fixes: claim tests, timeouts, alternatives, proposals."""

from __future__ import annotations

import json
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import __version__, focused_code, proposal_stability
from automated_grounded_theory.async_support import CallLimiter, run_sync
from automated_grounded_theory.backends import provider_http_timeout
from automated_grounded_theory.batching import schema_token_count
from automated_grounded_theory.focused import FOCUSED_CLAIM_TEST_SYSTEM, consolidate_code_proposals
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.schemas import (
    FocusedAlternativeReading,
    FocusedClaimJudgment,
    FocusedClaimTest,
    FocusedCodeVerdict,
    FocusedCompeteBatch,
    FocusedCompetePassage,
)

from tests.fake_focused import FakeFocusedCoder
from tests.test_decision_mode import FakeDecisionModel
from tests.test_v070 import _entry, _passages, _source
from tests.test_v074 import _passage, _segment


def test_version_is_075():
    assert __version__ == "0.8.6"
    assert POLICY_VERSION == "0.8.6"


def _prompt_text(result) -> str:
    chunks = []
    for row in result.prompt_log:
        for message in row.get("messages") or []:
            chunks.append(str(message.get("content") or ""))
    return "\n".join(chunks)


def _claim_source():
    passages = [
        _passage("p1", "I spent the morning packing my suitcase."),
        _passage("p2", "Packing the suitcase took all night."),
    ]
    return _source(
        passages=passages,
        codes=[_entry("c1", "leaving home", 2, ["p1", "p2"])],
        segments=[_segment("s1", "c1", "p1"), _segment("s2", "c1", "p2")],
    )


def _claim_run(model, **extra):
    return focused_code(
        _claim_source(),
        model=model,
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
        interrupt=extra.pop("interrupt", False),
        **extra,
    )


def _timing_property(result):
    return next(
        row
        for row in result.properties
        if row.get("name") == "timing" and str(row.get("kind") or "property") == "property"
    )


class _ContradictClaims(FakeFocusedCoder):
    def claim_test_batch(self, payload):
        rows = []
        for passage in payload.get("passages") or []:
            claim = passage.get("claim") or {}
            rows.append(
                FocusedClaimJudgment(
                    passage_id=str(passage.get("passage_id") or ""),
                    entry_id=str(claim.get("entry_id") or ""),
                    judgment="contradicts",
                    rationale="This passage is a negative case for the supplied claim.",
                    quote=str(passage.get("text") or ""),
                )
            )
        return FocusedClaimTest(passages=rows)


def test_claim_test_prompt_names_the_claim_and_supports_is_not_a_code_change():
    result = _claim_run(FakeFocusedCoder())
    text = _prompt_text(result)
    assert FOCUSED_CLAIM_TEST_SYSTEM in text
    assert "The incident is tied to this passage." in text
    assert "leaving home" in text
    assert "I spent the morning packing my suitcase." in text
    prop = _timing_property(result)
    assert prop.get("claim_support", "undefined") == "undefined"
    assert "model_claim_support" not in prop
    assert prop.get("review_actor") != "human"
    assert prop.get("evidence_condition") != "disputed"
    assert not prop.get("negative_cases")
    assert prop["claim_tests"][-1]["judgment"] == "supports"
    assert prop["claim_tests"][-1]["applied"] is False
    assert prop["claim_tests"][-1]["actor"] == "model"
    assert result.metadata["claim_test_counts"]["supports"] >= 1
    assert result.metadata["claim_test_counts"]["contradicts"] == 0


def test_claim_test_contradiction_is_a_negative_case_and_leaves_claim_support_unset():
    result = _claim_run(_ContradictClaims())
    prop = _timing_property(result)
    assert prop["evidence_condition"] == "disputed"
    assert prop["downstream_eligible"] is False
    assert prop.get("claim_support", "undefined") == "undefined"
    assert "model_claim_support" not in prop
    assert prop.get("review_actor") != "human"
    assert "claim_test_negative" in prop["flags"]
    case = prop["negative_cases"][-1]
    assert case["judgment"] == "contradicts"
    assert case["applied"] is False
    assert case["flagged"] is True
    assert case["actor"] == "model"
    assert case["source"] == "claim_test"
    assert result.metadata["claim_test_counts"]["contradicts"] >= 1


def test_llm_claim_test_keeps_model_provenance():
    decider = FakeDecisionModel(raised=["c1"], support="supported")
    result = _claim_run(
        _ContradictClaims(),
        decision_mode="llm",
        decision_model=decider,
        interrupt=True,
    )
    prop = _timing_property(result)
    assert prop.get("claim_support", "undefined") == "undefined"
    assert prop["model_claim_support"] == "supported"
    assert prop["review_actor"] == "llm"
    assert prop["evidence_condition"] == "disputed"
    assert prop["downstream_eligible"] is False
    assert prop["negative_cases"][-1]["judgment"] == "contradicts"
    assert prop["negative_cases"][-1]["actor"] == "model"
    assert decider.calls >= 1


def test_reallocated_claim_calls_are_sent_on_the_recode_schema():
    coder = FakeFocusedCoder()
    result = focused_code(
        _source(
            passages=[_passage("p1", "alpha bravo")],
            codes=[_entry("c1", "leaving home", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=coder,
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0.5,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0.5,
        max_focused_review_calls=4,
        interrupt=False,
    )
    assert coder.call_count >= 2
    assert result.purpose_coverage["claim_compete"]["calls"] == 0
    assert result.purpose_coverage["outside"]["calls"] >= 1
    assert result.purpose_coverage["reallocation"]["reallocated"] is True
    note = next(item for item in result.warnings if "claim_compete" in item and "of 4" in item)
    assert "reallocated to outside" in note
    assert FOCUSED_CLAIM_TEST_SYSTEM not in _prompt_text(result)


class _LoopBlock:
    async def ainvoke(self, payload):
        time.sleep(5)
        return {}


class _BlockingFocusedCoder(BaseChatModel):
    """time.sleep inside async ainvoke, on the loop thread. Not an Event().wait()."""

    @property
    def _llm_type(self) -> str:
        return "blocking-focused"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        async def _call(messages):
            time.sleep(5)
            return {}

        return RunnableLambda(_call)


def test_sync_read_on_the_call_thread_hits_the_request_timeout():
    limiter = CallLimiter(1, max_retries=4, backoff=0.5, max_backoff=8, request_timeout=0.3, wall_timeout=0.3)
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_sync(limiter.ainvoke(_LoopBlock(), {"call": "block"}), timeout=3.0)
    assert time.perf_counter() - started < 1.0

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
            model=_BlockingFocusedCoder(),
            seed=1,
            focused_min_score=0,
            focused_neighbor_limit=0,
            focused_outside_limit=0,
            focused_negative_case_limit=0,
            focused_share_develop=1,
            focused_share_compete=0,
            focused_share_outside=0,
            focused_share_claim_compete=0,
            focused_request_timeout=0.3,
            focused_request_retries=4,
            max_focused_review_calls=4,
            interrupt=False,
            raise_on_timeout=True,
        )
    # The outer bridge is request_timeout * call allowance + 1s (2.2s here).
    assert time.perf_counter() - started < 1.0


def test_provider_timeout_bounds_each_phase_and_schema_counts_are_cached():
    timeout = provider_http_timeout(30)
    assert timeout is not None
    assert timeout.connect == 10
    assert timeout.read == 30
    assert timeout.write == 30
    assert timeout.pool == 30
    assert provider_http_timeout(0) is None
    first = schema_token_count(FocusedClaimTest)
    assert schema_token_count(FocusedClaimTest) == first
    assert first > 0


def _alternative_result(alternative: FocusedAlternativeReading):
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
                    )
                ]
            )

    result = focused_code(
        _source(
            passages=passages,
            codes=[
                _entry("c-freq", "packing bags", 1, ["d1"]),
                _entry("c-rare", "writing letters", 1, ["d1"]),
                _entry("c-hour", "marking the hour", 1, ["d9"]),
                _entry("c-fold", "folding clothes", 1, ["d9"]),
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
    return result


def _compete_payload(result) -> dict:
    for row in result.prompt_log:
        for message in row.get("messages") or []:
            content = str(message.get("content") or "")
            if content.startswith("{"):
                return json.loads(content)
    raise AssertionError("compete prompt was not stored")


def test_alternatives_exclude_assigned_codes_and_reject_id_name_mismatches():
    kept_label = _alternative_result(
        FocusedAlternativeReading(
            code="packing bags",
            code_id="c-hour",
            rationale="The label is the kept code and the id is a different code.",
            alternative_fit="stronger",
            quote="packing",
        )
    )
    segment = next(item for item in kept_label.segments if item["segment_id"] == "s1")
    assert "alternative_is_kept_code" in segment["review_reasons"]
    assert segment["alternative_is_kept_code"]["invalid"] is True
    assert segment["alternative_is_kept_code"]["actor"] == "model"
    assert "alternative_named_not_chosen" not in segment["review_reasons"]

    mismatch = _alternative_result(
        FocusedAlternativeReading(
            code="folding clothes",
            code_id="c-hour",
            rationale="The id and the label name two different allowed codes.",
            alternative_fit="stronger",
            quote="folding",
        )
    )
    mismatched = next(item for item in mismatch.segments if item["segment_id"] == "s1")
    assert "alternative_id_name_mismatch" in mismatched["review_reasons"]
    assert mismatched["alternative_id_name_mismatch"]["invalid"] is True
    assert "alternative_named_not_chosen" not in mismatched["review_reasons"]

    novel = _alternative_result(
        FocusedAlternativeReading(
            code="inventing a ritual",
            code_id="",
            rationale="This label is not in the codebook.",
            alternative_fit="stronger",
            quote="ritual",
        )
    )
    novel_segment = next(item for item in novel.segments if item["segment_id"] == "s1")
    assert "alternative_named_not_chosen" in novel_segment["review_reasons"]
    assert "alternative_is_kept_code" not in novel_segment["review_reasons"]

    payload = _compete_payload(kept_label)
    passage = payload["passages"][0]
    excluded = set(passage["excluded_alternative_code_ids"])
    allowed = {item["code_id"] for item in passage["allowed_alternatives"]}
    assert {"c-freq", "c-rare"} <= excluded
    assert "c-hour" in allowed and "c-fold" in allowed
    assert not allowed & excluded


class _SplitRevise(FakeFocusedCoder):
    def compete_batch(self, payload):
        passage = payload["passages"][0]
        passage_id = str(passage.get("passage_id") or "")
        definition = "Packing as departure." if passage_id == "p1" else "Packing as storage."
        verdicts = []
        for code in passage.get("initial_codes") or []:
            code_id = str(code.get("code_id") or "")
            if code_id == "c-freq":
                verdicts.append(
                    FocusedCodeVerdict(
                        code_id="c-freq",
                        code="packing bags",
                        verdict="revise",
                        rationale="The definition should change for this passage.",
                        revised_definition=definition,
                        quote=str(passage.get("text") or ""),
                    )
                )
            else:
                verdicts.append(
                    FocusedCodeVerdict(
                        code_id=code_id,
                        code=str(code.get("code") or ""),
                        verdict="keep",
                        rationale="This code still fits the passage.",
                        quote=str(passage.get("text") or ""),
                    )
                )
        return FocusedCompeteBatch(
            passages=[
                FocusedCompetePassage(
                    passage_id=passage_id,
                    verdicts=verdicts,
                    strongest_alternative=FocusedAlternativeReading(
                        code="marking the hour",
                        code_id="c-hour",
                        rationale="A different code was considered and fits less well.",
                        alternative_fit="weaker",
                        quote=str(passage.get("text") or "")[:12],
                    ),
                )
            ]
        )


def _two_passage_compete(**extra):
    passages = _passages()[:2]
    segments = []
    for index, passage in enumerate(passages, start=1):
        for code_id, label in (("c-freq", "packing bags"), ("c-rare", "writing letters")):
            segments.append(
                {
                    "segment_id": f"s-{passage['passage_id']}-{code_id}",
                    "code_id": code_id,
                    "code": label,
                    "passage_id": passage["passage_id"],
                    "doc_id": passage["doc_id"],
                    "quote": passage["text"],
                    "review_reasons": [],
                }
            )
    return focused_code(
        _source(
            passages=passages,
            codes=[
                _entry("c-freq", "packing bags", 2, ["d1", "d2"]),
                _entry("c-rare", "writing letters", 2, ["d1", "d2"]),
                _entry("c-hour", "marking the hour", 1, ["d9"]),
            ],
            segments=segments,
        ),
        model=_SplitRevise(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0,
        focused_share_compete=1,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        max_focused_review_calls=2,
        interrupt=extra.pop("interrupt", False),
        **extra,
    )


def test_conflicting_revise_proposals_stay_unapplied_with_model_provenance():
    direct = consolidate_code_proposals(
        [
            {
                "code_id": "c1",
                "passage_id": "p1",
                "focused_revise_proposal": {"definition": "one", "applied": False, "actor": "model"},
            },
            {
                "code_id": "c1",
                "passage_id": "p2",
                "focused_drop_proposal": {"definition": "", "applied": False, "actor": "model"},
            },
        ]
    )
    assert direct[0]["conflict"] is True
    assert direct[0]["excluded"] is True
    assert direct[0]["applied"] is False
    assert direct[0]["actor"] == "model"
    assert {item["kind"] for item in direct[0]["proposals"]} == {
        "focused_revise_proposal",
        "focused_drop_proposal",
    }

    result = _two_passage_compete()
    group = next(item for item in result.metadata["proposal_consolidation"] if item["code_id"] == "c-freq")
    assert group["conflict"] is True
    assert group["excluded"] is True
    assert group["applied"] is False
    assert group["actor"] == "model"
    assert len({item["definition"] for item in group["proposals"]}) == 2
    assert all(item["actor"] == "model" and item["applied"] is False for item in group["proposals"])
    flagged = [item for item in result.segments if item.get("code_id") == "c-freq" and item.get("proposal_conflict")]
    assert flagged
    assert flagged[0]["focused_revise_proposal"]["applied"] is False
    assert flagged[0]["focused_revise_proposal"]["actor"] == "model"
    assert result.metadata["proposal_stability"]["code_level"] is None
    assert result.metadata["proposal_stability"]["key_level"] is None

    same = {
        "segments": [
            {
                "code_id": "c-freq",
                "passage_id": "p1",
                "focused_revise_proposal": {"definition": "Packing as departure.", "actor": "model"},
            }
        ]
    }
    other = {
        "segments": [
            {
                "code_id": "c-other",
                "passage_id": "p9",
                "focused_drop_proposal": {"definition": "", "actor": "model"},
            }
        ]
    }
    assert proposal_stability([same])["code_level"] is None
    overlap = proposal_stability([same, same])
    assert overlap["code_level"] == 1.0
    assert overlap["key_level"] == 1.0
    assert proposal_stability([same, other])["code_level"] == 0.0

    llm = _two_passage_compete(
        decision_mode="llm",
        decision_model=FakeDecisionModel(raised=["c-freq", "c-rare"], support="indeterminate"),
        interrupt=True,
    )
    llm_group = next(item for item in llm.metadata["proposal_consolidation"] if item["code_id"] == "c-freq")
    assert llm_group["actor"] == "model"
    assert llm_group["conflict"] is True
    assert all(item["actor"] == "model" for item in llm_group["proposals"])
    assert all(item.get("review_actor") != "human" for item in llm_group["proposals"])
