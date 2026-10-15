"""0.7.6 focused-coding fixes: one model loop, timeouts, relevant claim tests."""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory.async_support import CallLimiter, run_sync
from automated_grounded_theory.category_state import property_key, proposal_stability
from automated_grounded_theory.focused import (
    _absence_pole,
    _apply_claim_test,
    _claim_compete_items,
    _mark_degenerate_claims,
    _status_fields,
    _token_estimate_ratio,
    _usable_claims,
    build_focused_graph,
    focused_code,
)
from automated_grounded_theory.schemas import FocusedClaimJudgment, FocusedClaimTest

from tests.fake_focused import FakeFocusedCoder
from tests.test_decision_mode import FakeDecisionModel
from tests.test_v070 import _entry, _source
from tests.test_v074 import _passage, _segment, _state


class _PooledClient:
    """Remembers the loop that first used it, like a pooled httpx client."""

    def __init__(self):
        self.bound = None
        self.loops = []

    async def ainvoke(self, payload):
        loop = asyncio.get_running_loop()
        if self.bound is None:
            self.bound = loop
        elif loop is not self.bound:
            raise RuntimeError("Event is bound to a different event loop")
        self.loops.append(loop)
        return {"ok": len(self.loops)}


def test_three_sequential_calls_reuse_one_model_loop():
    client = _PooledClient()
    limiter = CallLimiter(1, max_retries=0, request_timeout=2, wall_timeout=2)

    async def _three():
        results = []
        for index in range(3):
            results.append(await limiter.ainvoke(client, {"n": index}))
        return results

    results = run_sync(_three(), timeout=10)
    assert [item["ok"] for item in results] == [1, 2, 3]
    assert len({id(loop) for loop in client.loops}) == 1


class _NeverReturns:
    async def ainvoke(self, payload):
        await asyncio.Event().wait()


def test_a_call_that_never_returns_times_out_on_the_wait():
    limiter = CallLimiter(1, max_retries=0, request_timeout=0.2, wall_timeout=0.2)
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_sync(limiter.ainvoke(_NeverReturns(), {"call": "hang"}), timeout=2)
    assert time.perf_counter() - started < 1.0


class _OpenConnection:
    def __init__(self):
        self.connection_open = False

    async def ainvoke(self, payload):
        self.connection_open = True
        await asyncio.Event().wait()


def test_timeout_fires_while_the_provider_connection_stays_open():
    client = _OpenConnection()
    limiter = CallLimiter(1, max_retries=0, request_timeout=0.2, wall_timeout=0.2)
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_sync(limiter.ainvoke(client, {"call": "open"}), timeout=2)
    assert time.perf_counter() - started < 1.0
    assert client.connection_open is True


class _HangingCoder(BaseChatModel):
    connection_open: bool = False

    @property
    def _llm_type(self) -> str:
        return "hanging-076"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        parent = self

        async def _call(messages):
            parent.connection_open = True
            await asyncio.Event().wait()

        return RunnableLambda(_call)


def test_a_hung_graph_step_records_incomplete_timeout():
    saver = InMemorySaver()
    coder = _HangingCoder()
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        focused_code(
            _source(
                passages=[_passage("p1", "I spent the morning packing my suitcase.")],
                codes=[_entry("c1", "leaving home", 1, ["p1"])],
                segments=[_segment("s1", "c1", "p1")],
            ),
            model=coder,
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
            focused_request_retries=0,
            max_focused_review_calls=2,
            interrupt=False,
            checkpointer=saver,
            thread_id="timeout-076",
            raise_on_timeout=True,
        )
    assert time.perf_counter() - started < 1.5
    assert coder.connection_open is True
    snapshot = run_sync(
        build_focused_graph().compile(checkpointer=saver).aget_state(
            {"configurable": {"thread_id": "timeout-076"}}
        )
    )
    values = dict(snapshot.values or {})
    assert values.get("recode_stop") == "timeout"
    assert values["metadata"]["status"] == "incomplete"
    assert values["metadata"]["stop"] == "timeout"
    assert values["metadata"]["complete"] is False
    assert any("Stop is timeout" in item for item in values.get("warnings") or [])


def test_absence_poles_are_flagged_and_unrelated_passages_are_not_tested():
    assert _absence_pole("no mention of staffing") is True
    assert _absence_pole("zzzz-absent") is False
    marked = _mark_degenerate_claims(
        [
            {
                "entry_id": "prop-1",
                "category_id": "c1",
                "name": "staffing",
                "text": "no mention of staffing",
                "pole": "no mention of staffing",
                "passage_id": "p1",
            }
        ]
    )
    assert marked[0]["degenerate_absence_pole"] is True
    assert marked[0]["downstream_eligible"] is False
    assert "degenerate_absence_pole" in marked[0]["flags"]
    assert marked[0]["needs_review"] is True
    assert _usable_claims(marked) == []
    found = _claim_compete_items(
        _state(
            [_passage("p1", "alpha"), _passage("p2", "bravo charlie")],
            [_segment("s1", "c1", "p1"), _segment("s2", "c1", "p2")],
            marked,
            reread=["p1"],
            codebook=[_entry("c1", "leaving home", 2, ["p1", "p2"])],
        ),
        [],
        slots=2,
    )
    assert found == []

    props = [{"entry_id": "prop-1", "flags": [], "negative_cases": [], "claim_tests": []}]
    counts = {"supports": 0, "contradicts": 0, "qualifies": 0, "irrelevant": 0}
    _apply_claim_test(
        props,
        [],
        {"entry_id": "prop-1", "judgment": "contradicts", "rationale": "unrelated", "quote": "bravo"},
        "p2",
        counts,
        valid=False,
    )
    assert counts["contradicts"] == 1
    assert props[0].get("evidence_condition") != "disputed"
    assert not props[0].get("negative_cases")
    assert "claim_test_invalid" in props[0]["flags"]
    assert props[0]["claim_tests"][-1]["judgment"] == "contradicts"


class _TwoJudgments(FakeFocusedCoder):
    claim_calls: int = 0

    def claim_test_batch(self, payload):
        rows = []
        for passage in payload.get("passages") or []:
            self.claim_calls += 1
            claim = passage.get("claim") or {}
            judgment = "contradicts" if self.claim_calls == 1 else "irrelevant"
            rows.append(
                FocusedClaimJudgment(
                    passage_id=str(passage.get("passage_id") or ""),
                    entry_id=str(claim.get("entry_id") or ""),
                    judgment=judgment,
                    rationale="Judged against this passage only.",
                    quote=str(passage.get("text") or ""),
                )
            )
        return FocusedClaimTest(passages=rows)


def test_claim_test_counts_accumulate_across_calls():
    passages = [
        _passage("p1", "I spent the morning packing my suitcase."),
        _passage("p2", "Packing bags up with boxes took all night."),
        _passage("p3", "The suitcase and the boxes were by the door at dawn."),
    ]
    result = focused_code(
        _source(
            passages=passages,
            codes=[_entry("c1", "leaving home", 3, ["p1", "p2", "p3"])],
            segments=[
                _segment("s1", "c1", "p1"),
                _segment("s2", "c1", "p2"),
                _segment("s3", "c1", "p3"),
            ],
        ),
        model=_TwoJudgments(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=1 / 3,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=2 / 3,
        max_focused_review_calls=3,
        interrupt=False,
    )
    counts = result.metadata["claim_test_counts"]
    assert counts["contradicts"] == 1
    assert counts["irrelevant"] == 1
    assert counts["supports"] == 0
    assert counts["qualifies"] == 0
    disputed = [row for row in result.properties + result.dimensions if row.get("evidence_condition") == "disputed"]
    assert disputed
    assert disputed[0]["negative_cases"][-1]["judgment"] == "contradicts"


def test_proposal_stability_is_undefined_without_proposals():
    empty = {"segments": [], "proposal_consolidation": []}
    scored = proposal_stability([empty, {"segments": []}])
    assert scored["code_level"] is None
    assert scored["key_level"] is None
    same = {
        "segments": [
            {
                "code_id": "c1",
                "passage_id": "p1",
                "focused_revise_proposal": {"definition": "Packing as departure.", "actor": "model"},
            }
        ]
    }
    assert proposal_stability([same, same])["code_level"] == 1.0


def test_compete_keep_only_is_flagged_and_a_revise_is_still_stored():
    source = _source(
        passages=[_passage("p1", "I spent the morning packing my suitcase.")],
        codes=[
            _entry("c1", "leaving home", 1, ["p1"]),
            _entry("c2", "packing bags", 1, ["p1"]),
        ],
        segments=[_segment("s1", "c1", "p1"), _segment("s2", "c2", "p1")],
    )
    kept = focused_code(
        source,
        model=FakeFocusedCoder(),
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
    assert kept.metadata["compete_keep_only"] is True
    assert kept.metadata["compete_verdict_counts"]["keep"] >= 1
    assert kept.metadata["compete_verdict_counts"]["revise"] == 0
    assert kept.metadata["compete_verdict_counts"]["drop"] == 0
    assert any("compete_keep_only" in item for item in kept.warnings)
    assert not any(row.get("focused_revise_proposal") or row.get("focused_drop_proposal") for row in kept.segments)

    class _Revise(FakeFocusedCoder):
        def compete_batch(self, payload):
            parsed = super().compete_batch(payload)
            rows = []
            for passage in parsed.passages:
                verdicts = [
                    verdict.model_copy(update={"verdict": "revise", "revised_definition": "Packing as departure."})
                    for verdict in passage.verdicts
                ]
                rows.append(passage.model_copy(update={"verdicts": verdicts}))
            return parsed.model_copy(update={"passages": rows})

    revised = focused_code(
        source,
        model=_Revise(),
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
    assert revised.metadata["compete_keep_only"] is False
    assert revised.metadata["compete_verdict_counts"]["revise"] >= 1
    assert any(row.get("focused_revise_proposal") for row in revised.segments)
    assert not any("compete_keep_only" in item for item in revised.warnings)


def test_property_key_drops_function_words_and_the_estimate_is_a_ceiling():
    assert property_key("timing during departure") == property_key("departure timing")
    assert property_key("Timing of the departure") == property_key("departure timings")
    assert property_key("timing during departure") != property_key("pace")
    assert _token_estimate_ratio(0, {}) is None
    assert _token_estimate_ratio(128, {"total_tokens": 100}) is None
    assert _token_estimate_ratio(128, {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140}) == 1.28
    fields = _status_fields(
        {
            "metadata": {"procedure": {}},
            "recode_stop": "",
            "focused_not_reread": [],
            "passages": [],
            "recode_queue": [],
            "focused_reread": [],
            "focused_calls_used": 0,
            "focused_tokens_used": 100,
            "focused_tokens_estimated": 128,
            "provider_token_usage": {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 4, "total_tokens": 120},
        }
    )
    assert fields["token_estimate_ratio"] == 1.28
    assert fields["focused_tokens_estimated"] == 128
    assert fields["provider_token_usage"]["input_tokens"] == 100
    assert fields["provider_token_usage"]["output_tokens"] == 20
    assert "heuristic" in fields["token_accounting_note"]
    assert "input_tokens" in fields["token_accounting_note"]


def test_llm_decision_mode_runs_both_checkpoints_offline():
    decider = FakeDecisionModel(raised=["c1"], support="supported")
    result = focused_code(
        _source(
            passages=[
                _passage("p1", "I spent the morning packing my suitcase."),
                _passage("p2", "Packing bags up with boxes took all night."),
            ],
            codes=[_entry("c1", "leaving home", 2, ["p1", "p2"])],
            segments=[_segment("s1", "c1", "p1"), _segment("s2", "c1", "p2")],
        ),
        model=FakeFocusedCoder(),
        decision_mode="llm",
        decision_model=decider,
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=1,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        max_focused_review_calls=1,
        interrupt=True,
    )
    assert result.interrupted is False
    assert result.metadata["selection_source"] == "llm"
    reviews = [row for row in result.prompt_log if row.get("kind") == "decision_review"]
    assert {row.get("checkpoint") for row in reviews} >= {"focused_selection", "focused_claims"}
    assert all(row.get("actor") == "llm" for row in reviews)
    reviewed = [row for row in result.properties + result.dimensions if row.get("review_actor") == "llm"]
    assert reviewed
    assert all(row.get("claim_support", "undefined") == "undefined" for row in reviewed)
    assert all(row.get("model_claim_support") == "supported" for row in reviewed)
    assert decider.calls >= 2
