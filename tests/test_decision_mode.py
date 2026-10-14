"""decision_mode human|llm glue around the user review files."""

from __future__ import annotations

import inspect
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    apply_model_claim_decisions,
    focused_code,
)
from automated_grounded_theory.api import CheckpointMismatch
from automated_grounded_theory.category_state import apply_claim_review, merge_properties
from automated_grounded_theory.focused import main
from automated_grounded_theory.prompts import message_text

from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _memos, _segments_for_raised, _source, _sweep


class FakeDecisionModel(BaseChatModel):
    """Returns selection and claim decisions. No network calls."""

    raised: list[str] = []
    support: str = "contradicted"
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-decision"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.calls += 1
            name = getattr(schema, "__name__", "")
            text = "\n".join(message_text(message) for message in messages)
            if name == "SelectionDecision":
                parsed = schema(raised_code_ids=list(self.raised), rationale="Selected from the evidence sample.")
            elif name == "ClaimDecisions":
                payload = json.loads(text[text.find("{") :])
                parsed = schema(
                    claims=[
                        {
                            "entry_id": claim["entry_id"],
                            "disposition": "rejected" if self.support == "contradicted" else "accepted",
                            "support": self.support,
                            "rationale": "Judged against the cited passage.",
                        }
                        for claim in payload["claims"]
                    ]
                )
            else:
                raise AssertionError(f"unexpected decision schema {name}")
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def test_decision_arguments_are_bounded():
    params = inspect.signature(focused_code).parameters
    assert params["decision_mode"].default == "human"
    assert params["max_decision_calls"].default == 8
    assert params["max_decision_tokens"].default == 24000
    assert params["decision_token_budget"].default == 6000
    source = _source(segments=_segments_for_raised()[:1])
    with pytest.raises(ValueError, match="decision_mode"):
        focused_code(source, model=FakeFocusedCoder(), decision_mode="auto", interrupt=False)
    with pytest.raises(ValueError, match="decision_model"):
        focused_code(
            source,
            model=FakeFocusedCoder(),
            decision_mode="human",
            decision_model=FakeDecisionModel(),
            interrupt=False,
        )


def test_human_empty_selection_is_kept_and_skips_the_reread():
    saver = InMemorySaver()
    coder = FakeFocusedCoder()
    source = _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep())
    common = dict(
        model=coder,
        seed=3,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        checkpointer=saver,
        thread_id="empty-selection",
        max_focused_review_calls=4,
    )
    paused = focused_code(source, interrupt=True, **common)
    assert paused.interrupt_payload["kind"] == "focused_selection"
    assert paused.interrupt_payload["raised_code_ids"]
    held = focused_code(source, interrupt=True, resuming=True, resume={"raised_code_ids": []}, **common)
    assert coder.call_count == 0
    assert held.raised_code_ids == []
    assert held.metadata["selection_source"] == "researcher"
    assert held.interrupt_payload["kind"] == "focused_claims"
    finished = focused_code(source, interrupt=True, resuming=True, resume={"leave_claims": True}, **common)
    assert finished.interrupted is False
    assert finished.raised_code_ids == []
    assert finished.focused_reread == []
    assert coder.call_count == 0


def test_human_contradicted_claims_stay_out_of_downstream():
    row = {
        "entry_id": "prop-1",
        "kind": "property",
        "category_id": "c1",
        "name": "timing",
        "text": "Morning packing.",
        "disposition": "accepted",
        "evidence_condition": "anchored",
        "downstream_eligible": True,
        "claim_support": "undefined",
        "needs_review": False,
        "quote_status": "exact",
    }
    reviewed = apply_claim_review(
        [row],
        [],
        {"claim_reviews": [{"entry_id": "prop-1", "claim_support": "contradicted"}]},
    )
    claim = reviewed["properties"][0]
    assert claim["claim_support"] == "contradicted"
    assert claim["downstream_eligible"] is False
    assert claim["review_actor"] == "human"
    assert claim["review_history"][-1]["actor"] == "human"
    assert "model_claim_support" not in claim


def test_conflicting_property_merges_keep_constituents_and_are_excluded():
    rows = [
        {
            "kind": "property",
            "category_id": "c1",
            "entry_id": "a",
            "name": "timing",
            "text": "Morning packing.",
            "passage_id": "p1",
            "quote": "morning",
            "disposition": "accepted",
            "evidence_condition": "anchored",
            "downstream_eligible": True,
        },
        {
            "kind": "property",
            "category_id": "c1",
            "entry_id": "b",
            "name": "timing",
            "text": "Evening packing.",
            "passage_id": "p2",
            "quote": "evening",
            "disposition": "accepted",
            "evidence_condition": "anchored",
            "downstream_eligible": True,
        },
    ]
    merged = merge_properties(rows)
    assert len(merged) == 1
    assert merged[0]["property_merge_conflict"] is True
    assert "property_merge_conflict" in merged[0]["flags"]
    assert merged[0]["downstream_eligible"] is False
    assert [item["entry_id"] for item in merged[0]["constituents"]] == ["a", "b"]
    assert merged[0]["constituents"][0]["text"] == "Morning packing."
    same = merge_properties(
        [
            {**rows[0], "text": "When the packing sits."},
            {**rows[1], "entry_id": "c", "text": "When the packing sits."},
        ]
    )
    assert same[0].get("property_merge_conflict") is not True
    assert [item["entry_id"] for item in same[0]["constituents"]] == ["a", "c"]


def test_model_claim_support_stays_separate_and_rejects_contradictions():
    original = {
        "entry_id": "prop-1",
        "claim_support": "undefined",
        "evidence_condition": "anchored",
        "needs_review": False,
        "disposition": "proposed",
    }
    updated = apply_model_claim_decisions(
        [original],
        [
            {
                "entry_id": "prop-1",
                "disposition": "accepted",
                "support": "contradicted",
                "rationale": "The quote points the other way.",
            }
        ],
        {"model": "FakeDecisionModel"},
    )
    row = updated[0]
    assert row["claim_support"] == "undefined"
    assert row["model_claim_support"] == "contradicted"
    assert row["disposition"] == "rejected"
    assert row["downstream_eligible"] is False
    assert row["review_actor"] == "llm"
    assert row["review_model"] == "FakeDecisionModel"
    assert row["review_history"][-1]["actor"] == "llm"


def test_llm_mode_can_take_over_a_paused_human_run():
    saver = InMemorySaver()
    coder = FakeFocusedCoder()
    decider = FakeDecisionModel(raised=[])
    source = _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep())
    common = dict(
        model=coder,
        seed=3,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        checkpointer=saver,
        thread_id="mode-switch",
        max_focused_review_calls=4,
    )
    paused = focused_code(source, interrupt=True, decision_mode="human", **common)
    assert paused.interrupted
    with pytest.raises(CheckpointMismatch):
        focused_code(
            source,
            interrupt=True,
            decision_mode="llm",
            decision_model=decider,
            resuming=True,
            **common,
        )
    finished = focused_code(
        source,
        interrupt=True,
        decision_mode="llm",
        decision_model=decider,
        resuming=True,
        allow_changes=True,
        **common,
    )
    assert finished.interrupted is False
    assert finished.raised_code_ids == []
    assert finished.metadata["selection_source"] == "llm"
    assert finished.metadata["decision_mode"] == "llm"
    assert coder.call_count == 0
    assert decider.calls == 1
    audit = finished.prompt_log[-1]
    assert audit["kind"] == "decision_review"
    assert audit["actor"] == "llm"
    assert audit["status"] == "validated"
    assert finished.metadata["continuations"]


def test_llm_claim_review_keeps_model_provenance_and_stops_on_the_allowance(tmp_path):
    coder = FakeFocusedCoder()
    decider = FakeDecisionModel(raised=["c-freq"], support="contradicted")
    source = _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep())
    result = focused_code(
        source,
        model=coder,
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
        output_dir=tmp_path,
    )
    assert result.interrupted is False
    assert result.raised_code_ids == ["c-freq"]
    assert coder.call_count >= 1
    assert any(row.get("model_claim_support") == "contradicted" for row in result.properties + result.dimensions)
    assert all(row.get("claim_support", "undefined") == "undefined" for row in result.properties + result.dimensions)
    contradicted = [row for row in result.properties + result.dimensions if row.get("model_claim_support") == "contradicted"]
    assert contradicted
    assert all(row["review_actor"] == "llm" and row["disposition"] == "rejected" for row in contradicted)
    assert all(row["downstream_eligible"] is False for row in contradicted)
    downstream = json.loads((tmp_path / "downstream_claims.json").read_text(encoding="utf-8"))
    assert all(row.get("model_claim_support") != "contradicted" for row in downstream["records"])

    stopped = focused_code(
        source,
        model=FakeFocusedCoder(),
        decision_mode="llm",
        decision_model=FakeDecisionModel(raised=["c-freq"]),
        seed=1,
        max_decision_calls=0,
        interrupt=True,
    )
    assert stopped.metadata["status"] == "incomplete"
    assert stopped.metadata["stop"] == "decision_call_allowance"
    assert stopped.metadata["decision_stop"] == "call_allowance"


def test_resume_cli_advertises_decision_flags(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["resume", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    for flag in (
        "--decision-mode",
        "--decision-model",
        "--max-decision-calls",
        "--max-decision-tokens",
        "--decision-token-budget",
    ):
        assert flag in help_text
