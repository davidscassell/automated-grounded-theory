"""0.9.44 reporting and accounting. Synthetic fixtures only.

These checks do not change eligibility, reviewer standards, or the 6,000 cap.
Analytic accuracy is unmeasured. Quote anchoring here is offline only.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.axial import (
    AxialRelationshipDraft,
    _call_usage_report,
    build_allowance_accounting,
    build_pair_assertions,
    build_relationship_funnel,
)
from automated_grounded_theory.records import POLICY_VERSION
from tests.test_v080 import FakeAxialCoder
from automated_grounded_theory.version import __version__


def test_version_is_0944():
    assert __version__ == "0.9.44"
    assert POLICY_VERSION == "0.9.44"


def test_a_core_on_one_eligible_relationship_is_provisional():
    from automated_grounded_theory.axial import _attach_run_accounting, _note_core_provisional

    metadata = {"core_outcome": "core_selected"}
    _note_core_provisional(metadata, [
        {"relationship_id": "rel-0003", "downstream_eligible": True},
        {"relationship_id": "rel-0004", "downstream_eligible": False},
        {"relationship_id": "rel-synthetic", "downstream_eligible": True, "synthetic": True},
    ])
    assert metadata["core_provisional"] is True
    assert metadata["core_eligible_relationship_ids"] == ["rel-0003"]
    assert metadata["core_support_note"] == (
        "The core is provisional. It rests on one eligible relationship (rel-0003)."
    )
    several = {"core_outcome": "core_selected"}
    _note_core_provisional(several, [
        {"relationship_id": "rel-1", "downstream_eligible": True},
        {"relationship_id": "rel-2", "downstream_eligible": True},
    ])
    assert several["core_provisional"] is False
    assert several["core_support_note"] is None
    unresolved = {"core_outcome": "selection_unresolved"}
    _note_core_provisional(unresolved, [{"relationship_id": "rel-0003", "downstream_eligible": True}])
    assert unresolved["core_provisional"] is None
    assert unresolved["core_support_note"] is None
    accounting = {}
    _attach_run_accounting(accounting, {"prompt_log": [], "relationships": []})
    assert accounting["quote_anchoring_live_exercise"] is False
    assert "offline" in accounting["quote_anchoring_note"]


def test_stage_usage_reconciles_without_double_counting():
    report = _call_usage_report(
        [
            {"kind": "axial_proposal", "estimated_input_tokens": 100, "usage": {"input_tokens": 10, "output_tokens": 1}},
            {
                "kind": "axial_proposal",
                "statement_copy_retry": True,
                "estimated_input_tokens": 80,
                "usage": {"input_tokens": 8, "output_tokens": 1},
            },
            {"kind": "axial_counterexample", "estimated_input_tokens": 50, "usage": {"input_tokens": 5, "output_tokens": 1}},
            {
                "kind": "decision_review",
                "checkpoint": "axial_relation_review",
                "estimated_input_tokens": 40,
                "usage": {"input_tokens": 4, "output_tokens": 1},
            },
            {
                "kind": "decision_review",
                "checkpoint": "axial_saturation_review",
                "estimated_input_tokens": 30,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
            {
                "kind": "decision_review",
                "checkpoint": "axial_core_review",
                "estimated_input_tokens": 70,
                "usage": {"input_tokens": 7, "output_tokens": 2},
            },
            {"kind": "axial_proposal", "status": "omitted", "estimated_input_tokens": 999, "usage": {"input_tokens": 999, "output_tokens": 999}},
        ],
        procedure={"include_memos": False, "include_storyline": False},
    )
    by_stage = {row["stage"]: row for row in report["stages"]}
    assert by_stage["axial_proposal"]["calls"] == 1
    assert by_stage["statement_copy_retry"]["calls"] == 1
    assert by_stage["axial_counterexample"]["provider_output_tokens"] == 1
    assert by_stage["axial_core_review"]["provider_input_tokens"] == 7
    assert by_stage["axial_core_review"]["provider_output_tokens"] == 2
    coding = by_stage["coding"]
    assert coding["subtotal"] is True
    assert coding["calls"] == 3
    assert coding["estimated_input_tokens"] == 230
    assert coding["provider_input_tokens"] == 23
    assert report["estimated_input_tokens"] == 100 + 80 + 50 + 40 + 30 + 70
    assert report["provider_input_tokens"] == 10 + 8 + 5 + 4 + 3 + 7
    assert report["provider_output_tokens"] == 7
    assert report["reconciles"] is True
    assert report["provider_input_tokens"] == (
        coding["provider_input_tokens"]
        + by_stage["axial_relation_review"]["provider_input_tokens"]
        + by_stage["axial_saturation_review"]["provider_input_tokens"]
        + by_stage["axial_core_review"]["provider_input_tokens"]
    )
    selective = report["selective"]
    assert selective["status"] == "not_requested"
    assert selective["calls"] == 0
    assert selective["core_usage_stage"] == "axial_core_review"
    assert selective["referenced_core_calls"] == 1
    missing = _call_usage_report(
        [{"kind": "axial_proposal", "estimated_input_tokens": 10, "usage": {}}],
    )
    proposal = next(row for row in missing["stages"] if row["stage"] == "axial_proposal")
    assert proposal["provider_input_tokens"] is None
    assert proposal["provider_output_tokens"] is None
    assert missing["provider_input_tokens"] is None


def test_a_returned_role_is_kept_and_an_omitted_role_stays_unspecified():
    returned = AxialRelationshipDraft(from_id="a", to_id="b", statement="A claim.", paradigm_role="context")
    omitted = AxialRelationshipDraft(from_id="a", to_id="b", statement="A claim.")
    explicit = AxialRelationshipDraft(from_id="a", to_id="b", statement="A claim.", paradigm_role="unspecified")
    assert returned.paradigm_role == "context"
    assert returned.paradigm_role_origin == "model_returned"
    assert omitted.paradigm_role == "unspecified"
    assert omitted.paradigm_role_origin == "schema_default_unspecified"
    assert explicit.paradigm_role_origin == "model_unspecified"


def test_accepted_rows_keep_empty_reasons_and_separate_limitations():
    funnel = build_relationship_funnel({
        "metadata": {"procedure": {"pattern_min_cases": 2}},
        "codebook": {},
        "passages": [
            {"passage_id": "p1", "doc_id": "d1"},
            {"passage_id": "p2", "doc_id": "d2"},
        ],
        "relationships": [{
            "relationship_id": "rel-0003",
            "from_id": "a",
            "to_id": "b",
            "statement": "One claim.",
            "disposition": "accepted",
            "reviewer_disposition": "accepted",
            "downstream_eligible": True,
            "evidence_tier": 2,
            "paradigm_role": "unspecified",
            "passage_id": "p1",
            "document_count": 2,
        }],
    })
    row = funnel["rows"][0]
    assert row["reasons"] == []
    assert row["evidence_limitations"] == ["tentative"]
    assert "tentative" not in row["reasons"]
    assert "limited_document_support" not in row["reasons"]


def test_different_statements_stay_separate_assertions():
    grouped = build_pair_assertions([
        {"relationship_id": "rel-1", "from_id": "a", "to_id": "b", "statement": "First claim.", "quote": "one"},
        {"relationship_id": "rel-2", "from_id": "a", "to_id": "b", "statement": "Second claim.", "quote": "two"},
        {"relationship_id": "rel-3", "from_id": "b", "to_id": "a", "statement": "Opposite.", "quote": "three"},
    ])
    pair = grouped["pairs"][0]
    assert pair["assertion_count"] == 2
    assert {item["relationship_id"] for item in pair["assertions"]} == {"rel-1", "rel-2"}
    assert {item["statement"] for item in pair["assertions"]} == {"First claim.", "Second claim."}
    assert len(grouped["pairs"]) == 2


def test_unused_allowance_does_not_mean_a_request_was_available():
    report = build_allowance_accounting(
        {
            "allowance_reserved": {
                "axial_proposal": 4,
                "statement_copy_retry": 2,
                "axial_counterexample": 4,
                "axial_relation_review": 4,
                "axial_saturation_review": 4,
                "axial_core_review": 6,
            },
            "allowance_transfers": [
                {"from": "counterexamples", "to": "proposals", "calls": 2, "reason": "multi_document_pairs_precede_counterexamples"},
            ],
            "procedure": {},
        },
        {"stages": [
            {"stage": "axial_proposal", "calls": 6},
            {"stage": "statement_copy_retry", "calls": 0},
            {"stage": "axial_counterexample", "calls": 1},
            {"stage": "axial_core_review", "calls": 6},
        ]},
    )
    by_stage = {row["stage"]: row for row in report["stages"]}
    assert by_stage["axial_counterexample"]["reserved"] == 4
    assert by_stage["axial_counterexample"]["released"] == 2
    assert by_stage["axial_counterexample"]["consumed"] == 1
    assert by_stage["axial_counterexample"]["unused"] == 1
    assert "does not mean" in by_stage["statement_copy_retry"]["unused_reason"]
    assert "tokens" in report["note"]


class _AdvanceFirst(FakeAxialCoder):
    """Select the first candidate in each core match. Counts every sent call."""

    core_seen: int = 0

    def with_structured_output(self, schema, **kwargs):
        name = getattr(schema, "__name__", "")
        if name != "ModelCoreDecision":
            return FakeAxialCoder.with_structured_output(self, schema, **kwargs)
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.core_seen += 1
            body = {}
            for message in messages:
                content = getattr(message, "content", "")
                if isinstance(content, str) and content.startswith("{"):
                    body = __import__("json").loads(content)
            candidates = body.get("candidates") or []
            chosen = str((candidates[0] or {}).get("code_id") or "") if candidates else ""
            parsed = schema(core_id=chosen, rationale=f"Select {chosen}.")
            if not include_raw:
                return parsed
            raw = AIMessage(content="{}", usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42})
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def _core_rows(result):
    return [
        row for row in result.prompt_log
        if row.get("checkpoint") == "axial_core_review" and row.get("status") not in {"omitted", "refused", "not_sent"}
    ]


def _round_roles(rows):
    found = []
    for row in rows:
        for message in row.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else ""
            if isinstance(content, str) and content.startswith("{"):
                body = __import__("json").loads(content)
                scope = body.get("comparison_scope") if isinstance(body.get("comparison_scope"), dict) else {}
                found.append(str(scope.get("round_role") or body.get("round_role") or ""))
    return found


def test_a_seven_call_core_stops_at_six_and_the_seventh_consumes_a_raised_allowance(tmp_path):
    from automated_grounded_theory.axial import export_axial_state, load_axial_export, selective_code
    from tests.test_v0934 import _core_source

    source = _core_source("packing " * 450, 8)
    common = dict(
        seed=3,
        decision_mode="llm",
        interrupt=False,
        max_decision_calls=4,
        max_core_review_tokens=10**6,
        max_axial_review_tokens=10**6,
        axial_token_budget=6000,
        allow_partial_core_review=True,
    )
    stopped_model = _AdvanceFirst()
    stopped = selective_code(source, stopped_model, max_core_review_calls=6, **common)
    stopped_rows = _core_rows(stopped)
    assert len(stopped_rows) == 6
    assert stopped_model.core_seen == 6
    assert stopped.metadata["provider_budget"]["stages"]["axial_core_review"]["calls_used"] == 6
    assert stopped.metadata["core_call_accounting"]["sent_calls"] == 6
    assert stopped.metadata["core_call_accounting"]["sent_within_allowance"] is True
    assert stopped.metadata["core_cost_projection"]["opening_calls"] + stopped.metadata["core_cost_projection"]["contingent_calls"] == 7
    assert stopped.metadata["core_cost_projection"]["final_assessment_in_plan"] is True
    assert "final" not in _round_roles(stopped_rows)
    full_model = _AdvanceFirst()
    full = selective_code(source, full_model, max_core_review_calls=7, **common)
    full_rows = _core_rows(full)
    assert len(full_rows) == 7
    assert full_model.core_seen == 7
    assert full.metadata["provider_budget"]["stages"]["axial_core_review"]["calls_used"] == 7
    assert full.metadata["core_call_accounting"]["sent_within_allowance"] is True
    assert "final" in _round_roles(full_rows)
    export_axial_state(
        {
            "metadata": stopped.metadata,
            "warnings": list(stopped.warnings),
            "prompt_log": list(stopped.prompt_log),
            "relationships": [],
            "core_candidates": list(stopped.core_candidates),
            "saturation": {},
            "coverage": {},
        },
        tmp_path,
    )
    resumed_model = _AdvanceFirst()
    resumed = selective_code(
        load_axial_export(tmp_path),
        resumed_model,
        max_core_review_calls=7,
        **common,
    )
    assert resumed_model.core_seen == 1
    assert resumed.metadata["provider_budget"]["stages"]["axial_core_review"]["calls_used"] == 7
    assert "final" in _round_roles(_core_rows(resumed))
    assert resumed.metadata["core_call_accounting"]["sent_within_allowance"] is True
