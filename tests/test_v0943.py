"""0.9.43 coverage, budgeting, and reporting. Synthetic fixtures only.

These checks are engineering and evidence-traceability coverage.
Analytic accuracy is unmeasured. Eligibility and reviewer standards are unchanged.
"""

from __future__ import annotations

from automated_grounded_theory.axial import (
    AxialProposalBatch,
    AxialRelationshipDraft,
    _annotate_quote_scope,
    _call_usage_report,
    axial_code,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder


def test_version_is_0943():
    assert __version__ == "0.9.43"
    assert POLICY_VERSION == "0.9.43"


def test_a_punctuation_only_difference_is_not_a_missing_quote():
    row = {
        "passage_id": "p1",
        "quote": "the tool\u2019s bench",
        "flags": [],
    }
    passages = {"p1": {"passage_id": "p1", "doc_id": "d1", "text": "the tool's bench"}}
    _annotate_quote_scope(row, passages)
    assert "quotes_not_in_cited_passage" not in (row.get("flags") or [])
    assert row["quote_scope_cause"] is None


def test_a_quote_in_another_passage_keeps_the_flag_and_names_the_cause():
    row = {
        "passage_id": "p1",
        "quote": "alpha and beta sit together",
        "flags": [],
    }
    passages = {
        "p1": {"passage_id": "p1", "doc_id": "d1", "text": "nothing about either category"},
        "p2": {"passage_id": "p2", "doc_id": "d2", "text": "alpha and beta sit together"},
    }
    _annotate_quote_scope(row, passages)
    assert "quotes_not_in_cited_passage" in row["flags"]
    assert row["quote_scope_cause"] == "quote_found_in_other_passage"
    absent = {"passage_id": "p1", "quote": "not in any passage here", "flags": []}
    _annotate_quote_scope(absent, passages)
    assert "quotes_not_in_cited_passage" in absent["flags"]
    assert absent["quote_scope_cause"] == "quote_absent_from_corpus"


def test_missing_core_calls_make_usage_partial():
    report = _call_usage_report(
        [{
            "kind": "axial_proposal",
            "estimated_input_tokens": 40,
            "usage": {"input_tokens": 30, "output_tokens": 12},
        }],
        provider_budget={"stages": {"coding": {"calls_used": 1}, "axial_core_review": {"calls_used": 6}}},
    )
    assert report["complete"] is False
    assert report["usage_completeness"] == "partial"
    core = next(row for row in report["stages"] if row["stage"] == "axial_core_review")
    assert core["calls_in_ledger"] == 6
    assert core["missing_calls"] == 6
    assert core["provider_input_tokens"] is None


class _MultiThenRetry(FakeAxialCoder):
    def proposals(self, body):
        pair = (body.get("pairs") or [{"from_id": "a", "to_id": "b"}])[0]
        left = str(pair.get("from_id") or "")
        right = str(pair.get("to_id") or "")
        quote = "alpha and beta sit together"
        if {left, right} == {"c", "d"}:
            quote = "not in any passage here"
        if body.get("statement_copy_retry"):
            statement = "Alpha and beta co-occur across these documents."
        elif {left, right} == {"c", "d"}:
            statement = quote
        else:
            statement = "The two categories co-occur across these documents."
        cited_quote = quote
        return AxialProposalBatch(relationships=[
            AxialRelationshipDraft(
                from_id=left,
                to_id=right,
                statement=statement,
                passage_id="p1",
                quote=cited_quote,
            )
        ])

    def relation_review(self, body):
        from automated_grounded_theory.axial import AxialRelationDecision, AxialRelationDecisions

        return AxialRelationDecisions(relationships=[
            AxialRelationDecision(
                relationship_id=str(row.get("relationship_id") or ""),
                disposition="rejected",
                paradigm_role="context",
                rationale="The claim does not hold.",
            )
            for row in body.get("relationships") or []
        ])


def test_multi_document_pairs_are_examined_before_counterexamples_and_single_document_slots():
    codes = ["a", "b", "c", "d", "e", "f"]
    source = {
        "codebook": {code: {"code_id": code, "code": code, "definition": f"the {code} category"} for code in codes},
        "passages": [
            {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "alpha and beta sit together"},
            {"passage_id": "p2", "doc_id": "d2", "respondent_id": "r2", "text": "alpha and beta sit together"},
            {"passage_id": "p3", "doc_id": "d3", "respondent_id": "r3", "text": "only the single document pair"},
        ],
        "segments": [
            {"segment_id": f"s{doc}{code}", "code_id": code, "passage_id": passage}
            for doc, passage in (("1", "p1"), ("2", "p2"))
            for code in ("a", "b", "c", "d")
        ] + [
            {"segment_id": "s3e", "code_id": "e", "passage_id": "p3"},
            {"segment_id": "s3f", "code_id": "f", "passage_id": "p3"},
        ],
        "relationships": [],
        "memos": [],
    }
    result = axial_code(
        source,
        _MultiThenRetry(),
        seed=1,
        decision_mode="llm",
        interrupt=False,
        also_selective=True,
        max_axial_review_calls=16,
        axial_share_propose=0.5,
        axial_share_counterexample=0.5,
        axial_token_budget=6000,
        max_axial_review_tokens=24000,
        max_decision_calls=8,
        batch_size=1,
        negative_case_search=True,
    )
    multi = result.metadata["multi_document_pairs"]
    assert multi
    assert result.metadata["multi_document_reconciles"] is True
    terminals = {"proposed", "nothing_to_propose", "failed", "budget_blocked"}
    assert {row["status"] for row in multi} <= terminals
    assert all(row["status"] != "unplanned" for row in multi)
    for row in multi:
        if row["status"] in {"budget_blocked", "failed"}:
            assert row["reason"]
    multi_keys = {tuple(sorted((row["from_id"], row["to_id"]))) for row in multi}
    assert ("e", "f") not in multi_keys
    events = result.metadata.get("axial_event_order") or []
    kinds = [event["kind"] for event in events]
    proposal_events = [event for event in events if event["kind"] == "proposal_sent"]
    assert proposal_events
    proposed_pairs = [
        tuple(sorted((pair["from_id"], pair["to_id"])))
        for event in proposal_events
        for pair in event["pairs"]
    ]
    assert ("e", "f") not in proposed_pairs
    assert set(proposed_pairs) <= multi_keys
    if "counterexample_sent" in kinds:
        assert kinds.index("counterexample_sent") > kinds.index("proposal_sent")
        assert all(pair in multi_keys for pair in proposed_pairs[: len(multi)])
    else:
        assert "counterexample_sent" not in kinds
    priority = result.metadata["multi_document_call_priority"]
    assert priority["moved_from_counterexamples"] > 0 or priority["moved_from_retries"] > 0
    assert any(
        item.get("from") in {"counterexamples", "statement_copy_retry"} and item.get("to") == "proposals"
        for item in result.metadata.get("allowance_transfers") or []
    )
    assert len(proposal_events) >= int(priority["needed_requests"]) - int(priority["still_short"])
    calls = result.metadata["relationship_funnel"]["pair_funnel"]["proposal_calls"]
    assert calls["statement_copy_retries"] >= 1
    assert calls["calls"] == calls["proposal_requests"] + calls["statement_copy_retries"]
    retry_rows = [row for row in calls["retries"] if row.get("retry_sent")]
    assert retry_rows
    assert {row["outcome"] for row in retry_rows} <= {"retried", "retry_still_invalid", "no_supported_relationship", "unsent"}
    funnel_rows = result.metadata["relationship_funnel"]["rows"]
    matched = [row for row in funnel_rows if row.get("retries")]
    assert matched
    assert matched[0]["retries"][0]["reason"] == "invalid_restated_quote"
    reviewed = [row for row in result.relationships if row.get("reviewer_disposition") == "rejected"]
    assert reviewed
    assert all(
        any(item.get("stage") == "axial_relation_review" for item in (row.get("review_history") or []))
        for row in reviewed
    )
    mismatched = [
        row for row in result.relationships
        if "quotes_not_in_cited_passage" in (row.get("flags") or [])
    ]
    assert mismatched
    assert mismatched[0]["quote_scope_cause"] in {"quote_absent_from_corpus", "quote_found_in_other_passage"}
    targeting = result.metadata.get("counterexample_targeting") or {}
    assert targeting.get("applicable_to_reviewer_rejected") is False
    usage = result.metadata["call_usage"]
    stage_names = {row["stage"] for row in usage["stages"]}
    assert "coding" in stage_names
    assert "axial_core_review" in stage_names
    assert usage["complete"] is (usage["usage_completeness"] == "complete")
    if any(row["missing_calls"] for row in usage["stages"]):
        assert usage["complete"] is False
        assert usage["usage_completeness"] == "partial"


class _OverflowRound(FakeAxialCoder):
    """One accepted pair whose quote uses a curly apostrophe. Every other pair is rejected."""

    def proposals(self, body):
        pair = (body.get("pairs") or [{"from_id": "a", "to_id": "b"}])[0]
        left = str(pair.get("from_id") or "")
        right = str(pair.get("to_id") or "")
        curly = "the tool\u2019s bench sits beside the a marker"
        if {left, right} == {"a", "b"}:
            quote = curly
        else:
            quote = "the a marker"
        return AxialProposalBatch(relationships=[
            AxialRelationshipDraft(
                from_id=left,
                to_id=right,
                statement="These categories co-occur across the documents.",
                passage_id="p1",
                quote=quote,
            )
        ])

    def relation_review(self, body):
        from automated_grounded_theory.axial import AxialRelationDecision, AxialRelationDecisions

        rows = []
        for row in body.get("relationships") or []:
            ends = {str(row.get("from_id") or ""), str(row.get("to_id") or "")}
            accepted = ends == {"a", "b"}
            rows.append(AxialRelationDecision(
                relationship_id=str(row.get("relationship_id") or ""),
                disposition="accepted" if accepted else "rejected",
                paradigm_role="context",
                rationale="The curly-apostrophe quote is in the cited passage." if accepted else "The claim does not hold.",
            ))
        return AxialRelationDecisions(relationships=rows)


def test_overflow_blocks_multi_document_pairs_then_sends_a_single_document_pair_and_a_counterexample():
    multi_codes = ["a", "b", "c", "d", "e", "f", "g"]
    codes = multi_codes + ["s", "t"]
    shared = (
        "the tool's bench sits beside the a marker and the b marker and the c marker "
        "and the d marker and the e marker and the f marker and the g marker"
    )
    source = {
        "codebook": {
            code: {"code_id": code, "code": code, "definition": f"the {code} marker"}
            for code in codes
        },
        "passages": [
            {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": shared},
            {"passage_id": "p2", "doc_id": "d2", "respondent_id": "r2", "text": shared},
            {"passage_id": "p3", "doc_id": "d3", "respondent_id": "r3", "text": "the s marker and the t marker only"},
            {"passage_id": "p-neg", "doc_id": "d9", "respondent_id": "r9", "text": "the a marker without the other"},
        ],
        "segments": [
            {"segment_id": f"s{doc}{code}", "code_id": code, "passage_id": passage}
            for doc, passage in (("1", "p1"), ("2", "p2"))
            for code in multi_codes
        ] + [
            {"segment_id": "s3s", "code_id": "s", "passage_id": "p3"},
            {"segment_id": "s3t", "code_id": "t", "passage_id": "p3"},
            {"segment_id": "sneg", "code_id": "a", "passage_id": "p-neg"},
        ],
        "relationships": [],
        "memos": [],
    }
    result = axial_code(
        source,
        _OverflowRound(),
        seed=1,
        decision_mode="llm",
        interrupt=False,
        also_selective=True,
        max_axial_review_calls=16,
        axial_share_propose=0.5,
        axial_share_counterexample=0.5,
        axial_token_budget=6000,
        max_axial_review_tokens=64000,
        max_decision_calls=24,
        max_core_review_calls=12,
        batch_size=1,
        negative_case_search=True,
    )
    multi = [row for row in result.metadata["multi_document_pairs"] if isinstance(row, dict)]
    multi_keys = {tuple(sorted((row["from_id"], row["to_id"]))) for row in multi}
    blocked = [row for row in multi if row["status"] == "budget_blocked"]
    assert len(multi) > result.metadata["multi_document_call_priority"]["propose_calls_before_transfer"]
    assert blocked
    assert all(row["reason"] for row in blocked)
    assert result.metadata["multi_document_reconciles"] is True
    events = result.metadata["axial_event_order"]

    def _pair(event):
        pair = event["pairs"][0]
        return tuple(sorted((pair["from_id"], pair["to_id"])))

    proposal_events = [event for event in events if event["kind"] == "proposal_sent"]
    multi_at = [index for index, event in enumerate(events) if event["kind"] == "proposal_sent" and _pair(event) in multi_keys]
    single_at = [index for index, event in enumerate(events) if event["kind"] == "proposal_sent" and _pair(event) not in multi_keys]
    assert multi_at and single_at
    last_multi = max(multi_at)
    assert min(single_at) > last_multi
    assert all(index > last_multi for index in single_at)
    priority = result.metadata["multi_document_call_priority"]
    assert priority["moved_from_counterexamples"] > 0
    assert priority["moved_from_retries"] > 0
    moved = priority["moved_from_counterexamples"] + priority["moved_from_retries"]
    assert len(proposal_events) - priority["propose_calls_before_transfer"] == moved
    counter_at = [index for index, event in enumerate(events) if event["kind"] == "counterexample_sent"]
    assert counter_at
    assert min(counter_at) > last_multi
    accepted = [
        row for row in result.relationships
        if row.get("reviewer_disposition") == "accepted"
    ]
    assert len(accepted) == 1
    accepted_id = accepted[0]["relationship_id"]
    assert "\u2019" in str(accepted[0].get("quote") or "")
    assert "quotes_not_in_cited_passage" not in (accepted[0].get("flags") or [])
    assert accepted[0].get("quote_scope_cause") is None
    sent_ids = [event.get("relationship_id") for event in events if event["kind"] == "counterexample_sent"]
    assert accepted_id in sent_ids
    rejected = [
        row for row in result.relationships
        if row.get("reviewer_disposition") == "rejected"
    ]
    assert rejected
    rejected_ids = {row["relationship_id"] for row in rejected}
    assert not (rejected_ids & set(sent_ids))
    requests = result.metadata.get("counterexample_requests") or []
    assert not rejected_ids & {row.get("relationship_id") for row in requests}
    skipped = {
        row["relationship_id"]: row["reason"]
        for row in (result.metadata.get("counterexample_targeting") or {}).get("skipped") or []
    }
    assert rejected_ids <= set(skipped)
    assert set(skipped.values()) == {"reviewer_rejected_not_applicable"}
    from automated_grounded_theory.axial import _usage_stage_name

    logged_counts: dict[str, int] = {}
    logged_tokens: dict[str, int] = {}
    for row in result.prompt_log:
        status = str(row.get("status") or "")
        if status in {"omitted", "deferred", "refused", "not_sent"}:
            continue
        stage = _usage_stage_name(row)
        logged_counts[stage] = logged_counts.get(stage, 0) + 1
        usage_row = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        if usage_row.get("input_tokens") is not None:
            logged_tokens[stage] = logged_tokens.get(stage, 0) + int(usage_row["input_tokens"])
    usage = result.metadata["call_usage"]
    for stage in usage["stages"]:
        assert stage["calls"] == logged_counts.get(stage["stage"], 0)
        assert stage["missing_calls"] == 0
    core_tokens = logged_tokens.get("axial_core_review", 0)
    assert core_tokens > 0
    core_stage = next(row for row in usage["stages"] if row["stage"] == "axial_core_review")
    assert core_stage["provider_input_tokens"] == core_tokens
    assert usage["provider_input_tokens"] == sum(logged_tokens.values())
    assert usage["complete"] is True
    assert usage["usage_completeness"] == "complete"
