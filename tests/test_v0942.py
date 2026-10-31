"""0.9.42 coverage, budgeting, and reporting. Synthetic fixtures only.

These checks are engineering and evidence-traceability coverage.
Analytic accuracy is unmeasured. Eligibility is not relaxed.
"""

from __future__ import annotations

from automated_grounded_theory.axial import (
    AxialCounterexampleBatch,
    AxialProposalBatch,
    AxialRelationshipDraft,
    _messages,
    _shrink_counterexample_payload,
    axial_code,
    build_relationship_funnel,
    consolidate_duplicate_assertions,
    counterexample_token_breakdown,
    reconcile_proposal_lineage,
)
from automated_grounded_theory.batching import request_input_tokens
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder


def test_version_is_0942():
    assert __version__ == "0.9.42"
    assert POLICY_VERSION == "0.9.42"
    assert PATTERN_MIN_CASES == 2


def _fat_compacted_request(other_chars: int) -> dict:
    """The pre-0.9.42 shape: the whole relationship row, passage already compacted."""
    relationship = {
        "relationship_id": "rel-0001",
        "from_id": "c1",
        "to_id": "c2",
        "statement": "Tools sit with the bench when the shift runs long.",
        "quote": "tools on the bench",
        "passage_id": "p1",
        "doc_id": "d1",
        "evidence_tier": 2,
        "supporting_cases": [{"case_id": f"r{index}", "quotes": ["tools on the bench"]} for index in range(12)],
        "claims": {"pattern": {"present": True, "evidence_tier": 2}},
        "cooccurrence_2x2": {"both": 12, "a_only": 4, "b_only": 3, "neither": 40},
        "review_history": [{"actor": "llm", "disposition": "proposed", "rationale": "x" * 80} for _ in range(6)],
        "flags": ["endpoint_evidence_partial"],
        "derived_metadata": "m" * other_chars,
    }
    return {
        "counterexamples": [
            {
                "relationship": relationship,
                "passage": {"passage_id": "p-long", "doc_id": "d9", "text": "x" * 40, "excerpt": True},
                "complementary_passages": [],
                "respondent_id": "r9",
                "source": "passage_comparison",
                "comparison": "",
            }
        ]
    }


def test_a_compacted_full_row_is_9466_tokens_because_of_other_context():
    payload = _fat_compacted_request(24180)
    total = request_input_tokens(_messages("axial_counterexample", payload), AxialCounterexampleBatch)
    assert total == 9466
    assert total > 6000
    breakdown = counterexample_token_breakdown(payload)
    assert breakdown["total"] == 9466
    assert breakdown["instructions"] == 407
    assert breakdown["schema"] == 241
    assert breakdown["claim"] == 17
    assert breakdown["evidence"] == 14
    assert breakdown["provenance"] == 27
    assert breakdown["other_context"] == 8760
    assert breakdown["other_context"] > breakdown["evidence"]
    assert (
        breakdown["instructions"]
        + breakdown["schema"]
        + breakdown["claim"]
        + breakdown["category_definitions"]
        + breakdown["evidence"]
        + breakdown["provenance"]
        + breakdown["other_context"]
    ) == 9466


def test_required_content_that_cannot_fit_is_blocked_not_truncated_or_skipped():
    claim = "Tools sit with the bench. " * 800
    payload = {
        "counterexamples": [
            {
                "relationship": {
                    "relationship_id": "rel-1",
                    "from_id": "c1",
                    "to_id": "c2",
                    "statement": claim,
                    "categories": [
                        {"code_id": "c1", "code": "tools", "definition": "tools"},
                        {"code_id": "c2", "code": "bench", "definition": "bench"},
                    ],
                    "provenance": {"relationship_id": "rel-1", "passage_id": "p1", "doc_id": "d1", "quote": "tools"},
                },
                "passage": {"passage_id": "p1", "doc_id": "d1", "text": "tools " * 20},
                "complementary_passages": [],
            }
        ]
    }
    before = payload["counterexamples"][0]["relationship"]["statement"]
    estimate, fitted = _shrink_counterexample_payload(payload, AxialCounterexampleBatch, 6000)
    assert fitted is False
    assert estimate > 6000
    assert payload["counterexamples"][0]["relationship"]["statement"] == before
    assert payload["counterexamples"][0]["passage"]["text"].count("tools") >= 1
    breakdown = counterexample_token_breakdown(payload)
    required = (
        breakdown["instructions"]
        + breakdown["schema"]
        + breakdown["claim"]
        + breakdown["category_definitions"]
        + breakdown["evidence"]
        + breakdown["provenance"]
    )
    assert required > 6000
    assert breakdown["claim"] > breakdown["other_context"]


def test_consolidation_keeps_both_decisions_and_reviews_the_merged_row():
    accepted = {
        "relationship_id": "rel-0002",
        "proposal_id": "prop-0002",
        "from_id": "c1",
        "to_id": "c2",
        "statement": "Tools sit with the bench.",
        "disposition": "accepted",
        "reviewer_disposition": "accepted",
        "review_history": [{"actor": "llm", "disposition": "accepted"}],
        "passage_id": "p1",
        "quote": "tools",
    }
    rejected = {
        "relationship_id": "rel-0001",
        "proposal_id": "prop-0001",
        "from_id": "c1",
        "to_id": "c2",
        "statement": "Tools sit with the bench.",
        "disposition": "rejected",
        "reviewer_disposition": "rejected",
        "review_history": [{"actor": "llm", "disposition": "rejected"}],
        "passage_id": "p2",
        "quote": "bench",
    }
    merged = consolidate_duplicate_assertions([accepted, rejected])
    assert len(merged) == 1
    row = merged[0]
    assert row["disposition"] == "proposed"
    assert row["reviewer_disposition"] is None
    assert {item["disposition"] for item in row["prior_decisions"]} == {"accepted", "rejected"}
    assert len(row["review_history"]) == 2
    assert row["conflicting_evidence"]
    assert row["source_proposal_ids"] == ["prop-0001", "prop-0002"]


def test_lineage_fails_when_merged_ids_are_relabelled_as_proposals():
    raw = [
        {"proposal_id": "prop-0001", "disposition": "retained", "relationship_id": "rel-0001"},
        {"proposal_id": "prop-0002", "disposition": "merged_into:rel-0001", "relationship_id": "rel-0001"},
        {"proposal_id": "prop-0003", "disposition": "retained", "relationship_id": "rel-0002"},
        {"proposal_id": "prop-0004", "disposition": "merged_into:rel-0001", "relationship_id": "rel-0001"},
        {"proposal_id": "prop-0005", "disposition": "merged_into:rel-0002", "relationship_id": "rel-0002"},
    ]
    relationships = [
        {"relationship_id": "rel-0001", "proposal_id": "prop-0001", "source_proposal_ids": ["prop-0001", "prop-0002", "prop-0004"]},
        {"relationship_id": "rel-0002", "proposal_id": "prop-0003", "source_proposal_ids": ["prop-0003", "prop-0005"]},
    ]
    assert reconcile_proposal_lineage(raw, relationships)["reconciles"] is True
    assert reconcile_proposal_lineage(raw, relationships)["raw_count"] == 5
    assert reconcile_proposal_lineage(raw, relationships)["relationship_count"] == 2
    mislabeled = [dict(row, disposition="retained") for row in raw]
    assert reconcile_proposal_lineage(mislabeled, relationships)["reconciles"] is False


def test_missing_provider_usage_stays_null():
    from automated_grounded_theory.axial import _call_usage_report

    report = _call_usage_report([
        {"kind": "axial_proposal", "estimated_input_tokens": 120, "usage": {}},
        {
            "kind": "axial_counterexample",
            "estimated_input_tokens": 80,
            "usage": {"input_tokens": 30, "output_tokens": 12},
        },
    ])
    assert report["calls"][0]["estimated_input_tokens"] == 120
    assert report["calls"][0]["provider_input_tokens"] is None
    assert report["calls"][0]["provider_output_tokens"] is None
    assert report["calls"][0]["usage_completeness"] == "unknown"
    assert report["calls"][1]["provider_input_tokens"] == 30
    assert report["calls"][1]["provider_output_tokens"] == 12
    assert report["usage_completeness"] == "partial"
    assert report["estimated_input_tokens"] == 200
    assert report["provider_input_tokens"] == 30


def test_summary_reasons_do_not_drop_partial_endpoint_evidence():
    funnel = build_relationship_funnel({
        "codebook": {},
        "passages": [],
        "relationships": [
            {
                "relationship_id": "rel-1",
                "from_id": "c1",
                "to_id": "c2",
                "reviewer_disposition": "rejected",
                "document_count": 1,
                "downstream_eligible": False,
                "endpoint_anchoring": [
                    {"endpoint": "from", "anchored": True},
                    {"endpoint": "to", "anchored": False},
                ],
                "flags": ["participant_link_hypothetical"],
                "eligibility_blocks": ["hypothetical_link"],
            }
        ],
        "metadata": {"procedure": {"pattern_min_cases": 2, "one_document_one_respondent": False}},
    })
    row = funnel["rows"][0]
    assert row["summary_reasons"] == [
        "reviewer_rejected",
        "insufficient_document_count",
        "hypothetical_only_evidence",
    ]
    assert "partial_endpoint_evidence" in row["reasons"]
    assert "incomplete_review" not in row["reasons"]


class _Sequence(FakeAxialCoder):
    def __init__(self):
        super().__init__()
        object.__setattr__(self, "proposal_pairs", [])

    def proposals(self, body):
        pair = (body.get("pairs") or [{}])[0]
        self.proposal_pairs.append((str(pair.get("from_id") or ""), str(pair.get("to_id") or "")))
        quote = "tools sit with the bench"
        if len(self.proposal_pairs) == 1:
            drafts = [
                AxialRelationshipDraft(
                    from_id=pair.get("from_id") or "a",
                    to_id=pair.get("to_id") or "b",
                    statement="Categories co-occur across the two documents.",
                    passage_id="p1",
                    quote=quote,
                )
                for _ in range(4)
            ]
            drafts.append(
                AxialRelationshipDraft(
                    from_id=pair.get("from_id") or "a",
                    to_id=pair.get("to_id") or "b",
                    statement="A different assertion links the same two categories.",
                    passage_id="p1",
                    quote=quote,
                )
            )
            return AxialProposalBatch(relationships=drafts)
        return AxialProposalBatch(
            relationships=[
                AxialRelationshipDraft(
                    from_id=pair.get("from_id") or "a",
                    to_id=pair.get("to_id") or "b",
                    statement="The later pair has its own short claim.",
                    passage_id="p1",
                    quote=quote,
                )
            ]
        )

    def relation_review(self, body):
        from automated_grounded_theory.axial import AxialRelationDecision, AxialRelationDecisions

        return AxialRelationDecisions(
            relationships=[
                AxialRelationDecision(
                    relationship_id=str(row.get("relationship_id") or ""),
                    disposition="rejected",
                    paradigm_role="context",
                    rationale="The endpoints are not both anchored.",
                )
                for row in body.get("relationships") or []
            ]
        )


def test_duplicate_proposals_consolidate_before_review_and_transferred_calls_run():
    codes = ["a", "b", "c", "d"]
    source = {
        "codebook": {code: {"code_id": code, "code": code, "definition": f"the {code} category"} for code in codes},
        "passages": [
            {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "tools sit with the bench"},
            {"passage_id": "p2", "doc_id": "d2", "respondent_id": "r2", "text": "tools sit with the bench"},
        ],
        "segments": [
            {"segment_id": f"s{doc}{code}", "code_id": code, "passage_id": passage}
            for doc, passage in (("1", "p1"), ("2", "p2"))
            for code in codes
        ],
        "relationships": [
            {
                "relationship_id": "rel-v46",
                "from_id": "a",
                "to_id": "b",
                "statement": "Tools sit with the bench across these documents.",
                "quote": "tools sit with the bench",
                "passage_id": "p1",
                "doc_id": "d1",
                "disposition": "proposed",
                "evidence_tier": 2,
                "paradigm_role": "context",
                "review_history": [
                    {"actor": "llm", "disposition": "proposed", "rationale": "history " * 40}
                    for _ in range(30)
                ],
                "derived_metadata": "m" * 24180,
                "supporting_cases": [{"case_id": "r1", "quotes": ["tools sit with the bench"]}],
                "cooccurrence_2x2": {"both": 2, "a_only": 0, "b_only": 0, "neither": 0},
                "flags": ["endpoint_evidence_partial"],
            },
            {
                "relationship_id": "rel-huge",
                "from_id": "c",
                "to_id": "d",
                "statement": "The required counterexample claim. " * 700,
                "quote": "tools sit with the bench",
                "passage_id": "p1",
                "doc_id": "d1",
                "disposition": "proposed",
                "evidence_tier": 2,
                "paradigm_role": "context",
                "categories_note": "definitions stay on the codebook; the claim itself does not fit",
            },
        ],
        "memos": [],
    }
    coder = _Sequence()
    result = axial_code(
        source,
        coder,
        seed=1,
        decision_mode="llm",
        interrupt=False,
        max_axial_review_calls=4,
        axial_share_propose=0.5,
        axial_share_counterexample=0.5,
        axial_token_budget=6000,
        max_axial_review_tokens=24000,
        batch_size=1,
        negative_case_search=True,
    )
    initial = result.metadata["initial_relation_round"]
    assert len(initial) == 1
    initial_pair = tuple(sorted((initial[0]["from_id"], initial[0]["to_id"])))
    assert tuple(sorted(coder.proposal_pairs[0])) == initial_pair
    events = result.metadata["axial_event_order"]
    kinds = [event["kind"] for event in events]
    assert kinds.index("proposal_sent") < kinds.index("consolidation_before_review")
    assert kinds.index("consolidation_before_review") < kinds.index("relation_review_sent")
    assert kinds.index("relation_review_sent") < kinds.index("counterexample_sent")
    assert kinds.index("counterexample_sent") < kinds.index("counterexample_blocked")
    assert kinds.index("counterexample_blocked") < kinds.index("allowance_transfer")
    transfer_at = kinds.index("allowance_transfer")
    extra_events = [event for event in events[transfer_at + 1 :] if event["kind"] == "proposal_sent"]
    assert extra_events
    assert all(event["reallocated"] is True for event in extra_events)
    assert events[kinds.index("proposal_sent")]["reallocated"] is False
    reallocation = result.metadata["counterexample_reallocation"]
    assert reallocation["calls_executed"] == reallocation["calls_transferred"] == len(extra_events)
    assert reallocation["calls_unsent"] == 0
    assert reallocation["calls_unsent"] != reallocation["calls_executed"]
    queued = [tuple(sorted((pair["from_id"], pair["to_id"]))) for pair in reallocation["pairs"]]
    executed = [tuple(sorted((pair["from_id"], pair["to_id"]))) for event in extra_events for pair in event["pairs"]]
    assert executed == queued
    assert initial_pair not in executed
    multi = {
        tuple(sorted((row["from_id"], row["to_id"])))
        for row in result.metadata["multi_document_pairs"]
        if int(row["document_count"]) >= 2
    }
    assert executed
    assert all(pair in multi for pair in executed)
    sent = next(row for row in result.metadata["counterexample_requests"] if row["status"] == "sent")
    assert sent["relationship_id"] == "rel-v46"
    assert sent["estimated_input_tokens"] <= 6000
    assert sent["token_breakdown"]["total"] <= 6000
    assert sent["token_breakdown"]["other_context"] < 1000
    assert len(result.relationships[0]["derived_metadata"]) == 24180
    blocked = next(row for row in result.metadata["counterexample_requests"] if row["status"] == "required_but_blocked")
    blocked_parts = blocked["token_breakdown"]
    required = (
        blocked_parts["instructions"]
        + blocked_parts["schema"]
        + blocked_parts["claim"]
        + blocked_parts["category_definitions"]
        + blocked_parts["evidence"]
        + blocked_parts["provenance"]
    )
    assert required > 6000
    assert blocked["relationship_id"] == "rel-huge"
    stage = result.metadata["stage_outcomes"]["counterexamples"]
    assert stage["status"] == "required_but_blocked"
    assert stage["required"] is True
    review = result.metadata["relation_review_rounds"][0]
    assert review["review_calls"] <= review["consolidated_rows"]
    assert review["raw_proposal_members"] == 5
    assert review["review_calls"] < review["raw_proposal_members"]
    lineage = result.metadata["proposal_lineage"]
    assert lineage["reconciles"] is True
    raw = result.metadata["raw_proposals"]
    first_raw = [row for row in raw if {row["from_id"], row["to_id"]} == set(initial_pair)]
    assert len(first_raw) == 5
    assert sum(1 for row in first_raw if row["disposition"] == "retained") == 2
    assert sum(1 for row in first_raw if str(row["disposition"]).startswith("merged_into:")) == 3
    summary_codes = (
        "reviewer_rejected",
        "insufficient_document_count",
        "hypothetical_only_evidence",
        "incomplete_review",
    )
    funnel_rows = {row["relationship_id"]: row for row in result.metadata["relationship_funnel"]["rows"]}
    consolidated = [
        row for row in result.relationships
        if {row.get("from_id"), row.get("to_id")} == set(initial_pair)
    ]
    assert len(consolidated) == 2
    for row in consolidated:
        assert row.get("reviewer_disposition") == "rejected"
        assert "endpoint_evidence_partial" in (row.get("flags") or [])
        funnel_row = funnel_rows[row["relationship_id"]]
        assert funnel_row["summary_reasons"] == [code for code in summary_codes if code in funnel_row["reasons"]]
        assert "reviewer_rejected" in funnel_row["summary_reasons"]
        assert "partial_endpoint_evidence" in funnel_row["reasons"]
        assert "partial_endpoint_evidence" not in funnel_row["summary_reasons"]
        assert "incomplete_review" not in funnel_row["reasons"]
        if len(row.get("source_proposal_ids") or []) > 1:
            assert len(row["prior_decisions"]) == len(row["source_proposal_ids"])
            assert {item["disposition"] for item in row["prior_decisions"]} == {"proposed"}
            assert row["review_history"]
    usage = result.metadata["call_usage"]
    assert usage["estimated_input_tokens"] is not None
    assert usage["provider_input_tokens"] is not None
    assert usage["estimated_input_tokens"] != usage["provider_input_tokens"]
    assert all(row["usage_completeness"] == "complete" for row in usage["calls"] if row["kind"] == "axial_proposal")
