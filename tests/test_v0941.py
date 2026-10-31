"""0.9.41 coverage, budgeting, and reporting. Synthetic fixtures only.

These checks are engineering and evidence-traceability coverage.
Analytic accuracy is unmeasured. They do not relax eligibility.
"""

from __future__ import annotations

from automated_grounded_theory.async_support import ProviderAllowance
from automated_grounded_theory.axial import (
    AxialProposalBatch,
    _axial_token_plan,
    _procedure_from_kwargs,
    axial_code,
    build_relationship_funnel,
    relationship_funnel_reconciles,
)
from automated_grounded_theory.category_state import plan_relation_pairs
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder
from tests.test_v081 import _procedure


def test_version_is_0941_and_the_minimum_stays_two():
    assert __version__ == "0.9.41"
    assert POLICY_VERSION == "0.9.41"
    assert PATTERN_MIN_CASES == 2


def test_retry_reserve_fits_inside_the_call_allowance():
    procedure = _procedure()
    assert procedure["propose_calls"] == 2
    assert procedure["statement_copy_retry_calls"] == 2
    assert procedure["counterexample_calls"] == 4
    assert procedure["counterexample_token_reserve"] > 0
    assert procedure["counterexample_token_reserve"] < procedure["max_axial_review_tokens"]
    wide = _procedure_from_kwargs(
        {"codebook": {f"c{index}": {"code": str(index)} for index in range(20)}, "passages": []},
        seed=0,
        coding_lens="paradigm",
        paradigm_version="1990",
        min_relation_documents=2,
        max_axial_review_calls=24,
        axial_share_propose=0.5,
        obvious_share=0.5,
        axial_relations=True,
        include_memos=False,
        include_storyline=False,
        axial_synthetic_counterexample=False,
        decision_mode="human",
        max_selective_review_calls=4,
        batch_size=1,
        max_axial_review_tokens=48000,
        axial_token_budget=6000,
    )
    assert wide["propose_calls"] == 6
    assert wide["statement_copy_retry_calls"] == 6
    assert wide["counterexample_calls"] == 12
    plan = _axial_token_plan(
        {"performance": {"axial_token_budget": 6000, "batch_size": 1}},
        wide,
        planned_pairs=20,
    )
    assert plan["planned_calls"] == 24
    assert plan["planned_calls"] <= 24
    assert plan["shortfall"] == 24 * 6000 - 48000
    assert str(plan["shortfall"]) in plan["warning"]
    assert "not raised" in plan["warning"]


def test_counterexample_tokens_are_reserved_before_proposals_spend_them():
    allowance = ProviderAllowance(
        max_tokens=12000,
        stages={"coding": 8},
        counterexample_token_reserve=6000,
    )
    allowance.tokens_estimated = 6000
    proposal = allowance.preview(6000, stage="coding", kind="axial_proposal")
    counter = allowance.preview(6000, stage="coding", kind="axial_counterexample")
    assert proposal["status"] == "refused"
    assert proposal["reason"] == "token_allowance"
    assert counter["status"] == "acquired"


def test_document_spread_outranks_evidence_class_and_keeps_the_minimum():
    evidence = {
        "same_passage": [
            {"from_id": f"s{index}", "to_id": f"s{index + 1}", "document_cooccurrence": 1}
            for index in range(8)
        ],
        "same_respondent": [
            {"from_id": "m1", "to_id": "m2", "document_cooccurrence": 3},
            {"from_id": "q1", "to_id": "q2", "document_cooccurrence": 1},
        ],
        "lexical_cross_case": [
            {"from_id": "m3", "to_id": "m4", "document_cooccurrence": 2},
            {"from_id": "m5", "to_id": "m6", "document_cooccurrence": 4},
        ],
    }
    ids = [f"s{index}" for index in range(9)] + ["m1", "m2", "m3", "m4", "m5", "m6", "q1", "q2"]
    plan = plan_relation_pairs(ids, round_size=8, seed=0, evidence_classes=evidence, exploration_share=0)
    pairs = [{row["from_id"], row["to_id"]} for row in plan["round"]]
    assert {"m1", "m2"} in pairs
    assert {"m3", "m4"} in pairs
    assert {"m5", "m6"} in pairs
    bands = {row["band"] for row in plan["round"]}
    assert "same_respondent" in bands
    assert "lexical_cross_case" in bands
    single_same = [pair for pair in pairs if any(item.startswith("s") for item in pair)]
    assert len(single_same) < 8


class _Nothing(FakeAxialCoder):
    def proposals(self, body):
        return AxialProposalBatch(nothing_to_propose=True, relationships=[])


def test_every_multi_document_pair_has_a_terminal_status():
    source = {
        "codebook": {
            "a": {"code_id": "a", "code": "alpha", "definition": "alpha"},
            "b": {"code_id": "b", "code": "beta", "definition": "beta"},
            "c": {"code_id": "c", "code": "gamma", "definition": "gamma"},
        },
        "passages": [
            {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "alpha beta gamma"},
            {"passage_id": "p2", "doc_id": "d2", "respondent_id": "r2", "text": "alpha beta gamma"},
        ],
        "segments": [
            {"segment_id": "s1", "code_id": "a", "passage_id": "p1"},
            {"segment_id": "s2", "code_id": "b", "passage_id": "p1"},
            {"segment_id": "s3", "code_id": "c", "passage_id": "p1"},
            {"segment_id": "s4", "code_id": "a", "passage_id": "p2"},
            {"segment_id": "s5", "code_id": "b", "passage_id": "p2"},
            {"segment_id": "s6", "code_id": "c", "passage_id": "p2"},
        ],
        "relationships": [],
        "memos": [],
    }
    result = axial_code(
        source,
        _Nothing(),
        seed=1,
        decision_mode="llm",
        interrupt=False,
        max_axial_review_calls=8,
        axial_token_budget=6000,
        batch_size=1,
    )
    rows = result.metadata["multi_document_pairs"]
    assert rows
    assert {row["status"] for row in rows} <= {
        "unplanned",
        "nothing_to_propose",
        "budget_blocked",
        "failed",
        "proposed",
    }
    assert any(row["status"] == "nothing_to_propose" for row in rows)
    assert all(row["unit"] == "documents" for row in rows)
    assert all(row["document_count"] >= 2 for row in rows)
    assert not any(row["status"] == "failed" for row in rows)


def test_an_oversized_counterexample_is_shortened_or_the_reason_is_exact():
    long_text = "tools " * 8000
    source = {
        "codebook": {
            "c1": {"code_id": "c1", "code": "tools", "definition": "tools"},
            "c2": {"code_id": "c2", "code": "bench", "definition": "bench"},
        },
        "passages": [
            {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "tools on the bench"},
            {"passage_id": "p-long", "doc_id": "d9", "respondent_id": "r9", "text": long_text},
        ],
        "segments": [
            {"segment_id": "s1", "code_id": "c1", "passage_id": "p1"},
            {"segment_id": "s2", "code_id": "c2", "passage_id": "p1"},
            {"segment_id": "s3", "code_id": "c1", "passage_id": "p-long"},
        ],
        "relationships": [
            {
                "relationship_id": "rel-1",
                "from_id": "c1",
                "to_id": "c2",
                "passage_id": "p1",
                "statement": "Tools sit with the bench.",
                "quote": "tools on the bench",
                "disposition": "proposed",
                "evidence_tier": 2,
            }
        ],
        "memos": [],
    }
    common = dict(
        seed=1,
        decision_mode="human",
        interrupt=False,
        axial_relations=False,
        max_axial_review_calls=4,
        axial_token_budget=6000,
        batch_size=1,
    )
    sent = axial_code(source, FakeAxialCoder(), max_axial_review_tokens=20000, **common)
    blocked = axial_code(source, FakeAxialCoder(), max_axial_review_tokens=1000, **common)
    assert sent.metadata.get("counterexample_omitted_context")
    assert sent.metadata.get("counterexample_adequacy") == "unassessed"
    kept = sent.metadata["counterexample_omitted_context"]["kept"]
    assert "proposed_claim" in kept
    assert "provenance" in kept
    unsent = blocked.metadata["counterexample_unsent"]
    assert unsent["reason"] == "token_allowance"
    assert "Adequacy is unassessed" in unsent["detail"]
    assert unsent["adequacy"] == "unassessed"
    assert any(item.get("from") == "counterexamples" and item.get("reason") == "token_allowance" for item in blocked.metadata["allowance_transfers"])


def test_reasons_stay_independent_and_a_mismatch_does_not_reconcile():
    state = {
        "codebook": {"c1": {}, "c2": {}},
        "passages": [],
        "relationships": [
            {
                "relationship_id": "both",
                "from_id": "c1",
                "to_id": "c2",
                "reviewer_disposition": "rejected",
                "downstream_eligible": False,
                "document_count": 1,
                "eligibility_blocks": ["hypothetical_link"],
                "flags": ["participant_link_hypothetical"],
            }
        ],
        "metadata": {"procedure": {"pattern_min_cases": 2, "one_document_one_respondent": False}},
    }
    funnel = build_relationship_funnel(state)
    reasons = funnel["rows"][0]["reasons"]
    assert "reviewer_rejected" in reasons
    assert "insufficient_document_count" in reasons
    assert "hypothetical_only_evidence" in reasons
    assert funnel["rows"][0]["document_count_unit"] == "documents"
    assert funnel["row_funnel"]["reconciles"] is True
    assert relationship_funnel_reconciles(
        [
            {"stage": "consolidated", "count": 2, "ids": ["a", "b"], "dropped_from_previous": 0, "reasons": {}},
            {"stage": "reviewed", "count": 2, "ids": ["a"], "dropped_from_previous": 0, "reasons": {}},
        ]
    ) is False
    assert relationship_funnel_reconciles(
        [
            {"stage": "consolidated", "count": 1, "ids": ["a"], "dropped_from_previous": 0, "reasons": {}},
            {
                "stage": "reviewed",
                "count": 0,
                "ids": [],
                "dropped_from_previous": 1,
                "dropped_ids": ["a"],
                "reasons_by_id": {"a": []},
                "reasons": {},
            },
        ]
    ) is False


def test_relation_review_is_one_assessment_per_chunk():
    source = {
        "codebook": {
            "c-prep": {"code_id": "c-prep", "code": "packing", "definition": "packing"},
            "c-return": {"code_id": "c-return", "code": "return", "definition": "return"},
        },
        "passages": [{"passage_id": "p1", "doc_id": "d1", "text": "Packing the suitcase took all night."}],
        "segments": [],
        "relationships": [
            {
                "relationship_id": f"rel-{index:04d}",
                "from_id": "c-prep",
                "to_id": "c-return",
                "statement": f"Packing is a condition of returning {index}.",
                "quote": "Packing the suitcase took all night.",
                "passage_id": "p1",
                "evidence_tier": 2,
                "disposition": "proposed",
                "paradigm_role": "context",
            }
            for index in range(4)
        ],
        "memos": [],
    }
    coder = FakeAxialCoder()
    result = axial_code(
        source,
        coder,
        seed=1,
        decision_mode="llm",
        interrupt=False,
        axial_relations=False,
        max_axial_review_calls=0,
        max_decision_calls=8,
        axial_token_budget=6000,
    )
    report = result.metadata["decision_reviews"]["axial_relation_review"]
    assert report["chunks_sent"] == 1
    assert report["items_reviewed"] == 4
    assert result.metadata["relation_review_assessment"]["assessments_per_row"] == 1
    assert result.metadata["relation_review_assessment"]["schema_repair"] is False
    relation_calls = [row for row in result.prompt_log if row.get("checkpoint") == "axial_relation_review"]
    assert len(relation_calls) == 1


def test_coding_stage_tokens_come_from_the_ledger():
    source = {
        "codebook": {
            "c1": {"code_id": "c1", "code": "tools", "definition": "tools"},
            "c2": {"code_id": "c2", "code": "bench", "definition": "bench"},
        },
        "passages": [{"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "the tools sat on the bench"}],
        "segments": [
            {"segment_id": "s1", "code_id": "c1", "passage_id": "p1"},
            {"segment_id": "s2", "code_id": "c2", "passage_id": "p1"},
        ],
        "relationships": [],
        "memos": [],
    }
    result = axial_code(
        source,
        FakeAxialCoder(),
        seed=1,
        decision_mode="human",
        interrupt=False,
        max_axial_review_calls=8,
        axial_token_budget=6000,
        batch_size=1,
    )
    coding = result.metadata["provider_budget"]["stages"]["coding"]
    assert coding["tokens_used"] > 0
    assert result.metadata["coding_tokens_used"] == coding["tokens_used"]
    assert result.metadata["axial_token_plan"]["coding_tokens_used"] == coding["tokens_used"]
    assert result.metadata["axial_token_plan"]["tokens_used_source"] == "coding_stage_ledger"
