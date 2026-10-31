"""0.9.33 review fixes. Fakes and synthetic fixtures only. No network calls.

These checks are processing coverage. They do not establish selection validity.
"""

from __future__ import annotations

from automated_grounded_theory import (
    CASE_IDENTITY_SOURCE_RULE,
    CONDITION_STATEMENT_RULE,
    CORE_COMPARISON_RULE,
    REREAD_REQUEST_RULE,
    __version__,
)
from automated_grounded_theory.axial import _axial_token_plan
from automated_grounded_theory.decision_review import (
    BUDGET_LEDGER_RULE,
    CHUNKING_CHECKPOINT_MESSAGE,
    PER_STEP_DECISION_REVIEW,
    plan_core_comparison,
    review_procedure_fields,
    same_chunking_fingerprint,
)
from automated_grounded_theory.focused import (
    REREAD_REQUEST_EXCEEDS_TOKEN_BUDGET,
    _focused_reread_preflight,
    _prepare_node,
    _reread_estimate_for,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import (
    PATTERN_MIN_CASES,
    apply_evidence_tier,
    effective_statement,
)
from tests.test_v0926 import SENTENCE, _book, _both
from tests.test_v0931 import _LINK, _tier


def test_version_fingerprint_and_unchanged_defaults():
    assert __version__ == "0.9.41"
    assert POLICY_VERSION == "0.9.41"
    assert PATTERN_MIN_CASES == 2
    assert BUDGET_LEDGER_RULE == "per_step_review_ledger_v1"
    assert CORE_COMPARISON_RULE == "deterministic_tournament_v1"
    assert REREAD_REQUEST_RULE == "pack_to_request_budget_v1"
    assert CONDITION_STATEMENT_RULE == "retain_recorded_condition_v1"
    assert CASE_IDENTITY_SOURCE_RULE == "source_or_declared_document_v1"
    assert PER_STEP_DECISION_REVIEW == "per_step_decision_review"
    assert "0.9.28" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.33" in CHUNKING_CHECKPOINT_MESSAGE
    current = review_procedure_fields()
    assert same_chunking_fingerprint(current, current) is True
    older = dict(current)
    older.pop("core_comparison_rule")
    assert same_chunking_fingerprint(older, current) is False


def _packed_state() -> dict:
    definition = "packing tools for the night shift " * 80
    codes = {
        f"c{index:02d}": {"code_id": f"c{index:02d}", "code": f"code {index}", "definition": definition}
        for index in range(40)
    }
    short = {"passage_id": "short", "doc_id": "d0", "text": "tools for the shift"}
    huge = {"passage_id": "huge", "doc_id": "d1", "text": "Packing the suitcase before the night shift. " * 800}
    return {
        "queue_ready": True,
        "recode_queue": [
            {"passage_id": "huge", "purpose": "develop"},
            {"passage_id": "short", "purpose": "develop"},
        ],
        "recode_cursor": 0,
        "passages": [huge, short],
        "codebook": codes,
        "raised_code_ids": list(codes),
        "segments": [
            {"segment_id": "s0", "code_id": "c00", "passage_id": "short", "quote": "tools for the shift"},
        ],
        "properties": [],
        "dimensions": [],
        "memos": [],
        "focused_calls_used": 0,
        "focused_tokens_estimated": 0,
        "metadata": {
            "procedure": {
                "max_focused_review_calls": 8,
                "max_focused_review_tokens": 24000,
                "validation_retries": 0,
                "focused_claim_compete_round": True,
                "seed": 1,
                "focused_neighbor_limit": 0,
                "focused_outside_limit": 0,
                "focused_negative_case_limit": 0,
                "focused_share_develop": 1.0,
                "focused_share_compete": 0.0,
                "focused_share_outside": 0.0,
                "focused_share_claim_compete": 0.0,
            },
            "performance": {"batch_size": 1, "token_budget": 6000},
            "reread_call_targets": {"claim_compete": 2},
        },
    }


def test_a_reread_is_packed_to_the_cap_and_one_oversized_passage_does_not_block_the_rest():
    state = _packed_state()
    prepared = _prepare_node(state)
    assert prepared.get("recode_stop") != "reread_allowance_insufficient"
    jobs = prepared.get("round_jobs") or []
    assert len(jobs) == 1
    assert jobs[0]["passage_ids"] == ["short"]
    assert jobs[0]["estimate"] <= 6000
    assert int(jobs[0]["code_level"]) >= 1
    over = prepared["metadata"]["reread_requests_over_budget"]
    assert over[0]["passage_id"] == "huge"
    assert over[0]["reason"] == REREAD_REQUEST_EXCEEDS_TOKEN_BUDGET
    assert over[0]["estimate"] > 6000

    estimate = _reread_estimate_for(state, list(state["raised_code_ids"]), state["recode_queue"])
    assert estimate is not None
    assert estimate["short"] is False
    assert estimate["focused_review_requests_withheld"] == 1
    assert estimate["focused_review_requests_projected"] == 1
    assert estimate["focused_review_tokens_required"] <= 6000
    assert estimate["focused_review_tokens_required"] > 0
    preflight = _focused_reread_preflight(state)
    assert preflight is not None
    assert preflight["blocked"] is False
    assert preflight["focused_review_requests_projected_worst_case"] is not None
    assert preflight["focused_review_tokens_required_worst_case"] > 0

    only_huge = _packed_state()
    only_huge["recode_queue"] = [{"passage_id": "huge", "purpose": "develop"}]
    only_huge["metadata"] = {
        **only_huge["metadata"],
        "reread_call_targets": {"claim_compete": 2},
    }
    held = _prepare_node(only_huge)
    assert held.get("round_jobs") in (None, [])
    assert "recode_stop" not in held
    assert held["recode_cursor"] == 1
    assert held["metadata"]["reread_requests_over_budget"][0]["passage_id"] == "huge"


def test_core_comparison_is_one_request_or_a_withhold():
    fitted = plan_core_comparison(["c0", "c1", "c2"], lambda ids: 100 * len(list(ids)), per_request=6000)
    assert fitted["chunks"] == [["c0", "c1", "c2"]]
    assert fitted["compared"] == 3
    assert fitted["eligible"] == 3
    assert fitted["reason"] is None
    alone = plan_core_comparison(["c0"], lambda ids: 100, per_request=6000)
    assert alone["chunks"] == [["c0"]]
    assert alone["compared"] == 1
    blocked = plan_core_comparison(["c0", "c1", "c2"], lambda ids: 5000 * len(list(ids)), per_request=6000)
    assert blocked["chunks"] == []
    assert blocked["compared"] == 0
    assert blocked["eligible"] == 3
    assert blocked["reason"] == "core_pair_exceeds_request_budget"


def test_a_recorded_condition_stays_on_the_presented_statement():
    row = _tier(f"{_LINK} if it rains")
    assert row["statement"] == _LINK
    assert row["condition"] == "if it rains"
    assert row["condition_retained"] is False
    assert row["statement_omits_condition"] is True
    assert row["presented_statement"] == f"{_LINK} (if it rains)"
    assert effective_statement(row) == row["presented_statement"]
    assert "condition_not_in_statement" in row["flags"]
    assert row["evidence_tier"] != 3

    retained = _tier(f"{_LINK} if it rains", statement=f"{_LINK} if it rains")
    assert retained["condition_retained"] is True
    assert retained["statement_omits_condition"] is False
    assert retained["presented_statement"] == f"{_LINK} if it rains"
    assert retained["presented_statement"].count("if it rains") == 1
    assert "condition_not_in_statement" not in retained["flags"]


def test_the_coding_pool_warning_names_per_step_allowances():
    plan = _axial_token_plan(
        {"performance": {"axial_token_budget": 6000, "batch_size": 1}},
        {
            "max_axial_review_tokens": 24000,
            "propose_calls": 4,
            "counterexample_calls": 4,
            "memo_calls": 0,
            "storyline_calls": 0,
        },
    )
    assert plan["statement_copy_retry_calls_reserved"] == 4
    assert plan["estimate"] == 72000
    assert plan["ceiling"] == 24000
    assert plan["shortfall"] == 48000
    assert plan["changes_default"] is False
    assert "48000" in plan["warning"]
    assert "24000" in plan["warning"]
    assert "not raised" in plan["warning"]
    assert "Relation review" in plan["warning"]
    assert "coding_tokens_used" in plan["note"]
    quiet = _axial_token_plan(
        {"performance": {"axial_token_budget": 6000, "batch_size": 1}},
        {"max_axial_review_tokens": 24000, "propose_calls": 2, "counterexample_calls": 0},
        planned_pairs=2,
        pool_pairs=10,
    )
    assert quiet["warning"] == ""
    assert quiet["pool_estimate_label"] == "raw unordered pool, not compared with the ceiling"


def _document_row(declared: bool = False) -> dict:
    passages = {}
    for index, doc in enumerate(("question-9", "question-10"), start=1):
        passages[f"p{index}"] = {
            "passage_id": f"p{index}",
            "doc_id": doc,
            "text": f"The shift notes say {SENTENCE}",
        }
    row = {
        "relationship_id": "rel-docs",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": "",
        "statement": "tools for the shift may be related to coworkers on the crew",
        "disposition": "proposed",
        "case_quotes": [_both("p1", SENTENCE), _both("p2", SENTENCE)],
    }
    apply_evidence_tier(
        row,
        passages=passages,
        codebook=_book(),
        segments=[],
        memos=[],
        one_document_one_respondent=declared,
    )
    return row


def test_case_identity_names_a_document_fallback_and_a_declared_respondent():
    fallback = _document_row(False)
    assert fallback["evidence_tier"] == 2
    assert fallback["case_counts_are_respondents"] is False
    assert "documents" in fallback["allowed_wording"]
    assert "respondents" not in fallback["allowed_wording"]
    assert {case["case_identity_source"] for case in fallback["supporting_cases"]} == {"document_fallback"}
    assert {case["case_id"] for case in fallback["supporting_cases"]} == {"question-9", "question-10"}
    assert fallback["case_identity_summary"]["document_fallback"] == 2
    assert fallback["case_identity_summary"]["case_counts_are_respondents"] is False

    declared = _document_row(True)
    assert declared["evidence_tier"] == 2
    assert declared["one_document_one_respondent"] is True
    assert declared["case_counts_are_respondents"] is True
    assert "respondents" in declared["allowed_wording"]
    assert {case["case_identity_source"] for case in declared["supporting_cases"]} == {"one_document_one_respondent"}

    named = {
        "relationship_id": "rel-people",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "statement": "tools for the shift may be related to coworkers on the crew",
        "disposition": "proposed",
        "case_quotes": [_both("p1", SENTENCE), _both("p2", SENTENCE)],
    }
    passages = {
        "p1": {"passage_id": "p1", "doc_id": "question-9", "respondent_id": "r1", "text": f"The shift notes say {SENTENCE}"},
        "p2": {"passage_id": "p2", "doc_id": "question-10", "respondent_id": "r2", "text": f"The shift notes say {SENTENCE}"},
    }
    apply_evidence_tier(named, passages=passages, codebook=_book(), segments=[], memos=[])
    assert named["evidence_tier"] == 2
    assert named["case_counts_are_respondents"] is True
    assert "respondents" in named["allowed_wording"]
    assert {case["case_identity_source"] for case in named["supporting_cases"]} == {"respondent"}

    one_person = {
        "relationship_id": "rel-one",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "statement": "tools for the shift may be related to coworkers on the crew",
        "disposition": "proposed",
        "case_quotes": [_both("p1", SENTENCE), _both("p2", SENTENCE)],
    }
    same = {
        "p1": {"passage_id": "p1", "doc_id": "question-9", "respondent_id": "r1", "text": f"The shift notes say {SENTENCE}"},
        "p2": {"passage_id": "p2", "doc_id": "question-10", "respondent_id": "r1", "text": f"The shift notes say {SENTENCE}"},
    }
    apply_evidence_tier(one_person, passages=same, codebook=_book(), segments=[], memos=[])
    assert one_person["evidence_tier"] == 1
    assert len(one_person["supporting_cases"]) == 1
    assert one_person["case_counts_are_respondents"] is True
