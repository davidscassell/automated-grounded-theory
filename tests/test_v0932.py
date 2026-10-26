"""0.9.32 review fixes. Fakes and synthetic fixtures only. No network calls.

These checks are processing coverage. They do not establish selection validity.
"""

from __future__ import annotations

from automated_grounded_theory import (
    CASE_COUNT_RULE,
    DECISION_TOKEN_SCOPE,
    REREAD_ESTIMATE_RULE,
    __version__,
    export_gate_relationships,
    selective_code,
)
from automated_grounded_theory.async_support import ProviderAllowance, run_sync
from automated_grounded_theory.axial import (
    DEFAULT_MAX_AXIAL_REVIEW_TOKENS,
    AxialDecisionReviewer,
    AxialRelationDecisions,
    ModelCoreDecision,
    _commit_relations,
    _core_review_payload,
    _relation_review_payload,
)
from automated_grounded_theory.decision_review import (
    CHUNKING_CHECKPOINT_MESSAGE,
    DECISION_REVIEW_CHUNKING,
    budget_ledger,
    review_procedure_fields,
    same_chunking_fingerprint,
)
from automated_grounded_theory.focused import (
    REREAD_ALLOWANCE_INSUFFICIENT,
    REREAD_REQUEST_EXCEEDS_TOKEN_BUDGET,
    _coverage_statement,
    _focused_reread_preflight,
    _prepare_node,
    _reread_estimate_for,
    _status_fields,
)
from automated_grounded_theory.gate_evaluation import relationship_assessment, stamp_relationship_assessment
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES
from tests.test_v080 import FakeAxialCoder
from tests.test_v0926 import SENTENCE, _book, _both

STATEMENT = "tools for the shift may be related to coworkers on the crew"


def test_version_fingerprint_and_unchanged_defaults():
    assert __version__ == "0.9.35"
    assert POLICY_VERSION == "0.9.35"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v6"
    assert DECISION_TOKEN_SCOPE == "per_step_v1"
    assert CASE_COUNT_RULE == "respondent_identity_v1"
    assert REREAD_ESTIMATE_RULE == "post_selection_v1"
    assert PATTERN_MIN_CASES == 2
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000
    assert "0.9.28" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.32" in CHUNKING_CHECKPOINT_MESSAGE
    current = review_procedure_fields()
    assert same_chunking_fingerprint(current, current) is True
    older = dict(current)
    older.pop("decision_token_scope")
    older.pop("case_count_rule")
    older.pop("reread_estimate_rule")
    assert same_chunking_fingerprint(older, current) is False


def _reviewer(ceiling: int) -> tuple[AxialDecisionReviewer, ProviderAllowance]:
    reviewer = AxialDecisionReviewer(FakeAxialCoder(), max_calls=4, max_tokens=ceiling, token_budget=10**9)
    allowance = ProviderAllowance(
        max_tokens=ceiling,
        stages={
            "axial_relation_review": 4,
            "axial_saturation_review": 4,
            "axial_core_review": 4,
        },
        stage_token_ceilings={
            "axial_relation_review": ceiling,
            "axial_saturation_review": ceiling,
            "axial_core_review": ceiling,
        },
    )
    reviewer.limiter.provider_allowance = allowance
    reviewer.max_tokens = ceiling
    return reviewer, allowance


def test_each_decision_step_has_its_own_token_allowance():
    reviewer, allowance = _reviewer(24000)
    blank = {"decision_calls_used": 0}
    relation = _relation_review_payload(
        [{"relationship_id": "rel-0001", "from_id": "tools", "to_id": "crew", "statement": STATEMENT, "quote": SENTENCE}],
        chunk_index=0,
        chunk_count=3,
    )
    core = _core_review_payload(
        [{"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"}],
        preferred=[],
        snapshot={},
        chunk_index=0,
        chunk_count=1,
    )
    relation_estimate = reviewer.refusal_reason(relation, AxialRelationDecisions, blank)[1]
    tight, tight_allowance = _reviewer(relation_estimate)
    metadata: dict = {"decision_calls_used": 0}
    parsed, audit = run_sync(tight.decide(relation, AxialRelationDecisions, metadata))
    core_stop, _estimate = tight.refusal_reason(core, ModelCoreDecision, metadata)
    assert parsed is not None and audit["status"] == "validated"
    assert tight_allowance.stage_used["axial_relation_review"] == 1
    assert tight_allowance.stage_tokens_used["axial_relation_review"] == relation_estimate
    assert tight_allowance.tokens_estimated == 0
    assert tight_allowance.stage_used["axial_core_review"] == 0
    assert tight_allowance.stage_tokens_used["axial_core_review"] == 0
    assert core_stop is None
    assert tight_allowance.remaining_tokens("axial_core_review") == relation_estimate


def test_selective_core_does_not_inherit_the_axial_token_pool():
    book = {
        f"c{index:02d}": {"code_id": f"c{index:02d}", "code": "packing the bag", "definition": "packing the bag"}
        for index in range(4)
    }
    source = {
        "codebook": book,
        "passages": [{"passage_id": "p1", "doc_id": "d1", "text": "A short note about packing the bag."}],
        "segments": [{"segment_id": "s1", "code_id": "c00", "passage_id": "p1", "quote": "packing the bag"}],
        "relationships": [],
        "properties": [],
        "dimensions": [],
        "metadata": {
            "stage": "axial",
            "axial_tokens_estimated": 24000,
            "provider_budget": {
                "calls_used": 2,
                "tokens_estimated": 24000,
                "max_tokens": 24000,
                "stages": {
                    "axial_relation_review": {"calls_used": 1, "max_calls": 4, "tokens_used": 24000, "max_tokens": 24000},
                    "axial_saturation_review": {"calls_used": 1, "max_calls": 4, "tokens_used": 0, "max_tokens": 24000},
                },
            },
        },
    }
    finished = selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=4,
        max_axial_review_tokens=24000,
        batch_size=1,
    )
    report = finished.metadata["decision_reviews"]["axial_core_review"]
    step = next(item for item in finished.metadata["decision_budget_steps"] if item["step"] == "axial_core_review")
    assert report["chunks_sent"] >= 1
    assert finished.metadata.get("decision_stop") != "token_allowance"
    assert step["token_allowance"] == 24000
    assert step["starting_tokens"] == 0
    assert step["tokens_required"] is not None
    assert step["tokens_available"] == 24000
    assert step["tokens_required"] <= step["tokens_available"]
    assert step["prior_step_tokens_reduce_allowance"] is False
    assert step["tokens_used"] >= 1
    assert step["remaining_tokens"] == 24000 - step["tokens_used"]
    public = budget_ledger(finished.metadata)
    assert public[-1]["token_allowance"] == 24000
    assert public[-1]["starting_tokens"] == 0


def _passages(person: str, *, field: str = "respondent", where: str = "metadata", people: list[str] | None = None) -> dict:
    names = people or [person, person]
    rows = {}
    for index, name in enumerate(names, start=1):
        passage = {
            "passage_id": f"p{index}",
            "doc_id": f"question-{index}",
            "text": f"The shift notes say {SENTENCE}",
        }
        if where == "top":
            passage[field] = name
        else:
            passage["metadata"] = {field: name}
        rows[f"p{index}"] = passage
    return rows


def _merged(passages: dict, order: list[tuple[str, str]] | None = None) -> dict:
    pairs = order or [("rel-0005", "p1"), ("rel-0006", "p1"), ("rel-0007", "p2")]
    rows = [
        {
            "relationship_id": relationship_id,
            "from_id": "tools",
            "to_id": "crew",
            "passage_id": passage_id,
            "quote": SENTENCE,
            "statement": STATEMENT,
            "disposition": "proposed",
            "evidence_tier": 1,
            "case_quotes": [_both(passage_id, SENTENCE)],
        }
        for relationship_id, passage_id in pairs
    ]
    committed = _commit_relations(
        {
            "metadata": {"procedure": {"min_relation_documents": 2, "coding_lens": "paradigm", "pattern_min_cases": 2}},
            "relationships": rows,
            "passages": list(passages.values()),
            "codebook": _book(),
            "segments": [],
            "memos": [],
            "warnings": [],
        }
    )["relationships"]
    return committed[0]


def test_one_respondent_many_quotes_stay_below_tier_2():
    for passages in (
        _passages("alice"),
        _passages("alice", where="top"),
        _passages("alice", field="participant"),
    ):
        merged = _merged(passages)
        assert merged["evidence_tier"] == 1
        assert merged["cases_with_both_endpoints"] == 1
        assert merged["supporting_cases"][0]["case_id"] == "alice"


def test_two_respondents_reach_tier_2_and_id_order_does_not_matter():
    passages = _passages("alice", field="respondent_id", people=["alice", "bob"])
    forward = _merged(passages)
    backward = _merged(
        passages,
        [("rel-0007", "p2"), ("rel-0006", "p1"), ("rel-0005", "p1")],
    )
    assert forward["evidence_tier"] == 2
    assert forward["cases_with_both_endpoints"] == 2
    assert backward["evidence_tier"] == forward["evidence_tier"]
    assert backward["cases_with_both_endpoints"] == forward["cases_with_both_endpoints"]
    assert [case["case_id"] for case in backward["supporting_cases"]] == [
        case["case_id"] for case in forward["supporting_cases"]
    ]
    assert {case["case_id"] for case in forward["supporting_cases"]} == {"alice", "bob"}


def test_reread_is_estimated_after_selection_and_a_shortfall_does_not_block_it():
    """The worst case plans a queue as if every code were raised.

    After selection the estimate uses the raised code and that shorter queue.
    Both figures use the same request builder. The develop share is the whole
    allowance here, so the difference is the number of assigned passages.
    """
    codes = [
        {"code_id": f"c{index:03d}", "code": f"code {index}", "definition": "packing tools " * 40}
        for index in range(8)
    ]
    passages = [
        {
            "passage_id": f"d{index}-p0",
            "doc_id": f"d{index}",
            "text": "tools for the shift sit beside the crew. " * 40,
        }
        for index in range(8)
    ]
    segments = [
        {
            "segment_id": f"s{index}",
            "code_id": f"c{index:03d}",
            "passage_id": f"d{index}-p0",
            "quote": "tools for the shift",
        }
        for index in range(8)
    ]
    state = {
        "codebook": {row["code_id"]: row for row in codes},
        "passages": passages,
        "segments": segments,
        "raised_code_ids": ["c000"],
        "properties": [],
        "dimensions": [],
        "memos": [],
        "metadata": {
            "procedure": {
                "max_focused_review_calls": 8,
                "max_focused_review_tokens": 24000,
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
        },
    }
    worst = _focused_reread_preflight(state)
    post = _reread_estimate_for(state, ["c000"])
    assert worst is not None and post is not None
    assert worst["blocked"] is False
    assert worst["worst_case"] is True
    assert worst["worst_case_assumes_all_codes_raised"] is True
    assert "does not block" in worst["note"]
    assert post["raised_codes"] == 1
    assert post["queue_items"] == 1
    assert post["queue_items"] < 8
    assert worst["focused_review_tokens_required_worst_case"] > post["focused_review_tokens_required"]


def test_a_single_reread_request_cannot_exceed_the_per_request_budget():
    huge = {"passage_id": "huge", "doc_id": "d0", "text": "Packing the suitcase before the night shift. " * 800}
    prepared = _prepare_node(
        {
            "queue_ready": True,
            "recode_queue": [{"passage_id": "huge", "purpose": "develop"}],
            "recode_cursor": 0,
            "passages": [huge],
            "codebook": {"c000": {"code_id": "c000", "code": "packing", "definition": "packing"}},
            "raised_code_ids": ["c000"],
            "segments": [],
            "properties": [],
            "dimensions": [],
            "memos": [],
            "focused_calls_used": 0,
            "focused_tokens_estimated": 0,
            "metadata": {
                "procedure": {"max_focused_review_calls": 4, "max_focused_review_tokens": 500000, "validation_retries": 0},
                "performance": {"batch_size": 1, "token_budget": 6000},
            },
        }
    )
    assert prepared.get("round_jobs") in (None, [])
    assert prepared["recode_stop"] == REREAD_REQUEST_EXCEEDS_TOKEN_BUDGET
    recorded = prepared["metadata"]["reread_requests_over_budget"]
    assert recorded[0]["token_budget"] == 6000
    assert recorded[0]["estimate"] > 6000
    assert "6000" in prepared["metadata"]["reread_request_bound"]


def test_complete_for_queue_reports_passage_and_document_coverage():
    passages = [
        {"passage_id": f"d{doc}-p{pos}", "doc_id": f"d{doc}", "text": "a note"}
        for doc in range(6)
        for pos in range(4)
    ]
    reread = [f"d0-p{pos}" for pos in range(4)] + ["d1-p0"]
    state = {
        "passages": passages,
        "recode_queue": [{"passage_id": item, "purpose": "develop"} for item in reread],
        "focused_reread": list(reread),
        "focused_not_reread": [],
        "recode_stop": "complete",
        "metadata": {"procedure": {}},
    }
    coverage = _coverage_statement(state, [])
    status = _status_fields(state)
    assert status["status"] == "complete_for_queue"
    assert coverage["complete_for_queue_is_corpus_coverage"] is False
    assert coverage["passages_covered"] == 5
    assert coverage["passages_in_corpus"] == 24
    assert coverage["documents_covered"] == 2
    assert coverage["documents_in_corpus"] == 6
    assert "not corpus coverage" in coverage["note"]


def test_relationship_assessment_is_exported_under_that_name():
    row = {"relationship_id": "rel-0005", "evidence_tier": 1, "disposition": "proposed", "from_id": "a", "to_id": "b"}
    assessment = stamp_relationship_assessment(row)
    assert row["relationship_assessment"] == assessment
    assert row["evidence_tier"] == assessment["evidence_tier"]
    exported = export_gate_relationships([row])
    listed = exported["proposed"]
    assert listed[0]["relationship_assessment"] == relationship_assessment(row)
    assert listed[0]["evidence_tier"] == listed[0]["relationship_assessment"]["evidence_tier"]
