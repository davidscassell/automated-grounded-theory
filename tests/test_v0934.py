"""0.9.34 review fixes. Fakes and synthetic fixtures only. No network calls.

These checks are processing coverage. They do not establish selection validity.
"""

from __future__ import annotations

from automated_grounded_theory import __version__
from automated_grounded_theory.axial import (
    DEFAULT_MAX_AXIAL_REVIEW_TOKENS,
    AxialDecisionReviewer,
    ModelCoreDecision,
    _axial_token_plan,
    _core_match_upper_bound,
    _score_candidates,
    graph_snapshot,
    selective_code,
)
from automated_grounded_theory.decision_review import (
    CHUNKING_CHECKPOINT_MESSAGE,
    CORE_CALL_ALLOWANCE_INSUFFICIENT,
    CORE_COMPARISON_RULE,
    CORE_PAIR_EXCEEDS_REQUEST_BUDGET,
    DECISION_REVIEW_CHUNKING,
    CONDITION_COMPLETENESS_RULE,
    ELIGIBILITY_DECISION_RULE,
    REREAD_BOUND_RULE,
    REREAD_COVERAGE_RULE,
    order_core_candidates,
    partition_core_matches,
    plan_core_comparison,
    review_procedure_fields,
    same_chunking_fingerprint,
)
from automated_grounded_theory.focused import (
    REREAD_REQUEST_EXCEEDS_TOKEN_BUDGET,
    _focused_reread_preflight,
    _prepare_node,
    _reread_upper_bound,
    _status_fields,
)
from automated_grounded_theory.gate_evaluation import relationship_assessment, withholding_reasons
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import (
    PATTERN_MIN_CASES,
    apply_evidence_tier,
    condition_span_is_incomplete,
)
from tests.test_v080 import FakeAxialCoder
from tests.test_v0924 import _KeepSelector, _code, _run_focused
from tests.test_v0926 import _book, _passages
from tests.test_v0931 import _LINK, _tier
from tests.test_v0933 import _packed_state


def test_version_fingerprint_and_unchanged_defaults():
    assert __version__ == "0.9.39"
    assert POLICY_VERSION == "0.9.39"
    assert PATTERN_MIN_CASES == 2
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000
    assert DECISION_REVIEW_CHUNKING == "request_budget_v6"
    assert CORE_COMPARISON_RULE == "deterministic_tournament_v1"
    assert CONDITION_COMPLETENESS_RULE == "full_clause_or_incomplete_v1"
    assert ELIGIBILITY_DECISION_RULE == "observed_conditional_eligible_v1"
    assert REREAD_COVERAGE_RULE == "withheld_not_complete_v1"
    assert REREAD_BOUND_RULE == "cap_times_reserved_calls_v1"
    assert "0.9.28" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.33" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.34" in CHUNKING_CHECKPOINT_MESSAGE
    current = review_procedure_fields()
    assert same_chunking_fingerprint(current, current) is True
    older = dict(current)
    older.pop("condition_completeness_rule")
    assert same_chunking_fingerprint(older, current) is False


def test_core_candidates_are_ordered_into_matches_of_at_least_two():
    assert order_core_candidates(["c2", "c0", "c0", "c1"]) == ["c0", "c1", "c2"]
    fitted = plan_core_comparison(["c2", "c0", "c1"], lambda ids: 100 * len(list(ids)), per_request=6000)
    assert fitted["whole_set"] is True
    assert fitted["chunks"] == [["c0", "c1", "c2"]]
    assert fitted["compared"] == 3
    assert fitted["eligible"] == 3
    alone = plan_core_comparison(["only"], lambda ids: 100, per_request=6000)
    assert alone["chunks"] == [["only"]]
    assert alone["whole_set"] is True
    tournament = partition_core_matches(
        ["c0", "c1", "c2"],
        lambda ids: 2500 * len(list(ids)),
        per_request=6000,
    )
    assert tournament["reason"] is None
    assert tournament["whole_set"] is False
    assert tournament["matches"] == [["c0", "c1"]]
    assert tournament["byes"] == ["c2"]
    assert all(len(match) >= 2 for match in tournament["matches"])
    blocked = plan_core_comparison(["c0", "c1", "c2"], lambda ids: 5000 * len(list(ids)), per_request=6000)
    assert blocked["chunks"] == []
    assert blocked["byes"] == []
    assert blocked["compared"] == 0
    assert blocked["eligible"] == 3
    assert blocked["reason"] == CORE_PAIR_EXCEEDS_REQUEST_BUDGET


def _core_source(width_label: str, count: int) -> dict:
    book = {
        f"c{index:02d}": {"code_id": f"c{index:02d}", "code": width_label, "definition": width_label}
        for index in range(count)
    }
    return {
        "codebook": book,
        "passages": [{"passage_id": "p1", "doc_id": "d1", "text": "A short note about packing."}],
        "segments": [],
        "relationships": [],
        "metadata": {"stage": "axial"},
    }


def _payload_estimate(reviewer, candidates, ids, *, whole_set: bool, eligible: int) -> int:
    """Token count of the same final-shaped upper bound the planner uses.

    ``whole_set`` is kept so callers can still name a complete eligible set.
    The bound always reserves one full prior rationale, matching
    ``_core_match_upper_bound``.
    """
    by_id = {row["code_id"]: row for row in candidates}
    subset = [by_id[item] for item in ids if item in by_id]
    known = [row["code_id"] for row in candidates]
    if whole_set:
        eligible_ids = [str(item) for item in ids]
    elif len(known) == int(eligible):
        eligible_ids = known
    else:
        eligible_ids = [str(item) for item in ids] + [
            f"extra-{index}" for index in range(max(0, int(eligible) - len(ids)))
        ]
    payload = _core_match_upper_bound(
        subset,
        list(ids),
        preferred=[],
        snapshot=graph_snapshot([]),
        eligible_ids=eligible_ids,
    )
    return reviewer.refusal_reason(payload, ModelCoreDecision, {"decision_calls_used": 0})[1]


def test_a_core_tournament_compares_finalists_and_an_unfinished_bracket_picks_nobody():
    label = "packing " * 40
    source = _core_source(label, 3)
    candidates = _score_candidates(source)
    assert [row["code_id"] for row in candidates] == ["c00", "c01", "c02"]
    probe = AxialDecisionReviewer(FakeAxialCoder(), max_calls=8, max_tokens=10**6, token_budget=10**6)
    pair = _payload_estimate(probe, candidates, ["c00", "c01"], whole_set=False, eligible=3)
    triple = _payload_estimate(probe, candidates, ["c00", "c01", "c02"], whole_set=True, eligible=3)
    assert triple > pair
    budget = pair
    assert _payload_estimate(
        AxialDecisionReviewer(FakeAxialCoder(), max_calls=8, max_tokens=10**6, token_budget=budget),
        candidates,
        ["c00", "c01"],
        whole_set=False,
        eligible=3,
    ) <= budget
    finished = selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=8,
        max_axial_review_tokens=10**6,
        axial_token_budget=budget,
        batch_size=1,
    )
    report = finished.metadata["decision_reviews"]["axial_core_review"]
    assert report["complete"] is True
    assert report["chunks_sent"] == 2
    assert finished.metadata["model_core_id"] == "c00"
    assert finished.metadata["core_selection_withheld"] is False
    assert finished.metadata["core_candidates_eligible"] == 3
    bracket = finished.metadata["core_bracket"]
    assert [len(match) for match in bracket["rounds"][0]["matches"]] == [2]
    assert bracket["rounds"][0]["byes"] == ["c02"]
    assert bracket["rounds"][0]["compared"] == 2
    assert bracket["rounds"][1]["matches"] == [["c00", "c02"]]
    assert bracket["rounds"][1]["compared"] == 2
    assert finished.metadata["core_candidates_compared_by_round"] == [2, 2]
    assert finished.metadata["core_candidates_compared"] == 4
    assert all(len(match) >= 2 for round_row in bracket["rounds"] for match in round_row["matches"])

    stopped = selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=1,
        max_axial_review_tokens=10**6,
        axial_token_budget=budget,
        batch_size=1,
    )
    assert stopped.metadata["decision_stop"] == CORE_CALL_ALLOWANCE_INSUFFICIENT
    assert stopped.metadata["core_outcome"] == "selection_unresolved"
    assert stopped.metadata["model_core_id"] == ""
    assert stopped.metadata["provisional_core_id"] == ""
    assert stopped.metadata["core_status"] == "withheld"
    assert stopped.metadata["core_selection_withheld"] is True
    assert stopped.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 1
    assert stopped.metadata["core_candidates_eligible"] == 3
    assert stopped.metadata["core_candidates_compared_by_round"][0] == 2
    assert stopped.metadata["core_candidates_compared"] == 2

    one = _payload_estimate(probe, candidates[:1], ["c00"], whole_set=True, eligible=1)
    refused = selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=8,
        max_axial_review_tokens=10**6,
        axial_token_budget=max(1, one - 1),
        batch_size=1,
    )
    assert refused.metadata["decision_stop"] == CORE_PAIR_EXCEEDS_REQUEST_BUDGET
    assert refused.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 0
    assert refused.metadata["model_core_id"] == ""
    assert refused.metadata["core_candidates_compared"] == 0
    assert refused.metadata["core_candidates_eligible"] == 3


def test_a_condition_span_keeps_the_full_clause_and_a_bare_pronoun_is_incomplete():
    assert condition_span_is_incomplete("if you") is True
    assert condition_span_is_incomplete("unless you") is True
    assert condition_span_is_incomplete("provided that you") is True
    assert condition_span_is_incomplete("if you help") is False
    assert condition_span_is_incomplete("if it rains") is False
    quote = "if you, violated the data sharing agreement, tools for the shift lead to coworkers on the crew"
    full = _tier(quote)
    assert full["condition"] == "if you, violated the data sharing agreement"
    assert full["condition"] != "if you"
    assert full["condition_incomplete"] is False
    assert full["statement_omits_condition"] is True
    assert "if you, violated the data sharing agreement" in full["presented_statement"]
    assert full["flags"].count("condition_not_in_statement") == 1
    passages = _passages(quote, ["r1", "r2"])
    apply_evidence_tier(full, passages=passages, codebook=_book(), segments=[], memos=[])
    apply_evidence_tier(full, passages=passages, codebook=_book(), segments=[], memos=[])
    assert full["flags"].count("condition_not_in_statement") == 1

    bare = _tier(f"{_LINK} if you")
    assert bare["condition"] == "if you"
    assert bare["condition_incomplete"] is True
    assert bare["condition_retained"] is None
    assert bare["statement_omits_condition"] is None
    assert bare["presented_statement"] == _LINK
    assert "(if you)" not in bare["presented_statement"]
    assert "condition_incomplete" in bare["flags"]
    assert bare["flags"].count("condition_incomplete") == 1
    assert bare["downstream_eligible"] is False
    assert bare["claims"]["pattern"]["supports_core"] is False


def test_unreviewed_hypothetical_and_incomplete_rows_are_ineligible_with_a_cause():
    associated = "tools for the shift are related to coworkers on the crew"
    bare = _tier(f"{associated} if you", statement=associated)
    bare["reviewer_decided"] = False
    bare["reviewer_disposition"] = None
    quote = f"{associated} if you"
    apply_evidence_tier(bare, passages=_passages(quote, ["r1", "r2"]), codebook=_book(), segments=[], memos=[])
    assert bare["evidence_tier"] == 2
    assert bare["downstream_eligible"] is False
    assert bare["eligibility_blocks"] == [
        "incomplete_condition",
        "hypothetical_link",
        "no_review_decision",
    ]
    reasons = withholding_reasons(bare)
    assert "incomplete_condition" in reasons
    assert "hypothetical_link" in reasons
    assert "no_review_decision" in reasons
    assessment = relationship_assessment(bare)
    assert assessment["downstream_eligible"] is False
    assert "incomplete_condition" in assessment["withholding_reasons"]
    assert "hypothetical_link" in assessment["withholding_reasons"]
    assert "no_review_decision" in assessment["withholding_reasons"]
    assert assessment["primary_withholding_cause"] == "incomplete_condition"

    conditional = _tier(f"{_LINK} if it rains")
    assert conditional["condition_incomplete"] is False
    assert conditional["condition_pattern"] == "observed_conditional"
    assert conditional["downstream_eligible"] is False
    assert "hypothetical_link" not in conditional["eligibility_blocks"]
    assert "incomplete_condition" not in conditional["eligibility_blocks"]
    assert "wording_withheld" in withholding_reasons(conditional)
    assert "hypothetical_link" not in withholding_reasons(conditional)

    plain = _tier(_LINK)
    assert plain["downstream_eligible"] is True
    assert plain["eligibility_blocks"] == []
    assert withholding_reasons(plain) == []
    plain["reviewer_decided"] = False
    apply_evidence_tier(plain, passages=_passages(_LINK, ["r1", "r2"]), codebook=_book(), segments=[], memos=[])
    assert plain["downstream_eligible"] is False
    assert plain["eligibility_blocks"] == ["no_review_decision"]
    assert "no_review_decision" in withholding_reasons(plain)


def test_withheld_rereads_stay_in_coverage_and_the_queue_is_not_complete():
    state = _packed_state()
    state["metadata"]["procedure"]["focused_claim_compete_round"] = False
    state["metadata"]["reread_call_targets"] = {"claim_compete": 0}
    prepared = _prepare_node(state)
    assert prepared["round_jobs"][0]["passage_ids"] == ["short"]
    assert "huge" in prepared["focused_not_reread"]
    assert prepared["metadata"]["reread_requests_over_budget"][0]["passage_id"] == "huge"

    only = _packed_state()
    only["metadata"]["procedure"]["focused_claim_compete_round"] = False
    only["metadata"]["reread_call_targets"] = {"claim_compete": 0}
    only["recode_queue"] = [{"passage_id": "huge", "purpose": "develop"}]
    held = _prepare_node(only)
    assert held.get("round_jobs") in (None, [])
    assert held["focused_not_reread"] == ["huge"]
    assert held["recode_stop"] == REREAD_REQUEST_EXCEEDS_TOKEN_BUDGET
    finished = {
        **only,
        **held,
        "metadata": held["metadata"],
        "focused_reread": [],
        "focused_not_reread": [],
        "recode_stop": "complete",
    }
    fields = _status_fields(finished)
    assert "huge" in fields["focused_not_reread"]
    assert fields["status"] == "incomplete"
    assert fields["complete"] is False
    assert fields["stop"] == REREAD_REQUEST_EXCEEDS_TOKEN_BUDGET
    assert fields["coverage"]["queue_complete"] is False
    assert fields["coverage"]["withheld_over_budget"] == ["huge"]
    assert "huge" in fields["coverage"]["not_reread"]


def test_preflight_records_a_reread_upper_bound_and_the_pool_warning_is_gone():
    state = _packed_state()
    preflight = _focused_reread_preflight(state)
    assert preflight is not None
    assert preflight["focused_review_tokens_upper_bound"] == 6000 * 8
    assert preflight["focused_review_upper_bound_calls"] == 8
    assert preflight["focused_review_upper_bound_label"] == "per-request cap times reserved reread calls"
    assert preflight["focused_review_upper_bound_is_a_bound"] is True
    assert preflight["blocked"] is False
    bound = _reread_upper_bound(state)
    assert bound["tokens"] == bound["cap"] * bound["calls"]
    assert bound["label"] == "per-request cap times reserved reread calls"

    quiet = _packed_state()
    quiet["metadata"]["procedure"]["max_focused_review_calls"] = 0
    assert _focused_reread_preflight(quiet) is None
    zero = _reread_upper_bound(quiet)
    assert zero["calls"] == 0
    assert zero["tokens"] == 0
    assert zero["label"] == "per-request cap times reserved reread calls"
    ran = _run_focused([_code("c-a", "KEEP this code")], _KeepSelector())
    assert ran.metadata["focused_review_tokens_required_worst_case"] is None
    assert ran.metadata["focused_review_tokens_upper_bound"] == 0
    assert ran.metadata["focused_review_upper_bound_calls"] == 0
    assert ran.metadata["focused_review_upper_bound_label"] == "per-request cap times reserved reread calls"
    assert ran.metadata["focused_review_upper_bound_is_a_bound"] is True

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
    assert plan["estimate"] == 48000
    assert plan["ceiling"] == 24000
    assert plan["shortfall"] == 24000
    assert plan["warning"] == ""
    assert "48000" not in plan["warning"]
    assert "exceeds ceiling" not in plan["note"]
    assert "not a warning" in plan["coding_pool_bound_label"]
