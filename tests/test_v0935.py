"""0.9.35 review fixes. Fakes and synthetic fixtures only. No network calls.

These checks are processing coverage. They do not establish selection validity.
The conditional fixtures are synthetic. They are not adjudications.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from automated_grounded_theory import __version__
from automated_grounded_theory.axial import (
    DEFAULT_MAX_AXIAL_REVIEW_TOKENS,
    AxialDecisionReviewer,
    ModelCoreDecision,
    _core_match_instruction,
    _core_review_payload,
    _score_candidates,
    graph_snapshot,
    selective_code,
)
from automated_grounded_theory.batching import estimate_tokens
from automated_grounded_theory.decision_review import (
    CHUNKING_CHECKPOINT_MESSAGE,
    CORE_EVIDENCE_SUMMARY_RULE,
    CORE_FINAL_DECISION_RULE,
    CORE_PROMPT_SCOPE_RULE,
    CORE_SCHEDULE_NOTE,
    CONDITION_PATTERN_RULE,
    ELIGIBILITY_DECISION_RULE,
    bracket_order_report,
    order_core_candidates,
    plan_core_comparison,
    play_core_bracket,
    review_procedure_fields,
    same_chunking_fingerprint,
)
from automated_grounded_theory.gate_evaluation import withholding_reasons
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import (
    PATTERN_MIN_CASES,
    apply_evidence_tier,
    classify_condition_pattern,
)
from tests.test_v080 import FakeAxialCoder
from tests.test_v0926 import _book, _both, _passages
from tests.test_v0931 import _LINK
from tests.test_v0934 import _core_source, _payload_estimate

_ASSOCIATED = "tools for the shift are related to coworkers on the crew"
_PRIOR = "It is one candidate. Alternatives stay in the list."


class ScriptedCore(FakeAxialCoder):
    """One core response per match. Other schemas stay on the parent fake."""

    core_steps: list = Field(default_factory=list)
    core_seen: int = 0

    def with_structured_output(self, schema, **kwargs):
        name = getattr(schema, "__name__", "")
        if name != "ModelCoreDecision":
            return FakeAxialCoder.with_structured_output(self, schema, **kwargs)
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            step = self.core_steps[self.core_seen] if self.core_seen < len(self.core_steps) else "c00"
            self.core_seen += 1
            if step == "parse_failure":
                parsed = None
                raw = AIMessage(content="not json")
                if not include_raw:
                    raise ValueError("unreadable core decision")
                return {"raw": raw, "parsed": None, "parsing_error": "unreadable"}
            if isinstance(step, tuple) and step and step[0] == "abstain":
                parsed = schema(
                    decision="abstain",
                    core_id="",
                    abstention_reason=str(step[1]),
                    rationale="No defensible core",
                )
            elif step == "abstain":
                parsed = schema(
                    decision="abstain",
                    core_id="",
                    abstention_reason="no_suitable_candidate",
                    rationale="No defensible core",
                )
            elif step == "empty_select":
                parsed = schema(core_id="", rationale="No defensible core")
            elif step == "invalid":
                parsed = schema(core_id="not-a-candidate", rationale="Not in this match.")
            else:
                parsed = schema(core_id=str(step), rationale=f"Select {step}.")
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def _tournament_budget() -> int:
    source = _core_source("packing " * 40, 3)
    candidates = _score_candidates(source)
    probe = AxialDecisionReviewer(FakeAxialCoder(), max_calls=8, max_tokens=10**6, token_budget=10**6)
    return _payload_estimate(probe, candidates, ["c00", "c01"], whole_set=False, eligible=3)


def _run_core(steps: list, **kwargs):
    source = _core_source("packing " * 40, 3)
    budget = kwargs.pop("axial_token_budget", _tournament_budget())
    calls = kwargs.pop("max_decision_calls", 8)
    return selective_code(
        source,
        ScriptedCore(core_steps=list(steps)),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=calls,
        max_axial_review_tokens=10**6,
        axial_token_budget=budget,
        batch_size=1,
        **kwargs,
    )


def _core_bodies(result) -> list[dict]:
    found = []
    for row in result.prompt_log:
        if str(row.get("checkpoint") or "") != "axial_core_review":
            continue
        if str(row.get("status") or "") in {"omitted", "refused", "not_sent"}:
            continue
        for message in row.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else ""
            if isinstance(content, str) and content.startswith("{"):
                found.append(json.loads(content))
    return found


def _conditional(quote: str, statement: str, *, reviewed: bool = True) -> dict:
    """Synthetic fixture. Not an adjudication."""
    row = {
        "relationship_id": "rel-synthetic-condition",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": quote,
        "statement": statement,
        "disposition": "proposed",
        "fixture": "synthetic",
        "case_quotes": [_both("p1", quote), _both("p2", quote)],
        "counterexamples": [
            {
                "passage_id": "p9",
                "quote": "the crew met without tools for the shift",
                "judgment": "contradicts",
                "valid_test": True,
            }
        ],
    }
    if reviewed:
        row["reviewer_decided"] = True
        row["reviewer_disposition"] = "accepted"
    apply_evidence_tier(
        row,
        passages=_passages(quote, ["r1", "r2"]),
        codebook=_book(),
        segments=[],
        memos=[],
    )
    return row


def _cycle(match: list[str]) -> str:
    beats = {("c00", "c01"): "c00", ("c01", "c02"): "c01", ("c02", "c00"): "c02"}
    pair = (str(match[0]), str(match[1]))
    if pair in beats:
        return beats[pair]
    return beats[(pair[1], pair[0])]


def test_version_fingerprint_and_unchanged_defaults():
    assert __version__ == "0.9.42"
    assert POLICY_VERSION == "0.9.42"
    assert PATTERN_MIN_CASES == 2
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000
    assert CORE_FINAL_DECISION_RULE == "final_assessment_only_v1"
    assert CORE_PROMPT_SCOPE_RULE == "eligible_match_finalists_no_abstain_cue_v1"
    assert CONDITION_PATTERN_RULE == "observed_conditional_eligible_v1"
    assert ELIGIBILITY_DECISION_RULE == "observed_conditional_eligible_v1"
    assert CORE_EVIDENCE_SUMMARY_RULE == "match_budget_candidate_evidence_v1"
    assert "reproducible scheduling" in CORE_SCHEDULE_NOTE
    assert "not evidence of substantive stability" in CORE_SCHEDULE_NOTE
    assert "0.9.28" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.33" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.34" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.35" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.36" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.37" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.38" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.39" in CHUNKING_CHECKPOINT_MESSAGE
    current = review_procedure_fields()
    assert current["core_final_decision_rule"] == CORE_FINAL_DECISION_RULE
    assert current["core_prompt_scope_rule"] == CORE_PROMPT_SCOPE_RULE
    assert current["condition_pattern_rule"] == CONDITION_PATTERN_RULE
    assert current["core_evidence_summary_rule"] == CORE_EVIDENCE_SUMMARY_RULE
    assert same_chunking_fingerprint(current, current) is True
    older = dict(current)
    older.pop("core_final_decision_rule")
    assert same_chunking_fingerprint(older, current) is False
    changed = dict(current)
    changed["eligibility_decision_rule"] = "reviewed_complete_condition_v1"
    assert same_chunking_fingerprint(changed, current) is False


def test_an_empty_final_core_does_not_keep_the_earlier_nomination():
    finished = _run_core(["c00", "empty_select"])
    assert finished.metadata["model_core_id"] == ""
    assert finished.metadata["core_status"] != "applied"
    assert finished.metadata["core_status"] != "abstained"
    assert finished.metadata.get("model_core_unknown") is not True
    assert finished.metadata["decision_stop"] == "review_error"
    assert all(row.get("model_core") is not True for row in finished.core_candidates)


def test_an_explicit_abstention_ends_with_no_core_in_either_round():
    final = _run_core(["c00", "abstain"])
    assert final.metadata["model_core_id"] == ""
    assert final.metadata["core_status"] == "no_core_supported"
    assert final.metadata["core_outcome"] == "no_core_supported"
    assert final.metadata["model_core_decision"] == "abstain"
    assert final.metadata["model_core_rationale"] == "No defensible core"
    assert final.metadata["core_selection_withheld"] is False
    assert final.metadata["core_selection_complete"] is True
    assert final.metadata["core_downstream_eligible"] is False
    assert final.metadata.get("model_core_unknown") is not True
    assert "Chosen by the final match" not in str(final.metadata.get("model_core_rationale") or "")
    assert all(not row.get("model_core") for row in final.core_candidates)

    opening = _run_core(["abstain", "abstain"])
    assert opening.metadata["model_core_id"] == ""
    assert opening.metadata["core_status"] == "no_core_supported"
    assert opening.metadata["core_outcome"] == "no_core_supported"
    assert opening.metadata["model_core_decision"] == "abstain"
    assert opening.metadata["model_core_rationale"] == "No defensible core"
    assert opening.metadata["core_selection_complete"] is True
    assert opening.metadata["core_candidates_compared"] == 3
    assert opening.metadata["core_downstream_eligible"] is False


def test_invalid_ids_and_parse_failures_set_no_core_in_either_round():
    opening = _run_core(["invalid"])
    assert opening.metadata["model_core_id"] == ""
    assert opening.metadata["decision_stop"] == "invalid_decision"
    assert opening.metadata["core_status"] == "withheld"
    assert opening.metadata.get("model_core_unknown") is not True

    final = _run_core(["c00", "invalid"])
    assert final.metadata["model_core_id"] == ""
    assert final.metadata["decision_stop"] == "invalid_decision"
    assert final.metadata["core_status"] == "withheld"
    assert final.metadata["core_status"] != "applied"
    assert final.metadata.get("model_core_unknown") is not True
    assert not any("was kept and flagged" in note for note in final.warnings)

    bad_open = _run_core(["parse_failure"])
    assert bad_open.metadata["model_core_id"] == ""
    assert bad_open.metadata["decision_stop"] == "review_error"
    assert bad_open.metadata["core_status"] != "applied"

    bad_final = _run_core(["c00", "parse_failure"])
    assert bad_final.metadata["model_core_id"] == ""
    assert bad_final.metadata["decision_stop"] == "review_error"
    assert bad_final.metadata["core_status"] != "applied"
    assert bad_final.metadata["core_status"] != "abstained"


def test_a_valid_final_selection_still_sets_that_core():
    finished = _run_core(["c00", "c02"])
    assert finished.metadata["model_core_id"] == "c02"
    assert finished.metadata["model_core_decision"] == "select"
    assert finished.metadata["core_status"] == "applied"
    assert finished.metadata["model_core_rationale"] == "Select c02."
    assert finished.metadata["core_selection_complete"] is True
    assert "Chosen by the final match" not in finished.metadata["model_core_rationale"]


def test_the_final_prompt_names_the_eligible_set_the_round_and_the_finalists():
    finished = _run_core(["c00", "c02"])
    bodies = _core_bodies(finished)
    assert len(bodies) == 2
    opening = bodies[0]
    final = bodies[1]
    assert "This request contains every eligible candidate" not in final["instruction"]
    assert "contains_every_eligible_candidate is false" not in final["instruction"]
    assert "partial" not in final["instruction"].casefold()
    assert "original eligible set has 3" in final["instruction"]
    assert "This round is final." in final["instruction"]
    assert "c00" in final["instruction"] and "c02" in final["instruction"]
    assert "Select c00." in final["instruction"]
    assert opening["comparison_scope"]["round_role"] == "opening"
    assert opening["comparison_scope"]["eligible_total"] == 3
    assert opening["comparison_scope"]["contains_every_eligible_candidate"] is False
    assert opening["comparison_scope"]["finalists"] == []
    assert final["comparison_scope"]["round_role"] == "final"
    assert final["comparison_scope"]["eligible_total"] == 3
    assert final["comparison_scope"]["match_ids"] == ["c00", "c02"]
    assert final["comparison_scope"]["finalists"] == ["c00", "c02"]
    assert final["comparison_scope"]["contains_every_eligible_candidate"] is False
    assert final["comparison_scope"]["prior_match_rationales"][0]["winner"] == "c00"
    assert "Select c00." in final["comparison_scope"]["prior_match_rationales"][0]["rationale"]
    recorded = finished.metadata["core_bracket"]["rounds"][1]["comparison_scope"]
    assert recorded["round_role"] == "final"
    assert recorded["eligible_total"] == 3
    assert recorded["finalists"] == ["c00", "c02"]
    assert recorded["contains_every_eligible_candidate"] is False
    assert finished.metadata["core_bracket_is_substantive_stability"] is False
    assert finished.metadata["core_bracket_schedule_note"] == CORE_SCHEDULE_NOTE


def test_bracket_order_is_scheduling_and_a_cycle_can_change_the_core():
    orders = (["c00", "c01", "c02"], ["c02", "c01", "c00"], ["c01", "c00", "c02"])
    assert all(order_core_candidates(order) == ["c00", "c01", "c02"] for order in orders)
    plans = [
        plan_core_comparison(order, lambda ids: 2500 * len(list(ids)), per_request=6000)["chunks"]
        for order in orders
    ]
    assert plans[0] == plans[1] == plans[2] == [["c00", "c01"]]
    report = bracket_order_report(["c02", "c00", "c01"], _cycle)
    assert report["input_order_changes_bracket"] is False
    assert report["alternative_bracket_changes_core"] is True
    assert report["code_id_core"] == "c02"
    assert report["alternative_core"] == "c00"
    assert report["substantive_stability"] is False
    assert report["note"] == CORE_SCHEDULE_NOTE
    assert play_core_bracket(["c00", "c01", "c02"], _cycle) == "c02"
    assert play_core_bracket(["c01", "c02", "c00"], _cycle) == "c00"
    first = lambda match: match[0]
    assert play_core_bracket(order_core_candidates(["c02", "c00", "c01"]), first) == "c00"


def test_a_supported_observed_condition_stays_eligible_and_speculative_ones_do_not():
    quote = f"{_ASSOCIATED} if it rains"
    observed = _conditional(quote, quote)
    assert observed["fixture"] == "synthetic"
    assert observed["evidence_tier"] == 2
    assert observed["evidence_tier"] != 3
    assert observed["condition_retained"] is True
    assert observed["condition_pattern"] == "observed_conditional"
    assert classify_condition_pattern(observed) == "observed_conditional"
    assert observed["downstream_eligible"] is True
    assert "hypothetical_link" not in observed["eligibility_blocks"]
    assert "incomplete_condition" not in observed["eligibility_blocks"]
    assert observed["downstream_condition"] == "if it rains"
    assert observed["claims"]["pattern"]["condition"] == "if it rains"
    assert observed["claims"]["pattern"]["condition_pattern"] == "observed_conditional"
    assert observed["claims"]["pattern"]["supports_core"] is True
    assert observed["participant_link"]["authorizes_tier_3"] is False
    assert withholding_reasons(observed) == []

    speculative = _conditional(
        f"someone said {_ASSOCIATED} if it rains",
        f"someone said {_ASSOCIATED} if it rains",
    )
    assert speculative["condition_pattern"] == "speculative_counterfactual"
    assert speculative["downstream_eligible"] is False
    assert "hypothetical_link" in speculative["eligibility_blocks"]
    assert speculative.get("downstream_condition") in (None, "")
    assert "hypothetical_link" in withholding_reasons(speculative)

    would = _conditional(f"{_ASSOCIATED} would hold if it rains", f"{_ASSOCIATED} would hold if it rains")
    assert would["condition_pattern"] == "speculative_counterfactual"
    assert "hypothetical_link" in would["eligibility_blocks"]
    assert would["downstream_eligible"] is False

    incomplete = _conditional(f"{_ASSOCIATED} if you", _ASSOCIATED)
    assert incomplete["condition_pattern"] == "incomplete"
    assert incomplete["downstream_eligible"] is False
    assert "incomplete_condition" in incomplete["eligibility_blocks"]
    assert "hypothetical_link" in incomplete["eligibility_blocks"]

    causal = _conditional(f"{_LINK} if it rains", f"{_LINK} if it rains")
    assert causal["condition_pattern"] == "observed_conditional"
    assert "hypothetical_link" not in causal["eligibility_blocks"]
    assert causal["downstream_eligible"] is False
    reasons = withholding_reasons(causal)
    assert "wording_withheld" in reasons
    assert "hypothetical_link" not in reasons
    assert causal.get("downstream_condition") in (None, "")


def test_candidate_evidence_summaries_are_capped_and_configurable():
    definition = "Tools for the shift cover the dock, the radio, and the late board. " * 8
    statement = "Tools for the shift are related to coworkers on the crew when the dock is open."
    variation = "The link is weaker on the night shift and stronger when the dock is open."
    counter = "The crew finished the shift without those tools and the work still moved."
    state = {
        "codebook": {
            "c00": {"code_id": "c00", "code": "tools", "definition": definition},
            "c01": {"code_id": "c01", "code": "crew", "definition": "Coworkers on the crew."},
        },
        "relationships": [
            {
                "relationship_id": "r1",
                "from_id": "c00",
                "to_id": "c01",
                "statement": statement,
                "variation_note": variation,
                "disposition": "accepted",
                "evidence_tier": 2,
                "counterexamples": [
                    {"quote": counter, "judgment": "contradicts", "valid_test": True}
                ],
            }
        ],
        "metadata": {
            "procedure": {
                "core_evidence_summaries": True,
                "core_evidence_token_cap": 80,
                "max_axial_review_tokens": 24000,
            },
            "performance": {"axial_token_budget": 6000},
        },
    }
    rows = _score_candidates(state)
    by_id = {row["code_id"]: row for row in rows}
    summary = by_id["c00"]["evidence_summary"]
    assert summary["definition"]
    assert summary["representative_statements"]
    assert statement[:40] in summary["representative_statements"][0]
    assert summary["key_variation"]
    assert summary["counterevidence"]
    assert summary["tokens"] == estimate_tokens(
        " ".join(
            piece
            for piece in (
                summary["definition"],
                "; ".join(summary["representative_statements"]),
                summary["key_variation"],
                summary["counterevidence"],
            )
            if piece
        )
    )
    assert summary["tokens"] <= summary["token_cap"] <= 80
    assert summary["improves_selection"] is None
    assert by_id["c00"]["core_evidence_improves_selection"] is None
    assert by_id["c00"]["core_evidence_extra_tokens_per_candidate"] == summary["tokens"]
    assert by_id["c00"]["core_evidence_extra_tokens_per_candidate"] > 0

    tight = {
        **state,
        "metadata": {
            "procedure": {
                "core_evidence_summaries": True,
                "core_evidence_token_cap": 80,
                "max_axial_review_tokens": 24000,
            },
            "performance": {"axial_token_budget": 40},
        },
    }
    stored = {row["code_id"]: row for row in _score_candidates(tight)}
    tight_summary = stored["c00"]["evidence_summary"]
    assert tight_summary["token_cap"] == 80
    assert tight_summary["source_definition"]
    assert tight_summary["source_statements"]
    assert tight_summary["source_counterevidence"]

    off = {
        **state,
        "metadata": {
            "procedure": {"core_evidence_summaries": False, "core_evidence_token_cap": 80},
            "performance": {"axial_token_budget": 6000},
        },
    }
    quiet = {row["code_id"]: row for row in _score_candidates(off)}
    assert quiet["c00"]["evidence_summary"] is None
    assert quiet["c00"]["core_evidence_extra_tokens_per_candidate"] == 0
    assert quiet["c00"]["core_evidence_improves_selection"] is None


def test_core_instruction_scope_is_explicit_without_a_model_call():
    text = _core_match_instruction(
        match_ids=["c00", "c02"],
        eligible_ids=["c00", "c01", "c02"],
        round_role="final",
        prior_rationales=[{"match": ["c00", "c01"], "winner": "c00", "rationale": "c00 held " * 80}],
    )
    assert "This request contains every eligible candidate" not in text
    assert "contains_every_eligible_candidate is false" not in text
    assert "partial" not in text.casefold()
    assert "Do not abstain because this match is not the whole eligible set." in text
    assert "original eligible set has 3" in text
    assert "This round is final." in text
    assert "c00, c02" in text
    assert len(text.split("c00 held")[1]) < 400
    payload = _core_review_payload(
        [{"code_id": "c00"}, {"code_id": "c02"}],
        preferred=[],
        snapshot=graph_snapshot([]),
        chunk_index=1,
        chunk_count=2,
        instruction=text,
    )
    assert payload["instruction"] == text
