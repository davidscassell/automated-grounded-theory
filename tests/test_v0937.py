"""0.9.37 core budget. Fakes and synthetic fixtures only. No network calls.

These checks reproduce the v42 bracket stop and the decisions that replace it.
They are processing coverage. They do not establish selection validity.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.axial import (
    _core_resource_stop,
    _runtime,
    export_axial_state,
    load_axial_export,
    resume_core_selection,
    selective_code,
)
from automated_grounded_theory.decision_review import (
    CHUNKING_CHECKPOINT_MESSAGE,
    CORE_AUTO_SIZE_RULE,
    CORE_BUDGET_RULE,
    CORE_CALL_ALLOWANCE_INSUFFICIENT,
    CORE_MATCH_REUSE_RULE,
    CORE_PARTIAL_EXECUTION_RULE,
    CORE_PREFLIGHT_RULE,
    CORE_REQUEST_TOO_LARGE,
    CORE_STOP_REASON_RULE,
    CORE_TOKEN_ALLOWANCE_INSUFFICIENT,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.version import __version__
from tests.test_v0934 import _core_source
from tests.test_v0935 import ScriptedCore, _core_bodies, _run_core
from tests.test_v0936 import _match_records


def _run(source, steps, **kwargs):
    model = ScriptedCore(core_steps=list(steps))
    result = selective_code(
        source,
        model,
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        batch_size=1,
        **kwargs,
    )
    return result, model


def test_version_names_the_core_budget_rules():
    assert __version__ == "0.9.40"
    assert POLICY_VERSION == "0.9.40"
    assert CORE_BUDGET_RULE == "separate_core_allowance_v1"
    assert CORE_PREFLIGHT_RULE == "opening_unavoidable_later_contingent_v1"
    assert CORE_PARTIAL_EXECUTION_RULE == "partial_core_off_by_default_v1"
    assert CORE_STOP_REASON_RULE == "request_call_or_token_v1"
    assert CORE_MATCH_REUSE_RULE == "reuse_validated_match_v1"
    assert CORE_AUTO_SIZE_RULE == "bounded_core_auto_size_off_v1"
    assert "0.9.36" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.37" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.38" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.39" in CHUNKING_CHECKPOINT_MESSAGE


def _wide_source():
    """54 candidates whose opening matches sit near the 6000 request cap."""
    return _core_source("packing " * 40, 54)


def test_v42_stops_before_spending_and_names_the_allowance():
    result, model = _run(
        _wide_source(),
        ["abstain"] * 12,
        max_decision_calls=4,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
    )
    preflight = result.metadata["core_bracket_preflight"]
    assert preflight["unavoidable_opening_calls"] == 6
    assert preflight["unavoidable_opening_tokens"] > 24000
    assert preflight["contingent_later_calls"] >= 0
    assert preflight["calls_available"] == 4
    assert preflight["tokens_available"] == 24000
    assert preflight["call_shortfall"] > 0
    assert preflight["token_shortfall"] > 0
    assert "advance nobody" in preflight["abstention_note"]
    assert "not a safe assumption for every provider" in preflight["provider_cost_note"]
    assert any("Unavoidable opening matches" in item for item in result.warnings)
    assert any("Contingent later rounds" in item for item in result.warnings)
    assert result.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 0
    assert model.core_seen == 0
    assert result.metadata["core_outcome"] == "selection_unresolved"
    assert result.metadata["core_selection_complete"] is False
    assert result.metadata["model_core_id"] == ""
    stops = result.metadata["decision_reviews"]["axial_core_review"]["stop_reasons"]
    assert CORE_CALL_ALLOWANCE_INSUFFICIENT in stops
    assert CORE_TOKEN_ALLOWANCE_INSUFFICIENT in stops
    assert result.metadata["decision_stop"] in {
        CORE_CALL_ALLOWANCE_INSUFFICIENT,
        CORE_TOKEN_ALLOWANCE_INSUFFICIENT,
        CORE_REQUEST_TOO_LARGE,
    }
    assert result.metadata["decision_stop"] != "core_pair_exceeds_request_budget"
    assert result.metadata["core_comparison_stop"] != "core_pair_exceeds_request_budget"
    assert "Examining" in result.metadata["core_coverage_note"]
    assert "does not mean that no core exists" in result.metadata["core_coverage_note"]


def test_partial_execution_sends_what_fits_and_stays_resumable():
    result, model = _run(
        _wide_source(),
        ["abstain"] * 12,
        max_decision_calls=4,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
        allow_partial_core_review=True,
    )
    sent = result.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"]
    assert sent == model.core_seen
    assert 1 <= sent < 6
    assert result.metadata["core_outcome"] == "selection_unresolved"
    assert result.metadata["core_resume"]["resumable"] is True
    assert result.metadata["decision_stop"] in {
        CORE_CALL_ALLOWANCE_INSUFFICIENT,
        CORE_TOKEN_ALLOWANCE_INSUFFICIENT,
    }
    assert result.metadata["decision_stop"] != "core_pair_exceeds_request_budget"
    assert result.metadata["allow_partial_core_review"] is True


def test_a_request_over_the_per_request_cap_is_not_a_pair_failure():
    """A live payload over 6000 is core_request_too_large, not a pair-plan failure.

    A pair the planner cannot split still stops as core_pair_exceeds_request_budget
    before a send. That plan-time name is a different stop.
    """
    assert (
        _core_resource_stop(
            "request_budget",
            estimate=6001,
            calls_sent=0,
            tokens_sent=0,
            call_ceiling=8,
            token_ceiling=24000,
            per_request=6000,
        )
        == CORE_REQUEST_TOO_LARGE
    )
    assert (
        _core_resource_stop(
            None,
            estimate=6001,
            calls_sent=0,
            tokens_sent=0,
            call_ceiling=8,
            token_ceiling=24000,
            per_request=6000,
        )
        == CORE_REQUEST_TOO_LARGE
    )
    assert (
        _core_resource_stop(
            None,
            estimate=6000,
            calls_sent=0,
            tokens_sent=0,
            call_ceiling=8,
            token_ceiling=24000,
            per_request=6000,
        )
        is None
    )
    assert (
        _core_resource_stop(
            "call_allowance",
            estimate=1000,
            calls_sent=4,
            tokens_sent=4000,
            call_ceiling=4,
            token_ceiling=24000,
            per_request=6000,
        )
        == CORE_CALL_ALLOWANCE_INSUFFICIENT
    )
    assert (
        _core_resource_stop(
            "token_allowance",
            estimate=1000,
            calls_sent=1,
            tokens_sent=20000,
            call_ceiling=8,
            token_ceiling=24000,
            per_request=6000,
        )
        == CORE_TOKEN_ALLOWANCE_INSUFFICIENT
    )
    assert _core_resource_stop(
        "request_budget",
        estimate=6001,
        calls_sent=0,
        tokens_sent=0,
        call_ceiling=8,
        token_ceiling=24000,
        per_request=6000,
    ) != "core_pair_exceeds_request_budget"


def test_call_and_token_shortfalls_are_distinct():
    calls, _model = _run(
        _wide_source(),
        ["abstain"] * 8,
        max_decision_calls=4,
        max_core_review_calls=1,
        max_core_review_tokens=10**6,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
    )
    assert calls.metadata["decision_stop"] == CORE_CALL_ALLOWANCE_INSUFFICIENT
    assert calls.metadata["core_outcome"] == "selection_unresolved"
    assert calls.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 0
    assert CORE_TOKEN_ALLOWANCE_INSUFFICIENT not in calls.metadata["decision_reviews"]["axial_core_review"]["stop_reasons"]

    tokens, _token_model = _run(
        _wide_source(),
        ["abstain"] * 8,
        max_decision_calls=4,
        max_core_review_calls=20,
        max_core_review_tokens=8000,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
    )
    assert tokens.metadata["decision_stop"] == CORE_TOKEN_ALLOWANCE_INSUFFICIENT
    assert tokens.metadata["core_outcome"] == "selection_unresolved"
    assert tokens.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 0
    preflight = tokens.metadata["core_bracket_preflight"]
    assert preflight["token_shortfall"] > 0
    assert preflight["calls_available"] == 20
    assert preflight["call_shortfall"] == 0


def test_omitted_core_allowance_matches_the_shared_step_defaults():
    runtime = _runtime(
        ScriptedCore(core_steps=["c00"]),
        decision_mode="llm",
        decision_model=None,
        request_timeout=1,
        include_memos=False,
        include_storyline=False,
        propose_calls=1,
        counter_calls=1,
        selective_calls=1,
        decision_calls=4,
        max_review_tokens=24000,
        token_budget=6000,
        review_calls=4,
    )
    allowance = runtime.limiter.provider_allowance
    assert allowance.stage_ceilings["axial_core_review"] == 4
    assert allowance.stage_ceilings["axial_relation_review"] == 4
    assert allowance.stage_ceilings["axial_saturation_review"] == 4
    assert allowance.stage_token_ceilings["axial_core_review"] == 24000
    assert allowance.stage_token_ceilings["axial_relation_review"] == 24000
    assert runtime.reviewer.token_budget == 6000

    raised = _runtime(
        ScriptedCore(core_steps=["c00"]),
        decision_mode="llm",
        decision_model=None,
        request_timeout=1,
        include_memos=False,
        include_storyline=False,
        propose_calls=1,
        counter_calls=1,
        selective_calls=1,
        decision_calls=4,
        max_review_tokens=24000,
        token_budget=6000,
        review_calls=4,
        core_review_calls=12,
        core_review_tokens=80000,
    )
    raised_allowance = raised.limiter.provider_allowance
    assert raised_allowance.stage_ceilings["axial_core_review"] == 12
    assert raised_allowance.stage_ceilings["axial_relation_review"] == 4
    assert raised_allowance.stage_ceilings["axial_saturation_review"] == 4
    assert raised_allowance.stage_token_ceilings["axial_core_review"] == 80000
    assert raised_allowance.stage_token_ceilings["axial_relation_review"] == 24000
    assert raised.reviewer.max_calls == 4


def test_a_larger_budget_reuses_validated_matches_until_the_payload_or_policy_changes(tmp_path):
    source = _wide_source()
    first, first_model = _run(
        source,
        ["abstain"] * 6,
        max_decision_calls=4,
        max_core_review_calls=2,
        max_core_review_tokens=10**6,
        max_axial_review_tokens=10**6,
        axial_token_budget=6000,
        allow_partial_core_review=True,
    )
    assert first.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 2
    assert first_model.core_seen == 2
    assert [row["abstention_reason"] for row in _match_records(first)[:2]] == [
        "no_suitable_candidate",
        "no_suitable_candidate",
    ]
    export_axial_state(
        {
            "metadata": first.metadata,
            "warnings": list(first.warnings),
            "prompt_log": list(first.prompt_log),
            "relationships": [],
            "core_candidates": list(first.core_candidates),
            "saturation": {},
            "coverage": {},
        },
        tmp_path,
    )
    loaded = load_axial_export(tmp_path)
    assert loaded["metadata"]["freeze_core_candidates"] is True
    assert len(loaded["core_candidates"]) == 54
    resumed, resumed_model = _run(
        loaded,
        ["abstain"] * 6,
        max_decision_calls=4,
        max_core_review_calls=8,
        max_core_review_tokens=10**6,
        max_axial_review_tokens=10**6,
        axial_token_budget=6000,
    )
    assert resumed_model.core_seen < 6
    assert resumed.metadata["core_matches_reused"] >= 2
    assert resumed_model.core_seen == 6 - 2 or resumed.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] >= 2
    # The two stored abstentions are not sent again. New calls are only the rest.
    assert resumed_model.core_seen == max(0, int(resumed.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"]) - 2)

    changed = load_axial_export(tmp_path)
    stored_match = changed["metadata"]["decision_chunk_state"]["core_bracket"]["rounds"][0]["match_records"][0]["match"]
    target = str(stored_match[0])
    for index, row in enumerate(changed["core_candidates"]):
        if str(row.get("code_id") or "") != target:
            continue
        edited = dict(row)
        edited["code"] = "CHANGED " + str(edited.get("code") or "")
        changed["core_candidates"][index] = edited
        break
    again, again_model = _run(
        changed,
        ["abstain"] * 6,
        max_decision_calls=4,
        max_core_review_calls=8,
        max_core_review_tokens=10**6,
        max_axial_review_tokens=10**6,
        axial_token_budget=6000,
    )
    assert again.metadata["core_reuse_invalidated"] == "payload_changed"
    assert again_model.core_seen >= 1

    policy = load_axial_export(tmp_path)
    state = policy["metadata"]["decision_chunk_state"]
    records = state["core_bracket"]["rounds"][0]["match_records"]
    records[0]["policy_fingerprint"] = "not-the-policy"
    records[1]["policy_fingerprint"] = "not-the-policy"
    policy_run, policy_model = _run(
        policy,
        ["abstain"] * 6,
        max_decision_calls=4,
        max_core_review_calls=8,
        max_core_review_tokens=10**6,
        max_axial_review_tokens=10**6,
        axial_token_budget=6000,
    )
    assert policy_run.metadata["core_reuse_invalidated"] == "policy_changed"
    assert policy_model.core_seen >= 1


def test_auto_size_stays_off_and_a_hard_limit_bounds_it():
    untouched, _model = _run(
        _wide_source(),
        ["abstain"],
        max_decision_calls=4,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
    )
    assert untouched.metadata["auto_size_core_review"] is False
    assert untouched.metadata["auto_size_core_applied"] is False
    assert untouched.metadata["core_bracket_preflight"]["calls_available"] == 4

    bounded, _bounded_model = _run(
        _wide_source(),
        ["abstain"] * 8,
        max_decision_calls=4,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
        auto_size_core_review=True,
        max_core_review_call_hard_limit=5,
        max_core_review_token_hard_limit=20000,
    )
    preflight = bounded.metadata["core_bracket_preflight"]
    assert bounded.metadata["auto_size_core_review"] is True
    assert preflight["calls_available"] == 5
    assert preflight["tokens_available"] == 20000
    assert preflight["calls_available"] < preflight["unavoidable_opening_calls"] + preflight["contingent_later_calls"] or (
        preflight["tokens_available"] <= 20000
    )
    assert bounded.metadata["core_outcome"] == "selection_unresolved"
    assert "not a safe assumption for every provider" in preflight["provider_cost_note"]


def test_stopping_semantics_still_hold():
    none = _run_core(["abstain", "abstain"])
    assert none.metadata["core_outcome"] == "no_core_supported"

    unresolved = _run_core(["abstain"], max_decision_calls=1)
    assert unresolved.metadata["core_outcome"] == "selection_unresolved"

    survivor = _run_core(["abstain", "c02"])
    assert _core_bodies(survivor)[-1]["review_chunk"]["code_ids"] == ["c02"]
    assert survivor.metadata["core_outcome"] == "core_selected"
    assert survivor.metadata["model_core_id"] == "c02"

    selected = _run_core(["c00", "c00"])
    assert selected.metadata["core_outcome"] == "core_selected"
    assert selected.metadata["model_core_id"] == "c00"


def test_resume_core_selection_uses_the_frozen_export(tmp_path):
    source = _wide_source()
    first, first_model = _run(
        source,
        ["abstain"] * 8,
        max_decision_calls=4,
        max_core_review_calls=2,
        max_core_review_tokens=10**6,
        axial_token_budget=6000,
        allow_partial_core_review=True,
    )
    assert first_model.core_seen == 2
    assert first.metadata["core_outcome"] == "selection_unresolved"
    export_axial_state(
        {
            "metadata": first.metadata,
            "warnings": list(first.warnings),
            "prompt_log": list(first.prompt_log),
            "relationships": [],
            "core_candidates": list(first.core_candidates),
            "saturation": {},
            "coverage": {},
        },
        tmp_path,
    )
    model = ScriptedCore(core_steps=["c02", "c02", "c02"])
    resumed = resume_core_selection(
        tmp_path,
        model,
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=4,
        max_core_review_calls=8,
        max_core_review_tokens=10**6,
        axial_token_budget=6000,
        batch_size=1,
    )
    assert model.core_seen >= 1
    assert resumed.metadata["core_matches_reused"] >= 1
    bodies = _core_bodies(resumed)
    assert bodies
    assert json.loads((tmp_path / "core_candidates.json").read_text(encoding="utf-8"))["candidates"]


class _CountingCore(ScriptedCore):
    """Abstain with a long rationale, the shape that pushed v42 over 6000."""

    def with_structured_output(self, schema, **kwargs):
        name = getattr(schema, "__name__", "")
        if name != "ModelCoreDecision":
            return ScriptedCore.with_structured_output(self, schema, **kwargs)

        def _call(_messages):
            self.call_count += 1
            self.core_seen += 1
            parsed = schema(
                decision="abstain",
                core_id="",
                abstention_reason="no_suitable_candidate",
                rationale="m" * 240,
            )
            raw = AIMessage(content="{}", usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42})
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def test_same_round_rationales_do_not_make_the_next_opening_match_a_pair_overflow():
    source = _wide_source()
    model = _CountingCore(core_steps=["abstain"] * 8)
    result = selective_code(
        source,
        model,
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=4,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
        allow_partial_core_review=True,
        batch_size=1,
    )
    assert result.metadata["decision_stop"] != "core_pair_exceeds_request_budget"
    assert result.metadata["decision_stop"] != CORE_REQUEST_TOO_LARGE
    assert model.core_seen == 4
    estimates = [
        row.get("estimated_input_tokens")
        for row in result.prompt_log
        if row.get("checkpoint") == "axial_core_review" and row.get("status") == "validated"
    ]
    assert estimates
    assert max(estimates) <= 6000
