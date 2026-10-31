"""0.9.28 review fixes. Fakes and fixtures only. No network calls."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    AUTO_SIZE_DECISION_CALLS,
    BUDGET_LEDGER_RULE,
    CHUNK_REVIEW_RECOVERY,
    CONDITIONAL_SCOPE_RULE,
    DECISION_REVIEW_CHUNKING,
    DUPLICATE_ASSERTION_RULE,
    EXPORT_ASSESSMENT_RULE,
    DecisionChunkingCheckpointError,
    DecisionReviewer,
    axial_code,
    compare_pipeline_runs,
    export_gate_relationships,
    focused_code,
    measure_chunk_context_sensitivity,
    relationship_assessment,
    repeat_frozen_review,
    repeat_pipeline,
    selective_code,
)
from automated_grounded_theory.async_support import ProviderAllowance, run_sync
from automated_grounded_theory.prompts import message_text
from automated_grounded_theory.axial import (
    AxialDecisionReviewer,
    AxialError,
    ModelCoreDecision,
    _commit_relations,
    _core_review_payload,
    _score_candidates,
    aaxial_code,
    aselective_code,
    consolidate_duplicate_assertions,
)
from automated_grounded_theory.graph_grounding import graph_snapshot
from automated_grounded_theory.backends import TEMPERATURE_REPRODUCIBILITY_NOTE
from automated_grounded_theory.decision_review import (
    CHUNKING_CHECKPOINT_MESSAGE,
    CHUNK_ALLOWANCE_INSUFFICIENT,
    PARTIAL_SELECTION_BIAS,
    PARTIAL_SELECTION_SHUFFLE_BIAS,
    SELECTION_CONSOLIDATION,
    PER_STEP_DECISION_REVIEW,
    SYSTEM,
    SelectionDecision,
    chunk_input_fingerprint,
    selection_policy_fingerprint,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder
from tests.test_v0924 import _KeepSelector, _code, _focused_source, _rewrite_metadata, _selection_state
from tests.test_v0926 import SENTENCE, _book, _both, _passages, _row
from tests.test_v0927 import _focused, _large_codes, _tier_two_source

_ASIDE = Path(__file__).resolve().parents[0] / "fixtures" / "discourse_aside_0928.json"


def test_version_rules_and_unchanged_defaults():
    assert __version__ == "0.9.40"
    assert POLICY_VERSION == "0.9.40"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v6"
    assert CONDITIONAL_SCOPE_RULE == "condition_scope_v1"
    assert DUPLICATE_ASSERTION_RULE == "merge_all_evidence_v1"
    assert BUDGET_LEDGER_RULE == "per_step_review_ledger_v1"
    assert EXPORT_ASSESSMENT_RULE == "shared_assessment_v1"
    assert CHUNK_REVIEW_RECOVERY == "reuse_validated_v1"
    assert "validated-chunk recovery" in CHUNKING_CHECKPOINT_MESSAGE
    assert AUTO_SIZE_DECISION_CALLS is False
    import inspect

    focused = inspect.signature(focused_code).parameters
    assert focused["selection_chunk_order"].default == "id"
    assert focused["selection_chunk_order_seed"].default is None
    assert focused["selection_consolidation"].default is False
    assert inspect.signature(focused_code).parameters["max_decision_calls"].default == 8
    assert inspect.signature(focused_code).parameters["decision_token_budget"].default == 6000
    assert inspect.signature(aaxial_code).parameters["max_axial_review_tokens"].default == 24000
    assert inspect.signature(aaxial_code).parameters["axial_token_budget"].default == 6000
    assert inspect.signature(aselective_code).parameters["max_decision_calls"].default == 4


def test_ineligible_accepted_row_is_explained_and_not_downstream_accepted(tmp_path):
    source = _tier_two_source([])
    finished = axial_code(
        source,
        FakeAxialCoder(),
        seed=3,
        interrupt=False,
        max_axial_review_calls=0,
        axial_relations=False,
        batch_size=1,
        output_dir=tmp_path,
    )
    row = finished.relationships[0]
    assert row["evidence_tier"] == 2
    assert row["downstream_eligible"] is False
    assert row["endpoint_review_status"] == "unreviewed"
    assert row["primary_withholding_cause"] == "endpoint_unreviewed_by_focus"
    assert "endpoint_unreviewed_by_focus" in row["withholding_reasons"]
    saved = json.loads((tmp_path / "axial_run.json").read_text(encoding="utf-8"))
    saved_row = saved["relationships"][0]
    exported = export_gate_relationships(saved["relationships"])
    copied = exported["accepted"][0]
    for field in (
        "evidence_tier",
        "reviewer_disposition",
        "endpoint_review_status",
        "downstream_eligible",
        "withholding_reasons",
        "primary_withholding_cause",
        "withholding_cause",
    ):
        assert copied[field] == saved_row[field] == relationship_assessment(saved_row)[field]
    assert exported["accepted_downstream"] == []
    assert exported["withholding_cause_counts"]["endpoint_unreviewed_by_focus"] == 1
    assert sum(exported["withholding_cause_counts"].values()) == exported["explained_rows"] == 1
    ineligible = [item for item in exported["accepted"] + exported["withheld"] + exported["proposed"] if item["downstream_eligible"] is False]
    assert ineligible
    assert all(item["withholding_reasons"] for item in ineligible)


def test_policy_rejection_does_not_hide_insufficient_cases():
    exported = export_gate_relationships(
        [
            {
                "relationship_id": "rel-policy",
                "disposition": "rejected",
                "review_actor": "llm",
                "reviewer_decided": True,
                "reviewer_disposition": "rejected",
                "model_disposition": "rejected",
                "evidence_tier": 1,
                "downstream_eligible": False,
                "flags": ["endpoint_evidence_partial"],
                "endpoint_anchoring": [{"anchored": False}],
            }
        ]
    )
    row = exported["withheld"][0]
    assert row["primary_withholding_cause"] == "partial_endpoint_evidence"
    assert "insufficient_cases" in row["withholding_reasons"]
    assert "reviewer_policy_rejection" in row["withholding_reasons"]
    assert "llm_reviewer_rejected" not in row["withholding_reasons"]
    assert exported["withholding_cause_counts"]["partial_endpoint_evidence"] == 1
    assert exported["withholding_reason_counts"]["insufficient_cases"] == 1
    assert exported["withholding_reason_counts"]["reviewer_policy_rejection"] == 1
    assert sum(exported["withholding_cause_counts"].values()) == 1
    assert sum(exported["withholding_reason_counts"].values()) > 1


def test_discourse_aside_does_not_erase_a_real_conditional():
    payload = json.loads(_ASIDE.read_text(encoding="utf-8"))
    assert "not study data" in payload["note"].casefold()
    readings = {}
    for case in payload["cases"]:
        row = _row(case["quote"], ["r1", "r2"], [_both("p1", case["quote"]), _both("p2", case["quote"])])
        link = row["participant_link"]
        assert link["reading_basis"] == "deterministic_provisional"
        assert link["reading_provisional"] is True
        assert link["status"] == case["status"], case["id"]
        assert (link["status"] == "hypothetical") is case["hypothetical"], case["id"]
        readings[case["id"]] = link["status"]
    assert readings["aside-alone"] != "hypothetical"
    assert readings["aside-and-conditional"] == "hypothetical"


def test_identical_assertions_are_merged_and_reassessed():
    statement = "tools for the shift may be related to coworkers on the crew"
    passages = _passages(SENTENCE, ["r1", "r2"])
    high = {
        "relationship_id": "rel-0002",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": SENTENCE,
        "statement": statement,
        "disposition": "accepted",
        "evidence_tier": 2,
        "case_quotes": [_both("p1", SENTENCE)],
    }
    low = {
        "relationship_id": "rel-0001",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": SENTENCE,
        "statement": statement,
        "disposition": "proposed",
        "reviewer_disposition": "proposed",
        "evidence_tier": 1,
        "case_quotes": [_both("p1", SENTENCE)],
    }
    other = {
        "relationship_id": "rel-0003",
        "from_id": "tools",
        "to_id": "crew",
        "statement": "tools for the shift shape the return",
        "disposition": "proposed",
        "evidence_tier": 1,
        "case_quotes": [_both("p1", SENTENCE)],
    }
    opposite = {
        "relationship_id": "rel-0004",
        "from_id": "crew",
        "to_id": "tools",
        "statement": statement,
        "disposition": "proposed",
        "evidence_tier": 1,
        "case_quotes": [_both("p1", SENTENCE)],
    }
    committed = _commit_relations(
        {
            "metadata": {"procedure": {"min_relation_documents": 2, "coding_lens": "paradigm", "pattern_min_cases": 2}},
            "relationships": [high, low, other, opposite],
            "passages": list(passages.values()),
            "codebook": _book(),
            "segments": [],
            "memos": [],
            "warnings": [],
        }
    )
    rows = committed["relationships"]
    assert len(rows) == 3
    merged = next(row for row in rows if row["relationship_id"] == "rel-0001")
    assert merged["evidence_tier"] != 2
    assert merged["conflicting_evidence"]
    assert {item["relationship_id"] for item in merged["assertion_provenance"]} == {"rel-0001", "rel-0002"}
    assert any(row["relationship_id"] == "rel-0003" for row in rows)
    assert any(row["from_id"] == "crew" and row["to_id"] == "tools" for row in rows)
    again = consolidate_duplicate_assertions(rows)
    assert len(again) == 3
    assert again[0]["conflicting_evidence"]

    risen_a = dict(low, relationship_id="rel-0001", disposition="proposed", evidence_tier=1, case_quotes=[_both("p1", SENTENCE)])
    risen_b = dict(low, relationship_id="rel-0002", disposition="proposed", evidence_tier=1, passage_id="p2", case_quotes=[_both("p2", SENTENCE)])
    risen = _commit_relations(
        {
            "metadata": {"procedure": {"min_relation_documents": 2, "coding_lens": "paradigm", "pattern_min_cases": 2}},
            "relationships": [risen_a, risen_b],
            "passages": list(passages.values()),
            "codebook": _book(),
            "segments": [],
            "memos": [],
            "warnings": [],
        }
    )["relationships"]
    assert len(risen) == 1
    assert risen[0]["conflicting_evidence"] is None
    assert risen[0]["evidence_tier"] == 2


def test_one_decision_call_charges_the_shared_review_stage_once():
    reviewer = AxialDecisionReviewer(FakeAxialCoder(), max_calls=8, max_tokens=24000, token_budget=6000)
    assert reviewer.limiter.max_retries == 0
    allowance = ProviderAllowance(max_tokens=24000, stages={"review": 8})
    reviewer.limiter.provider_allowance = allowance
    metadata = {"decision_calls_used": 0}
    payload = _core_review_payload(
        [{"code_id": "c0", "code": "tools"}],
        preferred=[],
        snapshot={},
        chunk_index=0,
        chunk_count=1,
    )
    parsed, _audit = run_sync(reviewer.decide(payload, ModelCoreDecision, metadata))
    assert parsed["core_id"] == "c0"
    assert metadata["decision_calls_used"] == 1
    assert allowance.stage_used["review"] == 1


def test_core_selection_keeps_its_own_call_allowance():
    label = "packing " * 40
    book = {f"c{index:02d}": {"code_id": f"c{index:02d}", "code": label, "definition": label} for index in range(8)}
    passages = [{"passage_id": "p1", "doc_id": "d1", "text": "A short note about packing."}]
    state = {
        "codebook": book,
        "passages": passages,
        "segments": [],
        "relationships": [],
        "properties": [],
        "dimensions": [],
        "metadata": {},
    }
    candidates = _score_candidates(state)
    reviewer = AxialDecisionReviewer(FakeAxialCoder(), max_calls=8, max_tokens=24000, token_budget=10**9)
    blank = {"decision_calls_used": 0}

    def estimate(ids):
        subset = [row for row in candidates if row["code_id"] in set(ids)]
        payload = _core_review_payload(
            subset, preferred=[], snapshot=graph_snapshot([]), chunk_index=10**6, chunk_count=10**6,
        )
        return reviewer.refusal_reason(payload, ModelCoreDecision, blank)[1]

    one = estimate([candidates[0]["code_id"]])
    two = estimate([candidates[0]["code_id"], candidates[1]["code_id"]])
    assert two > one
    source = {
        "codebook": book,
        "passages": passages,
        "segments": [],
        "relationships": [],
        "metadata": {
            "stage": "axial",
            "decision_calls_used": 4,
            "provider_budget": {
                "calls_used": 4,
                "tokens_estimated": 0,
                "stages": {"review": {"calls_used": 4, "max_calls": 8}},
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
        max_decision_calls=8,
        max_axial_review_tokens=24000,
        axial_token_budget=10**9,
        batch_size=1,
    )
    report = finished.metadata["decision_reviews"]["axial_core_review"]
    assert report["stop_reasons"] in (None, [])
    assert report["complete"] is True
    assert report["chunks_sent"] == 1
    assert report["starting_usage"] == 0
    assert report["calls_made"] == 1
    assert report["remaining_allowance"] == 7
    assert report["prior_step_calls"] == 4
    assert report["prior_step_calls_reduce_allowance"] is False
    assert report["allowance_scope"] == PER_STEP_DECISION_REVIEW
    assert report["allowance"] == 8
    assert report["shared_with"] == []
    assert finished.metadata["core_candidates_eligible"] == 8
    assert finished.metadata["core_candidates_compared"] == 8
    step = finished.metadata["decision_budget_steps"][-1]
    assert step["step"] == "axial_core_review"
    assert step["starting_usage"] == 0
    assert step["calls_made"] == 1
    assert step["remaining_allowance"] == 7
    assert step["stop_reason"] is None
    assert step["prior_step_calls"] == 4
    assert step["prior_step_calls_reduce_allowance"] is False
    assert step["shared_with"] == []
    assert finished.metadata["decision_calls_used"] == 1
    assert "temperature 0" in finished.metadata["model_configuration"]["note"]
    assert "not a reproducibility guarantee" in TEMPERATURE_REPRODUCIBILITY_NOTE


def test_a_0927_checkpoint_cannot_resume():
    saver = InMemorySaver()
    paused = focused_code(
        _focused_source(_large_codes()),
        model=FakeAxialCoder(),
        decision_model=_KeepSelector(),
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=True,
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_decision_calls=1,
        max_decision_tokens=200000,
        checkpointer=saver,
        thread_id="chunk-0928",
    )
    assert paused.interrupted

    def downgrade(metadata):
        procedure = metadata["procedure"]
        procedure["decision_review_chunking"] = "request_budget_v4"
        procedure["withholding_cause_rule"] = "primary_cause_v1"
        procedure["conditional_scope_rule"] = "link_governing_if_v1"
        for key in (
            "duplicate_assertion_rule",
            "budget_ledger_rule",
            "export_assessment_rule",
            "chunk_review_recovery",
        ):
            procedure.pop(key, None)
        manifest = metadata.get("checkpoint_manifest") or {}
        stored = dict(manifest.get("procedure") or {})
        stored.update(procedure)
        manifest["procedure"] = stored
        metadata["checkpoint_manifest"] = manifest

    _rewrite_metadata(saver, "chunk-0928", downgrade)
    with pytest.raises(DecisionChunkingCheckpointError, match="decision-review"):
        focused_code(
            _focused_source(_large_codes()),
            model=FakeAxialCoder(),
            decision_model=_KeepSelector(),
            decision_mode="llm",
            seed=3,
            batch_size=1,
            interrupt=True,
            max_focused_review_calls=0,
            focused_neighbor_limit=0,
            focused_outside_limit=0,
            focused_negative_case_limit=0,
            max_decision_calls=8,
            max_decision_tokens=200000,
            checkpointer=saver,
            thread_id="chunk-0928",
            resuming=True,
            resume={"continue_selection": True},
            allow_changes=True,
        )
    assert "chunked" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.28" in CHUNKING_CHECKPOINT_MESSAGE


def test_frozen_review_and_pipeline_repeat_stay_distinct():
    reviewer = DecisionReviewer(FakeAxialCoder(), max_calls=4, token_budget=6000)
    payload = {"kind": "focused_selection", "codes": [{"code_id": "c-a"}]}
    frozen = repeat_frozen_review(reviewer, payload, SelectionDecision, 2)
    assert frozen["experiment"] == "frozen_reviewer"
    assert frozen["prompt_hash_stable"] is True
    assert len(frozen["decisions"]) == 2
    assert frozen["decisions"][0]["prompt_hash"] == frozen["decisions"][1]["prompt_hash"]

    calls = {"n": 0}

    def pipeline():
        calls["n"] += 1
        return {
            "raised_code_ids": ["c-a"] if calls["n"] == 1 else ["c-a", "c-b"],
            "relationships": [
                {
                    "from_id": "tools",
                    "to_id": "crew",
                    "statement": "tools may be related to crew",
                    "downstream_eligible": calls["n"] == 1,
                }
            ],
        }

    piped = repeat_pipeline(pipeline, 2)
    assert piped["experiment"] == "full_pipeline"
    assert piped["repetitions"] == 2
    compared = compare_pipeline_runs(piped["runs"])
    pair = compared["pairwise"][0]
    assert pair["selected_code_jaccard"] == pytest.approx(0.5)
    assert pair["directed_pair_jaccard"] == 1.0
    assert pair["statement_jaccard"] == 1.0
    assert pair["eligibility_changes"]["matched"]
    assert pair["eligibility_changes"]["additions"] == []
    assert compared["summary"]["selected_code_jaccard"] == pytest.approx(0.5)
    assert compared["summary"]["eligibility_changes"] == pair["eligibility_changes"]["matched"]
    empty = compare_pipeline_runs([{}, {}])
    assert empty["pairwise"][0]["selected_code_jaccard"] is None
    assert empty["pairwise"][0]["directed_pair_jaccard"] is None
    assert empty["pairwise"][0]["eligibility_changes"] is None
    missing = compare_pipeline_runs(
        [
            {"raised_code_ids": [], "relationships": [{"from_id": "a", "to_id": "b", "statement": "same"}]},
            {"raised_code_ids": [], "relationships": [{"from_id": "a", "to_id": "b", "statement": "same", "downstream_eligible": True}]},
        ]
    )
    assert missing["pairwise"][0]["selected_code_jaccard"] is None
    unknown = missing["pairwise"][0]["eligibility_changes"]
    assert unknown["matched"] == []
    assert unknown["additions"] == []
    assert unknown["removals"] == []
    assert unknown["unknown_eligibility"][0]["left"] is None
    assert unknown["unknown_eligibility"][0]["right"] is True


def test_token_shortfall_stops_before_the_first_request(tmp_path):
    """Auto-sizing may raise the call allowance. It must not raise the token allowance."""
    capped = _focused(
        _large_codes(),
        tmp_path,
        max_decision_calls=1,
        auto_size_decision_calls=True,
        max_decision_tokens=1000,
    )
    assert capped.metadata["decision_stop"] == CHUNK_ALLOWANCE_INSUFFICIENT
    assert capped.metadata["decision_token_allowance"] == 1000
    assert capped.metadata["procedure"]["max_decision_tokens"] == 1000
    assert capped.metadata["selection_tokens_required"] > 1000
    assert capped.metadata["selection_token_shortfall"] == capped.metadata["selection_tokens_required"] - 1000
    assert capped.metadata["decision_call_allowance_raised"] is True
    assert capped.raised_code_ids == []
    report = capped.metadata["decision_reviews"]["focused_selection"]
    assert report["chunks_sent"] == 0
    assert report["tokens_required"] == capped.metadata["selection_tokens_required"]
    assert report["tokens_required"] == report["tokens_available"] + report["token_shortfall"]
    assert report["calls_required"] == report["chunks_required"]
    assert report["claim_review_calls_reserved"] == 1
    assert report["decision_call_allowance"] == report["chunks_required"] + report["claim_review_calls_reserved"]
    assert report["calls_available"] == report["chunks_required"]
    assert report["calls_required"] <= report["calls_available"]
    assert report["preflight_can_finish"] is False
    assert report["auto_size_covers_calls_only"] is True
    assert report["auto_size_raised_tokens"] is False
    assert report["token_estimate_includes_output_tokens"] is False
    assert report["token_estimate_includes_retry_or_repair_margin"] is False
    assert report["token_estimate_includes_fixed_prompt_overhead"] is True
    assert "output tokens" in report["token_estimate_scope"]
    assert "fixed system prompt" in report["token_estimate_scope"]
    assert "retry" in report["token_estimate_scope"]
    assert report["decision_token_allowance"] == 1000
    assert report["token_shortfall"] > 0
    assert CHUNK_ALLOWANCE_INSUFFICIENT in (report["stop_reasons"] or [])
    assert report["partial_selection_is_code_id_slice"] is True
    assert report["partial_selection_bias"] == PARTIAL_SELECTION_BIAS
    assert not any(entry.get("kind") == "decision_review" for entry in capped.prompt_log)
    selection = json.loads((tmp_path / "focused_selection.json").read_text(encoding="utf-8"))
    run = json.loads((tmp_path / "focused_run.json").read_text(encoding="utf-8"))
    for payload in (selection, run, run["metadata"]):
        assert payload["partial_selection_is_code_id_slice"] is True
        assert payload["partial_selection_bias"] == PARTIAL_SELECTION_BIAS
        assert "code-id order" in payload["partial_selection_bias"]
        assert "not a sample" in payload["partial_selection_bias"]
        assert payload["preflight_can_finish"] is False
        assert payload["auto_size_covers_calls_only"] is True
        assert payload["auto_size_raised_tokens"] is False
        assert payload["token_estimate_includes_output_tokens"] is False
        assert payload["token_estimate_includes_fixed_prompt_overhead"] is True
        assert payload["tokens_required"] > payload["tokens_available"]

    finished = _focused(
        [
            _code("c-b", "DROP this code"),
            _code("c-a", "KEEP this code"),
        ],
        tmp_path / "complete",
        max_decision_calls=8,
        max_decision_tokens=24000,
    )
    assert finished.metadata["selection_review_complete"] is True
    assert finished.metadata["partial_selection_is_code_id_slice"] is False
    assert finished.metadata["partial_selection_bias"] is None
    complete = json.loads((tmp_path / "complete" / "focused_selection.json").read_text(encoding="utf-8"))
    assert complete["partial_selection_is_code_id_slice"] is False
    assert complete["partial_selection_bias"] is None


def test_proposed_and_unreviewed_ineligible_rows_are_explained():
    exported = export_gate_relationships(
        [
            {
                "relationship_id": "rel-0005",
                "disposition": "proposed",
                "reviewer_decided": False,
                "reviewer_disposition": None,
                "evidence_tier": 0,
                "supporting_cases": [],
                "cases_with_both_endpoints": 0,
                "pattern_min_cases": 2,
            },
            {
                "relationship_id": "rel-unreviewed",
                "disposition": "proposed",
                "evidence_tier": 2,
                "supporting_cases": [{"both_endpoints": True}, {"both_endpoints": True}],
                "cases_with_both_endpoints": 2,
                "flags": ["endpoint_unreviewed_by_focus"],
                "eligibility_reason": "endpoint_unreviewed_by_focus",
            },
            {
                "relationship_id": "rel-open",
                "disposition": "proposed",
                "evidence_tier": 2,
                "supporting_cases": [{"both_endpoints": True}, {"both_endpoints": True}],
                "cases_with_both_endpoints": 2,
            },
        ]
    )
    by_id = {row["relationship_id"]: row for row in exported["proposed"]}
    zero = by_id["rel-0005"]
    assert zero["primary_withholding_cause"] == "insufficient_cases"
    assert zero["withholding_cause"] == "insufficient_cases"
    assert zero["reviewer_disposition"] is None
    assert "llm_reviewer_rejected" not in zero["withholding_reasons"]
    assert "reviewer_policy_rejection" not in zero["withholding_reasons"]
    assert by_id["rel-unreviewed"]["primary_withholding_cause"] == "endpoint_unreviewed_by_focus"
    assert by_id["rel-unreviewed"]["endpoint_review_status"] == "unreviewed"
    assert by_id["rel-open"]["primary_withholding_cause"] is None
    assert by_id["rel-open"]["withholding_reasons"] == []
    ineligible = [by_id["rel-0005"], by_id["rel-unreviewed"]]
    assert all(row["withholding_reasons"] for row in ineligible)
    assert sum(exported["withholding_cause_counts"].values()) == exported["explained_rows"] == len(ineligible)
    assert exported["accepted_downstream"] == []


class _ScriptedSelector(_KeepSelector):
    """Fake reviewer. Fails on one call, or narrows a consolidation pass."""

    def __init__(self, *, fail_at: int | None = None, consolidate_to: list[str] | None = None):
        super().__init__()
        object.__setattr__(self, "fail_at", fail_at)
        object.__setattr__(self, "consolidate_to", consolidate_to)
        object.__setattr__(self, "payloads", [])
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.calls += 1
            name = getattr(schema, "__name__", "")
            text = "\n".join(message_text(message) for message in messages)
            payload = json.loads(text[text.find("{"):])
            self.payloads.append(payload)
            if self.fail_at is not None and self.calls == self.fail_at and name == "SelectionDecision":
                raise RuntimeError("scripted review failure")
            if name == "SelectionDecision":
                if payload.get("review_pass") == SELECTION_CONSOLIDATION and self.consolidate_to is not None:
                    raised = list(self.consolidate_to)
                else:
                    raised = [
                        str(row.get("code_id") or "")
                        for row in payload.get("codes") or []
                        if "KEEP" in str(row.get("definition") or "")
                    ]
                parsed = schema(raised_code_ids=raised, rationale="Kept the marked codes in this request.")
            elif name == "ClaimDecisions":
                parsed = schema(
                    claims=[
                        {
                            "entry_id": claim["entry_id"],
                            "disposition": "proposed",
                            "support": "indeterminate",
                            "rationale": "Left proposed.",
                        }
                        for claim in payload.get("claims") or []
                    ]
                )
            else:
                raise AssertionError(name)
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def _selection_requests(model: _ScriptedSelector) -> list[dict]:
    return [
        payload
        for payload in model.payloads
        if payload.get("kind") == "focused_selection" and not payload.get("review_pass")
    ]


def _focused_common(saver, thread_id, decision_model, **extra):
    return dict(
        model=FakeAxialCoder(),
        decision_model=decision_model,
        decision_mode="llm",
        seed=3,
        batch_size=1,
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_decision_tokens=extra.pop("max_decision_tokens", 200000),
        checkpointer=saver,
        thread_id=thread_id,
        **extra,
    )


def _rewrite_channel(saver: InMemorySaver, thread_id: str, channel: str, mutate) -> None:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    saved = saver.get_tuple(config)
    assert saved is not None
    version = saved.checkpoint["channel_versions"][channel]
    namespace = saved.config["configurable"].get("checkpoint_ns", "")
    key = (thread_id, namespace, channel, version)
    type_tag, blob = saver.blobs[key]
    value = saver.serde.loads_typed((type_tag, blob))
    mutate(value)
    saver.blobs[key] = saver.serde.dumps_typed(value)


def test_budget_only_resume_sends_after_a_token_raise(tmp_path):
    saver = InMemorySaver()
    source = _focused_source(_large_codes())
    first = _ScriptedSelector()
    paused = focused_code(
        source,
        **_focused_common(
            saver,
            "budget-only",
            first,
            interrupt=True,
            max_decision_calls=1,
            auto_size_decision_calls=True,
            max_decision_tokens=1000,
        ),
    )
    assert paused.interrupted
    assert paused.metadata["preflight_can_finish"] is False
    assert paused.metadata["auto_size_raised_tokens"] is False
    assert first.calls == 0
    second = _ScriptedSelector()
    resumed = focused_code(
        source,
        **_focused_common(
            saver,
            "budget-only",
            second,
            interrupt=True,
            resuming=True,
            resume={"continue_selection": True},
            allow_changes=False,
            max_decision_calls=1,
            auto_size_decision_calls=True,
            max_decision_tokens=200000,
        ),
    )
    assert resumed.metadata["continuations"][0]["event"] == "budget_resume"
    assert resumed.raised_code_ids == ["c-a", "c-c"]
    assert resumed.metadata["preflight_can_finish"] is True
    assert second.calls >= 1
    assert resumed.metadata["decision_reviews"]["focused_selection"]["chunks_sent"] == len(_selection_requests(second))


def test_validated_chunks_are_reused_when_only_the_allowance_changes(tmp_path):
    saver = InMemorySaver()
    source = _focused_source(_large_codes())
    failing = _ScriptedSelector(fail_at=3)
    paused = focused_code(
        source,
        **_focused_common(
            saver,
            "reuse-chunks",
            failing,
            interrupt=True,
            max_decision_calls=4,
        ),
    )
    assert paused.interrupted
    assert paused.metadata["decision_stop"] == "review_error"
    stored = paused.metadata["decision_chunk_state"]["validated_reviews"]
    assert len(stored) == 2
    for review in stored:
        assert review["input_fingerprint"]
        assert review["prompt_hash"]
        assert review["policy_fingerprint"]
        assert len(review["code_ids"]) == 1
    kept = {tuple(review["code_ids"]) for review in stored}
    healthy = _ScriptedSelector()
    resumed = focused_code(
        source,
        **_focused_common(
            saver,
            "reuse-chunks",
            healthy,
            interrupt=True,
            resuming=True,
            resume={"continue_selection": True},
            allow_changes=False,
            max_decision_calls=8,
        ),
    )
    assert resumed.metadata["continuations"][0]["event"] == "budget_resume"
    resent = [tuple(payload["review_chunk"]["code_ids"]) for payload in _selection_requests(healthy)]
    assert kept.isdisjoint(resent)
    assert len(resent) == 2
    assert resumed.raised_code_ids == ["c-a", "c-c"]
    report = resumed.metadata["decision_reviews"]["focused_selection"]
    assert report["chunks_reused"] >= 2
    assert report["invalidated_reviews"] in ([], None)


def test_a_changed_definition_invalidates_only_that_chunk(tmp_path):
    saver = InMemorySaver()
    source = _focused_source(_large_codes())
    failing = _ScriptedSelector(fail_at=3)
    paused = focused_code(
        source,
        **_focused_common(saver, "invalidate-one", failing, interrupt=True, max_decision_calls=4),
    )
    assert len(paused.metadata["decision_chunk_state"]["validated_reviews"]) == 2

    def rewrite(book):
        for entry in book.values():
            if str(entry.get("code_id") or "") == "c-a":
                definition = str(entry.get("definition") or "")
                entry["definition"] = "packed  " + definition[len("packed  "):]

    _rewrite_channel(saver, "invalidate-one", "codebook", rewrite)
    healthy = _ScriptedSelector()
    resumed = focused_code(
        source,
        **_focused_common(
            saver,
            "invalidate-one",
            healthy,
            interrupt=True,
            resuming=True,
            resume={"continue_selection": True},
            allow_changes=False,
            max_decision_calls=8,
        ),
    )
    report = resumed.metadata["decision_reviews"]["focused_selection"]
    invalidated = {tuple(item["code_ids"]) for item in (report["invalidated_reviews"] or [])}
    assert ("c-a",) in invalidated
    assert ("c-b",) not in invalidated
    resent = [tuple(payload["review_chunk"]["code_ids"]) for payload in _selection_requests(healthy)]
    assert ("c-a",) in resent
    assert ("c-b",) not in resent
    assert resumed.raised_code_ids == ["c-a", "c-c"]


def test_a_changed_policy_invalidates_the_validated_chunks(tmp_path, monkeypatch):
    saver = InMemorySaver()
    source = _focused_source(_large_codes())
    failing = _ScriptedSelector(fail_at=3)
    paused = focused_code(
        source,
        **_focused_common(saver, "invalidate-policy", failing, interrupt=True, max_decision_calls=4),
    )
    kept = {tuple(row["code_ids"]) for row in paused.metadata["decision_chunk_state"]["validated_reviews"]}
    assert kept == {("c-a",), ("c-b",)}
    monkeypatch.setattr(
        "automated_grounded_theory.decision_review.SYSTEM",
        SYSTEM + "\nApply the revised review policy.",
    )
    healthy = _ScriptedSelector()
    resumed = focused_code(
        source,
        **_focused_common(
            saver,
            "invalidate-policy",
            healthy,
            interrupt=True,
            resuming=True,
            resume={"continue_selection": True},
            allow_changes=False,
            max_decision_calls=8,
        ),
    )
    report = resumed.metadata["decision_reviews"]["focused_selection"]
    invalidated = {tuple(item["code_ids"]) for item in (report["invalidated_reviews"] or [])}
    assert invalidated == kept
    resent = {tuple(payload["review_chunk"]["code_ids"]) for payload in _selection_requests(healthy)}
    assert resent == {("c-a",), ("c-b",), ("c-c",), ("c-d",)}


def test_chunk_fingerprint_tracks_definition_quote_and_policy():
    state = _selection_state([_code("c-a", "KEEP one"), _code("c-b", "DROP two")])
    state["segments"] = [{"code_id": "c-a", "passage_id": "p1", "quote": "alpha"}]
    original = chunk_input_fingerprint(state, ["c-a"])
    redefined = _selection_state([_code("c-a", "KEEP one more"), _code("c-b", "DROP two")])
    redefined["segments"] = state["segments"]
    assert chunk_input_fingerprint(redefined, ["c-a"]) != original
    requoted = _selection_state([_code("c-a", "KEEP one"), _code("c-b", "DROP two")])
    requoted["segments"] = [{"code_id": "c-a", "passage_id": "p1", "quote": "beta"}]
    assert chunk_input_fingerprint(requoted, ["c-a"]) != original
    other = _selection_state([_code("c-a", "KEEP one"), _code("c-b", "DROP two changed")])
    other["segments"] = state["segments"]
    assert chunk_input_fingerprint(other, ["c-a"]) == original
    assert selection_policy_fingerprint(chunk_order="id", chunk_order_seed=4) == selection_policy_fingerprint(
        chunk_order="id", chunk_order_seed=9
    )
    assert selection_policy_fingerprint(chunk_order="seeded_shuffle", chunk_order_seed=4) != selection_policy_fingerprint(
        chunk_order="seeded_shuffle", chunk_order_seed=9
    )


def test_seeded_shuffle_is_recorded_and_a_partial_selection_stays_incomplete(tmp_path):
    ids = ["c-a", "c-b", "c-c", "c-d"]
    seed = 1
    shuffled = list(ids)
    while shuffled == ids:
        shuffled = list(ids)
        random.Random(seed).shuffle(shuffled)
        if shuffled == ids:
            seed += 1
    stopped = _focused(
        _large_codes(),
        tmp_path,
        max_decision_calls=1,
        selection_chunk_order="seeded_shuffle",
        selection_chunk_order_seed=seed,
    )
    assert stopped.metadata["selection_chunk_order"] == "seeded_shuffle"
    assert stopped.metadata["selection_chunk_order_seed"] == seed
    assert stopped.metadata["decision_chunk_state"]["ordered_ids"] == shuffled
    assert stopped.metadata["selection_review_complete"] is False
    assert stopped.metadata["partial_selection_is_code_id_slice"] is False
    assert stopped.metadata["partial_selection_bias"] == PARTIAL_SELECTION_SHUFFLE_BIAS
    assert stopped.raised_code_ids == []
    report = stopped.metadata["decision_reviews"]["focused_selection"]
    assert report["partial_selection_bias"] == PARTIAL_SELECTION_SHUFFLE_BIAS
    assert report["chunk_order_selects_codes"] is False
    plain = _focused(_large_codes(), tmp_path / "id-order", max_decision_calls=1)
    assert plain.metadata["decision_chunk_state"]["ordered_ids"] == ids
    assert plain.metadata["partial_selection_is_code_id_slice"] is True
    assert plain.metadata["partial_selection_bias"] == PARTIAL_SELECTION_BIAS
    with pytest.raises(ValueError, match="selection_chunk_order"):
        _focused(_large_codes(), tmp_path / "bad-order", selection_chunk_order="random")


def test_consolidation_is_off_unless_requested_and_reviews_raised_and_near_miss(tmp_path):
    codes = [_code("c-b", "DROP this code"), _code("c-a", "KEEP this code")]
    plain_model = _ScriptedSelector()
    plain = focused_code(
        _focused_source(codes),
        model=FakeAxialCoder(),
        decision_model=plain_model,
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=False,
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_decision_tokens=200000,
        output_dir=tmp_path / "off",
    )
    assert len(_selection_requests(plain_model)) == 1
    assert plain.metadata.get("selection_consolidation") in (None, False)
    assert plain.raised_code_ids == ["c-a"]

    consolidating = _ScriptedSelector(consolidate_to=["c-a"])
    finished = focused_code(
        _focused_source(codes),
        model=FakeAxialCoder(),
        decision_model=consolidating,
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=False,
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_decision_tokens=200000,
        selection_consolidation=True,
        output_dir=tmp_path / "on",
    )
    passes = [payload for payload in consolidating.payloads if payload.get("review_pass") == SELECTION_CONSOLIDATION]
    assert len(passes) == 1
    assert passes[0]["provisional_raised_code_ids"] == ["c-a"]
    assert "c-b" in passes[0]["near_miss_code_ids"]
    assert finished.metadata["selection_consolidation"] == SELECTION_CONSOLIDATION
    assert finished.raised_code_ids == ["c-a"]
    assert finished.metadata["selection_consolidation_enabled"] is True


def test_chunk_context_sensitivity_reports_a_changed_selection():
    state = _selection_state([
        _code("target", "the code under test"),
        _code("friend", "a companion"),
        _code("other", "another companion"),
    ])

    class _CompanionReviewer:
        async def decide(self, payload, schema, metadata):
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            raised = ["target"] if "friend" in ids else []
            return {"raised_code_ids": raised, "rationale": "scripted"}, {"prompt_hash": "hash-" + ",".join(ids)}

    changed = run_sync(
        measure_chunk_context_sensitivity(_CompanionReviewer(), state, "target", [["friend"], ["other"]])
    )
    assert changed["selection_changed"] is True
    assert [row["code_raised"] for row in changed["outcomes"]] == [True, False]
    assert "across the corpus" in changed["chunk_context_note"]
    assert "selection_consolidation" in changed["chunk_context_note"]

    class _StableReviewer:
        async def decide(self, payload, schema, metadata):
            return {"raised_code_ids": ["target"], "rationale": "same"}, {"prompt_hash": "stable"}

    stable = run_sync(
        measure_chunk_context_sensitivity(_StableReviewer(), state, "target", [["friend"], ["other"]])
    )
    assert stable["selection_changed"] is False


def test_every_export_bucket_explains_ineligible_rows():
    exported = export_gate_relationships(
        [
            {
                "relationship_id": "rel-accepted-ineligible",
                "disposition": "accepted",
                "reviewer_disposition": "accepted",
                "downstream_eligible": False,
                "evidence_tier": 2,
                "cases_with_both_endpoints": 2,
                "flags": ["endpoint_unreviewed_by_focus"],
                "eligibility_reason": "endpoint_unreviewed_by_focus",
            },
            {
                "relationship_id": "rel-accepted-ok",
                "disposition": "accepted",
                "reviewer_disposition": "accepted",
                "downstream_eligible": True,
                "evidence_tier": 2,
                "cases_with_both_endpoints": 2,
            },
            {
                "relationship_id": "rel-rejected",
                "disposition": "rejected",
                "reviewer_disposition": "rejected",
                "reviewer_decided": True,
                "downstream_eligible": False,
                "evidence_tier": 2,
                "cases_with_both_endpoints": 2,
            },
            {
                "relationship_id": "rel-0005",
                "disposition": "proposed",
                "reviewer_decided": False,
                "reviewer_disposition": None,
                "evidence_tier": 0,
                "supporting_cases": [],
                "cases_with_both_endpoints": 0,
            },
            {
                "relationship_id": "rel-unreviewed",
                "disposition": "proposed",
                "evidence_tier": 2,
                "cases_with_both_endpoints": 2,
                "flags": ["endpoint_unreviewed_by_focus"],
                "eligibility_reason": "endpoint_unreviewed_by_focus",
            },
            {
                "relationship_id": "rel-open",
                "disposition": "proposed",
                "evidence_tier": 2,
                "cases_with_both_endpoints": 2,
            },
        ]
    )
    buckets = {
        "accepted": {row["relationship_id"] for row in exported["accepted"]},
        "withheld": {row["relationship_id"] for row in exported["withheld"]},
        "proposed": {row["relationship_id"] for row in exported["proposed"]},
    }
    assert "rel-accepted-ineligible" in buckets["accepted"]
    assert "rel-accepted-ok" in buckets["accepted"]
    assert "rel-rejected" in buckets["withheld"]
    assert {"rel-0005", "rel-unreviewed", "rel-open"} <= buckets["proposed"]
    ineligible_ids = {"rel-accepted-ineligible", "rel-rejected", "rel-0005", "rel-unreviewed"}
    open_ids = {"rel-accepted-ok", "rel-open"}
    seen = []
    for name in ("accepted", "withheld", "proposed"):
        assert exported[name]
        for row in exported[name]:
            seen.append(row["relationship_id"])
            if row["relationship_id"] in ineligible_ids:
                assert row["withholding_reasons"], row["relationship_id"]
                assert row["primary_withholding_cause"], row["relationship_id"]
            if row["relationship_id"] in open_ids:
                assert row["withholding_reasons"] == []
                assert row["primary_withholding_cause"] is None
    assert set(seen) == ineligible_ids | open_ids
    accepted = {row["relationship_id"]: row for row in exported["accepted"]}
    held = accepted["rel-accepted-ineligible"]
    assert held["reviewer_disposition"] == "accepted"
    assert held["downstream_eligible"] is False
    assert held["reviewer_disposition"] != str(held["downstream_eligible"]).casefold()
    downstream_ids = {row["relationship_id"] for row in exported["accepted_downstream"]}
    assert downstream_ids == {"rel-accepted-ok"}
    assert "rel-accepted-ineligible" not in downstream_ids
    assert sum(exported["withholding_cause_counts"].values()) == exported["explained_rows"] == len(ineligible_ids)


def test_complete_selection_withholds_endpoints_that_were_not_raised():
    skipped = _tier_two_source(None)
    skipped.update(
        selection_review_complete=True,
        selection_withheld=False,
        selection_status="applied",
        selection_reviewed_code_ids=["tools", "crew"],
        raised_code_ids=["tools"],
    )
    held = axial_code(
        skipped,
        FakeAxialCoder(),
        seed=3,
        interrupt=False,
        max_axial_review_calls=0,
        axial_relations=False,
        batch_size=1,
    )
    row = held.relationships[0]
    assert row["eligibility_reason"] == "endpoint_not_selected_by_focus"
    assert "endpoint_not_selected_by_focus" in row["flags"]
    assert "endpoint_unreviewed_by_focus" not in row["flags"]
    assert row["downstream_eligible"] is False
    assert row["evidence_use"] == "hypothesis_exploration"
    exported = export_gate_relationships([row])
    accepted = exported["accepted"][0]
    assert accepted["primary_withholding_cause"] == "endpoint_not_selected_by_focus"
    assert accepted["reviewer_disposition"] != "ineligible"

    both = _tier_two_source(None)
    both.update(
        selection_review_complete=True,
        selection_withheld=False,
        selection_status="applied",
        selection_reviewed_code_ids=["tools", "crew"],
        raised_code_ids=["tools", "crew"],
    )
    kept = axial_code(
        both,
        FakeAxialCoder(),
        seed=3,
        interrupt=False,
        max_axial_review_calls=0,
        axial_relations=False,
        batch_size=1,
    )
    assert kept.relationships[0]["downstream_eligible"] is True
    assert "endpoint_not_selected_by_focus" not in (kept.relationships[0].get("flags") or [])

    pool = _tier_two_source(None)
    book = dict(pool["codebook"])
    book["other"] = {"code_id": "other", "code": "other work", "definition": "other work"}
    pool["codebook"] = book
    pool.update(
        selection_review_complete=True,
        selection_withheld=False,
        selection_status="applied",
        selection_reviewed_code_ids=["tools", "crew", "other"],
        raised_code_ids=["tools", "crew"],
    )
    planned = axial_code(
        pool,
        FakeAxialCoder(),
        seed=3,
        interrupt=False,
        max_axial_review_calls=0,
        axial_relations=True,
        batch_size=1,
    )
    assert planned.metadata["axial_pair_pool"] == "raised_focused_categories"
    assert planned.metadata["pair_coverage"]["pair_count"] == 1
    assert planned.metadata["axial_pair_pool_excluded_code_ids"] == ["other"]


def test_auto_size_reserves_claim_review_calls(tmp_path):
    finished = _focused(
        [
            _code("c-b", "DROP this code"),
            _code("c-a", "KEEP this code"),
        ],
        tmp_path,
        max_decision_calls=2,
        auto_size_decision_calls=True,
    )
    assert finished.metadata["selection_review_complete"] is True
    assert finished.metadata["claim_review_calls_reserved"] == 2
    assert finished.metadata["decision_call_allowance"] >= 2
    assert finished.metadata.get("decision_stop") != "call_allowance"
    assert "16%" in finished.metadata["token_estimate_scope"]
    assert finished.metadata["token_estimate_is_dollar_bound"] is False
    assert "dollar bound" in finished.metadata["token_estimate_scope"]


def test_focused_review_token_allowance_is_in_the_preflight(tmp_path):
    capped = focused_code(
        _focused_source([_code("c-a", "KEEP this code")]),
        model=FakeAxialCoder(),
        decision_model=_KeepSelector(),
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=False,
        max_focused_review_calls=2,
        max_focused_review_tokens=1,
        max_decision_calls=4,
        max_decision_tokens=200000,
        output_dir=tmp_path,
    )
    assert capped.metadata.get("decision_stop") != CHUNK_ALLOWANCE_INSUFFICIENT
    assert capped.metadata["focused_review_tokens_required_worst_case"] > 1
    assert "every codebook code" in capped.metadata["focused_review_worst_case_note"]
    assert capped.metadata["focused_review_tokens_required"] > 1
    assert capped.metadata["focused_review_tokens_available"] == 1
    assert capped.metadata["focused_review_token_shortfall"] > 0
    assert capped.metadata["focused_review_estimate_basis"] == "raised_codes_and_queue"
    assert capped.metadata["reread_stop"] == "reread_allowance_insufficient"
    assert capped.metadata["decision_reviews"]["focused_selection"]["chunks_sent"] >= 1
    assert capped.raised_code_ids == ["c-a"]
    assert any(entry.get("kind") == "decision_review" for entry in capped.prompt_log)


def test_finished_incomplete_selection_resumes_on_a_larger_budget():
    saver = InMemorySaver()
    source = _focused_source(_large_codes())
    first = _ScriptedSelector()
    finished = focused_code(
        source,
        **_focused_common(
            saver,
            "finished-budget",
            first,
            interrupt=False,
            max_decision_calls=1,
            auto_size_decision_calls=True,
            max_decision_tokens=1000,
        ),
    )
    assert finished.interrupted is False
    assert finished.metadata["decision_stop"] == CHUNK_ALLOWANCE_INSUFFICIENT
    assert first.calls == 0
    second = _ScriptedSelector()
    resumed = focused_code(
        source,
        **_focused_common(
            saver,
            "finished-budget",
            second,
            interrupt=False,
            resuming=True,
            allow_changes=False,
            max_decision_calls=1,
            auto_size_decision_calls=True,
            max_decision_tokens=200000,
        ),
    )
    assert resumed.metadata["continuations"][0]["event"] == "budget_resume"
    assert resumed.raised_code_ids == ["c-a", "c-c"]
    assert second.calls >= 1


def test_finished_incomplete_core_resumes_on_a_larger_budget():
    label = "packing " * 40
    book = {f"c{index:02d}": {"code_id": f"c{index:02d}", "code": label, "definition": label} for index in range(8)}
    passages = [{"passage_id": "p1", "doc_id": "d1", "text": "A short note about packing."}]
    source = {
        "codebook": book,
        "passages": passages,
        "segments": [],
        "relationships": [],
        "metadata": {"stage": "axial"},
    }
    saver = InMemorySaver()
    common = dict(
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        axial_token_budget=10**9,
        batch_size=1,
        checkpointer=saver,
        thread_id="finished-core",
    )
    finished = selective_code(
        source,
        FakeAxialCoder(),
        max_decision_calls=1,
        max_axial_review_tokens=0,
        **common,
    )
    assert finished.interrupted is False
    assert finished.metadata["decision_stop"] == "core_token_allowance_insufficient"
    assert finished.metadata["core_selection_complete"] is False
    assert finished.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 0
    assert finished.metadata["core_candidates_compared"] == 0
    assert finished.metadata["core_candidates_eligible"] == 8
    resumed = selective_code(
        source,
        FakeAxialCoder(),
        max_decision_calls=1,
        max_axial_review_tokens=24000,
        resuming=True,
        allow_changes=False,
        **common,
    )
    report = resumed.metadata["decision_reviews"]["axial_core_review"]
    assert report["complete"] is True
    assert report["chunks_sent"] == 1
    assert report["starting_usage"] == 0
    assert report["calls_made"] == 1
    assert report["remaining_allowance"] == 0
    assert report["budget_stop_reason"] is None
    assert report["prior_step_calls_reduce_allowance"] is False
    assert report["allowance_scope"] == PER_STEP_DECISION_REVIEW
    assert report["shared_with"] == []
    assert resumed.metadata["core_candidates_compared"] == 8
    assert resumed.metadata["core_selection_withheld"] is False
    assert resumed.model_core_id
    with pytest.raises(AxialError, match="not waiting on a checkpoint"):
        selective_code(
            source,
            FakeAxialCoder(),
            max_decision_calls=8,
            resuming=True,
            allow_changes=False,
            **common,
        )
