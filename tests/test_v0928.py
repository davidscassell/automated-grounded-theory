"""0.9.28 review fixes. Fakes and fixtures only. No network calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    AUTO_SIZE_DECISION_CALLS,
    BUDGET_LEDGER_RULE,
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
    relationship_assessment,
    repeat_frozen_review,
    repeat_pipeline,
    selective_code,
)
from automated_grounded_theory.async_support import ProviderAllowance, run_sync
from automated_grounded_theory.axial import (
    AxialDecisionReviewer,
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
    SHARED_DECISION_REVIEW,
    SHARED_REVIEW_STEPS,
    SelectionDecision,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder
from tests.test_v0924 import _KeepSelector, _focused_source, _rewrite_metadata
from tests.test_v0926 import SENTENCE, _book, _both, _passages, _row
from tests.test_v0927 import _large_codes, _tier_two_source

_ASIDE = Path(__file__).resolve().parents[0] / "fixtures" / "discourse_aside_0928.json"


def test_version_rules_and_unchanged_defaults():
    assert __version__ == "0.9.28"
    assert POLICY_VERSION == "0.9.28"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v5"
    assert CONDITIONAL_SCOPE_RULE == "discourse_aside_v1"
    assert DUPLICATE_ASSERTION_RULE == "merge_identical_statement_v1"
    assert BUDGET_LEDGER_RULE == "shared_review_ledger_v1"
    assert EXPORT_ASSESSMENT_RULE == "shared_assessment_v1"
    assert AUTO_SIZE_DECISION_CALLS is False
    import inspect

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


def test_core_selection_stops_on_the_shared_allowance_remainder():
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
        axial_token_budget=one,
        batch_size=1,
    )
    report = finished.metadata["decision_reviews"]["axial_core_review"]
    assert report["stop_reasons"] == ["call_allowance"]
    assert report["chunks_sent"] == 4
    assert report["starting_usage"] == 4
    assert report["calls_made"] == 4
    assert report["remaining_allowance"] == 0
    assert report["allowance_scope"] == SHARED_DECISION_REVIEW
    assert report["allowance"] == 8
    step = finished.metadata["decision_budget_steps"][-1]
    assert step["step"] == "axial_core_review"
    assert step["starting_usage"] == 4
    assert step["calls_made"] == 4
    assert step["remaining_allowance"] == 0
    assert step["stop_reason"] == "call_allowance"
    assert step["shared_with"] == list(SHARED_REVIEW_STEPS)
    assert finished.metadata["decision_calls_used"] == 8
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
        for key in ("duplicate_assertion_rule", "budget_ledger_rule", "export_assessment_rule"):
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
    assert pair["eligibility_changes"]
    assert compared["summary"]["selected_code_jaccard"] == pytest.approx(0.5)
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
    assert missing["pairwise"][0]["eligibility_changes"] is None
