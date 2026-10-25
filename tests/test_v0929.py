"""0.9.29 review fixes. Fakes and fixtures only. No network calls."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from automated_grounded_theory.async_support import run_sync
from automated_grounded_theory.axial import _commit_relations, consolidate_duplicate_assertions
from automated_grounded_theory.decision_review import (
    SELECTION_CONSOLIDATION,
    SelectionDecision,
    selection_payload,
)
from automated_grounded_theory.decision_review import DecisionReviewer, measure_chunk_context_sensitivity
from automated_grounded_theory.focused import _consolidate_selection_chunks
from automated_grounded_theory.relationship_tiers import apply_evidence_tier
from automated_grounded_theory.repeatability import compare_pipeline_runs
from tests.test_v0924 import _KeepSelector, _code, _selection_state
from tests.test_v0926 import _both, _passages

_CONDITIONS = Path(__file__).resolve().parents[0] / "fixtures" / "conditional_adjudication_0929.json"
_STATEMENT = "tools for the shift may be related to coworkers on the crew"


def _evidence_signature(row: dict) -> dict:
    flags = {
        str(flag)
        for flag in (row.get("flags") or [])
        if str(flag) in {"relationship_qualified", "absence_is_not_contradiction"}
    }
    return {
        "counterexamples": json.dumps(row.get("counterexamples") or [], sort_keys=True),
        "variation_cases": json.dumps(row.get("variation_cases") or [], sort_keys=True),
        "negative_cases": json.dumps(row.get("negative_cases") or [], sort_keys=True),
        "qualification_reason": str(row.get("qualification_reason") or ""),
        "qualification_reasons": tuple(row.get("qualification_reasons") or []),
        "qualification_flags": tuple(sorted(flags)),
        "review_history": json.dumps(row.get("review_history") or [], sort_keys=True),
        "claim_revision_history": json.dumps(row.get("claim_revision_history") or [], sort_keys=True),
    }


def _assertion(relationship_id: str, **extra) -> dict:
    row = {
        "relationship_id": relationship_id,
        "from_id": "tools",
        "to_id": "crew",
        "statement": _STATEMENT,
        "disposition": "accepted",
        "reviewer_disposition": "accepted",
        "evidence_tier": 2,
        "passage_id": "p1",
        "quote": "tools for the shift sit beside coworkers on the crew",
        "case_quotes": [_both("p1", "tools for the shift sit beside coworkers on the crew")],
        "counterexamples": [],
        "variation_cases": [],
        "negative_cases": [],
        "flags": [],
        "review_history": [],
        "qualification_reason": "",
    }
    row.update(extra)
    return row


def test_duplicate_merge_keeps_counterexamples_qualifications_and_review_history():
    """Two accepted tier-2 copies. The higher id carries the counterexample."""
    counter = {
        "passage_id": "p9",
        "judgment": "qualifies",
        "quote": "the night shift does not follow that pattern",
        "valid_test": True,
        "synthetic": False,
        "rationale": "a limiting case",
    }
    variation = {
        "passage_id": "p9",
        "judgment": "qualifies",
        "quote": "the night shift does not follow that pattern",
        "reclassification_reason": "absence_is_not_contradiction",
    }
    history = [{"actor": "human", "disposition": "accepted", "rationale": "kept the qualification"}]
    claim_history = [{"evidence_tier": 2, "note": "qualified before the merge"}]
    bare = _assertion("rel-0001")
    qualified = _assertion(
        "rel-0002",
        counterexamples=[counter],
        variation_cases=[variation],
        negative_cases=[dict(counter, judgment="contradicts")],
        qualification_reason="absence_is_not_contradiction",
        flags=["relationship_qualified", "absence_is_not_contradiction"],
        review_history=history,
        claim_revision_history=claim_history,
    )
    merged = consolidate_duplicate_assertions([bare, qualified])
    assert len(merged) == 1
    row = merged[0]
    assert row["relationship_id"] == "rel-0001"
    assert row["counterexamples"]
    assert row["variation_cases"]
    assert row["qualification_reason"] == "absence_is_not_contradiction"
    assert "relationship_qualified" in row["flags"]
    assert row["review_history"] == history
    assert row["conflicting_evidence"]
    provenance = row["assertion_provenance"]
    assert any(item.get("counterexamples") for item in provenance)
    assert any(item.get("variation_cases") for item in provenance)
    assert any(item.get("qualification_reason") for item in provenance)
    assert any(item.get("qualification_flags") for item in provenance)
    assert any(item.get("review_history") for item in provenance)
    assert any(item.get("claim_revision_history") for item in provenance)

    signatures = set()
    for left_id, right_id in itertools.permutations(("rel-0001", "rel-0002")):
        for order in itertools.permutations((
            _assertion(left_id),
            _assertion(
                right_id,
                counterexamples=[dict(counter)],
                variation_cases=[dict(variation)],
                negative_cases=[dict(counter, judgment="contradicts")],
                qualification_reason="absence_is_not_contradiction",
                flags=["relationship_qualified", "absence_is_not_contradiction"],
                review_history=[dict(history[0])],
                claim_revision_history=[dict(claim_history[0])],
            ),
        )):
            result = consolidate_duplicate_assertions(order)[0]
            signatures.add(json.dumps(_evidence_signature(result), sort_keys=True))
            saved = result["assertion_provenance"]
            assert any(item.get("counterexamples") for item in saved)
            assert result["conflicting_evidence"]
    assert len(signatures) == 1

    passages = _passages("tools for the shift sit beside coworkers on the crew", ["r1", "r2"])
    committed = _commit_relations(
        {
            "metadata": {"procedure": {"min_relation_documents": 2, "coding_lens": "paradigm", "pattern_min_cases": 2}},
            "relationships": [bare, qualified],
            "passages": list(passages.values()),
            "codebook": {
                "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
                "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
            },
            "segments": [],
            "memos": [],
            "warnings": [],
        }
    )["relationships"]
    kept = committed[0]
    assert kept["counterexamples"]
    assert kept["variation_cases"]
    assert kept["qualification_reason"] == "absence_is_not_contradiction"
    assert "relationship_qualified" in kept["flags"]
    assert kept["review_history"]
    assert kept["conflicting_evidence"]
    assert any(item.get("counterexamples") for item in kept["assertion_provenance"])


def _condition_row(case: dict) -> dict:
    from_code = str(case["from_code"])
    to_code = str(case["to_code"])
    quote = str(case["quote"])
    book = {
        "from-end": {"code_id": "from-end", "code": from_code, "definition": from_code},
        "to-end": {"code_id": "to-end", "code": to_code, "definition": to_code},
    }
    row = {
        "relationship_id": "rel-0001",
        "from_id": "from-end",
        "to_id": "to-end",
        "passage_id": "p1",
        "quote": quote,
        "statement": f"{from_code} may be related to {to_code}",
        "disposition": "proposed",
        "flags": [],
        "evidence_condition": "anchored",
        "paradigm_role": "unspecified",
        "case_quotes": [_both("p1", quote), _both("p2", quote)],
        "counterexamples": [
            {
                "passage_id": "p9",
                "judgment": "supports",
                "valid_test": True,
                "synthetic": False,
                "quote": "checked against another shift",
            }
        ],
    }
    apply_evidence_tier(
        row,
        passages=_passages(quote, ["r1", "r2"]),
        codebook=book,
        segments=[],
        memos=[],
    )
    return row


def test_genuine_and_ambiguous_conditions_do_not_authorize_tier_3():
    payload = json.loads(_CONDITIONS.read_text(encoding="utf-8"))
    assert "not study data" in payload["note"].casefold()
    assert set(payload["labels"]) == {
        "conversational_hedge",
        "genuine_condition",
        "ambiguous_condition",
    }
    by_label: dict[str, list] = {}
    for case in payload["cases"]:
        row = _condition_row(case)
        link = row["participant_link"]
        assert link["reading_basis"] == "deterministic_provisional", case["id"]
        assert link["reading_provisional"] is True, case["id"]
        assert link["status"] == case["status"], case["id"]
        assert (link["status"] == "hypothetical") is case["hypothetical"], case["id"]
        assert link["authorizes_tier_3"] is case["authorizes_tier_3"], case["id"]
        if case["evidence_tier_3"]:
            assert row["evidence_tier"] == 3, case["id"]
        else:
            assert row["evidence_tier"] < 3, case["id"]
        if case["label"] != "conversational_hedge":
            assert link["authorizes_tier_3"] is False, case["id"]
        by_label.setdefault(case["label"], []).append(case["id"])
    assert by_label["ambiguous_condition"]
    assert by_label["genuine_condition"]
    assert by_label["conversational_hedge"]


class _RecordingReviewer(DecisionReviewer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.payloads: list[dict] = []

    def refusal_reason(self, payload, schema, metadata):
        self.payloads.append(json.loads(json.dumps(payload, default=str)))
        return super().refusal_reason(payload, schema, metadata)


def _consolidation_estimate(reviewer, state, code_ids: list[str]) -> int:
    payload = selection_payload(state, code_ids, chunk_index=10**6, chunk_count=10**6)
    payload["review_pass"] = SELECTION_CONSOLIDATION
    payload["provisional_raised_code_ids"] = ["c-a"]
    payload["near_miss_code_ids"] = [item for item in code_ids if item != "c-a"]
    return reviewer.estimate(payload, SelectionDecision)


def test_consolidation_does_not_rebuild_the_full_reviewed_set_over_the_ceiling():
    blob = "packing " * 180
    codes = [
        _code("c-a", f"KEEP {blob}"),
        _code("c-b", f"DROP {blob}"),
        _code("c-c", f"DROP {blob}"),
        _code("c-d", f"DROP {blob}"),
    ]
    state = _selection_state(codes)
    probe = DecisionReviewer(_KeepSelector(), max_calls=8, max_tokens=10**9, token_budget=10**9)
    one = _consolidation_estimate(probe, state, ["c-a"])
    pair = _consolidation_estimate(probe, state, ["c-a", "c-b"])
    full = _consolidation_estimate(probe, state, ["c-a", "c-b", "c-c", "c-d"])
    assert pair < full
    budget = pair
    assert one <= budget < full
    reviewer = _RecordingReviewer(
        _KeepSelector(), max_calls=8, max_tokens=10**9, token_budget=budget,
    )
    progress = {
        "raised_ids": ["c-a"],
        "reviewed_ids": ["c-a", "c-b", "c-c", "c-d"],
        "compaction": {},
    }
    metadata = {"decision_calls_used": 0, "decision_tokens_used": 0}
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=reviewer, selection_consolidation=True),
            state,
            progress,
            metadata,
            [],
            [],
        )
    )
    assert reviewer.payloads
    reviewed = {"c-a", "c-b", "c-c", "c-d"}
    for payload in reviewer.payloads:
        sent = {str(row.get("code_id") or "") for row in (payload.get("codes") or [])}
        assert sent != reviewed
        assert sent
        assert reviewer.estimate(payload, SelectionDecision) <= budget
    assert metadata["selection_consolidation_status"] == "completed"
    assert progress.get("consolidation_done") is True
    assert isinstance(metadata["consolidation_calls_reserved"], int)
    assert metadata["consolidation_calls_reserved"] >= 1
    assert metadata["consolidation_request_budget"] == budget
    assert metadata["consolidation_request_budget"] == reviewer.token_budget

    blocked = _RecordingReviewer(
        _KeepSelector(), max_calls=8, max_tokens=10**9, token_budget=1,
    )
    blocked_progress = {
        "raised_ids": ["c-a"],
        "reviewed_ids": ["c-a", "c-b", "c-c", "c-d"],
        "compaction": {},
    }
    blocked_meta = {"decision_calls_used": 0, "decision_tokens_used": 0}
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=blocked, selection_consolidation=True),
            state,
            blocked_progress,
            blocked_meta,
            [],
            [],
        )
    )
    assert blocked_meta["selection_consolidation_status"] == "skipped"
    assert blocked_progress.get("consolidation_done") is not True
    assert blocked_meta["selection_consolidation_skipped"] == "request_budget"
    assert blocked_meta["consolidation_request_budget"] == 1


def test_failed_context_reviews_are_unknown():
    state = _selection_state([
        _code("target", "the code under test"),
        _code("friend", "a companion"),
        _code("other", "another companion"),
    ])

    class _Failing:
        async def decide(self, payload, schema, metadata):
            metadata["decision_stop"] = "review_error"
            return None, {"status": "error", "error": "parse failed", "prompt_hash": "err"}

    failed = run_sync(
        measure_chunk_context_sensitivity(_Failing(), state, "target", [["friend"], ["other"]])
    )
    assert [row["code_raised"] for row in failed["outcomes"]] == [None, None]
    assert [row["status"] for row in failed["outcomes"]] == ["error", "error"]
    assert failed["selection_changed"] is None
    assert failed["reviews_attempted"] == 2
    assert failed["reviews_successful"] == 0

    class _Mixed:
        def __init__(self):
            self.calls = 0

        async def decide(self, payload, schema, metadata):
            self.calls += 1
            if self.calls == 1:
                return {"raised_code_ids": ["target"], "rationale": "ok"}, {"status": "validated", "prompt_hash": "ok"}
            metadata["decision_stop"] = "review_error"
            return None, {"status": "error", "error": "boom", "prompt_hash": "bad"}

    mixed = run_sync(
        measure_chunk_context_sensitivity(_Mixed(), state, "target", [["friend"], ["other"]])
    )
    assert mixed["outcomes"][0]["code_raised"] is True
    assert mixed["outcomes"][1]["code_raised"] is None
    assert mixed["outcomes"][1]["status"] == "error"
    assert mixed["selection_changed"] is None
    assert mixed["reviews_attempted"] == 2
    assert mixed["reviews_successful"] == 1

    class _ValidSplit:
        async def decide(self, payload, schema, metadata):
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            if "boom" in ids:
                metadata["decision_stop"] = "review_error"
                return None, {"status": "error", "error": "skip"}
            raised = ["target"] if "friend" in ids else []
            return {"raised_code_ids": raised, "rationale": "scripted"}, {"status": "validated"}

    compared = run_sync(
        measure_chunk_context_sensitivity(
            _ValidSplit(),
            state,
            "target",
            [["friend"], ["other"], ["boom"]],
        )
    )
    assert compared["reviews_attempted"] == 3
    assert compared["reviews_successful"] == 2
    assert compared["outcomes"][2]["code_raised"] is None
    assert compared["selection_changed"] is True


def test_eligibility_changes_keep_a_matched_flip_when_a_run_adds_a_relationship():
    shared = {
        "from_id": "tools",
        "to_id": "crew",
        "statement": "tools may be related to crew",
    }
    added = {
        "from_id": "tools",
        "to_id": "night",
        "statement": "tools may be related to the night shift",
        "downstream_eligible": True,
    }
    compared = compare_pipeline_runs(
        [
            {"relationships": [{**shared, "downstream_eligible": True}]},
            {"relationships": [{**shared, "downstream_eligible": False}, added]},
        ]
    )
    report = compared["pairwise"][0]["eligibility_changes"]
    assert report is not None
    assert report["matched_changes"]
    assert report["matched_changes"][0]["left"] is True
    assert report["matched_changes"][0]["right"] is False
    assert len(report["additions"]) == 1
    assert report["additions"][0]["from_id"] == "tools"
    assert report["additions"][0]["to_id"] == "night"
    assert report["removals"] == []
    assert report["unknown_eligibility"] == []
    summary = compared["summary"]["eligibility_changes"]
    assert summary["matched_changes"]
    assert summary["additions"]

    unknown = compare_pipeline_runs(
        [
            {"relationships": [{**shared}]},
            {"relationships": [{**shared, "downstream_eligible": True}]},
        ]
    )["pairwise"][0]["eligibility_changes"]
    assert unknown is not None
    assert unknown["matched_changes"] == []
    assert unknown["unknown_eligibility"]
    assert unknown["unknown_eligibility"][0]["left"] is None
    assert unknown["unknown_eligibility"][0]["right"] is True
