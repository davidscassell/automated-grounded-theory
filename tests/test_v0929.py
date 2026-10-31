"""0.9.29 review fixes. Fakes and synthetic fixtures only. No network calls."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from automated_grounded_theory import (
    CONDITIONAL_SCOPE_RULE,
    DECISION_REVIEW_CHUNKING,
    DUPLICATE_ASSERTION_RULE,
    compare_pipeline_runs,
    focused_code,
    measure_chunk_context_sensitivity,
    score_conditional_readings,
)
from automated_grounded_theory.async_support import run_sync
from automated_grounded_theory.axial import _commit_relations
from automated_grounded_theory.decision_review import (
    CONSOLIDATION_NEAR_MISS_CAP,
    SELECTION_CONSOLIDATION,
    consolidation_allowance_reserve,
    rank_near_misses,
    review_procedure_fields,
    same_chunking_fingerprint,
)
from automated_grounded_theory.focused import _consolidate_selection_chunks
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import _distinctive, _interpret_quote, apply_evidence_tier
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder
from tests.test_v0924 import _code, _focused_source, _selection_state
from tests.test_v0926 import SENTENCE, _book, _both, _passages
from tests.test_v0928 import _ScriptedSelector

_HEDGES = Path(__file__).resolve().parents[0] / "fixtures" / "conditional_hedges_0929.json"
_STATEMENT = "tools for the shift may be related to coworkers on the crew"


def test_version_and_fingerprint():
    assert __version__ == "0.9.39"
    assert POLICY_VERSION == "0.9.39"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v6"
    assert CONDITIONAL_SCOPE_RULE == "condition_scope_v1"
    assert DUPLICATE_ASSERTION_RULE == "merge_all_evidence_v1"
    current = review_procedure_fields()
    assert same_chunking_fingerprint(current, current) is True
    older = dict(current)
    older["decision_review_chunking"] = "request_budget_v5"
    older["conditional_scope_rule"] = "discourse_aside_v1"
    older["duplicate_assertion_rule"] = "merge_identical_statement_v1"
    assert same_chunking_fingerprint(older, current) is False


def _evidence_row(relationship_id: str, *, extra: dict | None = None) -> dict:
    row = {
        "relationship_id": relationship_id,
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": SENTENCE,
        "statement": _STATEMENT,
        "disposition": "accepted",
        "reviewer_disposition": "accepted",
        "evidence_tier": 2,
        "case_quotes": [_both("p1", SENTENCE), _both("p2", SENTENCE)],
        "counterexamples": [],
        "variation_cases": [],
        "negative_cases": [],
        "qualifications": [],
        "qualification_reason": None,
        "review_history": [],
        "conflicting_evidence": None,
    }
    if extra:
        row.update(extra)
    return row


def _qualified_extra() -> dict:
    return {
        "counterexamples": [
            {
                "passage_id": "p9",
                "quote": "the shift had no tools and the crew still met",
                "judgment": "contradicts",
                "valid_test": False,
            }
        ],
        "variation_cases": [{"passage_id": "p8", "quote": "a night shift kept the same crew"}],
        "negative_cases": [{"passage_id": "p7", "quote": "no tools were on the bench"}],
        "qualifications": ["shift_length"],
        "qualification_reason": "only when the shift is short",
        "review_history": [{"actor": "human", "note": "qualified the second row"}],
        "conflicting_evidence": {"source": "second-row"},
    }


def _substance(row: dict) -> dict:
    def blobs(key: str) -> list[str]:
        return sorted(json.dumps(item, sort_keys=True, default=str) for item in (row.get(key) or []))

    history = sorted(
        json.dumps(item, sort_keys=True, default=str) for item in (row.get("review_history") or [])
    )
    provenance_ids = sorted(
        str(item.get("relationship_id") or "") for item in (row.get("assertion_provenance") or [])
    )
    return {
        "evidence_tier": row.get("evidence_tier"),
        "counterexamples": blobs("counterexamples"),
        "variation_cases": blobs("variation_cases"),
        "negative_cases": blobs("negative_cases"),
        "qualifications": list(row.get("qualifications") or []),
        "qualification_reason": row.get("qualification_reason"),
        "review_history": history,
        "provenance_ids": provenance_ids,
        "conflict_ids": provenance_ids if row.get("conflicting_evidence") else [],
    }


def _commit(rows: list[dict]) -> dict:
    passages = _passages(SENTENCE, ["r1", "r2"])
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
    )
    found = [row for row in committed["relationships"] if row["statement"] == _STATEMENT and row["from_id"] == "tools"]
    assert len(found) == 1
    return found[0]


def test_duplicate_merge_keeps_every_rows_evidence_before_reassessment():
    plain = _evidence_row("rel-0001", extra={"review_history": [{"actor": "human", "note": "accepted the first row"}]})
    qualified = _evidence_row("rel-0002", extra=_qualified_extra())
    forward = _commit([plain, qualified])
    assert forward["relationship_id"] == "rel-0001"
    assert forward["counterexamples"]
    assert forward["counterexamples"][0]["passage_id"] == "p9"
    assert forward["variation_cases"]
    assert forward["negative_cases"]
    assert forward["qualifications"] == ["shift_length"]
    assert forward["qualification_reason"] == "only when the shift is short"
    assert "relationship_qualified" in set(forward.get("flags") or [])
    notes = {item.get("note") for item in forward["review_history"]}
    assert notes == {"accepted the first row", "qualified the second row"}
    assert {item["relationship_id"] for item in forward["assertion_provenance"]} == {"rel-0001", "rel-0002"}
    assert forward["conflicting_evidence"]
    assert {item["relationship_id"] for item in forward["conflicting_evidence"]} == {"rel-0001", "rel-0002"}
    for item in forward["assertion_provenance"]:
        if item["relationship_id"] == "rel-0002":
            assert item["counterexamples"]
            assert item["qualification_reason"] == "only when the shift is short"
            assert item["review_history"]

    reordered = _commit([qualified, plain])
    swapped_plain = _evidence_row("rel-0002", extra={"review_history": [{"actor": "human", "note": "accepted the first row"}]})
    swapped_qualified = _evidence_row("rel-0001", extra=_qualified_extra())
    swapped = _commit([swapped_plain, swapped_qualified])
    assert _substance(forward) == _substance(reordered) == _substance(swapped)
    assert forward["evidence_tier"] == reordered["evidence_tier"] == swapped["evidence_tier"]


def test_closed_hedges_do_not_authorize_a_genuine_condition():
    payload = json.loads(_HEDGES.read_text(encoding="utf-8"))
    note = payload["note"].casefold()
    assert "not study data" in note
    assert "not adjudication" in note
    assert "not selection validity" in note
    empty = score_conditional_readings(None)
    assert empty["accuracy"] is None
    assert empty["correct_count"] is None
    assert empty["adjudication_data_included"] is False
    assert empty["selection_validity"] is False
    assert "not selection validity" in empty["note"]
    assert "Human-adjudicated examples are not included" in empty["note"]
    scored = score_conditional_readings(payload["cases"])
    assert scored["denominator"] == len(payload["cases"])
    assert scored["accuracy"] == 1
    assert scored["selection_validity"] is False
    assert scored["adjudication_data_included"] is False
    assert "processing coverage" in scored["note"]
    assert "not adjudication" in scored["note"]
    for case in payload["cases"]:
        reading = _interpret_quote(
            case["quote"],
            _distinctive(" ".join(case["from_terms"])),
            _distinctive(" ".join(case["to_terms"])),
        )
        assert reading["status"] == case["expected_status"], case["id"]
        assert reading["authorizes_tier_3"] is case["authorizes_tier_3"], case["id"]
        assert reading["reading_provisional"] is True
        if case["expected_status"] != "validated":
            assert reading["authorizes_tier_3"] is False

    def tier_for(quote: str) -> dict:
        passages = _passages(quote, ["r1", "r2"])
        row = {
            "relationship_id": "rel-cond",
            "from_id": "tools",
            "to_id": "crew",
            "passage_id": "p1",
            "quote": quote,
            "statement": "tools for the shift lead to coworkers on the crew",
            "disposition": "proposed",
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
        apply_evidence_tier(row, passages=passages, codebook=_book(), segments=[], memos=[])
        return row

    genuine = tier_for("If you help, tools for the shift lead to coworkers on the crew")
    assert genuine["participant_link"]["status"] == "hypothetical"
    assert genuine["participant_link"]["authorizes_tier_3"] is False
    assert genuine["participant_link"]["reading_provisional"] is True
    assert genuine["evidence_tier"] != 3
    hedge = tier_for("tools for the shift lead to coworkers on the crew, if that makes sense")
    assert hedge["participant_link"]["status"] == "validated"
    assert hedge["participant_link"]["authorizes_tier_3"] is True
    assert hedge["participant_link"]["reading_provisional"] is True
    assert hedge["evidence_tier"] == 3


def test_near_misses_are_capped_and_consolidation_status_is_not_a_bool(tmp_path):
    reviewed = ["raise"] + [f"n{index:02d}" for index in range(12)] + ["alone"]
    chunks = [["raise", *reviewed[1:11]], ["alone", "n10", "n11"]]
    near, omitted = rank_near_misses(chunks, ["raise"], reviewed, cap=CONSOLIDATION_NEAR_MISS_CAP)
    assert near == [f"n{index:02d}" for index in range(8)]
    assert omitted == ["n08", "n09"]
    assert "alone" not in near
    assert "n10" not in near and "n11" not in near
    assert len(near) == CONSOLIDATION_NEAR_MISS_CAP

    codes = [_code("c-keep", "KEEP this code")] + [_code(f"c-{index:02d}", "DROP this code") for index in range(11)]
    model = _ScriptedSelector(consolidate_to=["c-keep"])
    finished = focused_code(
        _focused_source(codes),
        model=FakeAxialCoder(),
        decision_model=model,
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
        output_dir=tmp_path / "capped",
    )
    assert finished.metadata["selection_consolidation_status"] == "completed"
    assert finished.metadata["selection_consolidation_reason"] is None
    assert finished.metadata["selection_consolidation"] == SELECTION_CONSOLIDATION
    assert "consolidation_done" not in finished.metadata
    assert len(finished.metadata["selection_near_miss_code_ids"]) == CONSOLIDATION_NEAR_MISS_CAP
    assert finished.metadata["selection_near_miss_omitted_code_ids"]
    assert finished.metadata["consolidation_calls_reserved"] >= 1
    assert finished.metadata["consolidation_tokens_reserved"] > 0
    assert finished.metadata["consolidation_reserve_separate"] is True
    allowance = int(finished.metadata["decision_call_allowance"])
    assert finished.metadata["calls_available"] == allowance
    assert finished.raised_code_ids == ["c-keep"]

    off = focused_code(
        _focused_source(codes[:2]),
        model=FakeAxialCoder(),
        decision_model=_ScriptedSelector(),
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
    assert off.metadata["selection_consolidation_status"] is None
    assert off.metadata["consolidation_calls_reserved"] == 0
    assert off.metadata["consolidation_tokens_reserved"] == 0
    assert off.metadata.get("selection_consolidation") in (None, False)


def test_codes_outside_a_raised_chunk_are_not_near_misses_and_a_split_can_be_attempted(tmp_path):
    codes = [
        _code("c-0", "KEEP this code"),
        _code("c-1", "DROP beside the raise"),
        _code("c-2", "DROP in the other chunk"),
        _code("c-3", "DROP in the other chunk too"),
    ]
    state = _selection_state(codes)
    # 1000 input tokens fits these two short codes and does not fit three.
    # A code reviewed only in the second chunk is not a near miss.
    model = _ScriptedSelector(consolidate_to=["c-0"])
    finished = focused_code(
        _focused_source(codes),
        model=FakeAxialCoder(),
        decision_model=model,
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=False,
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_decision_tokens=200000,
        decision_token_budget=1000,
        selection_consolidation=True,
        output_dir=tmp_path / "narrow",
    )
    assert finished.metadata["selection_near_miss_code_ids"] == ["c-1"]
    assert "c-2" not in finished.metadata["selection_near_miss_code_ids"]
    assert "c-3" not in finished.metadata["selection_near_miss_code_ids"]
    assert finished.metadata["selection_review_complete"] is True
    assert finished.raised_code_ids == ["c-0"]

    class _OneAtATime:
        token_budget = 50
        max_calls = 8
        max_tokens = 1000

        def estimate(self, payload, schema):
            count = len(payload.get("codes") or [])
            return 40 if count <= 1 else 90

        def refusal_reason(self, payload, schema, metadata):
            if self.estimate(payload, schema) > self.token_budget:
                return "request_budget"
            return None

        async def decide(self, payload, schema, metadata):
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            return {"raised_code_ids": ids, "rationale": "kept this request"}, {"status": "validated"}

    progress = {
        "raised_ids": ["c-0", "c-1"],
        "reviewed_ids": ["c-0", "c-1"],
        "chunks": [["c-0"], ["c-1"]],
        "compaction": {},
    }
    metadata = {"consolidation_calls_reserved": 4}
    warnings: list[str] = []
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=_OneAtATime()),
            state,
            progress,
            metadata,
            [],
            warnings,
        )
    )
    assert metadata["selection_consolidation_status"] == "unresolved-synthesis"
    assert metadata["selection_consolidation_reason"] == "synthesis_cannot_fit"
    assert metadata["selection_consolidation_error"] is None
    assert metadata["selection_consolidation"] is None
    assert metadata.get("selection_consolidation_skipped") is None
    assert progress["consolidation_closed"] is False
    assert "consolidation_done" not in metadata
    assert progress["raised_ids"] == ["c-0", "c-1"]
    assert progress["consolidation_calls_sent"] >= 1
    assert len(progress["consolidation_validated_groups"]) == 2
    assert metadata.get("decision_stop") is None

    class _NeverFits:
        token_budget = 10
        max_calls = 4
        max_tokens = 100

        def estimate(self, payload, schema):
            return 100

        def refusal_reason(self, payload, schema, metadata):
            return "request_budget"

        async def decide(self, payload, schema, metadata):
            raise AssertionError("an oversized consolidation request must not be sent")

    skipped_progress = {
        "raised_ids": ["c-0", "c-1"],
        "reviewed_ids": ["c-0", "c-1"],
        "chunks": [["c-0", "c-1"]],
        "compaction": {},
    }
    skipped_meta: dict = {}
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=_NeverFits()),
            state,
            skipped_progress,
            skipped_meta,
            [],
            [],
        )
    )
    assert skipped_meta["selection_consolidation_status"] == "skipped"
    assert skipped_meta["selection_consolidation_reason"] == "request_budget"
    assert skipped_meta["selection_consolidation"] is None
    assert skipped_progress["consolidation_closed"] is True
    assert skipped_progress["raised_ids"] == ["c-0", "c-1"]

    lone = focused_code(
        _focused_source([_code("c-0", "KEEP this code")]),
        model=FakeAxialCoder(),
        decision_model=_ScriptedSelector(),
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
        output_dir=tmp_path / "lone",
    )
    assert lone.metadata["selection_consolidation_status"] == "skipped"
    assert lone.metadata["selection_consolidation_reason"] == "nothing_to_consolidate"
    assert lone.metadata["selection_consolidation"] is None
    assert lone.metadata["selection_review_complete"] is True
    assert lone.raised_code_ids == ["c-0"]

    def estimate(ids):
        return 10 * len(list(ids))

    reserve = consolidation_allowance_reserve(["a", "b", "c"], estimate, per_request=25)
    assert reserve["consolidation_calls_reserved"] >= 1
    assert reserve["consolidation_tokens_reserved"] > 0


def test_sensitivity_counts_failures_as_unknown():
    state = _selection_state([
        _code("target", "the code under test"),
        _code("friend", "a companion"),
        _code("other", "another companion"),
    ])

    class _Fails:
        async def decide(self, payload, schema, metadata):
            raise RuntimeError("scripted review failure")

    failed = run_sync(measure_chunk_context_sensitivity(_Fails(), state, "target", [["friend"], ["other"]]))
    assert failed["attempted_reviews"] == 2
    assert failed["successful_reviews"] == 0
    assert failed["selection_changed"] is None
    assert [row["code_raised"] for row in failed["outcomes"]] == [None, None]
    assert [row["status"] for row in failed["outcomes"]] == ["error", "error"]
    assert all(row["error"] for row in failed["outcomes"])

    class _Unparsed:
        async def decide(self, payload, schema, metadata):
            return None, {"status": "unparsed"}

    missing = run_sync(measure_chunk_context_sensitivity(_Unparsed(), state, "target", [["friend"], ["other"]]))
    assert missing["successful_reviews"] == 0
    assert missing["selection_changed"] is None
    assert [row["code_raised"] for row in missing["outcomes"]] == [None, None]
    assert [row["status"] for row in missing["outcomes"]] == ["unparsed", "unparsed"]

    class _ThenFails:
        def __init__(self):
            self.calls = 0

        async def decide(self, payload, schema, metadata):
            self.calls += 1
            if self.calls == 1:
                return {"raised_code_ids": ["target"], "rationale": "kept"}, {"status": "validated"}
            raise RuntimeError("second review failed")

    mixed = run_sync(measure_chunk_context_sensitivity(_ThenFails(), state, "target", [["friend"], ["other"]]))
    assert mixed["attempted_reviews"] == 2
    assert mixed["successful_reviews"] == 1
    assert mixed["selection_changed"] is None
    assert mixed["outcomes"][0]["code_raised"] is True
    assert mixed["outcomes"][1]["code_raised"] is None
    assert mixed["outcomes"][1]["status"] == "error"


def test_eligibility_reports_a_flip_when_the_second_run_adds_a_row():
    shared = "tools may be related to crew"
    added = "crew may be related to the return"
    compared = compare_pipeline_runs(
        [
            {
                "raised_code_ids": ["c-a"],
                "relationships": [
                    {
                        "from_id": "tools",
                        "to_id": "crew",
                        "statement": shared,
                        "downstream_eligible": True,
                    }
                ],
            },
            {
                "raised_code_ids": ["c-a"],
                "relationships": [
                    {
                        "from_id": "tools",
                        "to_id": "crew",
                        "statement": shared,
                        "downstream_eligible": False,
                    },
                    {
                        "from_id": "crew",
                        "to_id": "return",
                        "statement": added,
                        "downstream_eligible": True,
                    },
                ],
            },
        ]
    )
    report = compared["pairwise"][0]["eligibility_changes"]
    assert report is not None
    assert len(report["matched"]) == 1
    assert report["matched"][0]["left"] is True
    assert report["matched"][0]["right"] is False
    assert len(report["additions"]) == 1
    assert report["additions"][0]["right"] is True
    assert report["removals"] == []
    assert report["unknown_eligibility"] == []
    assert compared["summary"]["eligibility_changes"] == report["matched"]
