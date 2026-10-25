"""0.9.29 review fixes. Fakes and synthetic fixtures only. No network calls."""

from __future__ import annotations

import itertools
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
from automated_grounded_theory.decision_review import DecisionReviewer
from automated_grounded_theory.focused import _consolidate_selection_chunks
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import _distinctive, _interpret_quote, apply_evidence_tier
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder
from tests.test_v0924 import _KeepSelector, _code, _focused_source, _selection_state
from tests.test_v0926 import SENTENCE, _book, _both, _passages
from tests.test_v0928 import _ScriptedSelector

_HEDGES = Path(__file__).resolve().parents[0] / "fixtures" / "conditional_hedges_0929.json"
_HOLDOUT = Path(__file__).resolve().parents[0] / "fixtures" / "false_promotion_holdout_0929.json"
_STATEMENT = "tools for the shift may be related to coworkers on the crew"


def test_version_and_fingerprint():
    assert __version__ == "0.9.29"
    assert POLICY_VERSION == "0.9.29"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v6"
    assert CONDITIONAL_SCOPE_RULE == "ability_permission_preserved_v1"
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
    allowance = int(finished.metadata["decision_call_allowance"])
    assert finished.metadata["calls_available"] == allowance - int(finished.metadata["consolidation_calls_reserved"])
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
    assert metadata["selection_consolidation_status"] == "attempted"
    assert metadata["selection_consolidation_reason"] == "request_budget"
    assert metadata["selection_consolidation"] is None
    assert "consolidation_done" not in metadata
    assert progress["raised_ids"] == ["c-0", "c-1"]
    assert progress["consolidation_calls_sent"] >= 1
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
    summary = compared["summary"]["eligibility_changes"]
    assert summary["matched"] == report["matched"]
    assert summary["additions"]
    assert summary["comparisons_attempted"] == 1
    assert summary["comparisons_available"] == 1
    assert summary["comparisons_unavailable"] == 0


def _tier_for(quote: str) -> dict:
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


def test_ability_and_permission_conditions_do_not_authorize_tier_3():
    """Minimal pairs, plus natural wording that is not the rule's phrase list."""
    from automated_grounded_theory.relationship_tiers import _DISCOURSE_ASIDES

    payload = json.loads(_HEDGES.read_text(encoding="utf-8"))
    for case_id in ("hedge-if-i-can", "hedge-if-possible"):
        labelled = next(case for case in payload["cases"] if case["id"] == case_id)
        assert labelled["expected_status"] == "hypothetical", case_id
        assert labelled["authorizes_tier_3"] is False, case_id

    clear = _tier_for("tools for the shift lead to coworkers on the crew")
    ability = _tier_for("tools for the shift lead to coworkers on the crew, if you can.")
    exact = _tier_for("tools for the shift lead to coworkers on the crew, if I can")
    possible = _tier_for("tools for the shift lead to coworkers on the crew, if possible")
    assert clear["participant_link"]["authorizes_tier_3"] is True
    assert clear["evidence_tier"] == 3
    assert clear["rule_authorized_attribution"] is True
    assert clear["human_validated_attribution"] is False
    assert clear["attribution_source"] == "participant"
    for row in (ability, exact, possible):
        link = row["participant_link"]
        assert link["status"] == "hypothetical", row["quote"]
        assert link["conditional_scope"] == "preserved", row["quote"]
        assert link["authorizes_tier_3"] is False, row["quote"]
        assert row["evidence_tier"] != 3, row["quote"]
        assert row["rule_authorized_attribution"] is False, row["quote"]
        assert row["human_validated_attribution"] is False, row["quote"]

    natural = [
        "If the trainee is allowed to practice on the line, tools for the shift lead to coworkers on the crew.",
        "tools for the shift lead to coworkers on the crew if she could ask one question during the drill.",
        "tools for the shift lead to coworkers on the crew, if you are able to stay for the demonstration.",
        "If we may stop the drill early, tools for the shift lead to coworkers on the crew.",
        "tools for the shift lead to coworkers on the crew if the night crew might follow the new steps.",
    ]
    for quote in natural:
        folded = " ".join(quote.casefold().split())
        aside_hits = [phrase for phrase in _DISCOURSE_ASIDES if phrase in folded]
        assert aside_hits == [], quote
        row = _tier_for(quote)
        assert row["participant_link"]["authorizes_tier_3"] is False, quote
        assert row["evidence_tier"] != 3, quote
        assert row["rule_authorized_attribution"] is False, quote
        assert row["human_validated_attribution"] is False, quote

    marked = _tier_for("tools for the shift lead to coworkers on the crew")
    marked["human_validated_attribution"] = True
    apply_evidence_tier(
        marked,
        passages=_passages(marked["quote"], ["r1", "r2"]),
        codebook=_book(),
        segments=[],
        memos=[],
    )
    assert marked["rule_authorized_attribution"] is True
    assert marked["human_validated_attribution"] is True


def test_false_promotions_are_measured_on_a_held_out_set():
    """Tier 3 authorized against an adjudicated label that says it should not be."""
    from automated_grounded_theory.relationship_tiers import _DISCOURSE_ASIDES

    payload = json.loads(_HOLDOUT.read_text(encoding="utf-8"))
    assert "not a study participant" in payload["adjudicator"].casefold()
    assert "live model" in payload["limits"].casefold()
    withheld = [case for case in payload["cases"] if case["should_authorize_tier_3"] is False]
    assert len(withheld) == 8
    false_promotions = []
    for case in withheld:
        folded = " ".join(str(case["quote"]).casefold().split()).strip(" ,;.")
        assert folded not in _DISCOURSE_ASIDES, case["id"]
        row = _tier_for(case["quote"])
        authorized = bool(row["participant_link"]["authorizes_tier_3"]) or row["evidence_tier"] == 3
        if authorized:
            false_promotions.append(case["id"])
    denominator = len(withheld)
    rate = len(false_promotions) / denominator
    assert rate == 0
    assert {
        "false_promotions": len(false_promotions),
        "denominator": denominator,
        "rate": rate,
        "adjudicator": payload["adjudicator"],
    }["denominator"] == 8


class _FailingConsolidationReviewer(DecisionReviewer):
    """Records one decision call, then returns an error. No network."""

    async def decide(self, payload, schema, metadata):
        metadata["decision_calls_used"] = int(metadata.get("decision_calls_used") or 0) + 1
        metadata["decision_stop"] = "review_error"
        return None, {"status": "error", "error": "boom"}


def test_failed_consolidation_is_not_skipped_and_can_resume():
    """A request that errors is failed, not skipped, and does not close the pass."""
    codes = [_code("c-a", "KEEP this"), _code("c-b", "DROP that")]
    state = _selection_state(codes)
    reviewer = _FailingConsolidationReviewer(
        _KeepSelector(), max_calls=8, max_tokens=10**9, token_budget=10**9,
    )
    progress = {
        "raised_ids": ["c-a"],
        "reviewed_ids": ["c-a", "c-b"],
        "chunks": [["c-a", "c-b"]],
        "compaction": {},
    }
    metadata = {"decision_calls_used": 0, "decision_tokens_used": 0}
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=reviewer),
            state,
            progress,
            metadata,
            [],
            [],
        )
    )
    assert metadata["decision_calls_used"] == 1
    assert metadata["selection_consolidation_status"] == "failed"
    assert metadata["consolidation_requests_attempted"] == 1
    assert metadata["consolidation_responses_completed"] == 0
    assert progress.get("consolidation_closed") is not True
    assert progress["raised_ids"] == ["c-a"]
    resume = progress["consolidation_resume"]
    assert resume["available"] is True
    assert resume["requests_attempted"] == 1
    assert resume["responses_completed"] == 0

    assert not progress.get("consolidation_closed")
    working = DecisionReviewer(_KeepSelector(), max_calls=8, max_tokens=10**9, token_budget=10**9)
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=working),
            state,
            progress,
            metadata,
            [],
            [],
        )
    )
    assert metadata["selection_consolidation_status"] == "completed"
    assert progress.get("consolidation_closed") is True
    assert "c-a" in progress["raised_ids"]
    assert metadata["consolidation_responses_completed"] >= 1


def test_consolidation_allowance_preserves_claim_review_calls():
    """Multi-call consolidation does not start when it would spend claim-review calls."""
    codes = [
        _code("c-a", "KEEP this"),
        _code("c-b", "DROP that"),
        _code("c-c", "DROP the other"),
        _code("c-d", "DROP the rest"),
    ]
    state = _selection_state(codes)
    claim_reserve = 2
    spent_by_selection = 1

    class _MultiBatch:
        def __init__(self, max_calls: int):
            self.max_calls = max_calls
            self.max_tokens = 10**9
            self.token_budget = 50
            self.decide_calls = 0

        def estimate(self, payload, schema):
            count = len(payload.get("codes") or [])
            return 40 if count <= 1 else 90

        def refusal_reason(self, payload, schema, metadata):
            if int(metadata.get("decision_calls_used") or 0) >= self.max_calls:
                return "call_allowance"
            if self.estimate(payload, schema) > self.token_budget:
                return "request_budget"
            return None

        async def decide(self, payload, schema, metadata):
            self.decide_calls += 1
            metadata["decision_calls_used"] = int(metadata.get("decision_calls_used") or 0) + 1
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            raised = [item for item in ids if item == "c-a"]
            return {"raised_code_ids": raised, "rationale": "kept"}, {"status": "validated"}

    def progress_for() -> dict:
        return {
            "raised_ids": ["c-a"],
            "reviewed_ids": ["c-a", "c-b", "c-c", "c-d"],
            "chunks": [["c-a", "c-b", "c-c", "c-d"]],
            "compaction": {},
        }

    blocked = _MultiBatch(spent_by_selection + claim_reserve + 1)
    blocked_progress = progress_for()
    blocked_meta = {
        "decision_calls_used": spent_by_selection,
        "decision_tokens_used": 0,
        "claim_review_calls_reserved": claim_reserve,
    }
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=blocked),
            state,
            blocked_progress,
            blocked_meta,
            [],
            [],
        )
    )
    assert blocked.decide_calls == 0
    assert blocked_meta["decision_calls_used"] == spent_by_selection
    assert blocked_meta["consolidation_completion_guaranteed"] is False
    assert blocked_meta["consolidation_calls_needed"] > blocked_meta["consolidation_call_allowance"]
    assert blocked_meta["consolidation_batches_required"] > 1
    assert blocked_meta["consolidation_synthesis_call_budget"] >= 1
    assert blocked_meta["claim_review_calls_reserved"] == claim_reserve
    assert blocked.max_calls - blocked_meta["decision_calls_used"] >= claim_reserve
    assert blocked_meta["selection_consolidation_status"] == "skipped"
    assert blocked_meta["selection_consolidation_reason"] == "completion_not_guaranteed"
    assert blocked_progress["raised_ids"] == ["c-a"]

    covered = _MultiBatch(30)
    covered_progress = progress_for()
    covered_meta = {
        "decision_calls_used": spent_by_selection,
        "decision_tokens_used": 0,
        "claim_review_calls_reserved": claim_reserve,
    }
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=covered),
            state,
            covered_progress,
            covered_meta,
            [],
            [],
        )
    )
    assert covered_meta["consolidation_completion_guaranteed"] is True
    assert covered_meta["selection_consolidation_status"] == "completed"
    spent = covered_meta["decision_calls_used"] - spent_by_selection
    assert spent <= covered_meta["consolidation_call_allowance"]
    assert spent <= covered_meta["consolidation_calls_needed"]
    assert covered.max_calls - covered_meta["decision_calls_used"] >= claim_reserve
    assert covered_meta["claim_review_calls_reserved"] == claim_reserve
    assert covered.decide_calls == spent


def test_conflicting_eligibility_duplicates_are_order_invariant():
    """Conflicting eligibility on one statement key is reported, and row order does not matter."""
    base = {
        "from_id": "tools",
        "to_id": "crew",
        "statement": "tools may be related to crew",
    }
    true_row = {**base, "downstream_eligible": True}
    false_row = {**base, "downstream_eligible": False}

    def report_for(rows: list[dict], *, duplicates_on_left: bool) -> str:
        conflicted = {"relationships": rows}
        single = {"relationships": [true_row]}
        runs = [conflicted, single] if duplicates_on_left else [single, conflicted]
        report = compare_pipeline_runs(runs)["pairwise"][0]["eligibility_changes"]
        assert report["conflicting_duplicates"]
        assert report["matched"] == []
        return json.dumps(report, sort_keys=True)

    for duplicates_on_left in (True, False):
        forms = {
            report_for(list(rows), duplicates_on_left=duplicates_on_left)
            for rows in itertools.permutations((true_row, false_row))
        }
        assert len(forms) == 1

    one_sided = []
    for rows in itertools.permutations((true_row, false_row)):
        report = compare_pipeline_runs(
            [
                {"relationships": []},
                {"relationships": list(rows)},
            ]
        )["pairwise"][0]["eligibility_changes"]
        assert report["conflicting_duplicates"]
        assert report["additions"] == []
        assert report["matched"] == []
        one_sided.append(json.dumps(report, sort_keys=True))
    assert len(set(one_sided)) == 1

    clean = compare_pipeline_runs(
        [
            {"relationships": [true_row]},
            {"relationships": [false_row]},
        ]
    )["pairwise"][0]["eligibility_changes"]
    assert clean["matched"]
    assert clean["matched"][0]["left"] is True
    assert clean["matched"][0]["right"] is False
    assert clean["conflicting_duplicates"] == []


def test_eligibility_summary_keeps_valid_comparisons_when_a_run_has_no_relationships():
    """A run with no relationship list is an unavailable comparison. It does not erase a flip."""
    shared = {
        "from_id": "tools",
        "to_id": "crew",
        "statement": "tools may be related to crew",
    }
    runs = [
        {"relationships": [{**shared, "downstream_eligible": True}], "raised_code_ids": ["a"]},
        {"relationships": [{**shared, "downstream_eligible": False}], "raised_code_ids": ["a"]},
        {"raised_code_ids": ["b"]},
    ]
    signatures = []
    for order in itertools.permutations(runs):
        compared = compare_pipeline_runs(order)
        unavailable = [item for item in compared["pairwise"] if item["eligibility_changes"] is None]
        available = [item for item in compared["pairwise"] if item["eligibility_changes"] is not None]
        assert len(available) == 1
        assert len(unavailable) == 2
        assert available[0]["eligibility_changes"]["matched"]
        summary = compared["summary"]["eligibility_changes"]
        assert summary is not None
        assert summary["matched"]
        signatures.append(
            (
                len(summary["matched"]),
                summary["comparisons_attempted"],
                summary["comparisons_available"],
                summary["comparisons_unavailable"],
            )
        )
    assert len(set(signatures)) == 1
    assert signatures[0] == (1, 3, 1, 2)
    assert compare_pipeline_runs([{}, {}])["summary"]["eligibility_changes"] is None
