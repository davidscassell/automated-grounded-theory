"""0.9.36 review fixes. Fakes and synthetic fixtures only. No network calls.

These checks reproduce the v41 failures and the decisions that replace them.
They are processing coverage. They do not establish selection validity.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from automated_grounded_theory.axial import (
    AxialDecisionReviewer,
    AxialRelationDecisions,
    _relation_review_payload,
    _score_candidates,
    aaxial_code,
    axial_code,
)
from automated_grounded_theory.batching import estimate_tokens
from automated_grounded_theory.decision_review import RAISED_VARIANCE_RULE
from automated_grounded_theory.focused import _result_from_state, _status_fields, export_focused_state
from automated_grounded_theory.graph_grounding import (
    candidate_evidence_summary,
    evidence_cap_for_match,
    refit_evidence_summary,
)
from automated_grounded_theory.repeatability import compare_pipeline_runs
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v0934 import _core_source, _payload_estimate
from tests.test_v0935 import ScriptedCore, _core_bodies, _run_core, selective_code


def _match_records(result) -> list[dict]:
    found = []
    for round_row in result.metadata["core_bracket"]["rounds"]:
        for item in round_row.get("match_records") or []:
            found.append(item)
    return found


def test_an_opening_abstention_does_not_skip_later_candidates():
    finished = _run_core(["abstain", "c02"])
    assert finished.metadata["model_core_id"] == "c02"
    assert finished.metadata["core_outcome"] == "core_selected"
    assert finished.metadata["core_status"] == "applied"
    assert finished.metadata["core_selection_complete"] is True
    assert finished.metadata["core_candidates_compared"] == 3
    assert finished.metadata["core_uncovered_code_ids"] == []
    bodies = _core_bodies(finished)
    assert len(bodies) == 2
    assert bodies[0]["comparison_scope"]["match_ids"] == ["c00", "c01"]
    assert bodies[1]["comparison_scope"]["round_role"] == "final"
    assert bodies[1]["comparison_scope"]["match_ids"] == ["c02"]
    assert "final assessment" in bodies[1]["instruction"]
    assert "does not make it the core" in bodies[1]["instruction"]
    assert "partial" not in bodies[0]["instruction"].casefold()
    assert "original eligible set has 3" in bodies[0]["instruction"]
    assert "This match compares 2 candidates" in bodies[0]["instruction"]
    assert bodies[0]["comparison_scope"]["finalists"] == []
    assert bodies[1]["comparison_scope"]["finalists"] == ["c02"]

    source = _core_source("tools", 4)
    candidates = _score_candidates(source)
    probe = AxialDecisionReviewer(FakeAxialCoder(), max_calls=8, max_tokens=10**6, token_budget=10**6)
    pair = _payload_estimate(probe, candidates, ["c00", "c01"], whole_set=False, eligible=4)
    triple = _payload_estimate(probe, candidates, ["c00", "c01", "c02"], whole_set=False, eligible=4)
    assert triple > pair
    later = selective_code(
        source,
        ScriptedCore(core_steps=["abstain", "c02", "c02"]),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=8,
        max_axial_review_tokens=10**6,
        axial_token_budget=pair,
        batch_size=1,
    )
    assert later.metadata["model_core_id"] == "c02"
    assert later.metadata["core_outcome"] == "core_selected"
    sent = [body["review_chunk"]["code_ids"] for body in _core_bodies(later)]
    assert sent[0] == ["c00", "c01"]
    assert "c02" in sent[1] and "c03" in sent[1]
    assert sent[-1] == ["c02"]


def test_twelve_of_fifty_five_stops_early_and_counts_the_attempt():
    source = _core_source("tools", 55)
    candidates = _score_candidates(source)
    probe = AxialDecisionReviewer(FakeAxialCoder(), max_calls=8, max_tokens=10**6, token_budget=10**6)
    first = [f"c{index:02d}" for index in range(12)]
    thirteen = [f"c{index:02d}" for index in range(13)]
    fit = _payload_estimate(probe, candidates, first, whole_set=False, eligible=55)
    larger = _payload_estimate(probe, candidates, thirteen, whole_set=False, eligible=55)
    assert larger > fit
    finished = selective_code(
        source,
        ScriptedCore(core_steps=["abstain"]),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=1,
        max_axial_review_tokens=10**6,
        axial_token_budget=fit,
        batch_size=1,
    )
    assert finished.metadata["core_candidates_eligible"] == 55
    assert finished.metadata["core_candidates_compared"] == 12
    assert len(finished.metadata["core_uncovered_code_ids"]) == 43
    assert finished.metadata["core_selection_complete"] is False
    assert finished.metadata["core_outcome"] == "selection_unresolved"
    assert finished.metadata["core_status"] != "no_core_supported"
    assert finished.metadata["model_core_id"] == ""
    note = finished.metadata["core_coverage_note"]
    assert "Examining 12 of 55 candidates means selection stopped early." in note
    assert "does not mean that no core exists" in note
    assert finished.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 1
    assert _match_records(finished)[0]["status"] == "no_winner"
    assert _match_records(finished)[0]["abstention_reason"] == "no_suitable_candidate"


def test_every_completed_no_suitable_abstention_supports_no_core():
    finished = _run_core(["abstain", "abstain"])
    assert finished.metadata["core_outcome"] == "no_core_supported"
    assert finished.metadata["core_status"] == "no_core_supported"
    assert finished.metadata["core_selection_complete"] is True
    assert finished.metadata["model_core_id"] == ""
    assert finished.metadata["core_uncovered_code_ids"] == []
    assert [row["abstention_reason"] for row in _match_records(finished)] == [
        "no_suitable_candidate",
        "no_suitable_candidate",
    ]


def test_insufficient_evidence_retries_once_and_stays_unresolved():
    retried = _run_core([("abstain", "insufficient_evidence"), "c00", "c02"])
    bodies = _core_bodies(retried)
    assert bodies[1]["review_chunk"]["code_ids"] == ["c00", "c01"]
    assert bodies[1]["candidates"][0]["core_evidence_retry"] is True
    assert (
        bodies[1]["candidates"][0]["evidence_summary"]["token_cap"]
        >= bodies[0]["candidates"][0]["evidence_summary"]["token_cap"]
    )
    assert retried.metadata["model_core_id"] == "c02"
    assert retried.metadata["core_outcome"] == "core_selected"

    unresolved = _run_core(
        [("abstain", "insufficient_evidence"), ("abstain", "insufficient_evidence"), "c02"]
    )
    assert unresolved.metadata["core_outcome"] == "selection_unresolved"
    assert unresolved.metadata["core_selection_complete"] is False
    assert unresolved.metadata["model_core_id"] == ""
    assert _match_records(unresolved)[0]["status"] == "unresolved"
    assert _core_bodies(unresolved)[-1]["review_chunk"]["code_ids"] == ["c02"]
    assert "does not mean that no core exists" in unresolved.metadata["core_coverage_note"] or (
        "not a finding that no core exists" in unresolved.metadata["core_coverage_note"]
    )

    blocked = _run_core([("abstain", "insufficient_evidence")], max_decision_calls=1)
    assert blocked.metadata["core_outcome"] == "selection_unresolved"
    assert blocked.metadata["core_selection_complete"] is False
    assert blocked.metadata["core_candidates_compared"] == 2
    assert _match_records(blocked)[0]["status"] == "unresolved"
    assert all(not row.get("core_evidence_retry") for body in _core_bodies(blocked) for row in body["candidates"])


def test_a_sent_parse_failure_is_incomplete_and_not_an_abstention():
    failed = _run_core(["parse_failure"])
    assert failed.metadata["decision_stop"] == "review_error"
    assert failed.metadata["core_status"] == "withheld"
    assert failed.metadata["core_outcome"] == "selection_unresolved"
    assert failed.metadata["core_selection_complete"] is False
    assert failed.metadata["model_core_decision"] != "abstain"
    recorded = _match_records(failed)[0]
    assert recorded["decision"] == "incomplete"
    assert recorded["status"] == "incomplete"
    assert recorded["abstention_reason"] == ""
    assert failed.metadata["core_candidates_compared"] == 2


def test_match_budget_keeps_a_definition_a_statement_and_counterevidence():
    definition = "Tools for the shift cover the dock, the radio, and the late board."
    statement = "Tools for the shift are related to coworkers on the crew when the dock is open."
    variation = "The link is weaker on the night shift and stronger when the dock is open."
    counter = "The crew finished the shift without those tools and the work still moved."
    summary = candidate_evidence_summary(
        definition=definition,
        relationships=[
            {
                "statement": statement,
                "variation_note": variation,
                "counterexamples": [{"quote": counter, "judgment": "contradicts"}],
            }
        ],
        token_cap=80,
    )
    cap_12 = evidence_cap_for_match(12, per_request=6000, per_step=24000, configured=80)
    cap_55 = evidence_cap_for_match(55, per_request=6000, per_step=24000, configured=80)
    assert cap_12 == 80
    assert cap_55 == 27
    assert cap_12 > cap_55
    fitted = refit_evidence_summary(summary, cap_12)
    assert fitted["definition"]
    assert statement[:48] in fitted["representative_statements"][0]
    assert "finished the shift" in fitted["counterevidence"]
    assert fitted["tokens"] <= fitted["token_cap"] == 80
    assert fitted["tokens"] > 27
    shrunk = refit_evidence_summary(summary, cap_55)
    assert shrunk["definition"]
    assert shrunk["representative_statements"]
    assert shrunk["counterevidence"]
    assert shrunk["tokens"] <= 27
    book = {
        f"c{index:02d}": {"code_id": f"c{index:02d}", "code": "tools", "definition": definition}
        for index in range(55)
    }
    scored = _score_candidates(
        {
            "codebook": book,
            "relationships": [
                {
                    "relationship_id": "r1",
                    "from_id": "c00",
                    "to_id": "c01",
                    "statement": statement,
                    "variation_note": variation,
                    "disposition": "accepted",
                    "evidence_tier": 2,
                    "counterexamples": [{"quote": counter, "judgment": "contradicts"}],
                }
            ],
            "metadata": {
                "procedure": {"core_evidence_token_cap": 80, "max_axial_review_tokens": 24000},
                "performance": {"axial_token_budget": 6000},
            },
        }
    )
    rich = {row["code_id"]: row for row in scored}["c00"]["evidence_summary"]
    assert rich["token_cap"] == 80
    assert statement[:48] in rich["representative_statements"][0]
    assert "finished the shift" in rich["counterevidence"]
    assert estimate_tokens(rich["counterevidence"]) > 1


def _relationship(rel_id: str, statement: str) -> dict:
    return {
        "relationship_id": rel_id,
        "from_id": "c-prep",
        "to_id": "c-return",
        "statement": statement,
        "quote": "Packing the suitcase took all night before the trip.",
        "passage_id": "p1",
        "evidence_tier": 2,
        "evidence_status": "pattern_supported",
        "disposition": "proposed",
        "paradigm_role": "context",
    }


def test_relation_review_warns_and_names_unreviewed_rows_without_raising_the_limit():
    assert inspect.signature(aaxial_code).parameters["max_decision_calls"].default == 4
    reviewer = AxialDecisionReviewer(FakeAxialCoder(), max_calls=4, max_tokens=10**6, token_budget=6000)
    width = 200
    rows = []
    while width <= 8000:
        rows = [
            _relationship(f"rel-{index:04d}", f"Packing is a condition of returning {index}. " + ("note " * width))
            for index in range(6)
        ]
        one = reviewer.refusal_reason(
            _relation_review_payload(rows[:1], chunk_index=10**6, chunk_count=10**6),
            AxialRelationDecisions,
            {},
        )[1]
        two = reviewer.refusal_reason(
            _relation_review_payload(rows[:2], chunk_index=10**6, chunk_count=10**6),
            AxialRelationDecisions,
            {},
        )[1]
        if one <= 6000 < two:
            break
        width += 200
    else:
        raise AssertionError("could not size a relationship row to one request")
    source = _corpus()
    source["relationships"] = rows
    reviewed = axial_code(
        source,
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        axial_relations=False,
        interrupt=False,
        max_axial_review_calls=0,
        max_decision_calls=4,
        max_axial_review_tokens=10**6,
        axial_token_budget=6000,
    )
    warning = reviewed.metadata["relation_review_call_warning"]
    assert "6 rows" in warning
    assert "call limit is 4" in warning
    assert "not raised" in warning
    assert reviewed.metadata["relation_review_call_limit_raised"] is False
    assert reviewed.metadata["procedure"]["max_decision_calls"] == 4
    report = reviewed.metadata["decision_reviews"]["axial_relation_review"]
    assert report["chunks_sent"] == 4
    assert report["items_reviewed"] == 4
    assert report["complete"] is False
    assert report["stop_reasons"] == ["call_allowance"]
    unreviewed = reviewed.metadata["relation_unreviewed_rows"]
    assert len(unreviewed) == 2
    assert {row["reason"] for row in unreviewed} == {"call_allowance"}
    assert set(report["uncovered_ids"]) == {row["relationship_id"] for row in unreviewed}
    by_id = {row["relationship_id"]: row for row in reviewed.relationships}
    for item in unreviewed:
        assert by_id[item["relationship_id"]]["relation_review_status"] == "unreviewed"
        assert by_id[item["relationship_id"]]["relation_unreviewed_reason"] == "call_allowance"
    for relationship_id, row in by_id.items():
        if relationship_id not in report["uncovered_ids"]:
            assert row.get("relation_review_status") != "unreviewed"


def test_withheld_over_budget_matches_at_the_top_level(tmp_path):
    state = {
        "metadata": {
            "reread_requests_over_budget": [{"passage_id": "a"}, {"passage_id": "b"}],
            "procedure": {},
        },
        "passages": [
            {"passage_id": "a", "doc_id": "d1", "text": "one note"},
            {"passage_id": "b", "doc_id": "d1", "text": "another note"},
        ],
        "focused_reread": [],
        "focused_not_reread": [],
        "recode_queue": [],
        "recode_stop": "",
        "selection": [],
        "segments": [],
        "properties": [],
        "dimensions": [],
        "codebook": {},
    }
    fields = _status_fields(state)
    assert fields["withheld_over_budget"] == ["a", "b"]
    assert fields["withheld_over_budget_count"] == 2
    assert fields["withheld_over_budget"] == fields["coverage"]["withheld_over_budget"]
    assert fields["coverage"]["withheld_over_budget_count"] == 2
    paths = export_focused_state(state, tmp_path)
    run = json.loads(Path(paths["focused_run.json"]).read_text(encoding="utf-8"))
    assert run["withheld_over_budget"] == ["a", "b"]
    assert run["withheld_over_budget_count"] == 2
    assert run["metadata"]["withheld_over_budget"] == ["a", "b"]
    assert run["metadata"]["coverage"]["withheld_over_budget"] == ["a", "b"]
    result = _result_from_state(
        state,
        interrupted=False,
        payload=None,
        thread_id="withheld",
        resumable=False,
        checkpointer=None,
    )
    assert result.withheld_over_budget == ["a", "b"]
    assert result.to_dict()["withheld_over_budget"] == ["a", "b"]
    assert result.to_dict()["withheld_over_budget_count"] == 2
    assert result.metadata["withheld_over_budget"] == ["a", "b"]


def test_repeatability_records_raised_code_count_variance():
    left = [f"c{index:02d}" for index in range(57)]
    right = left[:55]
    compared = compare_pipeline_runs(
        [
            {"raised_code_ids": left, "relationships": []},
            {"raised_code_ids": right, "relationships": []},
        ]
    )
    summary = compared["summary"]
    assert summary["raised_variance_rule"] == RAISED_VARIANCE_RULE
    assert summary["raised_code_counts"] == [57, 55]
    assert summary["raised_code_count_min"] == 55
    assert summary["raised_code_count_max"] == 57
    assert summary["raised_code_count_range"] == 2
    assert summary["raised_code_counts_vary"] is True
    assert summary["raised_code_ids_not_in_every_run"] == ["c55", "c56"]
    assert summary["selected_code_jaccard"] == pytest.approx(55 / 57)
    pair = compared["pairwise"][0]
    assert pair["raised_code_count_diff"] == 2
    assert pair["raised_code_ids_only_left"] == ["c55", "c56"]
    assert pair["raised_code_ids_only_right"] == []
    missing = compare_pipeline_runs([{}, {"raised_code_ids": right}])
    assert missing["summary"]["raised_code_counts"] is None
    assert missing["summary"]["raised_code_counts_vary"] is None
    assert missing["pairwise"][0]["raised_code_count_diff"] is None
