"""0.9.2 negative-case search and graph-grounded selective coding. No network calls."""

from __future__ import annotations

import asyncio

from automated_grounded_theory.axial import (
    AxialRuntime,
    _apply_counterexample,
    _counterexample_node,
    _coverage,
    _memo_flags,
    _score_candidates,
    _select_counterexamples,
    core_stability,
)
from automated_grounded_theory.graph_grounding import graph_snapshot, integration_report
from automated_grounded_theory.negative_case_search import claim_respondent_search
from tests.test_v080 import QUOTE, FakeAxialCoder, _corpus
from automated_grounded_theory import axial_code, selective_code
from automated_grounded_theory.axial import StorylineBatch, StorylineStatementDraft


def _book():
    return {
        "prep": {"code": "preparing suitcase departure", "definition": "packing before a trip"},
        "back": {"code": "returning office schedule", "definition": "coming back to the desk"},
    }


def _relationship(**extra):
    row = {
        "relationship_id": "rel-1",
        "from_id": "prep",
        "to_id": "back",
        "passage_id": "cited",
        "doc_id": "d1",
        "respondent_id": "r-cited",
        "statement": "Preparing the suitcase before departure may be related to the office return.",
        "quote": "packing the suitcase before departure",
        "evidence_tier": 1,
    }
    row.update(extra)
    return row


def test_a_without_b_and_opposite_pole_are_primary_candidates():
    book = _book()
    passages = [
        {
            "passage_id": "cited",
            "doc_id": "d1",
            "respondent_id": "r-cited",
            "text": "Packing the suitcase before departure.",
        },
        {
            "passage_id": "only-prep",
            "doc_id": "d2",
            "respondent_id": "r-only-a",
            "text": "The suitcase was packed before the departure and nothing else was said.",
        },
        {
            "passage_id": "opposite",
            "doc_id": "d3",
            "respondent_id": "r-opposite",
            "text": "The suitcase packing did not happen before the departure.",
        },
        {
            "passage_id": "noise",
            "doc_id": "d4",
            "respondent_id": "r-noise",
            "text": "Format of the all-hands meeting was virtual.",
        },
    ]
    queued, stats = _select_counterexamples(
        [_relationship()],
        passages,
        slots=4,
        synthetic=False,
        codebook=book,
    )
    sources = {item["passage_id"]: item["source"] for item in queued}
    assert sources["only-prep"] == "a_without_b"
    assert sources["opposite"] == "opposite_pole"
    assert "noise" not in sources
    assert stats["filtered_count"] >= 1
    assert stats["candidates_considered"] >= 2
    assert stats["fallback"] is None
    matrix = stats["by_relationship"]["rel-1"]["cooccurrence_2x2"]
    assert matrix["a_only"] >= 1
    assert matrix["descriptive"] is True
    assert "do not validate" in matrix["note"]
    tier0, skipped = _select_counterexamples(
        [_relationship(evidence_tier=0)],
        passages,
        slots=4,
        synthetic=False,
        codebook=book,
    )
    assert tier0 == []
    assert skipped["by_relationship"]["rel-1"]["no_call_reason"] == "tier_0_category_evidence"


def test_budget_exhaustion_is_budget_limited_and_irrelevant_round_stays_undefined():
    book = _book()
    passages = [
        {"passage_id": "cited", "doc_id": "d1", "respondent_id": "r1", "text": "Packing the suitcase before departure."},
        {"passage_id": "only-prep", "doc_id": "d2", "respondent_id": "r2", "text": "The suitcase was packed before the departure."},
    ]
    queued, stats = _select_counterexamples([_relationship()], passages, slots=0, synthetic=False, codebook=book)
    assert queued == []
    assert stats["no_call_reason"] == "budget"
    assert stats["plausible_candidates"] >= 1
    runtime = AxialRuntime(coder=FakeAxialCoder(), request_timeout=0)
    state = {
        "relationships": [_relationship()],
        "passages": passages,
        "codebook": book,
        "segments": [],
        "warnings": [],
        "prompt_log": [],
        "metadata": {
            "procedure": {"counterexample_calls": 0, "negative_case_search": True, "axial_synthetic_counterexample": False},
            "performance": {"batch_size": 1},
            "stage_outcomes": {"proposals": {"status": "complete"}},
        },
        "axial_calls_used": 0,
    }
    result = asyncio.run(_counterexample_node(runtime)(state))
    stage = result["metadata"]["stage_outcomes"]["counterexamples"]
    assert stage["status"] == "budget_limited"
    assert stage["reason"] == "budget"
    coverage = _coverage(
        {
            "relationships": [
                {
                    "relationship_id": "rel-1",
                    "statement": "a link",
                    "counterexamples": [{"judgment": "irrelevant", "valid_test": True, "synthetic": False}],
                }
            ],
            "storyline": [],
            "codebook": {},
            "metadata": {},
            "passages": [],
        }
    )
    assert coverage["relationships_with_negative_case_search"] is None
    assert coverage["negative_case_relevance"] == "undefined_no_relevant_judgment"
    assert coverage["core_category_coverage"] is None
    assert coverage["core_respondent_coverage"] is None


def test_irrelevance_guard_and_contradict_or_qualify_feed_the_row():
    relationship = {"relationship_id": "rel-1", "flags": [], "counterexamples": [], "negative_cases": [], "statement": "a link"}
    passage = {"passage_id": "p9", "text": "The schedule changed and the departure did not lead to the office."}
    _apply_counterexample(
        relationship,
        {
            "judgment": "contradicts",
            "quote": "did not lead",
            "rationale": "This passage is irrelevant to the link.",
        },
        passage,
    )
    assert relationship["counterexamples"][0]["valid_test"] is False
    assert relationship["counterexamples"][0]["reason"] == "rationale_indicates_irrelevance"
    assert relationship.get("evidence_condition") != "disputed"
    held = {"relationship_id": "rel-2", "flags": [], "counterexamples": [], "negative_cases": [], "statement": "a link"}
    _apply_counterexample(
        held,
        {"judgment": "contradicts", "quote": "did not lead", "rationale": "The outcome is denied."},
        passage,
    )
    assert held["evidence_condition"] == "disputed"
    assert "relationship_contradicted" in held["flags"]
    assert held["contradiction_note"] == "did not lead"
    varied = {"relationship_id": "rel-3", "flags": [], "counterexamples": [], "negative_cases": [], "statement": "a link"}
    _apply_counterexample(
        varied,
        {"judgment": "qualifies", "quote": "the schedule changed", "rationale": "It holds only when the schedule changes."},
        {"passage_id": "p8", "text": "The schedule changed after the departure."},
    )
    assert varied.get("evidence_condition") != "disputed"
    assert varied["variation_note"] == "the schedule changed"
    assert varied["condition_note"] == "the schedule changed"
    assert varied["variation_search"] is True
    assert "relationship_qualified" in varied["flags"]
    memo_flags = _memo_flags("Packing leads to the office return.", held)
    assert "memo_ignores_contradiction" in memo_flags
    seen = _memo_flags("The departure does not lead to the office, however.", held)
    assert "memo_ignores_contradiction" not in seen


def test_core_requires_tier_two_and_storyline_cites_the_graph():
    supported = {
        "relationship_id": "rel-strong",
        "from_id": "core",
        "to_id": "other",
        "statement": "They co-occur.",
        "disposition": "accepted",
        "evidence_tier": 2,
        "evidence_status": "pattern_supported",
        "passage_id": "p1",
    }
    weak = {
        "relationship_id": "rel-weak",
        "from_id": "hint",
        "to_id": "other",
        "statement": "May be related.",
        "disposition": "proposed",
        "evidence_tier": 1,
    }
    denied = {
        "relationship_id": "rel-no",
        "from_id": "core",
        "to_id": "side",
        "statement": "A denied link.",
        "disposition": "accepted",
        "evidence_tier": 2,
        "flags": ["relationship_contradicted"],
        "contradiction_note": "did not happen",
        "counterexamples": [{"judgment": "contradicts", "valid_test": True, "quote": "did not happen"}],
    }
    state = {
        "codebook": {
            "core": {"code": "core category"},
            "other": {"code": "other category"},
            "hint": {"code": "hint category"},
            "side": {"code": "side category"},
            "orphan": {"code": "orphan category"},
        },
        "relationships": [supported, weak, denied],
        "segments": [],
        "properties": [],
        "dimensions": [],
        "passages": [
            {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "core and other"},
            {"passage_id": "p2", "doc_id": "d2", "respondent_id": "r2", "text": "unlinked"},
        ],
    }
    rows = _score_candidates(state)
    by_id = {row["code_id"]: row for row in rows}
    assert by_id["core"]["graph_supported"] is True
    assert by_id["core"]["pattern_supported_centrality"] == 2
    assert by_id["core"]["cases_covered"] >= 1
    assert by_id["hint"]["graph_supported"] is False
    assert by_id["hint"]["relationship_evidence"] is True
    assert by_id["orphan"]["graph_supported"] is False
    report = integration_report("core", state["codebook"], state["relationships"], state["passages"])
    assert "other" in report["direct_category_ids"]
    assert "side" in report["direct_category_ids"]
    assert "orphan" in report["orphan_category_ids"]
    assert "rel-no" in report["contradicted_relationship_ids"]
    assert report["core_category_coverage"] is not None
    empty = integration_report("core", state["codebook"], [], state["passages"])
    assert empty["core_category_coverage"] is None
    assert empty["core_respondent_coverage"] is None
    assert "orphan" in empty["orphan_category_ids"]
    snapshot = graph_snapshot(state["relationships"])
    assert snapshot["tiers"]["rel-strong"] == 2
    assert snapshot["tiers"]["rel-weak"] == 1
    same = core_stability(
        [{"metadata": {"model_core_id": "core"}}, {"metadata": {"researcher_core_id": "core"}}]
    )
    assert same["n"] == 2
    assert same["score"] == 1
    assert same["minimum_n"] == 2
    coded = axial_code(_corpus(), FakeAxialCoder(), seed=3, interrupt=False, max_axial_review_calls=2)
    coded.relationships[0]["flags"] = list(coded.relationships[0].get("flags") or []) + ["relationship_contradicted"]
    coded.relationships[0]["contradiction_note"] = "had no issues however"
    coded.relationships[0]["counterexamples"] = [
        {"judgment": "contradicts", "valid_test": True, "synthetic": False, "quote": "had no issues however"}
    ]

    class Draft(FakeAxialCoder):
        def storyline(self, body):
            del body
            return StorylineBatch(
                statements=[
                    StorylineStatementDraft(
                        text="Preparing to leave leads to returning to work.",
                        category_ids=["c-prep"],
                        relationship_ids=["rel-0001"],
                        passage_id="p1",
                        quote=QUOTE,
                    ),
                    StorylineStatementDraft(text="A descriptive aside with no link.", category_ids=["c-prep"]),
                ]
            )

        def memos(self, body):
            from automated_grounded_theory.axial import TheoreticalMemoBatch, TheoreticalMemoDraft

            drafts = []
            for row in body.get("relationships") or []:
                drafts.append(
                    TheoreticalMemoDraft(
                        relationship_id=str(row.get("relationship_id") or ""),
                        text="Preparing leads to returning.",
                    )
                )
            return TheoreticalMemoBatch(memos=drafts)

    finished = selective_code(coded, Draft(), include_storyline=True, include_memos=True, interrupt=False)
    claimed = next(row for row in finished.storyline if row["relationship_ids"])
    described = next(row for row in finished.storyline if not row["relationship_ids"])
    assert claimed["statement_kind"] == "relationship"
    assert "rel-0001" in claimed["relationship_tiers"]
    assert "storyline_contradicted_unqualified" in claimed["flags"]
    assert described["statement_kind"] == "description"
    assert "storyline_no_relationship" in described["flags"]
    assert finished.metadata["graph_snapshots"]["storyline"]["relationship_ids"]
    memo = next(row for row in finished.memos if row.get("role") == "theoretical_memo")
    assert "memo_ignores_contradiction" in memo["flags"]
    assert memo["graph_snapshot"]["relationship_ids"]


def test_claim_search_records_a_respondent_without_the_outcome():
    claims = [
        {
            "entry_id": "claim-1",
            "category_id": "prep",
            "name": "departure",
            "pole": "packed before departure",
            "text": "packed before departure",
            "quote": "packed before departure",
            "passage_id": "cited",
        }
    ]
    passages = [
        {"passage_id": "cited", "doc_id": "d1", "respondent_id": "r1", "text": "The suitcase was packed before departure."},
        {"passage_id": "only", "doc_id": "d2", "respondent_id": "r2", "text": "They were preparing the hall for guests."},
        {"passage_id": "other", "doc_id": "d3", "respondent_id": "r3", "text": "The meeting was virtual."},
    ]
    segments = [
        {"segment_id": "s1", "code_id": "prep", "passage_id": "cited"},
        {"segment_id": "s2", "code_id": "prep", "passage_id": "only"},
    ]
    found = claim_respondent_search(
        claims,
        passages,
        segments,
        [{"code_id": "prep", "code": "preparing the hall", "definition": "getting the hall ready"}],
    )
    kinds = {item["passage_id"]: item["kind"] for item in found["plausible"]}
    assert kinds.get("only") == "category_without_outcome"
    assert found["cooccurrence_2x2"]["descriptive"] is True
    assert found["candidates_considered"] >= 1
