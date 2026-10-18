"""0.9.0 regressions for the v29 live-eval findings. No network calls."""

from __future__ import annotations

import json

from automated_grounded_theory.axial import (
    AxialProposalBatch,
    AxialRelationshipDraft,
    StorylineBatch,
    StorylineStatementDraft,
    _allocate_evidence,
    _annotate_endpoints,
    _coverage,
    _endpoint_anchoring,
    _memo_flags,
    _score_candidates,
    _select_counterexamples,
    core_stability,
)
from automated_grounded_theory.backends import record_model_determinism
from automated_grounded_theory.focused import (
    SYNTHETIC_PASSAGE_ID,
    _apply_claim_test,
    _canonical_passage_id,
    _claim_test_relevance,
    focused_code,
)
from automated_grounded_theory.saturation import gate_model_saturation, pair_coverage_reading, stopping_criteria_report
from automated_grounded_theory import axial_code, selective_code
from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _entry, _source
from tests.test_v074 import _passage, _segment
from tests.test_v079 import FocusedClaimJudgment, FocusedClaimTest
from tests.test_v080 import QUOTE, FakeAxialCoder, _corpus


def test_storyline_flags_a_cause_on_an_unaccepted_cross_respondent_row():
    coded = axial_code(_corpus(), FakeAxialCoder(), seed=3, interrupt=False, max_axial_review_calls=2)
    coded.relationships[0]["disposition"] = "provisional"
    coded.relationships[0]["evidence_status"] = "cross_respondent_pattern"
    coded.relationships[0]["evidence_scope"] = "cross_respondent"

    class CausalStory(FakeAxialCoder):
        def storyline(self, body):
            del body
            return StorylineBatch(
                statements=[
                    StorylineStatementDraft(
                        text="Excessive approval layers reduce the scrappy efficiency of the team.",
                        category_ids=["c-prep", "c-return"],
                        relationship_ids=["rel-0001"],
                        passage_id="p1",
                        quote=QUOTE,
                    )
                ]
            )

    finished = selective_code(coded, CausalStory(), include_storyline=True, interrupt=False)
    row = finished.storyline[0]
    assert "storyline_unaccepted_relationship" in row["flags"]
    assert row["phrasing"] == "pattern"
    assert row["causal_claim_on_pattern"] is True
    assert "reduce" in row["text"]
    assert row["relationship_ids"] == ["rel-0001"]

    coded.relationships[0]["disposition"] = "accepted"
    coded.relationships[0]["evidence_status"] = "accepted"
    coded.relationships[0]["evidence_scope"] = "same_passage"
    coded.relationships[0]["flags"] = []
    accepted = selective_code(coded, CausalStory(), include_storyline=True, interrupt=False)
    assert "storyline_unaccepted_relationship" not in accepted.storyline[0]["flags"]


def test_synthetic_contrast_id_is_addressed_and_excluded_from_research_counts(tmp_path):
    assert _canonical_passage_id("p-contrast") == SYNTHETIC_PASSAGE_ID
    counts = {"supports": 0, "contradicts": 0, "qualifies": 0, "irrelevant": 0}
    properties = [{"entry_id": "e1", "pole": "visiting sites", "claim_tests": []}]
    _apply_claim_test(
        properties,
        [],
        {"entry_id": "e1", "judgment": "irrelevant", "quote": "neutral"},
        SYNTHETIC_PASSAGE_ID,
        counts,
        valid=True,
    )
    assert counts["irrelevant"] == 0
    assert properties[0]["claim_tests"][0]["excluded_from_research_outputs"] is True
    relevance = _claim_test_relevance(properties, [], {})
    assert relevance["judgments"] == 0
    assert relevance["relevance_rate"] is None

    class ContrastCoder(FakeFocusedCoder):
        def claim_test_batch(self, payload):
            rows = []
            for passage in payload.get("passages") or []:
                passage_id = str(passage.get("passage_id") or "")
                claim = passage.get("claim") or {}
                synthetic = passage_id in {SYNTHETIC_PASSAGE_ID, "p-contrast"}
                rows.append(
                    FocusedClaimJudgment(
                        passage_id=passage_id,
                        entry_id=str(claim.get("entry_id") or ""),
                        judgment="contradicts" if synthetic else "irrelevant",
                        rationale="Synthetic detection check." if synthetic else "Not the claim.",
                        quote=str(passage.get("text") or "") if synthetic else "",
                    )
                )
            return FocusedClaimTest(passages=rows)

    result = focused_code(
        _source(
            passages=[
                _passage("p1", "Leaders need to visit sites more often during the quarter."),
                _passage("p2", "The second visit waited until the following Monday."),
            ],
            codes=[_entry("c1", "visiting sites", 2, ["p1", "p2"])],
            segments=[_segment("s1", "c1", "p1"), _segment("s2", "c1", "p2")],
        ),
        model=ContrastCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0.5,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0.5,
        focused_synthetic_counterexample=True,
        focused_request_retries=0,
        validation_retries=0,
        max_focused_review_calls=4,
        interrupt=False,
        output_dir=tmp_path,
    )
    assert result.metadata.get("recode_stop") != "no_progress"
    assert result.metadata["claim_test_counts"]["contradicts"] == 0
    detected = result.metadata.get("synthetic_counterexample") or {}
    assert detected.get("detected") is True
    assert detected.get("excluded_from_research_outputs") is True


def test_same_passage_cooccurrence_is_shown_before_one_sided_passages():
    passages = {
        "p1": {"passage_id": "p1", "doc_id": "d1", "text": "Both categories are in this passage."},
        "p2": {"passage_id": "p2", "doc_id": "d1", "text": "Only the first category."},
        "p3": {"passage_id": "p3", "doc_id": "d2", "text": "Only the second category."},
    }
    evidence, _omitted = _allocate_evidence(
        [{"from_id": "a", "to_id": "b"}],
        {"a": ["p2", "p1"], "b": ["p3", "p1"]},
        passages,
        limit=1,
    )
    assert evidence[0]["passage_id"] == "p1"
    assert evidence[0]["same_passage_cooccurrence"] is True
    assert set(evidence[0]["code_ids"]) == {"a", "b"}


def test_low_pair_coverage_is_not_interpretable_as_sufficiency():
    reading = pair_coverage_reading(946, 6, 1)
    assert reading["low"] is True
    assert reading["status"] == "unmet"
    assert reading["pair_coverage"] == round(6 / 946, 4)
    report = stopping_criteria_report(
        {
            "n_documents": 3,
            "window_new_code_rate": 0.0,
            "window_new_property_rate": 0.0,
            "window_new_relationship_rate": 0.0,
            "code_frequency_shift": 0.0,
            "pairs_planned": 946,
            "pairs_completed": 6,
            "relationships_found": 1,
        },
        [],
    )
    by_name = {row["criterion"]: row for row in report["criteria"]}
    assert by_name["pair_coverage"]["status"] == "unmet"
    assert by_name["pair_coverage"]["observed"]["relationships_found"] == 1
    assert by_name["new_relationships"]["status"] == "undefined"
    assert by_name["new_relationships"]["undefined_reason"] == "low_pair_coverage"
    assert report["declares_saturation"] is False
    assert report["judgment"] is None
    gated = gate_model_saturation(
        {"assessment": "sufficient_for_this_question", "rationale": "The window looks quiet."},
        reading,
    )
    assert gated["assessment"] is None
    assert gated["model_said"] == "sufficient_for_this_question"
    assert gated["status"] == "not_interpretable"
    assert gated["withheld_reason"] == "low_pair_coverage"
    small = pair_coverage_reading(4, 2, 1)
    assert small["status"] == "undefined"
    assert small["low"] is False
    kept = gate_model_saturation({"assessment": "not_saturated", "rationale": "Still moving."}, small)
    assert kept["assessment"] == "not_saturated"
    missing = stopping_criteria_report({"n_documents": 3, "window_new_relationship_rate": 0.0}, [])
    pair = next(row for row in missing["criteria"] if row["criterion"] == "pair_coverage")
    assert pair["status"] == "undefined"
    assert pair["observed"]["pair_coverage"] is None


def test_a_category_label_is_not_anchored_unless_it_is_in_a_passage():
    book = {
        "roll": {"code_id": "roll", "code": "criticizing rollout without training", "definition": "the launch"},
        "grow": {"code_id": "grow", "code": "describing company as good with growing pains", "definition": "growth"},
    }
    relationship = {
        "from_id": "roll",
        "to_id": "grow",
        "passage_id": "p1",
        "quote_from": "criticizing rollout without training",
        "quote_to": "describing company as good with growing pains",
        "quote": "criticizing rollout without training",
        "flags": [],
    }
    without = _endpoint_anchoring(relationship, book)
    assert all(row["anchored"] for row in without)
    passages = {
        "p1": {
            "passage_id": "p1",
            "doc_id": "d1",
            "text": "The manager said the launch skipped the class and the office is still uneven.",
        }
    }
    _annotate_endpoints(relationship, book, passages=passages)
    assert all(row["anchored"] is False for row in relationship["endpoint_anchoring"])
    assert all(row["reason"] == "quote_not_verbatim" for row in relationship["endpoint_anchoring"])
    assert "quotes_not_in_cited_passage" in relationship["flags"]


def test_short_verbatim_quotes_use_an_embedding_second_signal():
    book = {
        "stay": {
            "code_id": "stay",
            "code": "settling into the role",
            "definition": "people who remain",
            "embedding": [1.0, 0.0],
        }
    }
    passages = {
        "p1": {
            "passage_id": "p1",
            "doc_id": "d1",
            "text": "Its a job. Glad to be here with the crew.",
            "embedding": [1.0, 0.0],
        }
    }
    rows = _endpoint_anchoring(
        {
            "from_id": "stay",
            "to_id": "stay",
            "passage_id": "p1",
            "quote_from": "Its a job.",
            "quote_to": "Glad to be here",
        },
        book,
        passages=passages,
    )
    assert rows[0]["anchored"] is True
    assert rows[0]["second_signal"] == "passage_embedding"
    assert rows[0]["anchoring_disagreement"]
    assert rows[1]["anchored"] is True
    bare = {
        "stay": {"code_id": "stay", "code": "settling into the role", "definition": "people who remain"},
    }
    missing = _endpoint_anchoring(
        {"from_id": "stay", "to_id": "stay", "passage_id": "p1", "quote": "Its a job."},
        bare,
        passages={"p1": {"passage_id": "p1", "doc_id": "d1", "text": "Its a job."}},
    )
    assert missing[0]["anchored"] is False
    assert missing[0]["second_signal"] == "unavailable"
    assert missing[0]["anchoring_disagreement"] == "lexical_unanchored_second_signal_unavailable"


def test_degenerate_and_causal_memos_are_flagged():
    cross = {
        "quote": "Packing the suitcase took all night",
        "quotes_share_passage": False,
        "evidence_status": "cross_respondent_pattern",
        "evidence_scope": "cross_respondent",
        "disposition": "provisional",
        "endpoint_anchoring": [{"anchored": False}, {"anchored": False}],
        "statement": "a pattern",
    }
    degenerate = _memo_flags("The evidence is two-sided.", cross)
    assert "memo_degenerate" in degenerate
    assert "memo_missing_quote" in degenerate
    assert "memo_quote_unsupported" in degenerate
    causal = _memo_flags("Shared events can foster appreciation for the mission.", cross)
    assert "memo_causal_on_pattern" in causal
    partial = _memo_flags("The evidence on this side is partial.", cross)
    assert "memo_quote_unsupported" not in partial
    assert "memo_degenerate" not in partial
    assert "memo_missing_quote" not in partial


def test_counterexample_search_reports_why_and_falls_back_to_the_opposite_pole():
    book = {
        "prep": {"code": "preparing suitcase departure", "definition": "packing before a trip"},
        "back": {"code": "returning office schedule", "definition": "coming back to the desk"},
    }
    relationship = {
        "relationship_id": "rel-1",
        "from_id": "prep",
        "to_id": "back",
        "passage_id": "cited",
        "doc_id": "d1",
        "statement": "Preparing the suitcase before departure shapes the office return.",
        "quote": "packing the suitcase before departure",
    }
    passages = [
        {"passage_id": "cited", "doc_id": "d1", "text": "Packing the suitcase before departure."},
        {
            "passage_id": "opposite",
            "doc_id": "d2",
            "text": "The suitcase packing did not happen before the departure.",
        },
    ]
    queued, stats = _select_counterexamples(
        [relationship],
        passages,
        slots=2,
        synthetic=False,
        codebook=book,
        segments=[],
    )
    assert stats["both_endpoint_queued"] == 0
    assert stats["fallback"] is None
    assert queued[0]["source"] == "opposite_pole"
    assert queued[0]["passage_id"] == "opposite"
    assert stats["candidates_considered"] >= 1
    assert stats["cooccurrence_2x2"]["a_only"] >= 1
    assert "not a validity" in stats["cooccurrence_2x2"]["note"] or "do not validate" in stats["cooccurrence_2x2"]["note"]
    empty, none = _select_counterexamples(
        [relationship],
        [{"passage_id": "cited", "doc_id": "d1", "text": "Packing the suitcase before departure."}],
        slots=2,
        synthetic=False,
        codebook=book,
    )
    assert empty == []
    assert none["no_call_reason"] == "no_plausible_candidates"
    assert none["fallback"] is None
    _budget_rows, budget = _select_counterexamples([relationship], passages, slots=0, synthetic=False, codebook=book)
    assert budget["no_call_reason"] == "budget"


def test_core_chooser_prefers_a_relationship_endpoint_and_stability_stays_undefined():
    rows = _score_candidates(
        {
            "codebook": {
                "lonely": {"code": "sitting alone"},
                "linked": {"code": "preparing to leave"},
            },
            "relationships": [
                {
                    "from_id": "linked",
                    "to_id": "other",
                    "statement": "A link.",
                    "disposition": "proposed",
                }
            ],
            "segments": [],
            "properties": [],
            "dimensions": [],
            "passages": [],
        }
    )
    assert rows[0]["code_id"] == "linked"
    assert rows[0]["relationship_evidence"] is True
    assert rows[1]["relationship_evidence"] is False
    coded = axial_code(_corpus(), FakeAxialCoder(), seed=4, interrupt=False, max_axial_review_calls=2)
    coded.codebook.append({"code_id": "lonely", "code": "sitting alone", "definition": "no link"})
    chosen = selective_code(
        coded,
        FakeAxialCoder(core_id="lonely"),
        decision_mode="llm",
        interrupt=False,
    )
    assert chosen.model_core_id == "lonely"
    assert chosen.researcher_core_id == ""
    assert chosen.metadata["core_unsupported_by_graph"] is True
    assert chosen.metadata.get("relationship_preferred_core_ids") == []
    assert "relationship_preferred_core_id" not in chosen.metadata
    assert chosen.metadata["core_stability"]["score"] is None
    assert chosen.metadata["core_stability"]["n"] == 1
    assert core_stability([{"metadata": {}}])["score"] is None
    assert rows[0]["graph_supported"] is False


def test_blank_rows_are_counted_separately_from_unknown_passages():
    """A blank statement is not a parsed row. Other discards stay per reason."""
    from pydantic import ValidationError

    import pytest

    with pytest.raises(ValidationError):
        AxialRelationshipDraft(from_id="c-prep", to_id="c-return", statement=" ")

    class Mixed(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or ""),
                        to_id=str(pair.get("to_id") or ""),
                        statement=QUOTE,
                        passage_id="p1",
                        quote=QUOTE,
                    ),
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or ""),
                        to_id=str(pair.get("to_id") or ""),
                        statement="A stated link that cites a passage that was not sent.",
                        passage_id="not-sent",
                        quote="some quote that is not blank",
                    ),
                ]
            )

    result = axial_code(_corpus(), Mixed(), seed=3, max_axial_review_calls=2, interrupt=False)
    diagnostics = result.metadata["proposal_diagnostics"]
    assert diagnostics["blank_rows"] == 0
    assert diagnostics["empty_statements"] == 0
    assert diagnostics["rejection_reasons"]["verbatim_quote_statement"] == 1
    assert diagnostics["rejection_reasons"]["unknown_passage"] == 1
    assert result.relationships == []
    counts = _coverage(
        {
            "relationships": [
                {
                    "statement": "A pattern across people.",
                    "evidence_status": "cross_respondent_pattern",
                    "evidence_scope": "cross_respondent",
                    "disposition": "provisional",
                    "passage_id": "p1",
                },
                {
                    "statement": "An accepted link.",
                    "disposition": "accepted",
                    "evidence_scope": "same_passage",
                    "passage_id": "p1",
                },
            ],
            "storyline": [],
            "metadata": {},
            "codebook": {},
        }
    )
    assert counts["relationship_counts"]["cross_respondent_pattern"] == 1
    assert counts["relationship_counts"]["accepted"] == 1
    assert counts["relationships"] == 2


def test_determinism_record_does_not_claim_reproducibility():
    note = record_model_determinism(None)
    assert "does not guarantee" in note["model_determinism_note"]
    assert "not a reproducibility check" in note["model_determinism_note"]
    assert "does not claim the provider returned the same text" in note["model_determinism_note"]
    assert note["model_determinism"] == "unset"


def test_proposal_prompt_includes_the_cooccurrence_list():
    corpus = _corpus()
    corpus["segments"] = [
        {"segment_id": "s1", "code_id": "c-prep", "passage_id": "p1"},
        {"segment_id": "s2", "code_id": "c-return", "passage_id": "p1"},
    ]
    result = axial_code(corpus, FakeAxialCoder(), seed=3, max_axial_review_calls=2, interrupt=False)
    proposal = next(row for row in result.prompt_log if row.get("kind") == "axial_proposal")
    body = json.loads(proposal["messages"][-1]["content"])
    assert body["candidate_same_passage_cooccurrence"]
    assert body["candidate_same_passage_cooccurrence"][0]["passage_id"] == "p1"
    assert "same_passage_cooccurrence" in proposal["messages"][0]["content"]
