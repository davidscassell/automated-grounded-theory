"""0.8.7 regressions for the v28 live-eval findings. No network calls."""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, axial_code, focused_code, selective_code
from automated_grounded_theory.async_support import ProviderAllowance
from automated_grounded_theory.axial import (
    AxialRelationshipDraft,
    _apply_model_relations,
    _coverage,
    _endpoint_anchoring,
    _memo_flags,
    _model_facing_passage,
    core_stability,
)
from automated_grounded_theory.backends import record_model_determinism
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.saturation import thematic_saturation_report
from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _book, _passages, _source
from tests.test_v080 import QUOTE, FakeAxialCoder, _corpus


def test_version_is_087():
    assert __version__ == "0.9.39"
    assert POLICY_VERSION == "0.9.39"


def test_generic_stem_does_not_anchor_and_disagreement_is_reported():
    book = {
        "bbq": {"code_id": "bbq", "code": "valuing annual bbq", "definition": "people who work the yearly bbq"},
        "crew": {"code_id": "crew", "code": "valuing shift coworkers", "definition": "people on the crew"},
    }
    generic = {
        "from_id": "bbq",
        "to_id": "crew",
        "quote": "I get to work with smart people",
        "passage_id": "p1",
    }
    rows = _endpoint_anchoring(generic, book)
    assert rows[0]["anchored"] is False
    assert rows[0]["reason"] == "generic_stem_only"
    distinctive = {
        "from_id": "crew",
        "to_id": "bbq",
        "quote": "coworkers on the shift",
        "passage_id": "p1",
        "embedding": [1.0, 0.0],
    }
    book["crew"]["embedding"] = [0.0, 1.0]
    anchored = _endpoint_anchoring(distinctive, book)
    assert anchored[0]["anchored"] is True
    assert anchored[0]["anchoring_disagreement"] == "lexical_anchored_embedding_unanchored"
    embedded = {
        "from_id": "bbq",
        "to_id": "crew",
        "quote": "I get to work with smart people",
        "passage_id": "p1",
        "embedding": [1.0, 0.0],
    }
    book["bbq"]["embedding"] = [1.0, 0.0]
    second = _endpoint_anchoring(embedded, book)
    assert second[0]["anchored"] is True
    assert second[0]["reason"] == "embedding"
    assert second[0]["anchoring_disagreement"] == "lexical_generic_only_embedding_anchored"


def test_cross_respondent_quotes_do_not_count_as_citation_support():
    book = {
        "collab": {"code_id": "collab", "code": "improved collaboration", "definition": "improved collaboration"},
        "hybrid": {"code_id": "hybrid", "code": "hybrid schedule", "definition": "keep hybrid schedule"},
    }
    passages = {
        "p-collab": {
            "passage_id": "p-collab",
            "doc_id": "d-a",
            "respondent_id": "r-a",
            "text": "improved collaboration between planning and customer service",
        },
        "p-hybrid": {
            "passage_id": "p-hybrid",
            "doc_id": "d-b",
            "respondent_id": "r-b",
            "text": "Keep the hybrid schedule please",
        },
    }
    row = {
        "relationship_id": "rel-0001",
        "from_id": "collab",
        "to_id": "hybrid",
        "paradigm_role": "consequence",
        "statement": "improved collaboration leads to pleading to keep hybrid schedule",
        "passage_id": "p-collab",
        "quote": "",
        "quote_from": "improved collaboration between planning and customer service",
        "quote_to": "Keep the hybrid schedule please",
        "flags": [],
        "endpoint_anchoring": [
            {"endpoint": "from", "anchored": True, "anchoring_disagreement": ""},
            {"endpoint": "to", "anchored": True, "anchoring_disagreement": ""},
        ],
        "evidence_scope": "cross_respondent",
        "quotes_share_passage": False,
    }
    coverage = _coverage({"codebook": book, "relationships": [row], "storyline": [], "metadata": {}, "empty_slots": []})
    assert coverage["citation_anchoring_rate"] is None
    assert coverage["citation_anchoring_cross_passage"] is None
    assert coverage["cross_passage_relationships"] == 1
    decided = _apply_model_relations(
        [dict(row)],
        {
            "relationships": [
                {
                    "relationship_id": "rel-0001",
                    "disposition": "accepted",
                    "paradigm_role": "consequence",
                    "rationale": "The quotes are in the same passage.",
                }
            ]
        },
        {},
        passages=passages,
        codebook=book,
    )[0]
    assert decided["disposition"] == "provisional"
    assert decided["evidence_tier"] == 1
    assert decided["evidence_label"] == "hypothesis"
    assert decided["evidence_status"] == "hypothesis"
    assert decided["cross_case"] is True
    assert "acceptance_withheld" in decided["flags"]
    assert "cross_case" in decided["flags"]
    assert "reviewer_same_passage_mismatch" in decided["flags"]
    assert decided["downstream_eligible"] is False
    held = _coverage({"codebook": book, "relationships": [decided], "storyline": [], "metadata": {}, "empty_slots": []})
    assert held["citation_anchoring_rate"] is None
    assert held["citation_anchoring_by_tier"]["hypothesis"]["citation_rate"] is None


def test_empty_statement_is_not_a_relationship_and_blank_rows_are_counted():
    empty = {
        "relationship_id": "rel-blank",
        "from_id": "a",
        "to_id": "b",
        "statement": "",
        "quote": "alpha side",
        "passage_id": "p1",
        "flags": [],
    }
    kept = {
        "relationship_id": "rel-kept",
        "from_id": "a",
        "to_id": "b",
        "statement": "alpha side shapes beta side",
        "quote": "alpha side",
        "passage_id": "p1",
        "flags": [],
        "endpoint_anchoring": [
            {"endpoint": "from", "anchored": False},
            {"endpoint": "to", "anchored": False},
        ],
    }
    coverage = _coverage(
        {
            "codebook": {"a": {"code": "alpha side"}, "b": {"code": "beta side"}},
            "relationships": [empty, kept],
            "storyline": [],
            "metadata": {},
            "empty_slots": [],
        }
    )
    assert coverage["relationships"] == 1
    assert coverage["empty_statements"] == 1

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AxialRelationshipDraft(statement="", passage_id="p1", quote=QUOTE)

    class BlankCoder(FakeAxialCoder):
        def proposals(self, body):
            from automated_grounded_theory.axial import AxialProposalBatch, AxialRelationshipDraft

            pair = (body.get("pairs") or [{}])[0]
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or ""),
                        to_id=str(pair.get("to_id") or ""),
                        paradigm_role="context",
                        statement="Packing the suitcase before departure shapes the return to work.",
                        passage_id="p1",
                        quote=QUOTE,
                    )
                ]
            )

    result = axial_code(_corpus(), BlankCoder(), seed=3, max_axial_review_calls=2, interrupt=False)
    diagnostics = result.metadata["proposal_diagnostics"]
    assert diagnostics["blank_rows"] == 0
    assert diagnostics["empty_statements"] == 0
    assert diagnostics["rejection_reasons"] == {}
    assert len(result.relationships) == 1
    assert result.relationships[0]["statement"]


def test_memo_same_passage_claim_is_flagged_and_partial_overlap_is_not():
    relationship = {
        "quote": "Packing the suitcase took all night",
        "quotes_share_passage": False,
        "endpoint_anchoring": [{"anchored": False}, {"anchored": False}],
        "statement": "a condition",
    }
    flags = _memo_flags("These quotes sit within the same passage.", relationship)
    assert "memo_same_passage_mismatch" in flags
    partial = _memo_flags("The evidence on this side is partial.", relationship)
    assert "memo_quote_unsupported" not in partial
    assert "memo_evidence_strength_unstated" not in partial


def test_core_without_a_relationship_is_unsupported_and_stability_stays_undefined():
    from automated_grounded_theory.axial import _note_core_support

    coded = axial_code(_corpus(), FakeAxialCoder(), seed=4, interrupt=False, max_axial_review_calls=2)
    _note_core_support(
        {"relationships": coded.relationships, "metadata": coded.metadata},
        coded.metadata,
        "not-in-the-graph",
    )
    assert coded.metadata["core_unsupported_by_graph"] is True
    assert coded.metadata["core_graph_support"] == "core_unsupported_by_graph"
    stability = core_stability([{"metadata": {"model_core_id": "a"}}, {"metadata": {"model_core_id": "b"}}])
    assert stability["score"] == 0
    assert stability["n"] == 2
    unnamed = core_stability([{"metadata": {}}])
    assert unnamed["n"] == 0
    assert unnamed["runs"] == 1
    assert unnamed["score"] is None


def test_counterexample_stage_without_a_relevant_judgment_is_not_plain_complete():
    class Irrelevant(FakeAxialCoder):
        def counterexamples(self, body):
            from automated_grounded_theory.axial import AxialCounterexample, AxialCounterexampleBatch

            rows = []
            for item in body.get("counterexamples") or []:
                passage = item.get("passage") or {}
                rows.append(
                    AxialCounterexample(
                        relationship_id=str((item.get("relationship") or {}).get("relationship_id") or ""),
                        passage_id=str(passage.get("passage_id") or ""),
                        judgment="irrelevant",
                        quote="",
                        rationale="Not about both categories.",
                    )
                )
            return AxialCounterexampleBatch(counterexamples=rows)

    result = axial_code(_corpus(), Irrelevant(), seed=5, max_axial_review_calls=4, interrupt=False)
    stage = result.metadata["stage_outcomes"]["counterexamples"]
    assert stage["status"] == "complete_no_negative_search"
    assert stage["reason"] in {"no_calls", "no_relevant_judgment"}


def test_synthetic_payload_is_neutral_and_the_audit_keeps_the_label(tmp_path):
    result = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=6,
        axial_synthetic_counterexample=True,
        max_axial_review_calls=4,
        interrupt=False,
        output_dir=tmp_path,
    )
    assert result.metadata["synthetic_counterexample"]["label"] == "[synthetic counterexample]"
    assert result.metadata["synthetic_counterexample"]["detected"] is True
    sent = [
        message["content"]
        for entry in result.prompt_log
        if entry.get("kind") == "axial_counterexample"
        for message in entry.get("messages") or []
    ]
    blob = "\n".join(sent).casefold()
    assert "\"synthetic\"" not in blob
    assert "audit_label" not in blob
    assert "does not hold" not in blob
    assert "[synthetic counterexample]" not in blob
    assert "synthetic-counterexample" not in blob
    assert "p-contrast" in blob
    facing = _model_facing_passage({"passage_id": "synthetic-counterexample", "synthetic": True, "audit_label": "[synthetic counterexample]", "text": "secret"})
    assert facing == {
        "passage_id": "p-contrast",
        "doc_id": "d-review",
        "text": "On this occasion the usual sequence ran differently, and the expected outcome did not occur.",
    }
    exported = (tmp_path / "category_relationships.json").read_text(encoding="utf-8")
    assert "p-contrast" not in exported
    assert "[synthetic counterexample]" not in exported
    assert json.loads((tmp_path / "axial_run.json").read_text(encoding="utf-8"))["metadata"]["synthetic_counterexample"]["label"]


def test_storyline_reserves_tokens_and_seed_is_on_metadata():
    coded = axial_code(_corpus(), FakeAxialCoder(), seed=9, interrupt=False, max_axial_review_calls=2)
    assert coded.metadata["seed"] == coded.metadata["procedure"]["seed"]
    assert coded.metadata["seed"] == coded.metadata["pair_coverage"]["seed"]
    finished = selective_code(
        coded,
        FakeAxialCoder(),
        seed=9,
        include_memos=True,
        include_storyline=True,
        max_selective_review_calls=2,
        max_axial_review_tokens=24000,
        interrupt=False,
    )
    reserve = finished.metadata["procedure"]["storyline_token_reserve"]
    assert reserve == 6000
    assert finished.metadata["seed"] == 9


def test_saturation_payload_separates_last_document_from_the_window():
    report = thematic_saturation_report(
        [
            {"document_id": "d1", "new_codes": 2, "new_properties": 0, "new_relationships": 1, "cumulative_codes": 2, "code_frequencies": {"a": 1}},
            {"document_id": "d2", "new_codes": 0, "new_properties": 0, "new_relationships": 0, "cumulative_codes": 2, "code_frequencies": {"a": 1}},
            {"document_id": "d3", "new_codes": 1, "new_properties": 0, "new_relationships": 1, "cumulative_codes": 3, "code_frequencies": {"a": 1}},
        ]
    )
    assert report["last_document_new_code_rate"] == 1
    assert report["new_code_rate"] == 1
    assert report["window_new_code_rate"] == round(3 / 3, 4)
    assert report["window_new_relationship_rate"] == round(2 / 3, 4)
    assert "newest document" in report["window_definition"]
    assert report["judgment"] is None
    small = thematic_saturation_report(
        [
            {"document_id": "d1", "new_codes": 1, "new_relationships": 0, "cumulative_codes": 1, "code_frequencies": {"a": 1}},
            {"document_id": "d2", "new_codes": 1, "new_relationships": 0, "cumulative_codes": 2, "code_frequencies": {"a": 1}},
        ]
    )
    assert small["saturation_window"] == "not_interpretable"
    assert small["new_code_rate"] == 1
    assert small["judgment"] is None


def test_allowance_reports_raw_and_compared_totals():
    allowance = ProviderAllowance(max_tokens=150000, token_reserve=0)
    allowance.observed_raw = 138
    allowance.observed_input = 100
    allowance.tokens_estimated = 160000
    snap = allowance.snapshot()
    assert snap["tokens_estimated"] == 160000
    assert snap["tokens_estimated_raw"] == 160000
    assert snap["tokens_estimated_compared"] < snap["tokens_estimated_raw"]
    assert snap["ceiling_comparison"] == "guarded"
    assert "guarded" in snap["ceiling_note"]
    note = record_model_determinism(None)
    assert "does not guarantee" in note["model_determinism_note"]


class _TemperatureCoder(FakeFocusedCoder):
    temperature: float | None = None
    seed: int | None = None


def test_focused_resume_with_the_same_model_object_matches_the_manifest():
    coder = _TemperatureCoder()
    passages = _passages()[:1]
    text = passages[0]["text"]
    source = _source(
        passages=passages,
        codes=[_book()[0]],
        segments=[
            {
                "segment_id": "s1",
                "code_id": "c-freq",
                "code": "packing bags",
                "passage_id": "p1",
                "doc_id": "d1",
                "quote": text,
                "review_reasons": [],
            }
        ],
    )
    saver = InMemorySaver()
    common = dict(
        model=coder,
        seed=1,
        focused_sampling_seed=11,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=2,
        checkpointer=saver,
        thread_id="same-model-resume",
    )
    paused = focused_code(source, interrupt=True, **common)
    assert paused.interrupted is True
    assert coder.temperature == 0.0
    assert coder.seed == 11
    settings = paused.metadata["checkpoint_manifest"]["model"]["model_settings"]
    assert settings["temperature"] == 0.0
    assert settings["seed"] == 11
    resumed = focused_code(source, interrupt=True, resuming=True, resume={"accept": True}, **common)
    assert resumed.metadata["checkpoint_manifest"]["model"]["model_settings"]["temperature"] == 0.0
    assert "does not guarantee" in resumed.metadata["model_determinism_note"]


def test_model_facing_passage_drops_audit_fields():
    hidden = _model_facing_passage(
        {
            "passage_id": "p1",
            "doc_id": "d1",
            "text": "Packing the suitcase took all night.",
            "synthetic": False,
            "audit_label": "[synthetic counterexample]",
        }
    )
    assert "audit_label" not in hidden
    assert "synthetic" not in hidden
    assert hidden["text"].startswith("Packing")
