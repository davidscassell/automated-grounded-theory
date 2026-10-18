"""0.7.0 focused coding: selection, reread, citations, interrupt, allowances."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from automated_grounded_theory import (
    FocusedError,
    __version__,
    afocused_code,
    focused_code,
    get_graph,
    read_focused_interrupt,
)
from automated_grounded_theory.api import CheckpointMismatch, CheckpointMigrationRequired
from automated_grounded_theory.async_support import run_sync
from automated_grounded_theory.focused import (
    DEFAULT_FOCUSED_BATCH_SIZE,
    FocusedRuntime,
    apply_selection_override,
    build_focused_graph,
    build_recode_queue,
    cite_note,
    focused_code as focused_code_fn,
    select_focused_codes,
)
from automated_grounded_theory.records import POLICY_VERSION, apply_comparisons
from automated_grounded_theory.results import OpenCodeResult
from automated_grounded_theory.validation import code_key

from tests.fake_focused import FakeFocusedCoder

ROOT = Path(__file__).resolve().parents[1]


def test_version_and_batch_size_stay_split_from_procedure():
    assert __version__ == "0.9.13"
    assert POLICY_VERSION == "0.9.13"
    assert inspect.signature(focused_code).parameters["batch_size"].default == 1
    assert inspect.signature(afocused_code).parameters["batch_size"].default == 1
    assert DEFAULT_FOCUSED_BATCH_SIZE == 1
    assert inspect.signature(focused_code).parameters["batch_size"].default == inspect.signature(
        __import__("automated_grounded_theory", fromlist=["open_code"]).open_code
    ).parameters["batch_size"].default


def _entry(code_id, label, frequency, docs, **extra):
    row = {
        "code_id": code_id,
        "code": label,
        "key": code_key(label),
        "definition": f"Incidents of {label}.",
        "frequency": frequency,
        "document_ids": list(docs),
        "in_vivo": bool(extra.get("in_vivo")),
        "dimensions": list(extra.get("dimensions") or []),
        "assignments": [],
    }
    if extra.get("pinned"):
        row["researcher_pin"] = True
    return row


def _passages():
    texts = {
        "p1": ("d1", "I spent the morning packing my suitcase."),
        "p2": ("d2", "Packing bags up with boxes took all night."),
        "p3": ("d3", "I was crying at the station when the train doors closed."),
        "p4": ("d4", "They hired me on a Friday and I start Monday."),
    }
    return [
        {
            "passage_id": pid,
            "doc_id": doc,
            "doc_name": doc,
            "text": text,
            "start_offset": 0,
            "end_offset": len(text),
            "metadata": {"respondent_id": doc},
        }
        for pid, (doc, text) in texts.items()
    ]


def _book():
    return [
        _entry("c-freq", "packing bags", 10, ["d1", "d2", "d3"]),
        _entry("c-rare", "writing letters", 1, ["d4"]),
        _entry("c-memo", "keeping letters", 1, ["d4"], in_vivo=True),
        _entry(
            "c-leave",
            "leaving home",
            4,
            ["d1", "d2"],
            dimensions=[{"name": "leaving", "value": "going"}],
        ),
        _entry(
            "c-stay",
            "staying home",
            1,
            ["d3"],
            dimensions=[{"name": "leaving", "value": "staying"}],
        ),
        _entry(
            "c-also",
            "going anyway",
            3,
            ["d1"],
            dimensions=[{"name": "leaving", "value": "going"}],
        ),
    ]


def _memos():
    return [
        {
            "memo_id": "m1",
            "kind": "passage",
            "text": "keeping letters is the participant's phrase for not throwing them out.",
            "passage_id": "p2",
        },
        {
            "memo_id": "m2",
            "kind": "provenance",
            "text": "writing letters was read in order. This provenance note must not raise it.",
        },
    ]


def _sweep():
    return [
        {
            "op": "refine_definition",
            "code": "keeping letters",
            "definition": "Holding on to letters instead of discarding them.",
        }
    ]


def test_selection_ranks_frequency_spread_and_analytic_weight():
    selected = select_focused_codes(
        _book(),
        passages=_passages(),
        memos=_memos(),
        sweep_log=_sweep(),
        pins=["c-rare"],
        min_score=1.0,
    )
    by_id = {row["code_id"]: row for row in selected["rows"]}
    assert by_id["c-freq"]["score"] > by_id["c-rare"]["score"]
    assert by_id["c-freq"]["frequency_component"] == 1.0
    assert by_id["c-freq"]["document_share"] > by_id["c-rare"]["document_share"]
    assert by_id["c-memo"]["substantive_memos"] == 1
    assert by_id["c-memo"]["model_judgment"] is True
    assert by_id["c-memo"]["in_vivo"] is True
    assert by_id["c-memo"]["analytic_component"] > by_id["c-rare"]["analytic_component"]
    assert by_id["c-rare"]["substantive_memos"] == 0
    assert by_id["c-stay"]["contrast"] is True
    assert by_id["c-leave"]["contrast"] is False
    assert "c-freq" in selected["raised_code_ids"]
    assert "c-rare" in selected["raised_code_ids"]
    assert "pin" in by_id["c-rare"]["raise_reason"]
    assert "c-stay" in selected["raised_code_ids"]
    assert "contrast" in by_id["c-stay"]["raise_reason"]
    assert "c-also" in selected["not_raised_code_ids"]
    assert len(selected["rows"]) == 6


def test_selection_override_replaces_the_proposal_and_keeps_unknown_ids_out():
    proposed = ["c-freq", "c-stay"]
    accepted, changed = apply_selection_override(["c-freq", "c-stay", "c-rare"], proposed, {"accept": True})
    assert accepted == proposed
    assert changed is False
    replaced, changed = apply_selection_override(
        ["c-freq", "c-stay", "c-rare"],
        proposed,
        {"raised_code_ids": ["c-rare"]},
    )
    assert replaced == ["c-rare"]
    assert changed is True
    with pytest.raises(FocusedError):
        apply_selection_override(["c-freq"], proposed, {"raised_code_ids": ["missing"]})
    with pytest.raises(FocusedError):
        apply_selection_override(["c-freq"], proposed, {"documents": []})


def test_recode_queue_is_seeded_and_puts_assigned_passages_first():
    passages = _passages()
    passages.append(
        {
            "passage_id": "p5",
            "doc_id": "d5",
            "doc_name": "d5",
            "text": "The bags stayed by the door.",
            "start_offset": 0,
            "end_offset": 29,
            "metadata": {"respondent_id": "d5"},
        }
    )
    segments = [
        {"segment_id": "s1", "code_id": "c-freq", "passage_id": "p1", "doc_id": "d1", "quote": passages[0]["text"]},
        {"segment_id": "s2", "code_id": "c-freq", "passage_id": "p2", "doc_id": "d2", "quote": passages[1]["text"]},
    ]
    codes = _book()
    first = build_recode_queue(
        passages, segments, ["c-freq"], codes, seed=7, neighbor_limit=1, outside_limit=0
    )
    again = build_recode_queue(
        passages, segments, ["c-freq"], codes, seed=7, neighbor_limit=1, outside_limit=0
    )
    assert first == again
    purposes = {item["passage_id"]: item["purpose"] for item in first}
    assert purposes["p1"] == "develop"
    assert purposes["p2"] == "develop"
    assert purposes["p5"] == "compete"
    ids = [item["passage_id"] for item in first]
    assert ids.index("p5") < max(ids.index("p1"), ids.index("p2"))


def test_property_citation_attributes_or_flags_and_never_drops():
    passages = _passages()
    supported = cite_note(
        {
            "category_id": "c-freq",
            "name": "timing",
            "text": "Packing in the morning.",
            "document_id": "d1",
            "passage_id": "p1",
            "quote": passages[0]["text"],
        },
        passages,
        kind="property",
        entry_id="prop-0001",
        coded_passage_id="p1",
    )
    assert supported["supported"] is True
    assert supported["quote_status"] == "exact"
    assert supported["passage_id"] == "p1"
    assert supported["document_id"] == "d1"

    moved = cite_note(
        {
            "category_id": "c-freq",
            "name": "elsewhere",
            "text": "The quote is in another passage.",
            "document_id": "d1",
            "passage_id": "p1",
            "quote": passages[2]["text"],
        },
        passages,
        kind="property",
        entry_id="prop-0002",
        coded_passage_id="p1",
    )
    assert moved["passage_id"] == "p3"
    assert moved["document_id"] == "d3"
    assert moved["cited_passage_id"] == "p1"
    assert "evidence_outside_passage" in moved["flags"]
    assert moved["supported"] is True

    missing = cite_note(
        {
            "category_id": "c-freq",
            "name": "hunch",
            "text": "No citation.",
            "document_id": "",
            "passage_id": "",
            "quote": "",
        },
        passages,
        kind="property",
        entry_id="prop-0003",
        coded_passage_id="p1",
    )
    assert missing["supported"] is False
    assert "unsupported" in missing["flags"]
    assert missing["passage_id"] == ""
    assert missing["entry_id"] == "prop-0003"

    invented = cite_note(
        {
            "category_id": "c-freq",
            "name": "invented",
            "text": "The quote is not in the corpus.",
            "document_id": "d1",
            "passage_id": "p1",
            "quote": "this quote is not in any passage",
        },
        passages,
        kind="dimension",
        entry_id="dim-0001",
        coded_passage_id="p1",
    )
    assert invented["supported"] is False
    assert "unsupported" in invented["flags"]
    assert invented["passage_id"] == "p1"
    assert invented["quote_status"] == "unanchored"


def _source(passages=None, codes=None, segments=None, memos=None, sweep=None) -> OpenCodeResult:
    passages = list(passages or _passages())
    codes = list(codes or _book())
    return OpenCodeResult(
        codebook=codes,
        segments=list(segments or []),
        memos=list(memos or []),
        passages=passages,
        saturation=[],
        saturation_report={"judgment": True, "points": []},
        validation_log=[],
        sweep_log=list(sweep or []),
        prompt_log=[],
        metadata={
            "context": {
                "research_question": "How do students describe leaving home?",
                "population": "Students",
                "setting": "",
                "sensitizing_concepts": ["leaving home"],
                "notes": "",
            },
            "unit": "paragraph",
            "seed": 1,
        },
    )


def _segments_for_raised():
    passages = _passages()
    return [
        {
            "segment_id": "s1",
            "code_id": "c-freq",
            "code": "packing bags",
            "passage_id": "p1",
            "doc_id": "d1",
            "quote": passages[0]["text"],
            "quote_status": "exact",
            "review_reasons": [],
        },
        {
            "segment_id": "s2",
            "code_id": "c-freq",
            "code": "packing bags",
            "passage_id": "p2",
            "doc_id": "d2",
            "quote": passages[1]["text"],
            "quote_status": "exact",
            "review_reasons": [],
        },
        {
            "segment_id": "s3",
            "code_id": "c-freq",
            "code": "packing bags",
            "passage_id": "p3",
            "doc_id": "d3",
            "quote": passages[2]["text"],
            "quote_status": "exact",
            "review_reasons": [],
        },
    ]


def _run(mode="echo", *, passages=None, segments=None, interrupt=False, **kwargs):
    coder = FakeFocusedCoder(mode=mode, outside_quote=kwargs.pop("outside_quote", ""))
    source = _source(passages=passages, segments=segments, memos=_memos(), sweep=_sweep())
    result = focused_code_fn(
        source,
        model=coder,
        seed=1,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        interrupt=interrupt,
        **kwargs,
    )
    return coder, result


def test_recoding_keeps_quotes_and_writes_cited_properties(tmp_path):
    coder, result = _run(segments=_segments_for_raised(), output_dir=tmp_path, max_focused_review_calls=8)
    assert coder.call_count == 5
    assert result.metadata["status"] == "complete_for_queue"
    assert result.metadata["stop"] == "complete_for_queue"
    assert result.metadata["coverage"]["corpus_complete"] is False
    assert result.focused_not_queued == ["p4"]
    assert "p4" not in result.purpose_coverage["claim_compete"]["passage_ids"]
    assert result.purpose_coverage["claim_compete"]["calls"] == 0
    assert any(
        "claim_compete" in item and "reallocated to outside" in item for item in result.warnings
    )
    assert result.metadata["performance"]["batch_size"] == 1
    assert "batch_size" not in result.metadata["procedure"]
    assert result.saturation_report["judgment"] is None
    assert result.metadata["saturation_judgment"] is None
    assert {entry["code_id"] for entry in result.codebook} >= {row["code_id"] for row in _book()}
    assert result.properties
    assert all(item["document_id"] and item["passage_id"] and item["quote"] for item in result.properties)
    assert all(item["supported"] for item in result.properties)
    assert result.dimensions
    assert not any(item.get("focused_disagreement") for item in result.segments)
    stored = json.loads((tmp_path / "category_development.json").read_text(encoding="utf-8"))
    assert stored["properties"]
    assert stored["relations"] is None
    assert stored["relations_required"] is False
    assert stored["views"]["category_development.json"].startswith("all records")
    assert result.focused_not_reread == []
    assert all(item["disposition"] == "proposed" for item in result.properties)
    assert all(item["claim_support"] == "undefined" for item in result.properties)


def test_outside_quote_is_attributed_and_missing_quote_is_flagged():
    outside = _passages()[3]["text"]
    _coder, moved = _run(
        "outside",
        segments=_segments_for_raised()[:1],
        outside_quote=outside,
        max_focused_review_calls=4,
    )
    flagged = [
        segment
        for segment in moved.segments
        if "evidence_outside_passage" in (segment.get("review_reasons") or [])
    ]
    assert flagged
    assert flagged[0]["passage_id"] == "p4"
    assert flagged[0]["coded_passage_id"] == "p1"
    assert flagged[0]["quote"]
    located = [item for item in moved.properties if item["name"] == "located elsewhere"]
    assert located
    assert located[0]["passage_id"] == "p4"
    assert located[0]["cited_passage_id"] == "p1"
    assert "evidence_outside_passage" in located[0]["flags"]

    _coder, thin = _run("unsupported", segments=_segments_for_raised()[:1], max_focused_review_calls=4)
    unsupported = [item for item in thin.properties + thin.dimensions if "unsupported" in item["flags"]]
    assert len(unsupported) >= 2
    assert any(item["name"] == "unsupported hunch" for item in unsupported)
    assert all(item["entry_id"] for item in unsupported)


def test_disagreement_keeps_the_initial_assignment():
    _coder, result = _run("drop", segments=_segments_for_raised()[:1], max_focused_review_calls=4)
    kept = [segment for segment in result.segments if segment.get("segment_id") == "s1"]
    assert kept
    assert "focused_disagreement" in kept[0]["review_reasons"]
    assert kept[0]["code_id"] == "c-freq"


def test_allowance_exhaustion_is_incomplete_and_skips_the_model():
    coder, starved = _run(
        segments=_segments_for_raised(),
        max_focused_review_calls=1,
        max_focused_review_tokens=24000,
    )
    assert coder.call_count == 1
    assert starved.metadata["status"] == "complete_for_queue"
    assert starved.metadata["stop"] == "complete_for_queue"
    assert starved.metadata["complete"] is False
    assert len(starved.focused_reread) == 1
    assert starved.focused_not_reread == []
    assert len(starved.focused_not_queued) >= 2
    assert {segment["segment_id"] for segment in starved.segments} >= {"s1", "s2", "s3"}

    silent, tokens = _run(
        segments=_segments_for_raised(),
        max_focused_review_calls=8,
        max_focused_review_tokens=1,
    )
    assert silent.call_count == 0
    assert tokens.metadata["status"] == "incomplete"
    assert tokens.metadata["stop"] == "token_allowance"
    assert set(tokens.focused_not_reread) == {"p1", "p2", "p3"}

    none, both = _run(
        segments=_segments_for_raised(),
        max_focused_review_calls=0,
        max_focused_review_tokens=0,
    )
    assert none.call_count == 0
    assert both.metadata["stop"] == "call_allowance"
    assert both.raised_code_ids

    quiet, empty = _run("empty", segments=_segments_for_raised()[:1], max_focused_review_calls=4)
    assert quiet.call_count == 1
    assert empty.metadata["stop"] == "no_progress"
    assert empty.metadata["status"] == "incomplete"


def test_interrupt_resume_uses_command_and_checks_provenance():
    saver = InMemorySaver()
    coder = FakeFocusedCoder()
    source = _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep())
    paused = focused_code(
        source,
        model=coder,
        seed=3,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        interrupt=True,
        checkpointer=saver,
        thread_id="focused-1",
        max_focused_review_calls=4,
    )
    assert paused.interrupted
    assert paused.interrupt_payload["kind"] == "focused_selection"
    assert "c-stay" in paused.interrupt_payload["raised_code_ids"]
    viewed = read_focused_interrupt(saver, "focused-1")
    assert viewed["payload"]["kind"] == "focused_selection"
    assert coder.call_count == 0

    with pytest.raises(CheckpointMismatch):
        focused_code(
            source,
            model=coder,
            seed=3,
            context={"research_question": "A different question"},
            pins=["c-rare"],
            focused_min_score=1.0,
            focused_neighbor_limit=0,
            resuming=True,
            resume={"accept": True},
            checkpointer=saver,
            thread_id="focused-1",
            max_focused_review_calls=4,
        )
    with pytest.raises(CheckpointMigrationRequired):
        focused_code(
            source,
            model=coder,
            seed=9,
            pins=["c-rare"],
            focused_min_score=1.0,
            focused_neighbor_limit=0,
            resuming=True,
            resume={"accept": True},
            allow_changes=True,
            checkpointer=saver,
            thread_id="focused-1",
            max_focused_review_calls=4,
        )

    graph = build_focused_graph(FocusedRuntime(coder=coder)).compile(checkpointer=saver)

    async def _resume():
        return await graph.ainvoke(
            Command(resume={"raised_code_ids": ["c-freq"]}),
            {"configurable": {"thread_id": "focused-1"}},
        )

    finished_state = run_sync(_resume())
    assert "c-freq" in (finished_state.get("raised_code_ids") or [])
    assert "c-stay" not in (finished_state.get("raised_code_ids") or [])
    assert coder.call_count >= 1
    assert finished_state["metadata"]["saturation_judgment"] is None


def test_api_resume_records_a_performance_continuation():
    saver = InMemorySaver()
    coder = FakeFocusedCoder()
    source = _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep())
    common = dict(
        model=coder,
        seed=3,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        checkpointer=saver,
        thread_id="focused-2",
        max_focused_review_calls=4,
    )
    paused = focused_code(source, interrupt=True, batch_size=1, **common)
    assert paused.interrupted
    with pytest.raises(CheckpointMismatch) as caught:
        focused_code(source, interrupt=True, batch_size=2, resuming=True, resume={"accept": True}, **common)
    assert any(path.startswith("performance.") for path in caught.value.differences)
    paused_claims = focused_code(
        source,
        interrupt=True,
        batch_size=2,
        resuming=True,
        resume={"raised_code_ids": ["c-memo"]},
        allow_changes=True,
        **common,
    )
    assert paused_claims.interrupted is True
    assert paused_claims.interrupt_payload["kind"] == "focused_claims"
    finished = focused_code(
        source,
        interrupt=True,
        batch_size=2,
        resuming=True,
        resume={"leave_claims": True},
        allow_changes=True,
        **common,
    )
    assert finished.interrupted is False
    assert finished.raised_code_ids == ["c-memo"]
    assert finished.metadata["continuations"]
    assert finished.metadata["procedure"]["focused_weight_frequency"] == 1.0
    assert finished.saturation_report["judgment"] is None
    assert "c-memo" in {entry["code_id"] for entry in finished.codebook}
    assert "c-freq" in {entry["code_id"] for entry in finished.codebook}


def test_label_retry_then_keeps_a_valid_code():
    coder, result = _run("retry_label", segments=_segments_for_raised()[:1], max_focused_review_calls=4)
    # The other packing passage shares one evidence token, so the claim slot is
    # not spent. That call is reallocated to outside, and the empty label is
    # retried there too: develop attempt + retry, outside attempt + retry.
    assert coder.call_count == 4
    assert result.purpose_coverage["claim_compete"]["calls"] == 0
    assert result.purpose_coverage["outside"]["calls"] == 1
    labels = {entry["code"] for entry in result.codebook}
    assert "packing bags" in labels
    assert not any(entry["code"] == "(empty label)" for entry in result.codebook)


def test_focused_graph_stays_separate_and_axial_graphs_exist():
    focused = get_graph("focused")
    assert "select_candidates" in focused.nodes
    assert "researcher_checkpoint" in focused.nodes
    assert "recode_worker" in focused.nodes
    assert "claim_checkpoint" in focused.nodes
    open_code = get_graph("open_code")
    themes = get_graph("themes")
    assert "recode_worker" not in open_code.nodes
    assert "select_candidates" not in themes.nodes
    assert "propose_relations" not in open_code.nodes
    assert "propose_relations" not in themes.nodes
    axial = get_graph("axial")
    assert "propose_relations" in axial.nodes
    assert "relation_checkpoint" in axial.nodes
    assert "saturation_checkpoint" in axial.nodes
    assert "export_axial" in axial.nodes
    selective = get_graph("selective")
    assert "score_candidates" in selective.nodes
    assert "core_checkpoint" in selective.nodes
    assert "draft_memos" in selective.nodes
    assert "draft_storyline" in selective.nodes
    assert "export_selective" in selective.nodes
    compared = apply_comparisons(
        {"speaking": {"code": "speaking", "code_id": "speaking"}},
        [],
        [{"op": "relate_categories", "source": "speaking", "target": "silence", "relation": "conditions"}],
    )
    assert compared["recorded"][0]["op"] == "relate_categories"
    assert compared["recorded"][0]["status"] == "recorded"


def test_cli_prints_the_interrupt_payload(tmp_path, capsys):
    path = tmp_path / "focused.sqlite"
    coder = FakeFocusedCoder()
    source = _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep())
    paused = focused_code(
        source,
        model=coder,
        seed=3,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        interrupt=True,
        checkpoint_path=path,
        thread_id="cli-1",
        max_focused_review_calls=4,
    )
    assert paused.interrupted
    from automated_grounded_theory.focused import main

    assert main(["interrupt", "--checkpoint", str(path), "--thread", "cli-1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["payload"]["kind"] == "focused_selection"


def test_reread_budget_is_interleaved_and_reported_per_purpose():
    passages = _passages()
    passages.append(
        {
            "passage_id": "p5",
            "doc_id": "d5",
            "text": "The bags stayed by the door.",
            "start_offset": 0,
            "end_offset": 29,
            "metadata": {"respondent_id": "d5"},
        }
    )
    segments = _segments_for_raised()[:2]
    coder = FakeFocusedCoder()
    result = focused_code(
        _source(passages=passages, codes=[_book()[0]], segments=segments),
        model=coder,
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=1,
        focused_outside_limit=1,
        focused_negative_case_limit=0,
        max_focused_review_calls=3,
        interrupt=False,
    )
    coverage = result.purpose_coverage
    assert coverage["develop"]["calls"] >= 1
    assert coverage["compete"]["reread"] == ["p5"]
    assert coverage["outside"]["calls"] == 1
    assert set(coverage["develop"]["reread"]) | set(coverage["compete"]["reread"]) | set(coverage["outside"]["reread"]) == set(
        result.focused_reread
    )
    assert result.metadata["status"] == "complete_for_queue"
    assert result.metadata["coverage"]["queue_complete"] is True
    assert result.metadata["coverage"]["corpus_complete"] is False
    assert result.focused_not_queued


def test_researcher_incident_can_enter_any_purpose(tmp_path):
    incidents = tmp_path / "focused_incidents.csv"
    incidents.write_text("passage_id,purpose\np4,compete\n", encoding="utf-8")
    saver = InMemorySaver()
    coder = FakeFocusedCoder()
    source = _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep())
    paused = focused_code(
        source,
        model=coder,
        seed=3,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        interrupt=True,
        checkpointer=saver,
        thread_id="incidents",
        max_focused_review_calls=4,
    )
    assert paused.interrupt_payload["kind"] == "focused_selection"
    resumed = focused_code(
        source,
        model=coder,
        seed=3,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        resuming=True,
        resume={"accept": True, "incidents_csv": str(incidents)},
        checkpointer=saver,
        thread_id="incidents",
        max_focused_review_calls=4,
    )
    assert resumed.interrupt_payload["kind"] == "focused_claims"
    compete = resumed.purpose_coverage["compete"]["passage_ids"]
    # p4 is the incident. The claim slot has no evidence overlap, so that call
    # is reallocated onto compete and p1 is queued after the incident.
    assert compete == ["p4", "p1"]
    assert resumed.metadata["claim_test_relevance"]["reallocated_calls"] == 1
    assert resumed.metadata["reread_reallocation"]["reallocated"] is True


def test_negative_cases_include_the_supporting_category():
    from automated_grounded_theory.category_state import negative_case_candidates

    passages = [
        {"passage_id": "p1", "text": "I was going to the station."},
        {"passage_id": "p2", "text": "I was staying despite the plan."},
        {"passage_id": "p3", "text": "The staying lasted all winter."},
    ]
    segments = [
        {"code_id": "c1", "passage_id": "p1"},
        {"code_id": "c1", "passage_id": "p2"},
    ]
    codebook = [
        {"code_id": "c1", "dimensions": [{"name": "leaving", "value": "going"}]},
        {"code_id": "c2", "dimensions": [{"name": "leaving", "value": "staying"}]},
    ]
    claims = [
        {
            "entry_id": "dim-c1",
            "category_id": "c1",
            "name": "leaving",
            "pole": "going",
        }
    ]
    hits = negative_case_candidates(passages, segments, claims, codebook=codebook)
    within = [hit for hit in hits if hit["reason"] == "within_category"]
    outside = [hit for hit in hits if hit["reason"] == "outside_support"]
    assert any(hit["passage_id"] == "p2" and hit["preflagged"] is False for hit in within)
    assert any(hit["passage_id"] == "p3" for hit in outside)


def test_category_edit_marks_dependents_and_keeps_history():
    from automated_grounded_theory.category_state import apply_category_edits, category_revision_id

    entry = {
        "code_id": "c1",
        "code": "packing bags",
        "definition": "Packing.",
        "revision": 1,
    }
    entry["category_revision_id"] = category_revision_id(entry)
    claim = {
        "entry_id": "prop-0001",
        "category_id": "c1",
        "category_revision_id": entry["category_revision_id"],
        "text": "Morning packing.",
        "disposition": "accepted",
        "evidence_condition": "anchored",
        "downstream_eligible": True,
        "flags": [],
    }
    edited = apply_category_edits(
        {"c1": entry},
        [{"segment_id": "s1", "code_id": "c1", "quote": "I packed.", "review_reasons": []}],
        [claim],
        [],
        [{"code_id": "c1", "definition": "Packing in order to leave."}],
    )
    revised = edited["codebook"]["c1"]
    assert revised["definition"] == "Packing in order to leave."
    assert revised["revision_history"][0]["definition"] == "Packing."
    assert revised["category_revision_id"] != entry["category_revision_id"]
    assert edited["properties"][0]["text"] == "Morning packing."
    assert edited["properties"][0]["previous_category_revision_id"] == entry["category_revision_id"]
    assert edited["properties"][0]["needs_review"] is True
    assert edited["properties"][0]["downstream_eligible"] is False
    assert edited["segments"][0]["needs_review"] is True
    assert "category_revised" in edited["segments"][0]["review_reasons"]
    assert edited["segments"][0]["quote"] == "I packed."


def test_taguette_roundtrip_requires_a_mapping_to_rename():
    from automated_grounded_theory.category_state import classify_codebook_roundtrip, code_id_sidecar

    sidecar = code_id_sidecar(
        [
            {"code_id": "c1", "code": "packing bags", "tag": "packing bags"},
            {"code_id": "c2", "code": "crying", "tag": "crying"},
        ]
    )
    ambiguous = classify_codebook_roundtrip(sidecar, ["packing luggage", "crying"])
    actions = {item.get("code_id") or item.get("tag"): item["action"] for item in ambiguous["actions"]}
    assert actions["c1"] == "remove_kept"
    assert actions["packing luggage"] == "ambiguous"
    renamed = classify_codebook_roundtrip(
        sidecar,
        ["packing luggage", "crying"],
        {"renames": [{"code_id": "c1", "tag": "packing luggage"}]},
    )
    by_action = {(item["action"], item.get("code_id"), item.get("tag")) for item in renamed["actions"]}
    assert ("rename", "c1", "packing luggage") in by_action
    assert ("unchanged", "c2", "crying") in by_action


def test_linked_run_preserves_the_parent_and_relations_are_optional():
    from automated_grounded_theory.category_state import (
        load_category_development,
        plan_linked_run,
        plan_relation_pairs,
        record_revision_cycle,
    )

    loaded = load_category_development({"state_version": 1, "properties": [], "dimensions": []})
    assert loaded["relations"] is None
    assert loaded["relations_required"] is False
    plan = plan_linked_run(
        parent_run_id="run-a",
        parent_output_dir="/tmp/run-a",
        additional_passages=[{"passage_id": "p9", "text": "A later interview."}],
        sampling_brief="Ask about staying.",
        new_run_id="run-b",
    )
    assert plan["preserves_parent"] is True
    assert plan["mutates_parent"] is False
    assert plan["relations_required"] is False
    cycled = record_revision_cycle(
        {"properties": [{"entry_id": "prop-0001", "text": "kept", "flags": []}], "dimensions": []},
        cycle="return_to_evidence",
        reason="The definition changed.",
        passage_ids=["p1"],
        claim_ids=["prop-0001"],
    )
    assert cycled["properties"][0]["text"] == "kept"
    assert cycled["properties"][0]["needs_review"] is True
    pairs = plan_relation_pairs([f"c{i}" for i in range(50)], round_size=8, seed=1)
    assert pairs["pair_count"] == 1225
    assert pairs["unexamined_pair_count"] == 1225 - len(pairs["round"])
    assert any(item["band"] == "less_obvious" for item in pairs["round"])
    assert "ceilings" in pairs["allowance_note"]


def test_analytic_ablation_and_live_estimate_make_no_api_call():
    from automated_grounded_theory.evaluation_plan import ablation_selection, estimate_live_cost

    selected = ablation_selection(
        _book(),
        passages=_passages(),
        memos=_memos(),
        sweep_log=_sweep(),
        pins=["c-rare"],
        min_score=1.0,
    )
    assert "memo" in selected["full"]["analytic_features"]
    assert "memo" not in selected["ablated"]["memo"]["analytic_features"]
    assert "verbose" in selected["memo_note"]
    estimate = estimate_live_cost()
    assert estimate["live_calls"] is False
    assert estimate["model"] == "grok-4.20-0309-non-reasoning"
    assert estimate["within_target"] is True
    assert estimate["total_usd"] <= 2.0
    from scripts.evaluate_focused import main

    assert main(["--live"]) == 2
