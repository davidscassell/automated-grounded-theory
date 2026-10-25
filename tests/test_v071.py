"""0.7.1 focused-coding fixes: selection, reread shares, ids, confirmations, export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from automated_grounded_theory import __version__, focused_code
from automated_grounded_theory.focused import FOCUSED_SYSTEM, plan_recode_queue, select_focused_codes
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.results import load_run
from automated_grounded_theory.schemas import FocusedDimensionNote, FocusedPropertyNote
from langgraph.checkpoint.memory import InMemorySaver

from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _book, _entry, _passages, _source


def test_version_is_071():
    assert __version__ == "0.9.29"
    assert POLICY_VERSION == "0.9.29"


def _tied_codes(n=6):
    return [
        _entry(f"c{i}", f"code {i}", 2, ["d1"])
        for i in range(n)
    ]


def test_tied_scores_do_not_raise_every_code():
    selected = select_focused_codes(_tied_codes(6), passages=_passages()[:1])
    assert selected["selection_rule"] == "top_k"
    assert selected["tie_policy"] == "code_id"
    assert selected["raised_code_ids"] == ["c0", "c1", "c2"]
    assert selected["not_raised_code_ids"] == ["c3", "c4", "c5"]
    assert selected["raised_fraction"] < 1
    assert selected["warnings"] == []

    warned = select_focused_codes(_tied_codes(4), passages=_passages()[:1], selection_k=3)
    assert len(warned["raised_code_ids"]) == 3
    assert warned["not_raised_code_ids"]
    assert warned["warnings"]
    assert "3 of 4" in warned["warnings"][0]

    opted = select_focused_codes(_tied_codes(4), passages=_passages()[:1], raise_all=True)
    assert len(opted["raised_code_ids"]) == 4
    assert any("focused_raise_all" in note for note in opted["warnings"])

    quantile = select_focused_codes(
        _tied_codes(6),
        passages=_passages()[:1],
        selection_rule="quantile",
        selection_quantile=0.5,
    )
    assert len(quantile["raised_code_ids"]) == 3
    assert quantile["selection_rule"] == "quantile"


def test_default_selection_warns_and_leaves_codes_unraised():
    coder = FakeFocusedCoder()
    result = focused_code(
        _source(passages=_passages()[:1], codes=_tied_codes(4)),
        model=coder,
        seed=1,
        focused_selection_k=3,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=0,
        interrupt=False,
    )
    assert coder.call_count == 0
    assert len(result.raised_code_ids) == 3
    assert result.not_raised_code_ids
    assert any("3 of 4" in note for note in result.warnings)

    everyone = focused_code(
        _source(passages=_passages()[:1], codes=_tied_codes(4)),
        model=coder,
        seed=1,
        focused_raise_all=True,
        max_focused_review_calls=0,
        interrupt=False,
    )
    assert len(everyone.raised_code_ids) == 4
    assert any("focused_raise_all" in note for note in everyone.warnings)


def test_empty_compete_and_outside_are_filled_or_reallocated():
    passages = _passages()[:3]
    segments = [
        {
            "segment_id": "s1",
            "code_id": "c-freq",
            "passage_id": "p1",
            "doc_id": "d1",
            "quote": passages[0]["text"],
            "confidence": 0.9,
        },
        {
            "segment_id": "s2",
            "code_id": "c-other",
            "passage_id": "p1",
            "doc_id": "d1",
            "quote": passages[0]["text"],
            "confidence": 0.9,
        },
        {
            "segment_id": "s3",
            "code_id": "c-freq",
            "passage_id": "p2",
            "doc_id": "d2",
            "quote": passages[1]["text"],
            "confidence": 0.2,
        },
        {
            "segment_id": "s4",
            "code_id": "c-freq",
            "passage_id": "p3",
            "doc_id": "d3",
            "quote": passages[2]["text"],
            "confidence": 0.8,
        },
    ]
    codes = [
        _entry("c-freq", "packing bags", 3, ["d1", "d2", "d3"]),
        _entry("c-other", "writing letters", 1, ["d1"]),
    ]
    plan = plan_recode_queue(
        passages,
        segments,
        ["c-freq", "c-other"],
        codes,
        seed=1,
        neighbor_limit=0,
        outside_limit=1,
        negative_case_limit=1,
    )
    by_purpose = {}
    for item in plan["queue"]:
        by_purpose.setdefault(item["purpose"], []).append(item)
    assert by_purpose["compete"][0]["source"] == "alternative_code"
    assert by_purpose["compete"][0]["passage_id"] == "p1"
    assert by_purpose["outside"][0]["source"] == "low_confidence"
    assert by_purpose["outside"][0]["passage_id"] == "p2"
    assert plan["reallocation"]["reallocated"] is False

    opposite = plan_recode_queue(
        [
            {"passage_id": "p1", "doc_id": "d1", "text": "I was staying despite the plan."},
        ],
        [{"segment_id": "s1", "code_id": "c-leave", "passage_id": "p1", "confidence": 0.4}],
        ["c-leave"],
        [
            _entry("c-leave", "leaving home", 1, ["d1"], dimensions=[{"name": "leaving", "value": "going"}]),
            _entry("c-stay", "staying home", 1, ["d1"], dimensions=[{"name": "leaving", "value": "staying"}]),
        ],
        seed=1,
        neighbor_limit=1,
        outside_limit=0,
        negative_case_limit=0,
    )
    assert opposite["queue"][0]["purpose"] == "compete"
    assert opposite["queue"][0]["source"] == "opposite_pole"
    assert "outside" in opposite["reallocation"]["purposes"]

    empty = plan_recode_queue(
        passages,
        segments[:1],
        ["c-freq"],
        codes,
        seed=1,
        neighbor_limit=0,
        outside_limit=0,
        negative_case_limit=0,
    )
    assert empty["reallocation"]["reallocated"] is True
    assert set(empty["reallocation"]["purposes"]) >= {"compete", "outside"}
    assert empty["warnings"]
    assert "reallocated" in empty["warnings"][0]

    coder = FakeFocusedCoder()
    result = focused_code(
        _source(passages=passages, codes=codes[:1], segments=segments[:1]),
        model=coder,
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=1,
        interrupt=False,
    )
    assert result.purpose_coverage["compete"]["reallocated_away"] is True
    assert result.purpose_coverage["outside"]["reallocated_away"] is True
    assert result.purpose_coverage["reallocation"]["reallocated"] is True
    assert any("reallocated" in note for note in result.warnings)


def test_loaded_run_continues_segment_and_memo_ids(tmp_path):
    passages = _passages()[:1]
    text = passages[0]["text"]
    source = _source(
        passages=passages,
        codes=[_book()[0]],
        segments=[
            {
                "segment_id": "s00007",
                "highlight_id": "h00004",
                "code_id": "c-freq",
                "code": "packing bags",
                "passage_id": "p1",
                "doc_id": "d1",
                "quote": text,
                "quote_status": "exact",
                "doc_start": 0,
                "doc_end": len(text),
            }
        ],
        memos=[{"memo_id": "m00003", "kind": "passage", "passage_id": "p1", "text": "An earlier analytic note."}],
    )
    place = tmp_path / "open"
    place.mkdir()
    (place / "run.json").write_text(json.dumps(source.to_dict()), encoding="utf-8")
    loaded = load_run(place)
    result = focused_code(
        loaded,
        model=FakeFocusedCoder(mode="new_label"),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=2,
        interrupt=False,
        output_dir=tmp_path / "focused",
    )
    ids = {segment["segment_id"] for segment in result.segments}
    assert "s00007" in ids
    assert "s00008" in ids
    assert "s00001" not in ids
    added = next(segment for segment in result.segments if segment["segment_id"] == "s00008")
    assert added["highlight_id"] == "h00005"
    memo_ids = {memo["memo_id"] for memo in result.memos}
    assert "m00003" in memo_ids
    assert "m00004" in memo_ids
    assert all(len(memo_id) == 6 and memo_id.startswith("m") for memo_id in memo_ids)


def test_reapplied_code_confirms_and_a_different_span_is_a_proposal():
    passages = _passages()[:1]
    text = passages[0]["text"]
    segments = [
        {
            "segment_id": "s00002",
            "code_id": "c-freq",
            "code": "packing bags",
            "passage_id": "p1",
            "doc_id": "d1",
            "quote": text,
            "quote_status": "exact",
            "doc_start": 0,
            "doc_end": len(text),
            "review_reasons": [],
        }
    ]
    confirmed = focused_code(
        _source(passages=passages, codes=[_book()[0]], segments=segments),
        model=FakeFocusedCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=2,
        interrupt=False,
    )
    same = [segment for segment in confirmed.segments if segment["code_id"] == "c-freq"]
    assert len(same) == 1
    assert same[0]["segment_id"] == "s00002"
    assert same[0]["quote"] == text
    assert same[0]["doc_start"] == 0
    assert same[0]["doc_end"] == len(text)
    assert same[0]["focused_confirmation"] is True
    assert same[0]["provenance"]["focused"]["confirmed"] is True
    assert same[0]["confirmations"][0]["stage"] == "focused"

    proposed = focused_code(
        _source(passages=passages, codes=[_book()[0]], segments=[dict(segments[0])]),
        model=FakeFocusedCoder(mode="alt_span"),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=2,
        interrupt=False,
    )
    original = next(segment for segment in proposed.segments if segment["segment_id"] == "s00002")
    assert original["quote"] == text
    assert original["doc_start"] == 0
    assert original["doc_end"] == len(text)
    assert not original.get("focused_confirmation")
    assert [segment for segment in proposed.segments if segment.get("code_id") == "c-freq"] == [original]
    proposals = list(original.get("span_proposals") or [])
    assert len(proposals) == 1
    assert proposals[0]["original_segment_id"] == "s00002"
    assert proposals[0]["quote"] != text
    assert proposals[0]["narrowed_span"] is True
    assert "focused_span_proposal" in original["review_reasons"]
    assert "narrowed_span" in original["review_reasons"]
    assert original["needs_review"] is True


def test_prompt_asks_for_analytic_claims_and_export_writes_them(tmp_path):
    assert "analytic properties" in FOCUSED_SYSTEM
    assert "competing interpretation" in FOCUSED_SYSTEM
    assert "descriptive summary" in FOCUSED_SYSTEM
    assert "Analytic property" in FocusedPropertyNote.model_fields["name"].description
    assert "Analytic dimension" in FocusedDimensionNote.model_fields["name"].description

    passages = _passages()[:1]
    text = passages[0]["text"]
    saver = InMemorySaver()
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
    common = dict(
        model=FakeFocusedCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=2,
        checkpointer=saver,
        thread_id="export-claims",
        output_dir=tmp_path,
    )
    paused = focused_code(source, interrupt=True, **common)
    assert paused.interrupt_payload["kind"] == "focused_selection"
    claims = focused_code(source, interrupt=True, resuming=True, resume={"accept": True}, **common)
    assert claims.interrupt_payload["kind"] == "focused_claims"
    finished = focused_code(
        source,
        interrupt=True,
        resuming=True,
        resume={"accept_claims": True},
        **common,
    )
    assert finished.interrupted is False
    assert finished.properties
    log = finished.prompt_log[0]
    assert log["raw_response"] == "{}"
    assert log["response"]["passages"]
    assert any("analytic properties" in message["content"] for message in log["messages"])
    with (tmp_path / "focused_claims.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["name"]
    assert rows[0]["quote"]
    downstream = json.loads((tmp_path / "downstream_claims.json").read_text(encoding="utf-8"))
    assert downstream["uses"]
    assert downstream["records"]
    assert all(row["downstream_eligible"] for row in downstream["records"])
    segments = json.loads((tmp_path / "coded_segments.json").read_text(encoding="utf-8"))
    assignments = json.loads((tmp_path / "assignments.json").read_text(encoding="utf-8"))
    assert segments and assignments
    assert segments[0]["segment_id"] == assignments[0]["segment_id"]
    stored_log = json.loads((tmp_path / "prompt_log.json").read_text(encoding="utf-8"))
    assert stored_log[0]["raw_response"] == "{}"
    assert stored_log[0]["response"]["passages"]
