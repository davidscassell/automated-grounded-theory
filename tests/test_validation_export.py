import csv
from xml.etree import ElementTree as ET

from automated_grounded_theory.codebook import apply_sweep
from automated_grounded_theory.export import QDC_NAMESPACE, codebook_qdc, export_result, taguette_tag
from automated_grounded_theory.results import OpenCodeResult
from automated_grounded_theory.validation import gerund_ok, is_generic_code, locate_quote


def _entry(label: str) -> dict:
    return {
        "code": label,
        "key": label,
        "definition": label,
        "in_vivo": False,
        "gerund_ok": True,
        "frequency": 1,
        "document_ids": ["d"],
        "examples": [],
        "first_seen": "p",
        "history": [],
        "flagged_overloaded": False,
        "relation_hints": [],
    }


def _segment(label: str, quote: str) -> dict:
    return {
        "segment_id": label,
        "code": label,
        "code_key": label,
        "quote": quote,
        "doc_id": "d",
        "passage_id": "p",
        "doc_start": 0,
        "doc_end": len(quote),
    }


def test_two_incidents_and_attribute_leak():
    from automated_grounded_theory.codebook import interpret_coding

    passage = {
        "text": "The team is great. Upper management is another story.",
        "start_offset": 0,
        "metadata": {"department": "cardiology"},
    }
    coding = {
        "codes": [
            {
                "action": "new",
                "code": "praising the team",
                "definition": "Praise for the immediate team.",
                "quote": "The team is great.",
                "confidence": 0.9,
            },
            {
                "action": "new",
                "code": "criticizing management",
                "definition": "Criticism of management from cardiology.",
                "quote": "Upper management is another story.",
                "confidence": 0.8,
            },
        ],
        "memo": "",
    }
    interpreted = interpret_coding(
        coding,
        passage,
        {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
    )
    assert [item["code"] for item in interpreted["accepted"]] == ["praising the team", "criticizing management"]
    leaked = interpreted["accepted"][1]["review_reasons"]
    assert "definition_contains_respondent_attribute" in leaked


def test_quote_must_be_in_the_passage():
    passage = "I packed my suitcase."
    assert locate_quote("I packed my suitcase.", passage) == (0, len(passage))
    assert locate_quote("I packed\nmy suitcase.", "I packed\nmy suitcase.") == (0, len("I packed\nmy suitcase."))
    assert locate_quote("I packed my suitcase.", "I packed  my suitcase.") == (0, len("I packed  my suitcase."))
    assert locate_quote("not in the text", passage) is None
    assert gerund_ok("packing bags")
    assert not gerund_ok("bags")
    assert is_generic_code("experience")
    assert not is_generic_code("packing bags")


def test_merge_map_relabels_earlier_segments():
    codebook = {"packing bags": _entry("packing bags"), "packing bags up": _entry("packing bags up")}
    segments = [
        _segment("packing bags", "I packed my suitcase."),
        _segment("packing bags up", "Packing bags up with boxes took all night."),
    ]
    applied = apply_sweep(
        codebook,
        segments,
        merges=[{
            "source": "packing bags",
            "target": "packing bags up",
            "rationale": "same incident",
            "same_action": True,
            "same_object": True,
            "contrast_checked": True,
            "material_contrast": False,
        }],
        candidates=[{"source": "packing bags", "target": "packing bags up", "score": 0.7}],
        definitions=[{"code": "packing bags up", "definition": "Bagging possessions while leaving."}],
    )
    assert list(applied["codebook"]) == ["packing bags"]
    assert applied["codebook"]["packing bags"]["frequency"] == 2
    assert applied["codebook"]["packing bags"]["definition"] == "packing bags"
    sharpened = apply_sweep(
        codebook,
        segments,
        merges=[{
            "source": "packing bags",
            "target": "packing bags up",
            "rationale": "same incident",
            "same_action": True,
            "same_object": True,
            "contrast_checked": True,
            "material_contrast": False,
        }],
        candidates=[{"source": "packing bags", "target": "packing bags up", "score": 0.7}],
        definitions=[{"code": "packing bags", "definition": "Bagging possessions while leaving."}],
    )
    assert sharpened["codebook"]["packing bags"]["definition"] == "Bagging possessions while leaving."
    assert {item["code"] for item in applied["segments"]} == {"packing bags"}


def test_sweep_may_propose_a_pair_that_was_not_a_candidate():
    codebook = {"crying at the station": _entry("crying at the station"), "starting a new job": _entry("starting a new job")}
    segments = [
        _segment("crying at the station", "I was crying."),
        _segment("starting a new job", "They hired me."),
    ]
    applied = apply_sweep(
        codebook,
        segments,
        merges=[{"source": "crying at the station", "target": "starting a new job", "rationale": "same incident"}],
        candidates=[],
    )
    assert applied["accepted_merges"] == []
    assert applied["rejected_merges"][0]["reason"] == "aspects_differ"
    assert set(applied["codebook"]) == {"crying at the station", "starting a new job"}
    unknown = apply_sweep(
        codebook,
        segments,
        merges=[{"source": "not a code", "target": "starting a new job"}],
        candidates=[],
    )
    assert unknown["accepted_merges"] == []
    assert len(unknown["codebook"]) == 2


def test_taguette_csv_and_qdc(tmp_path):
    tag = taguette_tag("leaving\\home")
    assert "\\" not in tag
    assert taguette_tag("leaving.home") == "leaving.home"
    result = OpenCodeResult(
        codebook=[
            {
                "code": 'leaving.home',
                "definition": 'Says "hello" & <goodbye>',
                "in_vivo": False,
                "gerund_ok": True,
                "frequency": 1,
                "document_ids": ["alex"],
                "examples": [{"quote": "I left", "doc_id": "alex"}],
                "first_seen": "alex-p0000",
                "flagged_overloaded": False,
            }
        ],
        segments=[
            {
                "segment_id": "s00001",
                "highlight_id": "h00001",
                "doc_id": "alex",
                "doc_name": "Alex",
                "passage_id": "alex-p0000",
                "code": "leaving.home",
                "quote": "I left",
                "doc_start": 0,
                "doc_end": 6,
                "in_vivo": False,
                "action": "new",
                "confidence": 0.9,
                "needs_review": False,
                "review_reasons": [],
                "rationale": "",
            }
        ],
        memos=[{"memo_id": "m00001", "kind": "passage", "doc_id": "alex", "passage_id": "alex-p0000", "sweep_index": None, "text": "A memo."}],
        passages=[],
        saturation=[{"batch": 1, "passages_in_batch": 1, "new_codes": 1, "new_codes_per_passage": 1, "cumulative_passages": 1, "cumulative_codes": 1}],
        saturation_report={"judgment": None, "note": "reported only", "points": []},
        validation_log=[],
        sweep_log=[],
        prompt_log=[],
        metadata={"seed": 1},
    )
    paths = export_result(result, tmp_path)
    with open(paths["taguette_codebook.csv"], encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["tag", "description", "number of highlights", "number of documents"]
    assert "\\" not in rows[1][0]
    assert "." in rows[1][0]
    root = ET.fromstring(codebook_qdc(result.codebook))
    assert root.tag == f"{{{QDC_NAMESPACE}}}CodeBook"
    description = root.find(f".//{{{QDC_NAMESPACE}}}Description")
    assert description is not None
    assert description.text == 'Says "hello" & <goodbye>\n\nExamples:\n- "I left" (alex)'
