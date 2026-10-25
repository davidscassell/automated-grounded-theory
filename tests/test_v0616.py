"""0.6.16: apply low-confidence codes, theme prompts, merge target, nonsubstantive replies, memos."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from automated_grounded_theory import open_code
from automated_grounded_theory.codebook import apply_sweep, commit_passage, interpret_coding
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.prompts import (
    CODING_SYSTEM,
    THEME_SYSTEM,
    build_theme_review_messages,
    format_theme_codebook,
)
from automated_grounded_theory.records import POLICY_VERSION, materialize_records
from automated_grounded_theory.schemas import CONFIDENCE_HELP
from automated_grounded_theory.segment import plan_coding
from automated_grounded_theory.themes import assign_codes, assignment_feedback
from automated_grounded_theory.validation import exclusion_reason, memo_review_reasons, nonsubstantive_kind
from automated_grounded_theory.version import __version__
from tests.corpus import CONTEXT
from tests.fake_llm import FakeCoder


def test_version_is_0_6_16():
    assert __version__ == "0.9.31"
    assert POLICY_VERSION == "0.9.31"
    assert "not a probability" in CONFIDENCE_HELP


def _passage(text: str = "I packed my suitcase.") -> dict:
    return {
        "passage_id": "p1",
        "doc_id": "d1",
        "text": text,
        "start_offset": 0,
        "end_offset": len(text),
        "prev_text": "",
        "next_text": "",
        "question": "What did you do about leaving?",
    }


def _coding(quote: str, label: str = "packing bags", confidence: float = 0.9, memo: str = "") -> dict:
    return {
        "codes": [
            {
                "action": "new",
                "code": label,
                "definition": f"Incidents of {label}.",
                "quote": quote,
                "confidence": confidence,
                "rationale": "The incident is in the participant's words.",
            }
        ],
        "memo": memo,
        "memo_confidence": 0.9,
    }


def _options(**extra):
    options = dict(max_codes=6, require_gerund=False, long_passage_chars=2000)
    options.update(extra)
    return options


def test_low_confidence_coding_is_applied_and_kept_for_review():
    interpreted = interpret_coding(
        _coding("I packed my suitcase.", confidence=0.4),
        _passage(),
        {},
        **_options(),
    )
    assert [item["code"] for item in interpreted["accepted"]] == ["packing bags"]
    kept = interpreted["accepted"][0]
    assert kept["needs_review"] is True
    assert "low_confidence" in kept["review_reasons"]
    reviews = [item for item in interpreted["review"] if item.get("reason") == "low_confidence"]
    assert reviews
    assert reviews[0]["status"] == "review"
    applied = [
        item
        for item in interpreted["ops"]
        if item.get("status") == "applied" and item.get("op") != "memo"
    ]
    assert applied

    spared = interpret_coding(
        _coding("I packed my suitcase.", confidence=0.4),
        _passage(),
        {},
        **_options(review_below=0.3),
    )
    assert "low_confidence" not in spared["accepted"][0]["review_reasons"]


def test_theme_prompts_allow_multiple_memberships_and_a_soft_share():
    assert "exactly one theme" not in THEME_SYSTEM
    assert "focused coding" not in THEME_SYSTEM.casefold()
    assert "more than one" in THEME_SYSTEM
    assert "primary" in THEME_SYSTEM
    assert "do not reread" in THEME_SYSTEM
    book = format_theme_codebook([{"code": "packing bags", "definition": "Bagging.", "frequency": 1}])
    assert "exactly one theme" not in book
    assert "more than one theme" in book
    assert "first theme is primary" in book
    messages = build_theme_review_messages(
        context=ResearchContext(research_question="How do people leave?"),
        themes=[{"name": "Leaving", "definition": "Leaving home.", "codes": ["packing bags"]}],
        groups=[],
        max_theme_share=0.30,
        prompt_cache=False,
    )
    note = messages[-1].content
    assert "soft presentation target" in note
    assert "does not block a fold" in note
    assert "must not hold more than" not in note
    assert "Do not build a group with more than" not in note

    proposal = {
        "themes": [
            {
                "name": "Leaving",
                "definition": "Getting ready to go.",
                "codes": ["packing bags", "crying at the station"],
                "confidence": 0.9,
            },
            {
                "name": "Emotion",
                "definition": "Feeling the departure.",
                "codes": ["crying at the station"],
                "confidence": 0.8,
            },
        ]
    }
    codebook = [{"code": "packing bags"}, {"code": "crying at the station"}]
    themes, report = assign_codes(proposal, codebook)
    by_name = {theme["name"]: theme for theme in themes}
    assert "crying at the station" in by_name["Leaving"]["codes"]
    assert "crying at the station" in by_name["Emotion"]["codes"]
    assert by_name["Leaving"]["primary_codes"] == ["packing bags", "crying at the station"]
    assert by_name["Emotion"]["primary_codes"] == []
    assert report["missing"] == []
    assert "crying at the station" in report["duplicates"]
    feedback = assignment_feedback(report)
    assert "Only the first assignment was kept" not in feedback
    assert "primary" in feedback


def _entry(label: str, **extra) -> dict:
    entry = {
        "code": label,
        "key": label,
        "definition": label,
        "in_vivo": False,
        "gerund_ok": True,
        "frequency": 1,
        "document_ids": ["d"],
        "examples": [],
        "first_seen": "p",
        "history": [{"event": "created", "code": label}],
        "aliases": list(extra.pop("aliases", [])),
        "flagged_overloaded": False,
        "relation_hints": [],
    }
    entry.update(extra)
    return entry


def _segment(label: str, quote: str) -> dict:
    return {
        "segment_id": label,
        "code": label,
        "code_key": label,
        "quote": quote,
        "doc_id": "d",
        "passage_id": "p",
        "provenance": {"coded_passage_id": "p", "note": f"from {label}"},
    }


def _merge(source: str, target: str, **extra) -> dict:
    row = {
        "source": source,
        "target": target,
        "rationale": "same incident",
        "confidence": 0.9,
        "same_action": True,
        "same_object": True,
        "contrast_checked": True,
        "material_contrast": False,
    }
    row.update(extra)
    return row


def test_merge_survivor_follows_the_model_target():
    codebook = {
        "packing bags": _entry("packing bags", aliases=["bagging"]),
        "packing bags up": _entry("packing bags up"),
    }
    segments = [
        _segment("packing bags", "I packed my suitcase."),
        _segment("packing bags up", "Packing bags up with boxes."),
    ]
    applied = apply_sweep(
        codebook,
        segments,
        merges=[_merge("packing bags", "packing bags up")],
        candidates=[{"source": "packing bags", "target": "packing bags up", "score": 0.7}],
        definitions=[{"code": "packing bags up", "definition": "Bagging possessions while leaving."}],
    )
    survivor = applied["codebook"]["packing bags up"]
    assert list(applied["codebook"]) == ["packing bags up"]
    assert survivor["definition"] == "Bagging possessions while leaving."
    assert survivor["aliases"] == ["bagging", "packing bags"]
    assert any(event.get("event") == "merged_from" for event in survivor["history"])
    assert any(event.get("event") == "survivor_chosen" and event.get("rule") == "target" for event in survivor["history"])
    assert any(event.get("event") == "created" and event.get("code") == "packing bags" for event in survivor["history"])
    assert {item["code"] for item in applied["segments"]} == {"packing bags up"}
    assert applied["segments"][0]["provenance"] == {"coded_passage_id": "p", "note": "from packing bags"}

    retired_definition = apply_sweep(
        codebook,
        segments,
        merges=[_merge("packing bags", "packing bags up")],
        candidates=[{"source": "packing bags", "target": "packing bags up"}],
        definitions=[{"code": "packing bags", "definition": "This wording belongs to the retired code."}],
    )
    assert retired_definition["codebook"]["packing bags up"]["definition"] == "packing bags up"


def test_trivial_and_conflicting_merges_tie_break_lexicographically():
    spelling = {
        "packing bag": _entry("packing bag", definition="Bagging possessions."),
        "packing bags": _entry("packing bags", definition="Bagging possessions."),
    }
    trivial = apply_sweep(
        spelling,
        [_segment("packing bag", "I packed a bag."), _segment("packing bags", "I packed my suitcase.")],
        merges=[_merge("packing bags", "packing bag", trivial=True)],
        candidates=[{"source": "packing bags", "target": "packing bag"}],
    )
    assert list(trivial["codebook"]) == ["packing bag"]
    history = trivial["codebook"]["packing bag"]["history"]
    assert any(event.get("rule") == "lexicographic" for event in history)
    assert "packing bags" in trivial["codebook"]["packing bag"]["aliases"]

    codebook = {
        "packing bags": _entry("packing bags", definition="Bagging possessions."),
        "packing bags up": _entry("packing bags up", definition="Bagging possessions."),
    }
    conflicted = apply_sweep(
        codebook,
        [
            _segment("packing bags", "I packed my suitcase."),
            _segment("packing bags up", "Packing bags up with boxes."),
        ],
        merges=[
            _merge("packing bags", "packing bags up"),
            _merge("packing bags up", "packing bags"),
        ],
        candidates=[
            {"source": "packing bags", "target": "packing bags up"},
            {"source": "packing bags up", "target": "packing bags"},
        ],
    )
    assert list(conflicted["codebook"]) == ["packing bags"]
    chosen = conflicted["codebook"]["packing bags"]["history"]
    assert any(event.get("rule") == "tie_break" for event in chosen)


def test_nonsubstantive_replies_are_coded_and_flagged():
    assert exclusion_reason("   ") == "empty"
    assert exclusion_reason("n/a") is None
    assert nonsubstantive_kind("I don't know") == "uncertainty"
    assert nonsubstantive_kind("prefer not to say") == "refusal"
    assert nonsubstantive_kind("I don't know what the policy says about leave") is None
    for substantive in ("我辞职了", "Я уволился", "تركت العمل", "I quit."):
        assert nonsubstantive_kind(substantive) is None
        assert exclusion_reason(substantive) is None
    assert nonsubstantive_kind("...") == "no_substantive_text"
    assert exclusion_reason("n/a", code_nonsubstantive=False) == "non_answer"

    plan = plan_coding(
        [{"id": "a", "text": "I don't know"}, {"id": "b", "text": "   "}, {"id": "c", "text": "n/a"}],
        unit="paragraph",
        seed=1,
    )
    by_text = {passage["text"].strip(): passage for passage in plan.passages}
    assert by_text[""]["skipped"] is True
    assert by_text[""]["exclusion_reason"] == "empty"
    assert by_text["I don't know"]["nonsubstantive_kind"] == "uncertainty"
    assert by_text["n/a"]["nonsubstantive"] is True
    assert by_text["I don't know"]["passage_id"] and not by_text["I don't know"]["skipped"]
    assert len(plan.queue) == 2

    held = plan_coding(
        [{"id": "a", "text": "I don't know"}],
        unit="paragraph",
        seed=1,
        code_nonsubstantive=False,
    )
    assert held.queue == []
    assert held.passages[0]["exclusion_reason"] == "non_answer"

    result = open_code(
        ["n/a", "   "],
        CONTEXT,
        model=FakeCoder(),
        seed=1,
        sweep_every=25,
        batch_size=1,
        concurrency=1,
    )
    assert result.metadata["code_nonsubstantive"] is True
    records = materialize_records(
        documents=[],
        passages=result.passages,
        segments=result.segments,
        codebook=result.codebook,
    )
    kinds = {row["nonsubstantive_kind"] for row in records["incidents"]}
    assert "non_answer" in kinds
    assert any(row["exclusion_reason"] == "empty" for row in records["incidents"])

    passage = _passage("I don't know")
    passage["nonsubstantive"] = True
    passage["nonsubstantive_kind"] = "uncertainty"
    interpreted = interpret_coding(
        _coding("I don't know", label="expressing uncertainty about leaving"),
        passage,
        {},
        **_options(),
    )
    committed = commit_passage({}, [], [], {}, {}, passage, interpreted)
    assert committed["segments"][0]["nonsubstantive"] is True
    assert committed["segments"][0]["nonsubstantive_kind"] == "uncertainty"
    assert "expressing uncertainty" in CODING_SYSTEM
    assert "declining to answer" in CODING_SYSTEM


def test_memos_are_kept_when_the_template_is_loose():
    assert memo_review_reasons("none") is None
    assert memo_review_reasons("n/a") is None
    assert memo_review_reasons("No property, dimension, or comparison was recorded.") is None
    short = "This incident adds time pressure versus packing."
    assert memo_review_reasons(short) == ["thin"]
    long = " ".join(["Compared with packing, the timing varies here."] * 4)
    assert memo_review_reasons(long) == []

    interpreted = interpret_coding(
        _coding("I packed my suitcase.", memo=short),
        _passage(),
        {},
        **_options(),
    )
    assert interpreted["memo"] == short
    assert interpreted["memo_reasons"] == ["thin"]
    committed = commit_passage({}, [], [], {}, {}, _passage(), interpreted)
    memo = committed["memos"][0]
    assert memo["text"] == short
    assert memo["needs_review"] is True
    assert memo["review_reasons"] == ["thin"]

    low = interpret_coding(
        {**_coding("I packed my suitcase.", memo=long), "memo_confidence": 0.2},
        _passage(),
        {},
        **_options(),
    )
    assert low["memo"] == long
    assert "low_confidence" in low["memo_reasons"]
    assert "thin" not in low["memo_reasons"]
    kept = commit_passage({}, [], [], {}, {}, _passage(), low)
    assert kept["memos"][0]["text"] == long
    assert "low_confidence" in kept["memos"][0]["review_reasons"]


def test_docs_describe_open_coding_and_a_theme_report():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    method = (ROOT / "docs/methodology.md").read_text(encoding="utf-8")
    for text in (readme, method):
        assert "0.5.4" in text
        assert "simulated fake-coder" in text.casefold()
        assert "one step of lag" in text
        assert "does not reread" in text
    assert "not a probability" in readme
    assert "code_nonsubstantive" in readme
    assert "not the theme report" in method.casefold()
    assert "judgment" in method
    assert "build_themes" in readme
