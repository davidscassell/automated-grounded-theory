from pathlib import Path

import pytest

from automated_grounded_theory.context import PresetCodesError, parse_context
from automated_grounded_theory.segment import plan_coding


def test_context_rejects_preset_codes():
    with pytest.raises(PresetCodesError):
        parse_context({"research_question": "What is happening?", "codes": ["leaving"]})


def test_context_string_is_the_question():
    context = parse_context("How do people leave?")
    assert context.research_question == "How do people leave?"
    assert context.sensitizing_concepts == []


def test_paragraphs_keep_offsets_and_neighbors():
    text = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
    plan = plan_coding([{"id": "doc", "text": text}], unit="paragraph", seed=1)
    passages = sorted(plan.passages, key=lambda item: item["position"])
    assert [item["text"] for item in passages] == [
        "Alpha paragraph.",
        "Beta paragraph.",
        "Gamma paragraph.",
    ]
    assert text[passages[1]["start_offset"] : passages[1]["end_offset"]] == "Beta paragraph."
    assert passages[1]["prev_text"] == "Alpha paragraph."
    assert passages[1]["next_text"] == "Gamma paragraph."


def test_sentence_keeps_abbreviation():
    plan = plan_coding(["Dr. Smith left home. I cried."], unit="sentence", seed=0)
    assert [item["text"] for item in plan.passages] == ["Dr. Smith left home.", "I cried."]


def test_speaker_turns():
    text = "Alex: I packed my suitcase before dawn.\nIt took all morning.\n\nSam: I was crying at the station.\n"
    plan = plan_coding([text], unit="speaker_turn", seed=0)
    assert len(plan.passages) == 2
    assert plan.passages[0]["text"].startswith("Alex:")
    assert "all morning" in plan.passages[0]["text"]
    assert plan.passages[1]["text"].startswith("Sam:")


def test_seed_is_reproducible_and_file_paths_load(tmp_path: Path):
    path = tmp_path / "interview.txt"
    path.write_text("One.\n\nTwo.\n\nThree.\n\nFour.\n\nFive.\n\nSix.\n", encoding="utf-8")
    first = plan_coding([path], unit="paragraph", seed=1)
    second = plan_coding([path], unit="paragraph", seed=1)
    other = plan_coding([path], unit="paragraph", seed=2)
    order = [item["passage_id"] for item in sorted(first.passages, key=lambda item: item["coding_order"])]
    assert order == [item["passage_id"] for item in sorted(second.passages, key=lambda item: item["coding_order"])]
    assert order != [item["passage_id"] for item in sorted(other.passages, key=lambda item: item["coding_order"])]
    assert first.passages[0]["doc_id"] == "interview"
