"""Focused coding with a fake model. No provider and no API key."""

import csv
import json

import pytest

from automated_grounded_theory import build_themes, open_code
from automated_grounded_theory.context import PresetCodesError, parse_context
from automated_grounded_theory.prompts import build_theme_messages
from automated_grounded_theory.results import OpenCodeResult
from automated_grounded_theory.themes import order_codes, respondent_of, theme_tag_path
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder
from tests.fake_themer import FakeThemer


def _passage(passage_id, doc_id, text, metadata, skipped=False):
    return {
        "passage_id": passage_id,
        "doc_id": doc_id,
        "text": text,
        "metadata": metadata,
        "skipped": skipped,
    }


def _segment(segment_id, passage_id, doc_id, code, quote):
    return {
        "segment_id": segment_id,
        "passage_id": passage_id,
        "doc_id": doc_id,
        "code": code,
        "quote": quote,
    }


def _result():
    alex = "I spent the morning packing my suitcase. The room looked empty once the books were gone."
    blair = "Packing bags up with boxes took all night. I kept finding letters I could not throw away."
    short = "Hi there. I will leave it at that."
    bad = "I packed carefully for the trip and left at dawn."
    casey = "I was crying at the station when the train doors closed."
    drew = "They hired me on a Friday. Starting the new job meant moving before the month ended."
    passages = [
        _passage("alex", "alex", alex, {"respondent_id": "R1", "question": "q1", "department": "IT"}),
        _passage("blair", "blair", blair, {"respondent_id": "R1", "question": "q1", "department": "IT"}),
        _passage("short", "short", short, {"respondent_id": "R2", "question": "q1", "department": "HR"}),
        _passage("bad", "bad", bad, {"respondent_id": "R3", "question": "q2", "department": "HR"}),
        _passage("casey", "casey", casey, {"respondent_id": "R4", "question": "q2", "department": "IT"}),
        _passage("drew", "drew", drew, {"respondent_id": "R5", "question": "q3", "department": "Finance"}),
        _passage("erin", "erin", "n/a", {"respondent_id": "R6", "question": "q2", "department": "IT"}, skipped=True),
    ]
    segments = [
        _segment("s1", "alex", "alex", "packing bags", "I spent the morning packing my suitcase."),
        _segment("s2", "blair", "blair", "packing bags", "Packing bags up with boxes took all night."),
        _segment("s3", "short", "short", "packing bags", "Hi there."),
        _segment("s4", "bad", "bad", "packing bags", "This quote was never said."),
        _segment("s5", "casey", "casey", "crying at the station", casey),
        _segment("s6", "drew", "drew", "starting a new job", "Starting the new job meant moving before the month ended."),
    ]
    codebook = [
        {
            "code": "packing bags",
            "definition": "Bagging possessions before a departure.",
            "frequency": 4,
            "examples": [{"quote": "I spent the morning packing my suitcase.", "passage_id": "alex"}],
        },
        {
            "code": "crying at the station",
            "definition": "Crying as the train leaves.",
            "frequency": 1,
            "examples": [{"quote": casey, "passage_id": "casey"}],
        },
        {
            "code": "starting a new job",
            "definition": "A job that forces the move.",
            "frequency": 1,
            "examples": [],
        },
    ]
    return OpenCodeResult(
        codebook=codebook,
        segments=segments,
        memos=[],
        passages=passages,
        saturation=[],
        saturation_report={},
        validation_log=[],
        sweep_log=[],
        prompt_log=[],
        metadata={},
    )


def _themes_of(result):
    return {theme["name"]: theme for theme in result.themes}


def test_prevalence_quotes_breakdowns_and_exports(tmp_path):
    model = FakeThemer()
    result = build_themes(
        _result(),
        CONTEXT,
        model=model,
        seed=1,
        output_dir=tmp_path,
        chart=True,
        n_themes=2,
    )
    assert model.call_count == 1
    assert result.metadata["calls"] == 1
    assert result.prompt_log[0]["usage"]["total_tokens"] == 30
    assert result.totals == {"n_passages": 7, "n_documents": 7, "n_respondents": 6, "n_codes": 3}
    labels = [code for theme in result.themes for code in theme["codes"]]
    assert sorted(labels) == ["crying at the station", "packing bags", "starting a new job"]
    assert any("Requested 2 themes" in warning for warning in result.warnings)

    packing = _themes_of(result)["Packing to leave"]
    assert packing["n_passages"] == 4
    assert packing["n_documents"] == 4
    assert packing["n_respondents"] == 3
    assert packing["pct_passages"] == 57.1
    assert packing["pct_respondents"] == 50.0
    assert packing["n_comments"] == packing["n_documents"]
    assert packing["properties"] == ["preparation", "what is taken"]
    assert packing["breakdowns"]["question"] == [{"value": "q1", "n": 3}, {"value": "q2", "n": 1}]
    assert packing["breakdowns"]["department"] == [{"value": "HR", "n": 2}, {"value": "IT", "n": 2}]
    assert [quote["respondent"] for quote in packing["quotes"]] == ["R1"]
    assert packing["quotes"][0]["quote"] == "Packing bags up with boxes took all night."
    assert packing["quotes"][0]["doc_id"] == "blair"
    assert "This quote was never said." not in {quote["quote"] for quote in packing["quotes"]}
    assert "Hi there." not in {quote["quote"] for quote in packing["quotes"]}

    assert result.uncoded == [
        {
            "passage_id": "erin",
            "doc_id": "erin",
            "text": "n/a",
            "skipped": True,
            "respondent": "R6",
        }
    ]
    passage_text = [item["text"] for item in _result().passages]
    assert any(item["quote"] for item in result.tensions)
    assert all(item["quote"] != "NOT IN THE DATA" for item in result.tensions)
    assert all(
        any(item["quote"] in text for text in passage_text)
        for item in result.tensions
        if item["quote"]
    )
    assert any("not verbatim" in warning for warning in result.warnings)
    assert result.negative_cases[0]["quote"] == ""
    assert result.negative_cases[0]["code"] in {
        "packing bags",
        "crying at the station",
        "starting a new job",
    }

    markdown = (tmp_path / "themes.md").read_text(encoding="utf-8")
    assert "## Packing to leave" in markdown
    assert "**Prevalence:** 4 passages (57.1%), 4 documents (57.1%), 3 respondents (50.0%)" in markdown
    assert "- By question: q1 3, q2 1" in markdown
    assert '— R1 (blair; code: packing bags)' in markdown
    assert "## Uncoded passages (1 of 7)" in markdown
    assert "erin" in markdown

    with (tmp_path / "themes.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "theme",
        "definition",
        "codes",
        "n_passages",
        "pct_passages",
        "n_documents",
        "pct_documents",
        "n_comments",
        "pct_comments",
        "n_respondents",
        "pct_respondents",
    ]
    packed_row = next(row for row in rows if row["theme"] == "Packing to leave")
    assert packed_row["codes"] == "packing bags"
    assert packed_row["n_comments"] == "4"
    assert packed_row["pct_respondents"] == "50.0"

    tags = (tmp_path / "taguette_themes.csv").read_text(encoding="utf-8")
    assert "Packing to leave\\packing bags" in tags
    assert (tmp_path / "taguette_themes.qdc").exists()
    saved = json.loads((tmp_path / "themes.json").read_text(encoding="utf-8"))
    assert saved["totals"]["n_codes"] == 3
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert any("matplotlib" in warning for warning in result.warnings)
        assert not (tmp_path / "theme_prevalence.png").exists()
    else:
        assert (tmp_path / "theme_prevalence.png").stat().st_size > 0


def test_seed_does_not_change_quotes():
    first = build_themes(_result(), CONTEXT, model=FakeThemer(), seed=1, chart=False)
    second = build_themes(_result(), CONTEXT, model=FakeThemer(), seed=2, chart=False)
    assert [theme["quotes"] for theme in first.themes] == [theme["quotes"] for theme in second.themes]


def test_retry_assigns_the_code_the_first_call_dropped():
    model = FakeThemer(drop_last="until_feedback")
    result = build_themes(_result(), "How do students describe leaving?", model=model, seed=3)
    assert model.call_count == 2
    assert "VALIDATION FAILED" in result.prompt_log[1]["messages"][1]["content"]
    assert len(model.seen_labels[0]) == 2
    assert len(model.seen_labels[1]) == 3
    labels = [code for theme in result.themes for code in theme["codes"]]
    assert sorted(labels) == ["crying at the station", "packing bags", "starting a new job"]


def test_still_missing_after_retry_is_ungrouped():
    model = FakeThemer(drop_last="always")
    result = build_themes(_result(), CONTEXT, model=model, seed=1, chart=False)
    assert model.call_count == 3
    labels = [code for theme in result.themes for code in theme["codes"]]
    assert sorted(labels) == ["crying at the station", "packing bags", "starting a new job"]
    assert any(theme["name"] == "Ungrouped incidents" for theme in result.themes)
    assert any("Ungrouped incidents" in warning for warning in result.warnings)


def test_unknown_code_is_dropped_and_not_required():
    model = FakeThemer(invent_code=True)
    result = build_themes(_result(), CONTEXT, model=model, seed=1)
    labels = [code for theme in result.themes for code in theme["codes"]]
    assert "not a real code" not in labels
    assert any("not a real code" in warning for warning in result.warnings)


def test_preset_codes_never_call_the_model():
    model = FakeThemer()
    with pytest.raises(PresetCodesError):
        build_themes(_result(), {"research_question": "What happened?", "codes": ["leaving"]}, model=model)
    assert model.call_count == 0


def test_empty_codebook_skips_the_model_and_lists_uncoded():
    source = _result()
    source.codebook = []
    source.segments = []
    result = build_themes(source, CONTEXT, model="xai:grok-4")
    assert result.metadata["calls"] == 0
    assert result.metadata["model"] == "xai:grok-4"
    assert result.themes == []
    assert len(result.uncoded) == 7
    assert any("No open codes" in warning for warning in result.warnings)


def test_chart_skipped_when_matplotlib_is_missing(monkeypatch, tmp_path):
    import automated_grounded_theory.themes as themes

    def _boom():
        raise ImportError("matplotlib")

    monkeypatch.setattr(themes, "_load_pyplot", _boom)
    result = build_themes(_result(), CONTEXT, model=FakeThemer(), output_dir=tmp_path)
    assert (tmp_path / "themes.md").exists()
    assert not (tmp_path / "theme_prevalence.png").exists()
    assert any("matplotlib" in warning for warning in result.warnings)


def test_open_code_result_is_fully_assigned(tmp_path):
    coded = open_code(CORPUS, CONTEXT, model=FakeCoder(), seed=7, sweep_every=25, batch_size=4, concurrency=1)
    themed = build_themes(coded, CONTEXT, model=FakeThemer(), output_dir=tmp_path, seed=4)
    assigned = [code for theme in themed.themes for code in theme["codes"]]
    assert sorted(assigned) == sorted(entry["code"] for entry in coded.codebook)
    passages = {item["passage_id"]: item["text"] for item in coded.passages}
    for theme in themed.themes:
        for quote in theme["quotes"]:
            assert quote["quote"] in passages[quote["passage_id"]]
            assert len(quote["quote"]) >= 15
    assert themed.uncoded == []
    assert (tmp_path / "taguette_themes.csv").exists()


def test_order_tag_path_and_respondent_key():
    codes = [{"code": f"code {index}", "frequency": index} for index in range(8)]
    assert [item["code"] for item in order_codes(codes, seed=1)] == [
        item["code"] for item in order_codes(codes, seed=1)
    ]
    assert [item["code"] for item in order_codes(codes, seed=1)] != [
        item["code"] for item in order_codes(codes, seed=2)
    ]
    assert [item["code"] for item in order_codes(codes, seed=None)] == [f"code {index}" for index in range(7, -1, -1)]
    assert theme_tag_path("Pay. Benefits", "asking.more") == "Pay. Benefits\\asking.more"
    assert theme_tag_path("Pay\\Benefits", "asking\\more") == "Pay·Benefits\\asking·more"
    metadata = {"respondent_id": "R1", "person": "Alex"}
    assert respondent_of(metadata, "alex", None) == "R1"
    assert respondent_of(metadata, "alex", "person") == "Alex"
    assert respondent_of({}, "alex", None) == "alex"
    messages = build_theme_messages(
        context=parse_context(CONTEXT),
        codes=codes[:1],
        n_themes=None,
        feedback="",
        prompt_cache=True,
    )
    assert messages[0].content[0]["cache_control"] == {"type": "ephemeral"}
