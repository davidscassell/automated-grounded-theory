"""Decision gates for the 0.6 representation and the snapshot commit."""

import inspect

from automated_grounded_theory import get_graph, open_code
from automated_grounded_theory.api import aopen_code
from automated_grounded_theory.async_support import cache_key
from automated_grounded_theory.benchmark import (
    audit_sample,
    codes_tagged_in,
    excerpt_offset_hits,
    normalize_doc_id,
    quote_and_coverage_rates,
    unit_grid_agreement,
)
from automated_grounded_theory.codebook import apply_sweep, commit_passage, interpret_coding
from automated_grounded_theory.graph import RunConfig, build_graph
from automated_grounded_theory.records import (
    apply_comparisons,
    proposal_buffer_reducer,
    proposal_id,
    select_committable,
)
from automated_grounded_theory.segment import plan_coding
from automated_grounded_theory.validation import anchor_quote, is_code_id_text
from automated_grounded_theory.version import __version__

from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder


def test_version_and_batch_size_default():
    assert __version__ == "0.8.0"
    assert inspect.signature(open_code).parameters["batch_size"].default == 1
    assert inspect.signature(aopen_code).parameters["batch_size"].default == 1
    assert RunConfig.__dataclass_fields__["batch_size"].default == 1


def test_two_processes_and_overlapping_spans():
    passage = {
        "passage_id": "p1",
        "doc_id": "d1",
        "text": "I praised the schedule and then criticized the manager.",
        "start_offset": 0,
        "end_offset": 52,
        "prev_text": "",
        "next_text": "",
    }
    coding = {
        "passage_id": "p1",
        "codes": [
            {
                "action": "new",
                "code": "praising the schedule",
                "definition": "Praising the posted schedule.",
                "quote": "praised the schedule and then",
                "rationale": "The first process is praise.",
                "valence": "positive",
                "confidence": 0.9,
            },
            {
                "action": "new",
                "code": "criticizing the manager",
                "definition": "Criticizing the manager.",
                "quote": "schedule and then criticized",
                "rationale": "The second process is criticism.",
                "valence": "negative",
                "confidence": 0.9,
            },
        ],
    }
    interpreted = interpret_coding(
        coding,
        passage,
        {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
    )
    assert len(interpreted["accepted"]) == 2
    committed = commit_passage({}, [], [], {}, {"segment": 0, "memo": 0, "highlight": 0, "op": 0}, passage, interpreted)
    assert len(committed["segments"]) == 2
    spans = [(row["doc_start"], row["doc_end"]) for row in committed["segments"]]
    assert spans[0][0] < spans[1][1] and spans[1][0] < spans[0][1]


def test_opposite_stances_share_a_category_without_merging():
    book = {
        "raising concerns": {
            "code": "raising concerns",
            "key": "raising concerns",
            "code_id": "a" * 16,
            "definition": "Speaking up.",
            "valence": "positive",
            "revision": 1,
            "history": [],
        },
        "withholding concerns": {
            "code": "withholding concerns",
            "key": "withholding concerns",
            "code_id": "b" * 16,
            "definition": "Staying silent.",
            "valence": "negative",
            "revision": 1,
            "history": [],
        },
    }
    result = apply_comparisons(
        book,
        [],
        [
            {
                "op": "link_to_category",
                "source": "raising concerns",
                "target": "withholding concerns",
                "category": "judging whether to speak",
            },
            {
                "op": "record_dimension",
                "code": "raising concerns",
                "dimension": "stance",
                "valence": "positive",
            },
            {"op": "keep_distinct", "source": "raising concerns", "target": "withholding concerns"},
            {"op": "relate_categories", "source": "speaking", "target": "silence", "relation": "conditions"},
            {"op": "defer", "source": "raising concerns", "target": "withholding concerns"},
        ],
    )
    assert set(result["codebook"]) == {"raising concerns", "withholding concerns"}
    assert "judging whether to speak" in result["codebook"]["raising concerns"]["category_links"]
    assert result["recorded"][0]["op"] == "relate_categories"
    assert result["deferred"][0]["status"] == "deferred"


def test_repeated_quote_is_not_silently_the_first_hit():
    text = "I packed the bags. Later I packed the bags again."
    quote = "packed the bags"
    flagged = anchor_quote(quote, text)
    assert "ambiguous_span" in flagged.flags
    second = anchor_quote(quote, text, occurrence=1)
    assert second.spans[0][0] > flagged.spans[0][0]
    assert "ambiguous_span" not in second.flags


def test_rare_case_survives_alias_and_hex_labels_are_rejected():
    book = {
        "packing bags": {
            "code": "packing bags",
            "key": "packing bags",
            "definition": "Bagging possessions.",
            "valence": "neutral",
            "history": [],
        },
        "naming a small perk": {
            "code": "naming a small perk",
            "key": "naming a small perk",
            "definition": "A rare perk.",
            "valence": "positive",
            "history": [],
        },
    }
    segments = [
        {"code": "packing bags", "code_key": "packing bags", "quote": "bags", "passage_id": "p", "doc_id": "d"},
        {"code": "naming a small perk", "code_key": "naming a small perk", "quote": "perk", "passage_id": "q", "doc_id": "d"},
    ]
    applied = apply_sweep(
        book,
        segments,
        merges=[{"source": "packing bags", "target": "naming a small perk", "confidence": 0.2, "rationale": "no"}],
        candidates=[],
        min_merge_confidence=0.5,
    )
    assert "naming a small perk" in applied["codebook"]
    assert is_code_id_text("36100ad339514b9e")
    rejected = interpret_coding(
        {"passage_id": "p", "codes": [{"action": "new", "code": "36100ad339514b9e", "quote": "bags", "definition": "x"}]},
        {"passage_id": "p", "text": "bags", "start_offset": 0},
        {},
        max_codes=4,
        require_gerund=False,
        long_passage_chars=2000,
    )
    assert rejected["accepted"] == []
    assert rejected["rejections"][0]["reason"] == "code_id_as_label"


def test_nontransitive_alias_is_not_applied():
    def entry(label):
        return {"code": label, "key": label, "definition": label, "valence": "neutral", "history": []}

    labels = ["recording the alpha delay", "recording an alpha delay", "recording that alpha delay"]
    book = {label: entry(label) for label in labels}
    segments = [
        {"code": label, "code_key": label, "quote": label, "passage_id": label, "doc_id": "d"} for label in labels
    ]
    applied = apply_sweep(
        book,
        segments,
        merges=[
            {
                "source": "recording an alpha delay",
                "target": "recording the alpha delay",
                "confidence": 0.9,
                "rationale": "pair",
                "same_action": True,
                "same_object": True,
                "contrast_checked": True,
                "material_contrast": False,
            },
            {
                "source": "recording that alpha delay",
                "target": "recording an alpha delay",
                "confidence": 0.9,
                "rationale": "chain",
                "same_action": True,
                "same_object": True,
                "contrast_checked": True,
                "material_contrast": False,
            },
        ],
        candidates=[],
        max_merges_per_sweep=10,
    )
    assert any(item.get("reason") == "non_transitive" for item in applied["rejected_merges"])
    assert applied["lost_distinction_candidates"]
    assert len(applied["codebook"]) == 2


def test_reducer_dedupes_and_stale_proposals_do_not_commit():
    proposal = {
        "proposal_id": proposal_id("s00001", "p1"),
        "snapshot_id": "s00001",
        "passage_index": 0,
        "coding": {"codes": [{"code": "packing bags"}]},
    }
    other = {
        "proposal_id": proposal["proposal_id"],
        "snapshot_id": "s00001",
        "passage_index": 0,
        "coding": {"codes": [{"code": "something else"}]},
    }
    once = proposal_buffer_reducer(None, {"items": [proposal]})
    twice = proposal_buffer_reducer(once, {"items": [proposal]})
    assert len(twice["items"]) == 1
    assert twice["duplicates"] == 1
    conflicted = proposal_buffer_reducer({"items": [], "duplicates": 0, "conflicts": []}, {"items": [proposal]})
    conflicted = proposal_buffer_reducer(conflicted, {"items": [other]})
    assert conflicted["items"] == []
    assert conflicted["conflicts"]
    fresh, stale, skipped = select_committable(
        [proposal, {**proposal, "proposal_id": "s00002:p1", "snapshot_id": "s00002"}],
        snapshot_id="s00001",
        committed_ids=[proposal["proposal_id"]],
    )
    assert fresh == []
    assert skipped == [proposal["proposal_id"]]
    assert stale[0]["snapshot_id"] == "s00002"
    cleared = proposal_buffer_reducer(twice, {"__reset__": True})
    assert cleared["items"] == []


def test_batch_size_one_codes_every_document(tmp_path):
    result = open_code(
        CORPUS,
        CONTEXT,
        model=FakeCoder(),
        seed=7,
        sweep_every=25,
        batch_size=1,
        concurrency=1,
        output_dir=tmp_path,
    )
    assert len(result.segments) == 4
    assert result.metadata["batch_size"] == 1
    assert (tmp_path / "taguette_codebook.csv").is_file()
    graph = get_graph("open_code")
    drawing = graph.compile().get_graph()
    assert "code_worker" in drawing.nodes
    assert "theme_worker" in get_graph("themes").compile().get_graph().nodes
    assert build_graph is not None


def test_incidents_keep_the_question_and_do_not_cut_mid_sentence():
    answer = " ".join(["Searching the database took longer than expected."] * 30)
    text = f"Interviewer: What did you do next?\nAlex: {answer}\n"
    plan = plan_coding([{"id": "6.0", "text": text}], unit="incident", seed=1)
    assert plan.passages
    assert all(item["question"].startswith("What did you do") for item in plan.passages)
    for item in plan.passages:
        assert item["text"] in text
        assert not item["text"].startswith("Interviewer")
        assert len(item["text"]) <= 1200 or item["text"].count(".") <= 1
    coded = plan_coding([{"id": "n", "text": "n/a"}], unit="incident", seed=1)
    assert coded.passages[0]["coverage_status"] == "pending"
    assert coded.passages[0]["exclusion_reason"] == ""
    assert coded.passages[0]["nonsubstantive"] is True
    assert coded.passages[0]["nonsubstantive_kind"] == "non_answer"
    assert coded.queue == [0]
    excluded = plan_coding([{"id": "n", "text": "n/a"}], unit="incident", seed=1, code_nonsubstantive=False)
    assert excluded.passages[0]["coverage_status"] == "excluded"
    assert excluded.passages[0]["exclusion_reason"] == "non_answer"
    assert excluded.queue == []


def test_cache_key_includes_versions_and_benchmark_ids():
    assert cache_key("prompt", snapshot_id="s1") != cache_key("prompt", snapshot_id="s2")
    assert cache_key("prompt", definition_revision="a") != cache_key("prompt", definition_revision="b")
    assert cache_key("prompt") == cache_key("prompt")
    assert normalize_doc_id("6.0") == normalize_doc_id("6") == "6"
    tags = [
        {"document": "6", "label": "in scope", "excerpt": "alpha beta"},
        {"document": "9.0", "label": "out of scope", "excerpt": "other"},
    ]
    assert codes_tagged_in(tags, ["6.0"]) == ["in scope"]
    hits = excerpt_offset_hits(
        [{"document": "6", "label": "in scope", "excerpt": "alpha beta"}],
        [{"doc_id": "6.0", "code": "llm", "doc_start": 0, "doc_end": 10, "quote": "alpha beta"}],
        [{"human_code": "in scope", "llm_codes": ["llm"], "grade": "full"}],
        documents=[{"id": "6", "text": "alpha beta gamma"}],
    )
    assert hits["excerpt_hits"] == 1
    grid = unit_grid_agreement(
        [{"human_codes": ["in scope"], "llm_codes": ["in scope"]}, {"human_codes": [], "llm_codes": []}]
    )
    assert grid["macro_f1"] == 1.0
    sample = audit_sample(
        [{"human_code": "a", "llm_code": "b"}, {"human_code": "c", "llm_code": "d"}],
        fraction=0.2,
        seed=1,
    )
    assert len(sample) == 1
    rates = quote_and_coverage_rates(
        [{"passage_id": "p", "skipped": False}, {"passage_id": "q", "skipped": True}],
        [{"passage_id": "p", "quote_status": "unanchored"}],
    )
    assert rates["uncoded_share"] == 0.0
    assert rates["quote_unanchored_rate"] == 1.0
