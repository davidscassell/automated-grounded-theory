"""0.6.12 regression tests for the review findings."""

import asyncio
import json
import warnings

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory.api import CheckpointMigrationRequired, CheckpointMismatch, open_code
from automated_grounded_theory.backends import chat_model_settings
from automated_grounded_theory.benchmark import ALIGNMENT_COVERAGE_MIN, unit_grid_from_annotations
from automated_grounded_theory.codebook import _apply_counts, commit_passage, empty_counters, interpret_coding
from automated_grounded_theory.context import parse_context
from automated_grounded_theory.evaluate import _passage_labels, _score, assignment_passage_ids
from automated_grounded_theory.prompts import build_sweep_messages, format_sweep_codebook, message_text
from automated_grounded_theory.retrieval import (
    cached_embed_texts,
    clear_embedding_cache,
    configure_embedding_cache,
    embedding_cache_stats,
    embedding_run_scope,
    select_prompt_examples,
    shared_codebook_vectors,
)
from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.version import __version__
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder


def test_version_is_0_6_12():
    assert __version__ == "0.9.4"


class ConfigCoder(FakeCoder):
    """A coder whose temperature lives on the instance, not the open_code argument."""

    temperature: float = 0
    top_p: float = 1
    max_tokens: int = 16
    model: str = "config-coder"
    api_key: str = ""


def _pause_kwargs(saver, thread_id: str) -> dict:
    return dict(seed=3, sweep_every=25, checkpointer=saver, thread_id=thread_id, interrupt_before_sweep=True)


def test_added_document_is_rejected_even_when_changes_are_allowed():
    saver = InMemorySaver()
    kwargs = _pause_kwargs(saver, "corpus-lock")
    paused = open_code(CORPUS, CONTEXT, model=FakeCoder(), **kwargs)
    assert {passage["doc_id"] for passage in paused.passages} == {"alex", "blair", "casey", "drew"}
    extra = CORPUS + [
        {"id": "erin", "name": "Erin", "text": "I came back in the spring and the room was still empty."}
    ]
    with pytest.raises(CheckpointMigrationRequired) as raised:
        open_code(extra, CONTEXT, model=FakeCoder(), resume=True, allow_changes=True, **kwargs)
    message = str(raised.value).lower()
    assert "start a new run" in message
    assert "no migration procedure yet" in message
    assert "corpus_hash" in raised.value.differences
    finished = open_code(CORPUS, CONTEXT, model=FakeCoder(), resume=True, **kwargs)
    assert {passage["doc_id"] for passage in finished.passages} == {"alex", "blair", "casey", "drew"}
    assert "erin" not in {passage["doc_id"] for passage in finished.passages}


def test_segmentation_change_is_rejected_and_performance_change_continues():
    saver = InMemorySaver()
    kwargs = _pause_kwargs(saver, "segment-lock")
    open_code(CORPUS, CONTEXT, model=FakeCoder(), **kwargs)
    with pytest.raises(CheckpointMigrationRequired) as raised:
        open_code(CORPUS, CONTEXT, model=FakeCoder(), unit="sentence", resume=True, allow_changes=True, **kwargs)
    assert "procedure.unit" in raised.value.differences
    assert "start a new run" in str(raised.value).lower()

    seeded = InMemorySaver()
    seed_kwargs = _pause_kwargs(seeded, "seed-lock")
    open_code(CORPUS, CONTEXT, model=FakeCoder(), **seed_kwargs)
    with pytest.raises(CheckpointMigrationRequired) as raised:
        open_code(
            CORPUS,
            CONTEXT,
            model=FakeCoder(),
            resume=True,
            allow_changes=True,
            **{**seed_kwargs, "seed": 9},
        )
    assert "procedure.seed" in raised.value.differences
    assert "start a new run" in str(raised.value).lower()

    other = InMemorySaver()
    other_kwargs = _pause_kwargs(other, "speed")
    open_code(CORPUS, CONTEXT, model=FakeCoder(), concurrency=4, **other_kwargs)
    with pytest.warns(UserWarning, match="changed checkpoint manifest"):
        continued = open_code(
            CORPUS,
            CONTEXT,
            model=FakeCoder(),
            concurrency=1,
            resume=True,
            allow_changes=True,
            **other_kwargs,
        )
    diff = continued.metadata["continuations"][-1]["differences"]
    assert "procedure.concurrency" in diff
    assert diff["procedure.concurrency"]["new"] == 1
    assert {passage["doc_id"] for passage in continued.passages} == {"alex", "blair", "casey", "drew"}


_ORDERED = [
    {"id": "a", "name": "A", "text": "Alpha one.\n\nAlpha two.\n\nAlpha three."},
    {"id": "b", "name": "B", "text": "Beta one.\n\nBeta two.\n\nBeta three."},
    {"id": "c", "name": "C", "text": "Gamma one.\n\nGamma two."},
]


def _stored_spans(passages) -> list[tuple]:
    return [
        (
            passage["passage_id"],
            passage["text"],
            passage["start_offset"],
            passage["end_offset"],
            passage["coding_order"],
        )
        for passage in passages
    ]


def test_review_mode_change_continues_without_resegmenting():
    saver = InMemorySaver()
    kwargs = dict(
        seed=7,
        unit="paragraph",
        sweep_every=2,
        review_mode="document",
        checkpointer=saver,
        thread_id="review-mode",
        interrupt_before_sweep=True,
    )
    paused = open_code(_ORDERED, CONTEXT, model=FakeCoder(), **kwargs)
    assert len(paused.passages) > 2
    with pytest.raises(CheckpointMismatch) as raised:
        open_code(
            _ORDERED,
            CONTEXT,
            model=FakeCoder(),
            resume=True,
            **{**kwargs, "review_mode": "wave"},
        )
    assert type(raised.value) is CheckpointMismatch
    assert "procedure.review_mode" in raised.value.differences
    with pytest.warns(UserWarning, match="changed checkpoint manifest"):
        continued = open_code(
            _ORDERED,
            CONTEXT,
            model=FakeCoder(),
            resume=True,
            allow_changes=True,
            **{**kwargs, "review_mode": "wave", "interrupt_before_sweep": False},
        )
    diff = continued.metadata["continuations"][-1]["differences"]
    assert diff["procedure.review_mode"] == {"old": "document", "new": "wave"}
    assert continued.metadata["review_mode"] == "wave"
    assert _stored_spans(continued.passages) == _stored_spans(paused.passages)


def test_parallel_documents_change_continues_without_resegmenting():
    saver = InMemorySaver()
    kwargs = dict(
        seed=7,
        unit="paragraph",
        sweep_every=2,
        parallel_documents=False,
        checkpointer=saver,
        thread_id="parallel-docs",
        interrupt_before_sweep=True,
    )
    paused = open_code(_ORDERED, CONTEXT, model=FakeCoder(), **kwargs)
    with pytest.raises(CheckpointMismatch) as raised:
        open_code(
            _ORDERED,
            CONTEXT,
            model=FakeCoder(),
            resume=True,
            **{**kwargs, "parallel_documents": True},
        )
    assert type(raised.value) is CheckpointMismatch
    assert "procedure.parallel_documents" in raised.value.differences
    with pytest.warns(UserWarning, match="changed checkpoint manifest"):
        continued = open_code(
            _ORDERED,
            CONTEXT,
            model=FakeCoder(),
            resume=True,
            allow_changes=True,
            **{**kwargs, "parallel_documents": True, "interrupt_before_sweep": False},
        )
    diff = continued.metadata["continuations"][-1]["differences"]
    assert diff["procedure.parallel_documents"] == {"old": False, "new": True}
    assert continued.metadata["parallel_documents"] is True
    assert _stored_spans(continued.passages) == _stored_spans(paused.passages)

    cold = ConfigCoder(temperature=0, top_p=1, max_tokens=16, api_key="sk-secret")
    hot = ConfigCoder(temperature=1, top_p=0.2, max_tokens=32, api_key="sk-secret")
    settings = chat_model_settings(cold)
    assert settings["temperature"] == 0.0
    assert "sk-secret" not in json.dumps(settings)
    assert "api_key" not in settings

    saver = InMemorySaver()
    kwargs = _pause_kwargs(saver, "temperature")
    paused = open_code(CORPUS, CONTEXT, model=cold, temperature=0, **kwargs)
    assert paused.metadata["temperature"] == 0.0
    assert paused.metadata["model_settings"]["temperature"] == 0.0
    assert paused.metadata["model_settings"]["max_tokens"] == 16
    assert "sk-secret" not in json.dumps(paused.metadata)

    with pytest.raises(CheckpointMismatch) as raised:
        open_code(CORPUS, CONTEXT, model=hot, temperature=0, resume=True, **kwargs)
    assert type(raised.value) is CheckpointMismatch
    assert any("temperature" in path for path in raised.value.differences)

    with pytest.warns(UserWarning, match="changed checkpoint manifest"):
        continued = open_code(
            CORPUS,
            CONTEXT,
            model=hot,
            temperature=0,
            resume=True,
            allow_changes=True,
            **kwargs,
        )
    assert continued.metadata["temperature"] == 1.0
    assert continued.metadata["model_settings"]["temperature"] == 1.0
    assert continued.metadata["model_settings"]["top_p"] == 0.2
    assert continued.metadata["model_settings"]["max_tokens"] == 32
    assert continued.metadata["checkpoint_manifest"]["model"]["model_settings"]["temperature"] == 1.0
    assert "sk-secret" not in json.dumps(continued.metadata)


class _DimEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 4, model: str = "embed-a") -> None:
        self.dimensions = dimensions
        self.model = model

    def embed_documents(self, texts):
        width = int(self.dimensions)
        return [[1.0] * width for _ in texts]

    def embed_query(self, text):
        return [1.0] * int(self.dimensions)


def test_embedding_dimensions_are_in_the_manifest():
    saver = InMemorySaver()
    kwargs = _pause_kwargs(saver, "embeddings")
    paused = open_code(CORPUS, CONTEXT, model=FakeCoder(), embeddings=_DimEmbeddings(dimensions=4), **kwargs)
    assert paused.metadata["embeddings_settings"]["dimensions"] == 4
    assert paused.metadata["embeddings_settings"]["model"] == "embed-a"
    with pytest.raises(CheckpointMismatch) as raised:
        open_code(
            CORPUS,
            CONTEXT,
            model=FakeCoder(),
            embeddings=_DimEmbeddings(dimensions=8),
            resume=True,
            **kwargs,
        )
    assert type(raised.value) is CheckpointMismatch
    assert any("embeddings_settings" in path for path in raised.value.differences)


def test_ninth_contradictory_quote_is_in_the_next_review_prompt():
    applications = [
        {
            "segment_id": str(index),
            "passage_id": f"p{index}",
            "doc_id": "d1",
            "quote": f"I left home on day {index}.",
            "code_key": "leaving",
            "doc_start": index,
            "doc_end": index + 1,
        }
        for index in range(8)
    ]
    contradictory = {
        "segment_id": "8",
        "passage_id": "p8",
        "doc_id": "d1",
        "quote": "I did not leave. I stayed.",
        "code_key": "leaving",
        "doc_start": 8,
        "doc_end": 9,
    }
    entry = _apply_counts({"code": "leaving home", "key": "leaving", "definition": "Going away."}, applications, "leaving")
    entry["reviewed_assignment_ids"] = [str(index) for index in range(8)]
    entry["reviews_seen"] = [{"fingerprint": "1:seen", "assignment_ids": list(entry["reviewed_assignment_ids"])}]
    nine = _apply_counts(entry, applications + [contradictory], "leaving")
    assert len(nine["assignments"]) == 9
    assert nine["assignments"][-1]["quote"] == contradictory["quote"]
    chosen = select_prompt_examples(nine, limit=3)
    assert any(item["quote"] == contradictory["quote"] for item in chosen)
    assert contradictory["quote"] in format_sweep_codebook([nine])
    messages = build_sweep_messages(
        context=parse_context(CONTEXT),
        codes=[nine],
        candidates=[],
        prompt_cache=False,
    )
    prompt = "\n".join(message_text(message) for message in messages)
    assert contradictory["quote"] in prompt


def test_document_diversity_runs_before_passage_order():
    entry = {
        "code": "leaving",
        "definition": "Going away.",
        "examples": [
            {"quote": "I left home", "passage_id": "p1", "doc_id": "a"},
            {"quote": "I left again", "passage_id": "p2", "doc_id": "a"},
            {"quote": "I left a third time", "passage_id": "p3", "doc_id": "a"},
            {"quote": "She left too", "passage_id": "p4", "doc_id": "b"},
        ],
    }
    chosen = select_prompt_examples(entry, limit=2)
    assert {item["doc_id"] for item in chosen} == {"a", "b"}


def _span_passages():
    first = "I resigned."
    middle = "I spent the holiday walking the coast and reading in the evenings. " * 8
    middle = middle.strip()
    last = "I returned."
    document = f"{first} {middle} {last}"
    start_middle = len(first) + 1
    end_middle = start_middle + len(middle)
    start_last = end_middle + 1

    def passage(passage_id, text, start, end, prev, nxt):
        return {
            "passage_id": passage_id,
            "doc_id": "d1",
            "doc_name": "Diary",
            "text": text,
            "start_offset": start,
            "end_offset": end,
            "prev_text": prev,
            "next_text": nxt,
            "document_text": document,
        }

    passages = [
        passage("p0", first, 0, len(first), "", middle),
        passage("p1", middle, start_middle, end_middle, first, last),
        passage("p2", last, start_last, len(document), middle, ""),
    ]
    return passages


def test_discontinuous_quote_is_not_attributed_to_the_gap():
    passages = _span_passages()
    middle = passages[1]
    interpreted = interpret_coding(
        {
            "codes": [
                {
                    "action": "new",
                    "code": "leaving and returning",
                    "definition": "Going away and coming back.",
                    "quote": "I resigned. ... I returned.",
                    "confidence": 0.9,
                    "rationale": "The two ends are one incident.",
                }
            ],
            "memo": "",
        },
        middle,
        {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
        passages=passages,
        document_text=middle["document_text"],
    )
    kept = interpreted["accepted"][0]
    assert kept["needs_review"] is True
    assert "evidence_outside_passage" in kept["review_reasons"]
    assert kept["attribution"] == "multi"
    assert kept["attributed_passage_ids"] == ["p0", "p2"]
    assert kept["attributed_passage_id"] == ""
    assert kept["coded_passage_id"] == "p1"
    committed = commit_passage({}, [], [], {}, empty_counters(), middle, interpreted)
    segment = committed["segments"][0]
    assert segment["passage_id"] == ""
    assert segment["attributed_passage_id"] == ""
    assert segment["attributed_passage_ids"] == ["p0", "p2"]
    assert assignment_passage_ids(segment) == ["p0", "p2"]
    run = {
        "segments": committed["segments"],
        "passages": passages,
        "codebook": [],
        "memos": [],
        "sweep_log": [],
        "prompt_log": [],
        "saturation": [],
    }
    metrics = _score(run, "")
    assert metrics["coverage_pct"] == round(200 / 3, 1)
    assert set(_passage_labels(run)) == {"p0", "p2"}
    assert "p1" not in _passage_labels(run)


def _mapping(human: str, model: str) -> dict:
    return {"human_code": human, "llm_code": model, "grade": "full", "confidence": 0.9}


def test_extra_space_excerpt_stays_a_positive_match():
    resigned = "I resigned yesterday."
    holiday = "I enjoyed the holiday."
    document = f"{resigned} {holiday}"
    passages = [
        {
            "passage_id": "p1",
            "doc_id": "d1",
            "text": resigned,
            "start_offset": 0,
            "end_offset": len(resigned),
        },
        {
            "passage_id": "p2",
            "doc_id": "d1",
            "text": holiday,
            "start_offset": len(resigned) + 1,
            "end_offset": len(document),
        },
    ]
    documents = [{"id": "d1", "text": document}]
    tags = [
        {"document": "d1", "excerpt": "I  resigned yesterday.", "label": "resigning"},
        {"document": "d1", "excerpt": holiday, "label": "holiday"},
    ]
    segments = [
        {"passage_id": "p1", "code": "leaving a job", "doc_id": "d1"},
        {"passage_id": "p2", "code": "enjoying a holiday", "doc_id": "d1"},
    ]
    mappings = [_mapping("resigning", "leaving a job"), _mapping("holiday", "enjoying a holiday")]
    grid = unit_grid_from_annotations(passages, segments, tags, mappings, documents)
    assert grid["available"] is True
    assert grid["macro_f1"] == 1.0
    assert grid["alignment_aligned"] == 2
    assert grid["alignment_total"] == 2
    assert grid["alignment_coverage"] == 1.0


def test_repeated_excerpt_uses_occurrence_or_stays_ambiguous():
    text = "I resigned. Later I resigned."
    second = text.rfind("I resigned.")
    passages = [
        {
            "passage_id": "p1",
            "doc_id": "d1",
            "text": "I resigned.",
            "start_offset": 0,
            "end_offset": len("I resigned."),
        },
        {
            "passage_id": "p2",
            "doc_id": "d1",
            "text": "I resigned.",
            "start_offset": second,
            "end_offset": second + len("I resigned."),
        },
    ]
    documents = [{"id": "d1", "text": text}]
    mappings = [_mapping("resigning", "leaving a job")]
    segments = [{"passage_id": "p2", "code": "leaving a job", "doc_id": "d1"}]
    ambiguous = unit_grid_from_annotations(
        passages,
        segments,
        [{"document": "d1", "excerpt": "I resigned.", "label": "resigning"}],
        mappings,
        documents,
    )
    assert ambiguous["available"] is False
    assert ambiguous["reason"] == "span_alignment_unavailable"
    assert ambiguous["alignment_ambiguous"] == 1
    assert ambiguous["macro_f1"] is None

    placed = unit_grid_from_annotations(
        passages,
        segments,
        [{"document": "d1", "excerpt": "I resigned.", "label": "resigning", "occurrence": 1}],
        mappings,
        documents,
    )
    assert placed["available"] is True
    assert placed["macro_f1"] == 1.0
    assert placed["alignment_aligned"] == 1

    by_offset = unit_grid_from_annotations(
        passages,
        segments,
        [{"document": "d1", "excerpt": "I resigned.", "label": "resigning", "offset": second}],
        mappings,
        documents,
    )
    assert by_offset["available"] is True
    assert by_offset["macro_f1"] == 1.0


def test_low_alignment_coverage_is_unavailable_and_unknown_units_are_dropped():
    resigned = "I resigned yesterday."
    holiday = "I enjoyed the holiday."
    document = f"{resigned} {holiday}"
    passages = [
        {"passage_id": "p1", "doc_id": "d1", "text": resigned, "start_offset": 0, "end_offset": len(resigned)},
        {
            "passage_id": "p2",
            "doc_id": "d1",
            "text": holiday,
            "start_offset": len(resigned) + 1,
            "end_offset": len(document),
        },
    ]
    documents = [{"id": "d1", "text": document}]
    mappings = [_mapping("resigning", "leaving a job"), _mapping("holiday", "enjoying a holiday")]
    segments = [
        {"passage_id": "p1", "code": "leaving a job", "doc_id": "d1"},
        {"passage_id": "p2", "code": "enjoying a holiday", "doc_id": "d1"},
    ]
    thin = unit_grid_from_annotations(
        passages,
        segments,
        [
            {"document": "d1", "excerpt": "I  resigned yesterday.", "label": "resigning"},
            {"document": "d1", "excerpt": "not in the document", "label": "holiday", "offset": len(resigned) + 1},
        ],
        mappings,
        documents,
    )
    assert thin["alignment_aligned"] == 1
    assert thin["alignment_total"] == 2
    assert thin["alignment_coverage"] == 0.5
    assert thin["alignment_coverage"] < ALIGNMENT_COVERAGE_MIN
    assert thin["available"] is False
    assert thin["reason"] == "alignment_coverage_below_threshold"
    assert thin["macro_f1"] is None

    parts = [f"Sentence {index} stands alone." for index in range(5)]
    body = " ".join(parts)
    cursor = 0
    wide_passages = []
    for index, part in enumerate(parts):
        wide_passages.append(
            {
                "passage_id": f"u{index}",
                "doc_id": "d1",
                "text": part,
                "start_offset": cursor,
                "end_offset": cursor + len(part),
            }
        )
        cursor += len(part) + 1
    tags = [
        {"document": "d1", "excerpt": part, "label": f"label {index}"}
        for index, part in enumerate(parts[:4])
    ]
    tags.append(
        {
            "document": "d1",
            "excerpt": "missing words",
            "label": "label 4",
            "offset": wide_passages[4]["start_offset"],
        }
    )
    wide_segments = [
        {"passage_id": f"u{index}", "code": f"code {index}", "doc_id": "d1"} for index in range(5)
    ]
    wide_mappings = [_mapping(f"label {index}", f"code {index}") for index in range(5)]
    wide = unit_grid_from_annotations(
        wide_passages,
        wide_segments,
        tags,
        wide_mappings,
        [{"id": "d1", "text": body}],
    )
    assert wide["alignment_coverage"] == 0.8
    assert wide["available"] is True
    assert wide["unknown_units"] == 1
    assert wide["units"] == 4
    assert wide["macro_f1"] == 1.0


class _CapEmbeddings:
    model = "cap-test"
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_documents(self, texts):
        self.calls.extend(list(texts))
        return [[1.0, 0.0] for _ in texts]


def test_canonical_vector_cap_is_lru_and_optional():
    clear_embedding_cache()
    configure_embedding_cache(vector_limit=2)
    embeddings = _CapEmbeddings()
    try:
        cached_embed_texts(embeddings, ["one", "two", "three"])
        stats = embedding_cache_stats()
        assert stats["vectors"] == 2
        assert stats["vector_limit"] == 2
        cached_embed_texts(embeddings, ["one"])
        assert embeddings.calls.count("one") == 2
        clear_embedding_cache()
        configure_embedding_cache(vector_limit=None, vector_bytes=40)
        wide = _CapEmbeddings()
        wide.dimensions = 4

        def embed(texts):
            wide.calls.extend(list(texts))
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        wide.embed_documents = embed
        cached_embed_texts(wide, ["alpha", "beta"])
        assert embedding_cache_stats()["vectors"] == 1
        assert embedding_cache_stats()["vector_bytes_used"] <= 40
    finally:
        configure_embedding_cache(vector_limit=None, vector_bytes=None)
        clear_embedding_cache()


def test_snapshot_vectors_are_not_evicted_by_the_cap():
    clear_embedding_cache()

    class Growth:
        model = "protected-vectors"
        dimensions = 2

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        async def aembed_documents(self, texts):
            return self.embed_documents(texts)

    async def run():
        configure_embedding_cache(vector_limit=1)
        limiter = CallLimiter(2, max_retries=0, backoff=0, max_backoff=0, request_timeout=0)
        with embedding_run_scope("protect"):
            entries = [
                {"code": "one", "definition": "first definition"},
                {"code": "two", "definition": "second definition"},
            ]
            vectors = await shared_codebook_vectors(Growth(), entries, limiter=limiter, batch_size=8)
            assert len(vectors) == 2
            assert embedding_cache_stats()["vectors"] >= 2

    try:
        asyncio.run(run())
    finally:
        configure_embedding_cache(vector_limit=None)
        clear_embedding_cache()
