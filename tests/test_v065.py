"""0.6.5: compact sweep payloads, embedding cache, and token budgets."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from automated_grounded_theory import open_code
from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.batching import pack_indices
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.prompts import (
    build_coding_messages,
    build_sweep_messages,
    message_text,
)
from automated_grounded_theory.retrieval import (
    EmbeddingCodeRetriever,
    clear_embedding_cache,
    code_needs_review,
    code_text,
    full_audit_due,
    hint_pairs,
    review_fingerprint,
    select_cross_chunk,
)
from automated_grounded_theory.schemas import CodeInput
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder


def _benchmark():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_sweep_payload.py"
    spec = importlib.util.spec_from_file_location("benchmark_sweep_payload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sweep_payload_matches_review_table_and_grows_near_linear():
    table = _benchmark().payload_table()
    before = [row["before"] for row in table]
    after = [row["after"] for row in table]
    capped = [row["after_capped"] for row in table]
    for row in table:
        assert abs(row["before"] - row["published"]) / row["published"] < 0.03
        assert row["after"] < row["before"]
        assert row["after_capped"] < row["before"]
    for prev, nxt in zip(after, after[1:]):
        assert nxt / prev < 3.0
    assert capped[1] / capped[0] < 3.5
    for prev, nxt in zip(capped[1:], capped[2:]):
        assert nxt / prev < 3.0
    assert after[-1] / after[0] < 22
    assert before[-1] / before[0] > 50
    assert capped[-1] < before[-1] * 0.35


def test_sweep_request_is_compact_and_does_not_repeat_the_first_quote():
    assert "example" not in CodeInput.model_fields
    packing = {
        "code": "packing bags",
        "definition": "putting clothes in a bag",
        "examples": [
            {"quote": "packing my suitcase", "passage_id": "p1"},
            {"quote": "the books were gone", "passage_id": "p2"},
        ],
    }
    luggage = {
        "code": "packing luggage",
        "definition": "putting clothes in luggage",
        "examples": [{"quote": "packing bags up with boxes", "passage_id": "p3"}],
    }
    context = ResearchContext(research_question="How do people leave?")
    full = build_sweep_messages(
        context=context,
        codes=[packing],
        candidates=[],
        prompt_cache=False,
        all_codes=[packing, luggage],
        cross_chunk=[luggage],
        chunk_index=1,
        chunk_count=2,
        audit="full",
    )
    incremental = build_sweep_messages(
        context=context,
        codes=[packing],
        candidates=[],
        prompt_cache=False,
        all_codes=[packing, luggage],
        chunk_index=1,
        chunk_count=2,
        consolidation=True,
        audit="incremental",
    )
    assert message_text(full[0]) == message_text(incremental[0])
    body = message_text(full[-1])
    assert "\n" not in body
    payload = json.loads(body)
    code = payload["codes"][0]
    assert "example" not in code
    assert code["examples"][0]["quote"] == "packing my suitcase"
    assert json.dumps(code, separators=(",", ":")).count("packing my suitcase") == 1
    assert set(payload["label_index"][0]) == {"code_id", "label"}
    assert "definition" not in payload["label_index"][0]
    assert payload["cross_chunk"][0]["definition"] == "putting clothes in luggage"
    assert payload["audit"] == "full"
    assert json.loads(message_text(incremental[-1]))["audit"] == "incremental"
    coding = build_coding_messages(
        context=context,
        codebook_view=[packing],
        omitted_codes=0,
        passages=[{"passage_id": "p1", "text": "I was packing my suitcase."}],
        neighbor_chars=0,
        feedback=None,
        prompt_cache=False,
    )
    assert "\n" not in message_text(coding[-1])


def test_review_fingerprint_and_audit_schedule():
    entry = {
        "code": "packing bags",
        "definition": "putting clothes in a bag",
        "revision": 1,
        "frequency": 3,
        "examples": [{"quote": "packing my suitcase"}],
    }
    fingerprint = review_fingerprint(entry)
    assert review_fingerprint({**entry, "frequency": 99}) == fingerprint
    assert review_fingerprint({**entry, "definition": "something else"}) != fingerprint
    assert review_fingerprint({**entry, "code": "packing luggage"}) != fingerprint
    assert review_fingerprint({**entry, "revision": 2}) != fingerprint
    assert review_fingerprint({**entry, "examples": [{"quote": "other quote"}]}) != fingerprint
    assert code_needs_review(entry)
    assert not code_needs_review({**entry, "reviewed_revision": fingerprint})
    assert full_audit_due(1, consolidation=False, full_audit_every=4)
    assert not full_audit_due(2, consolidation=False, full_audit_every=4)
    assert full_audit_due(5, consolidation=False, full_audit_every=4)
    assert full_audit_due(2, consolidation=True, full_audit_every=0)
    assert not full_audit_due(1, consolidation=False, full_audit_every=0)


def test_cross_chunk_and_incremental_hints_stay_with_the_due_codes():
    packing = {
        "code": "packing bags",
        "definition": "packing bags for a trip",
        "examples": [{"quote": "packing my suitcase"}],
    }
    luggage = {
        "code": "packing luggage",
        "definition": "packing bags for a trip",
        "examples": [{"quote": "packing my suitcase"}],
    }
    crying = {
        "code": "crying at the station",
        "definition": "crying when the train left",
        "examples": [{"quote": "crying at the station"}],
    }
    codes = [packing, luggage, crying]
    found = [entry["code"] for entry in select_cross_chunk(codes, [packing], limit=8)]
    assert found == ["packing luggage"]
    assert select_cross_chunk(codes, [packing], limit=0) == []
    offered = hint_pairs(codes, lexical_threshold=0.3, limit=None)
    assert any({item["source"], item["target"]} == {"packing bags", "packing luggage"} for item in offered)
    touched = hint_pairs(codes, lexical_threshold=0.3, limit=None, chunk_indexes=[2])
    assert all(
        "crying at the station" in {item["source"], item["target"]}
        for item in touched
    )
    assert not any({item["source"], item["target"]} == {"packing bags", "packing luggage"} for item in touched)


class _CountingEmbeddings:
    def __init__(self, model: str) -> None:
        self.model = model
        self.documents: list[str] = []
        self.queries: list[str] = []
        self.fail_once = False

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [float(len(text)), 0.0]

    async def aembed_documents(self, texts):
        await asyncio.sleep(0.02)
        if self.fail_once:
            self.fail_once = False
            self.documents.append("__failed__")
            raise RuntimeError("embed failed")
        return self.embed_documents(texts)


def _cache_codes():
    return [
        {"code": "packing bags", "definition": "packing a bag", "examples": [{"quote": "suitcase"}]},
        {"code": "packing luggage", "definition": "packing a case", "examples": [{"quote": "boxes"}]},
        {"code": "crying at the station", "definition": "crying", "examples": [{"quote": "train"}]},
    ]


def test_rank_embeds_each_code_once_and_skips_unchanged_text():
    clear_embedding_cache()
    embeddings = _CountingEmbeddings("rank-once")
    retriever = EmbeddingCodeRetriever(embeddings)
    codes = _cache_codes()
    retriever.rank("I was packing bags", codes, k=1)
    assert embeddings.documents == [code_text(entry) for entry in codes]
    assert embeddings.queries == ["I was packing bags"]
    embeddings.documents.clear()
    embeddings.queries.clear()
    retriever.rank("I was packing bags", codes, k=1)
    assert embeddings.documents == []
    assert embeddings.queries == []
    changed = [dict(codes[0], definition="a new definition"), *codes[1:]]
    retriever.rank("I was packing bags", changed, k=1)
    assert embeddings.documents == [code_text(changed[0])]
    clear_embedding_cache()


def test_two_retrievers_with_the_same_model_share_the_cache():
    clear_embedding_cache()
    first = _CountingEmbeddings("shared-model")
    second = _CountingEmbeddings("shared-model")
    codes = _cache_codes()
    EmbeddingCodeRetriever(first).rank("passage", codes, k=1)
    EmbeddingCodeRetriever(second).rank("passage", codes, k=1)
    assert len(second.documents) == 0
    other = _CountingEmbeddings("other-model")
    EmbeddingCodeRetriever(other).rank("passage", codes, k=1)
    assert len(other.documents) == len(codes)
    clear_embedding_cache()


def test_arank_many_embeds_changed_entries_and_coalesces_concurrent_calls():
    asyncio.run(_arank_many_cache())


async def _arank_many_cache():
    clear_embedding_cache()
    embeddings = _CountingEmbeddings("arank-cache")
    retriever = EmbeddingCodeRetriever(embeddings)
    codes = _cache_codes()
    passages = ["passage alpha unique", "passage beta unique"]
    limiter = CallLimiter(4, max_retries=0, backoff=0)

    await retriever.arank_many(passages, codes, k=2, limiter=limiter)
    first = list(embeddings.documents)
    assert first.count(code_text(codes[0])) == 1
    embeddings.documents.clear()
    await retriever.arank_many(passages, codes, k=2, limiter=limiter)
    assert embeddings.documents == []

    changed = [dict(codes[0], definition="changed definition text"), *codes[1:]]
    await retriever.arank_many(passages, changed, k=2, limiter=limiter)
    assert embeddings.documents == [code_text(changed[0])]

    clear_embedding_cache()
    concurrent = _CountingEmbeddings("arank-concurrent")
    retriever = EmbeddingCodeRetriever(concurrent)
    await asyncio.gather(
        retriever.arank_many(passages, codes, k=2, limiter=CallLimiter(4, max_retries=0, backoff=0)),
        retriever.arank_many(passages, codes, k=2, limiter=CallLimiter(4, max_retries=0, backoff=0)),
    )
    expected = [code_text(entry) for entry in codes] + passages
    assert sorted(concurrent.documents) == sorted(expected)

    clear_embedding_cache()
    flaky = _CountingEmbeddings("arank-flaky")
    flaky.fail_once = True
    retriever = EmbeddingCodeRetriever(flaky)
    with pytest.raises(RuntimeError, match="embed failed"):
        await retriever.arank_many(passages, codes, k=2, limiter=CallLimiter(2, max_retries=0, backoff=0))
    failed = len(flaky.documents)
    await retriever.arank_many(passages, codes, k=2, limiter=CallLimiter(2, max_retries=0, backoff=0))
    assert len(flaky.documents) > failed
    clear_embedding_cache()


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


def test_pack_indices_counts_the_assembled_request_and_splits_or_warns():
    def measured(batch, overrides=None):
        del overrides
        return 50 + 40 * len(batch)

    batches, warnings = pack_indices(
        [0, 1],
        ["a", "b"],
        batch_size=8,
        token_budget=100,
        measure=measured,
    )
    assert warnings == []
    assert [batch.indices for batch in batches] == [[0], [1]]

    def never_fits(batch, overrides=None):
        del batch, overrides
        return 500

    batches, warnings = pack_indices(
        [0],
        ["nospaces"],
        batch_size=1,
        token_budget=100,
        names=["p-huge"],
        measure=never_fits,
    )
    assert batches[0].indices == [0]
    assert batches[0].slices is None
    assert "sent alone" in warnings[0]
    assert "p-huge" in warnings[0]

    text = " ".join(["word"] * 20)
    batches, warnings = pack_indices(
        [0],
        [text],
        batch_size=1,
        token_budget=25,
        measure=lambda batch, overrides=None: len((overrides or {}).get(batch[0], text)),
    )
    assert batches[0].slices
    assert "split into" in warnings[0]
    for start, end in batches[0].slices:
        assert len(text[start:end]) <= 25

    long_text = "hello world " * 10
    whole, quiet = pack_indices([0], [long_text], batch_size=1, token_budget=80)
    assert quiet == []
    assert whole[0].slices is None
    split, noisy = pack_indices(
        [0],
        [long_text],
        batch_size=1,
        token_budget=80,
        tokenizer=_CharTokenizer(),
    )
    assert split[0].slices
    assert "split into" in noisy[0]


def test_incremental_sweep_stamps_a_reviewed_revision_and_a_tiny_budget_warns():
    result = open_code(
        CORPUS,
        CONTEXT,
        model=FakeCoder(),
        seed=7,
        sweep_every=1,
        full_audit_every=0,
        pipeline=False,
        batch_size=1,
        concurrency=1,
    )
    assert result.codebook
    assert all(entry.get("reviewed_revision") for entry in result.codebook)
    assert any(not item.get("consolidation") and item.get("audit") == "incremental" for item in result.sweep_log)
    assert result.sweep_log[-1]["consolidation"] is True
    assert result.sweep_log[-1]["audit"] == "full"
    assert result.metadata["full_audit_every"] == 0
    user_payloads = []
    for item in result.prompt_log:
        if item["kind"] != "sweep":
            continue
        user_payloads.append(item["messages"][-1]["content"])
    assert user_payloads
    assert all("\n" not in payload for payload in user_payloads)
    assert any("Incremental review" in payload for payload in user_payloads)

    tight = open_code(
        ["I spent the morning packing my suitcase."],
        CONTEXT,
        model=FakeCoder(),
        seed=1,
        token_budget=200,
        sweep_every=100,
        pipeline=False,
        concurrency=1,
    )
    assert any("sent alone" in warning for warning in tight.warnings)
    assert tight.segments

    with pytest.raises(ValueError):
        open_code(CORPUS, CONTEXT, model=FakeCoder(), full_audit_every=-1)
