"""0.6.8: async retrieval, budget gate, bounded sweep comparisons, deterministic backoff."""

import asyncio
import json
import logging
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import open_code
from automated_grounded_theory.async_support import CallLimiter, ainvoke_structured
from automated_grounded_theory.batching import request_input_tokens
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.graph import (
    RunConfig,
    _attach_oversized,
    _coding_messages_for,
    _coding_request_cost,
    _invoke_coding,
    _sweep_chunks,
    _wave,
)
from automated_grounded_theory.results import result_from_state
from automated_grounded_theory.prompts import extract_payload, message_text, validation_feedback
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.retrieval import (
    EmbeddingCodeRetriever,
    LexicalCodeRetriever,
    clear_embedding_cache,
    code_text,
    jaccard,
    review_fingerprint,
)
from automated_grounded_theory.schemas import MergeRejection, PassageCoding, SweepCoding
from automated_grounded_theory.version import __version__
from tests.fake_llm import FakeCoder


def test_version_is_0_6_8():
    assert __version__ == "0.9.6"
    assert POLICY_VERSION == "0.9.6"


class _HeartbeatEmbeddings:
    """Sync methods block the loop. Async methods yield. Planning must use the async ones."""

    def __init__(self) -> None:
        self.model = "heartbeat-embed-v068"
        self.dimensions = 2
        self.documents: list[str] = []
        self.queries: list[str] = []
        self.blocked = False

    def embed_documents(self, texts):
        self.blocked = True
        time.sleep(0.2)
        self.documents.extend(texts)
        return [[1.0, 1.0] for _ in texts]

    def embed_query(self, text):
        self.blocked = True
        time.sleep(0.2)
        self.queries.append(text)
        return [1.0, 1.0]

    async def aembed_documents(self, texts):
        await asyncio.sleep(0.05)
        self.documents.extend(list(texts))
        return [[1.0, 1.0] for _ in texts]

    async def aembed_query(self, text):
        await asyncio.sleep(0.05)
        self.queries.append(text)
        return [1.0, 1.0]


def _messages_from_log(entry: dict) -> list:
    messages = []
    for item in entry["messages"]:
        content = item["content"]
        if item["role"] == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def test_planning_retrieves_async_and_embeds_each_passage_once():
    asyncio.run(_planning_retrieves_async_and_embeds_each_passage_once())


async def _planning_retrieves_async_and_embeds_each_passage_once():
    clear_embedding_cache()
    embeddings = _HeartbeatEmbeddings()
    entries = [
        {"code": "packing bags", "definition": "putting clothes in a bag", "examples": [{"quote": "suitcase"}]},
        {"code": "crying at the station", "definition": "crying as the train leaves", "examples": [{"quote": "crying"}]},
        {"code": "starting a new job", "definition": "being hired", "examples": [{"quote": "hired"}]},
    ]
    passages = [
        {"passage_id": "p1", "text": "I was packing the first suitcase on Monday.", "doc_id": "d1"},
        {"passage_id": "p2", "text": "I was packing the second suitcase on Tuesday.", "doc_id": "d1"},
    ]
    codebook = {entry["code"]: entry for entry in entries}
    context = ResearchContext(research_question="What is happening?")
    runtime = RunConfig(
        coder=FakeCoder(),
        sweeper=FakeCoder(),
        retriever=EmbeddingCodeRetriever(embeddings),
        embeddings=embeddings,
        context=context,
        prompt_cache=False,
        top_k_codes=3,
        frequent_codes=0,
        token_budget=8000,
        validation_retries=0,
        pipeline=False,
        limiter=CallLimiter(2, max_retries=0, backoff=0, request_timeout=0),
    )
    state = {
        "passages": passages,
        "queue": [0, 1],
        "next_index": 0,
        "codebook": codebook,
        "parallel": False,
        "metadata": {
            "token_budget": 8000,
            "passages_per_request": 2,
            "passages_per_codebook_update": 2,
            "review_mode": "wave",
            "sweep_every": 50,
        },
        "passages_since_sweep": 0,
    }
    beats = 0

    async def heartbeat():
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    before = beats
    await _wave(state, runtime)
    during = beats - before
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert during >= 3
    assert embeddings.blocked is False
    assert embeddings.queries.count(passages[0]["text"]) == 1
    assert embeddings.queries.count(passages[1]["text"]) == 1
    for entry in entries:
        assert embeddings.documents.count(code_text(entry)) == 1

    await _invoke_coding(runtime, passages, codebook, None)
    assert embeddings.queries.count(passages[0]["text"]) == 1
    assert embeddings.queries.count(passages[1]["text"]) == 1
    for entry in entries:
        assert embeddings.documents.count(code_text(entry)) == 1
    assert embeddings.blocked is False
    clear_embedding_cache()


def _grow_feedback(runtime, passage, per_passage, view, omitted) -> str:
    feedback = "quote_not_verbatim "
    while True:
        messages = _coding_messages_for(runtime, [passage], per_passage, {"p1": feedback}, view, omitted)
        if _coding_request_cost(runtime, messages, 1) >= 13031:
            return feedback
        feedback += "x" * 400


def test_validation_feedback_is_shortened_when_the_retry_can_fit():
    context = ResearchContext(research_question="What is happening?")
    entry = {
        "code": "packing bags",
        "definition": "putting clothes in a bag",
        "frequency": 2,
        "examples": [{"quote": "I was packing bags."}],
    }
    passages = [
        {"passage_id": "p1", "text": "I was packing bags.", "doc_id": "d1"},
        {"passage_id": "p2", "text": "The train left.", "doc_id": "d1"},
    ]
    codebook = {entry["code"]: entry}
    retriever = LexicalCodeRetriever()
    per_passage = [retriever.rank(passages[0]["text"], [entry], k=4)]
    runtime = RunConfig(
        coder=FakeCoder(),
        sweeper=FakeCoder(),
        retriever=retriever,
        embeddings=None,
        context=context,
        prompt_cache=False,
        top_k_codes=4,
        frequent_codes=0,
        token_budget=6000,
        validation_retries=1,
        pipeline=False,
        limiter=CallLimiter(1, max_retries=0, backoff=0, request_timeout=5),
    )
    feedback = "quote_not_verbatim " + ("x" * 20000)
    oversized = _coding_messages_for(runtime, [passages[0]], per_passage, {"p1": feedback}, [entry], 0)
    assert _coding_request_cost(runtime, oversized, 1) > 6000

    aligned = asyncio.run(_invoke_coding(runtime, passages, codebook, {"p1": feedback}))
    assert len(aligned) == 2
    assert runtime.coder.call_count >= 1
    sent = [record for record in runtime.prompt_records if record["kind"] == "code_passage"]
    assert len(sent) == 2
    assert any("Repacked" in note for note in runtime.notes)
    assert any("Shortened validation feedback" in note for note in runtime.notes)
    assert runtime.oversized_requests == []
    for record in sent:
        messages = _messages_from_log(record)
        assert _coding_request_cost(runtime, messages, 1) <= 6000
        payload = json.loads(message_text(messages[-1]))
        assert len("\n".join(payload.get("feedback") or [])) < len(feedback)
    packed = json.loads(message_text(_messages_from_log(sent[0])[-1]))
    assert any(row.get("label") == "packing bags" for row in packed.get("codebook") or [])


def test_validation_feedback_retry_over_budget_is_sent_and_flagged():
    """The 13,031-vs-6,000 retry is sent, warned, and flagged."""
    context = ResearchContext(research_question="What is happening?")
    entry = {
        "code": "packing bags",
        "definition": "putting clothes in a bag",
        "frequency": 2,
        "examples": [{"quote": "I was packing bags."}],
    }
    passage = {"passage_id": "p1", "text": "word " * 12000, "doc_id": "d1"}
    codebook = {entry["code"]: entry}
    retriever = LexicalCodeRetriever()
    per_passage = [retriever.rank(passage["text"], [entry], k=4)]
    coder = FakeCoder()
    runtime = RunConfig(
        coder=coder,
        sweeper=FakeCoder(),
        retriever=retriever,
        embeddings=None,
        context=context,
        prompt_cache=False,
        top_k_codes=4,
        frequent_codes=0,
        token_budget=6000,
        validation_retries=1,
        pipeline=False,
        limiter=CallLimiter(1, max_retries=0, backoff=0, request_timeout=5),
    )
    feedback = _grow_feedback(runtime, passage, per_passage, [entry], 0)
    oversized = _coding_messages_for(runtime, [passage], per_passage, {"p1": feedback}, [entry], 0)
    assert _coding_request_cost(runtime, oversized, 1) >= 13031

    aligned = asyncio.run(_invoke_coding(runtime, [passage], codebook, {"p1": feedback}))
    assert coder.call_count >= 1
    assert aligned[0]["passage_id"] == "p1"
    assert aligned[0].get("coverage_status") != "failed"
    sent = [record for record in runtime.prompt_records if record["kind"] == "code_passage"]
    assert len(sent) == 1
    sent_cost = _coding_request_cost(runtime, _messages_from_log(sent[0]), 1)
    assert sent_cost >= 13031
    assert any("flagged" in note and "sent" in note for note in runtime.notes)
    flag = runtime.oversized_requests[0]
    assert flag["flag"] == "oversized_request"
    assert flag["passage_id"] == "p1"
    assert flag["estimated"] >= 13031
    assert flag["estimated"] == sent_cost
    assert flag["budget"] == 6000
    metadata, review = _attach_oversized(runtime, {"metadata": {"seed": 1}}, [])
    result = result_from_state(
        {
            "metadata": metadata,
            "review": review,
            "warnings": list(runtime.notes),
            "passages": [],
            "documents": [],
            "segments": [],
            "codebook": {},
        },
        interrupted=False,
        thread_id="",
        resumable=False,
    )
    recorded = result.metadata["oversized_requests"][0]
    assert recorded["passage_id"] == "p1"
    assert recorded["estimated"] >= 13031
    assert result.review[0]["flag"] == "oversized_request"
    assert result.review[0]["passage_id"] == "p1"
    assert result.review[0]["estimated"] >= 13031


def test_oversized_coding_is_sent_and_recorded_on_the_run():
    result = open_code(
        ["I spent the morning packing my suitcase."],
        "What is happening?",
        model=FakeCoder(),
        seed=1,
        token_budget=200,
        sweep_every=100,
        pipeline=False,
        concurrency=1,
    )
    assert result.segments
    flags = result.metadata["oversized_requests"]
    assert flags
    passage_ids = {str(item.get("passage_id") or "") for item in result.passages}
    assert flags[0]["flag"] == "oversized_request"
    assert flags[0]["passage_id"] in passage_ids
    assert flags[0]["estimated"] > 200
    assert flags[0]["budget"] == 200
    assert any(
        item.get("flag") == "oversized_request"
        and item.get("passage_id") == flags[0]["passage_id"]
        and item.get("estimated") == flags[0]["estimated"]
        for item in result.review
    )
    assert any("flagged" in warning and "sent" in warning for warning in result.warnings)


def test_validation_feedback_is_bounded():
    rows = [
        {"code": f"label {index}", "reason": "quote_not_verbatim", "quote": "q" * 400}
        for index in range(40)
    ]
    text = validation_feedback(rows)
    assert len(text) <= 2200
    assert "omitted" in text


def test_schema_repair_over_budget_is_sent_and_flagged(caplog):
    messages = [SystemMessage(content="Code the passage."), HumanMessage(content='{"passages":[]}')]
    budget = request_input_tokens(messages, PassageCoding)
    calls = {"n": 0}
    flagged = []

    class Model:
        def with_structured_output(self, schema, **kwargs):
            def _call(payload):
                calls["n"] += 1
                return {
                    "raw": AIMessage(content="not json"),
                    "parsed": None,
                    "parsing_error": "bad",
                }

            return RunnableLambda(_call)

    class Limiter:
        async def ainvoke(self, runnable, payload, **kwargs):
            return runnable.invoke(payload)

    def _capture(flag):
        flagged.append(flag)

    with caplog.at_level(logging.WARNING, logger="automated_grounded_theory"):
        with pytest.raises(ValueError):
            asyncio.run(
                ainvoke_structured(
                    Model(),
                    PassageCoding,
                    messages,
                    Limiter(),
                    token_budget=budget,
                    reserved_output=0,
                    passage_id="p-repair",
                    on_overflow=_capture,
                )
            )
    assert calls["n"] == 2
    assert flagged[0].estimated > flagged[0].budget
    assert "p-repair" in str(flagged[0])
    assert "flagged" in caplog.text
    assert "sent" in caplog.text


class _DecideSweeper(FakeCoder):
    """Reject every hinted pair so the test can see that each one was judged."""

    def sweep(self, text: str) -> SweepCoding:
        payload = extract_payload(text)
        rejections = []
        for hint in payload.get("hints") or []:
            rejections.append(
                MergeRejection(
                    pair_id=str(hint.get("pair_id") or ""),
                    source=str(hint.get("source") or ""),
                    target=str(hint.get("target") or ""),
                    rationale="These name different incidents and stay as two codes.",
                    confidence=0.9,
                    same_action=False,
                    same_object=False,
                )
            )
        return SweepCoding(merges=[], rejections=rejections)


def _reviewed(label: str) -> dict:
    entry = {
        "code": label,
        "definition": f"Incidents described as {label}.",
        "revision": 1,
        "frequency": 1,
        "examples": [{"quote": f"quote for {label}", "passage_id": "ex"}],
    }
    entry["reviewed_revision"] = review_fingerprint(entry)
    return entry


def test_sweep_judges_every_partner_and_only_with_evidence():
    local = "packing luggage"
    tokens = ("aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii")
    partners = [f"packing luggage {token}" for token in tokens]
    assert len(partners) == 9
    assert jaccard(local, partners[0]) >= 0.5
    codes = [
        {
            "code": local,
            "definition": "Putting clothes into luggage.",
            "revision": 1,
            "frequency": 1,
            "examples": [{"quote": "packing the luggage", "passage_id": "ex"}],
        }
    ]
    codes.extend(_reviewed(label) for label in partners)
    context = ResearchContext(research_question="How do people leave?")
    runtime = RunConfig(
        coder=FakeCoder(),
        sweeper=_DecideSweeper(),
        retriever=LexicalCodeRetriever(),
        embeddings=None,
        context=context,
        prompt_cache=False,
        cross_chunk_limit=8,
        lexical_merge_threshold=0.5,
        candidate_limit=30,
        full_audit_every=4,
        sweep_chunk=20,
        pipeline=False,
        limiter=CallLimiter(2, max_retries=0, backoff=0, request_timeout=5),
    )

    parsed, candidates, _chunks = asyncio.run(
        _sweep_chunks(runtime, codes, 2, consolidation=False)
    )
    sweeps = [record for record in runtime.prompt_records if record["kind"] == "sweep"]
    assert len(sweeps) >= 2
    judged: set[frozenset[str]] = set()
    for record in sweeps:
        payload = json.loads(message_text(_messages_from_log(record)[-1]))
        assert len(payload.get("cross_chunk") or []) <= 8
        evidence = {}
        for entry in list(payload.get("codes") or []) + list(payload.get("cross_chunk") or []):
            evidence[entry["label"]] = entry
        for hint in payload.get("hints") or []:
            for side in (hint["source"], hint["target"]):
                entry = evidence[side]
                assert entry["definition"]
                assert entry["examples"]
            judged.add(frozenset((hint["source"], hint["target"])))
    for partner in partners:
        assert frozenset((local, partner)) in judged
    assert len(judged) >= 9
    decided = {
        frozenset((str(item.get("source") or ""), str(item.get("target") or "")))
        for item in parsed.get("rejections") or []
    }
    for partner in partners:
        assert frozenset((local, partner)) in decided
    offered = {
        frozenset((str(item.get("source") or ""), str(item.get("target") or "")))
        for item in candidates
    }
    for partner in partners:
        assert frozenset((local, partner)) in offered
