"""0.6.7: oversized reservations, embedding cancellation, cache identity, budgets, alignment, sweep size."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.async_support import (
    CallLimiter,
    OversizedReservation,
    ainvoke_structured,
    estimate_input_tokens,
)
from automated_grounded_theory.batching import (
    repair_allowance_tokens,
    request_input_tokens,
    reserved_output_tokens,
    schema_token_count,
)
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.evaluate import aligned_assignment_agreement
from automated_grounded_theory.graph import RunConfig, _invoke_coding, _pack_jobs
from automated_grounded_theory.prompts import build_candidate_discovery_messages, build_coding_messages, message_text
from automated_grounded_theory.retrieval import (
    LexicalCodeRetriever,
    cached_aembed_texts,
    cached_embed_texts,
    clear_embedding_cache,
    jaccard,
    select_prompt_codes,
    shared_codebook_vectors,
)
from automated_grounded_theory.schemas import BatchCoding, PassageCoding
from automated_grounded_theory.version import __version__
from tests.fake_llm import FakeCoder

TRUST = "trusting managers to share information"
DISTRUST = "not trusting managers to share information"


def test_version_is_0_6_7():
    assert __version__ == "0.6.15"


def test_oversized_token_reservation_fails_instead_of_waiting():
    text = "x"
    while estimate_input_tokens(text) <= 60:
        text += "x"
    assert estimate_input_tokens(text) > 60

    limiter = CallLimiter(
        1,
        tokens_per_minute=60,
        requests_per_minute=0,
        max_retries=0,
        backoff=0,
        request_timeout=30,
    )
    called = {"n": 0}

    class Model:
        async def ainvoke(self, payload):
            called["n"] += 1
            return "ok"

    async def run():
        with pytest.raises(OversizedReservation, match="60") as caught:
            await asyncio.wait_for(limiter.ainvoke(Model(), text), timeout=0.5)
        return str(caught.value)

    message = asyncio.run(run())
    assert called["n"] == 0
    assert "allow_oversized" in message
    assert "tokens_per_minute" in message


def test_allow_oversized_drains_a_full_bucket():
    text = "x"
    while estimate_input_tokens(text) <= 60:
        text += "x"
    clock = {"t": 0.0}

    async def sleep(seconds):
        clock["t"] += seconds

    limiter = CallLimiter(
        1,
        tokens_per_minute=60,
        requests_per_minute=0,
        allow_oversized=True,
        max_retries=0,
        backoff=0,
        request_timeout=0,
    )
    limiter._clock_override = lambda: clock["t"]
    limiter._sleep_override = sleep
    limiter._tokens = 0.0
    limiter._token_stamp = 0.0

    class Model:
        async def ainvoke(self, payload):
            return "ok"

    asyncio.run(asyncio.wait_for(limiter.ainvoke(Model(), text), timeout=1.0))
    assert clock["t"] == pytest.approx(60.0)
    assert limiter._tokens == pytest.approx(0.0)


class _DimEmbeddings:
    def __init__(self, dimensions: int, *, lie: int | None = None, base_url: str = "") -> None:
        self.model = "shared-name"
        self.dimensions = dimensions
        self.base_url = base_url
        self.lie = lie
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        width = self.dimensions if self.lie is None else self.lie
        return [[float(i + 1) for i in range(width)] for _ in texts]


def test_embedding_cache_keeps_dimensions_apart_and_checks_width():
    clear_embedding_cache()
    narrow = _DimEmbeddings(3)
    wide = _DimEmbeddings(5)
    cached_embed_texts(narrow, ["hello"])
    cached_embed_texts(wide, ["hello"])
    assert narrow.calls == 1
    assert wide.calls == 1
    again = _DimEmbeddings(3)
    assert len(cached_embed_texts(again, ["hello"])[0]) == 3
    assert again.calls == 0
    assert len(cached_embed_texts(wide, ["hello"])[0]) == 5
    assert wide.calls == 1
    other_endpoint = _DimEmbeddings(3, base_url="https://other.example")
    cached_embed_texts(other_endpoint, ["hello"])
    assert other_endpoint.calls == 1
    liar = _DimEmbeddings(5, lie=3)
    with pytest.raises(RuntimeError, match="configured for 5"):
        cached_embed_texts(liar, ["different text"])
    clear_embedding_cache()


class _SlowEmbeddings:
    def __init__(self, name: str) -> None:
        self.model = name
        self.dimensions = 3
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def aembed_documents(self, texts):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return [[1.0, 2.0, 3.0] for _ in texts]


def _limiter() -> CallLimiter:
    return CallLimiter(2, max_retries=0, backoff=0, request_timeout=0)


def test_cancelled_embedding_can_be_requested_again():
    asyncio.run(_cancelled_embedding_can_be_requested_again())


async def _cancelled_embedding_can_be_requested_again():
    clear_embedding_cache()
    embeddings = _SlowEmbeddings("cancel-then-retry")
    limiter = _limiter()

    async def once():
        return await cached_aembed_texts(embeddings, ["same text"], batch_size=8, limiter=limiter)

    task = asyncio.create_task(once())
    await embeddings.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    embeddings.release.set()
    result = await asyncio.wait_for(once(), timeout=1.0)
    assert result == [[1.0, 2.0, 3.0]]
    assert embeddings.calls == 2

    clear_embedding_cache()
    snap = _SlowEmbeddings("cancel-snapshot")
    entries = [{"code": "alpha", "definition": "alpha alone", "examples": [{"quote": "quote"}]}]

    async def book():
        return await shared_codebook_vectors(snap, entries, limiter=_limiter(), batch_size=8)

    task = asyncio.create_task(book())
    await snap.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    snap.release.set()
    vectors = await asyncio.wait_for(book(), timeout=1.0)
    assert vectors == [[1.0, 2.0, 3.0]]
    assert snap.calls == 2
    clear_embedding_cache()


def test_cancelling_one_embedding_waiter_leaves_the_other():
    asyncio.run(_cancelling_one_embedding_waiter_leaves_the_other())


async def _cancelling_one_embedding_waiter_leaves_the_other():
    clear_embedding_cache()
    embeddings = _SlowEmbeddings("two-waiters")
    limiter = _limiter()

    async def once():
        return await cached_aembed_texts(embeddings, ["shared"], batch_size=8, limiter=limiter)

    owner = asyncio.create_task(once())
    await embeddings.started.wait()
    waiter_a = asyncio.create_task(once())
    waiter_b = asyncio.create_task(once())
    await asyncio.sleep(0.05)
    waiter_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_a
    embeddings.release.set()
    assert await asyncio.wait_for(waiter_b, timeout=1.0) == [[1.0, 2.0, 3.0]]
    assert await asyncio.wait_for(owner, timeout=1.0) == [[1.0, 2.0, 3.0]]
    assert embeddings.calls == 1

    clear_embedding_cache()
    snap = _SlowEmbeddings("two-snapshot-waiters")
    entries = [{"code": "beta", "definition": "beta alone", "examples": [{"quote": "quote"}]}]

    async def book():
        return await shared_codebook_vectors(snap, entries, limiter=_limiter(), batch_size=8)

    owner = asyncio.create_task(book())
    await snap.started.wait()
    waiter_a = asyncio.create_task(book())
    waiter_b = asyncio.create_task(book())
    await asyncio.sleep(0.05)
    waiter_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_a
    snap.release.set()
    assert await asyncio.wait_for(waiter_b, timeout=1.0) == [[1.0, 2.0, 3.0]]
    assert await asyncio.wait_for(owner, timeout=1.0) == [[1.0, 2.0, 3.0]]
    assert snap.calls == 1
    clear_embedding_cache()


def _code(label: str, definition: str, frequency: int = 1) -> dict:
    return {
        "code": label,
        "definition": definition,
        "frequency": frequency,
        "examples": [{"quote": label, "passage_id": "ex"}],
    }


def _union_labels(per_passage: list[list[str]]) -> list[str]:
    ranked = []
    seen = set()
    for labels in per_passage:
        for label in labels:
            key = label.casefold()
            if key not in seen:
                seen.add(key)
                ranked.append(label)
    return ranked


def _messages_from_log(entry: dict) -> list:
    messages = []
    for item in entry["messages"]:
        content = item["content"]
        if item["role"] == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def test_assembled_coding_request_stays_within_budget():
    definition = "detail " * 40
    entries = []
    passages = []
    for passage_index in range(4):
        tokens = []
        for code_index in range(12):
            token = f"p{passage_index}c{code_index}zz"
            label = f"describing {token}"
            entries.append(_code(label, f"{token} {definition}"))
            tokens.append(token)
        passages.append(
            {
                "passage_id": f"pass-{passage_index}",
                "text": " ".join(tokens),
                "doc_id": "d1",
            }
        )
    for index in range(8):
        entries.append(_code(f"frequent code {index}", f"frequent {definition}", frequency=50))
    codebook = {entry["code"]: entry for entry in entries}
    retriever = LexicalCodeRetriever()
    per_passage = [retriever.rank(passage["text"], entries, k=12) for passage in passages]
    assert [len(labels) for labels in per_passage] == [12, 12, 12, 12]
    view, omitted = select_prompt_codes(entries, _union_labels(per_passage), frequent_limit=8)
    assert len(view) >= 48
    context = ResearchContext(research_question="What is happening?")
    real_messages = build_coding_messages(
        context=context,
        codebook_view=view,
        omitted_codes=omitted,
        passages=passages,
        neighbor_chars=0,
        feedback=None,
        prompt_cache=False,
        nearest=per_passage,
    )

    def bulk(entry: dict) -> int:
        return len(str(entry.get("definition") or ""))

    old_view = sorted(entries, key=bulk, reverse=True)[:20]
    old_messages = build_coding_messages(
        context=context,
        codebook_view=old_view,
        omitted_codes=len(entries) - len(old_view),
        passages=passages,
        neighbor_chars=0,
        feedback=None,
        prompt_cache=False,
        nearest=[[] for _ in passages],
    )
    reserved = reserved_output_tokens(4, max_codes_per_passage=6)
    old_cost = len(message_text(old_messages[0])) // 3 + len(message_text(old_messages[1])) // 3 + reserved
    quad_cost = request_input_tokens(real_messages, BatchCoding) + reserved
    one_messages = build_coding_messages(
        context=context,
        codebook_view=select_prompt_codes(entries, per_passage[0], frequent_limit=8)[0],
        omitted_codes=0,
        passages=passages[:1],
        neighbor_chars=0,
        feedback=None,
        prompt_cache=False,
        nearest=per_passage[:1],
    )
    one_cost = request_input_tokens(one_messages, BatchCoding) + reserved_output_tokens(1, max_codes_per_passage=6)
    assert len(old_view) == 20
    assert len(view) > 20
    assert old_cost < quad_cost
    budget = max(old_cost, one_cost)
    assert quad_cost > budget

    runtime = RunConfig(
        coder=FakeCoder(),
        sweeper=FakeCoder(),
        retriever=retriever,
        embeddings=None,
        context=context,
        prompt_cache=False,
        top_k_codes=12,
        frequent_codes=8,
        token_budget=budget,
        validation_retries=0,
        pipeline=False,
        limiter=CallLimiter(1, max_retries=0, backoff=0, request_timeout=5),
    )
    state = {"passages": passages, "codebook": codebook, "parallel": False}
    jobs = asyncio.run(_pack_jobs(state, runtime, [0, 1, 2, 3], budget, 4, None))
    assert jobs
    assert all(len(job[1]) < 4 for job in jobs)

    async def run():
        return await _invoke_coding(runtime, passages, codebook, None)

    aligned = asyncio.run(run())
    assert len(aligned) == 4
    assert runtime.prompt_records
    for record in runtime.prompt_records:
        if record["kind"] != "code_passage":
            continue
        messages = _messages_from_log(record)
        payload = json.loads(message_text(messages[-1]))
        n_passages = len(payload["passages"])
        size = request_input_tokens(messages, BatchCoding) + reserved_output_tokens(
            n_passages,
            max_codes_per_passage=6,
        )
        assert size <= budget
        assert len(payload["codebook"]) < len(view)


class _RepairModel:
    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            text = "\n".join(getattr(message, "content", str(message)) for message in messages)
            if "REPAIR THIS OUTPUT" in text:
                parsed = PassageCoding(passage_id="p1", codes=[], memo="")
                return {"raw": AIMessage(content="{}"), "parsed": parsed, "parsing_error": None}
            return {
                "raw": AIMessage(content="not json"),
                "parsed": None,
                "parsing_error": "e" * 8000,
            }

        return RunnableLambda(_call)


class _Limiter:
    async def ainvoke(self, runnable, messages, **kwargs):
        return runnable.invoke(messages)


def test_repair_request_is_remeasured_against_the_budget():
    messages = [SystemMessage(content="Code the passage."), HumanMessage(content='{"passages":[]}')]
    allowance = repair_allowance_tokens(PassageCoding)
    budget = request_input_tokens(messages, PassageCoding) + allowance
    long_repair = messages + [
        HumanMessage(
            content=(
                "REPAIR THIS OUTPUT\n"
                f"The previous output was not valid PassageCoding: {'e' * 8000}\n"
                "Return only the structured object."
            )
        )
    ]
    assert request_input_tokens(long_repair, PassageCoding) > budget

    async def run():
        return await ainvoke_structured(
            _RepairModel(),
            PassageCoding,
            messages,
            _Limiter(),
            token_budget=budget,
            reserved_output=0,
        )

    (payload, _usage), used = asyncio.run(run())
    assert payload["passage_id"] == "p1"
    assert "REPAIR THIS OUTPUT" in used[-1].content
    assert request_input_tokens(used, PassageCoding) <= budget


def test_polarity_near_duplicates_do_not_align_unless_mapped():
    assert jaccard(TRUST, DISTRUST) >= 0.5
    left = {"segments": [{"passage_id": "p1", "code": TRUST}]}
    right = {"segments": [{"passage_id": "p1", "code": DISTRUST}]}
    score = aligned_assignment_agreement(left, right)
    assert score["mean_jaccard"] == 0.0
    assert score["aligned_codes"] == 0
    spelling = aligned_assignment_agreement(
        {"segments": [{"passage_id": "p1", "code": "packing bags"}]},
        {"segments": [{"passage_id": "p1", "code": "packing bag"}]},
    )
    assert spelling["mean_jaccard"] == 1.0
    mapped = aligned_assignment_agreement(left, right, mapping={DISTRUST: TRUST})
    assert mapped["mean_jaccard"] == 1.0
    assert mapped["aligned_codes"] == 1


def test_discovery_request_is_labels_only():
    codes = [
        {"code": "packing bags", "definition": "putting clothes in a bag", "examples": [{"quote": "suitcase"}]},
        {"code": "crying at the station", "definition": "crying", "examples": [{"quote": "train"}]},
    ]
    messages = build_candidate_discovery_messages(
        context=ResearchContext(research_question="How do people leave?"),
        codes=codes,
        prompt_cache=False,
    )
    payload = json.loads(message_text(messages[-1]))
    assert set(payload["labels"][0]) == {"code_id", "label"}
    assert "definition" not in payload["labels"][0]
    assert "suitcase" not in message_text(messages[-1])
