"""0.6.6: one shared limiter, coding time, repair usage, and token buckets."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import open_code
from automated_grounded_theory.async_support import (
    CallLimiter,
    ainvoke_structured,
    estimate_input_tokens,
)
from automated_grounded_theory.graph import _abatch_structured, fold_coding_times
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.schemas import PassageCoding
from automated_grounded_theory.version import __version__
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder


def test_version_is_0_6_6():
    assert __version__ == "0.7.0"
    assert POLICY_VERSION == "0.7.0"


def test_concurrent_batches_share_the_ceiling_and_report_the_peak():
    current = 0
    peak = 0
    lock = asyncio.Lock()

    class Slow:
        async def ainvoke(self, payload):
            nonlocal current, peak
            async with lock:
                current += 1
                peak = max(peak, current)
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return payload

    async def run():
        limiter = CallLimiter(2, max_retries=0, backoff=0)
        slow = Slow()
        await asyncio.gather(
            limiter.abatch(slow, [1, 2, 3, 4]),
            limiter.abatch(slow, [5, 6, 7, 8]),
            limiter.ainvoke(slow, 9),
        )
        return limiter

    limiter = asyncio.run(run())
    assert peak == 2
    assert limiter.max_in_flight == 2
    assert limiter.activity_report()["max_in_flight"] == 2


def test_batch_wall_time_is_not_summed_once_per_passage():
    shared = {
        "job_id": "batch-a",
        "wall_start": 10.0,
        "wall_end": 12.5,
        "request_latencies": [1.1],
        "queue_waits": [0.4],
    }
    other = {
        "job_id": "batch-b",
        "wall_start": 11.0,
        "wall_end": 14.0,
        "request_latencies": [0.2, 0.3],
        "queue_waits": [0.1],
    }
    folded = fold_coding_times([dict(shared) for _ in range(4)] + [dict(other), dict(other)])
    assert folded["wall_seconds"] == 4.0
    assert folded["request_latencies"] == [1.1, 0.2, 0.3]
    assert folded["queue_wait_seconds"] == 0.5
    inflated = 2.5 * 4 + 3.0 * 2
    assert folded["wall_seconds"] != inflated


def test_open_code_reports_wall_latency_and_queue_separately():
    result = open_code(
        CORPUS,
        CONTEXT,
        model=FakeCoder(),
        seed=7,
        sweep_every=25,
        batch_size=4,
        concurrency=2,
        pipeline=False,
    )
    metrics = result.metadata["metrics"]
    code_prompts = [item for item in result.prompt_log if item["kind"] == "code_passage"]
    assert len(result.passages) > len(code_prompts)
    assert code_prompts
    assert len(metrics["coding_seconds"]) == len(code_prompts)
    assert metrics["coding_request_latency_seconds"] == metrics["coding_seconds"]
    assert metrics["coding_wall_seconds"] == result.metadata["stage_seconds"]["coding"]
    assert metrics["coding_queue_wait_seconds"] >= 0
    assert metrics["max_in_flight"] >= 1
    assert metrics["activity"]["max_in_flight"] == metrics["max_in_flight"]
    assert len(metrics["request_latency_seconds"]) >= len(metrics["coding_seconds"])
    assert "queue_wait_seconds" in metrics
    assert result.metadata["tokens_per_minute"] == 0
    assert result.metadata["requests_per_minute"] == 0
    assert result.metadata["params"]["tokens_per_minute"] == 0
    assert result.metadata["params"]["requests_per_minute"] == 0


def _usage_message(total_input: int, total_output: int) -> AIMessage:
    return AIMessage(
        content="not json",
        usage_metadata={
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        },
    )


def test_schema_repair_adds_the_failed_response_usage():
    class Model:
        def with_structured_output(self, schema, **kwargs):
            def _call(messages):
                text = "\n".join(getattr(message, "content", str(message)) for message in messages)
                if "REPAIR THIS OUTPUT" in text:
                    parsed = PassageCoding(passage_id="p1", codes=[], memo="")
                    raw = AIMessage(
                        content="{}",
                        usage_metadata={"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                    )
                    return {"raw": raw, "parsed": parsed, "parsing_error": None}
                return {
                    "raw": _usage_message(11, 3),
                    "parsed": None,
                    "parsing_error": "nope",
                }

            return RunnableLambda(_call)

    async def run():
        return await ainvoke_structured(
            Model(),
            PassageCoding,
            [HumanMessage(content="{}")],
            CallLimiter(1, max_retries=0, backoff=0),
        )

    (payload, usage), _messages = asyncio.run(run())
    assert payload["passage_id"] == "p1"
    assert usage["input_tokens"] == 19
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 26
    assert usage["attempt_usage"] == [
        {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
        {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
    ]


def test_batch_parse_failure_keeps_that_response_usage():
    class Model:
        calls = 0

        def with_structured_output(self, schema, **kwargs):
            def _call(messages):
                self.calls += 1
                if self.calls == 1:
                    return {"raw": _usage_message(6, 4), "parsed": None, "parsing_error": "nope"}
                parsed = PassageCoding(passage_id="p1", codes=[], memo="")
                raw = AIMessage(
                    content="{}",
                    usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
                )
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

            return RunnableLambda(_call)

    runtime = SimpleNamespace(
        limiter=CallLimiter(2, max_retries=0, backoff=0),
        sweep_timeout=30,
        prompt_records=[],
        _log_lock=threading.Lock(),
        prompt_log_path=None,
    )

    async def run():
        return await _abatch_structured(
            Model(),
            PassageCoding,
            [[HumanMessage(content="code")]],
            kind="code_passage",
            model_name="fake",
            passage_ids=["p1"],
            runtime=runtime,
        )

    asyncio.run(run())
    usage = runtime.prompt_records[0]["usage"]
    assert usage["total_tokens"] == 17
    assert [row["total_tokens"] for row in usage["attempt_usage"]] == [10, 7]


def test_tokens_per_minute_reserves_then_reconciles_to_actual_usage():
    clock = {"t": 50.0}

    async def sleep(seconds):
        clock["t"] += seconds

    limiter = CallLimiter(
        2,
        tokens_per_minute=100_000,
        requests_per_minute=0,
        reserved_output_tokens=40,
        max_retries=0,
        backoff=0,
    )
    limiter._clock_override = lambda: clock["t"]
    limiter._sleep_override = sleep

    class Model:
        async def ainvoke(self, payload):
            return {
                "raw": AIMessage(
                    content="ok",
                    usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                ),
                "parsed": {"ok": True},
            }

    payload = "hello token bucket"
    reserved = estimate_input_tokens(payload) + 40
    assert reserved > 1

    async def run():
        before = limiter._tokens
        await limiter.ainvoke(Model(), payload)
        return before

    before = asyncio.run(run())
    assert before - limiter._tokens == pytest.approx(30)
    assert clock["t"] == 50.0


def test_token_bucket_waits_for_the_reservation_not_one_request():
    clock = {"t": 10.0}

    async def sleep(seconds):
        clock["t"] += seconds

    payload = "hello"
    reserve = 25
    need = estimate_input_tokens(payload) + reserve
    limiter = CallLimiter(
        1,
        tokens_per_minute=60,
        requests_per_minute=0,
        reserved_output_tokens=reserve,
        max_retries=0,
        backoff=0,
    )
    limiter._clock_override = lambda: clock["t"]
    limiter._sleep_override = sleep
    limiter._tokens = 5
    limiter._token_stamp = clock["t"]

    class Model:
        async def ainvoke(self, payload):
            return {
                "raw": AIMessage(
                    content="ok",
                    usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                ),
                "parsed": {"ok": True},
            }

    asyncio.run(limiter.ainvoke(Model(), payload))
    assert clock["t"] == pytest.approx(10.0 + (need - 5))
    assert limiter._tokens == pytest.approx(need - 8)


def test_requests_per_minute_is_separate_from_the_token_bucket():
    clock = {"t": 0.0}

    async def sleep(seconds):
        clock["t"] += seconds

    limiter = CallLimiter(4, tokens_per_minute=0, requests_per_minute=60, max_retries=0, backoff=0)
    limiter._clock_override = lambda: clock["t"]
    limiter._sleep_override = sleep
    limiter._requests = 0.25
    limiter._request_stamp = 0.0

    class Ok:
        async def ainvoke(self, payload):
            return "ok"

    asyncio.run(limiter.ainvoke(Ok(), "x"))
    assert clock["t"] == pytest.approx(0.75)
    assert limiter._requests == pytest.approx(0.0)
    assert limiter._tokens == 0.0
