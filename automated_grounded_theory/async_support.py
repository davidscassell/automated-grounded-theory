"""Shared concurrency limit, rate-limit backoff, and a notebook-safe sync runner.

LLM calls go through :meth:`CallLimiter.ainvoke` or :meth:`CallLimiter.abatch`.
Embeddings use a separate semaphore of the same size so a coding task can embed
and then call the model without taking the model slot twice.

:func:`run_sync` drives a coroutine from ordinary Python and from a running
loop (Jupyter). Inside a running loop the coroutine is executed on a worker
thread with its own loop, so the notebook loop is not re-entered.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import random
import threading
from typing import Any, Callable, Sequence

class CallLimiter:
    """Cap in-flight work, adapt on 429, and retry with full-jitter backoff.

    ``ainvoke`` takes one slot from an AIMD limit (the configured concurrency
    is the ceiling; a rate limit halves it, and a streak of successes adds one
    back). ``abatch`` calls LangChain ``abatch`` with that same limit as
    ``max_concurrency``. A response cache keyed by the payload replays an
    identical prompt without calling the model again.

    Jitter changes only the wait, never the stored result. ``Retry-After``
    on the exception, when present, replaces the computed wait.
    """

    def __init__(
        self,
        concurrency: int = 2,
        *,
        max_retries: int = 4,
        backoff: float = 0.5,
        max_backoff: float = 8.0,
        tokens_per_minute: float = 0,
        cache: dict | None = None,
    ) -> None:
        self.concurrency = max(1, int(concurrency))
        self.max_concurrency = self.concurrency
        self.limit = self.concurrency
        self.max_retries = max(0, int(max_retries))
        self.backoff = max(0.0, float(backoff))
        self.max_backoff = max(self.backoff, float(max_backoff))
        self.tokens_per_minute = max(0.0, float(tokens_per_minute))
        self._tokens = self.tokens_per_minute
        self._token_stamp = 0.0
        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.embed_semaphore = asyncio.Semaphore(self.concurrency)
        self._gate = asyncio.Condition()
        self.max_in_flight = 0
        self._in_flight = 0
        self.rate_limit_retries = 0
        self._success_streak = 0
        self.cache = cache if cache is not None else {}
        self.cache_hits = 0

    async def ainvoke(self, runnable, payload):
        """``await runnable.ainvoke(payload)`` under the AIMD cap, with rate-limit retries."""
        key = _cache_key(payload)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        await self._acquire()
        try:
            result = await self._retry(lambda: runnable.ainvoke(payload))
            self._on_success()
            self.cache[key] = result
            return result
        finally:
            await self._release_async()

    async def abatch(self, runnable, inputs: Sequence[Any]) -> list[Any]:
        """``runnable.abatch`` with the current AIMD limit, then retry rate-limited rows.

        Results stay aligned with ``inputs``. Completion order does not matter.
        Cached rows are not sent again.
        """
        if not inputs:
            return []
        output: list[Any] = [None] * len(inputs)
        pending: list[int] = []
        for index, payload in enumerate(inputs):
            cached = self.cache.get(_cache_key(payload))
            if cached is not None:
                self.cache_hits += 1
                output[index] = cached
            else:
                pending.append(index)
        attempt_base = 0
        for attempt in range(self.max_retries + 1):
            if not pending:
                break
            batch = [inputs[index] for index in pending]
            try:
                results = await runnable.abatch(
                    batch,
                    config={"max_concurrency": self.limit},
                    return_exceptions=True,
                )
            except Exception as exc:
                if attempt >= self.max_retries or not is_rate_limit(exc):
                    raise
                self._on_rate_limit()
                await asyncio.sleep(self._wait(attempt, exc))
                continue
            still: list[int] = []
            rate_exc: BaseException | None = None
            for index, result in zip(pending, results):
                if isinstance(result, Exception) and is_rate_limit(result) and attempt < self.max_retries:
                    still.append(index)
                    rate_exc = result
                elif isinstance(result, Exception):
                    raise result
                else:
                    output[index] = result
                    self.cache[_cache_key(inputs[index])] = result
            if still:
                self._on_rate_limit(len(still))
                await asyncio.sleep(self._wait(attempt, rate_exc))
            else:
                self._on_success()
            pending = still
            attempt_base = attempt
        if pending:
            raise RuntimeError(f"{len(pending)} calls were still rate limited after {attempt_base + 1} attempts")
        return output

    async def call(self, func: Callable, /, *args, **kwargs):
        """Run a synchronous function on a worker thread under the same cap."""
        async with self.semaphore:
            self._enter()
            try:
                return await asyncio.to_thread(functools.partial(func, *args, **kwargs))
            finally:
                self._leave()

    async def _acquire(self) -> None:
        await self._take_tokens()
        async with self._gate:
            while self._in_flight >= self.limit:
                await self._gate.wait()
            self._enter()

    async def _release_async(self) -> None:
        async with self._gate:
            self._leave()
            self._gate.notify()

    async def _take_tokens(self) -> None:
        """Optional token bucket. ``tokens_per_minute`` of 0 leaves it open."""
        if self.tokens_per_minute <= 0:
            return
        loop = asyncio.get_running_loop()
        while True:
            now = loop.time()
            if self._token_stamp <= 0:
                self._token_stamp = now
            elapsed = now - self._token_stamp
            self._tokens = min(self.tokens_per_minute, self._tokens + elapsed * (self.tokens_per_minute / 60.0))
            self._token_stamp = now
            if self._tokens >= 1:
                self._tokens -= 1
                return
            await asyncio.sleep(max(0.01, (1 - self._tokens) * 60.0 / self.tokens_per_minute))

    def _on_success(self) -> None:
        self._success_streak += 1
        if self._success_streak >= self.limit and self.limit < self.max_concurrency:
            self.limit += 1
            self._success_streak = 0

    def _on_rate_limit(self, count: int = 1) -> None:
        self.rate_limit_retries += count
        self.limit = max(1, self.limit // 2)
        self._success_streak = 0

    def _wait(self, attempt: int, exc: BaseException | None) -> float:
        retry_after = _retry_after(exc)
        if retry_after is not None:
            return retry_after
        cap = min(self.max_backoff, self.backoff * (2**attempt))
        if cap <= 0:
            return 0.0
        return random.uniform(0.0, cap)

    async def _retry(self, factory: Callable):
        for attempt in range(self.max_retries + 1):
            try:
                return await factory()
            except Exception as exc:
                if attempt >= self.max_retries or not is_rate_limit(exc):
                    raise
                self._on_rate_limit()
                await asyncio.sleep(self._wait(attempt, exc))
        raise RuntimeError("retry loop exited without a result")

    def _enter(self) -> None:
        self._in_flight += 1
        if self._in_flight > self.max_in_flight:
            self.max_in_flight = self._in_flight

    def _leave(self) -> None:
        self._in_flight -= 1


def _cache_key(payload: Any) -> str:
    try:
        raw = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, default=repr)
    except TypeError:
        raw = repr(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    content = getattr(value, "content", None)
    if content is not None and not isinstance(value, (dict, list, tuple)):
        return {"role": getattr(value, "type", ""), "content": _jsonable(content)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _retry_after(exc: BaseException | None) -> float | None:
    if exc is None:
        return None
    for name in ("retry_after", "retry_after_seconds"):
        value = getattr(exc, name, None)
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = None
    if hasattr(headers, "get"):
        raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def is_rate_limit(exc: BaseException) -> bool:
    """True for HTTP 429 and provider exceptions that say the call was rate limited."""
    name = type(exc).__name__.casefold().replace("_", "")
    if "ratelimit" in name or "toomanyrequests" in name:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    text = str(exc).casefold()
    return "rate limit" in text or "too many requests" in text or "429" in text


def usage_from_message(raw: Any) -> dict[str, int]:
    """Read token counts from a LangChain message, or zeros when the provider sent none."""
    meta = getattr(raw, "usage_metadata", None) or {}
    if not meta and raw is not None:
        response = getattr(raw, "response_metadata", None) or {}
        token_usage = response.get("token_usage") or response.get("usage") or {}
        meta = {
            "input_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0,
            "output_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0,
            "total_tokens": token_usage.get("total_tokens") or 0,
        }
    input_tokens = int(meta.get("input_tokens") or 0)
    output_tokens = int(meta.get("output_tokens") or 0)
    total_tokens = int(meta.get("total_tokens") or 0)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def structured_runnable(model, schema):
    """Prefer ``include_raw`` so token usage can be recorded. Fall back if unsupported."""
    try:
        return model.with_structured_output(schema, include_raw=True)
    except TypeError:
        return model.with_structured_output(schema)


def parse_structured(result, schema) -> tuple[dict, dict]:
    """Normalize ``ainvoke`` / ``abatch`` output to ``(payload, usage)``."""
    if result is None:
        raise ValueError(f"{schema.__name__} returned no structured output.")
    raw = None
    if isinstance(result, dict) and "parsed" in result:
        raw = result.get("raw")
        parsed = result.get("parsed")
        if parsed is None:
            raise ValueError(f"{schema.__name__} output could not be parsed: {result.get('parsing_error')}")
    else:
        parsed = result
    if hasattr(parsed, "model_dump"):
        payload = parsed.model_dump()
    elif isinstance(parsed, dict):
        payload = schema.model_validate(parsed).model_dump()
    else:
        raise ValueError(f"{schema.__name__} returned {type(parsed).__name__}.")
    return payload, usage_from_message(raw)


def run_sync(coro):
    """Run ``coro`` to completion, including inside a running event loop.

    With no running loop this is :func:`asyncio.run`. With a running loop (a
    Jupyter cell, or a test that already started one) the coroutine runs on a
    worker thread so the caller's loop is left alone.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: dict[str, Any] = {}

    def _worker() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            outcome["error"] = exc

    thread = threading.Thread(target=_worker, name="agt-sync")
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
