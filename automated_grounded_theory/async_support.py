"""Shared concurrency limit, retry backoff, and a notebook-safe sync runner.

LLM calls go through :meth:`CallLimiter.ainvoke` or :meth:`CallLimiter.abatch`.
Both take slots from one adaptive gate. Embeddings use a separate semaphore
of the same size so a coding task can embed and then call the model without
taking the model slot twice.

:func:`run_sync` drives a coroutine from ordinary Python and from a running
loop (Jupyter). Inside a running loop the coroutine is executed on a worker
thread with its own loop, so the notebook loop is not re-entered.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import json
import random
import threading
import time
from typing import Any, Callable, Mapping, Sequence

class CallLimiter:
    """Cap in-flight work, adapt on 429, and retry transient failures with full-jitter backoff.

    Each call is cancelled when it runs longer than ``request_timeout`` (120
    seconds unless you set another value; 0 waits with no deadline). Timeouts
    and connection errors are retried the same way as rate limits. Only a rate
    limit lowers the AIMD ceiling.

    ``ainvoke`` takes one slot from an AIMD limit (the configured concurrency
    is the ceiling; a rate limit halves it, and a streak of successes adds one
    back). ``abatch`` runs one ``ainvoke`` per input under that same gate, so
    two batches cannot each open a private ceiling. ``max_in_flight`` is the
    peak number of model calls that held a slot.
    A response cache keyed by the payload replays an identical prompt without
    calling the model again.

    ``tokens_per_minute`` reserves estimated input tokens plus reserved output
    before the call, then reconciles to the provider's actual usage. It is not
    a request counter. ``requests_per_minute`` consumes one unit per attempt.
    A reservation larger than the bucket's capacity is rejected: the bucket
    never refills past that capacity, so waiting for it would not end.
    ``allow_oversized`` is the explicit exception. It waits until the bucket
    is full, then drains it and lets the call proceed.

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
        requests_per_minute: float = 0,
        allow_oversized: bool = False,
        reserved_output_tokens: int = 0,
        cache: dict | None = None,
        request_timeout: float = 120.0,
    ) -> None:
        self.concurrency = max(1, int(concurrency))
        self.max_concurrency = self.concurrency
        self.limit = self.concurrency
        self.max_retries = max(0, int(max_retries))
        self.backoff = max(0.0, float(backoff))
        self.max_backoff = max(self.backoff, float(max_backoff))
        self.tokens_per_minute = max(0.0, float(tokens_per_minute))
        self.requests_per_minute = max(0.0, float(requests_per_minute))
        self.allow_oversized = bool(allow_oversized)
        self.reserved_output_tokens = max(0, int(reserved_output_tokens))
        self.request_timeout = float(request_timeout)
        self._tokens = self.tokens_per_minute
        self._token_stamp: float | None = None
        self._requests = self.requests_per_minute
        self._request_stamp: float | None = None
        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.embed_semaphore = asyncio.Semaphore(self.concurrency)
        self._gate = asyncio.Condition()
        self._budget_lock = asyncio.Lock()
        self.max_in_flight = 0
        self._in_flight = 0
        self.rate_limit_retries = 0
        self.timeout_retries = 0
        self._success_streak = 0
        self.cache = cache if cache is not None else {}
        self.cache_hits = 0
        self.cache_context: dict = {}
        self.schema_repairs = 0
        self.request_latencies: list[float] = []
        self.queue_waits: list[float] = []
        self.timeline: list[dict] = []
        self._timeline_origin: float | None = None
        self._clock_override: Callable[[], float] | None = None
        self._sleep_override: Callable[[float], Any] | None = None
        self._accounting: contextvars.ContextVar[list | None] = contextvars.ContextVar(
            f"agt_acct_{id(self)}",
            default=None,
        )

    def push_accounting(self, bucket: list) -> contextvars.Token:
        """Attribute model calls on this task to ``bucket`` until :meth:`pop_accounting`."""
        return self._accounting.set(bucket)

    def pop_accounting(self, token: contextvars.Token) -> None:
        self._accounting.reset(token)

    async def ainvoke(
        self,
        runnable,
        payload,
        *,
        timeout: float | None = None,
        reserved_output: int | None = None,
    ):
        """``await runnable.ainvoke(payload)`` under the AIMD cap, with rate-limit retries.

        ``timeout`` overrides ``request_timeout`` for this call. ``None`` keeps
        the limiter default. ``0`` waits with no deadline. ``reserved_output``
        is added to the estimated input when the token bucket reserves a call.
        """
        key = cache_key(payload, **(self.cache_context or {}))
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        estimate = self._estimate(payload, reserved_output)
        result = await self._retry(lambda: self._attempt(runnable, payload, timeout=timeout, estimate=estimate))
        self.cache[key] = result
        return result

    async def abatch(
        self,
        runnable,
        inputs: Sequence[Any],
        *,
        timeout: float | None = None,
        reserved_output: int | None = None,
    ) -> list[Any]:
        """One :meth:`ainvoke` per input, all on the shared AIMD gate.

        Results stay aligned with ``inputs``. A private semaphore is not
        created, so concurrent batches share ``max_in_flight`` and the ceiling.
        Cached rows are not sent again.
        """
        if not inputs:
            return []

        async def one(payload):
            try:
                return await self.ainvoke(
                    runnable,
                    payload,
                    timeout=timeout,
                    reserved_output=reserved_output,
                )
            except Exception as exc:
                return exc

        results = list(await asyncio.gather(*(one(payload) for payload in inputs)))
        for result in results:
            if isinstance(result, Exception):
                raise result
        return results

    async def call(self, func: Callable, /, *args, **kwargs):
        """Run a synchronous function on a worker thread under the same cap."""
        async with self.semaphore:
            self._enter()
            try:
                return await asyncio.to_thread(functools.partial(func, *args, **kwargs))
            finally:
                self._leave()

    def _estimate(self, payload: Any, reserved_output: int | None) -> int:
        if self.tokens_per_minute <= 0:
            return 0
        reserve = self.reserved_output_tokens if reserved_output is None else max(0, int(reserved_output))
        return estimate_input_tokens(payload) + reserve

    async def _attempt(self, runnable, payload, *, timeout: float | None, estimate: int):
        """One provider attempt: reserve budgets, hold one shared slot, reconcile."""
        record = {"queue_wait": 0.0, "latency": None, "reserved": 0.0, "usage": None}
        held = False
        try:
            await self._reserve_budgets(record, estimate)
            await self._acquire_slot(record)
            held = True
            result = await self._invoke(runnable, payload, timeout=timeout, record=record)
            record["usage"] = usage_from_result(result)
            self._on_success()
            return result
        finally:
            await self._reconcile(record)
            self._remember(record)
            if held:
                await self._release_async()

    async def _reserve_budgets(self, record: dict, estimate: int) -> None:
        started = self._clock()
        await self._take_amount(1, kind="requests")
        if self.tokens_per_minute > 0 and estimate > 0:
            taken = await self._take_amount(estimate, kind="tokens")
            record["reserved"] = float(taken)
        record["queue_wait"] = max(0.0, self._clock() - started)

    async def _acquire_slot(self, record: dict) -> None:
        started = self._clock()
        async with self._gate:
            while self._in_flight >= self.limit:
                await self._gate.wait()
            self._enter()
        record["queue_wait"] = float(record.get("queue_wait") or 0) + max(0.0, self._clock() - started)

    async def _release_async(self) -> None:
        async with self._gate:
            self._leave()
            self._gate.notify()

    async def _take_amount(self, amount: float, *, kind: str) -> float:
        """Wait until ``kind`` (``tokens`` or ``requests``) can spare ``amount``.

        Returns the units actually removed from the bucket. A request larger
        than the capacity cannot be satisfied by waiting: refill stops at the
        capacity. That raises :class:`OversizedReservation` unless
        ``allow_oversized`` is set, in which case this waits for a full bucket
        and then drains it.
        """
        cap = self.tokens_per_minute if kind == "tokens" else self.requests_per_minute
        amount = float(amount)
        if cap <= 0 or amount <= 0:
            return 0.0
        if amount > cap + 1e-9:
            if not self.allow_oversized:
                unit = "tokens" if kind == "tokens" else "requests"
                setting = "tokens_per_minute" if kind == "tokens" else "requests_per_minute"
                raise OversizedReservation(
                    f"Reservation of {amount:g} {unit} exceeds the {setting} capacity of {cap:g}. "
                    "The bucket refills only up to that capacity, so waiting would never finish "
                    "and the provider timeout would never start. "
                    f"Raise {setting} to at least {amount:g}, send a smaller request, "
                    "or set allow_oversized=True to wait until the bucket is full and then drain it."
                )
            return await self._drain_full_bucket(kind)
        while True:
            async with self._budget_lock:
                self._refill(self._clock(), kind)
                level = self._tokens if kind == "tokens" else self._requests
                if level + 1e-9 >= amount:
                    if kind == "tokens":
                        self._tokens = level - amount
                    else:
                        self._requests = level - amount
                    return amount
                delay = (amount - level) * 60.0 / cap
            await self._sleep(max(0.0, delay))

    async def _drain_full_bucket(self, kind: str) -> float:
        """Wait until ``kind`` is full, then take the whole bucket."""
        cap = self.tokens_per_minute if kind == "tokens" else self.requests_per_minute
        while True:
            async with self._budget_lock:
                self._refill(self._clock(), kind)
                level = self._tokens if kind == "tokens" else self._requests
                if level + 1e-6 >= cap:
                    if kind == "tokens":
                        self._tokens = 0.0
                    else:
                        self._requests = 0.0
                    return float(cap)
                delay = (cap - level) * 60.0 / cap
            await self._sleep(max(0.0, delay))

    def _refill(self, now: float, kind: str) -> None:
        cap = self.tokens_per_minute if kind == "tokens" else self.requests_per_minute
        if cap <= 0:
            return
        if kind == "tokens":
            stamp = self._token_stamp
            level = self._tokens
        else:
            stamp = self._request_stamp
            level = self._requests
        if stamp is None:
            if kind == "tokens":
                self._token_stamp = now
            else:
                self._request_stamp = now
            return
        elapsed = max(0.0, now - float(stamp))
        level = min(cap, level + elapsed * (cap / 60.0))
        if kind == "tokens":
            self._tokens = level
            self._token_stamp = now
        else:
            self._requests = level
            self._request_stamp = now

    async def _reconcile(self, record: dict) -> None:
        """Replace the reservation with actual input + output when the provider reported it."""
        if self.tokens_per_minute <= 0:
            return
        reserved = float(record.get("reserved") or 0)
        if reserved <= 0:
            return
        usage = record.get("usage") or {}
        actual = int(usage.get("total_tokens") or 0)
        if actual <= 0:
            actual = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        if actual <= 0:
            return
        async with self._budget_lock:
            self._refill(self._clock(), "tokens")
            self._tokens += reserved - actual
            if self._tokens > self.tokens_per_minute:
                self._tokens = self.tokens_per_minute

    def _clock(self) -> float:
        if self._clock_override is not None:
            return float(self._clock_override())
        return asyncio.get_running_loop().time()

    async def _sleep(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if self._sleep_override is not None:
            await self._sleep_override(seconds)
            return
        # Sleep even for zero so a retry yields the slot to another waiter.
        await asyncio.sleep(seconds)

    def _remember(self, record: dict) -> None:
        latency = record.get("latency")
        queue = float(record.get("queue_wait") or 0)
        if latency is not None:
            self.request_latencies.append(float(latency))
        if queue > 0:
            self.queue_waits.append(queue)
        bucket = self._accounting.get()
        if bucket is not None:
            bucket.append(
                {
                    "latency": None if latency is None else float(latency),
                    "queue_wait": queue,
                }
            )

    def _add_queue_wait(self, seconds: float) -> None:
        seconds = float(seconds or 0)
        if seconds <= 0:
            return
        self.queue_waits.append(seconds)
        bucket = self._accounting.get()
        if bucket is not None:
            bucket.append({"latency": None, "queue_wait": seconds})

    def _on_success(self) -> None:
        self._success_streak += 1
        if self._success_streak >= self.limit and self.limit < self.max_concurrency:
            self.limit += 1
            self._success_streak = 0

    def _on_rate_limit(self, count: int = 1) -> None:
        self.rate_limit_retries += count
        self.limit = max(1, self.limit // 2)
        self._success_streak = 0

    def _note_transient(self, exc: BaseException | None, count: int = 1) -> None:
        """Count a retry. Only a rate limit narrows the AIMD ceiling."""
        if exc is not None and is_rate_limit(exc):
            self._on_rate_limit(count)
            return
        self.timeout_retries += count
        self._success_streak = 0

    def _mark(self, kind: str, **extra: Any) -> None:
        now = time.perf_counter()
        if self._timeline_origin is None:
            self._timeline_origin = now
        self.timeline.append(
            {
                "t": round(now - self._timeline_origin, 4),
                "kind": kind,
                "in_flight": self._in_flight,
                **extra,
            }
        )

    def activity_report(self) -> dict:
        """In-flight events and the gaps where nothing was calling the model."""
        gaps = []
        idle_at = None
        for event in self.timeline:
            kind = event.get("kind")
            if kind == "idle":
                idle_at = float(event.get("t") or 0)
            elif kind == "call_start" and idle_at is not None:
                seconds = round(float(event.get("t") or 0) - idle_at, 4)
                if seconds >= 0.05:
                    gaps.append({"start": idle_at, "end": event.get("t"), "seconds": seconds})
                idle_at = None
            elif kind == "call_start":
                idle_at = None
        idle_seconds = round(sum(float(item["seconds"]) for item in gaps), 4)
        return {
            "events": list(self.timeline),
            "idle_gaps": gaps,
            "idle_seconds": idle_seconds,
            "max_idle_seconds": round(max((float(item["seconds"]) for item in gaps), default=0.0), 4),
            "max_in_flight": self.max_in_flight,
            "request_latency_seconds": [round(item, 4) for item in self.request_latencies],
            "queue_wait_seconds": round(sum(self.queue_waits), 4),
        }

    async def _invoke(self, runnable, payload, *, timeout: float | None = None, record: dict | None = None):
        """One model call, cut off at ``timeout`` seconds, or ``request_timeout``.

        A timeout of 0 or less waits for the provider. The default is 120.
        Latency on ``record`` is the provider call only, not queue or backoff.
        """
        self._mark("call_start")
        started = time.perf_counter()
        limit = self.request_timeout if timeout is None else float(timeout)
        try:
            call = runnable.ainvoke(payload)
            if limit and limit > 0:
                return await asyncio.wait_for(call, limit)
            return await call
        finally:
            if record is not None:
                record["latency"] = time.perf_counter() - started
            self._mark("call_end")

    def _wait(self, attempt: int, exc: BaseException | None) -> float:
        """Backoff for one retry. ``Retry-After`` cannot exceed ``max_backoff``.

        An uncapped provider delay held the only slot for many minutes while
        no model call was in flight.
        """
        cap = min(self.max_backoff, self.backoff * (2**attempt)) if self.backoff else self.max_backoff
        retry_after = _retry_after(exc)
        if retry_after is not None:
            return min(max(0.0, retry_after), max(0.0, self.max_backoff))
        if cap <= 0:
            return 0.0
        return random.uniform(0.0, cap)

    async def _retry(self, factory: Callable):
        for attempt in range(self.max_retries + 1):
            try:
                return await factory()
            except Exception as exc:
                if attempt >= self.max_retries or not is_transient(exc):
                    raise
                self._note_transient(exc)
                wait = self._wait(attempt, exc)
                # The attempt already released its slot. Sleep with the gate
                # free so another passage or sweep can use it, then retry.
                self._mark("idle_sleep", seconds=round(wait, 4))
                self._add_queue_wait(wait)
                await self._sleep(wait)
        raise RuntimeError("retry loop exited without a result")

    def _enter(self) -> None:
        self._in_flight += 1
        if self._in_flight > self.max_in_flight:
            self.max_in_flight = self._in_flight

    def _leave(self) -> None:
        self._in_flight -= 1
        if self._in_flight <= 0:
            self._in_flight = 0
            self._mark("idle")


def cache_key(payload: Any, **context: Any) -> str:
    """Hash the request plus snapshot, definition, evidence, and policy versions.

    A frequency change does not alter the key. A new definition or evidence
    revision does. With no version fields this matches the older payload hash.
    """
    clean = {key: value for key, value in context.items() if value not in (None, "")}
    try:
        body = _jsonable(payload) if not clean else {"payload": _jsonable(payload), "context": clean}
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True, default=repr)
    except TypeError:
        raw = repr((payload, clean))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_key(payload: Any) -> str:
    return cache_key(payload)


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


class OversizedReservation(ValueError):
    """A rate-limit reservation is larger than the bucket can ever hold."""


def is_timeout(exc: BaseException) -> bool:
    """True for deadline and provider read-timeout errors."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    name = type(exc).__name__.casefold()
    if "timeout" in name:
        return True
    text = str(exc).casefold()
    return "timed out" in text or "timeout" in text or "deadline exceeded" in text


def is_connection_error(exc: BaseException) -> bool:
    """True for dropped connections. Not every OSError."""
    if isinstance(exc, (ConnectionError, ConnectionResetError, BrokenPipeError)):
        return True
    name = type(exc).__name__.casefold()
    if "connect" in name or "connection" in name:
        return True
    text = str(exc).casefold()
    return any(
        phrase in text
        for phrase in (
            "connection reset",
            "connection error",
            "connection aborted",
            "connect error",
            "server disconnected",
            "network error",
            "name or service not known",
        )
    )


def is_transient(exc: BaseException) -> bool:
    """True when the call should be retried: rate limit, timeout, or connection loss.

    Walks ``__cause__`` so a wrapped CodingError still counts.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if is_rate_limit(current) or is_timeout(current) or is_connection_error(current):
            return True
        current = current.__cause__
    return False


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


def usage_from_result(result: Any) -> dict[str, int]:
    """Token counts from a structured result, including a response that failed to parse."""
    raw = result
    if isinstance(result, dict) and "parsed" in result:
        raw = result.get("raw")
    return usage_from_message(raw)


def combine_usage(*parts: Mapping[str, Any] | None) -> dict[str, Any]:
    """Sum token usage. A schema repair keeps every attempt, not only the success."""
    keys = ("input_tokens", "output_tokens", "total_tokens")
    total = {key: 0 for key in keys}
    attempts: list[dict[str, int]] = []
    for part in parts:
        if not part:
            continue
        nested = part.get("attempt_usage")
        if isinstance(nested, list) and nested:
            for row in nested:
                attempts.append({key: int((row or {}).get(key) or 0) for key in keys})
        else:
            row = {key: int(part.get(key) or 0) for key in keys}
            if row["total_tokens"] == 0:
                row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
            if any(row.values()):
                attempts.append(row)
        for key in keys:
            total[key] += int(part.get(key) or 0)
    if total["total_tokens"] == 0:
        total["total_tokens"] = total["input_tokens"] + total["output_tokens"]
    if len(attempts) > 1:
        total["attempt_usage"] = attempts
    return total


def estimate_input_tokens(payload: Any) -> int:
    """Estimated input tokens for a limiter reservation.

    Chat messages use the coding tokenizer estimate. Anything else is the
    character estimate of its JSON form.
    """
    from automated_grounded_theory.batching import count_message_tokens, estimate_tokens

    if payload is None:
        return 0
    if isinstance(payload, str):
        return estimate_tokens(payload)
    if isinstance(payload, (list, tuple)) and payload and _is_message(payload[0]):
        return count_message_tokens(payload)
    try:
        raw = json.dumps(_jsonable(payload), ensure_ascii=False)
    except TypeError:
        raw = repr(payload)
    return estimate_tokens(raw)


def _is_message(value: Any) -> bool:
    return hasattr(value, "content") and not isinstance(value, (str, bytes, dict, list, tuple))


def structured_runnable(model, schema):
    """Prefer strict JSON schema, then looser structured output, always with raw usage when possible."""
    attempts = (
        {"method": "json_schema", "strict": True, "include_raw": True},
        {"method": "json_schema", "include_raw": True},
        {"method": "function_calling", "include_raw": True},
        {"include_raw": True},
        {},
    )
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return model.with_structured_output(schema, **kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return model.with_structured_output(schema)


def _raw_text(raw: Any) -> str:
    if raw is None:
        return ""
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    if isinstance(raw, str):
        return raw
    return ""


def _model_from_text(text: str, schema):
    decoder = json.JSONDecoder()
    start = 0
    while True:
        index = text.find("{", start)
        if index < 0:
            raise ValueError(f"{schema.__name__} output could not be parsed.")
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            start = index + 1
            continue
        if isinstance(obj, dict):
            return schema.model_validate(obj)
        start = index + 1


def parse_structured(result, schema) -> tuple[dict, dict]:
    """Normalize ``ainvoke`` / ``abatch`` output to ``(payload, usage)``.

    When the provider returns ``parsed=None``, the raw message is validated
    against ``schema`` once before this raises.
    """
    if result is None:
        raise ValueError(f"{schema.__name__} returned no structured output.")
    raw = None
    if isinstance(result, dict) and "parsed" in result:
        raw = result.get("raw")
        parsed = result.get("parsed")
        if parsed is None:
            try:
                parsed = _model_from_text(_raw_text(raw), schema)
            except ValueError as exc:
                detail = result.get("parsing_error") or exc
                raise ValueError(f"{schema.__name__} output could not be parsed: {detail}") from exc
    else:
        parsed = result
    if hasattr(parsed, "model_dump"):
        payload = parsed.model_dump()
    elif isinstance(parsed, dict):
        payload = schema.model_validate(parsed).model_dump()
    else:
        raise ValueError(f"{schema.__name__} returned {type(parsed).__name__}.")
    return payload, usage_from_message(raw)


def sweep_deadline(request_timeout: float, sweep_timeout: float) -> float:
    """How long a sweep call may run.

    Sweeps read the whole codebook chunk, so the deadline is longer than the
    coding deadline (300 seconds unless ``sweep_timeout`` is set higher).
    Zero on either side means no deadline.
    """
    coding = float(request_timeout)
    sweep = float(sweep_timeout)
    if coding <= 0 or sweep <= 0:
        return 0.0
    return max(coding, sweep)


async def ainvoke_structured(
    model,
    schema,
    messages: Sequence,
    limiter,
    *,
    timeout: float | None = None,
    reserved_output: int | None = None,
    token_budget: int | None = None,
    tokenizer=None,
) -> tuple[tuple[dict, dict], list]:
    """Invoke ``schema`` once, then repair a single malformed output.

    The follow-up user message starts with ``REPAIR THIS OUTPUT``. The second
    failure raises the first parse error. Returns ``((payload, usage), messages)``.
    Usage on a repaired call includes the failed response as well as the
    successful one.

    When ``token_budget`` is set, the repair request is measured again,
    including the schema, and the repair note is shortened until the input
    plus ``reserved_output`` fits that budget.
    """
    from langchain_core.messages import HumanMessage

    runnable = structured_runnable(model, schema)
    extra: dict[str, Any] = {}
    if timeout is not None:
        extra["timeout"] = timeout
    if reserved_output is not None:
        extra["reserved_output"] = reserved_output
    result = await limiter.ainvoke(runnable, list(messages), **extra)
    try:
        return parse_structured(result, schema), list(messages)
    except ValueError as first:
        failed_usage = usage_from_result(result)
        repair = list(messages) + [
            HumanMessage(
                content=(
                    "REPAIR THIS OUTPUT\n"
                    f"The previous output was not valid {schema.__name__}: {first}\n"
                    "Return only the structured object."
                )
            )
        ]
        if token_budget is not None:
            from automated_grounded_theory.batching import shrink_repair_messages

            reserved = 0 if reserved_output is None else max(0, int(reserved_output))
            ceiling = max(1, int(token_budget) - reserved)
            repair = shrink_repair_messages(repair, schema, ceiling, tokenizer)
        try:
            limiter.schema_repairs = int(getattr(limiter, "schema_repairs", 0) or 0) + 1
            repaired = await limiter.ainvoke(runnable, repair, **extra)
            payload, usage = parse_structured(repaired, schema)
            return (payload, combine_usage(failed_usage, usage)), repair
        except ValueError:
            raise first from None


async def _close_async_http_clients() -> None:
    """Close httpx and aiohttp clients bound to this loop before the loop shuts down.

    A client left open is closed from ``__del__`` after ``asyncio.run`` finishes,
    which raises ``RuntimeError: Event loop is closed``.
    """
    import gc

    running = asyncio.get_running_loop()
    closers = []
    seen: set[int] = set()
    for obj in gc.get_objects():
        marker = id(obj)
        if marker in seen:
            continue
        try:
            cls = type(obj)
            module = getattr(cls, "__module__", "")
            name = getattr(cls, "__name__", "")
        except Exception:
            continue
        if not isinstance(module, str) or not isinstance(name, str):
            continue
        root = module.split(".", 1)[0]
        if root not in {"httpx", "aiohttp"} or name not in {"AsyncClient", "ClientSession"}:
            continue
        if getattr(obj, "is_closed", False):
            seen.add(marker)
            continue
        bound = getattr(obj, "_loop", None)
        if bound is not None and bound is not running:
            continue
        close = getattr(obj, "aclose", None)
        if not callable(close):
            continue
        seen.add(marker)
        try:
            closers.append(close())
        except Exception:
            continue
    if closers:
        await asyncio.gather(*closers, return_exceptions=True)
    import gc

    gc.collect()
    extra = []
    for obj in gc.get_objects():
        try:
            cls = type(obj)
            module = getattr(cls, "__module__", "")
            name = getattr(cls, "__name__", "")
        except Exception:
            continue
        if not isinstance(module, str) or not isinstance(name, str):
            continue
        if "httpx" not in module and "aiohttp" not in module and "httpcore" not in module:
            continue
        if "Client" not in name and "Session" not in name:
            continue
        close = getattr(obj, "aclose", None)
        if not callable(close) or getattr(obj, "is_closed", False):
            continue
        try:
            extra.append(close())
        except Exception:
            continue
    if extra:
        await asyncio.gather(*extra, return_exceptions=True)


_LOOP_LOCK = threading.Lock()
_LOOP: asyncio.AbstractEventLoop | None = None


def _package_loop() -> asyncio.AbstractEventLoop:
    """One loop for every sync entry point, so a shared client keeps its loop."""
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is not None and not _LOOP.is_closed():
            return _LOOP
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _spin() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_spin, name="agt-loop", daemon=True)
        thread.start()
        ready.wait()
        _LOOP = loop
        return loop


def _loop_is_running() -> bool:
    """True when this thread already has a running loop.

    The check is finished before any new loop starts, so a later
    ``Event loop is closed`` is not chained under ``no running event loop``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def run_sync(coro):
    """Run ``coro`` to completion, including inside a running event loop.

    Every sync call shares one background loop. A chat model created during
    ``open_code`` stays bound to that loop, so ``build_themes`` can reuse the
    same instance. Shared httpx clients are not closed here: closing them
    made the next call raise ``Event loop is closed``.
    """
    loop = _package_loop()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        raise RuntimeError("run_sync cannot wait on the package event loop from inside it.")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result()
    except BaseException as exc:
        raise exc from None
