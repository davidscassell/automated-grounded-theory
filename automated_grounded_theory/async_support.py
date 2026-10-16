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
import concurrent.futures
import contextvars
import functools
import hashlib
import json
import logging
import random
import threading
import time
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger("automated_grounded_theory")

class ProviderAllowance:
    """Shared estimated-token ceiling and separate per-stage call ceilings.

    Coding, counterexample, memo, storyline, decision review, schema repair,
    and a retry all acquire from this object before the provider is called.
    Repairs and retries use the stage that originated the request. A request
    that does not fit is not sent. ``max_calls`` or a stage ceiling of ``None``
    leaves that side open. Zero is a closed ceiling. ``max_tokens`` is one
    pool across stages. ``snapshot`` and ``restore`` are the checkpoint copy
    of the counters, so a new runtime can continue the same allowance.
    """

    def __init__(
        self,
        *,
        max_calls: int | None = None,
        max_tokens: int | None = None,
        calls_used: int = 0,
        tokens_estimated: int = 0,
        stages: Mapping[str, int | None] | None = None,
        token_reserve: int = 0,
    ) -> None:
        self.max_calls = None if max_calls is None else max(0, int(max_calls))
        self.max_tokens = None if max_tokens is None else max(0, int(max_tokens))
        self.calls_used = max(0, int(calls_used))
        self.tokens_estimated = max(0, int(tokens_estimated))
        self.token_reserve = max(0, int(token_reserve))
        self.refusals: list[dict[str, Any]] = []
        self.stage_ceilings: dict[str, int | None] = {}
        self.stage_used: dict[str, int] = {}
        self.observed_raw = 0
        self.observed_input = 0
        for name, ceiling in (stages or {}).items():
            self.stage_ceilings[str(name)] = None if ceiling is None else max(0, int(ceiling))
            self.stage_used[str(name)] = 0

    def remaining_calls(self, stage: str | None = None) -> int | None:
        if stage and self.stage_ceilings:
            ceiling = self.stage_ceilings.get(stage)
            if ceiling is None:
                return None
            return max(0, ceiling - self.stage_used.get(stage, 0))
        if self.max_calls is None:
            return None
        return max(0, self.max_calls - self.calls_used)

    def remaining_tokens(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.tokens_estimated)

    def snapshot(self) -> dict[str, Any]:
        """Counters that must survive a human checkpoint and a new runtime."""
        return {
            "calls_used": self.calls_used,
            "tokens_estimated": self.tokens_estimated,
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
            "stages": {
                name: {
                    "calls_used": self.stage_used.get(name, 0),
                    "max_calls": self.stage_ceilings.get(name),
                }
                for name in set(self.stage_ceilings) | set(self.stage_used)
            },
            "observed_raw": self.observed_raw,
            "observed_input": self.observed_input,
            "token_estimate_ratio": self.estimate_ratio(),
            "tokens_estimated_raw": self.tokens_estimated,
            "tokens_estimated_compared": self._guarded(self.tokens_estimated),
            "token_reserve": self.token_reserve,
            "ceiling_comparison": "guarded",
            "ceiling_note": (
                "tokens_estimated is the raw heuristic accumulator. "
                "The ceiling comparison uses tokens_estimated_compared, the guarded reading of that raw total. "
                "A raw total can sit above the ceiling while the guarded reading still fits."
            ),
        }

    def restore(self, snapshot: Mapping[str, Any] | None) -> None:
        """Replace consumed counters from a checkpoint. Ceilings stay on this runtime."""
        if not snapshot:
            return
        self.calls_used = max(0, int(snapshot.get("calls_used") or 0))
        self.tokens_estimated = max(0, int(snapshot.get("tokens_estimated") or 0))
        self.observed_raw = max(0, int(snapshot.get("observed_raw") or 0))
        self.observed_input = max(0, int(snapshot.get("observed_input") or 0))
        for name, row in (snapshot.get("stages") or {}).items():
            if isinstance(row, Mapping):
                used = row.get("calls_used")
            else:
                used = row
            self.stage_used[str(name)] = max(0, int(used or 0))

    def preview(
        self,
        estimate: int,
        *,
        stage: str | None = None,
        kind: str = "preview",
        apply_reserve: bool = True,
    ) -> dict[str, Any]:
        return self._decide(estimate, commit=False, kind=kind, stage=stage, apply_reserve=apply_reserve)

    def would_refuse(
        self,
        estimate: int,
        *,
        stage: str | None = None,
        kind: str = "preview",
        apply_reserve: bool = True,
    ) -> bool:
        return self.preview(estimate, stage=stage, kind=kind, apply_reserve=apply_reserve)["status"] != "acquired"

    def try_acquire(self, estimate: int, *, kind: str, stage: str | None = None) -> dict[str, Any]:
        return self._decide(estimate, commit=True, kind=kind, stage=stage, apply_reserve=kind != "storyline")

    def note_estimate_observation(self, raw: int, usage: Mapping[str, Any] | None) -> None:
        """Record one call whose raw estimate and provider input tokens are both known."""
        reported = int((usage or {}).get("input_tokens") or 0)
        raw = max(0, int(raw or 0))
        if reported <= 0 or raw <= 0:
            return
        self.observed_raw += raw
        self.observed_input += reported

    def estimate_ratio(self) -> float | None:
        """Raw heuristic divided by provider input tokens. Null until both are known."""
        if self.observed_input <= 0 or self.observed_raw <= 0:
            return None
        return round(self.observed_raw / self.observed_input, 4)

    def _guarded(self, raw: int) -> int:
        from automated_grounded_theory.batching import guarded_allowance_estimate

        return guarded_allowance_estimate(
            raw, raw_total=self.observed_raw, provider_input=self.observed_input
        )

    def _bucket(self, kind: str, stage: str | None) -> str:
        if stage:
            return stage
        if kind in {"decision_review"}:
            return "review"
        if kind in {"theoretical_memo", "storyline"}:
            return "selective"
        if kind in {"focused_recode", "focused_label_retry"}:
            return "focused"
        return "coding"

    def _compared_ceiling(self, kind: str, *, apply_reserve: bool) -> int | None:
        if self.max_tokens is None:
            return None
        if not apply_reserve or kind == "storyline" or self.token_reserve <= 0:
            return self.max_tokens
        return max(0, self.max_tokens - self.token_reserve)

    def _decide(
        self,
        estimate: int,
        *,
        commit: bool,
        kind: str,
        stage: str | None = None,
        apply_reserve: bool = True,
    ) -> dict[str, Any]:
        raw_estimate = max(0, int(estimate))
        estimate = self._guarded(raw_estimate)
        spent = self._guarded(self.tokens_estimated)
        compared_ceiling = self._compared_ceiling(kind, apply_reserve=apply_reserve)
        bucket = self._bucket(kind, stage)
        stage_limited = bool(self.stage_ceilings)
        ceiling = self.stage_ceilings.get(bucket) if stage_limited else None
        used = self.stage_used.get(bucket, 0) if stage_limited else self.calls_used
        call_max = ceiling if stage_limited else self.max_calls
        if stage_limited and bucket not in self.stage_ceilings and self.max_calls is not None:
            call_max = self.max_calls
            used = self.calls_used
        if call_max is not None and used >= call_max:
            row = {
                "status": "refused",
                "reason": "call_allowance",
                "kind": kind,
                "stage": bucket,
                "estimated_input_tokens": raw_estimate,
                "guarded_estimate": estimate,
                "calls_used": used,
                "tokens_estimated": self.tokens_estimated,
                "max_calls": call_max,
                "max_tokens": self.max_tokens,
            }
        elif compared_ceiling is not None and spent + estimate > compared_ceiling:
            row = {
                "status": "refused",
                "reason": "token_allowance",
                "kind": kind,
                "stage": bucket,
                "estimated_input_tokens": raw_estimate,
                "guarded_estimate": estimate,
                "calls_used": used,
                "tokens_estimated": self.tokens_estimated,
                "max_calls": call_max,
                "max_tokens": self.max_tokens,
            }
        else:
            if commit:
                self.calls_used += 1
                self.tokens_estimated += raw_estimate
                if stage_limited:
                    self.stage_used[bucket] = self.stage_used.get(bucket, 0) + 1
                    used = self.stage_used[bucket]
                else:
                    used = self.calls_used
            row = {
                "status": "acquired",
                "reason": "",
                "kind": kind,
                "stage": bucket,
                "estimated_input_tokens": raw_estimate,
                "guarded_estimate": estimate,
                "calls_used": used if commit else used,
                "tokens_estimated": self.tokens_estimated,
                "max_calls": call_max,
                "max_tokens": self.max_tokens,
            }
        if commit and row["status"] != "acquired":
            self.refusals.append(dict(row))
        return row


class AllowanceExhausted(RuntimeError):
    """A provider request did not fit the shared call or token allowance."""

    def __init__(self, decision: Mapping[str, Any]):
        self.decision = dict(decision)
        self.reason = str(decision.get("reason") or "call_allowance")
        super().__init__(f"Provider allowance refused this request ({self.reason}).")


class CallLimiter:
    """Cap in-flight work, adapt on 429, and retry transient failures with full-jitter backoff.

    Each attempt is cancelled when it runs longer than ``request_timeout`` (120
    seconds unless you set another value; 0 waits with no per-attempt deadline).
    The wait returns when that clock expires even if the provider swallows
    cancellation. The task is cancelled, and ``TimeoutError`` is raised without
    waiting for the cancelled task to finish.
    ``wall_timeout``, when set above 0, is a single budget for the whole call:
    streaming, retries, and backoff share it, and a retry cannot start a fresh
    ``request_timeout``. A timeout is not retried unless the time still left
    in that budget covers another full ``request_timeout``. The error keeps
    the attempt's own duration. When a timeout was retried, the message also
    names how many attempts ran and the deadline those attempts shared.
    Connection errors and rate limits still retry while the budget remains.
    Only a rate limit lowers the AIMD ceiling.
    Timed calls share one model loop until a call fails to finish. That loop
    is then abandoned and the next call uses a new one. The abandoned thread
    is not killed. ``process_isolation=True`` runs a picklable provider in a
    child process that is terminated when the request timeout expires.

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
        wall_timeout: float | None = None,
        process_isolation: bool = False,
        provider_allowance: ProviderAllowance | None = None,
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
        self.wall_timeout = None if wall_timeout is None else float(wall_timeout)
        self.process_isolation = bool(process_isolation)
        self.provider_allowance = provider_allowance
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
        use_cache: bool = True,
        allowance_estimate: int | None = None,
        allowance_kind: str = "provider",
        allowance_stage: str | None = None,
    ):
        """``await runnable.ainvoke(payload)`` under the AIMD cap, with rate-limit retries.

        ``timeout`` overrides ``request_timeout`` for this call. ``None`` keeps
        the limiter default. ``0`` waits with no deadline. ``reserved_output``
        is added to the estimated input when the token bucket reserves a call.
        ``use_cache=False`` skips the response cache, so a retry of the same
        prompt calls the model again.
        """
        key = cache_key(payload, **(self.cache_context or {}))
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached
        estimate = self._estimate(payload, reserved_output)
        per_attempt = None if timeout is None else float(timeout)
        deadline = None
        if self.wall_timeout is not None and self.wall_timeout > 0:
            deadline = time.monotonic() + float(self.wall_timeout)

        async def once(remaining: float | None):
            attempt_timeout = per_attempt
            if remaining is not None:
                cap = self.request_timeout if attempt_timeout is None else attempt_timeout
                attempt_timeout = float(remaining) if not cap or cap <= 0 else min(float(cap), float(remaining))
                if attempt_timeout <= 0:
                    raise self._deadline_error()
            return await self._attempt(
                runnable,
                payload,
                timeout=attempt_timeout,
                estimate=estimate,
                deadline=deadline,
                allowance_estimate=allowance_estimate,
                allowance_kind=allowance_kind,
                allowance_stage=allowance_stage,
            )

        result = await self._retry(once, deadline=deadline)
        if use_cache:
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

    async def _attempt(
        self,
        runnable,
        payload,
        *,
        timeout: float | None,
        estimate: int,
        deadline: float | None = None,
        allowance_estimate: int | None = None,
        allowance_kind: str = "provider",
        allowance_stage: str | None = None,
    ):
        """One provider attempt: reserve budgets, hold one shared slot, reconcile."""
        record = {"queue_wait": 0.0, "latency": None, "reserved": 0.0, "usage": None}
        held = False
        limit = float(timeout or 0)
        if self.provider_allowance is not None:
            decision = self.provider_allowance.try_acquire(
                0 if allowance_estimate is None else int(allowance_estimate),
                kind=allowance_kind,
                stage=allowance_stage,
            )
            if decision["status"] != "acquired":
                raise AllowanceExhausted(decision)

        async def _before_call() -> None:
            await self._reserve_budgets(record, estimate, deadline=deadline)
            await self._acquire_slot(record, deadline=deadline)

        try:
            if limit > 0:
                await _await_with_hard_deadline(_before_call(), limit)
            else:
                await _before_call()
            held = True
            result = await self._invoke(runnable, payload, timeout=timeout, record=record)
            record["usage"] = usage_from_result(result)
            if self.provider_allowance is not None and allowance_estimate is not None:
                self.provider_allowance.note_estimate_observation(int(allowance_estimate), record["usage"])
            self._on_success()
            return result
        finally:
            await self._reconcile(record)
            self._remember(record)
            if held:
                await self._release_async()

    async def _reserve_budgets(self, record: dict, estimate: int, *, deadline: float | None = None) -> None:
        started = self._clock()
        await self._take_amount(1, kind="requests", deadline=deadline)
        if self.tokens_per_minute > 0 and estimate > 0:
            taken = await self._take_amount(estimate, kind="tokens", deadline=deadline)
            record["reserved"] = float(taken)
        record["queue_wait"] = max(0.0, self._clock() - started)

    async def _acquire_slot(self, record: dict, *, deadline: float | None = None) -> None:
        """Take one in-flight slot. The wait uses one monotonic deadline.

        ``deadline`` is the caller's wall clock when one was set. Otherwise a
        positive ``request_timeout`` is the cap, measured once. The wait is not
        restarted when the condition wakes early.
        """
        started = self._clock()
        cap = deadline
        limit = float(self.request_timeout or 0)
        if cap is None and limit > 0:
            cap = time.monotonic() + limit
        async with self._gate:
            while self._in_flight >= self.limit:
                if cap is None:
                    await self._gate.wait()
                    continue
                remaining = cap - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(_timeout_message(limit))
                try:
                    await asyncio.wait_for(self._gate.wait(), remaining)
                except TimeoutError:
                    raise TimeoutError(_timeout_message(limit)) from None
            self._enter()
        record["queue_wait"] = float(record.get("queue_wait") or 0) + max(0.0, self._clock() - started)

    async def _release_async(self) -> None:
        async with self._gate:
            self._leave()
            self._gate.notify()

    async def _take_amount(self, amount: float, *, kind: str, deadline: float | None = None) -> float:
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
            return await self._drain_full_bucket(kind, deadline=deadline)
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
            self._raise_if_past_deadline(delay, deadline)
            await self._sleep(max(0.0, delay))

    def _raise_if_past_deadline(self, delay: float, deadline: float | None) -> None:
        """A bucket refill must not wait past the call's monotonic deadline."""
        if deadline is None:
            return
        left = deadline - time.monotonic()
        if left <= 0 or float(delay) > left:
            raise TimeoutError(_timeout_message(float(self.request_timeout or 0)))

    async def _drain_full_bucket(self, kind: str, *, deadline: float | None = None) -> float:
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
            self._raise_if_past_deadline(delay, deadline)
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
            if limit and limit > 0:
                # A sync HTTP read inside the provider call must not run on this
                # loop. The call runs on the model loop. A pooled client is
                # constructed on that loop. The wait itself is
                # future.result(timeout) on a side thread: an idle loop cannot
                # skip the deadline. A call that does not finish is abandoned
                # with its loop so the next provider is not stuck behind it.
                # process_isolation runs the call in a child that can be killed.
                # That is the only way to stop code blocked in sleep or C.
                if self.process_isolation:
                    return await _ainvoke_in_subprocess(runnable, payload, timeout=limit)
                return await _ainvoke_off_loop(runnable, payload, timeout=limit)
            return await runnable.ainvoke(payload)
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

    def _deadline_error(self) -> TimeoutError:
        budget = 0.0 if self.wall_timeout is None else float(self.wall_timeout)
        return TimeoutError(
            f"The model call exceeded the {budget:g}s wall-clock timeout, including streaming and retries."
        )

    def _preserved_timeout(self, exc: BaseException, *, attempts: int) -> BaseException:
        """Keep the attempt's own timeout text.

        A wall-clock wrapper used to replace ``timed out after 120s`` with the
        overall budget. One attempt returns that exception unchanged. A retry
        that still fails names the attempt count and the shared deadline.
        Only a real ``TimeoutError`` is rewritten. A message that merely
        contains the word timeout (for example an oversized reservation) stays
        the original exception.
        """
        if not isinstance(exc, TimeoutError):
            return exc
        if attempts <= 1:
            return exc
        original = str(exc).strip() or "timed out"
        budget = None if self.wall_timeout is None else float(self.wall_timeout)
        if budget and budget > 0:
            return TimeoutError(f"{original} after {attempts} attempts; deadline {budget:g}s including retries")
        return TimeoutError(f"{original} after {attempts} attempts")

    def _timeout_retry_fits(self, deadline: float | None) -> bool:
        """True when another full request timeout still fits in the wall budget.

        No wall budget keeps the older retry behaviour for callers that do not
        set one. A budget that cannot hold another ``request_timeout`` does not
        start one: the attempt that just timed out already used its share.
        """
        if deadline is None:
            return True
        request = float(self.request_timeout or 0)
        if request <= 0:
            return False
        return deadline - time.monotonic() >= request

    async def _retry(self, factory: Callable, deadline: float | None = None):
        for attempt in range(self.max_retries + 1):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise self._deadline_error()
            try:
                return await factory(remaining)
            except Exception as exc:
                # is_timeout() also matches the word "timeout" in a message.
                # OversizedReservation says the provider timeout would never
                # start; that is not a provider read timeout and must not be
                # retried or rewritten as TimeoutError.
                real_timeout = isinstance(exc, TimeoutError)
                if real_timeout and not self._timeout_retry_fits(deadline):
                    raise self._preserved_timeout(exc, attempts=attempt + 1) from None
                message_only = is_timeout(exc) and not real_timeout
                if attempt >= self.max_retries or message_only or not is_transient(exc):
                    if real_timeout:
                        raise self._preserved_timeout(exc, attempts=attempt + 1) from None
                    raise
                self._note_transient(exc)
                wait = self._wait(attempt, exc)
                if deadline is not None:
                    left = deadline - time.monotonic()
                    if left <= 0 or wait >= left:
                        if real_timeout:
                            raise self._preserved_timeout(exc, attempts=attempt + 1) from None
                        raise self._deadline_error()
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


class OversizedRequest(RuntimeError):
    """A model call is still over ``token_budget`` after fitting.

    The call is sent anyway. ``estimated`` and ``budget`` are the counts the
    check compared. Callers record an ``oversized_request`` flag.
    """

    def __init__(self, message: str, *, estimated: int, budget: int) -> None:
        super().__init__(message)
        self.estimated = int(estimated)
        self.budget = int(budget)


def oversized_request_message(kind: str, estimated: int, budget: int, passage_id: str = "", *, sent: bool = True) -> str:
    """Warning text for a request that is still over ``token_budget``."""
    label = {"coding": "Coding request", "schema_repair": "Schema repair"}.get(kind, kind)
    target = f" for {passage_id}" if passage_id else ""
    outcome = "was sent" if sent else "was not sent"
    return (
        f"{label}{target} is {int(estimated)} estimated tokens, "
        f"over token_budget ({int(budget)}). "
        f"The overflow was flagged and the request {outcome}."
    )


def _timeout_message(seconds: float) -> str:
    """Label for one attempt. Sub-millisecond float noise still reads as the configured duration."""
    shown = round(float(seconds), 3)
    return f"timed out after {shown:g}s"


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


def _cached_tokens(meta: Mapping[str, Any]) -> int:
    """Provider cache reads. Missing details stay 0 and do not change the total."""
    details = meta.get("input_token_details") or meta.get("prompt_tokens_details") or {}
    if not isinstance(details, Mapping):
        details = {}
    cached = (
        details.get("cache_read")
        or details.get("cache_read_input_tokens")
        or details.get("cached_tokens")
        or meta.get("cached_tokens")
        or meta.get("cache_read_input_tokens")
        or 0
    )
    return int(cached or 0)


def usage_from_message(raw: Any) -> dict[str, int]:
    """Read token counts from a LangChain message, or zeros when the provider sent none."""
    meta = getattr(raw, "usage_metadata", None) or {}
    if not meta and raw is not None:
        response = getattr(raw, "response_metadata", None) or {}
        token_usage = response.get("token_usage") or response.get("usage") or {}
        prompt_details = token_usage.get("prompt_tokens_details") or {}
        meta = {
            "input_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0,
            "output_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0,
            "total_tokens": token_usage.get("total_tokens") or 0,
            "cached_tokens": (prompt_details or {}).get("cached_tokens") or token_usage.get("cached_tokens") or 0,
        }
    input_tokens = int(meta.get("input_tokens") or 0)
    output_tokens = int(meta.get("output_tokens") or 0)
    total_tokens = int(meta.get("total_tokens") or 0)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "cached_tokens": _cached_tokens(meta),
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
    keys = ("input_tokens", "cached_tokens", "output_tokens", "total_tokens")
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


def _record_raw(result: Any, holder: list | None) -> None:
    """Append the provider message text. Structured parsing does not keep it."""
    if holder is None:
        return
    raw = result.get("raw") if isinstance(result, dict) and "parsed" in result else result
    holder.append(_raw_text(raw))


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


def _request_overflow(
    messages,
    schema,
    tokenizer,
    *,
    budget: int | None,
    reserved: int,
    kind: str,
    passage_id: str = "",
) -> OversizedRequest | None:
    """Last measurement before a provider call.

    A request that is still over ``budget`` is flagged and still sent.
    """
    if budget is None:
        return None
    from automated_grounded_theory.batching import request_input_tokens

    estimated = request_input_tokens(messages, schema, tokenizer) + max(0, int(reserved))
    if estimated <= int(budget):
        return None
    message = oversized_request_message(kind, estimated, int(budget), passage_id)
    return OversizedRequest(message, estimated=estimated, budget=int(budget))


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
    passage_id: str = "",
    on_overflow: Callable[[OversizedRequest], None] | None = None,
    use_cache: bool = True,
    raw_out: list | None = None,
    preserve_failed: bool = False,
    send_repair_overflow: bool = True,
    attempt_repair: bool = True,
    allowance_kind: str = "structured",
    allowance_stage: str | None = None,
    call_audits: list | None = None,
) -> tuple[tuple[dict, dict], list]:
    """Invoke ``schema`` once, then repair a single malformed output.

    The follow-up user message starts with ``REPAIR THIS OUTPUT``. The second
    failure raises the first parse error. Returns ``((payload, usage), messages)``.
    Usage on a repaired call includes the failed response as well as the
    successful one.

    When ``token_budget`` is set, the repair request is measured again,
    including the schema, and the repair note is shortened until the input
    plus ``reserved_output`` fits that budget. A repair that is still over
    the budget is logged, reported to ``on_overflow``, and sent unless
    ``send_repair_overflow`` is false. A shared :class:`ProviderAllowance` on
    the limiter is acquired for the first request and again for the repair.
    Both acquisitions use ``allowance_stage``, so a repair spends the
    originating stage's call allowance. When the repair does not fit and
    ``preserve_failed`` is true, the failed output is returned with
    ``_agt_unparsed`` instead of being dropped.     A repair that also fails to
    parse keeps that second response and its usage, and the returned usage
    is the sum of both responses.

    ``attempt_repair=False`` returns after the first response. A parse failure
    is ``_agt_unparsed`` with ``needs_repair`` and is not followed by a second
    provider call. The caller can checkpoint that response before repairing.
    """
    from langchain_core.messages import HumanMessage

    from automated_grounded_theory.batching import request_input_tokens, shrink_repair_messages

    def _remember(audit: dict) -> None:
        if call_audits is not None:
            call_audits.append(audit)

    def _preserved(error: BaseException, usage: dict, *, reason: str, sent_messages: list, raw_text: str = "", received: bool = True) -> tuple[tuple[dict, dict], list]:
        if not raw_text and raw_out:
            raw_text = str(raw_out[-1] or "")
        payload = {
            "_agt_unparsed": True,
            "needs_review": True,
            "flagged": True,
            "reason": reason,
            "error": str(error),
            "raw_response": raw_text,
        }
        # A repair that was not sent is a preservation record. It must not
        # carry the original response's usage, or that response is counted twice.
        _remember(
            {
                "kind": "schema_repair",
                "messages": sent_messages,
                "estimated_input_tokens": 0,
                "usage": usage if received else {},
                "status": "schema_unparsed" if received else "not_sent",
                "reason": reason,
                "parsed": payload,
                "received": received,
            }
        )
        return (payload, usage), sent_messages

    runnable = structured_runnable(model, schema)
    extra: dict[str, Any] = {}
    if timeout is not None:
        extra["timeout"] = timeout
    if reserved_output is not None:
        extra["reserved_output"] = reserved_output
    if not use_cache:
        extra["use_cache"] = False
    reserved = 0 if reserved_output is None else max(0, int(reserved_output))
    if token_budget is None:
        budget = None
    elif preserve_failed:
        budget = max(0, int(token_budget))
    else:
        budget = max(1, int(token_budget))
    sent = list(messages)
    first_estimate = request_input_tokens(sent, schema, tokenizer) + reserved
    # The coding caller measures this payload and flags it when it is still over.
    # The repair below is a second invocation and acquires its own allowance.
    result = await limiter.ainvoke(
        runnable,
        sent,
        allowance_estimate=first_estimate,
        allowance_kind=allowance_kind,
        allowance_stage=allowance_stage,
        **extra,
    )
    _record_raw(result, raw_out)
    failed_raw = _raw_text(result.get("raw")) if isinstance(result, dict) else _raw_text(result)
    try:
        parsed = parse_structured(result, schema)
    except ValueError as first:
        failed_usage = usage_from_result(result)
        _remember(
            {
                "kind": allowance_kind,
                "messages": sent,
                "estimated_input_tokens": first_estimate,
                "usage": failed_usage,
                "status": "parse_failed",
                "reason": str(first),
                "parsed": {},
                "raw_response": failed_raw,
            }
        )
        if not attempt_repair:
            payload = {
                "_agt_unparsed": True,
                "needs_review": True,
                "flagged": True,
                "needs_repair": True,
                "reason": "parse_failed",
                "error": str(first),
                "raw_response": failed_raw,
            }
            return (payload, failed_usage), sent
        repair = list(messages) + [
            HumanMessage(
                content=(
                    "REPAIR THIS OUTPUT\n"
                    f"The previous output was not valid {schema.__name__}: {first}\n"
                    "Return only the structured object."
                )
            )
        ]
        if budget is not None and budget > 0:
            ceiling = max(1, budget - reserved)
            repair = shrink_repair_messages(repair, schema, ceiling, tokenizer)
        repair_estimate = request_input_tokens(repair, schema, tokenizer) + reserved
        overflow = _request_overflow(
            repair,
            schema,
            tokenizer,
            budget=budget,
            reserved=reserved,
            kind="schema_repair",
            passage_id=passage_id,
        )
        if overflow is not None:
            if on_overflow is not None:
                on_overflow(overflow)
            if not send_repair_overflow:
                logger.warning(
                    "%s",
                    oversized_request_message(
                        "schema_repair",
                        overflow.estimated,
                        overflow.budget,
                        passage_id,
                        sent=False,
                    ),
                )
                return _preserved(
                    first,
                    failed_usage,
                    reason="request_budget",
                    sent_messages=sent,
                    raw_text=failed_raw,
                    received=False,
                )
            logger.warning("%s", overflow)
        try:
            limiter.schema_repairs = int(getattr(limiter, "schema_repairs", 0) or 0) + 1
            repaired = await limiter.ainvoke(
                runnable,
                repair,
                allowance_estimate=repair_estimate,
                allowance_kind="schema_repair",
                allowance_stage=allowance_stage,
                **extra,
            )
        except AllowanceExhausted as exc:
            if preserve_failed:
                held = _preserved(first, failed_usage, reason=exc.reason, sent_messages=sent, raw_text=failed_raw)
                if call_audits:
                    call_audits[-1]["estimated_input_tokens"] = repair_estimate
                    call_audits[-1]["reason"] = exc.reason
                    call_audits[-1]["status"] = "refused"
                return held
            raise
        _record_raw(repaired, raw_out)
        repair_raw = _raw_text(repaired.get("raw")) if isinstance(repaired, dict) else _raw_text(repaired)
        repair_usage = usage_from_result(repaired)
        try:
            payload, usage = parse_structured(repaired, schema)
        except ValueError as repair_error:
            if preserve_failed:
                combined = combine_usage(failed_usage, repair_usage)
                payload = {
                    "_agt_unparsed": True,
                    "needs_review": True,
                    "flagged": True,
                    "reason": "schema_unparsed",
                    "error": str(repair_error),
                    "raw_response": failed_raw,
                    "repair_raw_response": repair_raw,
                    "repair_usage": repair_usage,
                }
                _remember(
                    {
                        "kind": "schema_repair",
                        "messages": repair,
                        "estimated_input_tokens": repair_estimate,
                        "usage": repair_usage,
                        "status": "schema_unparsed",
                        "reason": "schema_unparsed",
                        "parsed": payload,
                        "raw_response": repair_raw,
                    }
                )
                return (payload, combined), repair
            raise first from None
        _remember(
            {
                "kind": "schema_repair",
                "messages": repair,
                "estimated_input_tokens": repair_estimate,
                "usage": usage,
                "status": "validated",
                "reason": "",
                "parsed": payload,
            }
        )
        return (payload, combine_usage(failed_usage, usage)), repair
    else:
        payload, usage = parsed
        _remember(
            {
                "kind": allowance_kind,
                "messages": sent,
                "estimated_input_tokens": first_estimate,
                "usage": usage,
                "status": "validated",
                "reason": "",
                "parsed": payload,
            }
        )
        return parsed, sent


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
        if not ready.wait(30):
            raise TimeoutError("The package event loop did not start within 30s.")
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


def run_sync(coro, *, timeout: float | None = None):
    """Run ``coro`` to completion, including inside a running event loop.

    Every sync call shares one background loop. A chat model created during
    ``open_code`` stays bound to that loop, so ``build_themes`` can reuse the
    same instance. Shared httpx clients are not closed here: closing them
    made the next call raise ``Event loop is closed``.

    ``timeout`` above 0 is a wall-clock bound on this wait. When it expires the
    underlying task is cancelled and ``TimeoutError`` is raised. The caller
    does not keep waiting if that task ignores cancellation.
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
        if timeout is not None and float(timeout) > 0:
            return _result_with_deadline_watch(future, loop, float(timeout))
        return future.result()
    except TimeoutError as exc:
        # concurrent.futures.TimeoutError is TimeoutError on Python 3.11+.
        # A coroutine that already raised TimeoutError is a finished future.
        # Only an unfinished future means this wait itself expired. Rewriting
        # a finished call timeout as the bridge budget dropped the real label
        # (a 120s call was reported as the whole-run allowance). A hard
        # deadline the loop never resumed is also that call timeout.
        if future.done() and not future.cancelled():
            return _propagate_finished(future)
        if "timed out after" in str(exc) and "sync bridge" not in str(exc):
            raise
        future.cancel()
        try:
            return future.result(timeout=0.05)
        except (TimeoutError, concurrent.futures.CancelledError):
            if future.done() and not future.cancelled():
                return _propagate_finished(future)
        except BaseException:
            if future.done() and not future.cancelled():
                return _propagate_finished(future)
        raise TimeoutError(f"sync bridge timed out after {float(timeout):g}s") from None
    except BaseException as exc:
        raise exc from None


def _result_with_deadline_watch(future: concurrent.futures.Future, loop, timeout: float):
    """Wait out ``timeout`` and surface a call deadline the loop did not.

    ``future.result`` on this thread is already independent of the package
    loop. A per-call watchdog that fired and was not observed within a short
    grace means that loop did not resume the waiter. This wait then raises the
    call's own timeout and abandons the loop, instead of sitting until the
    bridge budget. A healthy timeout still arrives as the coroutine's exception.
    """
    deadline = time.monotonic() + float(timeout)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            return future.result(timeout=min(0.25, remaining))
        except TimeoutError:
            if future.done() and not future.cancelled():
                return _propagate_finished(future)
            late = _take_stale_deadline()
            if late is not None and not future.done():
                _abandon_package_loop(loop)
                raise late
            continue
    if future.done() and not future.cancelled():
        return _propagate_finished(future)
    future.cancel()
    try:
        return future.result(timeout=0.05)
    except (TimeoutError, concurrent.futures.CancelledError):
        if future.done() and not future.cancelled():
            return _propagate_finished(future)
    except BaseException:
        if future.done() and not future.cancelled():
            return _propagate_finished(future)
    raise TimeoutError(f"sync bridge timed out after {float(timeout):g}s") from None


def _abandon_package_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Stop using ``loop`` for later sync calls. Do not join its thread."""
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is loop:
            _LOOP = None
    if loop is None or not loop.is_running():
        return
    try:
        loop.call_soon_threadsafe(loop.stop)
    except RuntimeError:
        return


_DEADLINE_NOTICES: list[dict[str, Any]] = []
_DEADLINE_NOTICE_LOCK = threading.Lock()
_DEADLINE_GRACE = 0.75


def _notice_unobserved_deadline(exc: BaseException) -> None:
    """Record a call deadline that the waiter has not resumed yet."""
    with _DEADLINE_NOTICE_LOCK:
        _DEADLINE_NOTICES.append({"at": time.monotonic(), "exc": exc})


def _clear_deadline_notice() -> None:
    with _DEADLINE_NOTICE_LOCK:
        _DEADLINE_NOTICES.clear()


def _take_stale_deadline() -> BaseException | None:
    """Return a deadline notice older than the grace, and drop the list."""
    now = time.monotonic()
    with _DEADLINE_NOTICE_LOCK:
        stale = [item for item in _DEADLINE_NOTICES if now - float(item["at"]) >= _DEADLINE_GRACE]
        if not stale:
            return None
        exc = stale[0]["exc"]
        _DEADLINE_NOTICES.clear()
        return exc


def _propagate_finished(future: concurrent.futures.Future):
    """Return the finished future's value, or raise its exception.

    A cancelled future has no result. ``exception()`` raises ``CancelledError``
    in that case, which is not a call timeout.
    """
    if not future.done() or future.cancelled():
        return None
    try:
        inner = future.exception()
    except concurrent.futures.CancelledError:
        return None
    if inner is not None:
        raise inner
    return future.result()


_ABANDONED_TIMEOUTS: set[asyncio.Task] = set()


def _consume_task_exception(task: asyncio.Task) -> None:
    _ABANDONED_TIMEOUTS.discard(task)
    if not task.done() or task.cancelled():
        return
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        return


def _hold_abandoned(task: asyncio.Task) -> None:
    """Keep a task that ignored cancellation until it actually finishes.

    Dropping the only reference closes the coroutine from outside the loop and
    raises ``GeneratorExit`` inside the provider call.
    """
    _ABANDONED_TIMEOUTS.add(task)
    task.add_done_callback(_consume_task_exception)


_MODEL_LOOP: asyncio.AbstractEventLoop | None = None
_MODEL_LOOP_LOCK = threading.Lock()
_DEADLINE_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_DEADLINE_LOCK = threading.Lock()


def _deadline_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Threads that only block in ``future.result(timeout)``.

    The package loop's default executor can also run a provider's sync read.
    A deadline parked on that same pool does not start until the read returns,
    so the call waits out the sync bridge instead of the request timeout.
    """
    global _DEADLINE_EXECUTOR
    with _DEADLINE_LOCK:
        if _DEADLINE_EXECUTOR is None:
            _DEADLINE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=32,
                thread_name_prefix="agt-call-deadline",
            )
        return _DEADLINE_EXECUTOR


_ABANDONED_MODEL_LOOPS: list[asyncio.AbstractEventLoop] = []


def _model_loop() -> asyncio.AbstractEventLoop:
    """The live loop for timed model calls.

    Successful calls share one loop so a client built for that loop can be
    reused. A call that does not finish (the loop thread is blocked in
    synchronous provider code) abandons this loop. The next call starts
    another. The abandoned thread is not killed: ``time.sleep`` and a blocked
    C extension keep running until they return. Hard termination needs
    :func:`run_provider_in_subprocess`.
    """
    global _MODEL_LOOP
    with _MODEL_LOOP_LOCK:
        if _MODEL_LOOP is not None and _MODEL_LOOP.is_running():
            return _MODEL_LOOP
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        threading.Thread(target=_run, name="agt-model-loop", daemon=True).start()
        if not ready.wait(2):
            raise TimeoutError("The model-call loop did not start within 2s.")
        _MODEL_LOOP = loop
        return loop


def _abandon_model_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Stop scheduling new work on ``loop``. Do not join its thread.

    Cancelling a task does not interrupt synchronous code running on the loop
    thread. Leaving that loop in place makes every later call wait behind the
    blocker. The thread is abandoned. It is a daemon, and it is not killed.
    """
    global _MODEL_LOOP
    with _MODEL_LOOP_LOCK:
        if _MODEL_LOOP is loop:
            _MODEL_LOOP = None
        if loop not in _ABANDONED_MODEL_LOOPS:
            _ABANDONED_MODEL_LOOPS.append(loop)
    if not loop.is_running():
        return
    try:
        loop.call_soon_threadsafe(loop.stop)
    except RuntimeError:
        return


def _submit_on_loop(loop: asyncio.AbstractEventLoop, coro, context: contextvars.Context):
    """Schedule ``coro`` on ``loop`` in ``context`` and return a concurrent future."""
    future: concurrent.futures.Future = concurrent.futures.Future()

    def _start() -> None:
        try:
            task = loop.create_task(coro, context=context)
        except BaseException as exc:
            future.set_exception(exc)
            return

        def _done(done_task: asyncio.Task) -> None:
            # Always read the task exception. Skipping it when the outer wait
            # already finished logs "Task exception was never retrieved".
            if done_task.cancelled():
                if not future.done():
                    future.cancel()
                return
            exc = done_task.exception()
            if future.done():
                return
            if exc is not None:
                future.set_exception(exc)
            else:
                future.set_result(done_task.result())

        task.add_done_callback(_done)
        future.task = task  # type: ignore[attr-defined]

    loop.call_soon_threadsafe(_start)
    return future


# A live loop cancels the provider from its own wait at ``timeout``. This
# thread waits a little longer so that cancel is delivered before the loop is
# abandoned. A blocked loop never runs that wait, and this thread still fires.
_WATCHDOG_SLACK = 0.2


class _MonotonicDeadline:
    """One hard deadline on its own thread. It does not use an asyncio loop.

    The timer is ``threading.Event.wait``, which is a single monotonic wait.
    It is not ``future.result`` (that wait shares the future's condition with
    the loop that must resolve it) and it is not ``asyncio.sleep`` (that wait
    never runs while its loop is blocked). The clock is the request timeout
    plus a short slack, so a responsive loop can cancel the provider and raise
    its own ``TimeoutError`` first. On expiry the model loop is abandoned, a
    configured child process is killed, and ``gate`` is completed from this
    thread so a waiter blocked in ``future.result`` can resume without that
    loop. The error still names the request timeout, not the slack.
    """

    def __init__(self, timeout: float):
        self.timeout = float(timeout)
        self.deadline = time.monotonic() + self.timeout + _WATCHDOG_SLACK
        self.gate: concurrent.futures.Future = concurrent.futures.Future()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="agt-hard-deadline", daemon=True)
        self.work: concurrent.futures.Future | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.process = None

    def arm(self, work: concurrent.futures.Future, loop, process=None) -> None:
        self.work = work
        self.loop = loop
        self.process = process
        work.add_done_callback(lambda _future: self._wake.set())
        self._thread.start()

    def _run(self) -> None:
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                break
            if self._wake.wait(remaining):
                work = self.work
                if work is not None and work.done():
                    self._finish_from(work)
                    return
                continue
            break
        work = self.work
        if work is not None and work.done():
            self._finish_from(work)
            return
        self._expire()

    def _finish_from(self, work: concurrent.futures.Future) -> None:
        if self.gate.done():
            return
        try:
            exc = work.exception()
        except concurrent.futures.CancelledError:
            if not self.gate.done():
                self.gate.cancel()
            return
        if self.gate.done():
            return
        if exc is not None:
            self.gate.set_exception(exc)
        else:
            self.gate.set_result(work.result())

    def _expire(self) -> None:
        loop = self.loop
        work = self.work
        task = getattr(work, "task", None) if work is not None else None
        if task is not None and loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        if work is not None and not work.done():
            work.cancel()
        if loop is not None:
            _abandon_model_loop(loop)
        process = self.process
        if process is not None and getattr(process, "is_alive", lambda: False)():
            process.terminate()
            process.join(0.5)
            if process.is_alive():
                process.kill()
                process.join(0.5)
        exc = TimeoutError(_timeout_message(self.timeout))
        _notice_unobserved_deadline(exc)
        if not self.gate.done():
            self.gate.set_exception(exc)


async def _ainvoke_off_loop(runnable, payload, *, timeout: float):
    """Run one model call on the current model loop.

    The deadline is a dedicated thread waiting on ``time.monotonic``. It does
    not wait on the call's future, and it does not sleep on either asyncio
    loop. When the clock expires the model loop is abandoned and ``gate`` is
    failed from that thread. A pooled async client is constructed for this
    loop. httpx clients the provider builds during the call get a socket
    timeout slightly past this deadline, so an open connection does not stay
    open after the loop is abandoned. A timeout of 0 or less is not passed here.

    A finished call's ``TimeoutError`` is that call's own error. An unfinished
    call means the loop thread did not reach the deadline. That loop is
    abandoned. The blocked thread is not killed. ``process_isolation`` is the
    path that can kill the provider.
    """
    from automated_grounded_theory.backends import (
        arm_provider_clients,
        bind_http_timeout_cap,
        provider_socket_timeout,
        reset_http_timeout_cap,
    )

    loop = _model_loop()
    token = bind_http_timeout_cap(timeout)
    try:
        context = contextvars.copy_context()
    finally:
        reset_http_timeout_cap(token)
    socket_timeout = provider_socket_timeout(timeout)

    async def _call():
        await arm_provider_clients(runnable, socket_timeout)
        return await await_with_wall_timeout(runnable.ainvoke(payload), timeout)

    future = _submit_on_loop(loop, _call(), context)
    watch = _MonotonicDeadline(timeout)
    watch.arm(future, loop)
    try:
        return await asyncio.wrap_future(watch.gate)
    except TimeoutError:
        _clear_deadline_notice()
        if future.done() and not future.cancelled():
            exc = future.exception()
            if exc is not None:
                raise exc
            return future.result()
        task = getattr(future, "task", None)
        if task is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        _abandon_model_loop(loop)
        raise TimeoutError(_timeout_message(timeout)) from None


def _subprocess_provider_main(blob: bytes, result_queue) -> None:
    """Child entry. ``runnable`` and ``payload`` were pickled by the parent."""
    import pickle

    try:
        runnable, payload = pickle.loads(blob)
        result = asyncio.run(runnable.ainvoke(payload))
        result_queue.put(("ok", pickle.dumps(result)))
    except BaseException as exc:
        try:
            import pickle as _pickle

            result_queue.put(("err", f"{type(exc).__name__}: {exc}", _pickle.dumps(exc)))
        except Exception:
            result_queue.put(("err", f"{type(exc).__name__}: {exc}", None))


def run_provider_in_subprocess(runnable, payload, *, timeout: float):
    """Run ``runnable.ainvoke(payload)`` in a child process and kill it on timeout.

    Abandoning a loop thread does not stop ``time.sleep`` or a blocked C
    extension. A subprocess can be terminated. ``runnable`` and ``payload``
    must be picklable. A live HTTP client, a closure, or most chat-model
    instances are not: this raises ``TypeError`` instead of running them on
    the shared loop. The child uses ``spawn``, so it does not inherit this
    process's event loop.
    """
    import multiprocessing
    import pickle

    try:
        blob = pickle.dumps((runnable, payload))
    except Exception as exc:
        raise TypeError(
            "process_isolation needs a picklable provider and payload. "
            "A live HTTP client is not picklable. "
            "Abandoning a blocked loop thread does not kill that thread; "
            "only this subprocess can."
        ) from exc
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_subprocess_provider_main, args=(blob, result_queue), daemon=True)
    process.start()
    process.join(float(timeout))
    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join(1)
        exc = TimeoutError(_timeout_message(timeout))
        _notice_unobserved_deadline(exc)
        raise exc
    if result_queue.empty():
        raise RuntimeError("The provider process ended without a result.")
    message = result_queue.get()
    kind = message[0]
    if kind == "ok":
        return pickle.loads(message[1])
    text = message[1] if len(message) > 1 else "provider process failed"
    loaded = message[2] if len(message) > 2 else None
    if loaded:
        try:
            raise pickle.loads(loaded)
        except Exception:
            pass
    raise RuntimeError(text)


async def _ainvoke_in_subprocess(runnable, payload, *, timeout: float):
    """Blocking process wait, off the caller's loop.

    ``run_provider_in_subprocess`` joins the child until the monotonic deadline
    and then terminates it. That is the configured process-isolation path.
    """
    try:
        return await asyncio.to_thread(run_provider_in_subprocess, runnable, payload, timeout=timeout)
    except TimeoutError:
        _clear_deadline_notice()
        raise


async def _await_with_hard_deadline(awaitable, timeout: float):
    """Run ``awaitable`` on this loop under the monotonic deadline thread.

    Slot waits, budget waits, embedding, and client setup happen before
    ``_ainvoke_off_loop`` arms that thread. A stall there used to sit until the
    sync bridge, which is one request timeout times the call allowance. This
    wait uses the same clock as one model call. A live loop still finishes the
    awaitable itself. The thread is the backstop when this loop does not resume.
    """
    if timeout is None or float(timeout) <= 0:
        return await awaitable
    loop = asyncio.get_running_loop()
    future: concurrent.futures.Future = concurrent.futures.Future()
    task = asyncio.ensure_future(awaitable)

    def _done(done_task: asyncio.Task) -> None:
        if future.done():
            return
        if done_task.cancelled():
            if not future.done():
                future.cancel()
            return
        exc = done_task.exception()
        if exc is not None:
            future.set_exception(exc)
        else:
            future.set_result(done_task.result())

    task.add_done_callback(_done)
    future.task = task  # type: ignore[attr-defined]
    watch = _MonotonicDeadline(float(timeout))
    watch.arm(future, loop)
    try:
        return await asyncio.wrap_future(watch.gate)
    except TimeoutError:
        _clear_deadline_notice()
        if future.done() and not future.cancelled():
            exc = future.exception()
            if exc is not None:
                raise exc
            return future.result()
        task.cancel()
        _abandon_model_loop(loop)
        raise TimeoutError(_timeout_message(float(timeout))) from None


async def await_with_wall_timeout(awaitable, timeout: float | None):
    """Wait up to ``timeout`` seconds, then cancel and raise ``TimeoutError``.

    ``asyncio.wait_for`` cancels the task and then waits for it to finish. A
    provider that swallows ``CancelledError`` holds that second wait forever.
    This returns when the clock expires, cancels the task, and does not wait
    for the cancellation to complete. A timeout of 0 or less waits with no
    deadline.
    """
    if timeout is None or float(timeout) <= 0:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    task.add_done_callback(_consume_task_exception)
    try:
        done, _pending = await asyncio.wait({task}, timeout=float(timeout))
    except asyncio.CancelledError:
        task.cancel()
        _hold_abandoned(task)
        raise
    if task in done:
        return task.result()
    task.cancel()
    _hold_abandoned(task)
    await asyncio.sleep(0)
    raise TimeoutError(_timeout_message(timeout))
