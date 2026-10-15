"""0.7.8: claim-test pairs, loop isolation, httpx ownership, review packing."""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from automated_grounded_theory.async_support import CallLimiter, run_sync
from automated_grounded_theory.batching import TOKEN_ESTIMATE_ADJUSTMENT_NOTE, adjusted_token_estimate, estimate_tokens
from automated_grounded_theory.decision_review import DecisionReviewer, claims_payload, next_claim_batch
from automated_grounded_theory.focused import (
    _apply_claim_test,
    _claim_response_reason,
    _possible_direction_mismatch,
    focused_code,
)
from automated_grounded_theory.schemas import FocusedClaimJudgment, FocusedClaimTest

from tests.fake_focused import FakeFocusedCoder
from tests.test_decision_mode import FakeDecisionModel
from tests.test_v070 import _entry, _source
from tests.test_v074 import _passage, _segment
from tests.test_v075 import _claim_run


def test_unrequested_pair_and_unanchored_quote_do_not_dispute():
    passage = "The incident was recorded in the morning log."
    props = [
        {
            "entry_id": "right-claim",
            "evidence_condition": "anchored",
            "flags": [],
            "negative_cases": [],
            "claim_tests": [],
            "pole": "during the morning",
        },
        {
            "entry_id": "wrong-claim",
            "evidence_condition": "anchored",
            "flags": [],
            "negative_cases": [],
            "claim_tests": [],
            "pole": "during the morning",
        },
    ]
    counts = {"supports": 0, "contradicts": 0, "qualifies": 0, "irrelevant": 0}
    swapped = {
        "passage_id": "incident",
        "entry_id": "wrong-claim",
        "judgment": "contradicts",
        "quote": "FABRICATED QUOTE",
        "rationale": "not this claim",
    }
    reason = _claim_response_reason(swapped, "incident", passage, {"incident": "right-claim"})
    assert reason == "unrequested_pair"
    recorded = _apply_claim_test(
        props,
        [],
        swapped,
        "incident",
        counts,
        valid=False,
        passage_text=passage,
        invalid_reason=reason,
    )
    assert recorded["reason"] == "unrequested_pair"
    assert recorded["valid_test"] is False
    wrong = props[1]
    assert wrong["evidence_condition"] == "anchored"
    assert not wrong["negative_cases"]
    assert "claim_test_invalid" in wrong["flags"]
    assert wrong["claim_tests"][-1]["reason"] == "unrequested_pair"
    assert props[0]["evidence_condition"] == "anchored"
    assert not props[0]["claim_tests"]

    fabricated = {
        "passage_id": "incident",
        "entry_id": "right-claim",
        "judgment": "contradicts",
        "quote": "FABRICATED QUOTE",
        "rationale": "made up",
    }
    reason = _claim_response_reason(fabricated, "incident", passage, {"incident": "right-claim"})
    assert reason == "unanchored_quote"
    _apply_claim_test(
        props,
        [],
        fabricated,
        "incident",
        counts,
        valid=False,
        passage_text=passage,
        invalid_reason=reason,
    )
    assert props[0]["evidence_condition"] == "anchored"
    assert not props[0]["negative_cases"]
    assert props[0]["claim_tests"][-1]["reason"] == "unanchored_quote"
    assert props[0]["claim_tests"][-1]["valid_test"] is False

    anchored = {
        "passage_id": "incident",
        "entry_id": "right-claim",
        "judgment": "contradicts",
        "quote": "recorded in the morning",
        "rationale": "the log contradicts the claim",
    }
    assert _claim_response_reason(anchored, "incident", passage, {"incident": "right-claim"}) == ""
    _apply_claim_test(
        props,
        [],
        anchored,
        "incident",
        counts,
        valid=True,
        passage_text=passage,
    )
    assert props[0]["evidence_condition"] == "disputed"
    assert props[0]["negative_cases"][-1]["valid_test"] is True
    assert props[0]["negative_cases"][-1]["quote"] == "recorded in the morning"


class _WrongClaim(FakeFocusedCoder):
    def claim_test_batch(self, payload):
        rows = []
        for passage in payload.get("passages") or []:
            claim = passage.get("claim") or {}
            requested = str(claim.get("entry_id") or "")
            other = "dime-0001" if requested.startswith("prop") else "prop-0001"
            rows.append(
                FocusedClaimJudgment(
                    passage_id=str(passage.get("passage_id") or "incident"),
                    entry_id=other,
                    judgment="contradicts",
                    rationale="This names a different claim.",
                    quote="FABRICATED QUOTE",
                )
            )
        return FocusedClaimTest(passages=rows)


class _FabricatedQuote(FakeFocusedCoder):
    def claim_test_batch(self, payload):
        rows = []
        for passage in payload.get("passages") or []:
            claim = passage.get("claim") or {}
            rows.append(
                FocusedClaimJudgment(
                    passage_id=str(passage.get("passage_id") or ""),
                    entry_id=str(claim.get("entry_id") or ""),
                    judgment="contradicts",
                    rationale="The quote is not in the passage.",
                    quote="FABRICATED QUOTE",
                )
            )
        return FocusedClaimTest(passages=rows)


def _rejections(result):
    found = []
    for row in result.prompt_log:
        found.extend(row.get("claim_test_rejections") or [])
    return found


def test_live_repro_does_not_dispute_an_unrelated_claim():
    result = _claim_run(_WrongClaim())
    assert _rejections(result)
    assert any(item.get("reason") == "unrequested_pair" for item in _rejections(result))
    assert not any(
        row.get("evidence_condition") == "disputed" for row in result.properties + result.dimensions
    )
    assert not any(row.get("negative_cases") for row in result.properties + result.dimensions)


def test_unanchored_quote_is_kept_and_does_not_dispute():
    result = _claim_run(_FabricatedQuote())
    assert any(item.get("reason") == "unanchored_quote" for item in _rejections(result))
    flagged = [
        row
        for row in result.properties + result.dimensions
        if "claim_test_invalid" in (row.get("flags") or [])
    ]
    assert flagged
    assert flagged[0]["claim_tests"][-1]["reason"] == "unanchored_quote"
    assert flagged[0]["claim_tests"][-1]["valid_test"] is False
    assert flagged[0].get("evidence_condition") != "disputed"
    assert not flagged[0].get("negative_cases")


def test_direction_diagnostic_allows_a_paraphrase_and_scopes_clauses():
    assert not _possible_direction_mismatch("needs to visit more", "The CEO visits too rarely.")
    assert not _possible_direction_mismatch(
        "needs to visit more",
        "The CEO visits too rarely. The cafeteria should serve better coffee.",
    )
    assert _possible_direction_mismatch(
        "needs to visit the sites more",
        "The CEO walks the floor during site visits. The cafeteria should serve better coffee.",
    )


class _Blocker:
    async def ainvoke(self, payload):
        time.sleep(0.6)
        return "late"


class _Fast:
    async def ainvoke(self, payload):
        return "fast"


def test_a_blocked_provider_does_not_stall_the_next_call():
    limiter = CallLimiter(1, max_retries=0, request_timeout=0.05, wall_timeout=0.05)

    async def run():
        with pytest.raises(TimeoutError, match=r"timed out after 0\.05s"):
            await limiter.ainvoke(_Blocker(), {"call": "block"})
        started = time.perf_counter()
        result = await limiter.ainvoke(_Fast(), {"call": "next"})
        return result, time.perf_counter() - started

    result, elapsed = run_sync(run(), timeout=3)
    assert result == "fast"
    assert elapsed < 0.3


class _PicklableSleeper:
    def __init__(self, seconds: float = 30):
        self.seconds = seconds

    async def ainvoke(self, payload):
        time.sleep(self.seconds)
        return "late"


class _PicklableFast:
    async def ainvoke(self, payload):
        return "fast"


def test_process_isolation_terminates_a_blocked_provider():
    limiter = CallLimiter(1, max_retries=0, request_timeout=1, wall_timeout=1, process_isolation=True)
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"timed out after 1s"):
        run_sync(limiter.ainvoke(_PicklableSleeper(), {}, use_cache=False), timeout=20)
    assert time.perf_counter() - started < 15
    fast = CallLimiter(1, max_retries=0, request_timeout=10, wall_timeout=10, process_isolation=True)
    assert run_sync(fast.ainvoke(_PicklableFast(), {}, use_cache=False), timeout=20) == "fast"


def _keepalive_server():
    import socket

    def handle(conn):
        try:
            for _request in range(4):
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    data += chunk
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok")
        except OSError:
            return

    def serve(sock):
        while True:
            try:
                conn, _addr = sock.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    threading.Thread(target=serve, args=(sock,), daemon=True).start()
    return sock, sock.getsockname()[1]


class _HttpHolder:
    def __init__(self, client, url):
        self.http_async_client = client
        self.url = url

    async def ainvoke(self, payload):
        response = await self.http_async_client.get(self.url)
        return response.text


def test_httpx_client_used_on_one_loop_works_on_the_model_loop():
    assert httpx.__version__.startswith("0.28.")
    _sock, port = _keepalive_server()
    url = f"http://127.0.0.1:{port}/"
    loop = asyncio.new_event_loop()

    def spin():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=spin, daemon=True).start()
    client = httpx.AsyncClient()

    async def prime():
        response = await client.get(url)
        return response.text

    assert asyncio.run_coroutine_threadsafe(prime(), loop).result(timeout=3) == "ok"
    holder = _HttpHolder(client, url)
    limiter = CallLimiter(1, max_retries=0, request_timeout=2, wall_timeout=2)
    first = run_sync(limiter.ainvoke(holder, {"n": 1}, use_cache=False), timeout=5)
    second = run_sync(limiter.ainvoke(holder, {"n": 2}, use_cache=False), timeout=5)
    assert first == "ok"
    assert second == "ok"
    assert holder.http_async_client is not client
    loop.call_soon_threadsafe(loop.stop)


class _SizedReviewer:
    def __init__(self, *, max_tokens, token_budget, one=1000, two=1427):
        self.max_tokens = max_tokens
        self.token_budget = token_budget
        self.one = one
        self.two = two

    def estimate(self, payload, schema):
        count = len(payload.get("claims") or [])
        if count <= 1:
            return self.one
        return self.two


def test_claim_batches_pack_against_the_remaining_allowance():
    state = {"metadata": {}, "passages": [], "codebook": {}}
    claims = [
        {"entry_id": "a", "passage_id": "p1"},
        {"entry_id": "b", "passage_id": "p2"},
    ]
    reviewer = _SizedReviewer(max_tokens=1010, token_budget=6000)
    batch, stop, detail = next_claim_batch(reviewer, state, claims, 0, 0)
    assert stop is None
    assert [item["entry_id"] for item in batch] == ["a"]
    assert detail == "batch_shrunk_to_fit"
    batch, stop, detail = next_claim_batch(reviewer, state, claims, 1, 1000)
    assert batch == []
    assert stop == "token_allowance"
    assert detail == "single_claim_exceeds_remaining_allowance"
    oversized = _SizedReviewer(max_tokens=5000, token_budget=900, one=1000, two=2000)
    batch, stop, detail = next_claim_batch(oversized, state, claims[:1], 0, 0)
    assert batch == []
    assert stop == "request_budget"
    assert detail == "single_claim_exceeds_request_budget"


def test_llm_review_sends_the_claim_that_fits(monkeypatch):
    def estimate(self, payload, schema):
        if payload.get("kind") != "focused_claims":
            return 1
        count = len(payload.get("claims") or [])
        return 1000 if count <= 1 else 1427

    monkeypatch.setattr(DecisionReviewer, "estimate", estimate)
    result = focused_code(
        _source(
            passages=[_passage("p1", "I spent the morning packing my suitcase.")],
            codes=[_entry("c1", "leaving home", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=FakeFocusedCoder(),
        decision_mode="llm",
        decision_model=FakeDecisionModel(raised=["c1"], support="supported"),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=1,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        max_focused_review_calls=1,
        max_decision_tokens=1011,
        decision_token_budget=6000,
        interrupt=True,
    )
    reviewed = [
        row
        for row in result.properties + result.dimensions
        if row.get("model_claim_support")
    ]
    assert len(reviewed) == 1
    assert result.metadata["decision_calls_used"] == 2
    assert result.metadata["decision_stop"] == "token_allowance"
    assert result.metadata["decision_stop_detail"] == "single_claim_exceeds_remaining_allowance"
    assert result.metadata["decision_pending_claim_ids"]


def test_claims_payload_includes_challenging_passages_within_a_budget():
    state = {
        "metadata": {},
        "codebook": {},
        "passages": [
            {"passage_id": "p1", "doc_id": "d", "text": "alpha", "question": "q"},
            {"passage_id": "p2", "doc_id": "d", "text": "bravo " * 50, "question": "q"},
            {"passage_id": "p3", "doc_id": "d", "text": "charlie", "question": "q"},
        ],
    }
    claim = {
        "entry_id": "c",
        "passage_id": "p1",
        "cited_passage_id": "p1",
        "citations": [{"passage_id": "p1"}],
        "negative_cases": [{"passage_id": "p2"}, {"passage_id": "p9"}],
        "claim_tests": [{"passage_id": "p3"}],
    }
    payload = claims_payload(state, [claim], evidence_char_budget=20)
    included = [row["passage_id"] for row in payload["passages"]]
    omitted = {row["passage_id"]: row["reason"] for row in payload["omitted_context"]}
    assert included[0] == "p1"
    assert "p3" in included
    assert "p2" not in included
    assert omitted["p2"] == "evidence_budget"
    assert omitted["p9"] == "missing_passage"
    assert payload["evidence_char_budget"] == 20


def test_token_adjustment_is_labeled_as_empirical():
    text = "abcd" * 32
    heuristic = estimate_tokens(text)
    adjusted = adjusted_token_estimate(heuristic)
    assert adjusted == max(1, int(round(heuristic / 1.28)))
    assert "not a general calibration" in TOKEN_ESTIMATE_ADJUSTMENT_NOTE
    assert "input tokens" in TOKEN_ESTIMATE_ADJUSTMENT_NOTE
