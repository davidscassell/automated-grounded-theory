"""0.8.1 axial review fixes and v25 live-eval corrections. No network calls."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest

from automated_grounded_theory import axial_code, thematic_saturation_report
from automated_grounded_theory.async_support import CallLimiter, run_sync
from automated_grounded_theory.axial import (
    _allocate_evidence,
    _procedure_from_kwargs,
    _windows_from_state,
)
from automated_grounded_theory.backends import (
    CrossLoopClientError,
    ProviderClientFactory,
    client_for_running_loop,
)
from automated_grounded_theory.focused import _apply_claim_test, _fit_unspent_claim_calls
from tests.test_v080 import FakeAxialCoder, _corpus


def _procedure(**overrides):
    book = {f"c{index}": {"code": str(index)} for index in range(overrides.pop("categories", 5))}
    kwargs = dict(
        seed=0,
        coding_lens="paradigm",
        paradigm_version="1990",
        min_relation_documents=2,
        max_axial_review_calls=8,
        axial_share_propose=0.5,
        obvious_share=0.5,
        axial_relations=True,
        include_memos=False,
        include_storyline=False,
        axial_synthetic_counterexample=False,
        decision_mode="human",
        max_selective_review_calls=4,
        batch_size=3,
        max_axial_review_tokens=24000,
    )
    kwargs.update(overrides)
    return _procedure_from_kwargs({"codebook": book, "passages": []}, **kwargs)


def test_frequency_shift_is_total_variation_not_presence():
    report = thematic_saturation_report(
        [
            {
                "document_id": "d1",
                "new_codes": 2,
                "code_frequencies": {"A": 99, "B": 1},
                "cumulative_codes": 2,
            },
            {
                "document_id": "d2",
                "new_codes": 0,
                "code_frequencies": {"A": 1, "B": 99},
                "cumulative_codes": 2,
            },
        ]
    )
    assert report["code_frequency_comparison"] == "total_variation"
    assert report["points"][1]["new_codes"] == 0
    assert report["code_frequency_stabilization"] == pytest.approx(0.98)
    assert report["code_frequency_stabilization"] != 0.0
    assert "not comparable" in report["note"]
    assert report["judgment"] is None
    empty = thematic_saturation_report(
        [
            {"document_id": "d1", "new_codes": 0, "code_frequencies": {}},
            {"document_id": "d2", "new_codes": 1, "code_frequencies": {"A": 4}},
        ]
    )
    assert empty["code_frequency_stabilization"] is None
    assert "undefined" in empty["note"]
    state = {
        "passages": [
            {"passage_id": "p1", "doc_id": "d1", "text": "a"},
            {"passage_id": "p2", "doc_id": "d2", "text": "b"},
        ],
        "segments": [{"code_id": "A", "passage_id": "p1"}] * 99
        + [{"code_id": "B", "passage_id": "p1"}]
        + [{"code_id": "A", "passage_id": "p2"}]
        + [{"code_id": "B", "passage_id": "p2"}] * 99,
        "properties": [],
        "dimensions": [],
        "relationships": [],
    }
    windows = _windows_from_state(state)
    assert windows[0]["code_frequencies"] == {"A": 99, "B": 1}
    assert windows[1]["code_frequencies"] == {"A": 1, "B": 99}
    assert windows[1]["new_codes"] == 0
    measured = thematic_saturation_report(windows)
    assert measured["code_frequency_stabilization"] == pytest.approx(0.98)


def test_call_budget_uses_ceil_of_pairs_and_never_goes_negative():
    procedure = _procedure()
    assert procedure["propose_calls"] == 2
    assert procedure["statement_copy_retry_calls"] == 2
    assert procedure["counterexample_calls"] == 4
    assert (
        procedure["propose_calls"]
        + procedure["statement_copy_retry_calls"]
        + procedure["counterexample_calls"]
    ) == 8
    assert procedure["reallocated_propose_calls"] is False
    odd = _procedure(max_axial_review_calls=7)
    assert odd["propose_calls"] >= 0
    assert odd["counterexample_calls"] >= 0
    assert (
        odd["propose_calls"] + odd["statement_copy_retry_calls"] + odd["counterexample_calls"]
    ) == 7
    small = _procedure(categories=2, batch_size=3, max_axial_review_calls=8)
    assert small["propose_calls"] == 1
    assert small["statement_copy_retry_calls"] == 1
    assert small["counterexample_calls"] == 6
    assert small["reallocated_propose_calls"] is True
    with pytest.raises(Exception, match="zero or more"):
        _procedure(max_axial_review_calls=-1)


def test_zero_decision_calls_do_not_review_or_interrupt():
    coder = FakeAxialCoder()
    result = axial_code(
        _corpus(),
        coder,
        seed=3,
        max_axial_review_calls=4,
        decision_mode="llm",
        max_decision_calls=0,
        interrupt=True,
    )
    assert result.interrupted is False
    assert result.metadata["decision_stop"] == "call_allowance"
    assert not any(row.get("kind") == "decision_review" for row in result.prompt_log)
    assert result.saturation.get("model_saturation") is None
    with pytest.raises(Exception, match="zero or more"):
        axial_code(_corpus(), FakeAxialCoder(), decision_mode="llm", max_decision_calls=-1, interrupt=False)


def test_evidence_is_shared_across_both_categories():
    assigned = {"c-prep": [f"p{index}" for index in range(8)], "c-return": ["r1"]}
    passages = {
        passage_id: {"passage_id": passage_id, "doc_id": "d", "text": passage_id}
        for passage_id in [*assigned["c-prep"], "r1"]
    }
    rows, omitted = _allocate_evidence(
        [{"from_id": "c-prep", "to_id": "c-return"}],
        assigned,
        passages,
        8,
    )
    by_id = {row["passage_id"]: row for row in rows}
    assert "r1" in by_id
    assert "c-return" in by_id["r1"]["code_ids"]
    assert any("c-prep" in row["code_ids"] for row in rows)
    assert omitted["omitted_count"] == 1
    assert omitted["limit"] == 8


def test_combined_selective_keeps_the_axial_audit(tmp_path):
    result = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=3,
        max_axial_review_calls=4,
        decision_mode="llm",
        interrupt=False,
        also_selective=True,
        include_storyline=True,
        output_dir=tmp_path,
    )
    assert result.metadata["stage"] == "selective"
    assert result.metadata["axial_procedure"]["coding_lens"] == "paradigm"
    assert result.saturation["model_saturation"]["assessment"] == "not_saturated"
    assert result.saturation["model_saturation"]["model_judgment"] is True
    kinds = [row.get("kind") for row in result.prompt_log]
    assert "axial_proposal" in kinds
    assert "storyline" in kinds
    assert "decision_review" in kinds
    story = next(row for row in result.prompt_log if row.get("kind") == "storyline")
    body = json.loads(story["messages"][-1]["content"])
    assert body["core_id"]
    assert body["core_provenance"] == "model"
    assert body["core_id"] == body["model_core_id"]
    assert story["usage"]["input_tokens"] == 30
    assert result.metadata["provider_token_usage"]["reported_calls"] >= 1
    axial_run = json.loads((tmp_path / "axial_run.json").read_text(encoding="utf-8"))
    assert axial_run["metadata"]["stage"] == "axial"
    assert any(row.get("kind") == "axial_proposal" for row in axial_run["prompt_log"])
    assert "storyline" not in {row.get("kind") for row in axial_run["prompt_log"]}
    saturation = json.loads((tmp_path / "thematic_saturation.json").read_text(encoding="utf-8"))
    assert saturation["model_saturation"]["assessment"] == "not_saturated"
    relationships = json.loads((tmp_path / "category_relationships.json").read_text(encoding="utf-8"))
    assert relationships["relationships"]
    selective = json.loads((tmp_path / "selective_run.json").read_text(encoding="utf-8"))
    selective_kinds = {row.get("kind") for row in selective["prompt_log"]}
    assert "axial_proposal" in selective_kinds
    assert "storyline" in selective_kinds


def test_factory_clients_keep_separate_auth_and_foreign_clients_fail():
    async def inner():
        other = asyncio.new_event_loop()
        clients = []
        try:
            factory_a = ProviderClientFactory(
                auth=httpx.BasicAuth("alice", "secret"),
                cookies={"session": "aaa"},
                trust_env=False,
                base_url="http://127.0.0.1",
            )
            factory_b = ProviderClientFactory(
                auth=httpx.BasicAuth("bob", "other"),
                cookies={"session": "bbb"},
                trust_env=False,
                params={"lane": "b"},
                base_url="http://127.0.0.1",
            )
            first_a = factory_a.build(timeout=httpx.Timeout(5.0), loop=other)
            first_b = factory_b.build(timeout=httpx.Timeout(5.0), loop=other)
            clients.extend((first_a, first_b))
            repl_a = client_for_running_loop(first_a, httpx.Timeout(5.0))
            repl_b = client_for_running_loop(first_b, httpx.Timeout(5.0))
            clients.extend((repl_a, repl_b))
            assert repl_a is not repl_b
            assert repl_a is not first_a
            assert repl_a.auth._auth_header != repl_b.auth._auth_header
            assert repl_a.cookies["session"] == "aaa"
            assert repl_b.cookies["session"] == "bbb"
            assert repl_a.trust_env is False
            assert repl_b.trust_env is False
            assert dict(repl_b.params)["lane"] == "b"
            raw = httpx.AsyncClient(auth=httpx.BasicAuth("x", "y"), cookies={"session": "raw"}, trust_env=False)
            clients.append(raw)
            setattr(raw, "_agt_owner_loop", other)
            with pytest.raises(CrossLoopClientError):
                client_for_running_loop(raw, httpx.Timeout(5.0))
        finally:
            for client in clients:
                await client.aclose()
            other.close()

    asyncio.run(inner())


def test_pre_call_stall_and_open_connection_hit_the_request_deadline():
    ready = threading.Event()
    accepted = threading.Event()
    port = {}

    def serve():
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port["n"] = sock.getsockname()[1]
        ready.set()
        try:
            conn, _addr = sock.accept()
        except OSError:
            return
        accepted.set()
        try:
            time.sleep(30)
        finally:
            conn.close()
            sock.close()

    threading.Thread(target=serve, daemon=True).start()
    assert ready.wait(2)

    class _StallBeforeCall(CallLimiter):
        async def _reserve_budgets(self, record, estimate, *, deadline=None):
            self.opened = socket.create_connection(("127.0.0.1", port["n"]), timeout=2)
            time.sleep(30)

    limiter = _StallBeforeCall(1, max_retries=0, request_timeout=0.4, wall_timeout=0.4)
    called = {"model": False}

    class _Never:
        async def ainvoke(self, payload):
            called["model"] = True
            return "sent"

    started = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"timed out after 0\.4s"):
        run_sync(limiter.ainvoke(_Never(), {"call": "before"}, use_cache=False), timeout=5)
    assert time.perf_counter() - started < 0.4 + 1.6
    assert called["model"] is False
    assert accepted.wait(1)

    class _HoldOpen:
        def __init__(self):
            self.opened = False

        async def ainvoke(self, payload):
            self.opened = True
            reader, writer = await asyncio.open_connection("127.0.0.1", port["n"])
            del reader
            try:
                await asyncio.Event().wait()
            finally:
                writer.close()

    holder = _HoldOpen()
    open_limiter = CallLimiter(1, max_retries=0, request_timeout=0.4, wall_timeout=0.4)
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"timed out after 0\.4s"):
        run_sync(open_limiter.ainvoke(holder, {"call": "open"}, use_cache=False), timeout=5)
    assert time.perf_counter() - started < 0.4 + 1.6
    assert holder.opened is True


def test_ops_team_paraphrase_is_an_uncertain_dispute_not_irrelevance():
    props = [
        {
            "entry_id": "R039_q1",
            "name": "specificity of coworker reference",
            "pole": "the passage uses only the generic term coworkers",
            "flags": [],
            "negative_cases": [],
            "claim_tests": [],
            "evidence_condition": "anchored",
        }
    ]
    counts = {"supports": 0, "contradicts": 0, "qualifies": 0, "irrelevant": 0}
    quote = "The ops team is scrappy and gets things done without a lot of red tape"
    _apply_claim_test(
        props,
        [],
        {
            "entry_id": "R039_q1",
            "judgment": "contradicts",
            "quote": quote,
            "rationale": "The passage is about the ops team.",
        },
        "R039_q1",
        counts,
        valid=True,
        passage_text=quote,
    )
    assert props[0]["evidence_condition"] == "disputed"
    assert props[0]["negative_cases"]
    assert props[0]["negative_cases"][-1]["claim_subject_uncertain"] is True
    assert props[0]["needs_review"] is True
    assert "claim_subject_uncertain" in props[0]["flags"]
    assert "claim_test_invalid" not in props[0]["flags"]
    assert "rationale_indicates_irrelevance" not in props[0]["flags"]
    uncertain = [
        {
            "entry_id": "R039_q1",
            "name": "specificity of coworker reference",
            "pole": "the passage uses only the generic term coworkers",
            "flags": [],
            "negative_cases": [],
            "claim_tests": [],
            "evidence_condition": "anchored",
        }
    ]
    _apply_claim_test(
        uncertain,
        [],
        {
            "entry_id": "R039_q1",
            "judgment": "contradicts",
            "quote": "The ops team is scrappy and gets things done without a lot of red tape",
            "rationale": "This passage is a negative case for the supplied claim.",
        },
        "p-uncertain",
        counts,
        valid=True,
        passage_text=quote,
    )
    assert uncertain[0]["evidence_condition"] == "disputed"
    assert uncertain[0]["negative_cases"][-1]["claim_subject_uncertain"] is True
    engaged = [
        {
            "entry_id": "R039_q1",
            "name": "specificity of coworker reference",
            "pole": "the passage uses only the generic term coworkers",
            "flags": [],
            "negative_cases": [],
            "claim_tests": [],
        }
    ]
    _apply_claim_test(
        engaged,
        [],
        {
            "entry_id": "R039_q1",
            "judgment": "contradicts",
            "quote": "coworkers here are named, not generic",
            "rationale": "The quote names coworkers specifically.",
        },
        "p1",
        counts,
        valid=True,
        passage_text="coworkers here are named, not generic",
    )
    assert engaged[0]["evidence_condition"] == "disputed"
    assert engaged[0]["negative_cases"][-1]["valid_test"] is True


def test_reallocation_stays_inside_the_token_allowance():
    passage = {"passage_id": "p1", "text": "alpha beta gamma delta " * 80}
    state = {
        "passages": [passage],
        "segments": [],
        "codebook": {"c1": {"code_id": "c1", "code": "leaving", "definition": "leaving home"}},
        "raised_code_ids": ["c1"],
        "properties": [],
        "dimensions": [],
        "focused_tokens_estimated": 0,
        "metadata": {"procedure": {"max_focused_review_tokens": 1}, "performance": {}},
    }
    blocked = _fit_unspent_claim_calls(state, calls=3, batch_size=2)
    assert blocked["estimate_per_call"] > 1
    assert blocked["moved_calls"] == 0
    assert blocked["not_moved_calls"] == 3
    assert blocked["reason"] == "token_allowance"
    assert blocked["moved"] == []
    roomy = {
        **state,
        "metadata": {"procedure": {"max_focused_review_tokens": 10**9}, "performance": {}},
    }
    moved = _fit_unspent_claim_calls(roomy, calls=2, batch_size=2)
    assert moved["moved_calls"] == 2
    assert moved["not_moved_calls"] == 0
    assert len(moved["moved"]) == 4
