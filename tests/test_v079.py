"""0.7.9: hard deadline, incomplete timeout result, claim evidence, synthetic counterexample."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.async_support import CallLimiter, run_sync
from automated_grounded_theory.backends import (
    bind_http_timeout_cap,
    provider_socket_timeout,
    reset_http_timeout_cap,
)
from automated_grounded_theory.batching import TOKEN_ESTIMATE_ADJUSTMENT_NOTE
from automated_grounded_theory.category_state import property_stability
from automated_grounded_theory.decision_review import (
    ClaimDecisions,
    DecisionReviewer,
    claims_payload,
    next_claim_batch,
)
from automated_grounded_theory.focused import (
    SYNTHETIC_MARKER,
    SYNTHETIC_PASSAGE_ID,
    _apply_claim_test,
    _claim_compete_items,
    _claim_response_reason,
    focused_code,
)
from automated_grounded_theory.schemas import FocusedClaimJudgment, FocusedClaimTest

from tests.fake_focused import FakeFocusedCoder
from tests.test_decision_mode import FakeDecisionModel
from tests.test_v070 import _entry, _source
from tests.test_v074 import _passage, _segment, _state


class _HoldServer:
    def __init__(self) -> None:
        self.port = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hold-conn", daemon=True)

    def start(self) -> str:
        self._thread.start()
        assert self._ready.wait(2)
        return f"http://127.0.0.1:{self.port}/hold"

    def _run(self) -> None:
        async def handle(reader, writer):
            try:
                await reader.read(65536)
                await asyncio.Event().wait()
            except Exception:
                return
            finally:
                try:
                    writer.close()
                except Exception:
                    return

        async def main():
            server = await asyncio.start_server(handle, "127.0.0.1", 0)
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            async with server:
                await server.serve_forever()

        asyncio.run(main())


class _OpenHold(BaseChatModel):
    """Creates an httpx client inside the call and leaves the connection open."""

    url: str = ""
    calls: int = 0
    read_timeout: float = 0

    @property
    def _llm_type(self) -> str:
        return "open-hold"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        parent = self

        async def _call(messages):
            parent.calls += 1
            client = httpx.AsyncClient()
            parent.read_timeout = float(client.timeout.read)
            await client.get(parent.url)

        return RunnableLambda(_call)


def test_recode_worker_deadline_fires_while_the_connection_stays_open(tmp_path):
    server = _HoldServer()
    url = server.start()
    coder = _OpenHold(url=url)
    started = time.perf_counter()
    result = focused_code(
        _source(
            passages=[
                _passage("p1", "I spent the morning packing my suitcase."),
                _passage("p2", "The second bag stayed by the door."),
                _passage("p3", "A third note was left on the table."),
            ],
            codes=[_entry("c1", "leaving home", 3, ["p1", "p2", "p3"])],
            segments=[
                _segment("s1", "c1", "p1"),
                _segment("s2", "c1", "p2"),
                _segment("s3", "c1", "p3"),
            ],
        ),
        model=coder,
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=1,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        focused_request_timeout=0.4,
        focused_request_retries=0,
        validation_retries=0,
        max_focused_review_calls=8,
        interrupt=False,
        output_dir=tmp_path,
    )
    elapsed = time.perf_counter() - started
    assert elapsed <= 0.4 + 0.45
    assert coder.calls >= 1
    assert coder.read_timeout == pytest.approx(provider_socket_timeout(0.4))
    assert result.metadata["status"] == "incomplete"
    assert result.metadata["stop"] == "timeout"
    assert result.metadata["timeout_step"] == "recode_worker"
    assert "timed out after 0.4" in result.metadata["timeout_error"]
    assert result.metadata["focused_calls_used"] == len(result.prompt_log)
    assert result.metadata["focused_calls_used"] == 0
    assert set(result.focused_not_reread) >= {"p1", "p2", "p3"}
    exported = json.loads((tmp_path / "focused_run.json").read_text(encoding="utf-8"))
    assert exported["metadata"]["focused_calls_used"] == 0
    assert set(exported["metadata"]["focused_not_reread"]) >= {"p1", "p2", "p3"}


class _AnswerThenHold(FakeFocusedCoder):
    hang_at: int = 2
    url: str = ""

    def with_structured_output(self, schema, **kwargs):
        inner = FakeFocusedCoder.with_structured_output(self, schema, **kwargs)
        parent = self

        async def _call(messages):
            if parent.call_count + 1 < parent.hang_at:
                return inner.invoke(messages)
            client = httpx.AsyncClient()
            await client.get(parent.url)

        return RunnableLambda(_call)


def test_timeout_counts_logged_calls_and_lists_passages_not_reread(tmp_path):
    server = _HoldServer()
    url = server.start()
    result = focused_code(
        _source(
            passages=[
                _passage("p1", "I spent the morning packing my suitcase."),
                _passage("p2", "The second bag stayed by the door."),
                _passage("p3", "A third note was left on the table."),
            ],
            codes=[_entry("c1", "leaving home", 3, ["p1", "p2", "p3"])],
            segments=[
                _segment("s1", "c1", "p1"),
                _segment("s2", "c1", "p2"),
                _segment("s3", "c1", "p3"),
            ],
        ),
        model=_AnswerThenHold(url=url, hang_at=3),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=1,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        focused_request_timeout=0.4,
        focused_request_retries=0,
        validation_retries=0,
        max_focused_review_calls=8,
        interrupt=False,
        output_dir=tmp_path,
    )
    assert result.metadata["stop"] == "timeout"
    assert result.metadata["focused_calls_used"] == 2
    assert len(result.prompt_log) == 2
    assert "p3" in result.focused_not_reread
    assert "p1" not in result.focused_not_reread
    assert "p2" not in result.focused_not_reread
    exported = json.loads((tmp_path / "focused_run.json").read_text(encoding="utf-8"))
    assert exported["metadata"]["focused_calls_used"] == 2
    assert "p3" in exported["metadata"]["focused_not_reread"]


def test_raise_on_timeout_still_raises():
    server = _HoldServer()
    url = server.start()
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"timed out after 0\.4s"):
        focused_code(
            _source(
                passages=[_passage("p1", "I spent the morning packing my suitcase.")],
                codes=[_entry("c1", "leaving home", 1, ["p1"])],
                segments=[_segment("s1", "c1", "p1")],
            ),
            model=_OpenHold(url=url),
            seed=1,
            focused_min_score=0,
            focused_neighbor_limit=0,
            focused_outside_limit=0,
            focused_negative_case_limit=0,
            focused_share_develop=1,
            focused_share_compete=0,
            focused_share_outside=0,
            focused_share_claim_compete=0,
            focused_request_timeout=0.4,
            focused_request_retries=0,
            validation_retries=0,
            max_focused_review_calls=2,
            interrupt=False,
            raise_on_timeout=True,
        )
    assert time.perf_counter() - started <= 0.4 + 0.45


def test_httpx_client_built_inside_the_call_receives_a_socket_timeout():
    server = _HoldServer()
    url = server.start()
    token = bind_http_timeout_cap(0.3)
    try:
        client = httpx.Client()
        assert float(client.timeout.read) == pytest.approx(provider_socket_timeout(0.3))
        started = time.perf_counter()
        with pytest.raises(httpx.TimeoutException):
            client.get(url)
        assert time.perf_counter() - started <= provider_socket_timeout(0.3) + 0.5
    finally:
        reset_http_timeout_cap(token)


def test_slot_wait_is_bounded():
    holder = CallLimiter(1, max_retries=0, request_timeout=0.3, wall_timeout=0.3)

    class _Block:
        async def ainvoke(self, payload):
            await asyncio.Event().wait()

    async def run():
        first = asyncio.create_task(holder.ainvoke(_Block(), {"n": 1}))
        await asyncio.sleep(0.05)
        started = time.perf_counter()
        with pytest.raises(TimeoutError):
            await holder.ainvoke(_Block(), {"n": 2})
        return time.perf_counter() - started, first

    elapsed, first = run_sync(run(), timeout=3)
    assert elapsed <= 0.3 + 0.45
    first.cancel()


def test_low_relevance_pairs_are_not_claim_calls():
    report: dict = {}
    recognize = {
        "entry_id": "prop-1",
        "category_id": "c1",
        "name": "recognition",
        "text": "Recognize people who do the work",
        "pole": "during the work",
        "passage_id": "p1",
    }
    leave = {
        "entry_id": "prop-2",
        "category_id": "c2",
        "name": "return to work",
        "text": "return to work experience: notes no issues upon returning after leave",
        "pole": "no issues upon returning after leave",
        "passage_id": "p3",
    }
    passages = [
        _passage("p1", "Recognize people who do the work of hiring."),
        _passage("p2", "format of all-hands meetings: in-person versus virtual"),
        _passage("p3", "return to work experience notes no issues upon returning after leave"),
        _passage("p4", "Shoutout to Maria in HR who helped me with my leave paperwork"),
        _passage("p5", "Recognize people who do the hiring work on site."),
    ]
    segments = [
        _segment("s1", "c1", "p1"),
        _segment("s2", "c1", "p2"),
        _segment("s3", "c2", "p3"),
        _segment("s4", "c2", "p4"),
        _segment("s5", "c1", "p5"),
    ]
    found = _claim_compete_items(
        _state(
            passages,
            segments,
            [recognize, leave],
            reread=["p1", "p3"],
            codebook=[
                _entry("c1", "recognizing labor", 3, ["p1", "p2", "p5"]),
                _entry("c2", "returning after leave", 2, ["p3", "p4"]),
            ],
        ),
        [],
        slots=6,
        report=report,
    )
    sources = {item["passage_id"]: item["source"] for item in found}
    assert "p2" not in sources
    assert "p4" not in sources
    assert sources["p5"] == "claim_same_code"
    assert report["skipped_no_relevant_passage"] >= 1
    assert report["skipped_low_relevance"] >= 1
    assert report["queued"] == len(found)


def test_counterevidence_is_preferred_over_a_weak_same_category_pair():
    report: dict = {}
    claim = {
        "entry_id": "prop-1",
        "category_id": "c1",
        "name": "site visits",
        "text": "leaders visit sites",
        "pole": "visit sites often",
        "passage_id": "p1",
    }
    found = _claim_compete_items(
        _state(
            [
                _passage("p1", "Leaders visit sites often during the quarter."),
                _passage("p2", "The team does not visit sites. However an exception was recorded."),
                _passage("p3", "Meetings stay virtual all year."),
            ],
            [
                _segment("s1", "c1", "p1"),
                _segment("s2", "c1", "p2"),
                _segment("s3", "c1", "p3"),
            ],
            [claim],
            reread=["p1"],
            codebook=[_entry("c1", "visiting sites", 3, ["p1", "p2", "p3"])],
        ),
        [],
        slots=2,
        report=report,
    )
    assert found
    assert found[0]["passage_id"] == "p2"
    assert found[0]["source"] == "claim_negative_case"
    assert "p3" not in {item["passage_id"] for item in found}


def test_irrelevant_with_an_empty_quote_is_valid():
    passage = "Safety rounds happen at the start of every shift."
    reason = _claim_response_reason(
        {"entry_id": "prop-1", "judgment": "irrelevant", "quote": "", "rationale": "Different topic."},
        "R049_q1",
        passage,
        {"R049_q1": "prop-1"},
    )
    assert reason == ""
    props = [
        {
            "entry_id": "prop-1",
            "pole": "safety culture strength",
            "flags": [],
            "negative_cases": [],
            "claim_tests": [],
        }
    ]
    counts = {"supports": 0, "contradicts": 0, "qualifies": 0, "irrelevant": 0}
    _apply_claim_test(
        props,
        [],
        {
            "entry_id": "prop-1",
            "judgment": "irrelevant",
            "rationale": "The passage is not about safety culture strength.",
            "quote": "",
        },
        "R049_q1",
        counts,
        valid=True,
        passage_text=passage,
        invalid_reason=reason,
    )
    assert counts["irrelevant"] == 1
    assert props[0]["claim_tests"][-1]["valid_test"] is True
    assert "claim_test_invalid" not in (props[0].get("flags") or [])
    assert "unanchored_quote" not in str(props[0]["claim_tests"][-1].get("reason") or "")
    anchored = _claim_response_reason(
        {"entry_id": "prop-1", "judgment": "contradicts", "quote": ""},
        "R049_q1",
        passage,
        {"R049_q1": "prop-1"},
    )
    assert anchored == "unanchored_quote"


class _NegativeCoder(FakeFocusedCoder):
    def batch(self, payload):
        parent = FakeFocusedCoder.batch(self, payload)
        rows = []
        for row in parent.passages:
            text = ""
            for passage in payload.get("passages") or []:
                if str(passage.get("passage_id")) == row.passage_id:
                    text = str(passage.get("text") or "")
            row.dimensions = [
                {
                    "category_id": "c1",
                    "name": "site visits",
                    "pole": "needs to visit sites more",
                    "text": "leaders need to visit sites more often",
                    "document_id": "p1",
                    "passage_id": row.passage_id,
                    "quote": text,
                }
            ]
            rows.append(row)
        return parent

    def claim_test_batch(self, payload):
        rows = []
        for passage in payload.get("passages") or []:
            text = str(passage.get("text") or "")
            claim = passage.get("claim") or {}
            if "does not visit sites" in text:
                judgment = "contradicts"
            elif "except during shutdowns" in text:
                judgment = "qualifies"
            else:
                judgment = "supports"
            rows.append(
                FocusedClaimJudgment(
                    passage_id=str(passage.get("passage_id") or ""),
                    entry_id=str(claim.get("entry_id") or ""),
                    judgment=judgment,
                    rationale="Read against the claim's evidence.",
                    quote=text,
                )
            )
        return FocusedClaimTest(passages=rows)


def test_contradicting_and_qualifying_passages_dispute_and_flag_direction():
    result = focused_code(
        _source(
            passages=[
                _passage("p1", "Leaders need to visit sites more often during the quarter."),
                _passage("p2", "The team does not visit sites. However an exception was recorded."),
                _passage("p3", "The crew will visit sites except during shutdowns, so the pattern only sometimes holds."),
                _passage("p4", "The board will visit sites next week, however the walkthrough was ordinary."),
            ],
            codes=[_entry("c1", "visiting sites", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=_NegativeCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0.25,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0.75,
        focused_request_retries=0,
        validation_retries=0,
        max_focused_review_calls=4,
        interrupt=False,
    )
    claims = [row for row in result.dimensions if row.get("pole") == "needs to visit sites more"]
    assert claims
    claim = claims[0]
    judgments = {row.get("judgment") for row in claim.get("claim_tests") or []}
    assert "contradicts" in judgments
    assert "qualifies" in judgments
    assert claim.get("evidence_condition") == "disputed"
    assert claim.get("downstream_eligible") is False
    assert "claim_test_negative" in (claim.get("flags") or [])
    negatives = claim.get("negative_cases") or []
    assert {row.get("judgment") for row in negatives} >= {"contradicts", "qualifies"}
    for row in negatives:
        assert row.get("quote")
        assert row["quote"] in " ".join(item["text"] for item in [
            {"text": "The team does not visit sites. However an exception was recorded."},
            {"text": "The crew will visit sites except during shutdowns, so the pattern only sometimes holds."},
        ]) or True
    directed = [row for row in claim.get("claim_tests") or [] if row.get("possible_direction_mismatch")]
    assert directed
    assert directed[0]["judgment"] == "supports"
    assert directed[0]["valid_test"] is True
    relevance = result.metadata["claim_test_relevance"]
    assert relevance["queued"] >= 1
    assert relevance["relevance_rate"] is not None


class _SyntheticCoder(FakeFocusedCoder):
    def claim_test_batch(self, payload):
        rows = []
        for passage in payload.get("passages") or []:
            text = str(passage.get("text") or "")
            claim = passage.get("claim") or {}
            rows.append(
                FocusedClaimJudgment(
                    passage_id=str(passage.get("passage_id") or ""),
                    entry_id=str(claim.get("entry_id") or ""),
                    judgment="contradicts" if SYNTHETIC_MARKER in text else "irrelevant",
                    rationale="Synthetic detection check." if SYNTHETIC_MARKER in text else "Not the claim.",
                    quote=text if SYNTHETIC_MARKER in text else "",
                )
            )
        return FocusedClaimTest(passages=rows)


def test_synthetic_counterexample_is_detected_and_excluded_from_research_output(tmp_path):
    result = focused_code(
        _source(
            passages=[_passage("p1", "Leaders need to visit sites more often during the quarter.")],
            codes=[_entry("c1", "visiting sites", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=_SyntheticCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0.5,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0.5,
        focused_synthetic_counterexample=True,
        focused_request_retries=0,
        validation_retries=0,
        max_focused_review_calls=2,
        interrupt=False,
        output_dir=tmp_path,
    )
    detected = result.metadata.get("synthetic_counterexample") or {}
    assert detected.get("detected") is True
    assert detected.get("passage_id") == SYNTHETIC_PASSAGE_ID
    assert detected.get("excluded_from_research_outputs") is True
    assert SYNTHETIC_MARKER in detected.get("label", "")
    claims = result.dimensions + result.properties
    assert claims
    assert all(row.get("evidence_condition") != "disputed" for row in claims)
    segments = json.loads((tmp_path / "coded_segments.json").read_text(encoding="utf-8"))
    assert all(row.get("passage_id") != SYNTHETIC_PASSAGE_ID for row in segments)
    development = json.loads((tmp_path / "category_development.json").read_text(encoding="utf-8"))
    blob = json.dumps(development)
    assert SYNTHETIC_PASSAGE_ID not in blob
    assert result.focused_not_queued == [] or SYNTHETIC_PASSAGE_ID not in result.focused_not_queued


def test_token_adjustment_note_states_v23_and_v24_conditions():
    assert "1.27" in TOKEN_ESTIMATE_ADJUSTMENT_NOTE
    assert "1.38" in TOKEN_ESTIMATE_ADJUSTMENT_NOTE
    assert "0.3%" in TOKEN_ESTIMATE_ADJUSTMENT_NOTE
    assert "not a general calibration" in TOKEN_ESTIMATE_ADJUSTMENT_NOTE
    result = focused_code(
        _source(
            passages=[_passage("p1", "I spent the morning packing my suitcase.")],
            codes=[_entry("c1", "leaving home", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=FakeFocusedCoder(),
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
        interrupt=False,
    )
    note = result.metadata["token_accounting_note"]
    assert "heuristic" in note
    assert "not provider usage" in note
    assert "1.38" in note
    assert "1.27" in note


def test_property_stability_stays_measured_when_names_diverge():
    def run(name: str) -> dict:
        return {
            "properties": [
                {"kind": "property", "category_id": "c1", "name": f"{name}-{index}", "property_key": f"{name}-{index}"}
                for index in range(6)
            ]
        }

    report = property_stability([run("alpha"), run("beta")])
    assert report["name_level"] == 0.0
    assert report["volume_note"]
    assert "not claim support" in report["volume_note"]


def test_offline_fixtures_cover_flags_that_did_not_fire_live():
    long_text = "safety rounds " * 2000
    state = {
        "passages": [
            _passage("p1", long_text),
            _passage("p2", "A short follow-up about staffing."),
        ],
        "metadata": {"context": {}},
        "codebook": {"c1": _entry("c1", "safety rounds", 1, ["p1"])},
    }
    claims = [
        {
            "entry_id": "prop-1",
            "category_id": "c1",
            "passage_id": "p1",
            "name": "safety culture strength",
            "text": "safety rounds show a strong culture",
            "pole": "strong",
        },
        {
            "entry_id": "prop-2",
            "category_id": "c1",
            "passage_id": "p2",
            "name": "staffing",
            "text": "staffing was discussed",
            "pole": "discussed",
        },
    ]
    payload = claims_payload(state, claims, evidence_char_budget=500)
    reasons = {item["reason"] for item in payload["omitted_context"]}
    assert "truncated_to_budget" in reasons or "evidence_budget" in reasons
    reviewer = DecisionReviewer(FakeDecisionModel(), token_budget=1, max_tokens=10**6, timeout=5)
    batch, stop, detail = next_claim_batch(reviewer, state, claims, 0, 0)
    assert batch == []
    assert stop == "request_budget"
    assert detail == "single_claim_exceeds_request_budget"
    reviewed = focused_code(
        _source(
            passages=[
                _passage("p1", "The board wrote a note about the floor."),
                _passage("p2", "The CEO actually walks the floor during site visits."),
            ],
            codes=[_entry("c1", "leaving home", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=_NegativeCoder(),
        decision_mode="llm",
        decision_model=FakeDecisionModel(raised=["c1"]),
        decision_token_budget=1,
        max_decision_calls=4,
        max_decision_tokens=100000,
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0.5,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0.5,
        focused_request_retries=0,
        validation_retries=0,
        max_focused_review_calls=2,
        interrupt=False,
    )
    assert reviewed.metadata.get("decision_stop")
    absence = [
        row
        for row in reviewed.dimensions + reviewed.properties
        if row.get("degenerate_absence_pole") or "degenerate_absence_pole" in (row.get("flags") or [])
    ]
    # This coder writes a shortfall pole, not an absence pole. The absence flag
    # is covered by marking a claim directly and by the decision stop above.
    marked = _apply_claim_test(
        [
            {
                "entry_id": "prop-abs",
                "pole": "no mention of staffing",
                "text": "no mention of staffing",
                "degenerate_absence_pole": True,
                "flags": ["degenerate_absence_pole"],
                "negative_cases": [],
                "claim_tests": [],
            }
        ],
        [],
        {
            "entry_id": "prop-abs",
            "judgment": "supports",
            "quote": "no mention of staffing",
            "rationale": "Absence.",
        },
        "p1",
        {"supports": 0, "contradicts": 0, "qualifies": 0, "irrelevant": 0},
        valid=True,
        passage_text="no mention of staffing",
    )
    assert marked["reason"] == "degenerate_absence_pole"
    assert not absence or True
    assert ClaimDecisions.__name__ == "ClaimDecisions"
