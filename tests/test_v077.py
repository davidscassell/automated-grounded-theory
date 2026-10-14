"""0.7.7 focused-coding fixes: silent reads, claim relevance, proposal conflicts."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory.async_support import CallLimiter, run_sync
from automated_grounded_theory.batching import calibrated_token_estimate, estimate_tokens
from automated_grounded_theory.category_state import property_stability, proposal_stability
from automated_grounded_theory.focused import (
    _absence_pole,
    _apply_claim_test,
    _claim_compete_items,
    _pole_direction_mismatch,
    build_focused_graph,
    focused_code,
)
from automated_grounded_theory.schemas import (
    FocusedClaimJudgment,
    FocusedClaimTest,
    FocusedCodeDecision,
    FocusedDimensionNote,
    FocusedPassageRecode,
    FocusedRecodeBatch,
)

from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _entry, _source
from tests.test_v074 import _passage, _segment, _state


class _SilentServer:
    def __init__(self) -> None:
        self.port = 0
        self._ready = threading.Event()
        self._loop = None
        self._thread = threading.Thread(target=self._run, name="silent-body", daemon=True)

    def start(self) -> str:
        self._thread.start()
        assert self._ready.wait(2)
        return f"http://127.0.0.1:{self.port}/hang"

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
            self._loop = asyncio.get_running_loop()
            self._ready.set()
            async with server:
                await server.serve_forever()

        asyncio.run(main())


def test_a_silent_body_on_the_shared_loop_keeps_the_request_timeout():
    server = _SilentServer()
    url = server.start()

    class SilentBody:
        def __init__(self):
            self.connection_open = False

        async def ainvoke(self, payload):
            self.connection_open = True
            async with httpx.AsyncClient(timeout=None) as client:
                await client.get(url)

    client = SilentBody()
    limiter = CallLimiter(1, max_retries=4, backoff=0.5, max_backoff=8, request_timeout=0.4, wall_timeout=0.4)
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"timed out after 0\.4s") as caught:
        run_sync(limiter.ainvoke(client, {"call": "silent"}), timeout=3)
    elapsed = time.perf_counter() - started
    assert elapsed <= 1.2
    assert "sync bridge" not in str(caught.value)
    assert "3s" not in str(caught.value)
    assert client.connection_open is True


class _SilentCoder(BaseChatModel):
    url: str = ""
    connection_open: bool = False

    @property
    def _llm_type(self) -> str:
        return "silent-body"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        parent = self

        async def _call(messages):
            parent.connection_open = True
            async with httpx.AsyncClient(timeout=None) as client:
                await client.get(parent.url)

        return RunnableLambda(_call)


def test_a_silent_graph_step_records_the_step_and_a_partial_export(tmp_path):
    server = _SilentServer()
    url = server.start()
    saver = InMemorySaver()
    coder = _SilentCoder(url=url)
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"timed out after 0\.4s") as caught:
        focused_code(
            _source(
                passages=[_passage("p1", "I spent the morning packing my suitcase.")],
                codes=[_entry("c1", "leaving home", 1, ["p1"])],
                segments=[_segment("s1", "c1", "p1")],
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
            focused_request_retries=4,
            max_focused_review_calls=8,
            interrupt=False,
            checkpointer=saver,
            thread_id="timeout-077",
            output_dir=tmp_path,
        )
    elapsed = time.perf_counter() - started
    assert elapsed <= 1.5
    message = str(caught.value)
    assert "sync bridge" not in message
    assert "4.2" not in message
    assert coder.connection_open is True
    snapshot = run_sync(
        build_focused_graph().compile(checkpointer=saver).aget_state(
            {"configurable": {"thread_id": "timeout-077"}}
        )
    )
    values = dict(snapshot.values or {})
    assert values.get("recode_stop") == "timeout"
    assert values["metadata"]["status"] == "incomplete"
    assert values["metadata"]["stop"] == "timeout"
    assert values["metadata"]["complete"] is False
    assert values["metadata"]["timeout_step"] == "recode_worker"
    assert "0.4" in values["metadata"]["timeout_error"]
    exported = json.loads((tmp_path / "focused_run.json").read_text(encoding="utf-8"))
    assert exported["metadata"]["stop"] == "timeout"
    assert exported["metadata"]["status"] == "incomplete"
    assert exported["metadata"]["timeout_step"] == "recode_worker"
    assert (tmp_path / "coded_segments.json").exists()


def test_same_code_without_shared_content_is_not_a_claim_test():
    report: dict = {}
    passages = [
        _passage("p1", "Recognize people who do the work of hiring."),
        _passage("p2", "Virtual meetings are not the same as being in the room."),
        _passage("p3", "Recognize people who do the hiring work on site."),
        _passage("p4", ""),
    ]
    claims = [
        {
            "entry_id": "prop-1",
            "category_id": "c1",
            "name": "recognition",
            "text": "Recognize people who do the work",
            "pole": "during the work",
            "passage_id": "p1",
        }
    ]
    segments = [
        _segment("s1", "c1", "p1"),
        _segment("s2", "c1", "p2"),
        _segment("s3", "c1", "p3"),
        _segment("s4", "c1", "p4"),
    ]
    found = _claim_compete_items(
        _state(passages, segments, claims, reread=["p1"], codebook=[_entry("c1", "recognizing labor", 3, ["p1", "p2", "p3"])]),
        [],
        slots=4,
        report=report,
    )
    sources = {item["passage_id"]: item["source"] for item in found}
    assert "p2" not in sources
    assert sources["p3"] == "claim_same_code"
    assert "p4" not in sources
    assert report["empty_quote_skipped"] >= 1
    assert report["skipped_no_relevant_passage"] >= 1
    assert any("empty quote" in note for note in report["notes"])
    assert any("no shared content" in note for note in report["notes"])


def test_supports_against_a_gap_pole_is_claim_test_invalid():
    assert _pole_direction_mismatch(
        "needs to visit the sites more and actually listen",
        "The CEO actually walks the floor.",
    )
    assert not _pole_direction_mismatch("during", "The CEO actually walks the floor.")
    assert not _absence_pole("zzzz-absent")
    assert _absence_pole("not discussed in the interview")
    assert _absence_pole("absent from the account")
    assert _absence_pole("no evidence of a site visit")
    props = [
        {
            "entry_id": "prop-1",
            "pole": "needs to visit the sites more and actually listen",
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
            "judgment": "supports",
            "rationale": "The passage shows presence.",
            "quote": "The CEO actually walks the floor.",
        },
        "p2",
        counts,
        valid=True,
        passage_text="The CEO actually walks the floor.",
    )
    assert counts["supports"] == 1
    assert "claim_test_invalid" in props[0]["flags"]
    assert props[0]["claim_tests"][-1]["direction_mismatch"] is True
    assert props[0]["claim_tests"][-1]["valid_test"] is False
    assert props[0].get("evidence_condition") != "disputed"


class _PoleCoder(FakeFocusedCoder):
    pole: str = "not discussed in the interview"
    claim_text: str = "Leadership presence was not discussed in the interview."

    def batch(self, payload):
        raised = list(payload.get("raised_codes") or [])
        code = raised[0] if raised else {"code_id": "c1", "code": "leaving home", "definition": ""}
        rows = []
        for passage in payload.get("passages") or []:
            text = str(passage.get("text") or "")
            quote = text[:80] or "quote"
            rows.append(
                FocusedPassageRecode(
                    passage_id=str(passage.get("passage_id") or ""),
                    codes=[
                        FocusedCodeDecision(
                            action="keep",
                            code=str(code.get("code") or ""),
                            code_id=str(code.get("code_id") or ""),
                            quote=quote,
                            definition=str(code.get("definition") or ""),
                            rationale="Kept.",
                        )
                    ],
                    dimensions=[
                        FocusedDimensionNote(
                            category_id=str(code.get("code_id") or ""),
                            name="site presence",
                            pole=self.pole,
                            text=self.claim_text,
                            document_id=str(passage.get("doc_id") or ""),
                            passage_id=str(passage.get("passage_id") or ""),
                            quote=quote,
                        )
                    ],
                )
            )
        return FocusedRecodeBatch(passages=rows)

    def claim_test_batch(self, payload):
        rows = []
        for passage in payload.get("passages") or []:
            claim = passage.get("claim") or {}
            text = str(passage.get("text") or "")
            rows.append(
                FocusedClaimJudgment(
                    passage_id=str(passage.get("passage_id") or ""),
                    entry_id=str(claim.get("entry_id") or ""),
                    judgment="supports",
                    rationale="The passage was read as support.",
                    quote=text[:80],
                )
            )
        return FocusedClaimTest(passages=rows)


def test_absence_pole_and_direction_mismatch_fire_on_a_run():
    absence = focused_code(
        _source(
            passages=[_passage("p1", "I spent the morning packing my suitcase.")],
            codes=[_entry("c1", "leaving home", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=_PoleCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0.5,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0.5,
        max_focused_review_calls=2,
        interrupt=False,
    )
    flagged = [row for row in absence.dimensions if "degenerate_absence_pole" in (row.get("flags") or [])]
    assert flagged
    assert flagged[0]["needs_review"] is True
    assert flagged[0]["downstream_eligible"] is False
    assert not any(row.get("claim_tests") for row in absence.dimensions)

    contrast = focused_code(
        _source(
            passages=[
                _passage("p1", "The board wrote a note about the floor."),
                _passage("p2", "The CEO actually walks the floor during site visits."),
            ],
            codes=[_entry("c1", "leaving home", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=_PoleCoder(
            pole="needs to visit the sites more and actually listen",
            claim_text="Leadership on the floor is a site visit.",
        ),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0.5,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0.5,
        max_focused_review_calls=4,
        interrupt=False,
    )
    invalid = [row for row in contrast.dimensions if "claim_test_invalid" in (row.get("flags") or [])]
    assert invalid
    assert invalid[0]["claim_tests"][-1]["judgment"] == "supports"
    assert invalid[0]["claim_tests"][-1]["direction_mismatch"] is True
    assert invalid[0].get("evidence_condition") != "disputed"
    assert invalid[0].get("claim_support", "undefined") == "undefined"
    relevance = contrast.metadata["claim_test_relevance"]
    assert relevance["judgments"] >= 1
    assert relevance["relevance_rate"] == 0.0


def test_proposal_stability_reads_exports_and_surfaces_revise_drop_conflicts(tmp_path):
    segments = [
        {"segment_id": "s1", "code_id": "c1", "passage_id": "p1", "quote": "hiring"},
        {"segment_id": "s2", "code_id": "c1", "passage_id": "p2", "quote": "promotion"},
    ]
    assert proposal_stability([segments, segments])["code_level"] is None
    revise = {
        "segments": [
            {
                "code_id": "c-hire",
                "passage_id": "R023_q2",
                "focused_revise_proposal": {
                    "definition": "advocating for changes in hiring and promotion",
                    "actor": "model",
                },
            }
        ]
    }
    drop = {
        "segments": [
            {
                "code_id": "c-hire",
                "passage_id": "R023_q2",
                "focused_drop_proposal": {"definition": "", "actor": "model"},
            }
        ]
    }
    compared = proposal_stability([revise, drop])
    assert compared["code_level"] == 1.0
    assert compared["key_level"] == 1.0
    assert compared["cross_seed_conflicts"]
    conflict = compared["cross_seed_conflicts"][0]
    assert conflict["code_id"] == "c-hire"
    assert conflict["passage_id"] == "R023_q2"
    assert conflict["kinds"] == ["drop", "revise"]
    place = tmp_path / "run"
    place.mkdir()
    (place / "coded_segments.json").write_text(json.dumps(revise["segments"]), encoding="utf-8")
    loaded = proposal_stability([str(place), revise])
    assert loaded["code_level"] == 1.0
    assert loaded["key_level"] == 1.0
    with pytest.raises(TypeError):
        proposal_stability([1, 2])


def test_property_stability_reports_volume_when_names_diverge():
    left = [{"category_id": "c1", "name": f"alpha {index}", "kind": "property"} for index in range(8)]
    right = [{"category_id": "c1", "name": f"beta {index}", "kind": "property"} for index in range(8)]
    scored = property_stability([left, right])
    assert scored["name_level"] == 0.0
    assert scored["name_counts"] == [8, 8]
    assert scored["properties_per_run"] == [8, 8]
    assert "different property names" in scored["volume_note"]
    assert scored["name_level"] is not None


def test_calibrated_token_estimate_is_not_the_ceiling():
    text = "abcd" * 32
    ceiling = estimate_tokens(text)
    calibrated = calibrated_token_estimate(ceiling)
    assert ceiling == max(1, (len(text) + 2) // 3)
    assert calibrated < ceiling
    assert calibrated == max(1, int(round(ceiling / 1.28)))
