"""0.8.0 axial relationships, saturation indicators, and selective coding.

No network calls. The hanging provider holds a local socket open.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    __version__,
    axial_code,
    get_graph,
    read_axial_interrupt,
    relation_stability,
    selective_code,
    thematic_saturation_report,
)
from automated_grounded_theory.axial import (
    SYNTHETIC_PASSAGE_ID,
    AxialCheckpointMismatch,
    AxialError,
    AxialProposalBatch,
    AxialRelationshipDraft,
    _apply_counterexample,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.saturation import saturation_report


STATEMENT = "Packing the suitcase before departure shapes the return to work."
QUOTE = "Packing the suitcase took all night"


class _HoldServer:
    def __init__(self) -> None:
        self.port = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="axial-hold", daemon=True)

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


def _payload(messages) -> dict:
    for message in reversed(list(messages)):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.startswith("{"):
            return json.loads(content)
    return {}


class FakeAxialCoder(BaseChatModel):
    """Schema-shaped axial responses. ``mode`` selects a fixture. No network."""

    call_count: int = 0
    mode: str = "grounded"
    core_id: str = ""

    @property
    def _llm_type(self) -> str:
        return "fake-axial"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "AxialProposalBatch":
                parsed = self.proposals(body)
            elif name == "AxialCounterexampleBatch":
                parsed = self.counterexamples(body)
            elif name == "TheoreticalMemoBatch":
                parsed = self.memos(body)
            elif name == "StorylineBatch":
                parsed = self.storyline(body)
            elif name == "AxialRelationDecisions":
                parsed = self.relation_review(body)
            elif name == "ModelSaturationDecision":
                parsed = schema(assessment="not_saturated", rationale="A later document still adds a code.")
            elif name == "ModelCoreDecision":
                chosen = self.core_id or str((body.get("candidates") or [{}])[0].get("code_id") or "")
                parsed = schema(core_id=chosen, rationale="It is one candidate. Alternatives stay in the list.")
            else:
                parsed = schema()
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)

    def proposals(self, body: dict) -> AxialProposalBatch:
        pair = (body.get("pairs") or [{"from_id": "c-prep", "to_id": "c-return"}])[0]
        rows = [
            AxialRelationshipDraft(
                from_id=str(pair["from_id"]),
                to_id=str(pair["to_id"]),
                paradigm_role="causal_condition",
                statement=STATEMENT,
                passage_id="p1",
                quote=QUOTE,
            ),
            AxialRelationshipDraft(
                from_id="not-a-code",
                to_id="also-not",
                statement="This pair was not requested.",
                passage_id="p1",
                quote=QUOTE,
            ),
            AxialRelationshipDraft(
                from_id=str(pair["from_id"]),
                to_id=str(pair["to_id"]),
                statement=STATEMENT,
                passage_id="missing-passage",
                quote=QUOTE,
            ),
        ]
        return AxialProposalBatch(relationships=rows)

    def counterexamples(self, body: dict):
        from automated_grounded_theory.axial import AxialCounterexample, AxialCounterexampleBatch

        rows = []
        for item in body.get("counterexamples") or []:
            passage = item.get("passage") or {}
            text = str(passage.get("text") or "")
            folded = text.casefold()
            if "synthetic counterexample" in folded or "however" in folded:
                quote = "not in the passage" if self.mode == "bad_quote" else (
                    "the reported pattern does not hold" if "synthetic counterexample" in folded else "had no issues however"
                )
                judgment = "contradicts"
            elif "monday" in folded:
                judgment = "irrelevant"
                quote = ""
            else:
                judgment = "supports"
                quote = text
            rows.append(
                AxialCounterexample(
                    relationship_id=str((item.get("relationship") or {}).get("relationship_id") or ""),
                    passage_id=str(passage.get("passage_id") or ""),
                    judgment=judgment,
                    quote=quote,
                    rationale="Fixture verdict.",
                )
            )
        return AxialCounterexampleBatch(counterexamples=rows)

    def memos(self, body: dict):
        from automated_grounded_theory.axial import TheoreticalMemoBatch, TheoreticalMemoDraft

        drafts = []
        for row in body.get("relationships") or []:
            text = "" if self.mode == "empty_memo" else "Packing is a condition of returning."
            drafts.append(TheoreticalMemoDraft(relationship_id=str(row.get("relationship_id") or ""), text=text))
        return TheoreticalMemoBatch(memos=drafts)

    def storyline(self, body: dict):
        del body
        from automated_grounded_theory.axial import StorylineBatch, StorylineStatementDraft

        return StorylineBatch(
            statements=[
                StorylineStatementDraft(
                    text=STATEMENT,
                    category_ids=["c-prep", "c-return"],
                    relationship_ids=["rel-0001"],
                    passage_id="p1",
                    quote=QUOTE,
                ),
                StorylineStatementDraft(text="This statement has no citation.", category_ids=["c-prep"]),
            ]
        )

    def relation_review(self, body: dict):
        from automated_grounded_theory.axial import AxialRelationDecision, AxialRelationDecisions

        return AxialRelationDecisions(
            relationships=[
                AxialRelationDecision(
                    relationship_id=str(row.get("relationship_id") or ""),
                    disposition="accepted",
                    paradigm_role="context",
                    rationale="The cited quote supports a condition.",
                )
                for row in body.get("relationships") or []
            ]
        )


class _OpenHold(FakeAxialCoder):
    url: str = ""
    read_timeout: float = 0

    def with_structured_output(self, schema, **kwargs):
        parent = self

        async def _call(messages):
            parent.call_count += 1
            client = httpx.AsyncClient()
            parent.read_timeout = float(client.timeout.read)
            await client.get(parent.url)

        return RunnableLambda(_call)


def _code(code_id: str, label: str) -> dict:
    return {"code_id": code_id, "code": label, "definition": label}


def _corpus(*extra_passages, documents=("d1", "d2")) -> dict:
    passages = [
        {
            "passage_id": "p1",
            "doc_id": documents[0],
            "text": "Packing the suitcase took all night before the trip.",
            "embedding": [1.0, 0.0],
        },
        {
            "passage_id": "p2",
            "doc_id": documents[-1],
            "text": "The return to work after leave had no issues however the schedule changed.",
        },
        {
            "passage_id": "p-low",
            "doc_id": documents[-1],
            "text": "A single suitcase sat by the door.",
        },
        {
            "passage_id": "p-none",
            "doc_id": documents[-1],
            "text": "Format of the all-hands meeting was virtual.",
        },
        {
            "passage_id": "p-embed",
            "doc_id": documents[-1],
            "text": "qqqq zzzz vvvv",
            "embedding": [1.0, 0.0],
        },
        {
            "passage_id": "p-monday",
            "doc_id": documents[-1],
            "text": "The return to work waited until Monday.",
        },
    ]
    passages.extend(extra_passages)
    return {
        "codebook": {
            "c-prep": _code("c-prep", "preparing to leave"),
            "c-return": _code("c-return", "returning to work"),
        },
        "passages": passages,
        "segments": [
            {"segment_id": "s1", "code_id": "c-prep", "passage_id": "p1"},
            {"segment_id": "s2", "code_id": "c-return", "passage_id": "p2"},
        ],
        "properties": [
            {"entry_id": "prop-1", "property_key": "timing", "document_id": documents[0], "category_id": "c-prep"}
        ],
        "dimensions": [],
        "memos": [],
    }


def _run(model=None, **kwargs):
    coder = model or FakeAxialCoder(mode=str(kwargs.pop("mode", "grounded")))
    result = axial_code(
        _corpus(),
        coder,
        seed=3,
        max_axial_review_calls=4,
        batch_size=1,
        **kwargs,
    )
    return coder, result


def test_version_and_graphs():
    assert __version__ == "0.8.2"
    assert POLICY_VERSION == "0.8.2"
    from automated_grounded_theory.axial import aaxial_code

    assert inspect.signature(aaxial_code).parameters["batch_size"].default == 1
    axial = get_graph("axial")
    selective = get_graph("selective")
    assert "propose_relations" in axial.nodes
    assert "counterexample_worker" in axial.nodes
    assert "core_checkpoint" not in axial.nodes
    assert "draft_storyline" in selective.nodes
    assert "propose_relations" not in get_graph("open_code").nodes
    assert "propose_relations" not in get_graph("focused").nodes
    with pytest.raises(ValueError, match="axial"):
        get_graph("missing")
    assert saturation_report([]).get("judgment") is None


def test_indicators_are_undefined_without_enough_data():
    empty = thematic_saturation_report([])
    assert empty["judgment"] is None
    assert empty["new_code_rate"] is None
    assert empty["code_frequency_stabilization"] is None
    assert "undefined" in empty["note"]
    assert relation_stability([{"relationships": []}])["role_level"] is None
    assert relation_stability(
        [
            {"relationships": [{"from_id": "a", "to_id": "b", "paradigm_role": "context"}]},
            {"relationships": [{"from_id": "a", "to_id": "b", "paradigm_role": "context"}]},
        ]
    )["role_level"] == 1.0
    coder = FakeAxialCoder()
    result = axial_code(
        _corpus(documents=("d1", "d1")),
        coder,
        seed=3,
        axial_relations=False,
        interrupt=False,
        max_axial_review_calls=4,
    )
    assert coder.call_count == 0
    assert result.saturation["judgment"] is None
    assert result.saturation["code_frequency_stabilization"] is None
    assert "undefined" in result.saturation["note"]
    assert result.saturation["property_name_level"] is None
    assert result.coverage["citation_anchoring_rate"] is None
    assert result.coverage["relationships_with_negative_case_search"] is None
    assert "batch_size" not in result.metadata["procedure"]
    assert result.metadata["performance"]["batch_size"] == 1
    two = axial_code(
        _corpus(documents=("d1", "d2")),
        FakeAxialCoder(),
        seed=3,
        axial_relations=False,
        interrupt=False,
        max_axial_review_calls=4,
    )
    assert isinstance(two.saturation["code_frequency_stabilization"], float)
    assert two.saturation["judgment"] is None
    assert "undefined" in two.saturation["note"]


def test_human_checkpoints_accept_and_record_saturation(tmp_path):
    saver = InMemorySaver()
    coder = FakeAxialCoder()
    common = dict(
        model=coder,
        seed=3,
        max_axial_review_calls=4,
        batch_size=1,
        checkpointer=saver,
        thread_id="axial-human",
        output_dir=tmp_path,
    )
    paused = axial_code(_corpus(), interrupt=True, **common)
    assert paused.interrupted
    assert paused.interrupt_payload["kind"] == "axial_relations"
    assert paused.relationships[0]["claim_support"] == "undefined"
    assert paused.relationships[0]["disposition"] == "proposed"
    assert paused.relationships[0]["evidence_condition"] == "anchored"
    logged = json.dumps(paused.prompt_log)
    assert "unrequested_pair" in logged
    assert "unknown_passage" in logged
    held = read_axial_interrupt(saver, "axial-human")
    assert held["payload"]["kind"] == "axial_relations"
    with pytest.raises(AxialError, match="empty object"):
        axial_code(_corpus(), resuming=True, resume={}, interrupt=True, **common)
    changed = dict(common)
    changed["seed"] = 9
    with pytest.raises(AxialCheckpointMismatch):
        axial_code(_corpus(), resuming=True, resume={"accept": True}, allow_changes=True, interrupt=True, **changed)
    resumed = dict(common)
    resumed["batch_size"] = 4
    reviewed = axial_code(
        _corpus(),
        resuming=True,
        resume={"relationships": [{"relationship_id": "rel-0001", "action": "accept", "paradigm_role": "context"}]},
        interrupt=True,
        **resumed,
    )
    assert reviewed.interrupt_payload["kind"] == "thematic_saturation"
    row = reviewed.relationships[0]
    assert row["human_disposition"] == "accepted"
    assert row["human_paradigm_role"] == "context"
    assert row["claim_support"] == "undefined"
    assert row["evidence_condition"] == "disputed"
    assert row["negative_case_search"] is True
    assert row["negative_cases"]
    assert row["negative_cases"][0]["quote"] == "had no issues however"
    irrelevant = [item for item in row["counterexamples"] if item["judgment"] == "irrelevant"]
    assert irrelevant and irrelevant[0]["quote"] == "" and irrelevant[0]["valid_test"] is True
    assert "relationship_negative" in row["flags"]
    relevance = reviewed.metadata["axial_counterexample_relevance"]
    assert relevance["skipped_low_relevance"] >= 1
    assert relevance["skipped_no_relevant_passage"] >= 1
    assert reviewed.metadata["procedure"]["reallocated_propose_calls"] is True
    assert any("embedding" in json.dumps(entry) for entry in reviewed.prompt_log)
    empty_roles = {item["paradigm_role"] for item in reviewed.empty_slots if item["category_id"] == "c-prep"}
    assert "context" not in empty_roles
    assert "consequence" in empty_roles
    assert all(item["flag"] == "paradigm_slot_empty" for item in reviewed.empty_slots)
    assert isinstance(reviewed.saturation["code_frequency_stabilization"], float)
    assert reviewed.saturation["judgment"] is None
    finished = axial_code(
        _corpus(),
        resuming=True,
        resume={"decision": "not_saturated", "rationale": "The second document still adds a code."},
        interrupt=True,
        **common,
    )
    assert finished.interrupted is False
    assert finished.saturation["judgment"] is None
    assert finished.saturation["researcher_saturation"]["actor"] == "human"
    assert finished.saturation["researcher_saturation"]["decision"] == "not_saturated"
    assert finished.saturation["model_saturation"] is None
    assert finished.metadata["status"] == "complete"
    exported = json.loads((tmp_path / "category_relationships.json").read_text(encoding="utf-8"))
    assert exported["relationships"][0]["evidence_condition"] == "disputed"
    saturation = json.loads((tmp_path / "thematic_saturation.json").read_text(encoding="utf-8"))
    assert saturation["judgment"] is None
    assert "undefined" in saturation["note"]
    coverage = json.loads((tmp_path / "theory_coverage.json").read_text(encoding="utf-8"))
    assert coverage["disputed_relationships"] == 1
    assert coverage["relationships_with_negative_case_search"] == 1.0


def test_reject_revise_and_merge_keep_both_rows():
    saver = InMemorySaver()
    source = _corpus()
    source["codebook"]["c-other"] = _code("c-other", "other work")
    common = dict(model=FakeAxialCoder(), seed=3, max_axial_review_calls=4, checkpointer=saver, thread_id="axial-merge")
    paused = axial_code(source, interrupt=True, **common)
    assert len(paused.relationships) == 2
    merged = axial_code(
        source,
        resuming=True,
        resume={
            "relationships": [
                {
                    "relationship_id": "rel-0002",
                    "action": "merge",
                    "merge_into": "rel-0001",
                    "label": "clearer: packing shapes return",
                },
            ]
        },
        interrupt=True,
        **common,
    )
    by_id = {row["relationship_id"]: row for row in merged.relationships}
    assert set(by_id) == {"rel-0001", "rel-0002"}
    assert by_id["rel-0001"]["statement"] == "clearer: packing shapes return"
    assert by_id["rel-0002"]["disposition"] == "merged"
    assert by_id["rel-0002"]["merged_into"] == "rel-0001"
    assert by_id["rel-0002"]["downstream_eligible"] is False
    assert "relationship_merged" in by_id["rel-0001"]["flags"]
    assert "relationship_merged" in by_id["rel-0002"]["flags"]
    saver2 = InMemorySaver()
    paused2 = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=5,
        max_axial_review_calls=4,
        checkpointer=saver2,
        thread_id="axial-revise",
        interrupt=True,
    )
    updated = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=5,
        max_axial_review_calls=4,
        checkpointer=saver2,
        thread_id="axial-revise",
        resuming=True,
        resume={
            "relationships": [
                {
                    "relationship_id": "rel-0001",
                    "action": "revise",
                    "quote": "not in the passage",
                    "statement": "A revised condition.",
                }
            ]
        },
        interrupt=True,
    )
    row = updated.relationships[0]
    assert row["human_disposition"] == "revised"
    assert row["evidence_condition"] == "unanchored"
    assert "unanchored_quote" in row["flags"]
    assert row["statement"] == "A revised condition."
    assert paused2.relationships[0]["relationship_id"] == row["relationship_id"]
    reject_saver = InMemorySaver()
    axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=6,
        max_axial_review_calls=4,
        checkpointer=reject_saver,
        thread_id="axial-reject",
        interrupt=True,
    )
    kept = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=6,
        max_axial_review_calls=4,
        checkpointer=reject_saver,
        thread_id="axial-reject",
        resuming=True,
        resume={"relationships": [{"relationship_id": "rel-0001", "action": "reject"}]},
        interrupt=True,
    )
    assert kept.relationships[0]["disposition"] == "rejected"
    assert kept.relationships[0]["human_disposition"] == "rejected"
    assert "relationship_rejected" in kept.relationships[0]["flags"]


def test_llm_review_keeps_human_fields_undefined():
    coder, result = _run(decision_mode="llm", interrupt=True)
    assert result.interrupted is False
    row = result.relationships[0]
    assert row["model_disposition"] == "accepted"
    assert row["model_paradigm_role"] == "context"
    assert row["human_disposition"] is None
    assert row["human_paradigm_role"] is None
    assert row["claim_support"] == "undefined"
    assert row["review_actor"] == "llm"
    audits = [entry for entry in result.prompt_log if entry.get("kind") == "decision_review"]
    assert {entry["checkpoint"] for entry in audits} >= {"axial_relation_review", "axial_saturation_review"}
    assert all(entry["actor"] == "llm" and entry["status"] == "validated" for entry in audits)
    assert result.saturation["judgment"] is None
    assert result.saturation["researcher_saturation"] is None
    assert result.saturation["model_saturation"]["model_judgment"] is True
    assert result.saturation["model_saturation"]["actor"] == "llm"
    assert result.metadata.get("researcher_saturation") is None
    assert coder.call_count >= 2


def test_bad_anchor_does_not_dispute_and_synthetic_is_excluded(tmp_path):
    _coder, invalid = _run(mode="bad_quote", interrupt=False)
    row = invalid.relationships[0]
    assert row["evidence_condition"] != "disputed"
    assert row["negative_cases"] == []
    assert "axial_test_invalid" in row["flags"]
    assert any(item["reason"] == "unanchored_quote" for item in row["counterexamples"])
    coder, result = _run(axial_synthetic_counterexample=True, interrupt=False, output_dir=tmp_path)
    assert result.metadata["synthetic_counterexample"]["detected"] is True
    assert result.metadata["synthetic_counterexample"]["excluded_from_research_outputs"] is True
    assert all(item["passage_id"] != SYNTHETIC_PASSAGE_ID for item in result.passages)
    exported = (tmp_path / "category_relationships.json").read_text(encoding="utf-8")
    assert SYNTHETIC_PASSAGE_ID not in exported
    assert "[synthetic counterexample]" not in exported
    audit = json.loads((tmp_path / "axial_run.json").read_text(encoding="utf-8"))
    assert audit["metadata"]["synthetic_counterexample"]["label"] == "[synthetic counterexample]"
    assert coder.call_count >= 1
    relationship = {"relationship_id": "rel-0001", "flags": [], "counterexamples": [], "negative_cases": []}
    _apply_counterexample(
        relationship,
        {"judgment": "irrelevant", "quote": ""},
        {"passage_id": "p-monday", "text": "The return to work waited until Monday."},
    )
    assert relationship.get("evidence_condition") != "disputed"
    assert relationship["counterexamples"][0]["valid_test"] is True
    assert relationship["negative_cases"] == []


def test_selective_human_and_llm_stay_separate(tmp_path):
    source = _corpus()
    source["relationships"] = [
        {
            "relationship_id": "rel-0001",
            "from_id": "c-prep",
            "to_id": "c-return",
            "paradigm_role": "causal_condition",
            "statement": STATEMENT,
            "passage_id": "p1",
            "quote": QUOTE,
            "evidence_condition": "anchored",
            "disposition": "accepted",
            "human_disposition": "accepted",
            "claim_support": "undefined",
            "negative_case_search": True,
            "flags": [],
        }
    ]
    saver = InMemorySaver()
    paused = selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        include_memos=True,
        include_storyline=True,
        checkpointer=saver,
        thread_id="core",
        interrupt=True,
        output_dir=tmp_path,
    )
    assert paused.interrupt_payload["kind"] == "selective_core"
    assert {row["code_id"] for row in paused.core_candidates} == {"c-prep", "c-return"}
    assert paused.researcher_core_id == ""
    finished = selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        include_memos=True,
        include_storyline=True,
        checkpointer=saver,
        thread_id="core",
        resuming=True,
        resume={"core_id": "c-prep", "rationale": "Preparing to leave organizes the return."},
        interrupt=True,
        output_dir=tmp_path,
    )
    assert finished.researcher_core_id == "c-prep"
    assert finished.model_core_id == ""
    assert {row["code_id"] for row in finished.core_candidates} == {"c-prep", "c-return"}
    assert any(row["role"] == "theoretical_memo" and row["text"] for row in finished.memos)
    cited = [row for row in finished.storyline if row["evidence_condition"] == "anchored"]
    uncited = [row for row in finished.storyline if "storyline_uncited" in row["flags"]]
    assert cited and uncited
    assert finished.coverage["unsupported_statements"] >= 1
    assert finished.coverage["citation_anchoring_rate"] is not None
    assert (tmp_path / "storyline.json").exists()
    assert (tmp_path / "theoretical_memos.json").exists()
    assert (tmp_path / "core_candidates.json").exists()
    empty_memo = selective_code(
        source,
        FakeAxialCoder(mode="empty_memo"),
        seed=3,
        include_memos=True,
        interrupt=False,
    )
    assert empty_memo.memos[0]["flags"] == ["memo_empty"]
    assert empty_memo.memos[0]["text"] == ""
    llm = selective_code(source, FakeAxialCoder(core_id="missing-core"), seed=3, decision_mode="llm", interrupt=True)
    assert llm.interrupted is False
    assert llm.model_core_id == "missing-core"
    assert llm.researcher_core_id == ""
    assert llm.metadata["model_core_judgment"] is True
    assert llm.metadata["model_core_unknown"] is True
    assert any("not a candidate" in note for note in llm.warnings)
    core_saver = InMemorySaver()
    selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        checkpointer=core_saver,
        thread_id="empty-core",
        interrupt=True,
    )
    with pytest.raises(AxialError, match="empty object"):
        selective_code(
            source,
            FakeAxialCoder(),
            seed=3,
            checkpointer=core_saver,
            thread_id="empty-core",
            resuming=True,
            resume={},
        )


def test_timeout_returns_an_incomplete_result(tmp_path):
    from automated_grounded_theory.backends import provider_socket_timeout

    server = _HoldServer()
    url = server.start()
    coder = _OpenHold(url=url)
    started = time.perf_counter()
    result = axial_code(
        _corpus(),
        coder,
        seed=3,
        max_axial_review_calls=2,
        axial_request_timeout=0.4,
        interrupt=False,
        output_dir=tmp_path,
    )
    elapsed = time.perf_counter() - started
    assert elapsed <= 0.4 + 0.5
    assert coder.call_count >= 1
    assert coder.read_timeout == pytest.approx(provider_socket_timeout(0.4), abs=1e-4)
    assert result.metadata["status"] == "incomplete"
    assert result.metadata["stop"] == "timeout"
    assert result.metadata["timeout_step"] == "propose_relations"
    assert "timed out after 0.4" in result.metadata["timeout_error"]
    assert result.saturation["judgment"] is None
    exported = json.loads((tmp_path / "axial_run.json").read_text(encoding="utf-8"))
    assert exported["metadata"]["status"] == "incomplete"
    assert (tmp_path / "category_relationships.json").exists()
    assert (tmp_path / "thematic_saturation.json").exists()
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"timed out after 0\.4s"):
        axial_code(
            _corpus(),
            _OpenHold(url=url),
            seed=3,
            max_axial_review_calls=2,
            axial_request_timeout=0.4,
            interrupt=False,
            raise_on_timeout=True,
        )
    assert time.perf_counter() - started <= 0.4 + 0.5


def test_charmaz_lens_does_not_fill_paradigm_slots():
    _coder, result = _run(coding_lens="charmaz", interrupt=False, axial_relations=True)
    assert result.empty_slots == []
    with pytest.raises(AxialError):
        axial_code(_corpus(), FakeAxialCoder(), coding_lens="glaser")
    with pytest.raises(AxialError):
        axial_code(_corpus(), FakeAxialCoder(), decision_mode="human", decision_model=FakeAxialCoder())
