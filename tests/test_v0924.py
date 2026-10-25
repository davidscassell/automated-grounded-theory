"""0.9.24 decision-review chunking. Fakes only. No network calls."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    DECISION_REVIEW_CHUNKING,
    DecisionChunkingCheckpointError,
    axial_code,
    focused_code,
    merge_core_nominations,
    pack_decision_chunks,
    selective_code,
    selection_payload,
)
from automated_grounded_theory.axial import (
    DEFAULT_MAX_AXIAL_REVIEW_TOKENS,
    AxialRelationDecision,
    AxialRelationDecisions,
    aaxial_code,
)
from automated_grounded_theory.decision_review import (
    CHUNKING_CHECKPOINT_MESSAGE,
    DecisionReviewer,
    SelectionDecision,
    build_decision_plan,
    finish_decision_report,
    load_chunk_progress,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES
from automated_grounded_theory.results import OpenCodeResult
from automated_grounded_theory.version import __version__
from automated_grounded_theory.prompts import message_text

from tests.fake_focused import FakeFocusedCoder
from tests.test_v080 import FakeAxialCoder, _corpus


def test_version_and_defaults_stay_put():
    assert __version__ == "0.9.28"
    assert POLICY_VERSION == "0.9.28"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v5"
    assert PATTERN_MIN_CASES == 2
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000
    import inspect

    focused = inspect.signature(focused_code).parameters
    axial = inspect.signature(aaxial_code).parameters
    assert focused["batch_size"].default == 1
    assert axial["batch_size"].default == 1
    assert focused["decision_token_budget"].default == 6000
    assert axial["axial_token_budget"].default == 6000
    assert axial["max_axial_review_tokens"].default == 24000


class _KeepSelector(BaseChatModel):
    """Raises every code whose definition contains KEEP. No network."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "keep-selector"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.calls += 1
            name = getattr(schema, "__name__", "")
            text = "\n".join(message_text(message) for message in messages)
            if name == "SelectionDecision":
                payload = json.loads(text[text.find("{") :])
                raised = [
                    str(row.get("code_id") or "")
                    for row in payload.get("codes") or []
                    if "KEEP" in str(row.get("definition") or "")
                ]
                parsed = schema(raised_code_ids=raised, rationale="Kept the marked codes in this request.")
            elif name == "ClaimDecisions":
                payload = json.loads(text[text.find("{") :])
                parsed = schema(
                    claims=[
                        {
                            "entry_id": claim["entry_id"],
                            "disposition": "proposed",
                            "support": "indeterminate",
                            "rationale": "Left proposed.",
                        }
                        for claim in payload.get("claims") or []
                    ]
                )
            else:
                raise AssertionError(name)
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def _focused_source(codes: list[dict]) -> OpenCodeResult:
    passages = [{"passage_id": "p1", "doc_id": "d1", "text": "Packing the suitcase took all night."}]
    segments = [
        {
            "segment_id": "s1",
            "code_id": codes[0]["code_id"],
            "code": codes[0]["code"],
            "passage_id": "p1",
            "doc_id": "d1",
            "quote": "Packing the suitcase",
        }
    ]
    return OpenCodeResult(
        codebook=codes,
        segments=segments,
        memos=[],
        passages=passages,
        saturation=[],
        saturation_report={"judgment": None, "points": []},
        validation_log=[],
        sweep_log=[],
        prompt_log=[],
        metadata={"context": {"research_question": "How do people leave?", "population": "Travelers"}},
    )


def _code(code_id: str, definition: str) -> dict:
    return {"code_id": code_id, "code": code_id, "definition": definition}


def _selection_state(codes: list[dict]) -> dict:
    return {
        "codebook": {row["code_id"]: row for row in codes},
        "segments": [],
        "metadata": {},
        "proposed_raised_code_ids": [],
        "selection": [],
    }


def _codes_over_allowance(marker_every: int = 0) -> tuple[list[dict], int]:
    """Definitions whose one-shot selection payload is about 60k tokens."""
    reviewer = DecisionReviewer(_KeepSelector(), max_calls=40, max_tokens=10**9, token_budget=10**9)
    width = 9000
    count = 18
    codes: list[dict] = []
    estimate = 0
    for _ in range(6):
        codes = []
        for index in range(count):
            mark = " KEEP" if marker_every and index % marker_every == 0 else ""
            codes.append(_code(f"c-{index:02d}", ("packing " * (width // 8)) + mark))
        estimate = reviewer.estimate(selection_payload(_selection_state(codes)), SelectionDecision)
        if estimate >= 55000:
            break
        width = int(width * max(55000 / max(estimate, 1), 1.4)) + 500
    assert estimate >= 55000
    return codes, estimate


def _run_focused(codes, reviewer, **extra):
    return focused_code(
        _focused_source(codes),
        model=FakeFocusedCoder(),
        decision_model=reviewer,
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=False,
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        **extra,
    )


def _selection_report(result) -> dict:
    return result.metadata["decision_reviews"]["focused_selection"]


def _chunk_ids(result) -> list[list[str]]:
    found = []
    for row in result.prompt_log:
        if row.get("checkpoint") != "focused_selection":
            continue
        messages = row.get("messages") or []
        text = "\n".join(str(message.get("content") or "") for message in messages)
        start = text.find("{")
        if start < 0:
            continue
        payload = json.loads(text[start:])
        found.append(list((payload.get("review_chunk") or {}).get("code_ids") or []))
    return found


def test_pack_is_order_stable_and_is_not_batch_size():
    def estimate(ids):
        return 10 * len(ids)

    forward = pack_decision_chunks(["b", "a", "c"], estimate, per_request=25, omit_reason="too_big")
    backward = pack_decision_chunks(["c", "a", "b"], estimate, per_request=25, omit_reason="too_big")
    assert forward["chunks"] == [["b", "a"], ["c"]]
    assert backward["chunks"] == [["c", "a"], ["b"]]
    assert forward["chunks"] != backward["chunks"]
    omitted = pack_decision_chunks(
        ["a", "huge", "b"],
        lambda ids: 100 if "huge" in ids else 10,
        per_request=50,
        omit_reason="single_code_exceeds_request_budget",
    )
    assert omitted["chunks"] == [["a"], ["b"]]
    assert omitted["omissions"] == [{"id": "huge", "reason": "single_code_exceeds_request_budget"}]


def test_codebook_order_does_not_change_the_raised_codes():
    body = "packing " * 1200
    forward = [
        _code("c-b", body + " DROP"),
        _code("c-a", body + " KEEP"),
        _code("c-d", body + " DROP"),
        _code("c-c", body + " KEEP"),
    ]
    reverse = list(reversed(forward))
    left = _run_focused(forward, _KeepSelector(), max_decision_calls=8, max_decision_tokens=24000)
    right = _run_focused(reverse, _KeepSelector(), max_decision_calls=8, max_decision_tokens=24000)
    assert left.raised_code_ids == right.raised_code_ids == ["c-a", "c-c"]
    assert _chunk_ids(left) == _chunk_ids(right)
    assert len(_chunk_ids(left)) >= 2
    assert all(int(row.get("estimated_input_tokens") or 0) <= 6000 for row in left.prompt_log if row.get("checkpoint") == "focused_selection")
    report = _selection_report(left)
    assert report["complete"] is True
    assert report["chunk_count"] >= 2
    assert report["stop_reasons"] is None
    assert report["tokens_estimated"] is not None
    assert report["expenditure_complete"] is True
    assert left.metadata["performance"]["batch_size"] == 1


def test_oversized_code_list_makes_focused_selection_calls():
    codes, estimate = _codes_over_allowance()
    assert estimate > 24000
    reviewer = _KeepSelector()
    finished = _run_focused(
        codes,
        reviewer,
        max_decision_calls=40,
        max_decision_tokens=200000,
        decision_token_budget=6000,
    )
    sent = [row for row in finished.prompt_log if row.get("checkpoint") == "focused_selection" and row.get("status") == "validated"]
    assert reviewer.calls >= 2
    assert len(sent) >= 2
    assert all(int(row.get("estimated_input_tokens") or 0) <= 6000 for row in sent)
    report = _selection_report(finished)
    assert report["chunks_sent"] >= 2
    assert report["sized_by"] == "per_request_budget"
    assert report["expenditure_complete"] is True
    assert report["tokens_estimated"] is not None

    truncated = _run_focused(codes, _KeepSelector())
    truncated_report = _selection_report(truncated)
    # 0.9.27 does not spend calls on a plan the default allowance cannot finish.
    assert truncated_report["chunks_required"] > truncated_report["decision_call_allowance"]
    assert truncated_report["chunks_sent"] == 0
    assert truncated_report["complete"] is False
    assert truncated_report["items_uncovered"] > 0
    assert truncated_report["chunk_order_selects_codes"] is False
    assert "chunk_allowance_insufficient" in (truncated_report["stop_reasons"] or [])
    assert truncated.raised_code_ids == []
    assert truncated.metadata["selection_review_applied"] is False
    assert truncated.metadata["selection_downstream_eligible"] is False
    assert truncated.metadata["selection_status"] == "provisional"
    assert truncated.metadata["complete"] is False
    assert truncated.metadata["status"] == "incomplete"
    assert truncated.metadata["stop"] == "decision_chunk_allowance_insufficient"
    assert truncated.metadata["decision_stop"] == "chunk_allowance_insufficient"


def test_one_code_over_the_request_budget_is_compacted_and_reviewed():
    """A definition that does not fit is shortened and still reviewed.

    0.9.24 omitted it. 0.9.26 compacts the evidence and retries inside the
    same budget. The code is not dropped, and it is not raised: the
    truncated definition does not contain the KEEP marker.
    """
    huge = _code("c-huge", "packing " * 20000)
    keep = _code("c-keep", "KEEP a small code")
    other = _code("c-other", "DROP a small code")
    result = _run_focused([huge, keep, other], _KeepSelector(), max_decision_calls=8, max_decision_tokens=24000)
    report = _selection_report(result)
    assert report["complete"] is True
    assert report["omissions"] == []
    assert report["compaction"].get("c-huge", 0) >= 1
    assert report["chunks_sent"] >= 1
    assert result.raised_code_ids == ["c-keep"]
    assert result.metadata["selection_review_applied"] is True
    assert "c-huge" not in result.raised_code_ids
    sent = [row for row in result.prompt_log if row.get("checkpoint") == "focused_selection" and row.get("status") == "validated"]
    assert sent
    assert any("c-huge" in json.dumps(row.get("messages")) for row in sent)
    assert any("evidence_compaction" in json.dumps(row.get("messages")) for row in sent)


def test_a_review_with_no_sent_chunks_leaves_token_figures_null():
    codes = [_code("c-a", "KEEP"), _code("c-b", "DROP")]
    result = _run_focused(codes, _KeepSelector(), decision_token_budget=1, max_decision_calls=4)
    report = _selection_report(result)
    assert report["chunks_sent"] == 0
    assert report["complete"] is False
    assert report["stop_reasons"] is None
    assert report["tokens_estimated"] is None
    assert report["expenditure_complete"] is None
    assert report["items_omitted"] == 2
    assert result.metadata["complete"] is False
    assert result.raised_code_ids == []
    assert result.metadata["selection_review_applied"] is False
    assert result.metadata["selection_review_fallback"] == "proposal_unreviewed_over_request_budget"


def test_core_merge_prefers_graph_support_over_chunk_order():
    unsupported = {"code_id": "c-early", "graph_supported": False, "pattern_supported_centrality": 9, "accepted_centrality": 9, "proposed_centrality": 9}
    supported = {"code_id": "c-late", "graph_supported": True, "pattern_supported_centrality": 1, "accepted_centrality": 0, "proposed_centrality": 0}
    first = merge_core_nominations([unsupported, supported], ["c-early", "c-late"])
    second = merge_core_nominations([supported, unsupported], ["c-late", "c-early"])
    assert first == second == {"core_id": "c-late", "supported": True}
    tied = merge_core_nominations(
        [
            {"code_id": "c-b", "graph_supported": True, "pattern_supported_centrality": 1, "accepted_centrality": 0, "proposed_centrality": 0},
            {"code_id": "c-a", "graph_supported": True, "pattern_supported_centrality": 1, "accepted_centrality": 0, "proposed_centrality": 0},
        ],
        ["c-b", "c-a"],
    )
    assert tied["core_id"] == "c-a"


def _selective_source(order: list[str], *, width: int) -> dict:
    source = _corpus()
    label = "preparing " * (width // 10)
    source["codebook"] = {
        code_id: {"code_id": code_id, "code": label + (" KEEP" if code_id == "c-keep" else " OTHER"), "definition": code_id}
        for code_id in order
    }
    source["relationships"] = [
        {
            "relationship_id": "rel-0001",
            "from_id": "c-keep",
            "to_id": "c-outside",
            "statement": "Packing is a condition of returning.",
            "quote": "Packing the suitcase took all night before the trip.",
            "passage_id": "p1",
            "evidence_tier": 2,
            "evidence_status": "pattern_supported",
            "disposition": "proposed",
            "paradigm_role": "context",
        }
    ]
    source["segments"] = [
        {"segment_id": "s1", "code_id": "c-keep", "passage_id": "p1"},
        {"segment_id": "s2", "code_id": "c-other", "passage_id": "p2"},
    ]
    return source


class _ChunkCore(FakeAxialCoder):
    """Nominates KEEP when that candidate is in the request, otherwise the first id."""

    def with_structured_output(self, schema, **kwargs):
        parent = super().with_structured_output(schema, **kwargs)
        owner = self

        def _call(messages):
            name = getattr(schema, "__name__", "")
            if name != "ModelCoreDecision":
                return parent.invoke(messages)
            owner.call_count += 1
            text = "\n".join(message_text(message) for message in messages)
            payload = json.loads(text[text.find("{") :])
            ids = [str(row.get("code_id") or "") for row in payload.get("candidates") or []]
            chosen = "c-keep" if "c-keep" in ids else (ids[0] if ids else "")
            parsed = schema(core_id=chosen, rationale="Nominated from this request only.")
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def _core_calls(result) -> list[dict]:
    return [
        row
        for row in result.prompt_log
        if row.get("checkpoint") == "axial_core_review" and row.get("status") == "validated"
    ]


def test_oversized_candidate_list_makes_core_selection_calls():
    width = 12000
    source = _selective_source(["c-keep", "c-other"], width=width)
    extra = []
    for index in range(16):
        code_id = f"c-{index:02d}"
        extra.append(code_id)
        source["codebook"][code_id] = {
            "code_id": code_id,
            "code": "category " * (width // 9),
            "definition": code_id,
        }
    finished = selective_code(
        source,
        _ChunkCore(),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        axial_token_budget=6000,
        max_axial_review_tokens=200000,
        max_decision_calls=40,
        batch_size=1,
    )
    sent = _core_calls(finished)
    assert len(sent) >= 2
    assert all(int(row.get("estimated_input_tokens") or 0) <= 6000 for row in sent)
    report = finished.metadata["decision_reviews"]["axial_core_review"]
    assert report["chunks_sent"] >= 2
    assert report["expenditure_complete"] is True
    assert report["tokens_estimated"] is not None
    assert finished.metadata["model_core_judgment"] is True
    assert finished.metadata["performance"]["batch_size"] == 1

    truncated = selective_code(
        source,
        _ChunkCore(),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        axial_token_budget=6000,
        batch_size=1,
    )
    truncated_report = truncated.metadata["decision_reviews"]["axial_core_review"]
    assert truncated_report["chunks_sent"] >= 1
    assert truncated_report["complete"] is False
    assert truncated_report["items_uncovered"] > 0
    assert truncated.metadata["core_selection_complete"] is False
    assert truncated.metadata["core_selection_truncated"] is True
    assert truncated.metadata["model_core_id"] == ""
    assert truncated.metadata["core_downstream_eligible"] is False
    assert truncated.metadata["core_status"] == "provisional"
    assert truncated.metadata["complete"] is False


def test_chunk_order_does_not_change_the_core_and_tiers_still_win():
    left = selective_code(
        _selective_source(["c-keep", "c-other"], width=9000),
        _ChunkCore(),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        axial_token_budget=6000,
        max_axial_review_tokens=24000,
        max_decision_calls=8,
    )
    right = selective_code(
        _selective_source(["c-other", "c-keep"], width=9000),
        _ChunkCore(),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        axial_token_budget=6000,
        max_axial_review_tokens=24000,
        max_decision_calls=8,
    )
    assert len(_core_calls(left)) >= 2
    assert len(_core_calls(right)) >= 2
    assert left.model_core_id == right.model_core_id == "c-keep"
    assert left.metadata.get("core_unsupported_by_graph") is not True


class _VerifiedReview(FakeAxialCoder):
    def relation_review(self, body):
        return AxialRelationDecisions(
            relationships=[
                AxialRelationDecision(
                    relationship_id=str(row.get("relationship_id") or ""),
                    disposition="verified",
                    paradigm_role="causal_condition",
                    rationale="Participants attribute packing to the return, and this is verified.",
                )
                for row in body.get("relationships") or []
            ]
        )


def _relationship(rel_id: str, statement: str) -> dict:
    return {
        "relationship_id": rel_id,
        "from_id": "c-prep",
        "to_id": "c-return",
        "statement": statement,
        "quote": "Packing the suitcase took all night before the trip.",
        "passage_id": "p1",
        "evidence_tier": 2,
        "evidence_status": "pattern_supported",
        "disposition": "proposed",
        "paradigm_role": "context",
    }


def test_relation_review_chunks_and_cannot_verify():
    source = _corpus()
    source["relationships"] = [
        _relationship("rel-0001", "Packing is a condition of returning."),
        _relationship("rel-0002", "The return still depends on packing."),
    ]
    reviewed = axial_code(
        source,
        _VerifiedReview(),
        seed=3,
        decision_mode="llm",
        axial_relations=False,
        interrupt=False,
        max_axial_review_calls=0,
        max_decision_calls=4,
    )
    assert reviewed.metadata["decision_reviews"]["axial_relation_review"]["chunks_sent"] >= 1
    for row in reviewed.relationships:
        assert row.get("evidence_tier") != 4
        assert row.get("disposition") != "verified"
        assert row.get("model_disposition") != "verified"
        assert "model_cannot_verify" in (row.get("flags") or [])
        assert any(
            item.get("disposition") == "verified"
            for item in (row.get("review_history") or [])
            if isinstance(item, dict)
        )

    wide = _corpus()
    wide["relationships"] = [
        _relationship("rel-small", "Packing is a condition of returning."),
        _relationship("rel-huge", "packing " * 20000),
    ]
    limited = axial_code(
        wide,
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        axial_relations=False,
        interrupt=False,
        max_axial_review_calls=0,
        max_decision_calls=4,
        axial_token_budget=6000,
    )
    report = limited.metadata["decision_reviews"]["axial_relation_review"]
    assert {"id": "rel-huge", "reason": "single_relationship_exceeds_request_budget"} in report["omissions"]
    assert report["complete"] is False
    assert report["chunks_sent"] >= 1
    by_id = {row["relationship_id"]: row for row in limited.relationships}
    assert by_id["rel-small"].get("review_actor") == "llm"
    assert not by_id["rel-huge"].get("review_actor")
    assert limited.metadata["stage_outcomes"]["decision_review"]["status"] == "budget_limited"
    assert limited.metadata["complete"] is False


def _rewrite_metadata(saver: InMemorySaver, thread_id: str, mutate) -> None:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    saved = saver.get_tuple(config)
    assert saved is not None
    version = saved.checkpoint["channel_versions"]["metadata"]
    namespace = saved.config["configurable"].get("checkpoint_ns", "")
    key = (thread_id, namespace, "metadata", version)
    type_tag, blob = saver.blobs[key]
    metadata = saver.serde.loads_typed((type_tag, blob))
    mutate(metadata)
    saver.blobs[key] = saver.serde.dumps_typed(metadata)


def test_an_old_checkpoint_cannot_resume_into_chunked_review():
    saver = InMemorySaver()
    source = _focused_source([_code("c-a", "KEEP"), _code("c-b", "DROP")])
    paused = focused_code(
        source,
        model=FakeFocusedCoder(),
        seed=3,
        checkpointer=saver,
        thread_id="old-focused",
        interrupt=True,
        max_focused_review_calls=2,
    )
    assert paused.interrupted

    def strip_focused(metadata):
        metadata["procedure"].pop("decision_review_chunking", None)
        manifest = metadata.get("checkpoint_manifest") or {}
        procedure = manifest.get("procedure") or {}
        procedure.pop("decision_review_chunking", None)
        manifest["procedure"] = procedure
        metadata["checkpoint_manifest"] = manifest

    _rewrite_metadata(saver, "old-focused", strip_focused)
    with pytest.raises(DecisionChunkingCheckpointError, match="decision-review"):
        focused_code(
            source,
            model=FakeFocusedCoder(),
            seed=3,
            checkpointer=saver,
            thread_id="old-focused",
            interrupt=True,
            resuming=True,
            resume={"raised_code_ids": ["c-a"]},
            allow_changes=True,
            max_focused_review_calls=2,
        )

    axial_saver = InMemorySaver()
    corpus = _corpus()
    axial_code(
        corpus,
        FakeAxialCoder(),
        seed=3,
        checkpointer=axial_saver,
        thread_id="old-axial",
        interrupt=True,
        max_axial_review_calls=2,
        batch_size=1,
    )

    def strip_axial(metadata):
        metadata["procedure"].pop("decision_review_chunking", None)

    _rewrite_metadata(axial_saver, "old-axial", strip_axial)
    with pytest.raises(DecisionChunkingCheckpointError, match="chunked"):
        axial_code(
            corpus,
            FakeAxialCoder(),
            seed=3,
            checkpointer=axial_saver,
            thread_id="old-axial",
            interrupt=True,
            resuming=True,
            resume={"leave_relations": True},
            allow_changes=True,
            max_axial_review_calls=2,
            batch_size=1,
        )
    assert "decision-review" in CHUNKING_CHECKPOINT_MESSAGE


def test_a_changed_chunk_plan_does_not_resume():
    plan = build_decision_plan(
        ["c-a", "c-b"],
        lambda ids: 10 * len(list(ids)),
        kind="focused_selection",
        per_request=25,
        max_tokens=100,
        max_calls=4,
        omit_reason="single_code_exceeds_request_budget",
    )
    progress = load_chunk_progress({}, plan)
    progress["open"] = True
    other = dict(plan)
    other["plan_hash"] = "different"
    with pytest.raises(DecisionChunkingCheckpointError):
        load_chunk_progress({"decision_chunk_state": progress}, other)
    empty = finish_decision_report(
        {
            "kind": "focused_selection",
            "plan_hash": "abc",
            "ordered_ids": [],
            "chunks": [],
            "omissions": [],
            "reviewed_ids": [],
            "chunks_sent": 0,
            "chunks_validated": 0,
            "stop_reasons": [],
            "invalid_chunks": 0,
        }
    )
    assert empty["stop_reasons"] is None
    assert empty["tokens_estimated"] is None
    assert empty["expenditure_complete"] is None
    assert empty["complete"] is True
