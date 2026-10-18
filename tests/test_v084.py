"""0.8.4 request checkpoints, memo completion, usage, and evidence. No network."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, axial_code, selective_code
from automated_grounded_theory.records import POLICY_VERSION
from tests.test_v080 import STATEMENT, FakeAxialCoder, _code, _corpus
from tests.test_v082 import _install_estimates


def test_version_is_084():
    assert __version__ == "0.9.19"
    assert POLICY_VERSION == "0.9.19"


def _three_codes():
    source = _corpus()
    source["codebook"]["c-other"] = _code("c-other", "other work")
    source["segments"].append({"segment_id": "s3", "code_id": "c-other", "passage_id": "p1"})
    return source


class _TimeoutOnSecondProposal(FakeAxialCoder):
    """The second proposal request raises. Later requests succeed."""

    timeout_on: int = 2

    def with_structured_output(self, schema, **kwargs):
        parent = super().with_structured_output(schema, **kwargs)
        if getattr(schema, "__name__", "") != "AxialProposalBatch":
            return parent

        def _call(messages):
            upcoming = self.call_count + 1
            if upcoming == self.timeout_on:
                self.call_count += 1
                raise TimeoutError("timed out")
            return parent.invoke(messages)

        return RunnableLambda(_call)


def _proposal_kwargs():
    return dict(
        seed=3,
        max_axial_review_calls=3,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )


def test_timeout_keeps_the_finished_request_and_the_spent_call():
    coder = _TimeoutOnSecondProposal()
    saver = InMemorySaver()
    common = _proposal_kwargs()
    partial = axial_code(
        _three_codes(),
        coder,
        checkpointer=saver,
        thread_id="timeout-budget",
        **common,
    )
    budget = partial.metadata.get("provider_budget") or {}
    assert partial.relationships
    assert partial.prompt_log
    assert int(budget.get("calls_used") or 0) == 2
    assert partial.metadata["status"] == "incomplete"
    assert partial.metadata["stop"] == "timeout"
    assert partial.metadata["graph_finished"] is False
    assert any(row.get("status") == "validated" and row.get("kind") == "axial_proposal" for row in partial.prompt_log)
    assert any(row.get("status") == "timeout" for row in partial.prompt_log)
    assert coder.call_count == 2

    resumed = axial_code(
        _three_codes(),
        coder,
        checkpointer=saver,
        thread_id="timeout-budget",
        resuming=True,
        resume={"accept": True},
        **common,
    )
    recorded = [
        row
        for row in resumed.prompt_log
        if row.get("status") == "timeout"
        or (row.get("kind") == "axial_proposal" and row.get("status") == "validated")
    ]
    assert coder.call_count == 3
    assert int((resumed.metadata.get("provider_budget") or {}).get("calls_used") or 0) == 3
    assert len(recorded) == 3
    assert resumed.metadata["status"] == "incomplete"
    assert resumed.metadata["stop"] == "timeout"


class _BrokenMemo(FakeAxialCoder):
    """Every memo response fails to parse, including the repair."""

    memo_calls: int = 0

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))
        parent = super().with_structured_output(schema, **kwargs)
        if getattr(schema, "__name__", "") != "TheoreticalMemoBatch":
            return parent

        def _call(messages):
            self.memo_calls += 1
            self.call_count += 1
            raw = AIMessage(
                content=f"memo-{self.memo_calls}",
                usage_metadata={"input_tokens": 11, "output_tokens": 0, "total_tokens": 11},
            )
            if include_raw:
                return {"raw": raw, "parsed": None, "parsing_error": "malformed"}
            return None

        return RunnableLambda(_call)


def _one_relationship():
    source = _corpus()
    source["relationships"] = [
        {
            "relationship_id": "rel-0001",
            "from_id": "c-prep",
            "to_id": "c-return",
            "paradigm_role": "causal_condition",
            "statement": STATEMENT,
            "passage_id": "p1",
            "quote": "Packing the suitcase took all night",
            "evidence_condition": "anchored",
            "disposition": "accepted",
            "claim_support": "undefined",
            "flags": [],
        }
    ]
    return source


def test_failed_memo_stays_incomplete():
    coder = _BrokenMemo()
    result = selective_code(
        _one_relationship(),
        coder,
        seed=3,
        include_memos=True,
        include_storyline=False,
        interrupt=False,
    )
    accounting = result.metadata["memo_accounting"]
    assert coder.memo_calls == 2
    assert result.memos == []
    assert accounting["requested"] == 1
    assert accounting["produced"] == 0
    assert accounting["failed"] == 1
    assert result.metadata["axial_stop"] == "schema_unparsed"
    assert result.metadata["stage_outcomes"]["memos"]["status"] == "incomplete"
    assert result.metadata["stage_outcomes"]["memos"]["reason"] == "schema_unparsed"
    assert result.metadata["status"] == "incomplete"
    assert result.metadata["complete"] is False


def test_empty_memo_stays_incomplete():
    result = selective_code(
        _one_relationship(),
        FakeAxialCoder(mode="empty_memo"),
        seed=3,
        include_memos=True,
        include_storyline=False,
        interrupt=False,
    )
    assert result.memos[0]["text"] == ""
    assert result.memos[0]["flags"] == ["memo_empty"]
    accounting = result.metadata["memo_accounting"]
    assert accounting["requested"] == 1
    assert accounting["produced"] == 0
    assert accounting["empty"] == 1
    assert accounting["failed"] == 1
    assert accounting["not_attempted_rejected"] == 0
    assert accounting["requested"] == accounting["produced"] + accounting["omitted"] + accounting["failed"] + accounting["not_attempted_rejected"]
    assert result.metadata["stage_outcomes"]["memos"]["status"] == "incomplete"
    assert result.metadata["stage_outcomes"]["memos"]["reason"] == "memo_empty"
    assert result.metadata["status"] == "incomplete"
    assert result.metadata["complete"] is False


class _MalformedStory(FakeAxialCoder):
    """The storyline response is malformed. A repair must not be invented."""

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))
        parent = super().with_structured_output(schema, **kwargs)
        if getattr(schema, "__name__", "") != "StorylineBatch":
            return parent

        def _call(messages):
            self.call_count += 1
            raw = AIMessage(
                content="not-a-story",
                usage_metadata={"input_tokens": 30, "output_tokens": 0, "total_tokens": 30},
            )
            if include_raw:
                return {"raw": raw, "parsed": None, "parsing_error": "malformed"}
            return None

        return RunnableLambda(_call)


def test_unsent_repair_does_not_reuse_the_original_usage(monkeypatch, caplog):
    def estimate(messages, schema, tokenizer=None):
        text = " ".join(str(getattr(message, "content", "") or "") for message in messages)
        if "REPAIR THIS OUTPUT" in text:
            return 5000
        return 20

    _install_estimates(monkeypatch, estimate)
    coder = _MalformedStory()
    with caplog.at_level("WARNING"):
        result = selective_code(
            _one_relationship(),
            coder,
            include_storyline=True,
            include_memos=False,
            interrupt=False,
            axial_token_budget=100,
        )
    assert coder.call_count == 1
    usage = result.metadata["provider_token_usage"]
    assert usage["input_tokens"] == 30
    assert usage["reported_calls"] == 1
    repair = [row for row in result.prompt_log if row.get("kind") == "schema_repair"]
    assert repair
    assert repair[-1]["status"] == "not_sent"
    assert repair[-1].get("received") is False
    assert not (repair[-1].get("usage") or {})
    assert "request was not sent" in caplog.text
    assert "request was sent" not in caplog.text


class _MalformedRelationReview(FakeAxialCoder):
    """The relation review is received and does not parse."""

    def with_structured_output(self, schema, **kwargs):
        parent = super().with_structured_output(schema, **kwargs)
        if getattr(schema, "__name__", "") != "AxialRelationDecisions":
            return parent

        def _call(messages):
            self.call_count += 1
            raw = AIMessage(
                content="not-a-decision",
                usage_metadata={"input_tokens": 40, "output_tokens": 0, "total_tokens": 40},
            )
            return {"raw": raw, "parsed": None, "parsing_error": "malformed"}

        return RunnableLambda(_call)


def test_unparsed_decision_review_counts_its_usage():
    result = axial_code(
        _corpus(),
        _MalformedRelationReview(),
        seed=3,
        decision_mode="llm",
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        max_decision_calls=4,
        interrupt=False,
    )
    review = [row for row in result.prompt_log if row.get("kind") == "decision_review" and row.get("checkpoint") == "axial_relation_review"]
    assert review
    assert int((review[0].get("usage") or {}).get("input_tokens") or 0) == 40
    assert review[0].get("status") != "validated"
    logged = 0
    for row in result.prompt_log:
        if str(row.get("status") or "") in {"omitted", "refused", "not_sent", "deferred"}:
            continue
        logged += int((row.get("usage") or {}).get("input_tokens") or 0)
    assert result.metadata["provider_token_usage"]["input_tokens"] == logged
    assert logged >= 40


def test_missing_endpoint_evidence_defers_the_pair():
    source = _corpus()
    source["segments"] = [row for row in source["segments"] if row.get("code_id") != "c-return"]
    coder = FakeAxialCoder()
    result = axial_code(
        source,
        coder,
        seed=3,
        max_axial_review_calls=2,
        axial_share_propose=1,
        axial_share_counterexample=0,
        interrupt=False,
    )
    assert source["segments"]
    assert result.relationships == []
    assert coder.call_count == 0
    missing = result.metadata["missing_category_evidence"]
    assert missing
    assert missing[0]["unsupported_code_ids"] == ["c-return"]
    assert any(row.get("reason") == "missing_category_evidence" for row in result.prompt_log)
    assert result.coverage["missing_category_evidence"]


def test_fresh_process_script_derives_the_checkout():
    root = Path(__file__).resolve().parents[1]
    for name in ("test_v083.py", "test_v0915.py"):
        source = root.joinpath("tests", name).read_text(encoding="utf-8")
        assert 'sys.path.insert(0, "/workspace")' not in source
        assert 'cwd="/workspace"' not in source
        assert "Path(__file__).resolve().parents[1]" in source
