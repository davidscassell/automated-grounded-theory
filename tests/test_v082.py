"""0.8.2 irrelevance, shared provider allowance, coverage, and packing. No network."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from automated_grounded_theory import axial_code, focused_code, selective_code
from automated_grounded_theory.async_support import (
    CallLimiter,
    ProviderAllowance,
    ainvoke_structured,
    run_sync,
)
from automated_grounded_theory.axial import AxialProposalBatch
from automated_grounded_theory.focused import _apply_claim_test, _rationale_indicates_irrelevance
from automated_grounded_theory.saturation import thematic_saturation_report

from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _memos, _segments_for_raised, _source, _sweep
from tests.test_v080 import STATEMENT, FakeAxialCoder, _code, _corpus


def _claim(entry_id: str, name: str, pole: str) -> dict:
    return {
        "entry_id": entry_id,
        "name": name,
        "pole": pole,
        "text": pole,
        "flags": [],
        "negative_cases": [],
        "claim_tests": [],
        "evidence_condition": "anchored",
    }


def _contradict(props, *, entry_id, quote, rationale, passage_id="p1"):
    counts = {"supports": 0, "contradicts": 0, "qualifies": 0, "irrelevant": 0}
    _apply_claim_test(
        props,
        [],
        {
            "entry_id": entry_id,
            "judgment": "contradicts",
            "quote": quote,
            "rationale": rationale,
        },
        passage_id,
        counts,
        valid=True,
        passage_text=quote,
    )
    return props[0]


def test_paraphrased_contradiction_stays_a_disputed_negative_case():
    managers = _contradict(
        [_claim("m1", "manager access", "Managers are accessible.")],
        entry_id="m1",
        quote="Supervisors were impossible to reach.",
        rationale="This is about supervisors being unavailable.",
    )
    case = managers["negative_cases"][-1]
    assert managers["evidence_condition"] == "disputed"
    assert case["judgment"] == "contradicts"
    assert case["claim_subject_uncertain"] is True
    assert case["needs_review"] is True
    assert managers["downstream_eligible"] is False
    assert "claim_subject_uncertain" in managers["flags"]
    assert "rationale_indicates_irrelevance" not in managers["flags"]
    assert "claim_test_invalid" not in managers["flags"]

    synonym = _contradict(
        [_claim("s1", "boss access", "Bosses are reachable.")],
        entry_id="s1",
        quote="Supervisors were impossible to contact.",
        rationale="This describes leadership being unavailable.",
    )
    assert synonym["evidence_condition"] == "disputed"
    assert synonym["negative_cases"][-1]["claim_subject_uncertain"] is True
    assert "rationale_indicates_irrelevance" not in synonym["flags"]


def test_negated_irrelevance_still_disputes_and_explicit_cues_are_kept():
    assert _rationale_indicates_irrelevance("This is not unrelated to the claim.") is False
    assert _rationale_indicates_irrelevance("This isn't unrelated.") is False
    assert _rationale_indicates_irrelevance("This is not not unrelated.") is True
    assert _rationale_indicates_irrelevance("The passage does not address the claim.") is True
    assert _rationale_indicates_irrelevance("Nothing to do with the supplied claim.") is True
    assert _rationale_indicates_irrelevance("This is about supervisors being unavailable.") is False

    negated = _contradict(
        [_claim("m1", "manager access", "Managers are accessible.")],
        entry_id="m1",
        quote="Supervisors were impossible to reach.",
        rationale="This is not unrelated.",
    )
    assert negated["evidence_condition"] == "disputed"
    assert "rationale_indicates_irrelevance" not in negated["flags"]
    assert negated["negative_cases"][-1]["needs_review"] is True

    explicit = _contradict(
        [_claim("m2", "manager access", "Managers are accessible.")],
        entry_id="m2",
        quote="Supervisors were impossible to reach.",
        rationale="The quote is irrelevant.",
    )
    held = explicit["negative_cases"][-1]
    assert explicit["evidence_condition"] == "disputed_pending_review"
    assert held["reason"] == "rationale_indicates_irrelevance"
    assert held["needs_review"] is True
    assert held["flagged"] is True
    assert explicit["needs_review"] is True
    assert explicit["downstream_eligible"] is False
    assert "claim_test_invalid" not in explicit["flags"]
    assert "rationale_indicates_irrelevance" in explicit["flags"]

    addressed = _contradict(
        [_claim("m3", "manager access", "Managers are accessible.")],
        entry_id="m3",
        quote="Supervisors were impossible to reach.",
        rationale="The passage does not address the supplied topic and has nothing to do with it.",
    )
    assert addressed["evidence_condition"] == "disputed_pending_review"
    assert addressed["negative_cases"][-1]["reason"] == "rationale_indicates_irrelevance"


def test_ops_team_case_is_uncertain_rather_than_a_silent_dispute():
    ops = _contradict(
        [_claim("R039_q1", "specificity of coworker reference", "the passage uses only the generic term coworkers")],
        entry_id="R039_q1",
        quote="The ops team is scrappy and gets things done without a lot of red tape",
        rationale="The passage is about the ops team.",
    )
    assert ops["evidence_condition"] == "disputed"
    assert ops["negative_cases"]
    assert ops["negative_cases"][-1]["claim_subject_uncertain"] is True
    assert ops["needs_review"] is True
    assert "claim_subject_uncertain" in ops["flags"]
    assert "rationale_indicates_irrelevance" not in ops["flags"]


def test_total_variation_note_does_not_claim_saturation():
    report = thematic_saturation_report(
        [
            {"document_id": "d1", "code_frequencies": {"a": 2}, "new_codes": 1},
            {"document_id": "d2", "code_frequencies": {"a": 2}, "new_codes": 0},
        ]
    )
    assert report["judgment"] is None
    assert report["code_frequency_comparison"] == "total_variation"
    assert "does not establish theoretical saturation" in report["note"]
    assert "not a human confirmation" in report["note"]


def _install_estimates(monkeypatch, estimate):
    monkeypatch.setattr("automated_grounded_theory.axial.request_input_tokens", estimate)
    monkeypatch.setattr("automated_grounded_theory.batching.request_input_tokens", estimate)


def _blob(messages) -> str:
    return " ".join(str(getattr(message, "content", "")) for message in messages)


def test_shared_token_allowance_covers_coding_and_decision_review(monkeypatch):
    def estimate(messages, schema, tokenizer=None):
        name = getattr(schema, "__name__", "")
        if name == "AxialProposalBatch":
            return 1402
        if name in {"AxialRelationDecisions", "ModelSaturationDecision", "ModelCoreDecision"}:
            return 1166
        return 30

    _install_estimates(monkeypatch, estimate)
    coder = FakeAxialCoder()
    result = axial_code(
        _corpus(),
        coder,
        seed=3,
        decision_mode="llm",
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        max_decision_calls=4,
        max_axial_review_tokens=1500,
        interrupt=False,
    )
    validated = [
        int(row.get("estimated_input_tokens") or 0)
        for row in result.prompt_log
        if row.get("status") == "validated"
    ]
    assert validated
    assert sum(validated) <= 1500
    assert result.metadata["axial_tokens_estimated"] <= 1500
    assert result.metadata["decision_stop"] == "token_allowance"
    omitted = [row for row in result.prompt_log if row.get("kind") == "decision_review" and row.get("status") == "omitted"]
    assert omitted
    assert all(row.get("reason") == "token_allowance" and row.get("needs_review") for row in omitted)
    assert coder.call_count == 1


class _RepairCoder(FakeAxialCoder):
    """First axial proposals are malformed. Later calls use the parent fixture."""

    fail_proposals: int = 1

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))
        inner = super().with_structured_output(schema, **kwargs)

        def _call(messages):
            name = getattr(schema, "__name__", "")
            if name == "AxialProposalBatch" and self.call_count < int(self.fail_proposals):
                self.call_count += 1
                raw = AIMessage(
                    content="not json",
                    usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
                )
                if include_raw:
                    return {"raw": raw, "parsed": None, "parsing_error": "malformed"}
                return None
            return inner.invoke(messages)

        return RunnableLambda(_call)


def test_schema_repair_consumes_the_one_call_allowance_and_keeps_the_failure(tmp_path):
    coder = _RepairCoder(fail_proposals=1)
    result = axial_code(
        _corpus(),
        coder,
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        interrupt=False,
        output_dir=tmp_path,
    )
    assert coder.call_count == 1
    assert result.relationships == []
    assert result.metadata["status"] == "incomplete"
    assert result.metadata["complete"] is False
    accounting = result.metadata["pair_accounting"]
    assert accounting["attempted_count"] == 1
    assert accounting["completed_count"] == 0
    flagged = [row for row in result.prompt_log if row.get("needs_review") and row.get("flagged")]
    assert flagged
    assert any("not json" in json.dumps(row) for row in flagged)
    exported = json.loads((tmp_path / "axial_run.json").read_text(encoding="utf-8"))
    assert exported["metadata"]["status"] == "incomplete"
    assert exported["metadata"]["pair_accounting"]["completed_count"] == 0


def test_schema_repair_is_its_own_audit_when_a_call_remains():
    coder = _RepairCoder(fail_proposals=1)
    result = axial_code(
        _corpus(),
        coder,
        seed=3,
        max_axial_review_calls=8,
        axial_share_propose=1,
        axial_share_counterexample=0,
        interrupt=False,
    )
    assert coder.call_count >= 2
    assert result.relationships
    repairs = [row for row in result.prompt_log if row.get("kind") == "schema_repair" and row.get("status") == "validated"]
    failed = [row for row in result.prompt_log if row.get("status") == "parse_failed"]
    assert repairs and failed
    assert result.metadata["pair_accounting"]["completed_count"] >= 1
    assert result.metadata["status"] == "complete"


class _Tiny(BaseModel):
    ok: bool = True


class _StructuredFake(BaseChatModel):
    calls: int = 0
    malformed: int = 1

    @property
    def _llm_type(self) -> str:
        return "tiny-repair"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            self.calls += 1
            if self.calls <= int(self.malformed):
                return {
                    "raw": AIMessage(content="not json"),
                    "parsed": None,
                    "parsing_error": "malformed",
                }
            return {"raw": AIMessage(content="{}"), "parsed": schema(ok=True), "parsing_error": None}

        return RunnableLambda(_call)


def test_provider_allowance_blocks_repair_and_retry_at_the_call_boundary():
    model = _StructuredFake()
    messages = [HumanMessage(content="hello")]
    limiter = CallLimiter(
        1,
        max_retries=0,
        request_timeout=0,
        provider_allowance=ProviderAllowance(max_calls=1, max_tokens=10**6),
    )
    audits: list = []
    (payload, _usage), _sent = run_sync(
        ainvoke_structured(
            model,
            _Tiny,
            messages,
            limiter,
            preserve_failed=True,
            send_repair_overflow=False,
            allowance_kind="unit",
            call_audits=audits,
        )
    )
    assert model.calls == 1
    assert payload["_agt_unparsed"] is True
    assert payload["needs_review"] is True
    assert payload["flagged"] is True
    assert any(row.get("kind") == "schema_repair" and row.get("status") == "refused" for row in audits)

    roomy = _StructuredFake()
    roomy_limiter = CallLimiter(
        1,
        max_retries=0,
        request_timeout=0,
        provider_allowance=ProviderAllowance(max_calls=2, max_tokens=10**6),
    )
    roomy_audits: list = []
    (parsed, _usage), _sent = run_sync(
        ainvoke_structured(
            roomy,
            _Tiny,
            messages,
            roomy_limiter,
            preserve_failed=True,
            send_repair_overflow=False,
            call_audits=roomy_audits,
        )
    )
    assert roomy.calls == 2
    assert parsed["ok"] is True
    assert any(row.get("kind") == "schema_repair" and row.get("status") == "validated" for row in roomy_audits)

    class _Drop(BaseChatModel):
        calls: int = 0

        @property
        def _llm_type(self) -> str:
            return "drop"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

        async def ainvoke(self, payload):
            self.calls += 1
            raise ConnectionError("dropped")

    dropped = _Drop()
    retry_limiter = CallLimiter(
        1,
        max_retries=4,
        request_timeout=0,
        provider_allowance=ProviderAllowance(max_calls=2, max_tokens=10**6),
    )
    with pytest.raises(Exception) as caught:
        run_sync(retry_limiter.ainvoke(dropped, {"call": 1}, use_cache=False, allowance_estimate=10))
    assert dropped.calls == 2
    assert "allowance" in str(caught.value).casefold()


class _RepairFocused(FakeFocusedCoder):
    def with_structured_output(self, schema, **kwargs):
        inner = super().with_structured_output(schema, **kwargs)

        def _call(messages):
            if self.call_count == 0:
                self.call_count += 1
                return {
                    "raw": AIMessage(
                        content="not json",
                        usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
                    ),
                    "parsed": None,
                    "parsing_error": "malformed",
                }
            return inner.invoke(messages)

        return RunnableLambda(_call)


def test_focused_repair_uses_the_same_call_allowance():
    coder = _RepairFocused()
    result = focused_code(
        _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep()),
        model=coder,
        seed=1,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        interrupt=False,
        max_focused_review_calls=1,
    )
    assert coder.call_count == 1
    flagged = [row for row in result.prompt_log if row.get("status") == "schema_unparsed"]
    assert flagged
    assert flagged[0]["needs_review"] is True
    assert flagged[0]["flagged"] is True
    audits = flagged[0].get("provider_audits") or []
    assert any(row.get("kind") == "schema_repair" and row.get("status") == "refused" for row in audits)


def _two_relationships():
    source = _corpus()
    source["codebook"]["c-other"] = _code("c-other", "other work")
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
        },
        {
            "relationship_id": "rel-0002",
            "from_id": "c-prep",
            "to_id": "c-other",
            "paradigm_role": "context",
            "statement": "Other work sits beside packing.",
            "passage_id": "p1",
            "quote": "Packing the suitcase took all night",
            "evidence_condition": "anchored",
            "disposition": "accepted",
            "claim_support": "undefined",
            "flags": [],
        },
    ]
    return source


def test_oversized_memo_batch_shrinks_instead_of_skipping_both(monkeypatch):
    def estimate(messages, schema, tokenizer=None):
        name = getattr(schema, "__name__", "")
        if name == "TheoreticalMemoBatch":
            return 1562 if _blob(messages).count("relationship_id") >= 2 else 900
        return 30

    _install_estimates(monkeypatch, estimate)
    source = _two_relationships()
    alone = selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        include_memos=True,
        batch_size=1,
        axial_token_budget=910,
        max_selective_review_calls=4,
        interrupt=False,
    )
    together = selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        include_memos=True,
        batch_size=2,
        axial_token_budget=910,
        max_selective_review_calls=4,
        interrupt=False,
    )
    assert {row["relationship_id"] for row in alone.memos} == {"rel-0001", "rel-0002"}
    assert {row["relationship_id"] for row in together.memos} == {"rel-0001", "rel-0002"}


def test_proposal_trimming_keeps_both_categories_or_defers(monkeypatch):
    def fitting(messages, schema, tokenizer=None):
        name = getattr(schema, "__name__", "")
        blob = _blob(messages)
        if name == "AxialProposalBatch":
            return 80 if '"excerpt": true' in blob else 9000
        return 30

    _install_estimates(monkeypatch, fitting)
    sent = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        axial_token_budget=200,
        interrupt=False,
    )
    proposal = next(row for row in sent.prompt_log if row.get("kind") == "axial_proposal" and row.get("status") == "validated")
    human = next(message["content"] for message in proposal["messages"] if message.get("role") == "human")
    requested = json.loads(human)
    codes = {code_id for passage in requested["passages"] for code_id in passage.get("code_ids") or []}
    assert codes >= {"c-prep", "c-return"}
    excerpts = [passage for passage in requested["passages"] if passage.get("excerpt")]
    assert excerpts
    assert all("source_offset" in passage and "source_length" in passage for passage in excerpts)
    assert sent.relationships

    def never(messages, schema, tokenizer=None):
        name = getattr(schema, "__name__", "")
        if name == "AxialProposalBatch":
            return 9000
        return 30

    _install_estimates(monkeypatch, never)
    coder = FakeAxialCoder()
    deferred = axial_code(
        _corpus(),
        coder,
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        axial_token_budget=200,
        interrupt=False,
    )
    assert coder.call_count == 0
    assert deferred.relationships == []
    held = [row for row in deferred.prompt_log if row.get("status") == "deferred"]
    assert held
    assert held[0]["reason"] == "evidence_does_not_fit"
    assert held[0]["needs_review"] is True
    deferred_body = json.dumps(held[0])
    assert "c-prep" in deferred_body and "c-return" in deferred_body
    assert deferred.metadata["status"] == "budget_limited"
    assert deferred.metadata["complete"] is False


def test_zero_token_budget_is_not_complete_coverage(tmp_path):
    coder = FakeAxialCoder()
    result = axial_code(
        _corpus(),
        coder,
        seed=3,
        max_axial_review_calls=4,
        max_axial_review_tokens=0,
        interrupt=False,
        output_dir=tmp_path,
    )
    assert coder.call_count == 0
    assert result.relationships == []
    assert result.metadata["status"] == "budget_limited"
    assert result.metadata["complete"] is False
    assert result.metadata["graph_finished"] is True
    assert result.metadata["analytical_coverage"] == "budget_limited"
    assert result.metadata["stop"] == "token_allowance"
    assert result.metadata["axial_stop"] == "token_allowance"
    assert result.unexamined_pairs
    accounting = result.metadata["pair_accounting"]
    assert accounting["planned_count"] >= 1
    assert accounting["attempted_count"] == 0
    assert accounting["completed_count"] == 0
    assert accounting["omitted_count"] >= 1
    assert accounting["remaining_count"] == accounting["planned_count"]
    exported = json.loads((tmp_path / "category_relationships.json").read_text(encoding="utf-8"))
    assert exported["unexamined_pairs"]
    assert exported["pair_accounting"]["planned_count"] == accounting["planned_count"]
    assert exported["pair_accounting"]["completed_count"] == 0
    coverage = json.loads((tmp_path / "theory_coverage.json").read_text(encoding="utf-8"))
    assert coverage["pair_accounting"]["remaining_count"] == accounting["remaining_count"]


def test_parse_failure_after_a_repair_stays_incomplete():
    coder = _RepairCoder(fail_proposals=2)
    result = axial_code(
        _corpus(),
        coder,
        seed=3,
        max_axial_review_calls=2,
        axial_share_propose=1,
        axial_share_counterexample=0,
        interrupt=False,
    )
    assert coder.call_count == 2
    assert result.relationships == []
    assert result.metadata["status"] == "incomplete"
    assert result.metadata["complete"] is False
    assert result.metadata["analytical_coverage"] == "partial"
    accounting = result.metadata["pair_accounting"]
    assert accounting["attempted_count"] == 1
    assert accounting["completed_count"] == 0
    assert any(row.get("status") == "schema_unparsed" and row.get("needs_review") for row in result.prompt_log)
