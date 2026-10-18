"""0.9.17 review fixes. Fakes only. No network calls."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import axial_code, theoretical_sample
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs, _payload
from tests.test_v0916 import (
    _ASSOCIATION,
    _RawBatch,
    _Tier2Coder,
    _axial,
    _message_text,
    _sampling_visit_allowance,
    _tier2_corpus,
)


_KEPT = "Packing the suitcase before departure sits with the return to work."
_QUOTE = "Packing the suitcase took all night"
_PRICES = {"input_usd_per_million": 3.0, "output_usd_per_million": 15.0}


def _row(statement: str, *, from_id: str = "c-prep", to_id: str = "c-return") -> dict:
    return {
        "from_id": from_id,
        "to_id": to_id,
        "paradigm_role": "unspecified",
        "statement": statement,
        "passage_id": "p1",
        "quote": _QUOTE,
    }


def _mixed_payload() -> dict:
    return {"relationships": [_row(_KEPT), _row("   ")]}


def test_oversized_rereads_finish_inside_the_default_visit_limit():
    """Thirty-two requests over the token budget are one visit, not one each.

    A human can assign every passage. The default four-call allowance does
    not cap that assignment, and a request that is too large is not a provider
    call. The visit limit still finishes the batch. None of these requests is
    sent. The stop is not the recursion limit.
    """
    stem = "zephyrcrate"
    passages = [
        {
            "passage_id": f"extra-{index:02d}",
            "doc_id": f"d-{index:02d}",
            "respondent_id": f"r-{index:02d}",
            "text": f"{stem} " + ("padding " * 8000),
        }
        for index in range(32)
    ]
    corpus = {
        "codebook": {
            "c-pack": {
                "code_id": "c-pack",
                "code": stem,
                "definition": f"A {stem} that still needs another document.",
            }
        },
        "passages": [
            {
                "passage_id": "coded",
                "doc_id": "d-coded",
                "respondent_id": "r-coded",
                "text": f"The coded {stem} is already evidence.",
            },
            *passages,
        ],
        "segments": [{"segment_id": "s-coded", "code_id": "c-pack", "passage_id": "coded"}],
        "properties": [],
        "dimensions": [],
        "relationships": [],
        "memos": [],
    }
    saver = InMemorySaver()
    coder = FakeSamplingCoder()
    common = _kwargs(
        pursuit="reread",
        evidence_limit=32,
        category_documents=2,
        sampling_token_budget=6000,
        max_sampling_review_calls=4,
    )
    paused = theoretical_sample(corpus, coder, checkpointer=saver, thread_id="oversized", **common)
    assert coder.call_count == 0
    direction_id = next(
        str(row["direction_id"])
        for row in paused.directions
        if str(row.get("kind") or "") == "thin_category"
    )
    withheld = theoretical_sample(
        corpus,
        coder,
        checkpointer=saver,
        thread_id="oversized",
        resuming=True,
        resume={
            "directions": [
                {
                    "direction_id": direction_id,
                    "action": "accept",
                    "sampled_passage_ids": [row["passage_id"] for row in passages],
                }
            ]
        },
        **common,
    )
    assert withheld.metadata.get("reread_stop") == "request_budget"
    result = theoretical_sample(
        corpus,
        coder,
        checkpointer=saver,
        thread_id="oversized",
        resuming=True,
        resume={"leave_saturation": True},
        **common,
    )
    unsent = [row for row in result.metadata.get("reread_unsent_pairs") or [] if isinstance(row, dict)]
    assert {row["passage_id"] for row in unsent} == {row["passage_id"] for row in passages}
    assert {row["reason"] for row in unsent} == {"request_budget"}
    assert result.metadata.get("stop") == "request_budget"
    assert result.reread_notes == []
    assert coder.call_count == 0
    sent = [
        row
        for row in result.prompt_log
        if row.get("kind") == "sampling_reread" and row.get("received") is not False
    ]
    assert sent == []
    assert result.metadata["performance"]["recursion_limit"] == _sampling_visit_allowance(4)


def test_mixed_batch_keeps_json_text_tool_calls_and_a_repaired_response():
    """A valid row survives every supported shape of a mixed batch.

    JSON text, tool-call arguments, and a repair response each keep the one
    valid relationship. The repair is still the single follow-up, and an
    unrequested row is still rejected after it parses.
    """
    text = _RawBatch(payload=_mixed_payload())
    from_text = axial_code(_corpus(), text, **_axial())
    assert [row["statement"] for row in from_text.relationships] == [_KEPT]
    assert int(from_text.metadata["proposal_diagnostics"].get("schema_repairs") or 0) == 0
    assert text.call_count == 1

    class _ToolBatch(FakeAxialCoder):
        def with_structured_output(self, schema, **kwargs):
            def _call(messages):
                del messages
                self.call_count += 1
                raw = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "AxialProposalBatch",
                            "args": {
                                "relationships": [
                                    _row(_KEPT),
                                    _row("   "),
                                    _row(
                                        "This pair was not requested.",
                                        from_id="c-other",
                                        to_id="c-return",
                                    ),
                                ]
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                    usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                )
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("batch failed")}

            return RunnableLambda(_call)

    tool = _ToolBatch()
    from_tool = axial_code(_corpus(), tool, **_axial())
    assert [row["statement"] for row in from_tool.relationships] == [_KEPT]
    tool_diagnostics = from_tool.metadata["proposal_diagnostics"]
    assert int(tool_diagnostics.get("schema_repairs") or 0) == 0
    assert tool_diagnostics["rejection_reasons"]["blank_statement"] == 1
    assert tool_diagnostics["rejection_reasons"]["unrequested_pair"] == 1
    assert tool.call_count == 1

    class _RepairMixed(FakeAxialCoder):
        def with_structured_output(self, schema, **kwargs):
            def _call(messages):
                self.call_count += 1
                name = getattr(schema, "__name__", "")
                if name == "AxialProposalBatch":
                    body = _mixed_payload() if "REPAIR THIS OUTPUT" in _message_text(messages) else {
                        "relationships": [_row("   ")]
                    }
                    raw = AIMessage(
                        content=json.dumps(body),
                        usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                    )
                    return {"raw": raw, "parsed": None, "parsing_error": ValueError("batch failed")}
                return FakeAxialCoder.with_structured_output(self, schema, **kwargs).invoke(messages)

            return RunnableLambda(_call)

    repair = _RepairMixed()
    repaired = axial_code(_corpus(), repair, **_axial(max_axial_review_calls=2))
    assert [row["statement"] for row in repaired.relationships] == [_KEPT]
    assert int(repaired.metadata["proposal_diagnostics"].get("schema_repairs") or 0) == 1
    assert repair.call_count == 2


class _UsageCoder(FakeSamplingCoder):
    """Select the named passages and attach the usage map for each reread."""

    passage_ids: list[str] = []
    usage_by_passage: dict = {}

    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "ModelSamplingSelection":
                thin = ""
                for row in body.get("directions") or []:
                    if str(row.get("kind") or "") == "thin_category":
                        thin = str(row.get("direction_id") or "")
                parsed = schema(
                    selections=[{"direction_id": thin, "passage_id": passage_id} for passage_id in self.passage_ids],
                    rationale="These passages are the expenditure cases.",
                )
                usage = {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
            elif name == "SamplingRereadNote":
                passage_id = str(body.get("passage_id") or "")
                text = str(body.get("text") or "")
                parsed = schema(
                    passage_id=passage_id,
                    judgment="relevant",
                    note="This passage bears on the question.",
                    quote=text[:24],
                )
                usage = dict(self.usage_by_passage.get(passage_id) or {})
            elif name == "ModelSaturationJudgment":
                parsed = schema(assessment="not_saturated", rationale="A gap remains after these notes.")
                usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            else:
                parsed = schema()
                usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            raw = AIMessage(content="{}", usage_metadata=usage) if usage else AIMessage(content="{}")
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def _priced_corpus() -> dict:
    corpus = _corpus()
    corpus["metadata"] = {"token_prices": dict(_PRICES)}
    return corpus


def _expenditure(coder: _UsageCoder, **overrides):
    result = theoretical_sample(
        _priced_corpus(),
        coder,
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        **_kwargs(max_sampling_review_calls=4, sampling_token_budget=200000, max_sampling_review_tokens=200000, **overrides),
    )
    return result.metadata["reread_interpretations"]


def test_reread_expenditure_distinguishes_missing_usage_from_no_calls():
    """Unknown usage is not a known spend of zero.

    No reread calls can report a complete zero. A call with empty usage, and
    a batch where only some calls reported usage, leave the full rates null.
    Cost stays null in those cases even when prices are recorded.
    """
    quiet = theoretical_sample(
        _priced_corpus(),
        _UsageCoder(),
        decision_mode="llm",
        pursuit="brief_only",
        interrupt=False,
        **_kwargs(max_sampling_review_calls=2),
    ).metadata["reread_interpretations"]
    assert quiet["reread_expenditure_calls"] == 0
    assert quiet["reread_expenditure_calls_with_usage"] == 0
    assert quiet["reread_expenditure_usage_coverage"] is None
    assert quiet["reread_expenditure_complete"] is True
    assert quiet["reread_expenditure_tokens"] == 0
    assert quiet["relevant_reread_count"] == 0
    assert quiet["expenditure_tokens_per_relevant_reread"] is None
    assert quiet["reread_expenditure_cost_usd"] == 0.0
    assert quiet["expenditure_cost_per_relevant_reread_usd"] is None

    missing = _expenditure(_UsageCoder(passage_ids=["p1"], usage_by_passage={"p1": {}}))
    assert missing["relevant_reread_count"] == 1
    assert missing["reread_expenditure_calls"] == 1
    assert missing["reread_expenditure_calls_with_usage"] == 0
    assert missing["reread_expenditure_usage_coverage"] == 0.0
    assert missing["reread_expenditure_complete"] is False
    assert missing["reread_expenditure_tokens"] == 0
    assert missing["expenditure_tokens_per_relevant_reread"] is None
    assert missing["reread_expenditure_cost_usd"] is None
    assert missing["tokens_per_relevant_reread"] is None

    partial = _expenditure(
        _UsageCoder(
            passage_ids=["p1", "p-low"],
            usage_by_passage={
                "p1": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "p-low": {},
            },
        )
    )
    assert partial["relevant_reread_count"] == 2
    assert partial["reread_expenditure_calls"] == 2
    assert partial["reread_expenditure_calls_with_usage"] == 1
    assert partial["reread_expenditure_usage_coverage"] == 0.5
    assert partial["reread_expenditure_complete"] is False
    assert partial["reread_expenditure_tokens"] == 15
    assert partial["expenditure_tokens_per_relevant_reread"] is None
    assert partial["reread_expenditure_cost_usd"] is None
    assert partial["tokens_per_relevant_reread"] is None

    full = _expenditure(
        _UsageCoder(
            passage_ids=["p1", "p-low"],
            usage_by_passage={
                "p1": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "p-low": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
            },
        )
    )
    assert full["reread_expenditure_calls"] == 2
    assert full["reread_expenditure_calls_with_usage"] == 2
    assert full["reread_expenditure_usage_coverage"] == 1.0
    assert full["reread_expenditure_complete"] is True
    assert full["reread_expenditure_tokens"] == 25
    assert full["relevant_reread_count"] == 2
    assert full["expenditure_tokens_per_relevant_reread"] == 12.5
    assert full["tokens_per_relevant_reread"] == 12.5
    spent = (18 / 1_000_000) * 3.0 + (7 / 1_000_000) * 15.0
    assert full["reread_expenditure_cost_usd"] == round(spent, 6)
    assert full["expenditure_cost_per_relevant_reread_usd"] == round(spent / 2, 6)


def test_statement_edit_refreshes_the_basis_before_a_role_change():
    """A sentence that does not change flags is the basis a later role archives.

    Statement A and statement B are both non-causal, so the flags stay empty.
    The role change is the audited change, and the archived statement is B.
    """
    statement_b = "The tools for the shift sit beside coworkers on the crew tonight."
    saver = InMemorySaver()
    common = _axial(interrupt=True)
    paused = axial_code(
        _tier2_corpus(),
        _Tier2Coder(proposal_role="unspecified", proposal_statement=_ASSOCIATION),
        checkpointer=saver,
        thread_id="basis-refresh",
        **common,
    )
    proposed = paused.relationships[0]
    assert proposed["statement"] == _ASSOCIATION
    assert proposed["validation_basis"]["statement"] == _ASSOCIATION
    assert proposed["validation_basis"]["role"] == "unspecified"
    for flag in ("wording_exceeds_tier", "role_evidence_exceeds_tier", "causal_wording_withheld"):
        assert flag not in proposed["flags"]

    done = axial_code(
        _tier2_corpus(),
        _Tier2Coder(proposal_role="unspecified", proposal_statement=_ASSOCIATION),
        checkpointer=saver,
        thread_id="basis-refresh",
        resuming=True,
        resume={
            "relationships": [
                {
                    "relationship_id": proposed["relationship_id"],
                    "action": "accept",
                    "statement": statement_b,
                    "paradigm_role": "consequence",
                }
            ]
        },
        **_axial(interrupt=False),
    )
    corrected = done.relationships[0]
    assert corrected["human_statement"] == statement_b
    assert corrected["validation_basis"]["statement"] == statement_b
    assert corrected["validation_basis"]["role"] == "consequence"
    assert "role_evidence_exceeds_tier" in corrected["flags"]
    history = corrected["validation_revision_history"]
    assert history
    archived = history[-1]
    assert archived["statement"] == statement_b
    assert archived["role"] == "unspecified"
    assert "role_evidence_exceeds_tier" not in archived["flags"]
    assert all(item["statement"] != _ASSOCIATION for item in history)
