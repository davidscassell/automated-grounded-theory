"""0.9.8 regressions for the review of 0.9.7. Fake models only. No live evaluation."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import __version__, theoretical_sample
from automated_grounded_theory.async_support import ProviderAllowance
from automated_grounded_theory.batching import request_input_tokens
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.sampling import (
    ModelSamplingSelection,
    SamplingRuntime,
    _IdleModel,
    _analytic_new_keys,
    _closed_direction_keys,
    _messages,
    _number_directions,
    _proposals_for,
    _selection_payload,
    _store_model_selection,
    _target_key,
    _update_ledger,
    rank_sampling_directions,
    retrieve_sampling_passages,
)
from automated_grounded_theory.saturation import stopping_criteria_report
from tests.test_v080 import _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs, _payload


def test_version_is_098():
    assert __version__ == "0.9.19"
    assert POLICY_VERSION == "0.9.19"


def _prep_group(groups, directions):
    thin = next(row for row in directions if _target_key(row) == "thin_category:c-prep")
    return next(group for group in groups if group["direction_id"] == thin["direction_id"])


def test_a_relevant_note_does_not_close_the_direction():
    """The structured judgment is authoritative. Free text cannot close a direction."""
    before = rank_sampling_directions(_corpus())
    assert any(_target_key(item) == "thin_category:c-prep" for item in before)
    relevant = {
        "direction_key": "thin_category:c-prep",
        "judgment": "relevant",
        "note": "This evidence is not irrelevant to the packing question.",
        "passage_id": "p-low",
    }
    corpus = _corpus()
    corpus["metadata"] = {"sampling_ledger": [{"reread_notes": [relevant]}]}
    after = rank_sampling_directions(corpus)
    assert any(_target_key(item) == "thin_category:c-prep" for item in after)
    assert "thin_category:c-prep" not in _closed_direction_keys(corpus["metadata"]["sampling_ledger"])
    assert all(item["kind"] != "reread_follow_up" for item in after)

    rejected = {
        "direction_key": "thin_category:c-prep",
        "judgment": "irrelevant",
        "note": "This evidence is not irrelevant to the packing question.",
        "passage_id": "p-low",
    }
    held = _corpus()
    ledger = [{"reread_notes": [rejected]}]
    held["metadata"] = {"sampling_ledger": ledger}
    ranked = rank_sampling_directions(held)
    assert any(_target_key(item) == "thin_category:c-prep" for item in ranked)
    numbered = _number_directions(ranked, round_index=2)
    groups = retrieve_sampling_passages(numbered, {**held, "sampling_ledger": ledger})
    assert "p-low" not in [row["passage_id"] for row in _prep_group(groups, numbered)["proposals"]]
    control = _number_directions(rank_sampling_directions(_corpus()), round_index=2)
    offered = retrieve_sampling_passages(control, _corpus())
    assert "p-low" in [row["passage_id"] for row in _prep_group(offered, control)["proposals"]]

    closed_note = {
        **relevant,
        "direction_decision": "close",
        "direction_close_reason": "Further passages would not change this question.",
    }
    closed = _corpus()
    closed["metadata"] = {"sampling_ledger": [{"reread_notes": [closed_note]}]}
    assert all(_target_key(item) != "thin_category:c-prep" for item in rank_sampling_directions(closed))
    missing_reason = {**closed_note, "direction_close_reason": ""}
    still_open = _corpus()
    still_open["metadata"] = {"sampling_ledger": [{"reread_notes": [missing_reason]}]}
    assert any(_target_key(item) == "thin_category:c-prep" for item in rank_sampling_directions(still_open))
    human = _corpus()
    human["metadata"] = {"sampling_ledger": [{"rejected_direction_keys": ["thin_category:c-prep"]}]}
    assert all(_target_key(item) != "thin_category:c-prep" for item in rank_sampling_directions(human))

    kept = _analytic_new_keys(["thin_category:c-prep"], [rejected], [])
    assert kept == ["thin_category:c-prep"]
    dropped = _analytic_new_keys(["thin_category:c-prep"], [closed_note], [])
    assert dropped == []
    updated = _update_ledger(
        {
            "saturation": {"n_documents": 4, "points": []},
            "sampling_ledger": [{"round_index": 1, "new_direction_keys": ["thin_category:c-prep"]}],
            "reread_notes": [closed_note],
            "directions": [{"direction_id": "dir-1-0001", "kind": "thin_category", "target_id": "c-prep", "action": ""}],
            "relationships": [],
            "metadata": {"procedure": {"quiet_rounds": 2}},
        }
    )
    assert updated["sampling_ledger"][-1]["analytic_new_direction_keys"] == []
    assert updated["sampling_ledger"][-1]["closed_direction_keys"] == ["thin_category:c-prep"]
    assert updated["sampling_ledger"][-1]["indicators"]["judgment"] is None
    quiet = stopping_criteria_report(
        {"n_documents": 4},
        [
            {"new_direction_keys": [], "analytic_new_direction_keys": [], "reread_unsent_pairs": []},
            {"new_direction_keys": [], "analytic_new_direction_keys": [], "reread_unsent_pairs": []},
        ],
        quiet_rounds=2,
    )
    quiet_row = next(item for item in quiet["criteria"] if item["criterion"] == "sampling_rounds_without_new_directions")
    assert quiet_row["status"] == "met"
    closed_quiet = stopping_criteria_report(
        {"n_documents": 4},
        [
            {"new_direction_keys": ["thin_category:c-prep"], "analytic_new_direction_keys": [], "reread_unsent_pairs": []},
            {"new_direction_keys": [], "analytic_new_direction_keys": [], "reread_unsent_pairs": []},
        ],
        quiet_rounds=2,
    )
    closed_row = next(
        item for item in closed_quiet["criteria"] if item["criterion"] == "sampling_rounds_without_new_directions"
    )
    assert closed_row["status"] == "met"


def _one_pair_state():
    directions = [{"direction_id": "dir-1-0001", "rank": 1, "kind": "thin_category", "target_id": "c-prep"}]
    state = {
        "directions": directions,
        "passages": [{"passage_id": "p1"}],
        "passage_proposals": [],
        "warnings": [],
    }
    parsed = {
        "selections": [{"direction_id": "dir-1-0001", "passage_id": "p1"}],
        "direction_ids": ["dir-1-0001"],
        "passage_ids": ["p1"],
        "rationale": "This pair is the selection.",
    }
    return directions, state, parsed


def test_brief_only_keeps_the_selection_when_no_reread_is_requested():
    runtime = SamplingRuntime(coder=_IdleModel(), reviewer=_IdleModel(), decision_mode="llm")
    runtime.limiter.provider_allowance = ProviderAllowance(
        max_tokens=100000,
        stages={"coding": 0, "review": 2},
    )
    directions, state, parsed = _one_pair_state()
    metadata = {"decision_mode": "llm", "procedure": {"pursuit": "brief_only"}}
    _store_model_selection(runtime, state, metadata, directions, parsed, [])
    pair = {"direction_id": "dir-1-0001", "passage_id": "p1"}
    assert metadata["model_sampling_status"] == "complete"
    assert metadata["model_sampling_status"] != "budget_limited"
    assert metadata.get("sampling_stop") != "call_allowance"
    assert metadata["model_sampling_pairs"] == [pair]
    assert metadata["selected_sampling_pairs"] == [pair]
    assert metadata["scheduled_reread_pairs"] == []
    assert metadata.get("reread_unsent_pairs") in (None, [])

    runtime.limiter.provider_allowance = ProviderAllowance(
        max_tokens=100000,
        stages={"coding": 0, "review": 2},
    )
    reread = {"decision_mode": "llm", "procedure": {"pursuit": "reread"}}
    _store_model_selection(runtime, state, reread, directions, parsed, [])
    assert reread["model_sampling_pairs"] == [pair]
    assert reread["scheduled_reread_pairs"] == []
    assert reread["sampling_stop"] == "call_allowance"
    assert reread["model_sampling_status"] == "budget_limited"
    assert reread["reread_unsent_pairs"] == [
        {
            "direction_id": "dir-1-0001",
            "passage_id": "p1",
            "reason": "call_allowance",
            "question_key": "thin_category:c-prep",
            "direction_key": "thin_category:c-prep",
        }
    ]

    sent = theoretical_sample(
        _corpus(),
        FakeSamplingCoder(),
        decision_mode="llm",
        pursuit="brief_only",
        interrupt=False,
        **_kwargs(max_sampling_review_calls=0, max_decision_calls=4),
    )
    assert sent.metadata["model_sampling_pairs"]
    assert sent.metadata["model_sampling_status"] != "budget_limited"
    assert sent.metadata.get("sampling_stop") != "call_allowance"
    assert not sent.metadata.get("reread_unsent_pairs")


class _MalformedThenRepaired(FakeSamplingCoder):
    """The first selection does not parse. The repair returns one valid pair."""

    selection_calls: int = 0
    bodies: list = []

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "ModelSamplingSelection":
                self.selection_calls += 1
                self.bodies.append(body)
                if self.selection_calls == 1:
                    raw = AIMessage(
                        content="not json",
                        usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                    )
                    return {"raw": raw, "parsed": None, "parsing_error": "malformed"}
                direction = str((body.get("directions") or [{}])[0].get("direction_id") or "")
                groups = body.get("passage_proposals") or [{}]
                proposals = (groups[0] or {}).get("proposals") or [{}]
                passage = str((proposals[0] or {}).get("passage_id") or "")
                parsed = schema(
                    selections=[{"direction_id": direction, "passage_id": passage}],
                    rationale="The repaired selection names one pair from the packed request.",
                )
            elif name == "ModelSaturationJudgment":
                parsed = schema(
                    assessment="not_saturated",
                    rationale="The window is interpretable and a sampling gap is still open.",
                )
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


def test_selection_repair_reuses_the_packed_proposals():
    coder = _MalformedThenRepaired()
    result = theoretical_sample(
        _corpus(),
        coder,
        decision_mode="llm",
        pursuit="brief_only",
        interrupt=False,
        sampling_token_budget=2755,
        max_sampling_review_tokens=10**7,
        **_kwargs(max_decision_calls=6),
    )
    assert coder.selection_calls == 2
    sent_ids = set(result.metadata["selection_pack"]["sent_direction_ids"])
    assert sent_ids
    assert result.metadata["selection_pack"]["omitted_direction_ids"]
    repair_body = coder.bodies[1]
    repair_ids = {str(row.get("direction_id") or "") for row in repair_body.get("passage_proposals") or []}
    assert repair_ids
    assert repair_ids <= sent_ids
    initial_ids = {str(row.get("direction_id") or "") for row in coder.bodies[0].get("passage_proposals") or []}
    assert repair_ids == initial_ids
    repair_rows = [row for row in result.prompt_log if row.get("kind") == "schema_repair"]
    assert repair_rows
    assert all(row.get("received") for row in repair_rows)
    assert all(row.get("reason") != "request_budget" for row in repair_rows)
    assert result.metadata.get("sampling_stop") != "request_budget"
    assert result.metadata["model_sampling_pairs"]

    sent_directions = [row for row in result.directions if str(row.get("direction_id") or "") in sent_ids]
    repair_note = next(
        message["content"]
        for row in repair_rows
        for message in row["messages"]
        if isinstance(message, dict) and str(message.get("content") or "").startswith("REPAIR THIS OUTPUT")
    )
    filtered = _messages(
        "sampling_selection",
        _selection_payload(sent_directions, _proposals_for(result.passage_proposals, sent_directions)),
    )
    filtered.append(HumanMessage(content=repair_note))
    restored = _messages(
        "sampling_selection",
        _selection_payload(sent_directions, result.passage_proposals),
    )
    restored.append(HumanMessage(content=repair_note))
    filtered_estimate = request_input_tokens(filtered, ModelSamplingSelection)
    restored_estimate = request_input_tokens(restored, ModelSamplingSelection)
    assert filtered_estimate <= 2755
    assert restored_estimate > 2755
    logged = json.loads(repair_rows[0]["messages"][1]["content"])
    assert {str(row.get("direction_id") or "") for row in logged.get("passage_proposals") or []} == repair_ids
