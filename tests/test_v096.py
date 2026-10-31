"""0.9.6 review fixes. No network and no paid model calls."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, theoretical_sample
from automated_grounded_theory.records import POLICY_VERSION
from tests.test_v080 import _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs, _payload


def test_version_is_096():
    assert __version__ == "0.9.42"
    assert POLICY_VERSION == "0.9.42"


class _RereadRepairTimeout(FakeSamplingCoder):
    """The first reread does not parse. The repair times out. Resume repairs once."""

    initials: int = 0
    repairs: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _text(messages):
            return json.dumps(
                [{"content": getattr(message, "content", "")} for message in messages],
                ensure_ascii=False,
                default=str,
            )

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "SamplingRereadNote":
                if "REPAIR THIS OUTPUT" in _text(messages):
                    self.repairs += 1
                    if self.repairs == 1:
                        raise TimeoutError("repair timed out")
                    parsed = schema(
                        passage_id=str(body.get("passage_id") or ""),
                        note="Repaired note.",
                        quote=str(body.get("text") or "")[:48],
                    )
                else:
                    self.initials += 1
                    raw = AIMessage(
                        content="not json",
                        usage_metadata={"input_tokens": 18, "output_tokens": 4, "total_tokens": 22},
                    )
                    return {"raw": raw, "parsed": None, "parsing_error": "malformed"}
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


def test_a_reread_repair_timeout_keeps_the_original_response():
    saver = InMemorySaver()
    coder = _RereadRepairTimeout()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=4)
    paused = theoretical_sample(_corpus(), coder, checkpointer=saver, thread_id="reread-repair", **common)
    direction = str(paused.directions[0]["direction_id"])
    partial = theoretical_sample(
        _corpus(),
        coder,
        checkpointer=saver,
        thread_id="reread-repair",
        resuming=True,
        resume={"directions": [{"direction_id": direction, "action": "accept", "sampled_passage_ids": ["p-low"]}]},
        **common,
    )
    assert partial.metadata["status"] == "incomplete"
    assert partial.metadata["stop"] == "timeout"
    assert partial.resumable is True
    assert partial.metadata["reread_repair_pending"] is True
    assert partial.metadata["reread_active_pair"] == {"direction_id": direction, "passage_id": "p-low"}
    assert partial.metadata["provider_budget"]["calls_used"] == 2
    assert coder.initials == 1
    assert coder.repairs == 1
    failed = [row for row in partial.prompt_log if row.get("kind") == "sampling_reread" and row.get("status") == "parse_failed"]
    assert len(failed) == 1
    assert int((failed[0].get("usage") or {}).get("total_tokens") or 0) == 22
    assert any(int((row.get("usage") or {}).get("total_tokens") or 0) > 0 for row in partial.metadata["sampling_attempts"])
    spent = partial.metadata["provider_budget"]["calls_used"]
    resumed = theoretical_sample(
        _corpus(),
        coder,
        checkpointer=saver,
        thread_id="reread-repair",
        resuming=True,
        resume={"continue": True},
        **common,
    )
    assert coder.initials == 1
    assert coder.repairs == 2
    assert resumed.metadata["provider_budget"]["calls_used"] == spent + 1
    assert resumed.metadata["reread_repair_pending"] is False
    assert len([row for row in resumed.reread_notes if row["passage_id"] == "p-low"]) == 1
    assert len([row for row in resumed.prompt_log if row.get("status") == "parse_failed"]) == 1


class _TimeoutOnFirstSend(FakeSamplingCoder):
    rereads: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "SamplingRereadNote":
                self.rereads += 1
                raise TimeoutError("timed out")
            parsed = schema()
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def test_a_withheld_pair_is_not_labeled_as_the_timeout():
    saver = InMemorySaver()
    coder = _TimeoutOnFirstSend()
    huge = {"passage_id": "p-huge", "doc_id": "d1", "text": "pack " * 20000}
    source = _corpus(huge)
    common = _kwargs(pursuit="reread", max_sampling_review_calls=4)
    paused = theoretical_sample(source, coder, checkpointer=saver, thread_id="withheld-then-timeout", **common)
    first, second = [str(row["direction_id"]) for row in paused.directions[:2]]
    partial = theoretical_sample(
        source,
        coder,
        checkpointer=saver,
        thread_id="withheld-then-timeout",
        resuming=True,
        resume={
            "directions": [
                {"direction_id": first, "action": "accept", "sampled_passage_ids": ["p-huge"]},
                {"direction_id": second, "action": "accept", "sampled_passage_ids": ["p-low"]},
            ]
        },
        **common,
    )
    assert coder.rereads == 1
    assert partial.metadata["stop"] == "timeout"
    assert partial.metadata["status"] == "incomplete"
    failed = partial.metadata["reread_failed_pairs"]
    assert [(row["direction_id"], row["passage_id"]) for row in failed] == [(second, "p-low")]
    second_direction = next(row for row in paused.directions if row["direction_id"] == second)
    assert failed[0]["question_key"] == f"{second_direction['kind']}:{second_direction['target_id']}"
    assert partial.metadata["reread_active_pair"] == {"direction_id": second, "passage_id": "p-low"}
    withheld = partial.metadata["reread_unsent_pairs"]
    first_direction = next(row for row in paused.directions if row["direction_id"] == first)
    withheld_row = next(row for row in withheld if row["passage_id"] == "p-huge")
    assert withheld_row["direction_id"] == first
    assert withheld_row["reason"] == "request_budget"
    assert withheld_row["question_key"] == f"{first_direction['kind']}:{first_direction['target_id']}"
    assert all(row.get("passage_id") != "p-huge" for row in partial.metadata["reread_failed_pairs"])
    assert all(row.get("passage_id") != "p-huge" for row in partial.metadata.get("reread_unattempted_pairs") or [])
    assert partial.metadata["provider_budget"]["calls_used"] == 1


class _OneFlatList(FakeSamplingCoder):
    chosen: dict = {}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "ModelSamplingSelection":
                directions = [str(row.get("direction_id") or "") for row in (body.get("directions") or []) if row.get("direction_id")]
                passages = []
                for group in body.get("passage_proposals") or []:
                    for row in group.get("proposals") or []:
                        passage_id = str((row or {}).get("passage_id") or "")
                        if passage_id and passage_id not in passages:
                            passages.append(passage_id)
                self.chosen = {"direction_id": directions[0], "passage_id": passages[0]}
                parsed = schema(
                    selections=[{"direction_id": directions[0], "passage_id": passages[0]}],
                    direction_ids=[directions[0]],
                    rationale="The direction list matches the selection. The passage list was not sent.",
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


def test_one_matching_flat_list_is_not_a_mismatch():
    coder = _OneFlatList()
    result = theoretical_sample(
        _corpus(),
        coder,
        decision_mode="llm",
        pursuit="brief_only",
        interrupt=False,
        **_kwargs(),
    )
    choice = coder.chosen
    assert result.metadata["status"] == "complete"
    assert result.metadata["model_sampling_status"] == "complete"
    assert result.metadata["model_flat_list_mismatch"] is None
    assert result.metadata["model_direction_ids"] == [choice["direction_id"]]
    assert result.metadata["model_passage_ids"] == [choice["passage_id"]]
    assert result.metadata["model_sampling_pairs"] == [choice]
    assert not any("flat id lists" in str(row) for row in result.warnings)
