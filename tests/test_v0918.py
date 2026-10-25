"""0.9.18 review fixes. Fakes only. No network calls."""

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


_PRICES = {"input_usd_per_million": 3.0, "output_usd_per_million": 15.0}


def test_version_is_0918():
    assert __version__ == "0.9.26"
    assert POLICY_VERSION == "0.9.26"


def _priced_corpus() -> dict:
    corpus = _corpus()
    corpus["metadata"] = {"token_prices": dict(_PRICES)}
    return corpus


def _usage(incoming: int, outgoing: int) -> dict:
    return {"input_tokens": incoming, "output_tokens": outgoing, "total_tokens": incoming + outgoing}


class _InitialTimeoutThenSuccess(FakeSamplingCoder):
    """The first reread times out. The resume of that same passage succeeds."""

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
                if self.rereads == 1:
                    raise TimeoutError("timed out")
                parsed = schema(
                    passage_id=str(body.get("passage_id") or ""),
                    note="A note about this passage.",
                    quote=str(body.get("text") or "")[:48],
                    judgment="relevant",
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
            raw = AIMessage(content="{}", usage_metadata=_usage(30, 12))
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


class _RepairTimeoutThenSuccess(FakeSamplingCoder):
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
                        judgment="relevant",
                    )
                else:
                    self.initials += 1
                    raw = AIMessage(content="not json", usage_metadata=_usage(18, 4))
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
            raw = AIMessage(content="{}", usage_metadata=_usage(30, 12))
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


class _TimeoutAndStop(FakeSamplingCoder):
    rereads: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            if name == "SamplingRereadNote":
                self.rereads += 1
                raise TimeoutError("timed out")
            parsed = schema()
            if not include_raw:
                return parsed
            raw = AIMessage(content="{}", usage_metadata=_usage(30, 12))
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def _assign(direction_id: str) -> dict:
    return {
        "directions": [
            {"direction_id": direction_id, "action": "accept", "sampled_passage_ids": ["p-low"]}
        ]
    }


def _incomplete_rates(report: dict) -> None:
    assert report["reread_expenditure_complete"] is False
    assert report["expenditure_tokens_per_relevant_reread"] is None
    assert report["expenditure_cost_per_relevant_reread_usd"] is None
    assert report["reread_expenditure_cost_usd"] is None


def test_an_initial_reread_timeout_stays_unknown_after_a_successful_retry():
    """A timed-out reread is a second attempt. The later success does not cover it.

    The successful response reports 42 tokens. Prices are recorded. The full
    cost stays null because the timed-out attempt has no usage.
    """
    saver = InMemorySaver()
    coder = _InitialTimeoutThenSuccess()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=4)
    paused = theoretical_sample(_priced_corpus(), coder, checkpointer=saver, thread_id="initial-timeout", **common)
    direction = str(paused.directions[0]["direction_id"])
    partial = theoretical_sample(
        _priced_corpus(),
        coder,
        checkpointer=saver,
        thread_id="initial-timeout",
        resuming=True,
        resume=_assign(direction),
        **common,
    )
    assert partial.metadata["stop"] == "timeout"
    assert partial.metadata["provider_budget"]["calls_used"] == 1
    assert coder.rereads == 1
    timed_out = [row for row in partial.prompt_log if row.get("status") == "timeout"]
    assert len(timed_out) == 1
    assert timed_out[0]["kind"] == "sampling_reread"
    assert "usage" not in timed_out[0]
    resumed = theoretical_sample(
        _priced_corpus(),
        coder,
        checkpointer=saver,
        thread_id="initial-timeout",
        resuming=True,
        resume={"continue": True},
        **common,
    )
    assert coder.rereads == 2
    assert resumed.metadata["provider_budget"]["calls_used"] == 2
    assert len([row for row in resumed.reread_notes if row["passage_id"] == "p-low"]) == 1
    report = resumed.metadata["reread_interpretations"]
    assert report["reread_expenditure_calls"] == 2
    assert report["reread_expenditure_calls_with_usage"] == 1
    assert report["reread_expenditure_usage_coverage"] == 0.5
    assert report["reread_expenditure_tokens"] == 42
    assert report["reread_expenditure_input_tokens"] == 30
    assert report["reread_expenditure_output_tokens"] == 12
    assert report["relevant_reread_count"] == 1
    assert report["tokens_per_relevant_reread"] == 42
    _incomplete_rates(report)
    still = [row for row in resumed.prompt_log if row.get("status") == "timeout"]
    assert len(still) == 1
    assert still[0]["kind"] == "sampling_reread"


def test_a_reread_repair_timeout_stays_unknown_after_a_successful_retry():
    """The failed parse, the timed-out repair, and the later repair are three attempts.

    The known sum is the two responses that reported usage. The timeout is the
    third attempt and leaves the full rates null. The provider budget still
    counts every call.
    """
    saver = InMemorySaver()
    coder = _RepairTimeoutThenSuccess()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=4)
    paused = theoretical_sample(_priced_corpus(), coder, checkpointer=saver, thread_id="repair-timeout", **common)
    direction = str(paused.directions[0]["direction_id"])
    partial = theoretical_sample(
        _priced_corpus(),
        coder,
        checkpointer=saver,
        thread_id="repair-timeout",
        resuming=True,
        resume=_assign(direction),
        **common,
    )
    assert partial.metadata["stop"] == "timeout"
    assert partial.metadata["reread_repair_pending"] is True
    assert partial.metadata["provider_budget"]["calls_used"] == 2
    timed_out = [row for row in partial.prompt_log if row.get("status") == "timeout"]
    assert len(timed_out) == 1
    assert timed_out[0]["kind"] == "schema_repair"
    assert timed_out[0]["schema"] == "SamplingRereadNote"
    assert "usage" not in timed_out[0]
    spent = partial.metadata["provider_budget"]["calls_used"]
    resumed = theoretical_sample(
        _priced_corpus(),
        coder,
        checkpointer=saver,
        thread_id="repair-timeout",
        resuming=True,
        resume={"continue": True},
        **common,
    )
    assert resumed.metadata["provider_budget"]["calls_used"] == spent + 1
    assert coder.initials == 1
    assert coder.repairs == 2
    report = resumed.metadata["reread_interpretations"]
    assert report["reread_expenditure_calls"] == 3
    assert report["reread_expenditure_calls_with_usage"] == 2
    assert report["reread_expenditure_usage_coverage"] == round(2 / 3, 4)
    assert report["reread_expenditure_tokens"] == 64
    assert report["reread_expenditure_input_tokens"] == 48
    assert report["reread_expenditure_output_tokens"] == 16
    _incomplete_rates(report)
    assert len([row for row in resumed.prompt_log if row.get("status") == "timeout"]) == 1
    assert len([row for row in resumed.prompt_log if row.get("status") == "parse_failed"]) == 1


def test_a_reread_timeout_with_no_later_success_is_not_a_complete_zero():
    """Prices are set, and the only attempt timed out. Cost stays null, not 0."""
    saver = InMemorySaver()
    coder = _TimeoutAndStop()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=4)
    paused = theoretical_sample(_priced_corpus(), coder, checkpointer=saver, thread_id="timeout-only", **common)
    direction = str(paused.directions[0]["direction_id"])
    partial = theoretical_sample(
        _priced_corpus(),
        coder,
        checkpointer=saver,
        thread_id="timeout-only",
        resuming=True,
        resume=_assign(direction),
        **common,
    )
    assert partial.metadata["stop"] == "timeout"
    assert partial.metadata["status"] == "incomplete"
    assert partial.reread_notes == []
    assert coder.rereads == 1
    assert partial.metadata["provider_budget"]["calls_used"] == 1
    report = partial.metadata["reread_interpretations"]
    assert report["reread_expenditure_calls"] == 1
    assert report["reread_expenditure_calls_with_usage"] == 0
    assert report["reread_expenditure_usage_coverage"] == 0.0
    assert report["reread_expenditure_tokens"] == 0
    assert report["reread_expenditure_input_tokens"] == 0
    assert report["reread_expenditure_output_tokens"] == 0
    _incomplete_rates(report)
    timed_out = [row for row in partial.prompt_log if row.get("status") == "timeout"]
    assert len(timed_out) == 1
    assert timed_out[0]["kind"] == "sampling_reread"
