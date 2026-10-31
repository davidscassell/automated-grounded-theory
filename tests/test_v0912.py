"""0.9.12 regressions. Fake models only. No live evaluation."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, theoretical_sample
from automated_grounded_theory.axial import DEFAULT_MAX_AXIAL_REVIEW_TOKENS
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES
from automated_grounded_theory.saturation import STOPPING_QUIET_ROUNDS, stopping_criteria_report
from tests.test_v080 import _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs, _payload


def test_version_is_0912():
    assert __version__ == "0.9.42"
    assert POLICY_VERSION == "0.9.42"
    assert PATTERN_MIN_CASES == 2
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000
    assert STOPPING_QUIET_ROUNDS == 2


def _quiet(report):
    return next(row for row in report["criteria"] if row["criterion"] == "sampling_rounds_without_new_directions")


class _TimeoutOnce(FakeSamplingCoder):
    """The first reread raises. The resume completes that same pair."""

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
                    note="The passage bears on the question.",
                    quote=str(body.get("text") or "")[:48],
                    judgment="relevant",
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


def test_a_recovered_timeout_does_not_block_quiet_rounds():
    """Timeout, resume, complete. The default two-round window can then be met.

    An unresolved timeout still leaves the criterion not_evaluable.
    """
    seed = theoretical_sample(_corpus(), interrupt=False, **_kwargs())
    saver = InMemorySaver()
    coder = _TimeoutOnce()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=2)
    paused = theoretical_sample(seed, coder, checkpointer=saver, thread_id="recover-timeout", **common)
    direction = paused.directions[0]
    partial = theoretical_sample(
        seed,
        coder,
        checkpointer=saver,
        thread_id="recover-timeout",
        resuming=True,
        resume={
            "directions": [
                {
                    "direction_id": direction["direction_id"],
                    "action": "accept",
                    "sampled_passage_ids": ["p-low"],
                }
            ]
        },
        **common,
    )
    assert partial.metadata["stop"] == "timeout"
    assert partial.resumable is True
    blocked = _quiet(stopping_criteria_report(partial.saturation, partial.sampling_ledger))
    assert blocked["status"] == "not_evaluable"
    assert blocked["observed"] is None
    timed = partial.sampling_ledger[-1]
    assert timed["reread_stop"] == "timeout"
    assert timed["reread_status"] == "timeout"

    recovered = theoretical_sample(
        seed,
        coder,
        checkpointer=saver,
        thread_id="recover-timeout",
        resuming=True,
        resume={"continue": True},
        **common,
    )
    row = recovered.sampling_ledger[-1]
    assert row["reread_status"] == "complete"
    assert row["reread_stop"] == ""
    assert row["reread_failed_pairs"] == []
    assert row["reread_unsent_pairs"] == []
    assert row["reread_recovered_timeouts"]
    assert row["reread_recovered_timeouts"][0]["prior_reread_stop"] == "timeout"
    assert row["reread_recovered_timeouts"][0]["recovered"] is True
    assert any(
        item.get("status") == "recovered" and item.get("reason") == "timeout"
        for item in recovered.metadata.get("sampling_attempts") or []
    )
    later = theoretical_sample(recovered, interrupt=False, **_kwargs())
    quiet = _quiet(later.stopping_criteria)
    assert quiet["status"] == "met"
    assert quiet["observed"] is not None
    assert all(not keys for keys in quiet["observed"])


class _CloseTheQuestion(FakeSamplingCoder):
    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "SamplingRereadNote":
                parsed = schema(
                    passage_id=str(body.get("passage_id") or ""),
                    note="This passage settles the thin category.",
                    quote=str(body.get("text") or "")[:48],
                    judgment="relevant",
                    direction_decision="close",
                    direction_close_reason="The thin category is resolved for this question.",
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


def test_a_closed_question_exempts_the_unsent_passage_from_quiet_rounds():
    """One question, two passages, one reread. The completed note closes the question.

    The unsent passage is a production record. Its question key matches the
    closure. Later rounds can meet the quiet criterion.
    """
    seed = theoretical_sample(_corpus(), interrupt=False, **_kwargs())
    saver = InMemorySaver()
    coder = _CloseTheQuestion()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=1)
    paused = theoretical_sample(seed, coder, checkpointer=saver, thread_id="close-unsent", **common)
    prep = next(row for row in paused.directions if row["kind"] == "thin_category" and row["target_id"] == "c-prep")
    closed = theoretical_sample(
        seed,
        coder,
        checkpointer=saver,
        thread_id="close-unsent",
        resuming=True,
        resume={
            "directions": [
                {
                    "direction_id": prep["direction_id"],
                    "action": "accept",
                    "sampled_passage_ids": ["p-low", "p-monday"],
                }
            ]
        },
        **common,
    )
    row = closed.sampling_ledger[-1]
    assert row["closed_direction_keys"] == ["thin_category:c-prep"]
    assert len(row["reread_unsent_pairs"]) == 1
    unsent = row["reread_unsent_pairs"][0]
    assert unsent["reason"] == "call_allowance"
    assert unsent["direction_id"] == prep["direction_id"]
    assert unsent["direction_id"].startswith("dir-")
    assert unsent["question_key"] == "thin_category:c-prep"
    assert unsent["direction_key"] == "thin_category:c-prep"
    assert unsent["question_key"] != unsent["direction_id"]
    assert {(item["passage_id"], item["question_key"]) for item in row["sampled_pairs"]} == {
        ("p-low", "thin_category:c-prep"),
        ("p-monday", "thin_category:c-prep"),
    }
    later = theoretical_sample(closed, interrupt=False, **_kwargs())
    quiet = _quiet(later.stopping_criteria)
    assert quiet["status"] == "met"
    assert quiet["observed"] is not None
