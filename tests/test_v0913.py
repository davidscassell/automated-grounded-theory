"""0.9.13 regressions. Fake models only. No live evaluation."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, theoretical_sample
from automated_grounded_theory.axial import DEFAULT_MAX_AXIAL_REVIEW_TOKENS, _tiered_accept
from automated_grounded_theory.gate_evaluation import (
    export_gate_relationships,
    load_adjudications,
    score_gate_adjudications,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES
from automated_grounded_theory.saturation import STOPPING_QUIET_ROUNDS, stopping_criteria_report
from tests.test_v080 import _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs, _payload


def test_version_is_0913():
    assert __version__ == "0.9.33"
    assert POLICY_VERSION == "0.9.33"
    assert PATTERN_MIN_CASES == 2
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000
    assert STOPPING_QUIET_ROUNDS == 2


def _quiet(report):
    return next(row for row in report["criteria"] if row["criterion"] == "sampling_rounds_without_new_directions")


class _TimeoutAttempts(FakeSamplingCoder):
    """Raise TimeoutError on the listed 1-based reread attempts. Later attempts succeed."""

    rereads: int = 0
    timeout_on: str = "1"

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
                timed = {int(part) for part in str(self.timeout_on).split(",") if str(part).strip()}
                if self.rereads in timed:
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


def _assign(seed, coder, thread_id, passage_ids, *, calls):
    saver = InMemorySaver()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=calls)
    paused = theoretical_sample(seed, coder, checkpointer=saver, thread_id=thread_id, **common)
    direction = paused.directions[0]
    partial = theoretical_sample(
        seed,
        coder,
        checkpointer=saver,
        thread_id=thread_id,
        resuming=True,
        resume={
            "directions": [
                {
                    "direction_id": direction["direction_id"],
                    "action": "accept",
                    "sampled_passage_ids": list(passage_ids),
                }
            ]
        },
        **common,
    )
    return saver, common, partial


def _resume(seed, coder, saver, common, thread_id):
    return theoretical_sample(
        seed,
        coder,
        checkpointer=saver,
        thread_id=thread_id,
        resuming=True,
        resume={"continue": True},
        **common,
    )


def _ids(rows):
    return [str(row.get("passage_id") or "") for row in rows or [] if isinstance(row, dict)]


def test_a_recovered_pair_is_not_failed_when_a_later_pair_hits_the_allowance():
    """Three passages, three calls. The first times out and its retry succeeds.

    The second succeeds. The third is withheld for call_allowance. The recovered
    passage is completed and is not also failed. The timeout stays in the
    history, and the current stop is the allowance.
    """
    seed = theoretical_sample(_corpus(), interrupt=False, **_kwargs())
    coder = _TimeoutAttempts(timeout_on="1")
    _saver, _common, partial = _assign(seed, coder, "allowance-after-timeout", ["p-low", "p-monday", "p-none"], calls=3)
    assert partial.metadata["stop"] == "timeout"
    recovered = _resume(seed, coder, _saver, _common, "allowance-after-timeout")
    row = recovered.sampling_ledger[-1]
    noted = {item["passage_id"] for item in recovered.reread_notes}
    assert noted == {"p-low", "p-monday"}
    failed = _ids(row["reread_failed_pairs"])
    assert failed == []
    assert set(failed).isdisjoint(noted)
    completed = set(_ids(recovered.metadata.get("reread_completed_pairs")))
    assert completed == {"p-low", "p-monday"}
    assert row["reread_stop"] == "call_allowance"
    assert recovered.metadata["reread_stop"] == "call_allowance"
    assert row["reread_stop"] != "timeout"
    unsent = row["reread_unsent_pairs"]
    assert len(unsent) == 1
    assert unsent[0]["passage_id"] == "p-none"
    assert unsent[0]["reason"] == "call_allowance"
    history = row["reread_recovered_timeouts"]
    assert history
    assert history[0]["recovered"] is True
    assert history[0]["prior_reread_stop"] == "timeout"
    assert _ids(history[0]["pairs"]) == ["p-low"]
    assert any(
        item.get("status") == "recovered" and item.get("reason") == "timeout" and _ids(item.get("pairs")) == ["p-low"]
        for item in recovered.metadata.get("sampling_attempts") or []
    )


def test_one_of_two_timeouts_recovers_and_the_other_stays_failed():
    """The first passage times out and then succeeds. The second times out and stays failed."""
    seed = theoretical_sample(_corpus(), interrupt=False, **_kwargs())
    coder = _TimeoutAttempts(timeout_on="1,3")
    saver, common, first = _assign(seed, coder, "one-of-two-timeouts", ["p-low", "p-monday"], calls=3)
    assert first.metadata["stop"] == "timeout"
    second = _resume(seed, coder, saver, common, "one-of-two-timeouts")
    assert second.metadata["stop"] == "timeout"
    assert second.metadata["reread_stop"] == "timeout"
    assert _ids(second.metadata["reread_failed_pairs"]) == ["p-monday"]
    assert "p-low" not in _ids(second.metadata["reread_failed_pairs"])
    assert {item["passage_id"] for item in second.reread_notes} == {"p-low"}
    history = second.metadata["reread_recovered_timeouts"]
    assert _ids(history[0]["pairs"]) == ["p-low"]
    assert history[0]["recovered"] is True
    ledger = second.sampling_ledger[-1]
    assert _ids(ledger["reread_failed_pairs"]) == ["p-monday"]
    assert _ids(ledger["reread_recovered_timeouts"][0]["pairs"]) == ["p-low"]
    assert ledger["reread_stop"] == "timeout"


def test_a_fully_recovered_batch_still_meets_quiet_rounds():
    """One timeout that then completes still leaves the default window evaluable."""
    seed = theoretical_sample(_corpus(), interrupt=False, **_kwargs())
    coder = _TimeoutAttempts(timeout_on="1")
    saver, common, partial = _assign(seed, coder, "full-recovery", ["p-low"], calls=2)
    blocked = _quiet(stopping_criteria_report(partial.saturation, partial.sampling_ledger))
    assert blocked["status"] == "not_evaluable"
    assert blocked["observed"] is None
    recovered = _resume(seed, coder, saver, common, "full-recovery")
    row = recovered.sampling_ledger[-1]
    assert row["reread_status"] == "complete"
    assert row["reread_stop"] == ""
    assert row["reread_failed_pairs"] == []
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


def _adjudication(**overrides):
    row = {
        "relationship_id": "rel-causal",
        "rules": ["consequence_wording"],
        "verdict": "supported",
        "adjudicator": "example-adjudicator",
        "timestamp": "2026-01-01T00:00:00Z",
        "reason": "Example reason, not a study finding.",
    }
    row.update(overrides)
    return row


def test_downstream_withholding_is_scored_apart_from_disposition():
    """An accepted row blocked by causal wording is not a downstream accept.

    Disposition metrics stay on their old names. A 0.9.9 export without
    downstream_eligible scores the new rates as null.
    """
    row = {
        "relationship_id": "rel-causal",
        "statement": "The tools for the shift leads to coworkers on the crew.",
        "disposition": "proposed",
        "evidence_tier": 2,
        "evidence_label": "pattern_supported",
        "evidence_condition": "anchored",
        "flags": [],
    }
    _tiered_accept(row, field="human_disposition", actor="human")
    assert row["disposition"] == "accepted"
    assert row["human_disposition"] == "accepted"
    assert row["downstream_eligible"] is False
    assert "causal_wording_withheld" in row["flags"]
    assert "wording_exceeds_tier" in row["flags"]

    exported = export_gate_relationships([row])
    copied = exported["accepted"][0]
    assert copied["disposition"] == "accepted"
    assert copied["downstream_eligible"] is False
    assert copied["withholding_flags"] == ["causal_wording_withheld"]
    assert exported["withheld"] == []
    adjudication = _adjudication(downstream_eligible=True, withholding_flags=["ignored"])
    loaded = load_adjudications([adjudication])
    assert loaded[0]["downstream_eligible"] is True
    assert loaded[0]["withholding_flags"] == ["ignored"]
    scored = score_gate_adjudications(exported, loaded)
    assert scored["accepted_adjudicated"] == 1
    assert scored["withheld_adjudicated"] == 0
    assert scored["accept_precision"] == 1.0
    assert scored["withheld_supported_recall"] == 0.0
    assert scored["downstream_precision"] is None
    assert scored["downstream_withheld_supported_recall"] == 1.0
    assert scored["downstream_withheld_supported"] == 1
    assert scored["by_rule"]["causal_wording_withheld"]["downstream_withheld_supported_recall"] == 1.0
    assert "_downstream_known" not in scored

    legacy = {
        "accepted": [{"relationship_id": "rel-causal", "disposition": "accepted", "rules": ["consequence_wording"]}],
        "withheld": [],
        "proposed": [],
    }
    legacy_scored = score_gate_adjudications(legacy, [_adjudication()])
    assert legacy_scored["accept_precision"] == 1.0
    assert legacy_scored["withheld_supported_recall"] == 0.0
    assert legacy_scored["downstream_precision"] is None
    assert legacy_scored["downstream_withheld_supported_recall"] is None

    eligible = {
        "relationship_id": "rel-plain",
        "statement": "The tools for the shift sit beside coworkers on the crew.",
        "disposition": "proposed",
        "evidence_tier": 2,
        "evidence_label": "pattern_supported",
        "evidence_condition": "anchored",
        "flags": [],
    }
    _tiered_accept(eligible, field="human_disposition", actor="human")
    assert eligible["downstream_eligible"] is True
    plain = export_gate_relationships([eligible])
    plain_scored = score_gate_adjudications(plain, [_adjudication(relationship_id="rel-plain")])
    assert plain_scored["downstream_precision"] == 1.0
    assert plain_scored["downstream_withheld_supported_recall"] == 0.0
