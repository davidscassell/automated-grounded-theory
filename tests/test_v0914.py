"""0.9.14 regressions. Fake models only. No live evaluation."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    __version__,
    export_gate_relationships,
    load_adjudications,
    score_gate_adjudications,
    theoretical_sample,
)
from automated_grounded_theory.axial import DEFAULT_MAX_AXIAL_REVIEW_TOKENS
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES
from tests.test_v080 import _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs


def test_version_is_0914():
    assert __version__ == "0.9.34"
    assert POLICY_VERSION == "0.9.34"
    assert PATTERN_MIN_CASES == 2
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000


def _verdict(relationship_id: str, verdict: str = "supported") -> dict:
    return {
        "relationship_id": relationship_id,
        "verdict": verdict,
        "adjudicator": "example-adjudicator",
        "timestamp": "2026-01-01T00:00:00Z",
        "reason": "Example reason, not a study finding.",
    }


def _criterion(report: dict, name: str) -> dict:
    return next(row for row in report["criteria"] if row["criterion"] == name)


def test_partial_eligibility_does_not_publish_a_full_population_recall():
    """A supported+blocked row and a supported+unknown row are not a 0.5 recall.

    The full-population rate stays null. The known-only rate uses the one
    supported row whose eligibility is known. The same split is on the rule.
    """
    exported = export_gate_relationships(
        [
            {
                "relationship_id": "A",
                "disposition": "accepted",
                "flags": ["causal_wording_withheld"],
                "downstream_eligible": False,
            },
            {
                "relationship_id": "B",
                "disposition": "accepted",
                "flags": ["causal_wording_withheld"],
            },
        ]
    )
    assert exported["accepted"][1]["downstream_eligible"] is None
    scored = score_gate_adjudications(exported, load_adjudications([_verdict("A"), _verdict("B")]))
    assert scored["supported_adjudicated"] == 2
    assert scored["downstream_withheld_supported_recall"] is None
    assert scored["downstream_withheld_supported_recall_known"] == 1.0
    assert scored["downstream_withheld_supported_recall_known_denominator"] == 1
    assert scored["downstream_withheld_supported_recall_coverage"] == 0.5
    assert scored["downstream_precision"] is None
    assert scored["downstream_precision_known"] is None
    assert scored["downstream_precision_known_denominator"] == 0
    assert scored["downstream_precision_coverage"] == 0.5
    rule = scored["by_rule"]["causal_wording_withheld"]
    assert rule["downstream_withheld_supported_recall"] is None
    assert rule["downstream_withheld_supported_recall_known"] == 1.0
    assert rule["downstream_withheld_supported_recall_known_denominator"] == 1
    assert rule["downstream_withheld_supported_recall_coverage"] == 0.5
    assert rule["downstream_precision"] is None
    assert rule["downstream_precision_known"] is None


def test_known_eligibility_on_an_unsupported_row_does_not_zero_recall():
    """No supported row has known eligibility. An unrelated known row is not the denominator."""
    exported = export_gate_relationships(
        [
            {
                "relationship_id": "B",
                "disposition": "accepted",
                "flags": ["causal_wording_withheld"],
            },
            {
                "relationship_id": "C",
                "disposition": "accepted",
                "flags": ["causal_wording_withheld"],
                "downstream_eligible": True,
            },
        ]
    )
    scored = score_gate_adjudications(
        exported,
        load_adjudications([_verdict("B", "supported"), _verdict("C", "unsupported")]),
    )
    assert scored["supported_adjudicated"] == 1
    assert scored["downstream_withheld_supported_recall"] is None
    assert scored["downstream_withheld_supported_recall_known"] is None
    assert scored["downstream_withheld_supported_recall_known_denominator"] == 0
    assert scored["downstream_withheld_supported_recall_coverage"] == 0.0
    assert scored["downstream_precision"] is None
    assert scored["downstream_precision_known"] == 0.0
    assert scored["downstream_precision_known_denominator"] == 1
    assert scored["downstream_precision_coverage"] == 0.5
    rule = scored["by_rule"]["causal_wording_withheld"]
    assert rule["downstream_withheld_supported_recall"] is None
    assert rule["downstream_withheld_supported_recall_known"] is None
    assert rule["downstream_withheld_supported_recall_coverage"] == 0.0
    assert rule["downstream_precision"] is None
    assert rule["downstream_precision_known"] == 0.0


def test_identified_eligibility_keeps_the_full_population_rate():
    exported = export_gate_relationships(
        [
            {
                "relationship_id": "A",
                "disposition": "accepted",
                "flags": ["causal_wording_withheld"],
                "downstream_eligible": False,
            },
            {
                "relationship_id": "D",
                "disposition": "accepted",
                "downstream_eligible": True,
            },
        ]
    )
    scored = score_gate_adjudications(exported, load_adjudications([_verdict("A"), _verdict("D")]))
    assert scored["downstream_withheld_supported_recall"] == 0.5
    assert scored["downstream_withheld_supported_recall_known"] == 0.5
    assert scored["downstream_withheld_supported_recall_known_denominator"] == 2
    assert scored["downstream_withheld_supported_recall_coverage"] == 1.0
    assert scored["downstream_precision"] == 1.0
    assert scored["downstream_precision_known"] == 1.0
    assert scored["downstream_precision_known_denominator"] == 1
    assert scored["downstream_precision_coverage"] == 1.0


def test_an_export_without_eligibility_leaves_the_new_rates_null():
    exported = {
        "accepted": [{"relationship_id": "rel", "disposition": "accepted", "rules": ["stem_only"]}],
        "withheld": [],
        "proposed": [],
    }
    scored = score_gate_adjudications(exported, load_adjudications([_verdict("rel")]))
    assert scored["accept_precision"] == 1.0
    assert scored["downstream_precision"] is None
    assert scored["downstream_precision_known"] is None
    assert scored["downstream_withheld_supported_recall"] is None
    assert scored["downstream_withheld_supported_recall_known"] is None
    assert scored["downstream_withheld_supported_recall_coverage"] == 0.0
    rule = scored["by_rule"]["stem_only"]
    assert rule["downstream_withheld_supported_recall"] is None
    assert rule["downstream_withheld_supported_recall_known"] is None
    assert rule["downstream_precision"] is None
    assert rule["downstream_precision_known"] is None


def _three_document_source() -> dict:
    source = _corpus(
        {"passage_id": "p-mid", "doc_id": "d2", "text": "The suitcase was already by the door on Tuesday."},
        documents=("d1", "d2", "d3"),
    )
    assert {row["doc_id"] for row in source["passages"]} == {"d1", "d2", "d3"}
    source["saturation"] = {
        "n_documents": 3,
        "saturation_window": "interpretable",
        "window_new_code_rate": 0.0,
        "window_new_property_rate": 0.0,
        "window_new_relationship_rate": 0.0,
        "code_frequency_shift": 0.0,
        "judgment": None,
    }
    source["metadata"] = {
        "sampling_ledger": [
            {
                "round_index": 1,
                "direction_keys": ["thin_category:c-prep"],
                "new_direction_keys": [],
                "analytic_new_direction_keys": [],
                "reread_unsent_pairs": [],
                "indicators": {"window_new_code_rate": 0.0},
            }
        ]
    }
    return source


def _rate_rows(report: dict) -> list[dict]:
    return [_criterion(report, name) for name in ("new_codes", "new_properties", "new_relationships")]


def test_llm_saturation_call_follows_configured_min_documents():
    """Three documents and two rounds do not spend a sufficiency call when min_documents is 10.

    The same corpus with min_documents 2 does spend it. The stopping report's
    rate rows stay undefined at 10.
    """
    coder = FakeSamplingCoder(judgment_assessment="sufficient_for_this_question")
    withheld = theoretical_sample(
        _three_document_source(),
        coder,
        decision_mode="llm",
        interrupt=False,
        **_kwargs(min_documents=10),
    )
    judgment = withheld.metadata["model_saturation_judgment"]
    assert judgment["status"] == "not_interpretable"
    assert judgment["assessment"] is None
    assert judgment["minimum_documents"] == 10
    assert judgment["n_documents"] == 3
    assert judgment["sampling_rounds"] == 2
    assert "below 10" in judgment["note"]
    assert withheld.metadata["saturation_minimum_documents"] == 10
    assert withheld.metadata["saturation_window_ready"] is False
    assert withheld.metadata.get("saturation_decision") is None
    assert withheld.saturation["judgment"] is None
    assert coder.call_count == 1
    assert not any(row.get("kind") == "saturation_decision" for row in withheld.prompt_log)
    assert withheld.metadata["provider_budget"]["stages"]["review"]["calls_used"] == 1
    assert len(withheld.sampling_ledger) == 2
    for row in _rate_rows(withheld.stopping_criteria):
        assert row["status"] == "undefined"
        assert row["threshold"]["min_documents"] == 10

    spent = FakeSamplingCoder(judgment_assessment="sufficient_for_this_question")
    recorded = theoretical_sample(
        _three_document_source(),
        spent,
        decision_mode="llm",
        interrupt=False,
        **_kwargs(min_documents=2),
    )
    recorded_judgment = recorded.metadata["model_saturation_judgment"]
    assert recorded_judgment["status"] == "recorded"
    assert recorded_judgment["assessment"] == "sufficient_for_this_question"
    assert recorded_judgment["n_documents"] == 3
    assert recorded.metadata["saturation_minimum_documents"] == 2
    assert recorded.metadata["saturation_window_ready"] is True
    assert recorded.saturation["judgment"] is None
    assert recorded.metadata.get("saturation_decision") is None
    assert spent.call_count == 2
    assert any(row.get("kind") == "saturation_decision" for row in recorded.prompt_log)
    assert recorded.metadata["provider_budget"]["stages"]["review"]["calls_used"] == 2
    for row in _rate_rows(recorded.stopping_criteria):
        assert row["status"] == "met"
        assert row["threshold"]["min_documents"] == 2


def _human_saturation(min_documents: int, thread_id: str):
    source = _three_document_source()
    saver = InMemorySaver()
    common = _kwargs(min_documents=min_documents)
    paused = theoretical_sample(source, checkpointer=saver, thread_id=thread_id, **common)
    assert paused.interrupt_payload["kind"] == "theoretical_sampling"
    mid = theoretical_sample(
        source,
        checkpointer=saver,
        thread_id=thread_id,
        resuming=True,
        resume={"accept": True},
        **common,
    )
    assert mid.interrupted
    assert mid.interrupt_payload["kind"] == "saturation_decision"
    done = theoretical_sample(
        source,
        checkpointer=saver,
        thread_id=thread_id,
        resuming=True,
        resume={"decision": "not_saturated", "rationale": "The researcher has not claimed saturation."},
        **common,
    )
    assert done.interrupted is False
    return mid, done


def test_human_saturation_uses_configured_min_documents():
    below, finished = _human_saturation(10, "human-min-10")
    payload = below.interrupt_payload
    assert payload["minimum_documents"] == 10
    assert payload["window_ready"] is False
    assert "below 10" in payload["window_reason"]
    assert payload["saturation"]["n_documents"] == 3
    assert len(payload["sampling_ledger"]) == 2
    for row in _rate_rows(payload["stopping_criteria"]):
        assert row["status"] == "undefined"
        assert row["threshold"]["min_documents"] == 10
    decision = finished.metadata["saturation_decision"]
    assert decision["decision"] == "not_saturated"
    assert decision["actor"] == "human"
    assert decision["minimum_documents"] == 10
    assert decision["window_ready"] is False
    assert decision["n_documents"] == 3
    assert finished.metadata["saturation_minimum_documents"] == 10
    assert finished.metadata["saturation_window_ready"] is False
    assert finished.metadata.get("model_saturation_judgment") is None
    assert finished.saturation["judgment"] is None
    assert finished.metadata["provider_budget"]["stages"]["review"]["calls_used"] == 0

    ready, recorded = _human_saturation(2, "human-min-2")
    assert ready.interrupt_payload["minimum_documents"] == 2
    assert ready.interrupt_payload["window_ready"] is True
    for row in _rate_rows(ready.interrupt_payload["stopping_criteria"]):
        assert row["status"] == "met"
        assert row["threshold"]["min_documents"] == 2
    assert recorded.metadata["saturation_decision"]["minimum_documents"] == 2
    assert recorded.metadata["saturation_decision"]["window_ready"] is True
    assert recorded.metadata.get("model_saturation_judgment") is None
    assert recorded.saturation["judgment"] is None
