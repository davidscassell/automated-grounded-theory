"""0.9.21 discovery evaluation. Fakes only. No network calls and no real adjudications."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automated_grounded_theory import (
    DISCOVERY_SCHEMA_VERSION,
    DiscoveryEvaluationError,
    __version__,
    freeze_discovery_set,
    load_discovery_file,
    score_discovery,
)
from automated_grounded_theory.records import POLICY_VERSION

_RATES = (
    "discovery_recall",
    "discovery_precision",
    "discovery_distortion_rate",
    "gate_false_negative_rate",
    "gate_false_negative_rate_known",
    "discovery_recovered_per_token",
    "discovery_recovered_per_dollar",
)

_EXAMPLE = Path("examples/discovery_adjudication_example.json")
_TEMPLATE = Path("examples/discovery_adjudication_template.csv")


def test_package_version_is_0921():
    assert __version__ == "0.9.21"
    assert POLICY_VERSION == "0.9.21"
    assert DISCOVERY_SCHEMA_VERSION == "discovery-eval-1"


def _gold(adjudication_id, from_id, to_id, passage_id, **extra):
    row = {
        "adjudication_id": adjudication_id,
        "kind": "independent",
        "adjudicator_id": "example-adjudicator",
        "rationale": f"Synthetic rationale for {adjudication_id}.",
        "from_id": from_id,
        "to_id": to_id,
        "direction": "from_to",
        "supporting_passages": [{"passage_id": passage_id, "quote": "Synthetic quote only."}],
        "evidence_supported": True,
    }
    row.update(extra)
    return row


def _proposal(relationship_id, from_id, to_id, passage_id, **extra):
    row = {
        "relationship_id": relationship_id,
        "from_id": from_id,
        "to_id": to_id,
        "passage_id": passage_id,
        "stated_direction": "from_to",
        "downstream_eligible": True,
        "disposition": "accepted",
    }
    row.update(extra)
    return row


def _frozen(adjudications, disagreements=None):
    return freeze_discovery_set(
        {
            "label": "Synthetic example data only. Not a study result.",
            "note": "Synthetic example data only. Not a study result.",
            "adjudications": adjudications,
            "disagreements": disagreements or [],
        }
    )


def _by_id(report, adjudication_id):
    return next(row for row in report["relationships"] if row["adjudication_id"] == adjudication_id)


def test_no_adjudications_and_empty_frozen_set_leave_rates_null():
    proposals = [_proposal("r1", "c-a", "c-b", "p-1")]
    absent = score_discovery(proposals, None)
    assert absent["adjudications_absent"] is True
    assert absent["metric_source"] is None
    assert absent["model_judgments_used"] is False
    assert absent["tier_labels_used"] is False
    for name in _RATES:
        assert absent[name] is None
    assert absent["discovery_recall_denominator"] is None
    assert absent["recovered_count"] is None
    assert absent["withheld_but_supported_count"] is None

    empty_gold = _frozen([])
    no_proposals = score_discovery([], empty_gold)
    assert no_proposals["adjudications_absent"] is False
    assert no_proposals["metric_source"] == "human_adjudication"
    for name in _RATES:
        assert no_proposals[name] is None
    assert no_proposals["discovery_recall_denominator"] == 0
    assert no_proposals["discovery_precision_denominator"] == 0
    assert no_proposals["discovery_distortion_denominator"] == 0
    assert no_proposals["gate_false_negative_population"] == 0

    empty = score_discovery(proposals, empty_gold)
    assert empty["discovery_recall"] is None
    assert empty["discovery_distortion_rate"] is None
    assert empty["gate_false_negative_rate"] is None
    assert empty["gate_false_negative_rate_known"] is None
    assert empty["discovery_recovered_per_token"] is None
    assert empty["discovery_recovered_per_dollar"] is None
    assert empty["discovery_precision"] == 0.0
    assert empty["discovery_precision_denominator"] == 1
    assert empty["unmatched_proposal_count"] == 1
    assert empty["model_judgments_used"] is False
    assert empty["tier_labels_used"] is False


def test_a_missed_relationship_lowers_recall_and_is_not_a_gate_false_negative():
    discovery = _frozen(
        [
            _gold("g-hit", "C-A", "c-b", "p-1"),
            _gold("g-miss", "c-c", "c-d", "p-2"),
        ]
    )
    proposals = [
        _proposal("r-hit", "c-a", "c-b", "p-1", downstream_eligible=True),
        _proposal("r-other", "c-e", "c-f", "p-9"),
    ]
    report = score_discovery(proposals, discovery)
    assert report["discovery_recall"] == 0.5
    assert report["discovery_recall_denominator"] == 2
    assert report["discovery_recall_coverage"] == 1.0
    assert report["recovered_count"] == 1
    assert report["missed_count"] == 1
    assert _by_id(report, "g-hit")["discovery_status"] == "recovered"
    missed = _by_id(report, "g-miss")
    assert missed["discovery_status"] == "missed"
    assert missed["gate_status"] == "not_applicable"
    assert report["withheld_but_supported_count"] == 0
    assert report["gate_false_negative_rate"] == 0.0
    assert report["discovery_precision"] == 0.5
    assert report["discovery_precision_denominator"] == 2
    assert report["unmatched_proposal_count"] == 1
    assert report["discovery_distortion_rate"] == 0.0
    assert report["metric_source"] == "human_adjudication"
    assert report["tier_labels_used"] is False


def test_wrong_direction_is_distorted_not_recovered():
    discovery = _frozen([_gold("g-dir", "c-a", "c-b", "p-1", direction="from_to")])
    proposals = [
        _proposal(
            "r-dir",
            "c-a",
            "c-b",
            "p-1",
            stated_direction="to_from",
            evidence_tier=3,
            evidence_label="attribution",
            attribution_source="participant",
            allowed_wording="participants attribute c-a to c-b",
        )
    ]
    report = score_discovery(proposals, discovery)
    row = _by_id(report, "g-dir")
    assert row["discovery_status"] == "distorted"
    assert row["direction_agrees"] is False
    assert report["recovered_count"] == 0
    assert report["distorted_count"] == 1
    assert report["discovery_recall"] == 0.0
    assert report["discovery_recall"] != 1
    assert report["discovery_distortion_rate"] == 1.0
    assert report["discovery_distortion_denominator"] == 1
    assert report["discovery_precision"] == 0.0
    assert report["tier_labels_used"] is False
    assert report["model_judgments_used"] is False


def test_withheld_but_supported_is_a_gate_false_negative_and_can_still_be_recovered():
    discovery = _frozen([_gold("g-fn", "c-a", "c-b", "p-1")])
    proposals = [
        _proposal(
            "r-fn",
            "c-a",
            "c-b",
            "p-1",
            downstream_eligible=False,
            disposition="accepted",
        )
    ]
    report = score_discovery(proposals, discovery)
    row = _by_id(report, "g-fn")
    assert row["discovery_status"] == "recovered"
    assert row["gate_status"] == "withheld_but_supported"
    assert report["recovered_count"] == 1
    assert report["discovery_recall"] == 1.0
    assert report["withheld_but_supported_count"] == 1
    assert report["gate_false_negative_rate"] == 1.0
    assert report["gate_false_negative_rate_known"] == 1.0
    assert report["gate_false_negative_rate_known_denominator"] == 1
    assert report["gate_false_negative_rate_coverage"] == 1.0


def test_frozen_hash_change_is_detected():
    frozen = _frozen([_gold("g-hash", "c-a", "c-b", "p-1")])
    changed = json.loads(json.dumps(frozen))
    changed["adjudications"][0]["rationale"] = "Edited after the freeze."
    with pytest.raises(DiscoveryEvaluationError, match="frozen_hash"):
        score_discovery([], changed)
    with pytest.raises(DiscoveryEvaluationError, match="frozen_hash"):
        load_discovery_file(changed)
    assert load_discovery_file(frozen)["frozen_hash_status"] == "verified"


def test_unfrozen_discovery_is_not_scored():
    raw = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "label": "Synthetic example data only. Not a study result.",
        "adjudications": [_gold("g-open", "c-a", "c-b", "p-1")],
    }
    with pytest.raises(DiscoveryEvaluationError, match="frozen_hash"):
        score_discovery([], raw)


def test_incomplete_expenditure_gives_null_per_dollar_and_per_token():
    discovery = _frozen([_gold("g-cost", "c-a", "c-b", "p-1")])
    proposals = [_proposal("r-cost", "c-a", "c-b", "p-1")]
    incomplete = {
        "reread_expenditure_complete": False,
        "reread_expenditure_tokens": 100,
        "reread_expenditure_cost_usd": 1.5,
    }
    report = score_discovery(proposals, discovery, incomplete)
    assert report["recovered_count"] == 1
    assert report["discovery_recovered_per_token"] is None
    assert report["discovery_recovered_per_dollar"] is None

    complete = {
        "reread_expenditure_complete": True,
        "reread_expenditure_tokens": 100,
        "reread_expenditure_cost_usd": 2.0,
    }
    done = score_discovery(proposals, discovery, complete)
    assert done["discovery_recovered_per_token"] == 0.01
    assert done["discovery_recovered_per_dollar"] == 0.5

    state = {
        "prompt_log": [
            {
                "kind": "sampling_reread",
                "status": "ok",
                "received": True,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        ],
        "reread_notes": [],
        "metadata": {"token_prices": {"input_usd_per_million": 3, "output_usd_per_million": 15}},
    }
    from_state = score_discovery(proposals, discovery, state)
    assert from_state["discovery_recovered_per_token"] is None
    assert from_state["discovery_recovered_per_dollar"] is None


def test_disagreement_does_not_change_the_scored_row():
    gold = _gold("g-split", "c-a", "c-b", "p-1", direction="from_to")
    frozen = _frozen(
        [gold],
        disagreements=[
            {
                "adjudication_id": "g-split",
                "adjudicator_id": "example-adjudicator-2",
                "field": "direction",
                "value": "undirected",
                "rationale": "Synthetic disagreement only.",
            }
        ],
    )
    report = score_discovery([_proposal("r-split", "c-a", "c-b", "p-1")], frozen)
    assert report["adjudicator_disagreement_count"] == 1
    assert _by_id(report, "g-split")["discovery_status"] == "recovered"
    assert report["discovery_recall"] == 1.0


def test_passage_mismatch_and_unknown_eligibility_stay_out_of_the_wrong_rates():
    discovery = _frozen([_gold("g-pass", "c-a", "c-b", "p-1")])
    proposals = [
        _proposal("r-other-passage", "c-a", "c-b", "p-other", downstream_eligible=True),
        {"relationship_id": "r-synthetic", "from_id": "c-a", "to_id": "c-b", "passage_id": "p-1", "synthetic": True},
    ]
    report = score_discovery(proposals, discovery)
    assert _by_id(report, "g-pass")["discovery_status"] == "missed"
    assert _by_id(report, "g-pass")["gate_status"] == "not_applicable"
    assert report["withheld_but_supported_count"] == 0
    assert report["discovery_recall"] == 0.0
    assert report["unmatched_proposal_count"] == 1
    assert report["discovery_precision_denominator"] == 1

    unknown = score_discovery(
        [{"relationship_id": "r-unknown", "from_id": "c-a", "to_id": "c-b", "passage_id": "p-1"}],
        discovery,
    )
    assert unknown["discovery_recall"] == 1.0
    assert unknown["gate_false_negative_rate"] is None
    assert unknown["gate_false_negative_rate_known"] is None
    assert unknown["gate_false_negative_rate_coverage"] == 0.0
    assert unknown["withheld_but_supported_count"] == 0
    assert _by_id(unknown, "g-pass")["gate_status"] == "unknown"

    zero = score_discovery(
        [],
        discovery,
        {
            "reread_expenditure_complete": True,
            "reread_expenditure_tokens": 50,
            "reread_expenditure_cost_usd": 2,
        },
    )
    assert zero["recovered_count"] == 0
    assert zero["discovery_recovered_per_token"] == 0.0
    assert zero["discovery_recovered_per_dollar"] == 0.0


def test_template_is_headers_only_and_example_is_synthetic():
    header = _TEMPLATE.read_text(encoding="utf-8").strip().splitlines()
    assert len(header) == 1
    assert "adjudication_id" in header[0]
    assert "disagreement_adjudicator_id" in header[0]
    loaded_template = load_discovery_file(_TEMPLATE)
    assert loaded_template["frozen_hash_status"] == "absent"
    assert loaded_template["adjudications"] == []

    loaded = load_discovery_file(_EXAMPLE)
    assert "synthetic" in loaded["label"].casefold()
    assert "not a study result" in loaded["label"].casefold()
    assert loaded["frozen_hash_status"] == "verified"
    assert loaded["schema_version"] == "discovery-eval-1"
    assert loaded["disagreements"]
    report = score_discovery([], loaded)
    assert report["discovery_recall"] == 0.0
    assert report["missed_count"] == 1
    assert report["withheld_but_supported_count"] == 1
    assert report["gate_false_negative_rate"] == 1.0
    assert any(row["discovery_status"] == "missed" for row in report["relationships"])
    assert any(row["gate_status"] == "withheld_but_supported" for row in report["relationships"])
