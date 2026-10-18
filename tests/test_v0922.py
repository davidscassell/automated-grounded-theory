"""0.9.22 review fixes. Fakes only. No network calls and no real adjudications."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from automated_grounded_theory import (
    DISCOVERY_SCHEMA_VERSION,
    DiscoveryEvaluationError,
    axial_code,
    freeze_discovery_set,
    load_discovery_file,
    score_discovery,
)
from automated_grounded_theory.axial import AxialProposalBatch
from automated_grounded_theory.relationship_tiers import apply_evidence_tier
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v0920 import _assess, _base_row

_FIXTURE = Path("tests/fixtures/attribution_adversarial_1.json")
_EXAMPLE = Path("examples/discovery_adjudication_example.json")

_AXIAL = dict(
    seed=1,
    max_axial_review_calls=1,
    axial_share_propose=1,
    axial_share_counterexample=0,
    batch_size=1,
    interrupt=False,
    decision_mode="human",
)


def _pair_key(pair: dict) -> tuple[str, str]:
    left = str(pair.get("from_id") or "")
    right = str(pair.get("to_id") or "")
    return tuple(sorted((left, right)))


class _NothingCoder(FakeAxialCoder):
    def proposals(self, body):
        return AxialProposalBatch(relationships=[], nothing_to_propose=True)


class _Stall:
    def __init__(self, delay: float):
        self.delay = delay

    async def ainvoke(self, payload):
        await asyncio.sleep(self.delay)
        raw = AIMessage(content="{}")
        return {"raw": raw, "parsed": None, "parsing_error": ValueError("stall")}


class _TimeoutCoder(FakeAxialCoder):
    def __init__(self, delay: float):
        super().__init__()
        object.__setattr__(self, "_hold", _Stall(delay))

    def with_structured_output(self, schema, **kwargs):
        return object.__getattribute__(self, "_hold")


def test_an_unsent_pair_stays_retryable_on_a_funded_rerun():
    starved_coder = FakeAxialCoder()
    starved = axial_code(_corpus(), starved_coder, max_axial_review_tokens=0, **_AXIAL)
    assert starved_coder.call_count == 0
    assert starved.metadata["unique_pairs_explored"] == 0
    assert starved.metadata["examined_pairs"] == []
    assert starved.metadata["pairs_planned"] == 1
    assert starved.metadata["pairs_attempted"] == 0
    assert starved.metadata["pairs_completed"] == 0
    assert starved.metadata["pair_coverage"]["pairs_planned"] == 1
    assert starved.metadata["pair_coverage"]["pairs_completed"] == 0
    funded_coder = FakeAxialCoder()
    funded = axial_code(
        _corpus(),
        funded_coder,
        examined_pairs=starved.metadata["examined_pairs"],
        **_AXIAL,
    )
    assert funded_coder.call_count >= 1
    assert funded.metadata["pairs_completed"] == 1
    assert funded.metadata["unique_pairs_explored"] == 1
    assert funded.metadata["examined_pairs"]


def test_a_0919_style_examined_list_still_excludes_every_recorded_pair():
    """0.9.20 and 0.9.21 wrote planned pairs into examined_pairs before the request.

    That list cannot be split. Passing it still excludes every pair in it.
    """
    coder = FakeAxialCoder()
    history = [{"from_id": "c-prep", "to_id": "c-return"}]
    skipped = axial_code(_corpus(), coder, examined_pairs=history, **_AXIAL)
    assert coder.call_count == 0
    assert skipped.metadata["within_run_examined"] == []
    assert skipped.metadata["pairs_completed"] == 0
    assert skipped.metadata["unique_pairs_explored"] == 1
    recorded = {_pair_key(pair) for pair in skipped.metadata["examined_pairs"]}
    assert ("c-prep", "c-return") in recorded


def test_a_timed_out_pair_is_not_examined_and_can_be_retried():
    timed = axial_code(
        _corpus(),
        _TimeoutCoder(1.2),
        axial_request_timeout=0.35,
        **_AXIAL,
    )
    assert timed.metadata["stop"] == "timeout"
    assert timed.metadata["pairs_completed"] == 0
    assert timed.metadata["unique_pairs_explored"] == 0
    assert all(_pair_key(pair) != ("c-prep", "c-return") for pair in timed.metadata.get("examined_pairs") or [])
    coder = FakeAxialCoder()
    retried = axial_code(
        _corpus(),
        coder,
        examined_pairs=timed.metadata.get("examined_pairs") or [],
        **_AXIAL,
    )
    assert coder.call_count >= 1
    assert retried.metadata["pairs_completed"] == 1
    assert ("c-prep", "c-return") in {_pair_key(pair) for pair in retried.metadata["examined_pairs"]}


def test_a_completed_empty_response_is_examined_and_skipped_next_run():
    first_coder = _NothingCoder()
    first = axial_code(_corpus(), first_coder, **_AXIAL)
    assert first_coder.call_count >= 1
    assert first.relationships == []
    assert first.metadata["pairs_attempted"] == 1
    assert first.metadata["pairs_completed"] == 1
    assert first.metadata["unique_pairs_explored"] == 1
    assert ("c-prep", "c-return") in {_pair_key(pair) for pair in first.metadata["examined_pairs"]}
    second_coder = FakeAxialCoder()
    second = axial_code(
        _corpus(),
        second_coder,
        examined_pairs=first.metadata["examined_pairs"],
        **_AXIAL,
    )
    assert second_coder.call_count == 0
    assert second.metadata["within_run_examined"] == []
    assert second.metadata["pairs_completed"] == 0
    assert second.metadata["unique_pairs_explored"] == 1


def _tier_for_quote(quote: str) -> dict:
    row = _base_row()
    row["case_quotes"] = [
        {"passage_id": "p1", "quote": quote, "endpoint": "both"},
        {"passage_id": "p2", "quote": quote, "endpoint": "both"},
        {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
    ]
    return apply_evidence_tier_on(row, quote)


def apply_evidence_tier_on(row: dict, quote: str) -> dict:
    return _assess(row, quote=quote)


def test_adversarial_attribution_fixture_stays_unvalidated_outside_one_clear_pattern():
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert payload["fixture_version"] == "attribution-adversarial-1"
    assert "not general semantic reliability" in payload["note"].casefold()
    seen = set()
    for case in payload["cases"]:
        seen.add(case["outcome"])
        row = _tier_for_quote(case["quote"])
        link = row["participant_link"]
        assert link["status"] == case["status"], case["id"]
        assert link["stated_direction"] == case["stated_direction"], case["id"]
        assert link["authorizes_tier_3"] is case["authorizes_tier_3"], case["id"]
        assert link["validated_by"] == "deterministic_rule"
        assert row["evidence_tier"] == case["evidence_tier"], case["id"]
        if case["authorizes_tier_3"]:
            assert row["attribution_source"] == "participant"
        else:
            assert row["attribution_source"] is None
            assert row["evidence_tier"] < 3
    assert {
        "validated_forward",
        "validated_reverse",
        "denied",
        "hedged",
        "hypothetical",
        "ambiguous",
    } <= seen


def _gold(adjudication_id, passage_id, **extra):
    row = {
        "adjudication_id": adjudication_id,
        "kind": "independent",
        "adjudicator_id": "example-adjudicator",
        "rationale": f"Synthetic rationale for {passage_id}.",
        "from_id": "c-a",
        "to_id": "c-b",
        "supporting_passages": [{"passage_id": passage_id, "quote": "Synthetic quote only."}],
        "evidence_supported": True,
    }
    row.update(extra)
    return row


def _freeze(adjudications):
    return freeze_discovery_set(
        {
            "label": "Synthetic example data only. Not a study result.",
            "note": "Synthetic example data only. Not a study result.",
            "adjudications": adjudications,
        }
    )


def test_discovery_recall_is_invariant_to_ids_and_order():
    # Greedy id order recovers one of these rows. Maximum matching recovers both.
    gold_greedy_fail = [
        _gold("a-flex", "p-shared"),
        _gold("b-constrained", "p-only"),
    ]
    gold_greedy_pass = [
        _gold("a-constrained", "p-only"),
        _gold("b-flex", "p-shared"),
    ]
    first = {
        "relationship_id": "m-first",
        "from_id": "c-a",
        "to_id": "c-b",
        "passage_ids": ["p-only", "p-shared"],
        "stated_direction": "from_to",
        "downstream_eligible": True,
    }
    second = {
        "relationship_id": "m-second",
        "from_id": "c-a",
        "to_id": "c-b",
        "passage_ids": ["p-shared"],
        "stated_direction": "from_to",
        "downstream_eligible": True,
    }
    reports = []
    for gold in (gold_greedy_fail, list(reversed(gold_greedy_fail)), gold_greedy_pass, list(reversed(gold_greedy_pass))):
        for proposals in ((first, second), (second, first)):
            renamed = []
            for index, row in enumerate(proposals):
                item = dict(row)
                item["relationship_id"] = f"id-{index}-{row['passage_ids'][0]}"
                renamed.append(item)
            reports.append(score_discovery(renamed, _freeze(gold)))
    assert all(report["discovery_recall"] == 1.0 for report in reports)
    assert all(report["missed_count"] == 0 for report in reports)
    assert all(report["recovered_count"] == 2 for report in reports)
    # The constrained passage is always paired with the proposal that has p-only.
    for report in reports:
        by_passage = {}
        for row in report["relationships"]:
            if row["kind"] != "independent":
                continue
            by_passage[tuple(row["passage_ids"])] = row["discovery_status"]
        assert by_passage[("p-only",)] == "recovered"
        assert by_passage[("p-shared",)] == "recovered"


def test_full_agreement_beats_a_distorted_candidate():
    discovery = _freeze([_gold("g-dir", "p-1", direction="from_to")])
    wrong_first = [
        {
            "relationship_id": "a-wrong",
            "from_id": "c-a",
            "to_id": "c-b",
            "passage_id": "p-1",
            "stated_direction": "to_from",
            "downstream_eligible": True,
        },
        {
            "relationship_id": "z-right",
            "from_id": "c-a",
            "to_id": "c-b",
            "passage_id": "p-1",
            "stated_direction": "from_to",
            "downstream_eligible": True,
        },
    ]
    report = score_discovery(wrong_first, discovery)
    assert report["discovery_recall"] == 1.0
    row = next(item for item in report["relationships"] if item["adjudication_id"] == "g-dir")
    assert row["relationship_id"] == "z-right"
    assert row["discovery_status"] == "recovered"
    flipped = list(reversed(wrong_first))
    for index, proposal in enumerate(flipped):
        proposal = dict(proposal)
        proposal["relationship_id"] = f"renamed-{index}"
        proposal["statement"] = proposal.get("stated_direction")
        flipped[index] = proposal
    # Keep the agreeing proposal identifiable by stated direction, not by id.
    again = score_discovery(flipped, discovery)
    matched = next(item for item in again["relationships"] if item["discovery_status"] == "recovered")
    winner = next(item for item in flipped if item["relationship_id"] == matched["relationship_id"])
    assert winner["stated_direction"] == "from_to"
    assert again["discovery_recall"] == 1.0


def test_system_proposal_binds_only_its_named_relationship():
    discovery = _freeze(
        [
            {
                "adjudication_id": "s-named",
                "kind": "system_proposal",
                "adjudicator_id": "example-adjudicator",
                "rationale": "Synthetic system row.",
                "from_id": "c-a",
                "to_id": "c-b",
                "supporting_passages": [{"passage_id": "p-1", "quote": "Synthetic quote only."}],
                "evidence_supported": True,
                "system_relationship_id": "z-withheld",
                "system_outcome": "withheld",
            }
        ]
    )
    proposals = [
        {
            "relationship_id": "a-accepted",
            "from_id": "c-a",
            "to_id": "c-b",
            "passage_id": "p-1",
            "downstream_eligible": True,
            "disposition": "accepted",
        },
        {
            "relationship_id": "z-withheld",
            "from_id": "c-a",
            "to_id": "c-b",
            "passage_id": "p-1",
            "downstream_eligible": False,
            "disposition": "withheld",
        },
    ]
    report = score_discovery(proposals, discovery)
    row = next(item for item in report["relationships"] if item["adjudication_id"] == "s-named")
    assert row["discovery_status"] == "linked"
    assert row["relationship_id"] == "z-withheld"
    assert row["gate_status"] == "withheld_but_supported"
    assert report["gate_false_negative_rate"] == 1.0
    assert report["discovery_recall"] is None
    assert report["unmatched_system_relationship_ids"] == []


def test_a_missing_system_relationship_id_is_not_reassigned():
    discovery = _freeze(
        [
            {
                "adjudication_id": "s-missing",
                "kind": "system_proposal",
                "adjudicator_id": "example-adjudicator",
                "rationale": "Synthetic system row.",
                "from_id": "c-a",
                "to_id": "c-b",
                "supporting_passages": [{"passage_id": "p-1", "quote": "Synthetic quote only."}],
                "evidence_supported": True,
                "system_relationship_id": "not-exported",
                "system_outcome": "withheld",
            }
        ]
    )
    proposals = [
        {
            "relationship_id": "a-accepted",
            "from_id": "c-a",
            "to_id": "c-b",
            "passage_id": "p-1",
            "downstream_eligible": True,
            "disposition": "accepted",
        }
    ]
    report = score_discovery(proposals, discovery)
    row = next(item for item in report["relationships"] if item["adjudication_id"] == "s-missing")
    assert row["discovery_status"] == "unmatched_system_proposal"
    assert row["relationship_id"] is None
    assert report["unmatched_system_relationship_ids"] == ["not-exported"]
    assert report["gate_false_negative_rate"] == 1.0


def test_duplicate_system_relationship_ids_are_rejected_or_flagged():
    proposals = [
        {
            "relationship_id": "dup",
            "from_id": "c-a",
            "to_id": "c-b",
            "passage_id": "p-1",
            "downstream_eligible": False,
        },
        {
            "relationship_id": "dup",
            "from_id": "c-a",
            "to_id": "c-b",
            "passage_id": "p-1",
            "downstream_eligible": True,
        },
    ]
    discovery = _freeze(
        [
            {
                "adjudication_id": "s-dup",
                "kind": "system_proposal",
                "adjudicator_id": "example-adjudicator",
                "rationale": "Synthetic system row.",
                "from_id": "c-a",
                "to_id": "c-b",
                "supporting_passages": [{"passage_id": "p-1", "quote": "Synthetic quote only."}],
                "evidence_supported": True,
                "system_relationship_id": "dup",
                "system_outcome": "withheld",
            }
        ]
    )
    report = score_discovery(proposals, discovery)
    row = next(item for item in report["relationships"] if item["adjudication_id"] == "s-dup")
    assert row["discovery_status"] == "unmatched_system_proposal"
    assert row["relationship_id"] is None
    assert report["unmatched_system_relationship_ids"] == ["dup"]
    with pytest.raises(DiscoveryEvaluationError, match="Duplicate system_relationship_id"):
        _freeze(
            [
                {
                    "adjudication_id": "s-one",
                    "kind": "system_proposal",
                    "adjudicator_id": "example-adjudicator",
                    "rationale": "Synthetic.",
                    "evidence_supported": True,
                    "system_relationship_id": "same",
                    "system_outcome": "withheld",
                },
                {
                    "adjudication_id": "s-two",
                    "kind": "system_proposal",
                    "adjudicator_id": "example-adjudicator",
                    "rationale": "Synthetic.",
                    "evidence_supported": True,
                    "system_relationship_id": "same",
                    "system_outcome": "accepted",
                },
            ]
        )


def test_a_denying_statement_does_not_change_structural_recall():
    statement = "c-a does not cause c-b"
    gold = _gold(
        "g-deny",
        "p-1",
        claim_judgment={
            "verdict": "incorrect",
            "direction": "the statement reverses the gold direction",
            "qualification": "denied",
            "support": "the quote does not say this",
        },
    )
    proposal = {
        "relationship_id": "r-deny",
        "from_id": "c-a",
        "to_id": "c-b",
        "passage_id": "p-1",
        "stated_direction": "from_to",
        "statement": statement,
        "downstream_eligible": True,
    }
    judged = score_discovery([proposal], _freeze([gold]))
    assert judged["discovery_recall"] == 1.0
    assert judged["discovery_precision"] == 1.0
    assert judged["discovery_distortion_rate"] == 0.0
    assert judged["claim_correct_rate"] == 0.0
    assert judged["claim_correct_denominator"] == 1
    assert judged["claim_correct_coverage"] == 1.0
    assert judged["claim_matched_count"] == 1
    unjudged_gold = _gold("g-deny", "p-1")
    unjudged = score_discovery([proposal], _freeze([unjudged_gold]))
    assert unjudged["discovery_recall"] == 1.0
    assert unjudged["claim_correct_rate"] is None
    assert unjudged["claim_correct_denominator"] == 0
    assert unjudged["claim_correct_coverage"] == 0.0
    assert unjudged["claim_matched_count"] == 1


def test_discovery_eval_1_still_verifies_and_freeze_rewrites_the_hash():
    assert DISCOVERY_SCHEMA_VERSION == "discovery-eval-2"
    loaded = load_discovery_file(_EXAMPLE)
    assert loaded["schema_version"] == "discovery-eval-2"
    assert loaded["source_schema_version"] == "discovery-eval-1"
    assert loaded["frozen_hash_status"] == "verified"
    assert all(row["claim_judgment"]["verdict"] == "not_judged" for row in loaded["adjudications"])
    rewritten = freeze_discovery_set(loaded)
    assert rewritten["schema_version"] == "discovery-eval-2"
    assert rewritten["frozen_hash"] != loaded["frozen_hash"]
    assert rewritten["frozen_hash_status"] == "verified"
    scored = score_discovery([], loaded)
    assert scored["discovery_recall"] == 0.0
    assert scored["source_schema_version"] == "discovery-eval-1"
    tainted = json.loads(_EXAMPLE.read_text(encoding="utf-8"))
    tainted["adjudications"][0]["claim_judgment"] = {"verdict": "correct"}
    with pytest.raises(DiscoveryEvaluationError, match="discovery-eval-2"):
        load_discovery_file(tainted)
