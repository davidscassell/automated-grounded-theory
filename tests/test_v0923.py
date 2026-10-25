"""0.9.23 review fixes. Fakes only. No network calls and no real adjudications."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from automated_grounded_theory import (
    DISCOVERY_SCHEMA_VERSION,
    DiscoveryEvaluationError,
    __version__,
    claim_statement_hash,
    freeze_discovery_set,
    load_discovery_file,
    score_attribution_accuracy,
    score_discovery,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import apply_evidence_tier
from tests.test_v0920 import _assess

_FIXTURE = Path("tests/fixtures/attribution_adversarial_2.json")


def test_package_version_is_0923():
    assert __version__ == "0.9.29"
    assert POLICY_VERSION == "0.9.29"
    assert DISCOVERY_SCHEMA_VERSION == "discovery-eval-3"


def _gold(adjudication_id, passage_id, **extra):
    row = {
        "adjudication_id": adjudication_id,
        "kind": "independent",
        "adjudicator_id": "example-adjudicator",
        "rationale": f"Synthetic rationale for {passage_id}.",
        "from_id": extra.pop("from_id", "c-a"),
        "to_id": extra.pop("to_id", "c-b"),
        "supporting_passages": [{"passage_id": passage_id, "quote": "Synthetic quote only."}],
        "evidence_supported": True,
    }
    row.update(extra)
    return row


def _freeze(adjudications, claim_reviews=None):
    payload = {
        "label": "Synthetic example data only. Not a study result.",
        "note": "Synthetic example data only. Not a study result.",
        "adjudications": adjudications,
        "claim_reviews": claim_reviews or [],
    }
    return freeze_discovery_set(payload)


def _proposal(relationship_id, passage_ids, **extra):
    row = {
        "relationship_id": relationship_id,
        "from_id": "c-a",
        "to_id": "c-b",
        "passage_ids": list(passage_ids),
        "stated_direction": "from_to",
        "statement": "A affects B",
        "downstream_eligible": True,
        "disposition": "accepted",
    }
    row.update(extra)
    return row


def _by_id(report, adjudication_id):
    return next(row for row in report["relationships"] if row["adjudication_id"] == adjudication_id)


def test_a_proposal_review_does_not_erase_a_recovered_discovery():
    gold = [_gold("g-one", "p-real")]
    proposal = _proposal("rel-real", ["p-real"])
    before = score_discovery([proposal], _freeze(gold))
    assert before["discovery_recall"] == 1.0
    assert before["discovery_precision"] == 1.0
    assert _by_id(before, "g-one")["discovery_status"] == "recovered"

    review = {
        "adjudication_id": "s-real",
        "kind": "system_proposal",
        "adjudicator_id": "example-adjudicator",
        "rationale": "Synthetic review of the same proposal.",
        "from_id": "c-a",
        "to_id": "c-b",
        "supporting_passages": [{"passage_id": "p-real", "quote": "Synthetic quote only."}],
        "evidence_supported": True,
        "system_relationship_id": "rel-real",
        "system_outcome": "accepted",
    }
    after = score_discovery([proposal], _freeze(gold + [review]))
    assert after["discovery_recall"] == 1.0
    assert after["discovery_precision"] == 1.0
    assert _by_id(after, "g-one")["discovery_status"] == "recovered"
    assert _by_id(after, "g-one")["relationship_id"] == "rel-real"
    assert _by_id(after, "s-real")["discovery_status"] == "linked"
    assert _by_id(after, "s-real")["relationship_id"] == "rel-real"
    assert after["missed_count"] == 0


def test_recall_and_precision_are_invariant_to_proposal_reviews():
    gold = [_gold("g-a", "p-a"), _gold("g-b", "p-b", direction="from_to")]
    proposals = [
        _proposal("rel-a", ["p-a"], statement="tools affect the crew"),
        _proposal("rel-b", ["p-b"], statement="schedule affects handoff"),
        _proposal("rel-c", ["p-c"], statement="unrelated proposal"),
    ]
    bare = score_discovery(proposals, _freeze(gold))

    def review(adjudication_id, relationship_id):
        return {
            "adjudication_id": adjudication_id,
            "kind": "system_proposal",
            "adjudicator_id": "example-adjudicator",
            "rationale": "Synthetic.",
            "evidence_supported": True,
            "system_relationship_id": relationship_id,
            "system_outcome": "withheld",
        }

    bundles = [
        [],
        [review("s-a", "rel-a")],
        [review("s-a", "rel-a"), review("s-b", "rel-b"), review("s-c", "rel-c")],
        [review("s-c", "rel-c")],
    ]
    recalls = []
    precisions = []
    for bundle in bundles:
        report = score_discovery(proposals, _freeze(gold + bundle))
        recalls.append(report["discovery_recall"])
        precisions.append(report["discovery_precision"])
        assert report["recovered_count"] == bare["recovered_count"]
        assert report["missed_count"] == bare["missed_count"]
    assert recalls == [bare["discovery_recall"]] * len(bundles)
    assert precisions == [bare["discovery_precision"]] * len(bundles)
    # A withheld review still enlarges the gate population. That rate is not recall.
    with_review = score_discovery(proposals, _freeze(gold + [review("s-a", "rel-a")]))
    assert with_review["gate_false_negative_population"] > bare["gate_false_negative_population"]


def test_one_full_recovery_beats_two_distorted_matches():
    gold = [
        _gold("g-a", "p-a", direction="from_to"),
        _gold("g-b", "p-b", direction="to_from"),
    ]
    proposals = [
        _proposal("p-both", ["p-a", "p-b"], stated_direction="from_to", statement="both passages, forward"),
        _proposal("p-only", ["p-a"], stated_direction="to_from", statement="only the first passage, reversed"),
    ]
    report = score_discovery(proposals, _freeze(gold))
    assert report["recovered_count"] == 1
    assert report["distorted_count"] == 0
    assert report["missed_count"] == 1
    assert report["discovery_recall"] == 0.5
    assert _by_id(report, "g-a")["discovery_status"] == "recovered"
    assert _by_id(report, "g-a")["relationship_id"] == "p-both"
    assert _by_id(report, "g-b")["discovery_status"] == "missed"

    flipped = list(reversed(proposals))
    for index, proposal in enumerate(flipped):
        proposal = dict(proposal)
        proposal["relationship_id"] = f"renamed-{index}-zzzz" if index == 0 else f"renamed-{index}-aaaa"
        flipped[index] = proposal
    again = score_discovery(flipped, _freeze(list(reversed(gold))))
    assert again["recovered_count"] == 1
    assert again["distorted_count"] == 0
    winner = _by_id(again, "g-a")
    matched = next(item for item in flipped if item["relationship_id"] == winner["relationship_id"])
    assert matched["stated_direction"] == "from_to"
    assert set(matched["passage_ids"]) == {"p-a", "p-b"}


def _structural_edges(gold_rows, proposals):
    """Independent copy of the structural edges score_discovery is allowed to use."""
    edges = []
    for gold in gold_rows:
        if gold["kind"] != "independent":
            continue
        passages = {item["passage_id"] for item in gold["supporting_passages"]}
        pair = tuple(sorted((gold["from_id"].casefold(), gold["to_id"].casefold())))
        for proposal in proposals:
            proposal_pair = tuple(
                sorted((str(proposal["from_id"]).casefold(), str(proposal["to_id"]).casefold()))
            )
            proposal_passages = set(proposal.get("passage_ids") or [])
            if proposal_pair != pair or passages.isdisjoint(proposal_passages):
                continue
            stated = str(proposal.get("stated_direction") or "from_to")
            recovered = (not gold.get("direction")) or stated == gold.get("direction")
            content = (
                gold["from_id"],
                gold["to_id"],
                tuple(sorted(passages)),
                str(gold.get("direction") or ""),
                str(gold.get("rationale") or ""),
                tuple(sorted(proposal_passages)),
                stated,
                str(proposal.get("statement") or ""),
            )
            edges.append((gold["adjudication_id"], proposal["relationship_id"], recovered, content))
    return edges


def _brute_lex(edges):
    """Documented objective: recoveries, then structural matches, then content order."""
    gold_ids = sorted({gold for gold, _prop, _recovered, _content in edges})
    prop_ids = sorted({prop for _gold, prop, _recovered, _content in edges})
    by_gold = {}
    for gold, prop, recovered, content in edges:
        by_gold.setdefault(gold, []).append((prop, recovered, content))
    matchings = []

    def walk(index, used, chosen):
        if index == len(gold_ids):
            matchings.append(list(chosen))
            return
        gold = gold_ids[index]
        walk(index + 1, used, chosen)
        for prop, recovered, content in by_gold.get(gold, []):
            if prop in used:
                continue
            used.add(prop)
            chosen.append((gold, prop, recovered, content))
            walk(index + 1, used, chosen)
            chosen.pop()
            used.remove(prop)

    walk(0, set(), [])
    if not matchings:
        return []

    def primary(matching):
        return (
            sum(1 for _gold, _prop, recovered, _content in matching if recovered),
            len(matching),
        )

    target = max(primary(matching) for matching in matchings)
    remaining = [matching for matching in matchings if primary(matching) == target]
    chosen = []
    ordered = sorted(edges, key=lambda edge: edge[3])
    for edge in ordered:
        survivors = [
            matching
            for matching in remaining
            if edge[:3] in {(gold, prop, recovered) for gold, prop, recovered, _content in matching}
            and all(item[:3] in {(gold, prop, recovered) for gold, prop, recovered, _c in matching} for item in chosen)
        ]
        if survivors:
            chosen.append(edge)
            remaining = survivors
    return {(gold, prop) for gold, prop, _recovered, _content in chosen}


def test_component_assignment_matches_the_lexicographic_objective():
    families = []
    families.append(
        (
            [_gold("g-a", "p-a", direction="from_to"), _gold("g-b", "p-b", direction="to_from")],
            [
                _proposal("p-both", ["p-a", "p-b"], stated_direction="from_to"),
                _proposal("p-only", ["p-a"], stated_direction="to_from"),
            ],
        )
    )
    families.append(
        (
            [_gold("g-flex", "p-shared"), _gold("g-only", "p-only")],
            [
                _proposal("wide", ["p-only", "p-shared"]),
                _proposal("narrow", ["p-shared"]),
            ],
        )
    )
    families.append(
        (
            [_gold("g-1", "p-1", direction="from_to")],
            [
                _proposal("wrong", ["p-1"], stated_direction="to_from", statement="distorted"),
                _proposal("right", ["p-1"], stated_direction="from_to", statement="agrees"),
            ],
        )
    )
    for gold, proposals in families:
        for gold_order in (gold, list(reversed(gold))):
            for prop_order in (proposals, list(reversed(proposals))):
                renamed = []
                for index, proposal in enumerate(prop_order):
                    item = dict(proposal)
                    item["relationship_id"] = f"id-{index}-{proposal['passage_ids'][0]}"
                    renamed.append(item)
                report = score_discovery(renamed, _freeze(gold_order))
                got = {
                    (row["adjudication_id"], row["relationship_id"])
                    for row in report["relationships"]
                    if row["discovery_status"] in {"recovered", "distorted"}
                }
                # Brute force uses the pre-rename ids. Map through passage signature.
                signature = {
                    proposal["relationship_id"]: (
                        tuple(proposal["passage_ids"]),
                        proposal["stated_direction"],
                        proposal["statement"],
                    )
                    for proposal in renamed
                }
                inverse = {}
                for proposal in renamed:
                    inverse[signature[proposal["relationship_id"]]] = proposal["relationship_id"]
                expected_edges = _structural_edges(gold_order, renamed)
                expected = _brute_lex(expected_edges)
                assert got == expected


def _review(review_id, relationship_id, statement, direction="from_to", verdict="correct", run_id="run-1"):
    return {
        "review_id": review_id,
        "run_id": run_id,
        "relationship_id": relationship_id,
        "statement": statement,
        "direction": direction,
        "qualification": "stated cause",
        "support": "synthetic support",
        "verdict": verdict,
    }


def test_claim_judgment_follows_the_reviewed_statement_not_a_neighbor():
    gold = [_gold("g-claim", "p-1")]
    affects = "A affects B"
    denies = "A does not affect B"
    review = _review("rev-affects", "rel-1", affects)
    discovery = _freeze(gold, [review])
    proposal = _proposal("rel-1", ["p-1"], statement=affects, stated_direction="from_to")
    assert discovery["claim_reviews"][0]["statement_hash"] == claim_statement_hash(
        affects, "from_to", "stated cause"
    )
    held = score_discovery([proposal], discovery, run_id="run-1")
    assert held["discovery_recall"] == 1.0
    assert held["claim_correct_rate"] == 1.0
    assert held["claim_correct_denominator"] == 1
    assert held["claim_matched_count"] == 1
    assert held["claim_stale_count"] == 0
    assert _by_id(held, "g-claim")["claim_binding"] == "fresh"
    changed = dict(proposal)
    changed["statement"] = denies
    stale = score_discovery([changed], discovery, run_id="run-1")
    assert stale["discovery_recall"] == 1.0
    assert stale["claim_correct_rate"] is None
    assert stale["claim_correct_denominator"] == 0
    assert stale["claim_stale_count"] == 1
    assert _by_id(stale, "g-claim")["claim_verdict"] == "not_judged"
    assert _by_id(stale, "g-claim")["claim_binding"] == "stale"


def test_a_claim_review_from_another_run_is_stale():
    gold = [_gold("g-run", "p-1")]
    statement = "A affects B"
    discovery = _freeze(gold, [_review("rev-run", "rel-1", statement, run_id="run-a")])
    proposal = _proposal("rel-1", ["p-1"], statement=statement)
    report = score_discovery([proposal], discovery, run_id="run-b")
    assert report["claim_correct_rate"] is None
    assert report["claim_stale_count"] == 1
    assert report["discovery_recall"] == 1.0
    missing = score_discovery([proposal], discovery)
    assert missing["claim_correct_rate"] is None
    assert missing["claim_stale_count"] == 1


def test_a_proposal_bound_claim_review_counts_in_the_rate():
    system = {
        "adjudication_id": "s-bound",
        "kind": "system_proposal",
        "adjudicator_id": "example-adjudicator",
        "rationale": "Synthetic.",
        "evidence_supported": True,
        "system_relationship_id": "rel-bound",
        "system_outcome": "accepted",
    }
    statement = "A affects B"
    discovery = _freeze([system], [_review("rev-bound", "rel-bound", statement, verdict="incorrect")])
    proposal = _proposal("rel-bound", ["p-1"], statement=statement)
    report = score_discovery([proposal], discovery, run_id="run-1")
    assert report["discovery_recall"] is None
    assert _by_id(report, "s-bound")["discovery_status"] == "linked"
    assert report["claim_matched_count"] == 1
    assert report["claim_correct_rate"] == 0.0
    assert report["claim_correct_denominator"] == 1
    assert report["claim_correct_coverage"] == 1.0
    assert report["claim_stale_count"] == 0
    assert _by_id(report, "s-bound")["claim_binding"] == "fresh"


def test_the_same_relationship_is_counted_once_when_it_is_also_linked():
    gold = _gold("g-both", "p-1")
    system = {
        "adjudication_id": "s-both",
        "kind": "system_proposal",
        "adjudicator_id": "example-adjudicator",
        "rationale": "Synthetic.",
        "evidence_supported": True,
        "system_relationship_id": "rel-both",
        "system_outcome": "accepted",
    }
    statement = "A affects B"
    discovery = _freeze([gold, system], [_review("rev-both", "rel-both", statement)])
    report = score_discovery([_proposal("rel-both", ["p-1"], statement=statement)], discovery, run_id="run-1")
    assert report["discovery_recall"] == 1.0
    assert report["claim_matched_count"] == 1
    assert report["claim_correct_rate"] == 1.0
    assert _by_id(report, "g-both")["claim_binding"] == "fresh"
    assert _by_id(report, "s-both")["claim_binding"] == "fresh"


def _canonical_eval2(adjudication):
    body = {
        "schema_version": "discovery-eval-2",
        "label": "Synthetic example data only. Not a study result.",
        "note": "Synthetic example data only. Not a study result.",
        "adjudications": [adjudication],
        "disagreements": [],
    }
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _eval2_adjudication():
    return {
        "adjudication_id": "g-old",
        "kind": "independent",
        "adjudicator_id": "example-adjudicator",
        "rationale": "Synthetic.",
        "from_id": "c-a",
        "to_id": "c-b",
        "direction": "from_to",
        "role": None,
        "wording": None,
        "supporting_passages": [{"passage_id": "p-1", "quote": "Synthetic quote only."}],
        "conceptual_distinctions": [],
        "unsupported_interpretation": False,
        "over_merging": False,
        "loss_of_minority_account": False,
        "loss_of_contradictory_account": False,
        "evidence_supported": True,
        "system_relationship_id": None,
        "system_outcome": None,
        "claim_judgment": {"verdict": "correct", "direction": None, "qualification": None, "support": None},
    }


def test_old_schema_claim_judgments_load_as_stale_and_never_as_correct():
    adjudication = _eval2_adjudication()
    payload = {
        "schema_version": "discovery-eval-2",
        "label": "Synthetic example data only. Not a study result.",
        "note": "Synthetic example data only. Not a study result.",
        "adjudications": [adjudication],
        "disagreements": [],
        "frozen_hash": _canonical_eval2(adjudication),
    }
    loaded = load_discovery_file(payload)
    assert loaded["frozen_hash_status"] == "verified"
    assert loaded["schema_version"] == "discovery-eval-3"
    assert loaded["source_schema_version"] == "discovery-eval-2"
    proposal = _proposal("rel-old", ["p-1"], statement="A affects B")
    scored = score_discovery([proposal], loaded, run_id="run-1")
    assert scored["discovery_recall"] == 1.0
    assert scored["claim_correct_rate"] is None
    assert scored["claim_stale_count"] == 1
    assert _by_id(scored, "g-old")["claim_verdict"] == "not_judged"
    denies = dict(proposal)
    denies["statement"] = "A does not affect B"
    other = score_discovery([denies], loaded)
    assert other["claim_correct_rate"] is None
    assert other["claim_stale_count"] == 1
    rewritten = freeze_discovery_set(loaded)
    assert rewritten["schema_version"] == "discovery-eval-3"
    assert rewritten["claim_reviews"] == []
    carried = score_discovery([proposal], rewritten, run_id="run-1")
    assert carried["claim_correct_rate"] is None
    assert carried["claim_stale_count"] == 0

    fresh = {
        "schema_version": "discovery-eval-3",
        "label": "Synthetic example data only. Not a study result.",
        "note": "Synthetic.",
        "adjudications": [_gold("g-new", "p-1", claim_judgment={"verdict": "correct"})],
    }
    with pytest.raises(DiscoveryEvaluationError, match="claim_reviews"):
        freeze_discovery_set(fresh)


def test_absence_of_evidence_does_not_authorize_tier_3():
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert payload["fixture_version"] == "attribution-adversarial-2"
    assert "not a measure of general semantic accuracy" in payload["note"].casefold()
    outcomes = set()
    for case in payload["cases"]:
        outcomes.add(case["outcome"])
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
    assert "unsupported" in outcomes
    assert "validated_forward" in outcomes
    empty = score_attribution_accuracy(None)
    assert empty["accuracy"] is None
    assert empty["correct_count"] is None
    assert empty["denominator"] == 0
    assert empty["model_judgments_used"] is False
    assert score_attribution_accuracy([])["accuracy"] is None
    supplied = score_attribution_accuracy(
        [
            {"quote": case["quote"], "expected_status": case["status"], "authorizes_tier_3": case["authorizes_tier_3"]}
            for case in payload["cases"]
        ]
    )
    assert supplied["denominator"] == len(payload["cases"])
    assert supplied["accuracy"] == 1.0
    assert supplied["model_judgments_used"] is False
    # The hook does not read the fixture by itself.
    assert score_attribution_accuracy(None)["note"] != supplied["note"]


def _tier_for_quote(quote: str):
    row = _base_quote_row()
    for item in row["case_quotes"]:
        if item["endpoint"] == "both":
            item["quote"] = quote
    return _assess(row, quote=quote)


def _base_quote_row():
    from tests.test_v0920 import _base_row

    row = _base_row()
    row["case_quotes"] = [
        {"passage_id": "p1", "quote": "placeholder", "endpoint": "both"},
        {"passage_id": "p2", "quote": "placeholder", "endpoint": "both"},
        {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
    ]
    return row


def test_apply_evidence_tier_is_the_public_reader_for_the_repro_quote():
    quote = "There is no evidence that tools for the shift cause coworkers on the crew"
    row = _base_quote_row()
    row["case_quotes"][0]["quote"] = quote
    row["case_quotes"][1]["quote"] = quote
    assessed = apply_evidence_tier(
        row,
        passages={
            "p1": {
                "passage_id": "p1",
                "doc_id": "d1",
                "respondent_id": "r1",
                "text": f"The tools for the shift sit beside coworkers on the crew tonight {quote}",
            },
            "p2": {
                "passage_id": "p2",
                "doc_id": "d2",
                "respondent_id": "r2",
                "text": f"The tools for the shift sit beside coworkers on the crew tonight {quote}",
            },
            "p3": {
                "passage_id": "p3",
                "doc_id": "d3",
                "respondent_id": "r3",
                "text": "The schedule changed and nobody mentioned either side.",
            },
        },
        codebook={
            "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
            "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
        },
        segments=[
            {"segment_id": "s1", "code_id": "tools", "passage_id": "p1", "quote": "tools for the shift sit"},
            {"segment_id": "s2", "code_id": "crew", "passage_id": "p1", "quote": "coworkers on the crew tonight"},
        ],
        memos=[],
    )
    assert assessed["participant_link"]["status"] == "unsupported"
    assert assessed["participant_link"]["authorizes_tier_3"] is False
    assert row["evidence_tier"] == 2


def test_one_candidate_among_many_proposals_stays_fast():
    gold = [_gold("g-speed", "p-real")]
    discovery = _freeze(gold)
    elapsed = {}
    for count in (100, 200, 400, 1000):
        proposals = [_proposal("rel-real", ["p-real"], statement="the real candidate")]
        for index in range(count - 1):
            proposals.append(
                _proposal(
                    f"rel-{index}",
                    [f"p-{index}"],
                    from_id=f"c-{index}",
                    to_id=f"d-{index}",
                    statement=f"unrelated {index}",
                )
            )
        started = time.perf_counter()
        report = score_discovery(proposals, discovery)
        elapsed[count] = time.perf_counter() - started
        assert report["discovery_recall"] == 1.0
        assert report["discovery_precision"] == round(1 / count, 4)
    assert elapsed[400] < 0.5
    assert elapsed[1000] < 2.0
