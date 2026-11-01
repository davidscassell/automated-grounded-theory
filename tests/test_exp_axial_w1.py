"""Offline gate for exp-axial-w1. The model is a fake. Spend on the run is not a live bill."""

from __future__ import annotations

import json

import pytest

from automated_grounded_theory.axial import AxialError, AxialProposalBatch, AxialRelationshipDraft
from automated_grounded_theory.exp_axial_w1 import (
    CLASS_LEXICAL,
    CLASS_SAME,
    DRAW_LEXICAL,
    DRAW_SAME_RESPONDENT,
    counterexample_relevance_by_class,
    core_differs,
    exp_axial_w1,
    policy_fingerprint,
    preflight_workload,
    proposal_gate,
    proposal_request_fingerprint,
    reduce_scope,
    select_unexamined_draw,
    startup_fingerprint,
)
from automated_grounded_theory.relationship_tiers import classify_pair_evidence
from tests.test_v080 import FakeAxialCoder

PRICE = {"input_usd_per_million": 3.0}


def _code(code_id: str, word_a: str, word_b: str) -> dict:
    return {"code_id": code_id, "code": code_id, "definition": f"{word_a} {word_b}"}


def _corpus():
    codebook = {}
    passages = []
    segments = []
    quotes = {}
    for index in range(7):
        code_id = f"s{index:02d}"
        word_a = f"amber{index}"
        word_b = f"basalt{index}"
        codebook[code_id] = _code(code_id, word_a, word_b)
        passage_id = f"ps{index:02d}"
        text = f"Marker sits alone here for {word_a} {word_b}."
        quotes[code_id] = (passage_id, "Marker sits alone here")
        passages.append({
            "passage_id": passage_id,
            "doc_id": "d-same",
            "respondent_id": "r-same",
            "text": text,
        })
        segments.append({"segment_id": f"seg-{code_id}", "code_id": code_id, "passage_id": passage_id, "quote": text})
    for index, prefix in ((0, "x"), (1, "x"), (2, "x"), (3, "x"), (0, "y"), (1, "y")):
        code_id = f"{prefix}{index}"
        word_a = f"{prefix}quartz{index}"
        word_b = f"{prefix}velvet{index}"
        codebook[code_id] = _code(code_id, word_a, word_b)
        passage_id = f"p{code_id}"
        respondent = "r-x" if prefix == "x" else "r-y"
        text = f"Local note for {word_a} {word_b} only."
        quotes[code_id] = (passage_id, f"Local note for {word_a}")
        passages.append({
            "passage_id": passage_id,
            "doc_id": f"d-{code_id}",
            "respondent_id": respondent,
            "text": text,
        })
        segments.append({"segment_id": f"seg-{code_id}", "code_id": code_id, "passage_id": passage_id, "quote": text})
    cross_terms = []
    for code_id, entry in codebook.items():
        if code_id.startswith(("x", "y")):
            cross_terms.append(entry["definition"])
    passages.append({
        "passage_id": "p-cross",
        "doc_id": "d-cross",
        "respondent_id": "r-cross",
        "text": "A third respondent mentions " + " ".join(cross_terms) + ".",
    })
    return {"codebook": codebook, "passages": passages, "segments": segments, "memos": []}, quotes


class _RejectingCoder(FakeAxialCoder):
    proposal_calls: int = 0

    def proposals(self, body):
        self.proposal_calls += 1
        pair = (body.get("pairs") or [{}])[0]
        passages = [row for row in (body.get("passages") or []) if isinstance(row, dict) and row.get("text")]
        passage = passages[0] if passages else {"passage_id": "ps00", "text": "Marker sits alone here"}
        text = str(passage.get("text") or "")
        quote = text[:22] if len(text) >= 8 else "Marker sits alone here"
        return AxialProposalBatch(relationships=[
            AxialRelationshipDraft(
                from_id=str(pair.get("from_id") or ""),
                to_id=str(pair.get("to_id") or ""),
                statement="These categories come up together for this person.",
                passage_id=str(passage.get("passage_id") or ""),
                quote=quote,
                paradigm_role="context",
            )
        ])

    def relation_review(self, body):
        from automated_grounded_theory.axial import AxialRelationDecision, AxialRelationDecisions

        return AxialRelationDecisions(relationships=[
            AxialRelationDecision(
                relationship_id=str(row.get("relationship_id") or ""),
                disposition="rejected",
                rationale="The row does not meet the reviewer standard.",
            )
            for row in body.get("relationships") or []
        ])


def _history(path, examined, reuse):
    path.write_text(json.dumps({
        "examined_pairs": examined,
        "historical_expenditure": {"calls": 2, "input_tokens": 400, "dollars": 0.02},
        "reuse": reuse,
    }), encoding="utf-8")


def test_preflight_for_the_full_draw_and_the_reduction():
    original = preflight_workload(DRAW_SAME_RESPONDENT, DRAW_LEXICAL, token_prices=PRICE)
    assert original["calls"]["axial_proposal"] == 24
    assert original["calls"]["statement_copy_retry"] == 24
    assert original["calls"]["axial_counterexample"] == 4
    assert original["calls"]["axial_relation_review"] == 4
    assert original["calls"]["axial_saturation_review"] == 1
    assert original["coding_calls"] == 52
    assert original["coding_input_tokens"] == 312_000
    assert original["total_calls"] == 57
    assert original["total_input_tokens"] == 342_000
    assert original["dollars"] == pytest.approx(1.026)
    assert original["fits"] is False
    assert original["hard_cap_usd"] == 0.60
    assert original["kill_guard_usd"] == 0.55
    reduced = reduce_scope(PRICE)
    assert reduced["reduced_scope"] == {
        "same_respondent": 8,
        "lexical_cross_case": 4,
        "pairs": 12,
    }
    assert reduced["reduced"]["fits"] is True
    assert reduced["reduced"]["dollars"] == pytest.approx(0.594)
    assert reduced["reduced"]["coding_calls"] == 28
    assert reduced["original"]["pairs"] == 24


def test_a_short_class_stops_without_substitution():
    evidence = {"by_class": {
        CLASS_SAME: [{"from_id": "a", "to_id": "b", "document_cooccurrence": 1}],
        CLASS_LEXICAL: [{"from_id": "c", "to_id": "d"}],
    }}
    draw = select_unexamined_draw(evidence, [], seed=4401)
    assert draw["status"] == "stopped"
    assert draw["reason"] == "class_shortfall"
    assert draw["round"] == []
    assert draw["shortfall"][CLASS_SAME]["available"] == 1
    assert draw["shortfall"][CLASS_LEXICAL]["available"] == 1


def test_core_deferral_compares_the_eligible_set_and_the_category_counts():
    baseline = {"eligible_ids": [], "category_counts": {"a": 0}}
    assert core_differs({"eligible_ids": [], "category_counts": {"a": 0}}, baseline) is False
    assert core_differs({"eligible_ids": ["rel-1"], "category_counts": {"a": 1}}, baseline) is True
    assert core_differs({"eligible_ids": [], "category_counts": {"a": 0}}, None) is None


def test_kill_guard_refuses_before_the_call():
    metadata = {
        "exp_dollars_spent": 0.54,
        "procedure": {"exp_axial_w1": {
            "enabled": True,
            "token_prices": PRICE,
            "reuse": [],
        }},
    }
    gated = proposal_gate(
        metadata,
        {"pairs": [{"from_id": "a", "to_id": "b"}]},
        6000,
    )
    assert gated is not None
    assert gated["refused"] is True
    assert gated["reason"] == "kill_guard"
    assert gated["audits"][0]["status"] == "not_sent"


def test_exp_axial_w1_unexamined_single_document_round(tmp_path):
    source, quotes = _corpus()
    evidence = classify_pair_evidence(
        list(source["codebook"]),
        source["passages"],
        source["segments"],
        source["codebook"],
    )
    examined = [{"from_id": "s00", "to_id": "s01"}]
    draw = select_unexamined_draw(evidence, examined, seed=4401)
    assert draw["status"] == "selected"
    assert draw["available"][CLASS_SAME] >= 16
    assert draw["available"][CLASS_LEXICAL] >= 8
    assert ("s00", "s01") not in {
        tuple(sorted((row["from_id"], row["to_id"]))) for row in draw["round"]
    }
    scope = reduce_scope(PRICE)["reduced_scope"]
    chosen = (
        draw["sampled"][CLASS_SAME][: scope["same_respondent"]]
        + draw["sampled"][CLASS_LEXICAL][: scope["lexical_cross_case"]]
    )
    assert {row["evidence_class"] for row in chosen} == {CLASS_SAME, CLASS_LEXICAL}
    accepted = chosen[0]
    rejected = chosen[1]
    accepted_passage, accepted_quote = quotes[accepted["from_id"]]
    history = tmp_path / "examined-history.json"
    _history(history, examined, [
        {
            "from_id": accepted["from_id"],
            "to_id": accepted["to_id"],
            "request_fingerprint": proposal_request_fingerprint(accepted["from_id"], accepted["to_id"]),
            "policy_fingerprint": policy_fingerprint(),
            "relationship": {
                "from_id": accepted["from_id"],
                "to_id": accepted["to_id"],
                "statement": "A stored proposal kept because the fingerprints match.",
                "passage_id": accepted_passage,
                "quote": accepted_quote,
                "paradigm_role": "context",
            },
        },
        {
            "from_id": rejected["from_id"],
            "to_id": rejected["to_id"],
            "request_fingerprint": "not-the-request",
            "policy_fingerprint": policy_fingerprint(),
            "relationship": {
                "from_id": rejected["from_id"],
                "to_id": rejected["to_id"],
                "statement": "This stored proposal must not be reused.",
                "passage_id": quotes[rejected["from_id"]][0],
                "quote": quotes[rejected["from_id"]][1],
                "paradigm_role": "context",
            },
        },
    ])
    coder = _RejectingCoder()
    frozen = startup_fingerprint(source, coder, commit="offline-fixture")
    with pytest.raises(AxialError, match="refused to start"):
        exp_axial_w1(
            source,
            coder,
            history_path=history,
            frozen_fingerprints={**frozen, "commit": "different-commit"},
            baseline_core={"eligible_ids": [], "category_counts": {code_id: 0 for code_id in source["codebook"]}},
            token_prices=PRICE,
            seed=4401,
            commit="offline-fixture",
            decision_mode="llm",
            interrupt=False,
        )
    assert coder.proposal_calls == 0
    report = exp_axial_w1(
        source,
        coder,
        history_path=history,
        frozen_fingerprints=frozen,
        baseline_core={"eligible_ids": [], "category_counts": {code_id: 0 for code_id in source["codebook"]}},
        token_prices=PRICE,
        seed=4401,
        commit="offline-fixture",
        decision_mode="llm",
        interrupt=False,
    )
    assert report["status"] == "completed"
    assert report["preflight"]["original"]["pairs"] == 24
    assert report["preflight"]["original"]["coding_calls"] == 52
    assert report["preflight"]["original"]["total_input_tokens"] == 342_000
    assert report["preflight"]["reduced_scope"]["same_respondent"] == 8
    assert report["preflight"]["reduced_scope"]["lexical_cross_case"] == 4
    assert report["allowances"]["axial_proposal"] == 12
    assert report["allowances"]["statement_copy_retry"] == 12
    assert report["allowances"]["axial_counterexample"] == 4
    assert report["allowances"]["axial_relation_review"] == 4
    assert report["allowances"]["axial_saturation_review"] == 4
    assert report["historical_expenditure"]["dollars"] == 0.02
    assert report["new_spend"]["note"]
    assert coder.proposal_calls == len(chosen) - 1
    decisions = {row["decision"] for row in report["reuse_ledger"]}
    assert decisions == {"accepted", "rejected"}
    assert report["core"]["status"] == "deferred"
    assert report["core"]["dollars"] == 0
    assert report["core"]["calls"] == 0
    metrics = report["metrics"]
    assert metrics["by_class"][CLASS_SAME]["new_pairs_examined"] == 8
    assert metrics["by_class"][CLASS_LEXICAL]["new_pairs_examined"] == 4
    assert metrics["overall"]["new_pairs_examined"] == 12
    assert metrics["overall"]["grounded_assertions_per_pair"] == 0
    assert metrics["overall"]["eligibility_failures_by_reason"]["reviewer_rejected"] == 12
    assert metrics["overall"]["counterexample_relevance"] is None
    assert metrics["overall"]["elapsed_seconds"] is not None
    assert metrics["by_class"][CLASS_SAME]["additional_supporting_cases"] == 0
    assert metrics["by_class"][CLASS_LEXICAL]["additional_supporting_cases"] == 0
    relevance = counterexample_relevance_by_class([], chosen)
    assert relevance["overall"] is None
    procedure = report["result"].metadata["procedure"]
    assert procedure["max_axial_review_calls"] == 36
    assert procedure["max_axial_review_tokens"] == 216_000
    assert procedure["pattern_min_cases"] == 2
    assert procedure["one_document_one_respondent"] is False
    assert procedure["allowance_transfers"] == []
    rerun = _RejectingCoder()
    differed = exp_axial_w1(
        source,
        rerun,
        history_path=history,
        frozen_fingerprints=startup_fingerprint(source, rerun, commit="offline-fixture"),
        baseline_core={"eligible_ids": ["baseline-only"], "category_counts": {code_id: 0 for code_id in source["codebook"]}},
        token_prices=PRICE,
        seed=4401,
        commit="offline-fixture",
        decision_mode="llm",
        interrupt=False,
    )
    assert differed["core"]["status"] == "ran"
    assert differed["core"]["calls"] > 0
