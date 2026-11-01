"""Offline gate for exp-axial-w1. The model is a fake. Spend on the run is not a live bill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automated_grounded_theory.axial import AxialError, AxialProposalBatch, AxialRelationshipDraft
from automated_grounded_theory.exp_axial_w1 import (
    CLASS_LEXICAL,
    CLASS_SAME,
    DRAW_LEXICAL,
    DRAW_SAME_RESPONDENT,
    FROZEN_TOKEN_PRICES,
    LEXICAL_ELIGIBILITY_RULE,
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
    raised = list(codebook)
    codebook["not-raised"] = _code("not-raised", "unused", "label")
    passages.append({
        "passage_id": "p-counter",
        "doc_id": "d-counter",
        "respondent_id": "r-counter",
        "text": "however amber0 basalt0 showed up without the other pole.",
    })
    source = {
        "codebook": list(codebook.values()),
        "passages": passages,
        "segments": segments,
        "memos": [],
        "raised_code_ids": raised,
        "metadata": {
            "raised_code_ids": raised,
            "selection_review_complete": True,
            "selection_status": "applied",
            "selection_withheld": False,
        },
    }
    return source, quotes


class _RejectingCoder(FakeAxialCoder):
    proposal_calls: int = 0

    def proposals(self, body):
        self.proposal_calls += 1
        pair = (body.get("pairs") or [{}])[0]
        passages = [row for row in (body.get("passages") or []) if isinstance(row, dict) and row.get("text")]
        passage = passages[0] if passages else {"passage_id": "ps00", "text": "Marker sits alone here"}
        text = str(passage.get("text") or "")
        quote = text[:22] if len(text) >= 8 else "Marker sits alone here"
        if self.mode == "accept" and not body.get("statement_copy_retry") and self.proposal_calls == 1:
            return AxialProposalBatch(relationships=[
                AxialRelationshipDraft(
                    from_id=str(pair.get("from_id") or ""),
                    to_id=str(pair.get("to_id") or ""),
                    statement=quote,
                    passage_id=str(passage.get("passage_id") or ""),
                    quote=quote,
                    paradigm_role="context",
                )
            ])
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

        disposition = "accepted" if self.mode == "accept" else "rejected"
        rationale = (
            "The cited quote is enough to keep the row in play."
            if disposition == "accepted"
            else "The row does not meet the reviewer standard."
        )
        return AxialRelationDecisions(relationships=[
            AxialRelationDecision(
                relationship_id=str(row.get("relationship_id") or ""),
                disposition=disposition,
                rationale=rationale,
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
    original = preflight_workload(DRAW_SAME_RESPONDENT, DRAW_LEXICAL)
    assert original["prices"]["input_usd_per_million"] == 1.25
    assert original["prices"]["output_usd_per_million"] == 2.50
    assert FROZEN_TOKEN_PRICES["input_usd_per_million"] == 1.25
    assert original["calls"]["axial_proposal"] == 24
    assert original["calls"]["statement_copy_retry"] == 6
    assert original["calls"]["axial_counterexample_required"] == 24
    assert original["calls"]["axial_counterexample"] == 6
    assert original["counterexample_required_but_blocked"] == 18
    assert original["calls"]["axial_relation_review_required"] == 24
    assert original["calls"]["axial_relation_review"] == 24
    assert original["relation_review_shortfall"] == 0
    assert original["calls"]["axial_saturation_review"] == 1
    assert original["coding_calls"] == 36
    assert original["coding_input_tokens"] == 216_000
    assert original["total_calls"] == 61
    assert original["total_input_tokens"] == 366_000
    assert original["total_output_tokens"] == 55_567
    assert original["output_dollars"] is not None
    assert original["output_dollars"] == pytest.approx(0.138917)
    assert original["input_dollars"] == pytest.approx(0.4575)
    assert original["dollars"] == pytest.approx(0.596417)
    assert original["fits"] is False
    assert "counterexample_required_but_blocked" in original["reasons"]
    assert original["hard_cap_usd"] == 0.60
    assert original["kill_guard_usd"] == 0.55
    reduced = reduce_scope()
    assert reduced["reduced_scope"] == {
        "same_respondent": 8,
        "lexical_cross_case": 4,
        "pairs": 12,
    }
    fitted = reduced["reduced"]
    assert fitted["fits"] is True
    assert fitted["calls"]["statement_copy_retry"] == 6
    assert fitted["calls"]["axial_relation_review"] == 12
    assert fitted["counterexample_required_but_blocked"] == 0
    assert fitted["relation_review_shortfall"] == 0
    assert fitted["coding_calls"] == 30
    assert fitted["total_calls"] == 43
    assert fitted["total_input_tokens"] == 258_000
    assert fitted["total_output_tokens"] == 37_681
    assert fitted["output_dollars"] == pytest.approx(0.094202)
    assert fitted["dollars"] == pytest.approx(0.416702)
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
    assert draw["lexical_eligibility"]["rule"] == LEXICAL_ELIGIBILITY_RULE
    assert draw["lexical_eligibility"]["eligible"] is False
    assert draw["round"] == []


def test_core_deferral_compares_content_not_run_ids():
    same = {
        "from_id": "a",
        "to_id": "b",
        "statement": "One claim.",
        "quote": "A quote.",
        "evidence_tier": 2,
        "downstream_eligible": True,
        "paradigm_role": "context",
        "document_count": 1,
    }
    baseline = {"eligible_relationships": [same], "category_counts": {"a": 1, "b": 1}}
    renamed = dict(same)
    assert core_differs({"eligible_relationships": [renamed], "category_counts": {"a": 1, "b": 1}}, baseline) is False
    changed = dict(same)
    changed["statement"] = "A different claim."
    assert core_differs({"eligible_relationships": [changed], "category_counts": {"a": 1, "b": 1}}, baseline) is True
    assert core_differs({"eligible_relationships": [], "category_counts": {}}, None) is None


def test_kill_guard_refuses_before_the_call():
    metadata = {
        "exp_dollars_spent": 0.54,
        "procedure": {"exp_axial_w1": {
            "enabled": True,
            "token_prices": FROZEN_TOKEN_PRICES,
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


def _book(source):
    return {row["code_id"]: row for row in source["codebook"]}


def _evidence_for(source, code_id):
    passage_ids = [
        str(row.get("passage_id") or "")
        for row in source["segments"]
        if str(row.get("code_id") or "") == code_id
    ]
    return [
        row for row in source["passages"]
        if str(row.get("passage_id") or "") in passage_ids
    ]


def test_exp_axial_w1_unexamined_single_document_round(tmp_path):
    source, quotes = _corpus()
    raised = source["raised_code_ids"]
    book = _book(source)
    evidence = classify_pair_evidence(
        raised,
        source["passages"],
        source["segments"],
        book,
    )
    examined = [{"from_id": "s00", "to_id": "s01"}]
    draw = select_unexamined_draw(evidence, examined, seed=4401)
    assert draw["status"] == "selected"
    assert draw["available"][CLASS_SAME] >= 16
    assert draw["available"][CLASS_LEXICAL] >= 8
    lexical = draw["lexical_eligibility"]
    assert lexical["rule"] == LEXICAL_ELIGIBILITY_RULE
    assert lexical["pattern_min_cases_is_the_rule"] is False
    assert lexical["eligible"] is True
    assert lexical["meeting_rule"] == 8
    assert lexical["failed"] == []
    assert "two distinctive stems" in lexical["statement"]
    assert ("s00", "s01") not in {
        tuple(sorted((row["from_id"], row["to_id"]))) for row in draw["round"]
    }
    scope = reduce_scope(core_candidates=len(raised))["reduced_scope"]
    direct = select_unexamined_draw(
        evidence,
        examined,
        seed=4401,
        same_count=scope["same_respondent"],
        lexical_count=scope["lexical_cross_case"],
    )
    assert direct["status"] == "selected"
    chosen = direct["round"]
    assert {row["evidence_class"] for row in chosen} == {CLASS_SAME, CLASS_LEXICAL}
    accepted = chosen[0]
    rejected_reuse = chosen[1]
    coder_for_fp = _RejectingCoder()
    accepted_evidence = _evidence_for(source, accepted["from_id"]) + _evidence_for(source, accepted["to_id"])
    accepted_passage, accepted_quote = quotes[accepted["from_id"]]
    second_passage, second_quote = quotes[accepted["to_id"]]
    history = tmp_path / "examined-history.json"
    _history(history, examined, [
        {
            "from_id": accepted["from_id"],
            "to_id": accepted["to_id"],
            "request_fingerprint": proposal_request_fingerprint(
                accepted["from_id"],
                accepted["to_id"],
                evidence=accepted_evidence,
                model=coder_for_fp,
                passage_ids=[row["passage_id"] for row in accepted_evidence],
            ),
            "policy_fingerprint": policy_fingerprint(coder_for_fp),
            "relationships": [
                {
                    "from_id": accepted["from_id"],
                    "to_id": accepted["to_id"],
                    "statement": "A stored proposal kept because the fingerprints match.",
                    "passage_id": accepted_passage,
                    "quote": accepted_quote,
                    "paradigm_role": "context",
                },
                {
                    "from_id": accepted["from_id"],
                    "to_id": accepted["to_id"],
                    "statement": "A second stored claim on the same endpoints.",
                    "passage_id": second_passage,
                    "quote": second_quote,
                    "paradigm_role": "context",
                },
            ],
        },
        {
            "from_id": rejected_reuse["from_id"],
            "to_id": rejected_reuse["to_id"],
            "request_fingerprint": "not-the-request",
            "policy_fingerprint": policy_fingerprint(coder_for_fp),
            "relationships": [{
                "from_id": rejected_reuse["from_id"],
                "to_id": rejected_reuse["to_id"],
                "statement": "This stored proposal must not be reused.",
                "passage_id": quotes[rejected_reuse["from_id"]][0],
                "quote": quotes[rejected_reuse["from_id"]][1],
                "paradigm_role": "context",
            }],
        },
    ])
    coder = _RejectingCoder()
    frozen = json.loads(Path("tests/fixtures/exp_axial_w1_freeze.json").read_text(encoding="utf-8"))
    assert frozen["commit"] == "offline-fixture"
    assert len(frozen["prompts"]) == 64
    with pytest.raises(AxialError, match="refused to start"):
        exp_axial_w1(
            source,
            coder,
            history_path=history,
            frozen_fingerprints={**frozen, "commit": "different-commit"},
            baseline_core={"eligible_relationships": [], "category_counts": {code_id: 0 for code_id in raised}},
            token_prices=FROZEN_TOKEN_PRICES,
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
        baseline_core={"eligible_relationships": [], "category_counts": {code_id: 0 for code_id in raised}},
        token_prices=FROZEN_TOKEN_PRICES,
        seed=4401,
        commit="offline-fixture",
        decision_mode="llm",
        interrupt=False,
    )
    assert report["status"] == "completed"
    assert report["preflight"]["original"]["pairs"] == 24
    assert report["preflight"]["original"]["coding_calls"] == 36
    assert report["preflight"]["original"]["calls"]["axial_core_review"] == len(raised) - 1
    assert report["preflight"]["original"]["total_input_tokens"] == 366_000 + (len(raised) - 1) * 6000
    assert report["preflight"]["original"]["output_dollars"] is not None
    assert report["preflight"]["original"]["prices"]["output_usd_per_million"] == 2.50
    assert report["preflight"]["reduced"]["coding_calls"] == 30
    assert report["preflight"]["reduced"]["coding_input_tokens"] == 180_000
    assert report["preflight"]["reduced"]["total_calls"] > 36
    assert report["preflight"]["reduced"]["total_input_tokens"] > 216_000
    assert report["preflight"]["reduced_scope"]["same_respondent"] == 8
    assert report["preflight"]["reduced_scope"]["lexical_cross_case"] == 4
    assert report["allowances"]["axial_proposal"] == 12
    assert report["allowances"]["statement_copy_retry"] == 6
    assert report["allowances"]["axial_counterexample"] == 12
    assert report["allowances"]["axial_relation_review"] == 12
    assert report["allowances"]["axial_saturation_review"] == 1
    assert report["preflight"]["reduced"]["counterexample_required_but_blocked"] == 0
    assert report["preflight"]["reduced"]["relation_review_shortfall"] == 0
    assert report["preflight"]["original"]["counterexample_required_but_blocked"] == 18
    reviewed = [row for row in report["result"].relationships if not row.get("synthetic")]
    assert reviewed
    assert all(row.get("reviewer_disposition") for row in reviewed)
    assert report["historical_expenditure"]["dollars"] == 0.02
    assert report["new_spend"]["calls"] > report["metrics"]["overall"]["new_pairs_examined"]
    assert report["universe"]["codebook_codes"] == len(raised) + 1
    assert report["universe"]["pair_universe"] == len(raised) * (len(raised) - 1) // 2
    assert report["universe"]["restricted_to_raised"] is True
    assert report["draw_method"] == "direct_seeded_draw"
    assert "not-raised" not in {row["from_id"] for row in report["round"]}
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
        frozen_fingerprints=frozen,
        baseline_core={"eligible_relationships": [], "category_counts": {code_id: 1 for code_id in raised}},
        token_prices=FROZEN_TOKEN_PRICES,
        seed=4401,
        commit="offline-fixture",
        decision_mode="llm",
        interrupt=False,
    )
    assert differed["core"]["status"] == "ran"
    assert differed["core"]["calls"] > 0
    assert differed["preflight"]["reduced"]["calls"]["axial_core_review"] == len(raised) - 1


def test_the_package_does_not_shadow_the_experiment_module():
    import automated_grounded_theory as pkg
    import automated_grounded_theory.exp_axial_w1 as mod

    assert pkg.exp_axial_w1 is mod
    assert callable(mod.exp_axial_w1)


def test_lexical_eligibility_can_fail_without_substitution(tmp_path):
    source, _quotes = _corpus()
    examined = [{"from_id": f"x{i}", "to_id": f"y{j}"} for i in range(4) for j in range(2)]
    history = tmp_path / "examined-history.json"
    _history(history, examined, [])
    coder = _RejectingCoder()
    frozen = json.loads(Path("tests/fixtures/exp_axial_w1_freeze.json").read_text(encoding="utf-8"))
    report = exp_axial_w1(
        source,
        coder,
        history_path=history,
        frozen_fingerprints=frozen,
        baseline_core={"eligible_relationships": [], "category_counts": {}},
        seed=4401,
        commit="offline-fixture",
        decision_mode="llm",
        interrupt=False,
    )
    assert report["status"] == "stopped"
    assert report["model_calls"] == 0
    assert coder.proposal_calls == 0
    assert report["draw"]["lexical_eligibility"]["eligible"] is False
    assert report["round"] if "round" in report else [] == []


def test_kill_guard_stops_relation_review_before_the_model():
    import asyncio

    from automated_grounded_theory.axial import AxialDecisionReviewer, AxialRelationDecisions

    reviewer = AxialDecisionReviewer(_RejectingCoder(), max_calls=4, max_tokens=24000, token_budget=6000)
    metadata = {
        "procedure": {"exp_axial_w1": {"enabled": True}},
        "exp_possible_dollars": 0.55,
        "decision_calls_used": 0,
        "decision_calls_by_step": {},
    }
    payload = {
        "kind": "axial_relation_review",
        "relationships": [{"relationship_id": "rel-1", "statement": "A claim."}],
    }
    parsed, audit = asyncio.run(reviewer.decide(payload, AxialRelationDecisions, metadata))
    assert parsed is None
    assert audit["reason"] == "kill_guard"
    assert reviewer.model.call_count == 0


def test_unknown_and_timed_out_usage_stays_unmeasured():
    from automated_grounded_theory.exp_axial_w1 import account_call_spend, dispatch_spend_gate

    metadata = {"procedure": {"exp_axial_w1": {"enabled": True}}}
    account_call_spend(metadata, "axial_counterexample", None, known=False, timed_out=True)
    assert metadata["exp_spend_unknown"] is True
    assert "exp_dollars_spent" not in metadata
    assert metadata["exp_possible_dollars"] > 0
    metadata["exp_possible_dollars"] = 0.55
    assert dispatch_spend_gate(metadata, "axial_saturation_review", 6000) == "kill_guard"


def test_accepted_rows_send_counterexamples_and_a_malformed_proposal_retries(tmp_path):
    source, _quotes = _corpus()
    history = tmp_path / "examined-history.json"
    _history(history, [{"from_id": "s00", "to_id": "s01"}], [])
    coder = _RejectingCoder(mode="accept")
    frozen = json.loads(Path("tests/fixtures/exp_axial_w1_freeze.json").read_text(encoding="utf-8"))
    report = exp_axial_w1(
        source,
        coder,
        history_path=history,
        frozen_fingerprints=frozen,
        baseline_core={"eligible_relationships": [], "category_counts": {code_id: 0 for code_id in source["raised_code_ids"]}},
        seed=4401,
        commit="offline-fixture",
        decision_mode="llm",
        interrupt=False,
    )
    assert report["status"] == "completed"
    sent = [
        row for row in report["result"].prompt_log
        if row.get("kind") == "axial_counterexample" and row.get("status") not in {"omitted", "not_sent", "refused"}
    ]
    assert sent
    retries = [
        row for row in report["result"].prompt_log
        if row.get("statement_copy_retry") is True and row.get("status") not in {"omitted", "not_sent"}
    ]
    assert retries
    assert report["preflight"]["original"]["counterexample_required_but_blocked"] == 18


def test_an_interrupted_experiment_can_resume(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    source, _quotes = _corpus()
    history = tmp_path / "examined-history.json"
    _history(history, [{"from_id": "s00", "to_id": "s01"}], [])
    coder = _RejectingCoder()
    frozen = json.loads(Path("tests/fixtures/exp_axial_w1_freeze.json").read_text(encoding="utf-8"))
    saver = InMemorySaver()
    common = dict(
        history_path=history,
        frozen_fingerprints=frozen,
        baseline_core={"eligible_relationships": [], "category_counts": {code_id: 0 for code_id in source["raised_code_ids"]}},
        seed=4401,
        commit="offline-fixture",
        decision_mode="human",
        interrupt=True,
        checkpointer=saver,
        thread_id="exp-axial-w1-resume",
    )
    paused = exp_axial_w1(source, coder, **common)
    assert paused["result"].interrupted is True
    resumed = exp_axial_w1(
        source,
        coder,
        resuming=True,
        resume={"relationships": [{
            "relationship_id": paused["result"].relationships[0]["relationship_id"],
            "action": "reject",
        }]},
        **common,
    )
    assert resumed["result"].interrupted is True
    finished = exp_axial_w1(
        source,
        coder,
        resuming=True,
        resume={"decision": "not_saturated", "rationale": "Another document could still add a code."},
        **common,
    )
    assert finished["result"].interrupted is False
