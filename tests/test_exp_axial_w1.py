"""Offline gate for exp-axial-w1. The model is a fake. Spend on the run is not a live bill."""

from __future__ import annotations

import json
import sys
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
    repair_calls: int = 0
    counter_calls: int = 0

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

    def with_structured_output(self, schema, **kwargs):
        if self.mode != "gate":
            return super().with_structured_output(schema, **kwargs)
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        parent = self

        def _call(messages):
            parent.call_count += 1
            name = getattr(schema, "__name__", "")
            from tests.test_v080 import _payload

            body = _payload(messages)
            text = ""
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
                text += str(content)
            usage = {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42}
            raw_content = "{}"
            parsed = None
            if name == "AxialProposalBatch":
                if "REPAIR THIS OUTPUT" in text:
                    parent.repair_calls += 1
                    parsed = parent.proposals(body)
                else:
                    parent.proposal_calls += 1
                    if parent.proposal_calls == 1:
                        raw = AIMessage(
                            content="this is not valid schema output",
                            usage_metadata=usage,
                        )
                        return {"raw": raw, "parsed": None, "parsing_error": ValueError("malformed schema output")}
                    if parent.proposal_calls == 3:
                        parsed = AxialProposalBatch(relationships=[], nothing_to_propose=True)
                    else:
                        parsed = parent.proposals(body)
            elif name == "AxialCounterexampleBatch":
                parent.counter_calls += 1
                parsed = parent.counterexamples(body)
                if parent.counter_calls == 1:
                    # Settles near $0.56, under the $0.60 cap, so the next reservation hits the $0.55 kill guard.
                    usage = {"input_tokens": 448_000, "output_tokens": 20, "total_tokens": 448_020}
            elif name == "AxialRelationDecisions":
                parsed = parent.relation_review(body)
            elif name == "ModelSaturationDecision":
                parsed = schema(assessment="not_saturated", rationale="A later document still adds a code.")
            else:
                parsed = schema()
            raw = AIMessage(content=raw_content, usage_metadata=usage)
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


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
    assert "axial_core_review" not in report["preflight"]["original"]["calls"]
    assert "axial_core_review" not in report["preflight"]["original"]["calls_sent"]
    assert report["preflight"]["original"]["total_input_tokens"] == 366_000
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
    assert report["core"] == {
        "status": "not_requested",
        "reason": "evaluated separately",
        "calls": 0,
        "dollars": 0,
    }
    assert report["provider_max_attempts"] == 1
    assert report["provider_retries"] == "disabled"
    outcomes = report["pair_outcomes"]
    assert len(outcomes) == len(report["round"]) == 12
    assert {row["status"] for row in outcomes} <= {
        "attempted", "successfully_examined", "empty_response", "failed", "budget_blocked",
    }
    assert report["pair_coverage"]["successfully_examined"] == report["metrics"]["overall"]["new_pairs_examined"]
    assert "accepted" not in report["metrics"]["overall"]["eligibility_failures_by_reason"]
    assert all(row.get("stage") != "axial_core_review" for row in report["attempt_ledger"])
    assert report["saturation"]["status"] in {"executed", "blocked", "not_applicable"}
    assert report["saturation"]["reason"] is not None
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
    assert relevance["overall"]["counterexample_relevance"] is None
    assert relevance["overall"]["returned_assessments"] == 0
    assert relevance["overall"]["calls"] == 0
    procedure = report["result"].metadata["procedure"]
    assert procedure["max_axial_review_calls"] == 36
    assert procedure["max_axial_review_tokens"] == 216_000
    assert procedure["pattern_min_cases"] == 2
    assert procedure["one_document_one_respondent"] is False
    assert procedure["allowance_transfers"] == []
    assert report["result"].metadata["retry_policy"]["provider_max_attempts"] == 1


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
    assert report["status"] == "preflight_blocked"
    assert report["status"] != "completed"
    assert report["reason"] in {"lexical_ineligible", "class_shortfall"}
    assert report["model_calls"] == 0
    assert report["new_spend"]["calls"] == 0
    assert coder.proposal_calls == 0
    assert report["draw"]["lexical_eligibility"]["eligible"] is False
    assert report.get("round", []) == []
    assert report["core"]["status"] == "not_requested"
    assert report["core"]["reason"] == "evaluated separately"
    assert report["saturation"]["status"] == "not_applicable"


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
    assert metadata["exp_dollars_spent"] == 0
    assert metadata["exp_possible_dollars"] > 0
    row = metadata["exp_attempt_ledger"][-1]
    assert row["status"] == "timeout"
    assert row["provider_usage"] is None
    assert row["settlement"]["dollars"] is None
    assert row["estimated_input"] != row["provider_usage"]
    metadata["exp_baseline_possible"] = 0.55
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


def _round_is_empty(report) -> bool:
    return report.get("round", []) == []


def _coverage_holds(outcomes, selected) -> bool:
    if len(outcomes) != len(selected):
        return False
    allowed = {"attempted", "successfully_examined", "empty_response", "failed", "budget_blocked"}
    keys = set()
    for row in outcomes:
        if row.get("status") not in allowed:
            return False
        keys.add(tuple(sorted((row["from_id"], row["to_id"]))))
    return keys == set(selected)


def test_checks_fail_when_the_behavior_is_broken():
    """Mutation-style evidence. A broken report fails the same checks the gate uses."""
    present = {"round": [{"from_id": "a", "to_id": "b"}]}
    assert _round_is_empty(present) is False
    assert _round_is_empty({}) is True
    # The old lexical assertion was `report["round"] if "round" in report else [] == []`.
    # Comparison binds first, so a non-empty round is still true.
    old_expression = present["round"] if "round" in present else [] == []
    assert old_expression
    assert not _round_is_empty(present)
    outcomes = [{"from_id": "a", "to_id": "b", "status": "successfully_examined"}]
    selected = {("a", "b")}
    assert _coverage_holds(outcomes, selected) is True
    assert _coverage_holds(outcomes[:-1], selected) is False
    dropped_status = [{"from_id": "a", "to_id": "b", "status": "examined"}]
    assert _coverage_holds(dropped_status, selected) is False
    reasons = {"reviewer_rejected": 1, "limited_document_support": 1}
    assert "accepted" not in reasons
    broken_reasons = {"accepted": 2}
    assert "accepted" in broken_reasons


def test_a_reservation_is_not_expenditure_and_a_timeout_is_possible_spend():
    from automated_grounded_theory.exp_axial_w1 import reserve_attempt, settle_attempt

    metadata = {"procedure": {"exp_axial_w1": {"enabled": True}}}
    held = reserve_attempt(
        metadata,
        "axial_relation_review",
        6000,
        payload_fingerprint="payload-a",
    )
    assert held["status"] == "reserved"
    assert held["provider_usage"] is None
    assert held["estimated_input"] == 6000
    assert held["output_allowance"] > 0
    assert held["reservation"]["dollars"] > 0
    assert metadata["exp_dollars_spent"] == 0
    assert metadata["exp_possible_dollars"] == held["reservation"]["dollars"]
    assert metadata["exp_unresolved_reservation_dollars"] == held["reservation"]["dollars"]
    settle_attempt(metadata, held["attempt_id"], usage={"input_tokens": 30, "output_tokens": 12}, known=True)
    settled = metadata["exp_attempt_ledger"][0]
    assert settled["status"] == "settled"
    assert settled["provider_usage"] == {"input_tokens": 30, "output_tokens": 12}
    assert settled["provider_usage"]["input_tokens"] != settled["estimated_input"]
    assert settled["settlement"]["dollars"] != settled["reservation"]["dollars"]
    assert metadata["exp_dollars_spent"] == settled["settlement"]["dollars"]
    assert metadata["exp_unresolved_reservation_dollars"] == 0
    timed = reserve_attempt(metadata, "axial_saturation_review", 6000, payload_fingerprint="payload-b")
    settle_attempt(metadata, timed["attempt_id"], known=False, timed_out=True)
    timeout_row = metadata["exp_attempt_ledger"][-1]
    assert timeout_row["status"] == "timeout"
    assert timeout_row["provider_usage"] is None
    assert timeout_row["settlement"]["dollars"] is None
    assert metadata["exp_spend_unknown"] is True
    assert metadata["exp_dollars_spent"] == settled["settlement"]["dollars"]
    assert metadata["exp_possible_dollars"] > metadata["exp_dollars_spent"]


def test_an_empty_examined_universe_is_no_work(tmp_path):
    source, _quotes = _corpus()
    raised = source["raised_code_ids"]
    evidence = classify_pair_evidence(raised, source["passages"], source["segments"], _book(source))
    examined = []
    for name in (CLASS_SAME, CLASS_LEXICAL):
        for row in evidence["by_class"][name]:
            examined.append({"from_id": row["from_id"], "to_id": row["to_id"]})
    history = tmp_path / "examined-history.json"
    _history(history, examined, [])
    coder = _RejectingCoder()
    frozen = json.loads(Path("tests/fixtures/exp_axial_w1_freeze.json").read_text(encoding="utf-8"))
    report = exp_axial_w1(
        source,
        coder,
        history_path=history,
        frozen_fingerprints=frozen,
        seed=4401,
        commit="offline-fixture",
        decision_mode="llm",
        interrupt=False,
    )
    assert report["status"] == "no_work"
    assert report["status"] != "completed"
    assert report["model_calls"] == 0
    assert report["new_spend"]["calls"] == 0
    assert coder.proposal_calls == 0
    assert report["core"]["reason"] == "evaluated separately"
    assert report["saturation"]["status"] == "not_applicable"


def test_exp_axial_w1_integration_gate(tmp_path):
    """One resumed run: list codebook, mixed review, schema repair, blocked counterexample, guard, spend continuity."""
    from langgraph.checkpoint.memory import InMemorySaver

    source, _quotes = _corpus()
    assert isinstance(source["codebook"], list)
    assert len(source["codebook"]) > len(source["raised_code_ids"])
    history = tmp_path / "examined-history.json"
    _history(history, [{"from_id": "s00", "to_id": "s01"}], [])
    coder = _RejectingCoder(mode="gate")
    frozen = json.loads(Path("tests/fixtures/exp_axial_w1_freeze.json").read_text(encoding="utf-8"))
    saver = InMemorySaver()
    common = dict(
        history_path=history,
        frozen_fingerprints=frozen,
        seed=4401,
        commit="offline-fixture",
        decision_mode="human",
        interrupt=True,
        checkpointer=saver,
        thread_id="exp-axial-w1-gate",
        provider_max_attempts=4,
    )
    paused = exp_axial_w1(source, coder, **common)
    assert paused["result"].interrupted is True
    assert paused["result"].interrupt_payload["kind"] == "axial_relations"
    assert paused["provider_max_attempts"] == 1
    assert paused["result"].metadata["retry_policy"]["provider_max_attempts"] == 1
    ledger_at_pause = list(paused["attempt_ledger"])
    ids_at_pause = [row["attempt_id"] for row in ledger_at_pause]
    assert ids_at_pause == list(dict.fromkeys(ids_at_pause))
    repairs = [row for row in ledger_at_pause if row["stage"] == "schema_repair"]
    assert repairs
    assert repairs[0]["status"] == "settled"
    assert repairs[0]["reservation"]["dollars"] > 0
    assert repairs[0]["provider_usage"]["input_tokens"] != repairs[0]["estimated_input"]
    assert coder.repair_calls == 1
    spent_at_pause = paused["result"].metadata["exp_dollars_spent"]
    calls_at_pause = paused["new_spend"]["calls"]
    budget_at_pause = int(paused["result"].metadata["provider_budget"]["calls_used"])
    allowances_at_pause = dict(paused["result"].metadata["procedure"])
    assert allowances_at_pause["max_axial_review_calls"] == 36
    relationships = [row for row in paused["result"].relationships if not row.get("synthetic")]
    assert relationships
    actions = []
    accepted = 0
    rejected = 0
    for row in relationships:
        keep = "s00" in {str(row.get("from_id") or ""), str(row.get("to_id") or "")}
        actions.append({
            "relationship_id": row["relationship_id"],
            "action": "accept" if keep else "reject",
        })
        accepted += int(keep)
        rejected += int(not keep)
    assert accepted and rejected
    resumed = exp_axial_w1(
        source,
        coder,
        resuming=True,
        resume={"relationships": actions},
        **common,
    )
    assert resumed["result"].interrupted is True
    assert resumed["result"].interrupt_payload["kind"] == "thematic_saturation"
    ledger_mid = list(resumed["attempt_ledger"])
    ids_mid = [row["attempt_id"] for row in ledger_mid]
    assert len(ids_mid) == len(set(ids_mid))
    assert ids_mid[: len(ids_at_pause)] == ids_at_pause
    for before, after in zip(ledger_at_pause, ledger_mid):
        assert after["attempt_id"] == before["attempt_id"]
        assert after["status"] == before["status"]
        assert after["settlement"] == before["settlement"]
    new_rows = ledger_mid[len(ledger_at_pause):]
    assert new_rows
    assert all(row["stage"] != "axial_proposal" for row in new_rows)
    blocked = [row for row in new_rows if row["status"] == "budget_blocked"]
    assert blocked
    assert blocked[0]["stage"] == "axial_counterexample"
    assert blocked[0]["reason"] == "kill_guard"
    assert blocked[0]["reservation"] is None
    assert coder.counter_calls == 1
    assert resumed["guard_exhaustion"]["stage"] == "axial_counterexample"
    assert resumed["new_spend"]["calls"] == calls_at_pause + 1
    assert resumed["result"].metadata["exp_dollars_spent"] > spent_at_pause
    assert int(resumed["result"].metadata["provider_budget"]["calls_used"]) >= budget_at_pause
    assert resumed["result"].metadata["procedure"]["max_axial_review_calls"] == 36
    assert resumed["saturation"]["status"] == "not_applicable"
    assert resumed["saturation"]["reason"] == "awaiting_saturation_resume"
    finished = exp_axial_w1(
        source,
        coder,
        resuming=True,
        resume={"decision": "not_saturated", "rationale": "Another document could still add a code."},
        **common,
    )
    assert finished["result"].interrupted is False
    assert finished["status"] == "completed"
    final_ids = [row["attempt_id"] for row in finished["attempt_ledger"]]
    assert final_ids == ids_mid
    assert finished["new_spend"]["calls"] == resumed["new_spend"]["calls"]
    assert finished["result"].metadata["exp_dollars_spent"] == resumed["result"].metadata["exp_dollars_spent"]
    assert finished["saturation"]["status"] == "executed"
    assert finished["saturation"]["actor"] == "human"
    assert finished["saturation"]["reason"] == "not_saturated"
    assert finished["core"]["status"] == "not_requested"
    assert finished["core"]["reason"] == "evaluated separately"
    outcomes = finished["pair_outcomes"]
    selected = {tuple(sorted((row["from_id"], row["to_id"]))) for row in finished["round"]}
    assert _coverage_holds(outcomes, selected)
    by_status = finished["pair_coverage"]["by_status"]
    assert by_status.get("empty_response") == 1
    assert by_status.get("successfully_examined") == len(selected) - 1
    assert finished["metrics"]["overall"]["new_pairs_examined"] == len(selected) - 1
    assert "accepted" not in (finished["metrics"]["overall"].get("eligibility_failures_by_reason") or {})
    counter = finished["counterexamples"]
    assert counter["calls"] >= 1
    assert counter["returned_assessments"] >= 1
    assert counter["relevant_assessments"] >= 1
    assert counter["counterexample_relevance"] is not None
    kept = [row for row in finished["result"].relationships if not row.get("synthetic")]
    dispositions = {str(row.get("human_disposition") or row.get("disposition") or "") for row in kept}
    requested = {str(row.get("acceptance_requested") or "") for row in kept}
    assert "accepted" in requested
    assert "rejected" in dispositions
    assert "accepted" not in (finished["metrics"]["overall"].get("eligibility_failures_by_reason") or {})
    sent_counter = [
        row for row in finished["attempt_ledger"]
        if row["stage"] == "axial_counterexample" and row["status"] == "settled"
    ]
    assert sent_counter
    assert sent_counter[0]["reservation"]["dollars"] != sent_counter[0]["settlement"]["dollars"]
    assert finished["new_spend"]["dollars"] == finished["result"].metadata["exp_dollars_spent"]


def test_candidate_manifest_pins_the_package_head(tmp_path):
    import subprocess

    out = tmp_path / "exp-axial-w1-candidate.json"
    subprocess.check_call([sys.executable, "scripts/exp_axial_w1_manifest.py", "--out", str(out)], cwd=Path.cwd())
    payload = json.loads(out.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    frozen = json.loads(Path("tests/fixtures/exp_axial_w1_freeze.json").read_text(encoding="utf-8"))
    assert payload["candidate_manifest_version"] == 1
    assert payload["experiment"] == "exp-axial-w1"
    assert payload["commit"] == head
    assert payload["fingerprint_scheme"]["digest"] == "sha256"
    assert payload["fingerprint_scheme"]["json"]["ensure_ascii"] is False
    assert payload["prompts"] == frozen["prompts"]
    assert payload["eligibility_rules"] == frozen["eligibility_rules"]
    assert payload["export_manifest"] is None
    assert payload["prices"]["input_usd_per_million"] == 1.25
    assert payload["prices"]["output_usd_per_million"] == 2.50
    assert "6766bd28fc27cf41" not in out.read_text(encoding="utf-8")
