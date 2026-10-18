"""0.9.10 regressions. Fake models only. No live evaluation."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import __version__, axial_code, selective_code
from automated_grounded_theory.async_support import CallLimiter, ProviderAllowance, run_sync
from automated_grounded_theory.axial import (
    DEFAULT_AXIAL_TOKEN_BUDGET,
    AxialProposalBatch,
    AxialRelationshipDraft,
    _KIND_PROMPTS,
    _annotate_endpoints,
    _apply_counterexample,
    _axial_token_plan,
    _flag_reviewer_causal_claim,
    _tiered_accept,
)
from automated_grounded_theory.category_state import plan_relation_pairs
from automated_grounded_theory.focused import _incomplete_warning, _record_timeout
from automated_grounded_theory.gate_evaluation import export_gate_relationships
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import (
    PATTERN_MIN_CASES,
    apply_evidence_tier,
    candidate_cross_case_passages,
    case_id_of,
    classify_pair_evidence,
    denies_causal_language,
)
from automated_grounded_theory.sampling import (
    DEFAULT_SAMPLING_TOKEN_BUDGET,
    SamplingRuntime,
    _IdleModel,
    _apply_human_sampling,
    _baseline_sampling_stages,
    _pack_saturation,
    _selection_over_budget,
)
from automated_grounded_theory.saturation import stopping_criteria_report
from tests.test_v080 import STATEMENT, FakeAxialCoder, _code, _corpus
from tests.test_v084 import _one_relationship


def test_version_is_0910():
    assert __version__ == "0.9.14"
    assert POLICY_VERSION == "0.9.14"
    assert PATTERN_MIN_CASES == 2
    assert DEFAULT_AXIAL_TOKEN_BUDGET == 6000
    assert DEFAULT_SAMPLING_TOKEN_BUDGET == 6000


def test_evidence_aware_pairs_are_taken_before_the_fill():
    """Same-passage, then same-respondent, then lexical cross-case, then the fill."""
    ids = ["a", "b", "c", "d"]
    evidence = {
        "same_passage": [{"from_id": "a", "to_id": "d"}],
        "same_respondent": [{"from_id": "a", "to_id": "c"}],
        "lexical_cross_case": [{"from_id": "b", "to_id": "d"}],
    }
    first = plan_relation_pairs(ids, round_size=3, seed=1, evidence_classes=evidence)
    again = plan_relation_pairs(ids, round_size=3, seed=9, evidence_classes=evidence)
    bands = [row["band"] for row in first["round"]]
    pairs = [(row["from_id"], row["to_id"]) for row in first["round"]]
    assert bands == ["same_passage", "same_respondent", "lexical_cross_case"]
    assert pairs == [("a", "d"), ("a", "c"), ("b", "d")]
    assert again["round"] == first["round"]
    assert first["plausible_count"] == 3
    assert first["examined_plausible"] == 3
    assert first["prioritized_by_class"] == {
        "same_passage": 1,
        "same_respondent": 1,
        "lexical_cross_case": 1,
    }
    assert first["pair_count"] == 6
    assert first["examined_count"] == 3
    wider = plan_relation_pairs(ids, round_size=4, seed=2, evidence_classes=evidence)
    assert [row["band"] for row in wider["round"][:3]] == bands
    assert wider["round"][3]["band"] in {"obvious", "less_obvious"}
    blind = plan_relation_pairs(ids, round_size=1, seed=0, obvious_share=1)
    assert blind["plausible_count"] is None
    assert blind["round"][0]["band"] == "obvious"


def test_prompt_allows_a_tier1_hypothesis_and_counts_blank_rows_apart_from_empty_statements():
    prompt = _KIND_PROMPTS["axial_proposal"]
    assert "one passage" in prompt
    assert "one respondent" in prompt
    assert "nothing_to_propose" in prompt
    assert "unrelated passages" in prompt
    empty = AxialProposalBatch(nothing_to_propose=True, relationships=[])
    assert empty.nothing_to_propose is True
    assert empty.relationships == []
    assert AxialRelationshipDraft().statement is None

    class _None(FakeAxialCoder):
        def proposals(self, body):
            del body
            return AxialProposalBatch(relationships=[], nothing_to_propose=True)

    class _Blank(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or ""),
                        to_id=str(pair.get("to_id") or ""),
                        statement="",
                        quote="",
                        quote_from="",
                        quote_to="",
                    )
                ]
            )

    class _Statementless(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or ""),
                        to_id=str(pair.get("to_id") or ""),
                        statement="",
                        passage_id="p1",
                        quote="Packing the suitcase took all night",
                    )
                ]
            )

    class _Stitched(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or ""),
                        to_id=str(pair.get("to_id") or ""),
                        statement="Packing may be related to returning.",
                        passage_id="p1",
                        quote_from="Packing the suitcase took all night",
                        quote_to="The return to work after leave",
                    )
                ]
            )

    class _Single(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or ""),
                        to_id=str(pair.get("to_id") or ""),
                        statement="Packing may be related to the trip.",
                        passage_id="p1",
                        quote="Packing the suitcase took all night before the trip.",
                    )
                ]
            )

    common = dict(
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    none = axial_code(_corpus(), _None(), **common)
    assert none.metadata["proposal_diagnostics"]["explicit_none"] == 1
    assert none.metadata["proposal_diagnostics"]["blank_rows"] == 0
    assert none.relationships == []
    blank = axial_code(_corpus(), _Blank(), **common)
    assert blank.metadata["proposal_diagnostics"]["blank_rows"] == 1
    assert blank.metadata["proposal_diagnostics"]["empty_statements"] == 0
    assert blank.relationships == []
    statementless = axial_code(_corpus(), _Statementless(), **common)
    assert statementless.metadata["proposal_diagnostics"]["empty_statements"] == 1
    assert statementless.metadata["proposal_diagnostics"]["blank_rows"] == 0
    stitched = axial_code(_corpus(), _Stitched(), **common)
    assert stitched.metadata["proposal_diagnostics"]["rejection_reasons"]["stitched_unrelated_passages"] == 1
    assert stitched.relationships == []
    kept = axial_code(_corpus(), _Single(), **common)
    assert kept.relationships
    assert kept.relationships[0]["disposition"] == "proposed"
    assert "stitched_unrelated_passages" not in (kept.metadata.get("proposal_diagnostics") or {}).get("rejection_reasons", {})


def test_pattern_reach_warning_does_not_change_the_minimum():
    result = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    warning = result.metadata["pattern_reach_warning"]
    assert warning["behavior_unchanged"] is True
    assert warning["pattern_min_cases"] == 2
    assert warning["pairs_reaching_pattern_min_cases"] == 0
    assert warning["single_document_code_count"] == 2
    assert result.metadata["pair_coverage"]["round_count"] == 1
    assert "pattern_min_cases is unchanged" in warning["warning"]


def test_case_id_reads_metadata_respondent_and_tier2_gets_stricter():
    """Decision: question-documents that share metadata.respondent_id are one case.

    Top-level respondent_id still wins. pattern_min_cases stays 2, so two
    question-documents from one respondent no longer reach tier 2.
    """
    questions = [
        {
            "passage_id": "R010_q1",
            "doc_id": "R010_q1",
            "metadata": {"respondent_id": "R010"},
            "text": "The briefing covered the shift.",
        },
        {
            "passage_id": "R010_q2",
            "doc_id": "R010_q2",
            "metadata": {"respondent_id": "R010"},
            "text": "The debrief covered the shift.",
        },
    ]
    segments = [
        {"code_id": "c-a", "passage_id": "R010_q1"},
        {"code_id": "c-b", "passage_id": "R010_q1"},
        {"code_id": "c-a", "passage_id": "R010_q2"},
        {"code_id": "c-b", "passage_id": "R010_q2"},
    ]
    assert case_id_of(questions[0]) == "R010"
    assert case_id_of(questions[1]) == "R010"
    assert case_id_of({**questions[0], "respondent_id": "TOP"}) == "TOP"
    collapsed = classify_pair_evidence(["c-a", "c-b"], questions, segments)
    separate = classify_pair_evidence(
        ["c-a", "c-b"],
        [{key: value for key, value in row.items() if key != "metadata"} for row in questions],
        segments,
    )
    assert collapsed["pairs_reaching_pattern_min_cases"] == 0
    assert separate["pairs_reaching_pattern_min_cases"] == 1
    assert collapsed["pattern_min_cases"] == 2
    assert collapsed["by_class"]["same_passage"] == [{"from_id": "c-a", "to_id": "c-b"}]
    split = classify_pair_evidence(
        ["c-a", "c-b"],
        questions,
        [
            {"code_id": "c-a", "passage_id": "R010_q1"},
            {"code_id": "c-b", "passage_id": "R010_q2"},
        ],
    )
    assert split["by_class"]["same_respondent"] == [{"from_id": "c-a", "to_id": "c-b"}]
    assert split["by_class"]["same_passage"] == []


def test_stem_only_lexical_overlap_is_not_offered_as_cross_case():
    """Decision: one distinctive stem, the seed11 'training' shape, is lexical_only.

    It is not a candidate_cross_case passage and its support_cap is 1.
    Two distinctive stems on each endpoint are still offered.
    """
    codebook = {
        "c-dei": {"code_id": "c-dei", "code": "belonging", "definition": "diversity equity inclusion training"},
        "c-policy": {"code_id": "c-policy", "code": "policy", "definition": "handbook policy compliance training"},
    }
    passages = {
        "p-train": {
            "passage_id": "p-train",
            "doc_id": "d-other",
            "respondent_id": "R-other",
            "text": "The training session was optional.",
        },
        "p-both": {
            "passage_id": "p-both",
            "doc_id": "d-both",
            "respondent_id": "R-both",
            "text": "The training covered diversity equity and the handbook policy.",
        },
    }
    lexical_only: list[dict] = []
    offered = candidate_cross_case_passages(
        passages,
        codebook,
        [{"from_id": "c-dei", "to_id": "c-policy"}],
        limit=5,
        lexical_only=lexical_only,
    )
    assert [row["passage_id"] for row in offered] == ["p-both"]
    assert offered[0]["candidate_cross_case"] is True
    assert offered[0]["lexical_only"] is False
    assert [row["passage_id"] for row in lexical_only] == ["p-train"]
    assert lexical_only[0]["lexical_only"] is True
    assert lexical_only[0]["candidate_cross_case"] is False
    assert lexical_only[0]["support_cap"] == 1


def test_default_axial_series_records_the_small_fraction():
    readme = Path("README.md").read_text(encoding="utf-8")
    note = Path("docs/design-review-fixes-0910.md").read_text(encoding="utf-8")
    assert "small fraction" in readme
    assert "max_axial_review_calls" in readme
    assert "default 8" in note or "defaults to 8" in note
    assert "small fraction" in note
    result = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=1,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    coverage = result.metadata["pair_coverage"]
    assert "defaults to 8" in coverage["note"]
    assert "small fraction" in coverage["note"]
    assert "prioritized_by_class" in coverage
    assert coverage["examined_count"] < coverage["pair_count"] or coverage["pair_count"] == 1


def test_causal_wording_below_tier3_is_withheld_and_the_reviewer_mismatch_is_flagged():
    """Decision: wording_exceeds_tier below tier 3 withholds downstream_eligible.

    The statement is not rewritten. A reviewer rationale that says the text
    uses non-causal wording, applied to 'leads to', is a mismatch.
    """
    passages = {
        "p1": {
            "passage_id": "p1",
            "doc_id": "d1",
            "respondent_id": "r1",
            "text": "The tools for the shift sit beside coworkers on the crew tonight",
        }
    }
    book = {
        "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
        "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
    }
    row = {
        "relationship_id": "rel-0001",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote_from": "tools for the shift sit",
        "quote_to": "coworkers on the crew tonight",
        "statement": "The tools for the shift leads to coworkers on the crew.",
        "disposition": "proposed",
        "flags": [],
        "evidence_condition": "anchored",
    }
    apply_evidence_tier(row, passages=passages, codebook=book, pattern_min_cases=2)
    assert isinstance(row["evidence_tier"], int) and row["evidence_tier"] < 3
    assert "wording_exceeds_tier" in row["flags"]
    assert "causal_wording_withheld" in row["flags"]
    assert row["downstream_eligible"] is False
    assert row["statement"] == "The tools for the shift leads to coworkers on the crew."
    assert denies_causal_language("uses non-causal wording") is True
    reviewed = {
        "relationship_id": "rel-0003",
        "statement": "The schedule leads to the change.",
        "flags": [],
    }
    _flag_reviewer_causal_claim(reviewed, "The draft uses non-causal wording.")
    assert "reviewer_causal_claim_mismatch" in reviewed["flags"]


def test_sampling_token_baseline_does_not_charge_upstream_estimates():
    allowance = ProviderAllowance(
        max_calls=20,
        max_tokens=24000,
        calls_used=12,
        tokens_estimated=23000,
        stages={"coding": 8, "review": 8},
    )
    allowance.stage_used["coding"] = 6
    before = SamplingRuntime(_IdleModel())
    before.limiter.provider_allowance = allowance
    assert _selection_over_budget({}, 2000, before) == "token_allowance"
    metadata: dict = {}
    _baseline_sampling_stages(allowance, metadata)
    assert allowance.tokens_estimated == 0
    assert metadata["sampling_token_baseline"]["tokens_estimated_carried"] == 23000
    assert allowance.remaining_tokens() == 24000
    assert allowance.calls_used == 12
    assert allowance.stage_used["coding"] == 0
    after = SamplingRuntime(_IdleModel())
    after.limiter.provider_allowance = allowance
    assert _selection_over_budget({}, 2000, after) is None
    assert DEFAULT_SAMPLING_TOKEN_BUDGET == 6000


def test_saturation_request_is_packed_to_the_request_budget():
    runtime = SamplingRuntime(_IdleModel())
    runtime.limiter.provider_allowance = ProviderAllowance(max_tokens=24000, tokens_estimated=0)
    metadata = {"performance": {"sampling_token_budget": 200}}
    indicators = {
        "n_documents": 4,
        "relationships_found": 3,
        "pair_coverage": 0.01,
        "pairs_completed": 1,
        "pairs_planned": 900,
        "code_frequency_shift": 0.2,
        "window_new_code_rate": 0.1,
    }
    ledger = [{"round_index": index, "notes": "direction " * 80} for index in range(12)]
    criteria = {"criteria": [{"criterion": "new_codes", "status": "undefined", "note": "window " * 40}]}
    packed = _pack_saturation(metadata, indicators, ledger, criteria, runtime)
    assert packed["unpacked_estimate"] > 200
    assert packed["omitted_fields"]
    assert packed["omitted_fields"][0].startswith("sampling_ledger:")
    assert packed["estimate"] <= 200 or packed["reason"] == "request_budget"


def test_unlisted_sampling_directions_stay_open():
    directions = [{"direction_id": f"dir-{index:04d}"} for index in range(72)]
    state = {"directions": directions, "passages": [{"passage_id": "p1", "text": "a passage"}]}
    result = _apply_human_sampling(
        state,
        {
            "directions": [
                {"direction_id": "dir-0000", "action": "accept"},
                {"direction_id": "dir-0001", "action": "edit"},
                {"direction_id": "dir-0002", "action": "reject"},
            ]
        },
    )
    actions = [row.get("action") for row in result["directions"]]
    assert actions.count("reject") == 1
    assert actions.count("accept") == 1
    assert actions.count("edit") == 1
    assert sum(1 for action in actions if not action) == 69
    assert len(result["directions"]) == 72


def test_step_deadline_refuses_a_late_call_and_timeout_elapsed_is_recorded():
    limiter = CallLimiter(1, max_retries=0, request_timeout=5, wall_timeout=5)
    limiter.step_deadline = time.monotonic() - 0.05
    called = {"n": 0}

    async def _call(payload):
        del payload
        called["n"] += 1
        return {"ok": True}

    runnable = RunnableLambda(_call)

    async def _go():
        with pytest.raises(TimeoutError, match="step deadline"):
            await limiter.ainvoke(runnable, {"x": 1})
        limiter.step_deadline = None
        await limiter.ainvoke(runnable, {"x": 2})

    run_sync(_go())
    assert called["n"] == 1
    assert limiter.last_successful_call_at is not None

    class _Graph:
        def __init__(self):
            self.updated = None

        async def aget_state(self, config):
            del config

            class _Snap:
                values = {"metadata": {"status": "running"}, "warnings": [], "prompt_log": []}
                next = ("recode_worker",)

            return _Snap()

        async def aupdate_state(self, config, values):
            del config
            self.updated = values

    graph = _Graph()
    run_sync(
        _record_timeout(
            graph,
            {},
            TimeoutError("timed out after 120 seconds"),
            timeout_elapsed_seconds=201.0,
            timeout_limit_seconds=120.0,
        )
    )
    recorded = graph.updated["metadata"]
    assert recorded["timeout_elapsed_seconds"] == 201.0
    assert recorded["timeout_limit_seconds"] == 120.0
    assert recorded["timeout_step"] == "recode_worker"
    assert "step watchdog" in recorded["timeout_overrun_note"]


def test_absence_qualification_keeps_its_reason_through_accept():
    passage = {"passage_id": "p1", "text": "the expected outcome did not occur here"}
    quote = "the expected outcome did not occur"
    contradicted = {"relationship_id": "rel-abs", "flags": []}
    _apply_counterexample(
        contradicted,
        {"judgment": "contradicts", "quote": quote, "rationale": "one side is missing"},
        passage,
        candidate_type="a_without_b",
    )
    assert contradicted["counterexamples"][0]["judgment"] == "qualifies"
    assert contradicted["counterexamples"][0]["reclassification_reason"] == "absence_is_not_contradiction"
    assert contradicted["qualification_reason"] == "absence_is_not_contradiction"
    assert "relationship_qualified" in contradicted["flags"]
    already = {"relationship_id": "rel-qual", "flags": []}
    _apply_counterexample(
        already,
        {"judgment": "qualifies", "quote": quote, "rationale": "a condition"},
        passage,
        candidate_type="b_without_a",
    )
    assert already["qualification_reason"] == "absence_is_not_contradiction"
    assert "relationship_qualified" in already["flags"]
    plain = {"relationship_id": "rel-plain", "flags": []}
    _apply_counterexample(
        plain,
        {"judgment": "contradicts", "quote": quote, "rationale": "a denial"},
        passage,
        candidate_type="",
    )
    assert plain["counterexamples"][0]["judgment"] == "contradicts"
    assert plain["counterexamples"][0]["reclassification_reason"] == ""
    contradicted["evidence_tier"] = 2
    contradicted["evidence_label"] = "pattern_supported"
    contradicted["evidence_condition"] = "anchored"
    _tiered_accept(contradicted, field="human_disposition")
    assert contradicted["disposition"] == "accepted"
    assert contradicted["qualification_reason"] == "absence_is_not_contradiction"
    assert "relationship_qualified" in contradicted["flags"]


def test_same_passage_stem_only_stays_proposed():
    """Decision: the genuine same-passage weak-stem quote stays a tier-1 proposal.

    evidence_scope is same_passage_stem_only. The row is not rejected.
    """
    quote = "the people in my shift. management not so much"
    relationship = {
        "relationship_id": "rel-shift",
        "from_id": "c-shift",
        "to_id": "c-mgmt",
        "statement": "Shift may be related to management.",
        "passage_id": "p-shift",
        "quote": quote,
        "disposition": "proposed",
        "flags": [],
    }
    codebook = {
        "c-shift": {"code_id": "c-shift", "code": "shift", "definition": "shift"},
        "c-mgmt": {"code_id": "c-mgmt", "code": "management", "definition": "management"},
    }
    passages = {"p-shift": {"passage_id": "p-shift", "doc_id": "d1", "text": quote}}
    _annotate_endpoints(relationship, codebook, [], passages)
    assert relationship["evidence_scope"] == "same_passage_stem_only"
    assert relationship["disposition"] == "proposed"
    assert "weak_stem_anchor" in relationship["flags"]
    assert any(row.get("reason") == "stem_only" for row in relationship["endpoint_anchoring"])


def test_quiet_rounds_are_not_evaluable_while_selected_pairs_are_unsent():
    def _quiet(rows):
        report = stopping_criteria_report({"n_documents": 3}, rows)
        return next(row for row in report["criteria"] if row["criterion"] == "sampling_rounds_without_new_directions")

    unsent = {
        "analytic_new_direction_keys": [],
        "reread_unsent_pairs": [{"reason": "call_allowance", "direction_key": "gap:c1", "passage_id": "p9"}],
    }
    quiet = _quiet([dict(unsent), dict(unsent)])
    assert quiet["status"] == "not_evaluable"
    assert quiet["observed"] is None
    assert quiet["reason"] == "selected_pairs_unsent_or_unrecorded"
    missing = _quiet([{"analytic_new_direction_keys": []}, {"analytic_new_direction_keys": []}])
    assert missing["status"] == "not_evaluable"
    assert missing["observed"] is None
    met = _quiet(
        [
            {"analytic_new_direction_keys": [], "reread_unsent_pairs": []},
            {"analytic_new_direction_keys": [], "reread_unsent_pairs": []},
        ]
    )
    assert met["status"] == "met"
    closed = _quiet(
        [
            {
                "analytic_new_direction_keys": [],
                "closed_direction_keys": ["gap:c1"],
                "reread_unsent_pairs": [{"reason": "call_allowance", "direction_key": "gap:c1"}],
            },
            {
                "analytic_new_direction_keys": [],
                "closed_direction_keys": ["gap:c1"],
                "reread_unsent_pairs": [{"reason": "call_allowance", "direction_key": "gap:c1"}],
            },
        ]
    )
    assert closed["status"] == "met"


def test_memo_accounting_labels_rejected_rows_and_balances():
    source = _one_relationship()
    source["relationships"].append(
        {
            "relationship_id": "rel-0002",
            "from_id": "c-prep",
            "to_id": "c-return",
            "statement": STATEMENT,
            "passage_id": "p1",
            "quote": "Packing the suitcase took all night",
            "disposition": "rejected",
            "flags": [],
        }
    )
    result = selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        include_memos=True,
        include_storyline=False,
        interrupt=False,
    )
    accounting = result.metadata["memo_accounting"]
    assert accounting["rejected_label"] == "not_attempted_rejected"
    assert accounting["rejected_ids"] == ["rel-0002"]
    assert accounting["requested"] == (
        accounting["produced"] + accounting["omitted"] + accounting["failed"] + accounting["not_attempted_rejected"]
    )
    assert accounting["produced"] == 1
    assert accounting["not_attempted_rejected"] == 1
    assert result.metadata["stage_outcomes"]["memos"]["reason"] != "memo_empty"


def test_token_plan_uses_planned_pairs_and_names_the_guarded_figure():
    plan = _axial_token_plan(
        {"performance": {"axial_token_budget": 6000, "batch_size": 1}},
        {"max_axial_review_tokens": 24000, "propose_calls": 19, "counterexample_calls": 0},
        planned_pairs=2,
        pool_pairs=946,
    )
    assert plan["planned_pairs"] == 2
    assert plan["estimate"] == 12000
    assert plan["pool_estimate"] > plan["ceiling"]
    assert plan["warning"] == ""
    assert plan["pool_estimate_label"] == "raw unordered pool, not compared with the ceiling"
    assert plan["changes_default"] is False
    warning = _incomplete_warning(
        {
            "stop": "token_allowance",
            "focused_not_reread": ["p1"],
            "max_focused_review_tokens": 24000,
            "focused_tokens_estimated": 30000,
            "focused_tokens_estimated_compared": 20000,
        }
    )
    assert "30000" in warning
    assert "20000" in warning
    assert "not above the ceiling" in warning
    assert "exceeds the cap" not in warning


def test_synthetic_injection_without_a_relationship_is_not_run():
    class _None(FakeAxialCoder):
        def proposals(self, body):
            del body
            return AxialProposalBatch(relationships=[], nothing_to_propose=True)

    result = axial_code(
        _corpus(),
        _None(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
        axial_synthetic_counterexample=True,
    )
    synthetic = result.metadata["synthetic_counterexample"]
    assert synthetic["status"] == "not_run"
    assert synthetic["reason"] == "no_relationship_to_test"
    assert synthetic["detected"] is None


def test_gate_export_lists_proposed_rows():
    exported = export_gate_relationships(
        [
            {"relationship_id": "rel-ok", "disposition": "accepted", "flags": []},
            {"relationship_id": "rel-hold", "disposition": "provisional", "flags": ["acceptance_withheld"]},
            {"relationship_id": "rel-proposed", "disposition": "proposed", "flags": []},
            {"relationship_id": "rel-blank", "disposition": "", "flags": []},
            {"relationship_id": "rel-syn", "disposition": "accepted", "synthetic": True, "flags": []},
        ]
    )
    assert [row["relationship_id"] for row in exported["accepted"]] == ["rel-ok"]
    assert [row["relationship_id"] for row in exported["withheld"]] == ["rel-hold"]
    assert [row["relationship_id"] for row in exported["proposed"]] == ["rel-proposed", "rel-blank"]
