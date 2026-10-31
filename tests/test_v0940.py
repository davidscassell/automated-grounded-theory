"""0.9.40 upstream evidence trace. Synthetic fixtures only. No network calls.

These checks are engineering and evidence-traceability coverage.
Analytic accuracy is unmeasured.
"""

from __future__ import annotations

from automated_grounded_theory.axial import (
    AxialCounterexampleBatch,
    AxialProposalBatch,
    AxialRelationshipDraft,
    _messages,
    _shrink_counterexample_payload,
    axial_code,
    build_relationship_funnel,
)
from automated_grounded_theory.batching import request_input_tokens
from automated_grounded_theory.category_state import plan_relation_pairs
from automated_grounded_theory.focused import raised_code_document_reach, reread_queue_coverage
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES, classify_pair_evidence
from automated_grounded_theory.repeatability import compare_selection_replays
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder


def test_version_is_0940_and_pattern_minimum_stays_two():
    assert __version__ == "0.9.41"
    assert POLICY_VERSION == "0.9.41"
    assert PATTERN_MIN_CASES == 2


def test_identical_request_hashes_report_the_codes_that_flipped():
    request = "a" * 64
    left = [
        {
            "checkpoint": "focused_selection",
            "prompt_hash": request,
            "response_hash": "left-response",
            "response": {"raised_code_ids": ["c1", "c2", "c4"]},
        },
        {
            "checkpoint": "focused_selection",
            "prompt_hash": "other-request",
            "response_hash": "ignored",
            "response": {"raised_code_ids": ["c9"]},
        },
    ]
    right = [
        {
            "checkpoint": "focused_selection",
            "prompt_hash": request,
            "response_hash": "right-response",
            "response": {"raised_code_ids": ["c1", "c3", "c5"]},
        }
    ]
    report = compare_selection_replays(left, right)
    assert report["requests_matched"] == 1
    assert report["requests_with_different_decisions"] == 1
    assert report["flipped_code_ids"] == ["c2", "c3", "c4", "c5"]
    assert report["matches"][0]["request_hash"] == request
    assert report["matches"][0]["response_changed"] is True
    consequences = {row["code_id"]: row for row in report["matches"][0]["analytic_consequences"]}
    assert consequences["c2"]["change"] == "raised_only_in_left"
    assert "raised set" in consequences["c2"]["consequence"]
    assert consequences["c3"]["change"] == "raised_only_in_right"
    same = compare_selection_replays(
        [{"checkpoint": "focused_selection", "prompt_hash": request, "response_hash": "h1", "response": {"raised_code_ids": ["c1"]}}],
        [{"checkpoint": "focused_selection", "prompt_hash": request, "response_hash": "h2", "response": {"raised_code_ids": ["c1"]}}],
    )
    assert same["flipped_code_ids"] == []
    assert same["matches"][0]["analytic_consequences"][0]["change"] == "response_changed_without_parsed_difference"


def test_a_54_code_plan_warns_that_pair_coverage_is_low():
    codes = {f"c{index:02d}": {"code_id": f"c{index:02d}", "code": f"code {index}", "definition": "a code"} for index in range(54)}
    source = {
        "codebook": codes,
        "passages": [{"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "All of these codes were applied here."}],
        "segments": [
            {"segment_id": f"s{index:02d}", "code_id": f"c{index:02d}", "passage_id": "p1"}
            for index in range(54)
        ],
        "relationships": [],
        "memos": [],
    }
    result = axial_code(
        source,
        FakeAxialCoder(),
        seed=1,
        decision_mode="llm",
        interrupt=False,
        max_axial_review_calls=8,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
        batch_size=1,
    )
    warning = str(result.metadata["pair_coverage"].get("low_coverage_warning") or "")
    assert warning
    assert warning in result.warnings
    assert "pairs are needed" in warning
    assert result.metadata["pair_coverage"]["pairs_needed"] > result.metadata["pair_coverage"]["pairs_planned"]
    note = result.metadata["pair_coverage"]["denominator_note"]
    assert "candidate universe" in note
    assert "not the unexamined count" in note
    assert "1,386" in note and "1,423" in note
    assert result.metadata["pair_coverage"]["candidate_universe"] >= result.metadata["pair_coverage"]["plausible_pairs"]
    assert "not the corpus's total relationship potential" in warning
    assert result.metadata["axial_token_plan"]["pairs_needed"] == result.metadata["pair_coverage"]["pairs_needed"]
    assert result.metadata["procedure"]["one_document_one_respondent"] is False


def test_multi_document_pairs_are_planned_before_single_document_pairs():
    passages = [
        {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "alpha beta"},
        {"passage_id": "p2", "doc_id": "d2", "respondent_id": "r1", "text": "alpha beta"},
        {"passage_id": "p3", "doc_id": "d3", "respondent_id": "r2", "text": "gamma delta"},
    ]
    segments = [
        {"code_id": "wide-a", "passage_id": "p1"},
        {"code_id": "wide-b", "passage_id": "p1"},
        {"code_id": "wide-a", "passage_id": "p2"},
        {"code_id": "wide-b", "passage_id": "p2"},
        {"code_id": "narrow-a", "passage_id": "p3"},
        {"code_id": "narrow-b", "passage_id": "p3"},
    ]
    ids = ["narrow-a", "narrow-b", "wide-a", "wide-b"]
    evidence = classify_pair_evidence(ids, passages, segments, pattern_min_cases=2)
    wide = next(row for row in evidence["by_class"]["same_passage"] if row["from_id"] == "wide-a")
    narrow = next(row for row in evidence["by_class"]["same_passage"] if row["from_id"] == "narrow-a")
    assert wide["document_cooccurrence"] >= 2
    assert narrow["document_cooccurrence"] == 1
    plan = plan_relation_pairs(ids, round_size=1, seed=3, evidence_classes=evidence["by_class"])
    first = plan["round"][0]
    assert {first["from_id"], first["to_id"]} == {"wide-a", "wide-b"}


def test_a_long_counterexample_fits_or_its_calls_are_returned():
    long_text = "had no issues however " * 4000
    payload = {
        "counterexamples": [
            {
                "relationship": {"relationship_id": "rel-1", "statement": "A condition.", "quote": "had no issues however"},
                "passage": {"passage_id": "p-long", "doc_id": "d9", "text": long_text},
                "complementary_passages": [],
                "respondent_id": "r9",
                "source": "passage_comparison",
                "comparison": "",
            }
        ]
    }
    raw = request_input_tokens(_messages("axial_counterexample", payload), AxialCounterexampleBatch)
    assert raw > 6000
    claim = payload["counterexamples"][0]["relationship"]["statement"]
    quote = payload["counterexamples"][0]["relationship"]["quote"]
    estimate, fitted = _shrink_counterexample_payload(payload, AxialCounterexampleBatch, 6000)
    assert fitted is True
    assert estimate <= 6000
    kept = payload["counterexamples"][0]
    assert kept["relationship"]["statement"] == claim
    assert kept["relationship"]["quote"] == quote
    assert kept["passage"]["passage_id"] == "p-long"
    assert kept["passage"]["doc_id"] == "d9"
    assert payload["counterexample_adequacy"] == "unassessed"
    assert payload["omitted_context"]["omitted"]
    assert "not evidence" in payload["omitted_context"]["note"]

    source = {
        "codebook": {
            "c1": {"code_id": "c1", "code": "tools", "definition": "tools"},
            "c2": {"code_id": "c2", "code": "bench", "definition": "bench"},
        },
        "passages": [{"passage_id": "p-long", "doc_id": "d9", "respondent_id": "r9", "text": long_text}],
        "segments": [
            {"segment_id": "s1", "code_id": "c1", "passage_id": "p-long"},
            {"segment_id": "s2", "code_id": "c2", "passage_id": "p-long"},
        ],
        "relationships": [
            {
                "relationship_id": "rel-1",
                "from_id": "c1",
                "to_id": "c2",
                "passage_id": "p-long",
                "statement": "Tools sit with the bench.",
                "quote": "had no issues however",
                "disposition": "proposed",
            }
        ],
        "memos": [],
    }
    blocked = axial_code(
        source,
        FakeAxialCoder(),
        seed=1,
        decision_mode="llm",
        interrupt=False,
        axial_synthetic_counterexample=True,
        axial_token_budget=1,
        max_axial_review_calls=8,
        max_axial_review_tokens=24000,
        batch_size=1,
    )
    reallocation = blocked.metadata.get("counterexample_reallocation") or {}
    sent = [
        row
        for row in blocked.prompt_log
        if str(row.get("checkpoint") or "") == "axial_counterexample" and str(row.get("status") or "") not in {"omitted", "deferred"}
    ]
    under_cap = any(int(row.get("estimated_input_tokens") or 0) <= 6000 for row in sent)
    assert under_cap or (
        int(reallocation.get("calls_returned") or 0) > 0 and str(reallocation.get("reason") or "")
    )


class _CopyThenParaphrase(FakeAxialCoder):
    def proposals(self, body: dict) -> AxialProposalBatch:
        pair = (body.get("pairs") or [{"from_id": "c1", "to_id": "c2"}])[0]
        quote = "the tools sat on the bench"
        statement = (
            "The tools being on the bench is a condition of the next step."
            if body.get("statement_copy_retry")
            else quote
        )
        return AxialProposalBatch(
            relationships=[
                AxialRelationshipDraft(
                    from_id=str(pair["from_id"]),
                    to_id=str(pair["to_id"]),
                    statement=statement,
                    passage_id="p1",
                    quote=quote,
                )
            ]
        )


def test_a_copied_statement_is_retried_once_and_the_paraphrase_is_kept():
    source = {
        "codebook": {
            "c1": {"code_id": "c1", "code": "tools", "definition": "tools"},
            "c2": {"code_id": "c2", "code": "bench", "definition": "bench"},
        },
        "passages": [{"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "the tools sat on the bench"}],
        "segments": [
            {"segment_id": "s1", "code_id": "c1", "passage_id": "p1"},
            {"segment_id": "s2", "code_id": "c2", "passage_id": "p1"},
        ],
        "relationships": [],
        "memos": [],
    }
    result = axial_code(
        source,
        _CopyThenParaphrase(),
        seed=1,
        decision_mode="llm",
        interrupt=False,
        max_axial_review_calls=8,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
        batch_size=1,
    )
    kept = [
        row
        for row in result.relationships
        if {row.get("from_id"), row.get("to_id")} == {"c1", "c2"}
        and "condition of the next step" in str(row.get("statement") or "")
    ]
    assert kept
    diagnostics = result.metadata["proposal_diagnostics"]
    assert diagnostics["statement_copy_retries"] >= 1
    lineage = diagnostics["statement_copy_retry_lineage"][0]
    assert lineage["retry_sent"] is True
    assert lineage["rejected_response"][0]["statement"] == "the tools sat on the bench"
    assert lineage["outcome"] == "retried"
    assert result.metadata["axial_token_plan"]["statement_copy_retry_included_in_estimate"] is True
    funnel = result.metadata["relationship_funnel"]
    assert funnel["pair_funnel"]["reconciles"] is True
    assert funnel["row_funnel"]["reconciles"] is True
    assert funnel["documents_are_not_respondents"] is True
    assert funnel["one_document_one_respondent"] is False
    direct = build_relationship_funnel({"metadata": result.metadata, "codebook": source["codebook"], "relationships": result.relationships})
    assert direct["pair_funnel"]["reconciles"] is True


def test_raised_code_reach_and_reread_queue_are_reports_only():
    passages = [
        {"passage_id": "p1", "doc_id": "d1", "respondent_id": "same", "text": "one"},
        {"passage_id": "p2", "doc_id": "d2", "respondent_id": "same", "text": "two"},
        {"passage_id": "p3", "doc_id": "d1", "respondent_id": "same", "text": "three"},
    ]
    segments = [
        {"code_id": "solo", "passage_id": "p1"},
        {"code_id": "spread", "passage_id": "p1"},
        {"code_id": "spread", "passage_id": "p2"},
    ]
    reach = raised_code_document_reach(passages, segments, ["solo", "spread"])
    assert reach["single_document_code_count"] == 1
    assert reach["raised_code_count"] == 2
    by_id = {row["code_id"]: row for row in reach["codes"]}
    assert by_id["solo"]["document_count"] == 1
    assert by_id["spread"]["document_count"] == 2
    assert reach["note"] == "1 of 2 raised codes appear in only one document."
    coverage = reread_queue_coverage(passages, [{"passage_id": "p1"}, {"passage_id": "p2"}])
    assert coverage["passages_in_corpus"] == 3
    assert coverage["passages_queued"] == 2
    assert coverage["note"] == "2 of 3 passages were queued."
    assert raised_code_document_reach(passages, segments, None)["raised_code_count"] is None


def test_exploration_share_keeps_a_non_qualifying_pair_in_the_round():
    evidence = {
        "same_passage": [
            {"from_id": "q1", "to_id": "q2", "document_cooccurrence": 3},
            {"from_id": "q1", "to_id": "q3", "document_cooccurrence": 2},
            {"from_id": "q2", "to_id": "q3", "document_cooccurrence": 2},
            {"from_id": "q1", "to_id": "q4", "document_cooccurrence": 2},
            {"from_id": "n1", "to_id": "n2", "document_cooccurrence": 1},
        ]
    }
    ids = ["q1", "q2", "q3", "q4", "n1", "n2"]
    reserved = plan_relation_pairs(ids, round_size=4, seed=0, evidence_classes=evidence, exploration_share=0.25)
    pairs = [{row["from_id"], row["to_id"]} for row in reserved["round"]]
    assert {"n1", "n2"} in pairs
    assert reserved["pair_exploration_share"] == 0.25
    assert "not a ceiling" in reserved["qualifying_pair_note"]
    closed = plan_relation_pairs(ids, round_size=4, seed=0, evidence_classes=evidence, exploration_share=0)
    closed_pairs = [{row["from_id"], row["to_id"]} for row in closed["round"]]
    assert {"n1", "n2"} not in closed_pairs


def test_unique_pairs_do_not_grow_when_the_same_request_is_repeated():
    from automated_grounded_theory.axial import _publish_exploration

    metadata = {
        "examined_pairs": [],
        "within_run_examined": [{"from_id": "a", "to_id": "b"}],
        "pair_accounting": {
            "attempted": [
                {"from_id": "a", "to_id": "b"},
                {"from_id": "b", "to_id": "a"},
            ],
            "completed": [
                {"from_id": "a", "to_id": "b"},
                {"from_id": "a", "to_id": "b"},
            ],
        },
        "pair_coverage": {},
        "unique_pairs_attempted": [{"from_id": "a", "to_id": "b"}],
        "unique_pairs_completed": [],
    }
    _publish_exploration(metadata)
    assert metadata["unique_pairs_attempted_count"] == metadata["pair_coverage"]["unique_pairs_attempted_count"] == 1
    assert metadata["unique_pairs_completed_count"] == 1
    assert metadata["pairs_attempted"] == 1
    assert metadata["pairs_completed"] == 1


def test_reallocated_calls_fit_token_and_dollar_allowances():
    from automated_grounded_theory.axial import _calls_that_fit_remaining_allowance

    calls, fit = _calls_that_fit_remaining_allowance(
        {
            "token_prices": {"input_usd_per_million": 1.0},
            "max_dollar_allowance": 0.003,
            "dollars_estimated": 0,
        },
        wanted=4,
        per_request=6000,
        tokens_used=0,
        token_ceiling=24000,
    )
    assert fit["token_calls_fit"] == 4
    assert fit["dollar_calls_fit"] == 0
    assert calls == 0
    unset, unset_fit = _calls_that_fit_remaining_allowance(
        {},
        wanted=4,
        per_request=6000,
        tokens_used=0,
        token_ceiling=24000,
    )
    assert unset == 4
    assert unset_fit["dollar_allowance"] is None


class _NoSupportedRelationship(_CopyThenParaphrase):
    def proposals(self, body: dict) -> AxialProposalBatch:
        if body.get("statement_copy_retry"):
            return AxialProposalBatch(relationships=[], nothing_to_propose=True)
        return super().proposals(body)


def test_a_retry_may_say_there_is_no_supported_relationship():
    source = {
        "codebook": {
            "c1": {"code_id": "c1", "code": "tools", "definition": "tools"},
            "c2": {"code_id": "c2", "code": "bench", "definition": "bench"},
        },
        "passages": [{"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": "the tools sat on the bench"}],
        "segments": [
            {"segment_id": "s1", "code_id": "c1", "passage_id": "p1"},
            {"segment_id": "s2", "code_id": "c2", "passage_id": "p1"},
        ],
        "relationships": [],
        "memos": [],
    }
    result = axial_code(
        source,
        _NoSupportedRelationship(),
        seed=1,
        decision_mode="llm",
        interrupt=False,
        max_axial_review_calls=8,
        max_axial_review_tokens=24000,
        axial_token_budget=6000,
        batch_size=1,
    )
    lineage = result.metadata["proposal_diagnostics"]["statement_copy_retry_lineage"][0]
    assert lineage["outcome"] == "no_supported_relationship"
    assert lineage["rejected_response"]
    assert not any({row.get("from_id"), row.get("to_id")} == {"c1", "c2"} for row in result.relationships)


def test_funnel_keeps_reviewer_rejection_apart_from_case_coverage():
    state = {
        "codebook": {"c1": {}, "c2": {}, "c3": {}},
        "relationships": [
            {
                "relationship_id": "rev",
                "from_id": "c1",
                "to_id": "c2",
                "reviewer_disposition": "rejected",
                "downstream_eligible": False,
                "document_count": 2,
                "supporting_cases": [{"case_id": "r1"}, {"case_id": "r2"}],
                "evidence_tier": 2,
            },
            {
                "relationship_id": "cases",
                "from_id": "c1",
                "to_id": "c3",
                "downstream_eligible": False,
                "document_count": 1,
                "supporting_cases": [{"case_id": "r1"}],
                "evidence_tier": 1,
            },
            {
                "relationship_id": "other",
                "from_id": "c2",
                "to_id": "c3",
                "downstream_eligible": False,
                "evidence_tier": 1,
            },
        ],
        "metadata": {"procedure": {"pattern_min_cases": 2, "one_document_one_respondent": False}, "pair_coverage": {}},
    }
    funnel = build_relationship_funnel(state)
    by_id = {row["relationship_id"]: row["reasons"] for row in funnel["rows"]}
    assert "reviewer_rejected" in by_id["rev"]
    assert "insufficient_document_count" not in by_id["rev"]
    assert "insufficient_document_count" in by_id["cases"]
    assert "incomplete_review" in by_id["cases"]
    assert "reviewer_rejected" not in by_id["cases"]
    assert "incomplete_review" in by_id["other"]
    assert "insufficient_document_count" not in by_id["other"]
    assert funnel["row_funnel"]["reconciles"] is True
