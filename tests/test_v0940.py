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
    assert __version__ == "0.9.40"
    assert POLICY_VERSION == "0.9.40"
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
    assert report["matches"][0]["left_response_hash"] != report["matches"][0]["right_response_hash"]


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
    estimate, fitted = _shrink_counterexample_payload(payload, AxialCounterexampleBatch, 6000)
    assert fitted is True
    assert estimate <= 6000

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
    assert result.metadata["proposal_diagnostics"]["statement_copy_retries"] >= 1
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
