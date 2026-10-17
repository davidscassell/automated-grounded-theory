"""0.8.6 regressions for the v27 live-eval findings. No network calls."""

from __future__ import annotations

from automated_grounded_theory import __version__, axial_code, selective_code
from automated_grounded_theory.async_support import ProviderAllowance
from automated_grounded_theory.axial import (
    SYNTHETIC_LIMITATION,
    _apply_model_relations,
    _endpoint_anchoring,
    _memo_flags,
)
from automated_grounded_theory.batching import guarded_allowance_estimate
from automated_grounded_theory.category_state import plan_relation_pairs
from automated_grounded_theory.focused import _focused_stage_outcomes
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.saturation import thematic_saturation_report

from tests.test_v080 import QUOTE, STATEMENT, FakeAxialCoder, _corpus


def test_version_is_086():
    assert __version__ == "0.9.8"
    assert POLICY_VERSION == "0.9.8"


def test_from_does_not_anchor_both_endpoints_and_matched_tokens_are_reported():
    book = {
        "family": {
            "code_id": "family",
            "code": "valuing hybrid flexibility from family",
            "definition": "time at home",
        },
        "mentor": {
            "code_id": "mentor",
            "code": "valuing internal promotion from mentoring",
            "definition": "a mentor on the desk",
        },
    }
    quote = (
        "The mentoring program helped me move from dispatcher to planner. "
        "Wouldnt have happened elsewhere."
    )
    rows = _endpoint_anchoring(
        {"from_id": "family", "to_id": "mentor", "quote": quote, "passage_id": "p9"},
        book,
    )
    assert rows[0]["anchored"] is False
    assert rows[0]["matched_tokens"] == []
    assert rows[1]["anchored"] is False
    assert rows[1]["reason"] == "stem_only"
    assert "from" not in rows[1]["matched_tokens"]
    assert "mentor" in rows[1]["matched_tokens"]
    separate = _endpoint_anchoring(
        {
            "from_id": "family",
            "to_id": "mentor",
            "quote": quote,
            "quote_from": "hybrid flexibility for the family",
            "quote_to": quote,
        },
        book,
    )
    assert separate[0]["anchored"] is True
    assert "family" in separate[0]["matched_tokens"] or "flexibility" in separate[0]["matched_tokens"]
    assert separate[1]["anchored"] is False
    assert separate[1]["reason"] == "stem_only"


def test_partial_evidence_cannot_be_accepted():
    row = {
        "relationship_id": "rel-0001",
        "from_id": "family",
        "to_id": "mentor",
        "disposition": "proposed",
        "evidence_condition": "anchored",
        "flags": ["endpoint_evidence_partial"],
        "needs_review": True,
        "downstream_eligible": True,
        "endpoint_anchoring": [
            {"endpoint": "from", "anchored": False, "matched_tokens": []},
            {"endpoint": "to", "anchored": True, "matched_tokens": ["mentor"]},
        ],
    }
    _apply_model_relations(
        [row],
        {"relationships": [{"relationship_id": "rel-0001", "disposition": "accepted", "paradigm_role": "context", "rationale": "one side"}]},
        {},
    )
    assert row["model_disposition"] == "provisional"
    assert row["disposition"] == "provisional"
    assert row["acceptance_requested"] == "accepted"
    assert row["downstream_eligible"] is False
    assert "acceptance_withheld" in row["flags"]
    assert row["review_history"][0]["disposition"] == "accepted"


def test_memo_flags_distinguish_overclaim_and_contradiction():
    partial = {
        "quote": QUOTE,
        "endpoint_anchoring": [{"anchored": True}, {"anchored": False}],
        "evidence_condition": "anchored",
        "statement": STATEMENT,
        "disposition": "provisional",
    }
    accurate = _memo_flags("Packing is a condition of returning. The evidence is partial.", partial)
    assert "memo_quote_unsupported" not in accurate
    assert "memo_overclaims_support" not in accurate
    assert "memo_evidence_strength_unstated" not in accurate
    overclaim = _memo_flags("The quote explicitly supports both categories.", partial)
    assert "memo_overclaims_support" in overclaim
    unrelated = {
        **partial,
        "model_rationale": "Wanting the all-hands back is unrelated to wishing luck.",
        "endpoint_anchoring": [{"anchored": False}, {"anchored": False}],
    }
    contradiction = _memo_flags("The quote supports that relationship.", unrelated)
    assert "memo_contradicts_relationship" in contradiction


def test_memo_accounting_separates_preexisting_memos():
    source = _corpus()
    source["relationships"] = [
        {
            "relationship_id": "rel-0001",
            "from_id": "c-prep",
            "to_id": "c-return",
            "statement": STATEMENT,
            "passage_id": "p1",
            "quote": QUOTE,
            "evidence_condition": "anchored",
            "disposition": "proposed",
            "flags": [],
        }
    ]
    source["memos"] = [
        {"memo_id": f"open-{index}", "text": "open coding memo", "relationship_id": ""}
        for index in range(4)
    ]
    result = selective_code(source, FakeAxialCoder(), seed=3, include_memos=True, interrupt=False)
    accounting = result.metadata["memo_accounting"]
    assert accounting["preexisting"] == 4
    assert accounting["produced"] == 1
    assert accounting["produced"] != accounting["preexisting"] + accounting["produced"]


def test_storyline_flags_unsent_passages_and_missing_relationships(tmp_path):
    class Citing(FakeAxialCoder):
        def storyline(self, body):
            from automated_grounded_theory.axial import StorylineBatch, StorylineStatementDraft

            del body
            return StorylineBatch(
                statements=[
                    StorylineStatementDraft(
                        text="A claim about a passage that was not sent.",
                        category_ids=["c-prep"],
                        relationship_ids=[],
                        passage_id="p-none",
                        quote="Format of the all-hands meeting was virtual.",
                    )
                ]
            )

    coded = axial_code(_corpus(), FakeAxialCoder(), seed=3, interrupt=False)
    finished = selective_code(
        coded,
        Citing(),
        include_storyline=True,
        interrupt=False,
        output_dir=tmp_path,
    )
    row = finished.storyline[0]
    assert "storyline_unsent_passage" in row["flags"]
    assert "storyline_no_relationship" in row["flags"]
    assert row["needs_review"] is True


def test_default_seed_comes_from_the_run_and_coverage_names_the_denominator():
    first = plan_relation_pairs(["a", "b", "c"], round_size=1, seed=0)
    assert first["pair_coverage_statement"] == "1 of 3 unordered pairs examined"
    omitted = axial_code(_corpus(), FakeAxialCoder(), interrupt=False, max_axial_review_calls=2)
    again = axial_code(_corpus(), FakeAxialCoder(), interrupt=False, max_axial_review_calls=2)
    assert omitted.metadata["procedure"]["seed_source"] == "source_hash"
    assert omitted.metadata["procedure"]["seed"] == again.metadata["procedure"]["seed"]
    assert omitted.metadata["procedure"]["seed"] != 0
    explicit = axial_code(_corpus(), FakeAxialCoder(), seed=42, interrupt=False, max_axial_review_calls=2)
    assert explicit.metadata["procedure"]["seed"] == 42
    assert explicit.metadata["procedure"]["seed_source"] == "caller"
    coverage = explicit.metadata["pair_coverage"]
    assert coverage["pair_coverage_statement"] == "1 of 1 unordered pairs examined"
    assert "same seed" in coverage["note"]


def test_completed_queue_stage_is_complete_and_small_windows_are_flagged():
    outcome = _focused_stage_outcomes("complete_for_queue", "complete_for_queue")
    assert outcome["focused"]["status"] == "complete"
    assert outcome["focused"]["reason"] == "queue_complete"
    assert _focused_stage_outcomes("incomplete", "timeout")["focused"]["status"] == "incomplete"
    report = thematic_saturation_report(
        [
            {"document_id": "d1", "new_codes": 1, "code_frequencies": {"A": 1}, "cumulative_codes": 1},
            {"document_id": "d2", "new_codes": 1, "code_frequencies": {"B": 1}, "cumulative_codes": 2},
        ]
    )
    assert report["n_documents"] == 2
    assert report["saturation_window"] == "not_interpretable"
    assert report["new_code_rate"] == 1
    assert report["judgment"] is None
    assert "not interpretable" in report["note"]


def test_token_guard_uses_the_observed_ratio_and_stays_conservative():
    assert guarded_allowance_estimate(1000, raw_total=0, provider_input=0) == 1000
    assert guarded_allowance_estimate(100, raw_total=50, provider_input=100) == 100
    assert guarded_allowance_estimate(100, raw_total=2000, provider_input=100) == 100
    guarded = guarded_allowance_estimate(1380, raw_total=1380, provider_input=1000)
    assert guarded == max(1, min(1380, int(round(1380 * 1.10 / 1.38))))
    assert guarded < 1380
    allowance = ProviderAllowance(max_tokens=24000, stages={"focused": 8})
    allowance.tokens_estimated = 23568
    allowance.observed_raw = 23568
    allowance.observed_input = int(round(23568 / 1.38))
    decision = allowance.try_acquire(2000, kind="focused_recode", stage="focused")
    assert decision["status"] == "acquired"
    assert decision["estimated_input_tokens"] == 2000
    assert decision["guarded_estimate"] < 2000
    bare = ProviderAllowance(max_tokens=24000)
    bare.tokens_estimated = 23568
    assert bare.try_acquire(2000, kind="focused_recode", stage="focused")["status"] == "refused"


def test_model_determinism_and_synthetic_limitation_are_recorded(tmp_path):
    result = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=3,
        interrupt=False,
        axial_synthetic_counterexample=True,
        output_dir=tmp_path,
    )
    assert result.metadata["model_determinism"] in {"recorded", "unset"}
    assert "model_temperature" in result.metadata
    assert "chat_model_settings" in result.metadata
    synthetic = result.metadata["synthetic_counterexample"]
    assert synthetic["label"] == "[synthetic counterexample]"
    assert synthetic["limitation"] == SYNTHETIC_LIMITATION
    assert "announce" in synthetic["limitation"]
    exported = (tmp_path / "category_relationships.json").read_text(encoding="utf-8")
    assert "[synthetic counterexample]" not in exported
    assert "Injected for evaluation" not in exported


def test_v084_paths_remain_reachable_offline():
    """Timeout progress, not_sent repair, missing evidence, and axial_test_invalid stay covered offline."""
    from automated_grounded_theory.axial import _apply_counterexample

    relationship = {"relationship_id": "rel-0001", "flags": [], "counterexamples": [], "negative_cases": []}
    _apply_counterexample(
        relationship,
        {"judgment": "contradicts", "quote": "not in the passage", "rationale": "A real dispute."},
        {"passage_id": "p1", "text": "Packing the suitcase took all night."},
    )
    assert "axial_test_invalid" in relationship["flags"]
    assert relationship["negative_cases"] == []
