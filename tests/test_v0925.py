"""0.9.25 review fixes. Fakes and fixtures only. No network calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    CORE_CANDIDATE_RULE,
    DECISION_REVIEW_CHUNKING,
    ELIGIBILITY_RULE,
    INCOMPLETE_REVIEW_APPLICATION,
    LINK_DENIAL_RULE,
    PAIR_COUNT_RULE,
    PATTERN_QUOTE_RULE,
    DecisionChunkingCheckpointError,
    __version__,
    focused_code,
    read_focused_interrupt,
    selective_code,
)
from automated_grounded_theory.axial import axial_attrition
from automated_grounded_theory.decision_review import same_chunking_fingerprint
from automated_grounded_theory.graph_grounding import graph_supported
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import (
    PATTERN_MIN_CASES,
    apply_evidence_tier,
    assign_relationship_eligibility,
)
from automated_grounded_theory.sampling import reread_interpretation_report
from tests.fake_focused import FakeFocusedCoder
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v0924 import _KeepSelector, _code, _focused_source, _rewrite_metadata

FIXTURE = Path("tests/fixtures/link_denial_0925.json")


def test_version_rules_and_defaults():
    assert __version__ == "0.9.35"
    assert POLICY_VERSION == "0.9.35"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v6"
    assert INCOMPLETE_REVIEW_APPLICATION == "withhold_incomplete_v1"
    assert ELIGIBILITY_RULE == "row_gates_components_v1"
    assert LINK_DENIAL_RULE == "frequency_before_rejection_v1"
    assert PATTERN_QUOTE_RULE == "endpoint_support_in_span_v1"
    assert PAIR_COUNT_RULE == "space_and_examined_v1"
    assert CORE_CANDIDATE_RULE == "focused_categories_v1"
    assert PATTERN_MIN_CASES == 2
    assert same_chunking_fingerprint(
        {"decision_review_chunking": "request_budget_v1"},
        {"decision_review_chunking": DECISION_REVIEW_CHUNKING},
    ) is False


def _passages(quote: str) -> list[dict]:
    return [
        {
            "passage_id": "p1",
            "doc_id": "d1",
            "respondent_id": "r1",
            "text": f"The shift notes say {quote}",
        },
        {
            "passage_id": "p2",
            "doc_id": "d2",
            "respondent_id": "r2",
            "text": f"The other shift notes say {quote}",
        },
    ]


def _book() -> dict:
    return {
        "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
        "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
    }


def _assess(quote: str, *, case_quotes: list[dict] | None = None, disposition: str = "proposed") -> dict:
    row = {
        "relationship_id": "rel-0001",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": "",
        "statement": "tools for the shift may be related to coworkers on the crew",
        "disposition": disposition,
        "flags": [],
        "evidence_condition": "anchored",
        "paradigm_role": "unspecified",
        "case_quotes": case_quotes
        if case_quotes is not None
        else [
            {"passage_id": "p1", "quote": quote, "endpoint": "both"},
            {"passage_id": "p2", "quote": quote, "endpoint": "both"},
        ],
    }
    passages = {item["passage_id"]: item for item in _passages(quote)}
    apply_evidence_tier(row, passages=passages, codebook=_book(), segments=[], memos=[])
    return row


def test_denied_means_the_participant_rejects_the_link():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["link_denial_rule"] == LINK_DENIAL_RULE
    seen = []
    for pattern in fixture["patterns"]:
        seen.append(pattern["id"])
        quote = pattern["quote"]
        if pattern["id"] == "long_quote_without_negation":
            assert len(quote) == pattern["quote_characters"] == 721
            folded = f" {quote.casefold()} "
            for token in (" not ", " never ", " cannot ", " can't ", " isn't ", " aren't "):
                assert token not in folded
        row = _assess(quote)
        link = row["participant_link"]
        assert link["status"] == pattern["expected_status"]
        assert link["status"] != "denied"
        assert link["polarity"] == pattern["expected_polarity"]
        assert pattern["denied"] is False
        qualification = pattern["expected_qualification"]
        if qualification:
            assert qualification in link["qualifications"]
            assert qualification in row["flags"]
        assert "participant_denies_link" not in (row.get("flags") or [])
    assert seen == ["frequency_negation", "because_explanation", "long_quote_without_negation"]

    denied = _assess("tools for the shift do not lead to coworkers on the crew because the night ran long")
    assert denied["participant_link"]["status"] == "denied"
    assert denied["participant_link"]["polarity"] == "negative"
    assert "participant_denies_link" in denied["flags"]


def test_one_quote_for_both_endpoints_can_meet_pattern_min_cases():
    """0.9.26 reverses the 0.9.25 drop of a quotation used for both endpoints."""
    shared = "tools for the shift sit beside coworkers on the crew tonight"
    row = _assess(shared)
    assert row["shared_quote_for_both_endpoints"] is True
    assert row["shared_quote_inspect"] is True
    assert "shared_quote_inspect" in row["flags"]
    assert row["evidence_tier"] == 2
    assert row["downstream_eligible"] is True
    assert row["claims"]["pattern"]["supports_core"] is True
    assert row["claims"]["pattern"]["downstream_eligible"] is True
    assert row["core_support_component"] == "pattern"
    assert row.get("pattern_quote_rule") == PATTERN_QUOTE_RULE

    distinct = _assess(
        shared,
        case_quotes=[
            {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
            {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        ],
    )
    assert distinct["evidence_tier"] == 2
    assert distinct["downstream_eligible"] is True
    assert distinct["claims"]["pattern"]["downstream_eligible"] is True
    assert distinct["claims"]["pattern"]["supports_core"] is True
    assert distinct["core_support_component"] == "pattern"
    assert graph_supported(distinct) is True


def test_a_withheld_row_clears_a_supported_pattern_component():
    distinct = _assess(
        "tools for the shift sit beside coworkers on the crew tonight",
        case_quotes=[
            {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
            {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        ],
    )
    assert distinct["claims"]["pattern"]["supports_core"] is True
    distinct["disposition"] = "rejected"
    distinct["statement"] = "tools for the shift may be related to coworkers on the crew"
    assign_relationship_eligibility(distinct)
    assert distinct["evidence_use"] == "withheld"
    assert distinct["downstream_eligible"] is False
    assert distinct["claims"]["pattern"]["downstream_eligible"] is False
    assert distinct["claims"]["pattern"]["supports_core"] is False
    assert distinct["core_support_component"] is None
    assert distinct["claims"]["participant_explanation"]["downstream_eligible"] is False
    assert graph_supported(distinct) is False


def test_unique_pairs_is_the_examined_count_and_the_space_is_separate():
    state = {
        "relationships": [
            {
                "relationship_id": "rel-0001",
                "from_id": "c-a",
                "to_id": "c-b",
                "statement": "One examined pair.",
                "evidence_tier": 2,
                "disposition": "proposed",
            }
        ],
        "metadata": {
            "pair_coverage": {"pair_count": 62835},
            "examined_pairs": [
                {"from_id": "c-a", "to_id": "c-b"},
                {"from_id": "c-a", "to_id": "c-c"},
                {"from_id": "c-b", "to_id": "c-c"},
                {"from_id": "c-c", "to_id": "c-d"},
            ],
            "procedure": {"planned_pair_count": 4},
        },
    }
    attrition = axial_attrition(state)
    assert attrition["pairs_examined_unique"] == 4
    assert attrition["unique_pairs"] == attrition["pairs_examined_unique"]
    assert attrition["pair_space_size"] == 62835
    assert attrition["pair_space_size"] != attrition["unique_pairs"]

    missing = axial_attrition({"relationships": [], "metadata": {}})
    assert missing["unique_pairs"] is None
    assert missing["pairs_examined_unique"] is None
    assert missing["pair_space_size"] is None


def test_cost_status_follows_the_price_configuration():
    prices = {"input_usd_per_million": 3.0, "output_usd_per_million": 15.0}
    quiet = reread_interpretation_report(
        {
            "reread_notes": [],
            "prompt_log": [],
            "passages": [],
            "segments": [],
            "directions": [],
            "metadata": {"token_prices": prices},
        }
    )
    assert quiet["cost_status"] == "configured"
    assert quiet["relevant_reread_count"] == 0
    assert quiet["cost_per_relevant_reread_usd"] is None
    assert quiet["reread_expenditure_complete"] is True
    assert quiet["reread_expenditure_cost_usd"] == 0.0
    assert quiet["expenditure_cost_per_relevant_reread_usd"] is None

    unset = reread_interpretation_report(
        {
            "reread_notes": [],
            "prompt_log": [],
            "passages": [],
            "segments": [],
            "directions": [],
            "metadata": {},
        }
    )
    assert unset["cost_status"] == "unset"
    assert unset["cost_per_relevant_reread_usd"] is None


def _selective(source: dict):
    return selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        decision_mode="human",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_selective_review_calls=0,
        batch_size=1,
    )


def test_core_candidates_are_the_promoted_focused_categories():
    source = _corpus()
    source["codebook"]["c-other"] = {
        "code_id": "c-other",
        "code": "other open code",
        "definition": "not promoted",
    }
    source["raised_code_ids"] = ["c-prep"]
    source["selection_review_complete"] = True
    source["selection_status"] = "applied"
    source["selection_withheld"] = False
    limited = _selective(source)
    assert limited.metadata["core_candidate_source"] == "focused_categories"
    assert limited.metadata["procedure"]["core_candidate_rule"] == CORE_CANDIDATE_RULE
    ids = [row["code_id"] for row in limited.core_candidates]
    assert ids == ["c-prep"]
    assert "c-return" not in ids
    assert "c-other" not in ids

    direct = _selective(_corpus())
    direct_ids = {row["code_id"] for row in direct.core_candidates}
    assert direct.metadata["core_candidate_source"] == "caller_codebook"
    assert direct_ids == {"c-prep", "c-return"}

    incomplete = _corpus()
    incomplete["raised_code_ids"] = ["c-prep", "c-return"]
    incomplete["selection_review_complete"] = False
    incomplete["selection_withheld"] = True
    incomplete["selection_status"] = "provisional"
    withheld = _selective(incomplete)
    assert withheld.metadata["core_candidate_source"] == "focused_selection_incomplete"
    assert withheld.core_candidates == []
    assert withheld.model_core_id == ""
    assert withheld.metadata["core_selection_withheld"] is True
    assert withheld.metadata["core_downstream_eligible"] is False

    missing = _corpus()
    missing["metadata"] = {"stage": "focused", "selection_source": "llm"}
    unavailable = _selective(missing)
    assert unavailable.metadata["core_candidate_source"] == "focused_categories_unavailable"
    assert unavailable.core_candidates == []
    assert unavailable.model_core_id == ""
    assert unavailable.metadata["core_candidate_reason"] == "focused_categories_unavailable"


def test_an_incomplete_selection_is_withheld_and_can_resume():
    body = "packing " * 1200
    codes = [
        _code("c-b", body + " DROP"),
        _code("c-a", body + " KEEP"),
        _code("c-d", body + " DROP"),
        _code("c-c", body + " KEEP"),
    ]
    saver = InMemorySaver()
    paused = focused_code(
        _focused_source(codes),
        model=FakeFocusedCoder(),
        decision_model=_KeepSelector(),
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=True,
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_decision_calls=1,
        max_decision_tokens=200000,
        checkpointer=saver,
        thread_id="selection-0925",
    )
    assert paused.interrupted
    assert paused.raised_code_ids == []
    assert paused.metadata["selection_review_applied"] is False
    assert paused.metadata["selection_downstream_eligible"] is False
    assert paused.metadata["selection_status"] == "provisional"
    assert paused.metadata["selection_chunk_order_selects"] is False
    uncovered = set(paused.metadata["selection_uncovered_code_ids"])
    reviewed = set(paused.metadata["selection_reviewed_code_ids"])
    assert uncovered
    assert not reviewed & uncovered
    assert "c-c" in uncovered or "c-a" in uncovered
    payload = read_focused_interrupt(saver, "selection-0925")["payload"]
    assert payload["kind"] == "focused_selection_incomplete"
    assert set(payload["uncovered_code_ids"]) == uncovered

    resumed = focused_code(
        _focused_source(codes),
        model=FakeFocusedCoder(),
        decision_model=_KeepSelector(),
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=True,
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_decision_calls=8,
        max_decision_tokens=200000,
        checkpointer=saver,
        thread_id="selection-0925",
        resuming=True,
        resume={"continue_selection": True},
        allow_changes=True,
    )
    assert resumed.raised_code_ids == ["c-a", "c-c"]
    assert resumed.metadata["selection_review_applied"] is True
    assert resumed.metadata["selection_downstream_eligible"] is True
    assert resumed.metadata["selection_status"] == "applied"
    assert set(resumed.metadata["selection_uncovered_code_ids"]) == set()


def test_a_v1_chunking_checkpoint_cannot_resume():
    body = "packing " * 1200
    codes = [_code("c-a", body + " KEEP"), _code("c-b", body + " DROP")]
    saver = InMemorySaver()
    source = _focused_source(codes)
    paused = focused_code(
        source,
        model=FakeFocusedCoder(),
        decision_model=_KeepSelector(),
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=True,
        max_focused_review_calls=0,
        max_decision_calls=1,
        max_decision_tokens=200000,
        checkpointer=saver,
        thread_id="old-chunking",
    )
    assert paused.interrupted
    assert paused.raised_code_ids == []

    def downgrade(metadata):
        metadata["procedure"]["decision_review_chunking"] = "request_budget_v1"
        manifest = metadata.get("checkpoint_manifest") or {}
        procedure = dict(manifest.get("procedure") or {})
        procedure["decision_review_chunking"] = "request_budget_v1"
        manifest["procedure"] = procedure
        metadata["checkpoint_manifest"] = manifest

    _rewrite_metadata(saver, "old-chunking", downgrade)
    with pytest.raises(DecisionChunkingCheckpointError):
        focused_code(
            source,
            model=FakeFocusedCoder(),
            decision_model=_KeepSelector(),
            decision_mode="llm",
            seed=3,
            batch_size=1,
            interrupt=True,
            max_focused_review_calls=0,
            max_decision_calls=1,
            max_decision_tokens=200000,
            checkpointer=saver,
            thread_id="old-chunking",
            resuming=True,
            resume={"continue_selection": True},
            allow_changes=True,
        )
