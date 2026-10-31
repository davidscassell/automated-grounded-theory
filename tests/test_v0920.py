"""0.9.20 review fixes. Fakes only. No network calls."""

from __future__ import annotations

import copy

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, axial_code, selective_code
from automated_grounded_theory.axial import (
    AxialCheckpointMismatch,
    _evidence_source_hash,
    _selective_source_hash,
    _source_mapping,
)
from automated_grounded_theory.category_state import _plan_relation_pairs_0919, plan_relation_pairs
from automated_grounded_theory.graph_grounding import graph_supported
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import apply_evidence_tier
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v0919 import _rewrite_saved_source_hash, _source


def test_version_is_0920():
    assert __version__ == "0.9.40"
    assert POLICY_VERSION == "0.9.40"


def _quoted(text: str) -> dict:
    source = _source(attribution=False)
    source["passages"][0]["text"] = f"{source['passages'][0]['text']} {text}"
    source["passages"][1]["text"] = f"{source['passages'][1]['text']} {text}"
    source["relationships"][0]["case_quotes"] = [
        {"passage_id": "p1", "quote": text, "endpoint": "both"},
        {"passage_id": "p2", "quote": text, "endpoint": "both"},
        {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
    ]
    return source


def _row(source: dict) -> dict:
    result = axial_code(
        source,
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        interrupt=False,
        max_axial_review_calls=0,
        batch_size=1,
    )
    return next(row for row in result.relationships if row["relationship_id"] == "rel-0001")


def test_an_interviewer_reason_does_not_reach_tier_3():
    text = (
        "I discussed tools for the shift and coworkers on the crew "
        "because the interviewer asked about both"
    )
    row = _row(_quoted(text))
    link = row["participant_link"]
    assert row["evidence_tier"] == 2
    assert row["allowed_wording"] == "co-occurs in 2 respondents"
    assert row["attribution_source"] is None
    assert "participants attribute" not in str(row["allowed_wording"])
    assert link["status"] == "not_a_link"
    assert link["authorizes_tier_3"] is False
    assert link["validated_by"] == "deterministic_rule"
    assert link["link_text"] == text


def test_a_denial_stays_below_tier_3_and_is_flagged():
    text = "tools for the shift do not lead to coworkers on the crew because the night ran long"
    row = _row(_quoted(text))
    link = row["participant_link"]
    assert row["evidence_tier"] < 3
    assert row["attribution_source"] is None
    assert "participants attribute" not in str(row["allowed_wording"])
    assert link["polarity"] == "negative"
    assert link["status"] == "denied"
    assert link["authorizes_tier_3"] is False
    assert "participant_denies_link" in row["flags"]
    assert "participant_denies_link" in link["qualifications"]


def test_a_reversed_direction_is_not_accepted_for_the_stated_direction():
    text = "coworkers on the crew lead to tools for the shift because the night ran long"
    row = _row(_quoted(text))
    link = row["participant_link"]
    assert row["evidence_tier"] < 3
    assert row["attribution_source"] is None
    assert link["stated_direction"] == "to_from"
    assert link["status"] == "reversed"
    assert link["authorizes_tier_3"] is False
    assert "participant_direction_reversed" in row["flags"]


def test_a_hedged_quote_is_qualified_and_stays_below_tier_3():
    text = "tools for the shift might lead to coworkers on the crew because the night ran long"
    row = _row(_quoted(text))
    link = row["participant_link"]
    assert row["evidence_tier"] < 3
    assert "participants attribute" not in str(row["allowed_wording"])
    assert link["hedging"] == "hedged"
    assert link["polarity"] == "affirmative"
    assert link["stated_direction"] == "from_to"
    assert link["status"] == "hedged"
    assert link["authorizes_tier_3"] is False
    assert "participant_link_hedged" in link["qualifications"]
    assert link["validated_by"] == "deterministic_rule"


def test_a_genuine_participant_link_still_reaches_tier_3():
    text = "tools for the shift lead to coworkers on the crew because the night ran long"
    row = _row(_quoted(text))
    link = row["participant_link"]
    assert row["evidence_tier"] == 3
    assert row["allowed_wording"] == "participants attribute X to Y"
    assert row["attribution_source"] == "participant"
    assert link["status"] == "validated"
    assert link["stated_direction"] == "from_to"
    assert link["polarity"] == "affirmative"
    assert link["hedging"] == "none"
    assert link["validated_by"] == "deterministic_rule"
    assert link["authorizes_tier_3"] is True
    assert link["supports_core"] is False
    assert row["core_support_component"] == "pattern"
    assert row["claims"]["pattern"]["supports_core"] is True


def _passages_book():
    source = _source()
    return source["passages"], source["codebook"], source["segments"]


def _base_row() -> dict:
    return copy.deepcopy(_source()["relationships"][0])


def _assess(row: dict, *, memo: dict | None = None, quote: str = "") -> dict:
    passages, book, segments = _passages_book()
    if quote:
        for item in passages:
            if item["passage_id"] in {"p1", "p2"}:
                item["text"] = f"{item['text']} {quote}"
    by_id = {item["passage_id"]: item for item in passages}
    return apply_evidence_tier(
        row,
        passages=by_id,
        codebook=book,
        segments=segments,
        memos=[] if memo is None else [memo],
    )


def test_claim_flags_follow_the_current_evidence_up_and_down():
    """An analyst flag does not stick, and a later participant link does not erase the history."""
    row = _base_row()
    memo = {
        "relationship_id": "rel-0001",
        "actor": "human",
        "text": "The analyst interprets tools for the shift as related to coworkers on the crew.",
    }
    _assess(row, memo=memo)
    assert row["evidence_tier"] == 2
    assert row["downstream_eligible"] is True
    assert row["claims"]["analyst_interpretation"]["downstream_eligible"] is False
    assert row["claims"]["pattern"]["downstream_eligible"] is True
    assert row["core_support_component"] == "pattern"
    history_after_analyst = len(row.get("claim_revision_history") or [])
    _assess(row, memo=memo)
    assert len(row.get("claim_revision_history") or []) == history_after_analyst

    quote = "tools for the shift lead to coworkers on the crew because the night ran long"
    row["case_quotes"] = [
        {"passage_id": "p1", "quote": quote, "endpoint": "both"},
        {"passage_id": "p2", "quote": quote, "endpoint": "both"},
    ]
    _assess(row, memo=memo, quote=quote)
    assert row["evidence_tier"] == 3
    assert row["attribution_source"] == "participant"
    assert row["downstream_eligible"] is True
    assert row["claims"]["analyst_interpretation"]["present"] is True
    assert row["claims"]["analyst_interpretation"]["supports_core"] is False
    assert "analyst_interpretation" in row["flags"]
    assert any(item.get("evidence_tier") == 2 for item in row["claim_revision_history"])

    row["case_quotes"] = [
        {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
    ]
    _assess(row, memo=memo)
    assert row["evidence_tier"] == 2
    assert row["attribution_source"] is None
    assert row["downstream_eligible"] is True
    assert any(item.get("evidence_tier") == 3 for item in row["claim_revision_history"])

    _assess(row, memo=None)
    assert "analyst_interpretation" not in row["flags"]
    assert row["claims"]["analyst_interpretation"]["present"] is False
    assert row["downstream_eligible"] is True
    assert any(item.get("analyst_present") for item in row["claim_revision_history"])

    one_case = _base_row()
    one_case["case_quotes"] = [
        {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
    ]
    passages, book, segments = _passages_book()
    apply_evidence_tier(
        one_case,
        passages={item["passage_id"]: item for item in passages},
        codebook=book,
        segments=[item for item in segments if item["passage_id"] == "p1"],
        memos=[],
    )
    assert one_case["evidence_tier"] < 2
    assert one_case["downstream_eligible"] is False
    assert one_case["core_support_component"] is None


def test_a_0919_analyst_row_is_not_a_permanent_exclusion():
    """A stored 0.9.19 exclusion is history. The current pattern decides eligibility."""
    stale = _base_row()
    stale["evidence_tier"] = 2
    stale["attribution_source"] = "analyst_interpretation"
    stale["allowed_wording"] = "analyst interprets X as related to Y"
    stale["downstream_eligible"] = False
    stale["flags"] = ["analyst_interpretation"]
    assert "claims" not in stale
    assert graph_supported(stale) is False
    memo = {
        "relationship_id": "rel-0001",
        "actor": "human",
        "text": "The analyst interprets tools for the shift as related to coworkers on the crew.",
    }
    _assess(stale, memo=memo)
    assert stale["downstream_eligible"] is True
    assert stale["evidence_use"] == "supported_evidence"
    assert stale["claims"]["analyst_interpretation"]["present"] is True
    assert stale["claims"]["analyst_interpretation"]["supports_core"] is False
    assert stale["core_support_component"] == "pattern"
    assert graph_supported(stale) is True
    legacy = stale["claim_revision_history"][0]
    assert legacy["policy_version"] == "0.9.19"
    assert legacy["downstream_eligible"] is False
    assert "0.9.19" in legacy["note"]
    fresh = _row(_source(memo=memo))
    assert fresh["downstream_eligible"] is True
    assert fresh["core_support_component"] == "pattern"
    assert fresh["claims"]["analyst_interpretation"]["downstream_eligible"] is False


def _selective_source() -> dict:
    source = _corpus()
    source["relationships"] = [
        {
            "relationship_id": "rel-0001",
            "from_id": "c-prep",
            "to_id": "c-return",
            "statement": "preparing to leave may be related to returning to work",
            "disposition": "proposed",
            "evidence_tier": 2,
            "evidence_label": "pattern_supported",
            "flags": [],
            "paradigm_role": "unspecified",
        }
    ]
    source["memos"] = [
        {"memo_id": "memo-1", "relationship_id": "rel-0001", "actor": "human", "text": "A note about leaving."}
    ]
    source["dimensions"] = [
        {"entry_id": "dim-1", "category_id": "c-prep", "text": "pace of packing"}
    ]
    return source


def _pause_selective(source: dict, saver: InMemorySaver, thread_id: str):
    return selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        interrupt=True,
        checkpointer=saver,
        thread_id=thread_id,
        include_memos=False,
        include_storyline=False,
    )


def _resume_selective(source: dict, saver: InMemorySaver, thread_id: str, **extra):
    return selective_code(
        source,
        FakeAxialCoder(),
        seed=3,
        interrupt=True,
        checkpointer=saver,
        thread_id=thread_id,
        resuming=True,
        resume={"leave_core": True},
        include_memos=False,
        include_storyline=False,
        **extra,
    )


def test_an_unchanged_selective_source_resumes():
    source = _selective_source()
    saver = InMemorySaver()
    paused = _pause_selective(source, saver, "selective-same")
    assert paused.interrupted is True
    assert paused.interrupt_payload["kind"] == "selective_core"
    resumed = _resume_selective(source, saver, "selective-same")
    assert any(row["relationship_id"] == "rel-0001" for row in resumed.relationships)
    assert resumed.metadata["procedure"]["source_hash"] == _selective_source_hash(_source_mapping(source))


def test_changed_selective_inputs_are_rejected_on_resume():
    mutations = []

    def without_relationships(source):
        source["relationships"] = []

    def change_disposition(source):
        source["relationships"][0]["disposition"] = "rejected"
        source["relationships"][0]["human_disposition"] = "rejected"

    def change_memo(source):
        source["memos"][0]["text"] = "A different reading of leaving."

    def change_property(source):
        source["properties"][0]["property_key"] = "duration"

    def change_dimension(source):
        source["dimensions"][0]["text"] = "a different pace"

    for name, mutate in (
        ("relationships", without_relationships),
        ("disposition", change_disposition),
        ("memo", change_memo),
        ("property", change_property),
        ("dimension", change_dimension),
    ):
        source = _selective_source()
        saver = InMemorySaver()
        _pause_selective(source, saver, f"selective-{name}")
        changed = _selective_source()
        mutate(changed)
        with pytest.raises(AxialCheckpointMismatch, match="source_hash"):
            _resume_selective(changed, saver, f"selective-{name}", allow_changes=True)
        mutations.append(name)
    assert mutations == ["relationships", "disposition", "memo", "property", "dimension"]


def test_a_0919_selective_checkpoint_cannot_resume():
    source = _selective_source()
    saver = InMemorySaver()
    _pause_selective(source, saver, "selective-legacy")
    _rewrite_saved_source_hash(saver, "selective-legacy", _evidence_source_hash(_source_mapping(source)))
    with pytest.raises(AxialCheckpointMismatch, match="fingerprinted"):
        _resume_selective(source, saver, "selective-legacy", allow_changes=True)


def _exploration_evidence():
    same = [{"from_id": f"c{index}", "to_id": f"c{index + 1}"} for index in range(0, 12, 2)]
    return {
        "same_passage": same,
        "same_respondent": [{"from_id": "c0", "to_id": "c3"}],
        "lexical_cross_case": [{"from_id": "c1", "to_id": "c4"}],
    }


def test_exploration_reserves_later_classes_without_raising_the_allowance():
    ids = [f"c{index}" for index in range(8)]
    evidence = _exploration_evidence()
    plan = plan_relation_pairs(ids, round_size=4, seed=1, evidence_classes=evidence)
    assert len(plan["round"]) == 4
    bands = {row["band"] for row in plan["round"]}
    assert "same_respondent" in bands
    assert "lexical_cross_case" in bands
    assert plan["prioritized_by_class"]["same_passage"] < 4
    old = _plan_relation_pairs_0919(ids, round_size=4, seed=1, evidence_classes=evidence)
    assert {row["band"] for row in old["round"]} == {"same_passage"}
    assert len(old["round"]) == 4


def test_examined_history_is_not_revisited_and_seeds_change_coverage():
    ids = [f"c{index}" for index in range(8)]
    evidence = _exploration_evidence()
    first = plan_relation_pairs(ids, round_size=4, seed=1, evidence_classes=evidence)
    examined = [{"from_id": row["from_id"], "to_id": row["to_id"]} for row in first["round"]]
    second = plan_relation_pairs(ids, round_size=4, seed=1, evidence_classes=evidence, examined=examined)
    first_keys = {tuple(sorted((row["from_id"], row["to_id"]))) for row in first["round"]}
    second_keys = {tuple(sorted((row["from_id"], row["to_id"]))) for row in second["round"]}
    assert first_keys.isdisjoint(second_keys)
    assert second["unique_pairs_explored"] > first["within_run_pair_count"]
    selections = []
    for seed in (1, 2, 13):
        plan = plan_relation_pairs(ids, round_size=4, seed=seed, evidence_classes=evidence)
        selections.append(tuple(tuple(sorted((row["from_id"], row["to_id"]))) for row in plan["round"]))
    assert len(set(selections)) > 1


def test_seed_zero_matches_the_0919_planner_when_quotas_do_not_move_slots():
    ids = ["a", "b", "c", "d"]
    evidence = {
        "same_passage": [{"from_id": "a", "to_id": "d"}],
        "same_respondent": [{"from_id": "a", "to_id": "c"}],
        "lexical_cross_case": [{"from_id": "b", "to_id": "d"}],
    }
    current = plan_relation_pairs(ids, round_size=3, seed=1, evidence_classes=evidence)
    previous = _plan_relation_pairs_0919(ids, round_size=3, seed=1, evidence_classes=evidence)
    assert current["round"] == previous["round"]
    blind = plan_relation_pairs(ids, round_size=3, obvious_share=1, seed=0)
    blind_old = _plan_relation_pairs_0919(ids, round_size=3, obvious_share=1, seed=0)
    assert blind["round"] == blind_old["round"]
    wide = plan_relation_pairs([f"c{index}" for index in range(12)], round_size=4, seed=0)
    wide_old = _plan_relation_pairs_0919([f"c{index}" for index in range(12)], round_size=4, seed=0)
    assert wide["round"] == wide_old["round"]
    assert wide["unexamined_pair_count"] == wide_old["unexamined_pair_count"]
    large = plan_relation_pairs([f"c{index}" for index in range(40)], round_size=8, seed=1)
    assert large["pair_count"] == 780
    assert large["unexamined_materialized"] is False
    assert large["unexamined_pairs"] == []
    assert large["unexamined_pair_count"] == 780 - len(large["round"])
    assert large["selection_procedure"]["seed"] == 1


def test_axial_planning_stores_examined_pairs_and_skips_them_next_run():
    common = dict(
        seed=1,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
        decision_mode="llm",
    )
    first = axial_code(_corpus(), FakeAxialCoder(), **common)
    examined = first.metadata["examined_pairs"]
    assert examined
    assert first.metadata["unique_pairs_explored"] == len(examined)
    assert first.metadata["exploration_quotas"]["lexical_cross_case"] == 1
    second = axial_code(_corpus(), FakeAxialCoder(), examined_pairs=examined, **common)
    second_round = {
        tuple(sorted((row["from_id"], row["to_id"]))) for row in second.metadata["within_run_examined"]
    }
    already = {tuple(sorted((row["from_id"], row["to_id"]))) for row in examined}
    assert already.isdisjoint(second_round)
    assert second.metadata["unique_pairs_explored"] >= len(examined)
