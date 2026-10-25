"""0.9.7 regressions for the v30-0906 live-eval findings. No network and no paid model calls."""

from __future__ import annotations

from automated_grounded_theory import __version__, axial_code, selective_code, theoretical_sample
from automated_grounded_theory.async_support import ProviderAllowance
from automated_grounded_theory.axial import (
    DEFAULT_MAX_AXIAL_REVIEW_TOKENS,
    PARADIGM_ROLES,
    StorylineBatch,
    StorylineStatementDraft,
    _apply_counterexample,
    _apply_model_relations,
    _annotate_endpoints,
    _axial_token_plan,
    _endpoint_anchoring,
    _memo_flags,
    _note_core_support,
    _note_counterexample_accounting,
    _prompt_accounting,
    _synthetic_reading,
    core_stability,
)
from automated_grounded_theory.focused import _claim_round_stage, _status_fields
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import assess_relationship, wording_overreach
from automated_grounded_theory.sampling import (
    SamplingRuntime,
    _IdleModel,
    _pack_selection,
    _store_model_selection,
    _target_key,
    _update_ledger,
    publish_sampling_indicators,
    rank_sampling_directions,
)
from automated_grounded_theory.saturation import stopping_criteria_report
from tests.test_v080 import QUOTE, FakeAxialCoder, _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs


def test_version_is_097():
    assert __version__ == "0.9.31"
    assert POLICY_VERSION == "0.9.31"


def test_consequence_wording_fails_the_tier_gate_and_the_reviewer_is_checked():
    statement = "Feeling like a number is a consequence of noting culture change post-acquisition"
    assert wording_overreach(statement, 2) is True
    assert wording_overreach("The teams met weekly.", 2, role="consequence") is True
    assert wording_overreach("The teams met weekly.", 3, role="consequence") is False
    row = {
        "relationship_id": "rel-0001",
        "statement": statement,
        "paradigm_role": "consequence",
        "disposition": "proposed",
        "evidence_tier": 2,
        "evidence_label": "pattern_supported",
        "flags": [],
    }
    _apply_model_relations(
        [row],
        {
            "relationships": [
                {
                    "relationship_id": "rel-0001",
                    "disposition": "accepted",
                    "paradigm_role": "consequence",
                    "rationale": "No causal language is used",
                }
            ]
        },
        {},
    )
    assert "wording_exceeds_tier" in row["flags"]
    assert "reviewer_causal_claim_mismatch" in row["flags"]


def test_core_stability_includes_the_current_run():
    metadata = {
        "comparison_runs": [
            {"metadata": {"model_core_id": "de9e87d96e012cb9"}},
            {"metadata": {"model_core_id": "de9e87d96e012cb9"}},
        ],
        "model_core_id": "69e316821f3fb61a",
    }
    _note_core_support({"relationships": []}, metadata, "69e316821f3fb61a")
    stability = metadata["core_stability"]
    assert stability["cores"][-1] == "69e316821f3fb61a"
    assert stability["n"] == len(stability["cores"]) == 3
    assert stability["runs"] == 3
    assert stability["score"] == 1 / 3
    assert stability["score"] < 1
    direct = core_stability(
        [
            {"metadata": {"model_core_id": "de9e87d96e012cb9"}},
            {"metadata": {"model_core_id": "de9e87d96e012cb9"}},
            {"metadata": {"model_core_id": "69e316821f3fb61a"}},
        ]
    )
    assert direct["n"] == 3
    assert direct["score"] < 1


def test_one_distinctive_stem_does_not_anchor_same_passage():
    book = {
        "rollout": {
            "code_id": "rollout",
            "code": "criticizing rollout without training",
            "definition": "train",
        },
        "growth": {
            "code_id": "growth",
            "code": "describing company as good with growing pains",
            "definition": "change",
        },
        "job": {"code_id": "job", "code": "work", "definition": "work"},
    }
    quote = "The DEI training session and the hiring change were both mentioned."
    relationship = {
        "relationship_id": "rel-stem",
        "from_id": "rollout",
        "to_id": "growth",
        "quote": quote,
        "passage_id": "p-dei",
        "flags": [],
    }
    rows = _endpoint_anchoring(relationship, book)
    assert rows[0]["anchored"] is False
    assert rows[0]["reason"] == "stem_only"
    assert rows[1]["anchored"] is False
    assert rows[1]["reason"] == "stem_only"
    _annotate_endpoints(relationship, book)
    assert "weak_stem_anchor" in relationship["flags"]
    passage = {
        "passage_id": "p-dei",
        "doc_id": "d-dei",
        "respondent_id": "r-dei",
        "text": quote,
    }
    assessed = assess_relationship(relationship, passages={"p-dei": passage}, codebook=book)
    assert assessed["cases_with_both_endpoints"] != 1
    assert assessed["within_case_strength"] != "same_passage"
    generic = _endpoint_anchoring(
        {"from_id": "job", "to_id": "job", "quote": "The work continued today."},
        book,
    )
    assert generic[0]["reason"] == "generic_stem_only"
    assert generic[0]["anchored"] is False


def test_absence_is_not_a_contradiction():
    relationship = {"relationship_id": "rel-0001", "flags": [], "counterexamples": []}
    passage = {"passage_id": "p1", "doc_id": "d1", "text": "I feel undervalued after the change."}
    _apply_counterexample(
        relationship,
        {
            "judgment": "contradicts",
            "quote": "I feel undervalued",
            "rationale": "The other endpoint is missing.",
        },
        passage,
        candidate_type="a_without_b",
    )
    case = relationship["counterexamples"][0]
    assert case["model_judgment"] == "contradicts"
    assert case["judgment"] == "qualifies"
    assert case["reclassification_reason"] == "absence_is_not_contradiction"
    assert "relationship_contradicted" not in relationship["flags"]
    assert "absence_is_not_contradiction" in relationship["flags"]
    assert relationship.get("evidence_condition") != "disputed"


def test_negative_case_row_matches_the_aggregate_and_prompts_are_accounted():
    result = axial_code(_corpus(), FakeAxialCoder(), seed=7, max_axial_review_calls=4, interrupt=False)
    relevance = result.metadata["axial_counterexample_relevance"]
    by_relationship = relevance["by_relationship"]
    remaining = 0
    saw_attempt = False
    for row in result.relationships:
        record = row.get("negative_case_search_record")
        if not isinstance(record, dict):
            continue
        stats = by_relationship[row["relationship_id"]]
        assert record["attempted"] == stats["attempted"]
        assert record["remaining"] == stats["remaining"]
        remaining += int(record["remaining"])
        saw_attempt = saw_attempt or int(record["attempted"]) > 0
    assert saw_attempt
    assert relevance["remaining_candidates"] == remaining
    account = result.metadata["prompt_accounting"]
    assert account["prompt_log_rows"] == account["provider_call_rows"] + account["unsent_rows"]
    logs = [
        {"kind": "axial_counterexample", "status": "validated", "received": True, "usage": {"total_tokens": 10}},
        {"kind": "axial_counterexample", "status": "validated", "received": True, "usage": {"total_tokens": 10}},
        {"kind": "axial_counterexample", "status": "omitted", "received": False, "usage": {"total_tokens": 4}},
    ]
    gap = {}
    _note_counterexample_accounting(
        gap,
        logs,
        [
            {"judgment": "qualifies", "synthetic": False},
            {"judgment": "irrelevant", "synthetic": True, "passage_id": "p-contrast"},
        ],
    )
    assert gap["research_judgments"] == 1
    assert gap["synthetic_judgments"] == 1
    assert gap["omitted_counterexample_prompts"] == 1
    assert gap["counterexample_prompt_rows"] == gap["counterexample_judgments_stored"] + gap["omitted_counterexample_prompts"]
    logged = _prompt_accounting(logs)
    assert logged["prompt_log_rows"] == logged["provider_call_rows"] + logged["unsent_rows"]
    assert logged["unsent_total_tokens"] == 4
    assert logged["provider_total_tokens"] == 20


def test_sampling_ceilings_ignore_upstream_calls_and_the_request_is_packed():
    source = _corpus()
    source["metadata"] = {
        "provider_budget": {
            "calls_used": 58,
            "tokens_estimated": 1000,
            "max_tokens": 80000,
            "stages": {"coding": {"calls_used": 58, "max_calls": 80}},
        }
    }
    result = theoretical_sample(
        source,
        FakeSamplingCoder(),
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        max_sampling_review_tokens=24000,
        **_kwargs(max_sampling_review_calls=3, max_decision_calls=2),
    )
    baseline = result.metadata["sampling_call_baseline"]
    assert baseline["upstream_stage_calls"]["coding"] == 58
    assert baseline["calls_used_carried"] == 58
    coding_used = result.metadata["provider_budget"]["stages"]["coding"]["calls_used"]
    assert 1 <= coding_used <= 3
    assert result.reread_notes
    assert result.metadata["provider_budget"]["calls_used"] >= 58

    preview = theoretical_sample(_corpus(), interrupt=False, **_kwargs())
    runtime = SamplingRuntime(coder=_IdleModel(), reviewer=_IdleModel(), decision_mode="llm")
    runtime.limiter.provider_allowance = ProviderAllowance(
        max_tokens=10**7,
        stages={"coding": 4, "review": 2},
    )
    fitted = _pack_selection(
        {"performance": {"sampling_token_budget": 10**7}},
        preview.directions,
        preview.passage_proposals,
        runtime,
    )
    assert fitted["directions"]
    assert not fitted["omitted"]
    budget = int(fitted["unpacked_estimate"]) - 1
    packed = _pack_selection(
        {"performance": {"sampling_token_budget": budget}},
        preview.directions,
        preview.passage_proposals,
        runtime,
    )
    assert packed["reason"] is None
    assert packed["directions"]
    assert packed["omitted"]
    sent = theoretical_sample(
        _corpus(),
        FakeSamplingCoder(),
        decision_mode="llm",
        pursuit="brief_only",
        interrupt=False,
        sampling_token_budget=budget,
        max_sampling_review_tokens=10**7,
        **_kwargs(),
    )
    pack = sent.metadata["selection_pack"]
    assert pack["omitted_direction_ids"]
    assert pack["sent_direction_ids"]
    assert pack["per_request_budget"] == budget
    assert sent.metadata.get("sampling_stop") != "request_budget"
    assert any(row.get("kind") == "sampling_selection" and row.get("received") for row in sent.prompt_log)


def test_selection_above_the_call_allowance_stops_for_that_allowance():
    runtime = SamplingRuntime(coder=_IdleModel(), reviewer=_IdleModel(), decision_mode="llm")
    runtime.limiter.provider_allowance = ProviderAllowance(
        max_tokens=100000,
        stages={"coding": 1, "review": 2},
    )
    directions = [
        {"direction_id": "dir-1-0001", "rank": 1, "kind": "thin_category", "target_id": "c-prep"},
        {"direction_id": "dir-1-0002", "rank": 2, "kind": "thin_category", "target_id": "c-return"},
        {"direction_id": "dir-1-0003", "rank": 3, "kind": "no_negative_case", "target_id": "c-prep"},
    ]
    state = {
        "directions": directions,
        "passages": [{"passage_id": "p1"}, {"passage_id": "p2"}, {"passage_id": "p3"}],
        "passage_proposals": [],
        "warnings": [],
    }
    parsed = {
        "selections": [
            {"direction_id": "dir-1-0001", "passage_id": "p1"},
            {"direction_id": "dir-1-0002", "passage_id": "p2"},
            {"direction_id": "dir-1-0003", "passage_id": "p3"},
        ],
        "direction_ids": ["dir-1-0099"],
        "passage_ids": ["p1", "p2", "p3"],
        "rationale": "Three pairs, and the flat list names a different direction.",
    }
    metadata = {"decision_mode": "llm", "procedure": {"pursuit": "reread"}}
    _store_model_selection(runtime, state, metadata, directions, parsed, [])
    assert metadata["sampling_stop"] == "call_allowance"
    assert metadata["model_sampling_status"] == "budget_limited"
    assert metadata["model_flat_list_mismatch"]["disagreed"]
    assert metadata["model_sampling_pairs"] == [
        {"direction_id": "dir-1-0001", "passage_id": "p1"},
        {"direction_id": "dir-1-0002", "passage_id": "p2"},
        {"direction_id": "dir-1-0003", "passage_id": "p3"},
    ]
    assert metadata["scheduled_reread_pairs"] == [
        {
            "direction_id": "dir-1-0001",
            "passage_id": "p1",
            "question_key": "thin_category:c-prep",
            "direction_key": "thin_category:c-prep",
        }
    ]
    assert len(metadata["sampling_selection_unsent_pairs"]) == 2
    assert {row["reason"] for row in metadata["sampling_selection_unsent_pairs"]} == {"call_allowance"}


def test_reread_notes_change_the_next_round_and_the_ledger_rate():
    indicators = {
        "n_documents": 4,
        "window_new_property_rate": None,
        "points": [
            {"new_properties": 1},
            {"new_properties": 0},
            {"new_properties": 1},
            {"new_properties": 0},
        ],
    }
    published = publish_sampling_indicators(indicators)
    assert published["window_new_property_rate"] == 0.5
    assert published["judgment"] is None
    updated = _update_ledger(
        {
            "saturation": indicators,
            "sampling_ledger": [
                {
                    "round_index": 1,
                    "new_direction_keys": ["thin_category:c-prep"],
                    "indicators": {"window_new_property_rate": None},
                }
            ],
            "reread_notes": [
                {
                    "direction_id": "dir-1-0001",
                    "passage_id": "p1",
                    "note": "This passage is irrelevant.",
                    "judgment": "irrelevant",
                }
            ],
            "directions": [
                {
                    "direction_id": "dir-1-0001",
                    "kind": "thin_category",
                    "target_id": "c-prep",
                    "action": "",
                }
            ],
            "relationships": [],
            "metadata": {"procedure": {"quiet_rounds": 2}},
        }
    )
    row = updated["sampling_ledger"][-1]
    assert row["indicators"]["window_new_property_rate"] == 0.5
    assert row["indicators"]["judgment"] is None
    assert row["reread_notes"][0]["direction_key"] == "thin_category:c-prep"
    assert row["analytic_new_direction_keys"] == ["thin_category:c-prep"]
    assert row["rejected_passages"] == [{"direction_key": "thin_category:c-prep", "passage_id": "p1"}]
    assert "thin_category:c-prep" not in row["closed_direction_keys"]
    assert updated["metadata"]["reread_notes_do_not_change_saturation_indicators"] is False
    observed = next(
        item["observed"]
        for item in updated["stopping_criteria"]["criteria"]
        if item["criterion"] == "new_properties"
    )
    assert observed == row["indicators"]["window_new_property_rate"]
    quiet = stopping_criteria_report(
        {"n_documents": 4},
        [
            {"new_direction_keys": ["thin_category:c-prep"], "analytic_new_direction_keys": [], "reread_unsent_pairs": []},
            {"new_direction_keys": ["thin_category:c-prep"], "analytic_new_direction_keys": [], "reread_unsent_pairs": []},
        ],
        quiet_rounds=2,
    )
    quiet_row = next(
        item for item in quiet["criteria"] if item["criterion"] == "sampling_rounds_without_new_directions"
    )
    assert quiet_row["status"] == "met"
    slots = [{"paradigm_role": role, "category_id": "c-prep"} for role in PARADIGM_ROLES]
    slots.append(dict(slots[0]))
    keyed = rank_sampling_directions({**_corpus(), "empty_slots": slots})
    paradigm = [_target_key(item) for item in keyed if item["kind"] == "empty_paradigm_slot"]
    assert len(paradigm) == len(set(paradigm)) == len(PARADIGM_ROLES)
    assert all(any(role in key for key in paradigm) for role in PARADIGM_ROLES)
    closed = _corpus()
    closed["metadata"] = {
        "sampling_ledger": [
            {
                "reread_notes": [
                    {
                        "direction_key": "thin_category:c-prep",
                        "judgment": "irrelevant",
                        "note": "irrelevant",
                        "passage_id": "p1",
                    }
                ]
            }
        ]
    }
    reranked = rank_sampling_directions(closed)
    assert any(_target_key(item) == "thin_category:c-prep" for item in reranked)
    follow = _corpus()
    follow["metadata"] = {
        "sampling_ledger": [
            {
                "reread_notes": [
                    {
                        "direction_key": "thin_category:c-prep",
                        "judgment": "relevant",
                        "note": "A property showed up in this passage.",
                        "passage_id": "p1",
                    }
                ]
            }
        ]
    }
    followed = rank_sampling_directions(follow)
    assert any(_target_key(item) == "thin_category:c-prep" for item in followed)
    assert all(item["kind"] != "reread_follow_up" for item in followed)
    assert updated["sampling_ledger"][-1]["indicators"]["judgment"] is None


def test_focused_claim_round_is_budget_limited_when_candidates_remain():
    stage = _claim_round_stage(
        {
            "claim_round": {
                "requested": True,
                "plausible_count": 48,
                "items_queued": 2,
                "plausible_unsent": 46,
            },
            "claim_test_relevance": {"judgments": 0, "relevant_judgments": 0},
        }
    )
    assert stage["status"] == "budget_limited"
    assert stage["reason"] == "budget"
    assert stage["plausible_unsent"] == 46
    searched = _claim_round_stage(
        {
            "claim_round": {"requested": True, "plausible_count": 0, "items_queued": 0},
            "claim_test_relevance": {"judgments": 0, "relevant_judgments": 0},
        }
    )
    assert searched["status"] == "complete_no_negative_search"


def test_memo_echo_storyline_tokens_ceiling_and_synthetic_reading():
    echoed = _memo_flags(
        "When contradiction_note is set, the memo has to account for that contradicting quote.",
        {"quote": QUOTE, "disposition": "accepted", "evidence_tier": 2, "statement": "a pattern"},
    )
    assert "memo_echoes_prompt" in echoed
    coded = axial_code(_corpus(), FakeAxialCoder(), seed=8, interrupt=False, max_axial_review_calls=2)
    coded.relationships[0]["disposition"] = "rejected"
    skipped = selective_code(coded, FakeAxialCoder(), include_memos=True, interrupt=False)
    assert "rel-0001" in skipped.metadata["memo_skipped_rejected_ids"]
    assert all(str(row.get("relationship_id") or "") != "rel-0001" for row in skipped.memos)

    class PlainStory(FakeAxialCoder):
        def storyline(self, body):
            del body
            return StorylineBatch(
                statements=[
                    StorylineStatementDraft(
                        text="Preparing to leave and returning to work were both described.",
                        category_ids=["c-prep", "c-return"],
                        relationship_ids=["rel-0001"],
                        passage_id="p1",
                        quote=QUOTE,
                    )
                ]
            )

    coded.relationships[0]["disposition"] = "accepted"
    coded.relationships[0]["evidence_tier"] = 2
    coded.relationships[0]["evidence_status"] = "pattern_supported"
    coded.relationships[0]["evidence_label"] = "pattern_supported"
    plain = selective_code(coded, PlainStory(), include_storyline=True, interrupt=False)
    assert "storyline_unaccepted_relationship" not in plain.storyline[0]["flags"]

    fields = _status_fields(
        {
            "metadata": {
                "procedure": {},
                "provider_budget": {"tokens_estimated_compared": 140000},
            },
            "focused_tokens_estimated": 158000,
            "focused_not_reread": [],
            "recode_stop": "",
        }
    )
    assert fields["focused_tokens_estimated"] == 158000
    assert fields["focused_tokens_estimated_compared"] == 140000
    assert fields["focused_tokens_estimated_is_raw"] is True
    assert fields["ceiling_comparison"] == "guarded"
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000
    plan = _axial_token_plan(
        {"performance": {"axial_token_budget": 6000}},
        {
            "max_axial_review_tokens": 150000,
            "propose_calls": 19,
            "counterexample_calls": 8,
            "memo_calls": 4,
            "storyline_calls": 1,
        },
    )
    assert plan["shortfall"] == plan["estimate"] - 150000
    assert plan["shortfall"] > 0
    assert plan["changes_default"] is False
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000
    irrelevant = _synthetic_reading("irrelevant")
    assert irrelevant["status"] == "not_interpretable"
    assert irrelevant["detected"] is None
    missing = _synthetic_reading("")
    assert missing["status"] == "not_interpretable"
    assert missing["detected"] is None
    assert missing["reason"] == "no_call"
    caught = _synthetic_reading("contradicts")
    assert caught["detected"] is True
