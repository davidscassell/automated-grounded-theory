"""0.9.26 review fixes. Fakes and fixtures only. No network calls."""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    DECISION_REVIEW_CHUNKING,
    LINK_DENIAL_RULE,
    OVERSIZED_RECOVERY,
    PATTERN_QUOTE_RULE,
    DecisionChunkingCheckpointError,
    __version__,
    export_gate_relationships,
    focused_code,
    read_focused_interrupt,
)
from automated_grounded_theory.decision_review import (
    CHUNKING_CHECKPOINT_MESSAGE,
    DecisionReviewer,
    SelectionDecision,
    review_procedure_fields,
    same_chunking_fingerprint,
    selection_payload,
)
from automated_grounded_theory.focused import FocusedError
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES, apply_evidence_tier
from tests.fake_focused import FakeFocusedCoder
from tests.test_v0924 import _KeepSelector, _code, _focused_source, _rewrite_metadata, _run_focused, _selection_report

SENTENCE = "tools for the shift sit beside coworkers on the crew tonight"


def test_version_fingerprint_and_defaults():
    assert __version__ == "0.9.32"
    assert POLICY_VERSION == "0.9.32"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v6"
    assert PATTERN_QUOTE_RULE == "endpoint_support_in_span_v1"
    assert LINK_DENIAL_RULE == "frequency_before_rejection_v1"
    assert OVERSIZED_RECOVERY == "compact_then_human_v1"
    assert PATTERN_MIN_CASES == 2
    current = review_procedure_fields()
    assert same_chunking_fingerprint(current, current) is True
    older = dict(current)
    older["decision_review_chunking"] = "request_budget_v2"
    older["pattern_quote_rule"] = "distinct_endpoint_quotes_v1"
    older["link_denial_rule"] = "affirmative_rejection_v1"
    older.pop("oversized_recovery")
    assert same_chunking_fingerprint(older, current) is False
    assert "decision-review" in CHUNKING_CHECKPOINT_MESSAGE
    assert "chunked" in CHUNKING_CHECKPOINT_MESSAGE


def _book() -> dict:
    return {
        "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
        "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
    }


def _passages(text: str, respondents: list[str]) -> dict:
    rows = {}
    for index, respondent in enumerate(respondents, start=1):
        passage_id = f"p{index}"
        rows[passage_id] = {
            "passage_id": passage_id,
            "doc_id": f"d{index}",
            "respondent_id": respondent,
            "text": f"The shift notes say {text}",
        }
    return rows


def _row(text: str, respondents: list[str], case_quotes: list[dict]) -> dict:
    row = {
        "relationship_id": "rel-0001",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": "",
        "statement": "tools for the shift may be related to coworkers on the crew",
        "disposition": "proposed",
        "flags": [],
        "evidence_condition": "anchored",
        "paradigm_role": "unspecified",
        "case_quotes": case_quotes,
    }
    apply_evidence_tier(row, passages=_passages(text, respondents), codebook=_book(), segments=[], memos=[])
    return row


def _both(passage_id: str, quote: str) -> dict:
    return {"passage_id": passage_id, "quote": quote, "endpoint": "both"}


def test_whole_sentence_and_split_excerpts_have_the_same_tier():
    """Quote boundaries do not change tier or eligibility. One case still counts once."""
    respondents = ["r1", "r2"]
    whole = _row(
        SENTENCE,
        respondents,
        [_both("p1", SENTENCE), _both("p2", SENTENCE)],
    )
    split = _row(
        SENTENCE,
        respondents,
        [
            {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
            {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        ],
    )
    assert whole["evidence_tier"] == split["evidence_tier"] == 2
    assert whole["downstream_eligible"] is True
    assert split["downstream_eligible"] is True
    assert whole["claims"]["pattern"]["downstream_eligible"] == split["claims"]["pattern"]["downstream_eligible"]
    assert whole["shared_quote_for_both_endpoints"] is True
    assert whole["shared_quote_inspect"] is True
    assert "shared_quote_inspect" in whole["flags"]
    assert split["shared_quote_inspect"] is False
    assert whole["cases_with_both_endpoints"] == 2
    assert split["cases_with_both_endpoints"] == 2

    overlapping = _row(
        SENTENCE,
        respondents,
        [
            {"passage_id": "p1", "quote": "tools for the shift sit beside coworkers on the crew", "endpoint": "from"},
            {"passage_id": "p1", "quote": "beside coworkers on the crew tonight", "endpoint": "to"},
            {"passage_id": "p2", "quote": "tools for the shift sit beside coworkers on the crew", "endpoint": "from"},
            {"passage_id": "p2", "quote": "beside coworkers on the crew tonight", "endpoint": "to"},
        ],
    )
    assert overlapping["evidence_tier"] == whole["evidence_tier"]
    assert overlapping["downstream_eligible"] is True
    assert overlapping["shared_quote_inspect"] is False

    one_case = _row(
        SENTENCE,
        ["r1", "r1"],
        [_both("p1", SENTENCE), _both("p2", SENTENCE)],
    )
    assert one_case["cases_with_both_endpoints"] == 1
    assert one_case["evidence_tier"] == 1
    assert one_case["downstream_eligible"] is False


def _read(quote: str) -> dict:
    row = _row(quote, ["r1", "r2"], [_both("p1", quote), _both("p2", quote)])
    link = row["participant_link"]
    assert link["reading_basis"] == "deterministic_provisional"
    assert link["reading_provisional"] is True
    return row


def test_frequency_scope_is_qualified_and_true_rejections_stay_denied():
    qualified = [
        "tools for the shift do not always lead to coworkers on the crew",
        "tools for the shift doesn't always lead to coworkers on the crew",
        "tools for the shift don't usually lead to coworkers on the crew",
        "tools for the shift isn't always related to coworkers on the crew",
        "tools for the shift do not really always lead to coworkers on the crew",
        "tools for the shift does not necessarily lead to coworkers on the crew",
        "tools for the shift rarely lead to coworkers on the crew",
        "tools for the shift sometimes lead to coworkers on the crew",
        "tools for the shift are not often beside coworkers on the crew",
        "tools for the shift are not usually beside coworkers on the crew",
    ]
    for quote in qualified:
        row = _read(quote)
        link = row["participant_link"]
        assert link["status"] == "qualified", quote
        assert link["status"] != "denied"
        assert link["polarity"] == "affirmative"
        assert link["authorizes_tier_3"] is False
        assert "frequency_qualified" in link["qualifications"]
        assert "frequency_qualified" in row["flags"]
        assert "participant_denies_link" not in row["flags"]

    denied = [
        "tools for the shift do not lead to coworkers on the crew",
        "tools for the shift do not really lead to coworkers on the crew",
        "tools for the shift have nothing to do with coworkers on the crew",
        "tools for the shift are not what causes coworkers on the crew",
    ]
    for quote in denied:
        row = _read(quote)
        link = row["participant_link"]
        assert link["status"] == "denied", quote
        assert link["polarity"] == "negative"
        assert "participant_denies_link" in row["flags"]
        assert "frequency_qualified" not in row["flags"]


def test_withheld_reason_counts_are_zero_or_null():
    empty = export_gate_relationships([])
    assert empty["withheld_reason_counts"] is None
    assert empty["inspect_flag_counts"] is None
    synthetic = export_gate_relationships(
        [{"relationship_id": "s", "disposition": "rejected", "synthetic": True, "flags": ["participant_denies_link"]}]
    )
    assert synthetic["withheld_reason_counts"] is None

    exported = export_gate_relationships(
        [
            {
                "relationship_id": "rel-denied",
                "disposition": "rejected",
                "flags": ["participant_denies_link"],
                "downstream_eligible": False,
            },
            {
                "relationship_id": "rel-shared",
                "disposition": "accepted",
                "flags": ["shared_quote_inspect"],
                "downstream_eligible": True,
            },
        ]
    )
    counts = exported["withheld_reason_counts"]
    assert counts["participant_denies_link"] == 1
    assert counts["shared_quote_inspect"] == 0
    assert counts["frequency_qualified"] == 0
    assert counts["omitted_oversized"] == 0
    assert all(isinstance(value, int) for value in counts.values())
    assert exported["inspect_flag_counts"]["shared_quote_inspect"] == 1
    assert exported["inspect_flag_counts"]["frequency_qualified"] == 0


def _tight_budget() -> int:
    huge = _code("c-huge", "packing " * 20000)
    keep = _code("c-keep", "KEEP a small code")
    source = _focused_source([huge, keep])
    score = {
        "frequency": 0,
        "frequency_component": 0.0,
        "document_share": 1.0,
        "respondent_share": 1.0,
        "document_component": 1.0,
        "substantive_memos": 0,
        "in_vivo": False,
        "model_judgment": False,
        "analytic_component": 0.0,
        "analytic_features": ["memo", "in_vivo", "comparison"],
        "score": 1.0,
        "contrast": False,
        "pin": False,
        "raised": True,
        "raise_reason": ["score"],
    }
    state = {
        "codebook": {row["code_id"]: row for row in source.codebook},
        "segments": list(source.segments),
        "metadata": dict(source.metadata),
        "proposed_raised_code_ids": ["c-huge", "c-keep"],
        "selection": [
            {"code_id": "c-keep", "code": "c-keep", **score},
            {"code_id": "c-huge", "code": "c-huge", **score},
        ],
    }
    reviewer = DecisionReviewer(_KeepSelector(), max_calls=8, max_tokens=24000, token_budget=6000)
    small = reviewer.estimate(
        selection_payload(state, ["c-keep"], chunk_index=10**6, chunk_count=10**6),
        SelectionDecision,
    )
    compacted = reviewer.estimate(
        selection_payload(
            state, ["c-huge"], chunk_index=10**6, chunk_count=10**6, compaction={"c-huge": 3},
        ),
        SelectionDecision,
    )
    budget = small + 100
    assert small < budget < compacted
    return budget


def _selection_kwargs(budget: int) -> dict:
    return {
        "model": FakeFocusedCoder(),
        "decision_model": _KeepSelector(),
        "decision_mode": "llm",
        "seed": 3,
        "batch_size": 1,
        "max_focused_review_calls": 0,
        "focused_neighbor_limit": 0,
        "focused_outside_limit": 0,
        "focused_negative_case_limit": 0,
        "max_decision_calls": 8,
        "max_decision_tokens": 400000,
        "decision_token_budget": budget,
    }


def test_llm_only_compaction_failure_stays_provisional():
    budget = _tight_budget()
    huge = _code("c-huge", "packing " * 20000)
    keep = _code("c-keep", "KEEP a small code")
    result = _run_focused([huge, keep], _KeepSelector(), decision_token_budget=budget, max_decision_calls=8, max_decision_tokens=400000)
    report = _selection_report(result)
    assert report["complete"] is False
    assert report["chunks_validated"] >= 1
    assert {"id": "c-huge", "reason": "single_code_exceeds_request_budget", "detail": "omitted_oversized"} in report["omissions"]
    assert report["human_exclusions"] == []
    assert report["stop_reasons"] == ["oversized_unrecovered"]
    assert result.raised_code_ids == []
    assert result.metadata["provisional_raised_code_ids"] == ["c-keep"]
    assert result.metadata["selection_review_applied"] is False
    assert result.metadata["decision_stop"] == "oversized_unrecovered"
    assert result.metadata["decision_stop_detail"] == "omitted_oversized"
    assert result.metadata.get("selection_human_exclusions") in ([], None) or result.metadata["selection_human_exclusions"] == []
    reviewed = set(result.metadata["selection_reviewed_code_ids"])
    assert "c-keep" in reviewed
    assert "c-huge" not in reviewed


def test_oversized_item_can_be_reviewed_or_excluded_without_dropping_finished_chunks():
    budget = _tight_budget()
    huge = _code("c-huge", "packing " * 20000)
    keep = _code("c-keep", "KEEP a small code")
    source = _focused_source([huge, keep])
    saver = InMemorySaver()
    paused = focused_code(
        source,
        checkpointer=saver,
        thread_id="oversized-0926",
        interrupt=True,
        **_selection_kwargs(budget),
    )
    assert paused.interrupted
    assert paused.raised_code_ids == []
    payload = read_focused_interrupt(saver, "oversized-0926")["payload"]
    assert payload["kind"] == "oversized_recovery"
    assert payload["unrecovered_ids"] == ["c-huge"]
    assert "c-keep" in payload["reviewed_code_ids"]

    reviewed = focused_code(
        source,
        checkpointer=saver,
        thread_id="oversized-0926",
        interrupt=True,
        resuming=True,
        resume={"review_oversized": {"id": "c-huge", "decision_token_budget": 200000}},
        allow_changes=True,
        **_selection_kwargs(budget),
    )
    report = _selection_report(reviewed)
    assert report["complete"] is True
    assert report["omissions"] == []
    assert report["human_exclusions"] == []
    assert set(reviewed.metadata["selection_reviewed_code_ids"]) == {"c-keep", "c-huge"}
    assert reviewed.raised_code_ids == ["c-keep"]
    assert reviewed.metadata["selection_review_applied"] is True
    sent = [
        row for row in reviewed.prompt_log
        if row.get("checkpoint") == "focused_selection" and row.get("status") == "validated"
    ]
    assert len(sent) == 2
    first = json.dumps(sent[0].get("messages"))
    second = json.dumps(sent[1].get("messages"))
    assert "c-keep" in first and "c-huge" not in first
    assert "c-huge" in second

    excluded_saver = InMemorySaver()
    excluded_pause = focused_code(
        source,
        checkpointer=excluded_saver,
        thread_id="exclude-0926",
        interrupt=True,
        **_selection_kwargs(budget),
    )
    assert excluded_pause.interrupted
    excluded = focused_code(
        source,
        checkpointer=excluded_saver,
        thread_id="exclude-0926",
        interrupt=True,
        resuming=True,
        resume={"exclude_item": {"id": "c-huge", "reason": "outside this review"}},
        allow_changes=True,
        **_selection_kwargs(budget),
    )
    excluded_report = _selection_report(excluded)
    assert excluded_report["complete"] is True
    assert excluded_report["omissions"] == []
    assert excluded_report["items_excluded"] == 1
    assert excluded_report["human_exclusions"] == [
        {"id": "c-huge", "reason": "outside this review", "actor": "human"}
    ]
    assert excluded.raised_code_ids == ["c-keep"]
    assert "c-huge" not in excluded.raised_code_ids
    assert "c-keep" in excluded.metadata["selection_reviewed_code_ids"]
    assert excluded.metadata["selection_items_excluded"] == 1

    left_saver = InMemorySaver()
    focused_code(
        source,
        checkpointer=left_saver,
        thread_id="leave-0926",
        interrupt=True,
        **_selection_kwargs(budget),
    )
    left = focused_code(
        source,
        checkpointer=left_saver,
        thread_id="leave-0926",
        interrupt=True,
        resuming=True,
        resume={"leave_selection": True},
        allow_changes=True,
        **_selection_kwargs(budget),
    )
    assert left.raised_code_ids == []
    assert left.metadata["selection_review_applied"] is False
    assert left.metadata["selection_human_exclusions"] == []
    assert _selection_report(left)["omissions"]
    assert "c-keep" in left.metadata["selection_reviewed_code_ids"]


def test_an_oversized_resume_rejects_a_model_exclusion_shape():
    budget = _tight_budget()
    source = _focused_source([_code("c-huge", "packing " * 20000), _code("c-keep", "KEEP a small code")])
    saver = InMemorySaver()
    focused_code(
        source,
        checkpointer=saver,
        thread_id="bad-0926",
        interrupt=True,
        **_selection_kwargs(budget),
    )
    with pytest.raises(FocusedError):
        focused_code(
            source,
            checkpointer=saver,
            thread_id="bad-0926",
            interrupt=True,
            resuming=True,
            resume={"exclude_item": {"id": "c-huge"}},
            allow_changes=True,
            **_selection_kwargs(budget),
        )


def test_a_0925_checkpoint_cannot_resume():
    body = "packing " * 1200
    source = _focused_source([_code("c-a", body + " KEEP"), _code("c-b", body + " DROP")])
    saver = InMemorySaver()
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
        thread_id="old-0926",
    )
    assert paused.interrupted

    def downgrade(metadata):
        procedure = metadata["procedure"]
        procedure["decision_review_chunking"] = "request_budget_v2"
        procedure["pattern_quote_rule"] = "distinct_endpoint_quotes_v1"
        procedure["link_denial_rule"] = "affirmative_rejection_v1"
        procedure.pop("oversized_recovery", None)
        manifest = metadata.get("checkpoint_manifest") or {}
        stored = dict(manifest.get("procedure") or {})
        stored.update(procedure)
        manifest["procedure"] = stored
        metadata["checkpoint_manifest"] = manifest

    _rewrite_metadata(saver, "old-0926", downgrade)
    with pytest.raises(DecisionChunkingCheckpointError, match="decision-review"):
        focused_code(
            source,
            model=FakeFocusedCoder(),
            decision_model=_KeepSelector(),
            decision_mode="llm",
            seed=3,
            batch_size=1,
            interrupt=True,
            max_focused_review_calls=0,
            max_decision_calls=8,
            max_decision_tokens=200000,
            checkpointer=saver,
            thread_id="old-0926",
            resuming=True,
            resume={"continue_selection": True},
            allow_changes=True,
        )
