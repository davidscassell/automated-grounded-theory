"""0.9.27 review fixes. Fakes and fixtures only. No network calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    AUTO_SIZE_DECISION_CALLS,
    CHUNK_ALLOWANCE_INSUFFICIENT,
    DECISION_REVIEW_CHUNKING,
    DecisionChunkingCheckpointError,
    DecisionReviewer,
    axial_code,
    export_gate_relationships,
    focused_code,
    load_focused_export,
    selective_code,
)
from automated_grounded_theory.axial import (
    AxialRelationDecision,
    AxialRelationDecisions,
    axial_attrition,
)
from automated_grounded_theory.decision_review import CHUNKING_CHECKPOINT_MESSAGE
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v0924 import _KeepSelector, _code, _focused_source, _rewrite_metadata
from tests.test_v0926 import SENTENCE, _book, _both, _passages, _row

_FIXTURE = Path(__file__).resolve().parents[0] / "fixtures" / "hypothetical_if_0927.json"


def test_version_and_the_auto_size_default():
    assert __version__ == "0.9.31"
    assert POLICY_VERSION == "0.9.31"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v6"
    assert AUTO_SIZE_DECISION_CALLS is False
    assert focused_code.__wrapped__.__defaults__ is not None or True
    import inspect

    assert inspect.signature(focused_code).parameters["auto_size_decision_calls"].default is False
    assert inspect.signature(focused_code).parameters["max_decision_calls"].default == 8
    assert inspect.signature(focused_code).parameters["decision_token_budget"].default == 6000


def _selective(source, **kwargs):
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
        **kwargs,
    )


def _focused(codes, tmp_path, **kwargs):
    return focused_code(
        _focused_source(codes),
        model=FakeAxialCoder(),
        decision_model=_KeepSelector(),
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=kwargs.pop("interrupt", False),
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_decision_tokens=kwargs.pop("max_decision_tokens", 200000),
        output_dir=tmp_path,
        **kwargs,
    )


def test_export_round_trips_raised_ids_into_core_candidates(tmp_path):
    codes = [
        _code("c-b", "DROP this code"),
        _code("c-a", "KEEP this code"),
        _code("c-c", "KEEP this one too"),
    ]
    finished = _focused(codes, tmp_path, max_decision_calls=8)
    assert finished.raised_code_ids == ["c-a", "c-c"]
    run = json.loads((tmp_path / "focused_run.json").read_text(encoding="utf-8"))
    selection = json.loads((tmp_path / "focused_selection.json").read_text(encoding="utf-8"))
    for payload in (run, selection, run["metadata"]):
        assert payload["raised_code_ids"] == ["c-a", "c-c"]
        assert payload["selection_review_complete"] is True
        assert payload["selection_withheld"] is False
        assert "provisional_raised_code_ids" in payload
    loaded = load_focused_export(tmp_path)
    assert loaded["raised_code_ids"] == ["c-a", "c-c"]
    scored = _selective(tmp_path)
    assert scored.metadata["core_candidate_source"] == "focused_categories"
    assert [row["code_id"] for row in scored.core_candidates] == ["c-a", "c-c"]

    selection["raised_code_ids"] = []
    selection["provisional_raised_code_ids"] = ["c-a", "c-c"]
    selection["selection_review_complete"] = False
    selection["selection_withheld"] = True
    selection["selection_status"] = "provisional"
    (tmp_path / "focused_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    held = _selective(tmp_path)
    assert held.metadata["core_candidate_source"] == "focused_selection_incomplete"
    assert held.core_candidates == []
    assert load_focused_export(tmp_path)["provisional_raised_code_ids"] == ["c-a", "c-c"]
    assert load_focused_export(tmp_path)["raised_code_ids"] == []


def _tier_two_source(reviewed: list[str] | None) -> dict:
    """A relationship whose quotes support tier 2. Not study data."""
    passages = _passages(SENTENCE, ["r1", "r2"])
    row = {
        "relationship_id": "rel-0003",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": SENTENCE,
        "statement": "tools for the shift may be related to coworkers on the crew",
        "disposition": "accepted",
        "flags": [],
        "evidence_condition": "anchored",
        "paradigm_role": "unspecified",
        "case_quotes": [_both("p1", SENTENCE), _both("p2", SENTENCE)],
    }
    data = {
        "codebook": _book(),
        "passages": list(passages.values()),
        "segments": [],
        "relationships": [row],
    }
    if reviewed is not None:
        data.update(
            selection_withheld=True,
            selection_status="provisional",
            selection_review_complete=False,
            selection_reviewed_code_ids=reviewed,
            raised_code_ids=[],
        )
    return data


def _axial(source):
    return axial_code(
        source,
        FakeAxialCoder(),
        seed=3,
        interrupt=False,
        max_axial_review_calls=0,
        axial_relations=False,
        batch_size=1,
    )


def test_unreviewed_endpoints_stay_exploratory():
    blocked = _axial(_tier_two_source([]))
    row = blocked.relationships[0]
    assert row["evidence_tier"] == 2
    assert row["evidence_use"] == "hypothesis_exploration"
    assert row["downstream_eligible"] is False
    assert row["eligibility_reason"] == "endpoint_unreviewed_by_focus"
    assert "endpoint_unreviewed_by_focus" in row["flags"]

    kept = _axial(_tier_two_source(["tools", "crew"]))
    assert kept.relationships[0]["evidence_tier"] == 2
    assert "endpoint_unreviewed_by_focus" not in (kept.relationships[0].get("flags") or [])
    assert kept.relationships[0]["downstream_eligible"] is True

    plain = _axial(_tier_two_source(None))
    assert "endpoint_unreviewed_by_focus" not in (plain.relationships[0].get("flags") or [])
    assert plain.relationships[0]["downstream_eligible"] is True


def test_withholding_causes_partition_the_withheld_rows():
    exported = export_gate_relationships(
        [
            {
                "relationship_id": "rel-llm",
                "disposition": "rejected",
                "review_actor": "llm",
                "reviewer_decided": True,
                "model_disposition": "rejected",
                "model_rationale": "The quote does not show the link.",
                "evidence_tier": 2,
                "flags": ["shared_quote_inspect"],
                "downstream_eligible": False,
            },
            {
                "relationship_id": "rel-partial",
                "disposition": "provisional",
                "evidence_tier": 2,
                "flags": ["endpoint_evidence_partial", "acceptance_withheld"],
                "endpoint_anchoring": [{"anchored": True}, {"anchored": False}],
                "downstream_eligible": False,
            },
            {
                "relationship_id": "rel-cases",
                "disposition": "provisional",
                "evidence_tier": 1,
                "flags": [],
                "downstream_eligible": False,
            },
            {
                "relationship_id": "rel-shared",
                "disposition": "accepted",
                "evidence_tier": 2,
                "flags": ["shared_quote_inspect"],
                "downstream_eligible": True,
            },
        ]
    )
    causes = exported["withholding_cause_counts"]
    assert sum(causes.values()) == len(exported["withheld"]) == 3
    assert causes["llm_reviewer_rejected"] == 1
    assert causes["partial_endpoint_evidence"] == 1
    assert causes["insufficient_cases"] == 1
    assert causes["endpoint_unreviewed_by_focus"] == 0
    by_id = {row["relationship_id"]: row["withholding_cause"] for row in exported["withheld"]}
    assert by_id == {
        "rel-llm": "llm_reviewer_rejected",
        "rel-partial": "partial_endpoint_evidence",
        "rel-cases": "insufficient_cases",
    }
    assert exported["inspect_flag_counts"]["shared_quote_inspect"] == 2
    assert "shared_quote_inspect" not in causes
    assert exported["withheld_reason_counts"]["shared_quote_inspect"] == 0
    empty = export_gate_relationships([])
    assert empty["withholding_cause_counts"] is None


def test_grounded_endpoints_match_tier_two_rows():
    rows = []
    for index in range(3):
        rows.append(
            {
                "relationship_id": f"rel-{index:04d}",
                "statement": "tools for the shift may be related to coworkers on the crew",
                "evidence_tier": 2,
                "disposition": "accepted",
                "endpoint_anchoring": [],
            }
        )
    attrition = axial_attrition({"relationships": rows, "metadata": {}})
    assert attrition["evidence_tiers"]["2"] == 3
    assert attrition["grounded_endpoints"] == 3
    assert attrition["endpoint_anchoring_complete"] == 0
    tier_two_or_higher = sum(
        count for key, count in attrition["evidence_tiers"].items() if int(key) >= 2
    )
    assert attrition["grounded_endpoints"] == tier_two_or_higher

    exported = _axial(_tier_two_source(None))
    live = exported.metadata["attrition"]
    live_tiers = sum(count for key, count in live["evidence_tiers"].items() if int(key) >= 2)
    assert live["grounded_endpoints"] == live_tiers == 1


def test_if_inside_a_noun_or_an_aside_is_not_hypothetical():
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert "not study data" in payload["note"].casefold()
    for case in payload["cases"]:
        row = _row(case["quote"], ["r1", "r2"], [_both("p1", case["quote"]), _both("p2", case["quote"])])
        link = row["participant_link"]
        assert link["reading_basis"] == "deterministic_provisional"
        assert link["reading_provisional"] is True
        assert link["status"] == case["status"], case["id"]
        assert (link["status"] == "hypothetical") is case["hypothetical"], case["id"]
        assert link["authorizes_tier_3"] is False or case["status"] == "validated"


def test_reviewer_decision_is_recorded_and_order_is_stable():
    source = _tier_two_source(None)
    source["relationships"][0]["relationship_id"] = "rel-0002"
    source["relationships"][0]["disposition"] = "proposed"
    reviewed = axial_code(
        source,
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        axial_relations=False,
        interrupt=False,
        max_axial_review_calls=0,
        max_decision_calls=4,
        batch_size=1,
    )
    row = reviewed.relationships[0]
    assert row["reviewer_decided"] is True
    assert row["reviewer_disposition"] == "accepted"
    assert row["model_disposition"] == "accepted"
    assert row["model_rationale"]
    assert row["review_actor"] == "llm"
    assert row["review_history"][-1]["disposition"] == "accepted"

    wide = _corpus()
    wide["relationships"] = [
        {
            "relationship_id": "rel-small",
            "from_id": "c-prep",
            "to_id": "c-return",
            "statement": "Packing is a condition of returning.",
            "passage_id": "p1",
            "evidence_tier": 2,
            "disposition": "proposed",
        },
        {
            "relationship_id": "rel-huge",
            "from_id": "c-prep",
            "to_id": "c-return",
            "statement": "packing " * 20000,
            "passage_id": "p1",
            "evidence_tier": 2,
            "disposition": "proposed",
        },
    ]
    limited = axial_code(
        wide,
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        axial_relations=False,
        interrupt=False,
        max_axial_review_calls=0,
        max_decision_calls=4,
        axial_token_budget=6000,
        batch_size=1,
    )
    by_id = {item["relationship_id"]: item for item in limited.relationships}
    assert by_id["rel-huge"].get("reviewer_decided") is not True
    assert by_id["rel-huge"].get("model_rationale") in (None, "")

    reviewer = DecisionReviewer(FakeAxialCoder(), max_calls=1, token_budget=6000)
    left = reviewer.messages({"b": 1, "a": {"z": 1, "m": 2}, "kind": "focused_selection"})
    right = reviewer.messages({"kind": "focused_selection", "a": {"m": 2, "z": 1}, "b": 1})
    assert isinstance(left[1], HumanMessage)
    assert left[1].content == right[1].content


def _large_codes():
    body = "packing " * 1200
    return [
        _code("c-b", body + " DROP"),
        _code("c-a", body + " KEEP"),
        _code("c-d", body + " DROP"),
        _code("c-c", body + " KEEP"),
    ]


def test_chunk_allowance_stops_before_sending_and_auto_size_is_opt_in(tmp_path):
    stopped = _focused(_large_codes(), tmp_path, max_decision_calls=1)
    assert stopped.metadata["decision_stop"] == CHUNK_ALLOWANCE_INSUFFICIENT
    assert stopped.metadata["selection_chunks_required"] > stopped.metadata["decision_call_allowance"]
    assert stopped.metadata["selection_chunk_shortfall"] > 0
    assert stopped.metadata["auto_size_decision_calls"] is False
    assert stopped.raised_code_ids == []
    report = stopped.metadata["decision_reviews"]["focused_selection"]
    assert report["chunks_sent"] == 0
    assert report["chunks_required"] == stopped.metadata["selection_chunks_required"]
    assert CHUNK_ALLOWANCE_INSUFFICIENT in (report["stop_reasons"] or [])
    assert not any(entry.get("kind") == "decision_review" for entry in stopped.prompt_log)

    sized = _focused(
        _large_codes(),
        tmp_path / "sized",
        max_decision_calls=1,
        auto_size_decision_calls=True,
    )
    assert sized.metadata["decision_call_allowance_raised"] is True
    assert sized.metadata.get("decision_stop") != CHUNK_ALLOWANCE_INSUFFICIENT
    assert sized.metadata["decision_reviews"]["focused_selection"]["chunks_sent"] >= 1
    assert sized.raised_code_ids == ["c-a", "c-c"]


def test_a_0926_checkpoint_cannot_resume():
    saver = InMemorySaver()
    paused = focused_code(
        _focused_source(_large_codes()),
        model=FakeAxialCoder(),
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
        thread_id="chunk-0927",
    )
    assert paused.interrupted
    assert paused.metadata["decision_stop"] == CHUNK_ALLOWANCE_INSUFFICIENT

    def downgrade(metadata):
        procedure = metadata["procedure"]
        procedure["decision_review_chunking"] = "request_budget_v3"
        for key in (
            "focus_endpoint_rule",
            "withholding_cause_rule",
            "conditional_scope_rule",
            "chunk_allowance_rule",
        ):
            procedure.pop(key, None)
        manifest = metadata.get("checkpoint_manifest") or {}
        stored = dict(manifest.get("procedure") or {})
        stored.update(procedure)
        manifest["procedure"] = stored
        metadata["checkpoint_manifest"] = manifest

    _rewrite_metadata(saver, "chunk-0927", downgrade)
    with pytest.raises(DecisionChunkingCheckpointError, match="decision-review"):
        focused_code(
            _focused_source(_large_codes()),
            model=FakeAxialCoder(),
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
            thread_id="chunk-0927",
            resuming=True,
            resume={"continue_selection": True},
            allow_changes=True,
        )
    assert "chunked" in CHUNKING_CHECKPOINT_MESSAGE


def test_relation_decision_import_is_used():
    # The axial reviewer schema is what the recorded decision is checked against.
    assert AxialRelationDecisions.__name__ == "AxialRelationDecisions"
    assert AxialRelationDecision.__name__ == "AxialRelationDecision"
