"""0.9.30 review fixes. Fakes and synthetic fixtures only. No network calls."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import (
    CONDITIONAL_SCOPE_RULE,
    DECISION_REVIEW_CHUNKING,
    compare_pipeline_runs,
    focused_code,
)
from automated_grounded_theory.async_support import run_sync
from automated_grounded_theory.decision_review import (
    SELECTION_CONSOLIDATION,
    consolidation_allowance_reserve,
    review_procedure_fields,
    same_chunking_fingerprint,
)
from automated_grounded_theory.focused import _consolidate_selection_chunks
from automated_grounded_theory.prompts import message_text
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import apply_evidence_tier
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder
from tests.test_v0924 import _code, _focused_source, _selection_state
from tests.test_v0926 import _book, _both, _passages
from tests.test_v0928 import _ScriptedSelector

_LINK = "tools for the shift lead to coworkers on the crew"


def test_version_keeps_the_request_budget_and_replaces_the_hedge_rule():
    assert __version__ == "0.9.39"
    assert POLICY_VERSION == "0.9.39"
    assert DECISION_REVIEW_CHUNKING == "request_budget_v6"
    assert CONDITIONAL_SCOPE_RULE == "condition_scope_v1"
    current = review_procedure_fields()
    saved = dict(current)
    saved["conditional_scope_rule"] = "closed_hedge_v1"
    assert same_chunking_fingerprint(saved, current) is False


def _tier(quote: str) -> dict:
    passages = _passages(quote, ["r1", "r2"])
    row = {
        "relationship_id": "rel-cond",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": quote,
        "statement": _LINK,
        "disposition": "proposed",
        "case_quotes": [_both("p1", quote), _both("p2", quote)],
        "counterexamples": [
            {
                "passage_id": "p9",
                "quote": "the crew met without tools for the shift",
                "judgment": "contradicts",
                "valid_test": True,
            }
        ],
    }
    apply_evidence_tier(row, passages=passages, codebook=_book(), segments=[], memos=[])
    return row


def test_only_unambiguous_hedges_can_authorize_tier_3():
    clear = _tier(f"{_LINK}, if that makes sense")
    honest = _tier(f"If I'm honest, {_LINK}")
    can = _tier(f"{_LINK}, if I can")
    possible = _tier(f"{_LINK}, if possible")
    attached = _tier(f"{_LINK} if possible")
    for row in (clear, honest):
        assert row["participant_link"]["status"] == "validated"
        assert row["participant_link"]["authorizes_tier_3"] is True
        assert row["participant_link"]["reading_provisional"] is True
        assert row["evidence_tier"] == 3
    for row in (can, possible, attached):
        assert row["participant_link"]["status"] == "hypothetical"
        assert row["participant_link"]["authorizes_tier_3"] is False
        assert row["evidence_tier"] != 3


def test_failed_consolidation_stays_open_and_a_skip_does_not(tmp_path):
    state = _selection_state([_code("c-0", "KEEP"), _code("c-1", "DROP")])
    progress = {
        "raised_ids": ["c-0", "c-1"],
        "reviewed_ids": ["c-0", "c-1"],
        "chunks": [["c-0", "c-1"]],
        "compaction": {},
    }

    class _Raises:
        token_budget = 6000
        max_calls = 4
        max_tokens = 1000

        def estimate(self, payload, schema):
            return 20

        def refusal_reason(self, payload, schema, metadata):
            return None

        async def decide(self, payload, schema, metadata):
            raise RuntimeError("scripted consolidation failure")

    metadata: dict = {}
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=_Raises()), state, progress, metadata, [], [],
        )
    )
    assert metadata["selection_consolidation_status"] == "failed"
    assert "RuntimeError" in metadata["selection_consolidation_error"]
    assert metadata["selection_consolidation"] is None
    assert metadata.get("selection_consolidation_skipped") is None
    assert progress["consolidation_closed"] is False

    class _NoneParse(_Raises):
        async def decide(self, payload, schema, metadata):
            metadata["decision_stop"] = "review_error"
            metadata["decision_calls_used"] = int(metadata.get("decision_calls_used") or 0) + 1
            metadata["decision_tokens_used"] = int(metadata.get("decision_tokens_used") or 0) + 20
            return None, {"status": "error"}

    none_progress = dict(progress)
    none_meta = {"decision_calls_used": 1, "decision_tokens_used": 40}
    reviewer = _NoneParse()
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=reviewer), state, none_progress, none_meta, [], [],
        )
    )
    assert none_meta["selection_consolidation_status"] == "failed"
    assert none_meta["selection_consolidation_error"] == "review_error"
    assert none_meta["selection_consolidation"] is None
    assert none_meta.get("selection_consolidation_skipped") is None
    assert none_progress["consolidation_closed"] is False
    assert none_meta["decision_calls_used"] == 1
    assert none_meta["decision_tokens_used"] == 40
    assert reviewer.max_calls == 4
    assert reviewer.max_tokens == 1000

    class _FailConsolidation(_ScriptedSelector):
        def with_structured_output(self, schema, **kwargs):
            inner = super().with_structured_output(schema, **kwargs)

            def _call(messages):
                text = "\n".join(message_text(message) for message in messages)
                if SELECTION_CONSOLIDATION in text:
                    raise RuntimeError("scripted consolidation failure")
                return inner.invoke(messages)

            return RunnableLambda(_call)

    failed = focused_code(
        _focused_source([_code("c-0", "KEEP this code"), _code("c-1", "DROP this code")]),
        model=FakeAxialCoder(),
        decision_model=_FailConsolidation(),
        decision_mode="llm",
        seed=3,
        batch_size=1,
        interrupt=False,
        max_focused_review_calls=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_decision_tokens=200000,
        selection_consolidation=True,
        output_dir=tmp_path / "failed",
    )
    assert failed.metadata["selection_consolidation_status"] == "failed"
    assert failed.metadata["selection_consolidation_error"]
    assert failed.metadata.get("selection_consolidation") in (None, False)
    assert failed.metadata.get("selection_consolidation_skipped") is None
    assert failed.metadata["decision_chunk_state"]["consolidation_closed"] is False
    assert failed.metadata["selection_review_complete"] is False
    assert failed.metadata["selection_review_applied"] is False
    assert failed.metadata["selection_resume"]["resumable"] is True
    assert failed.raised_code_ids == []


def test_consolidation_reserve_is_separate_from_later_steps(tmp_path):
    def estimate(ids):
        names = [str(item) for item in ids]
        if len(names) <= 1:
            return 30 if names and names[0] in {"a", "c"} else 45
        if set(names) <= {"a", "c"}:
            return 40
        return 80

    reserve = consolidation_allowance_reserve(["a", "b", "c", "d"], estimate, per_request=50)
    assert reserve["consolidation_calls_reserved"] == 8
    assert reserve["consolidation_tokens_reserved"] == 300

    class _Meter:
        token_budget = 50
        max_calls = 4
        max_tokens = 200

        def estimate(self, payload, schema):
            count = len(payload.get("codes") or [])
            return 40 if count <= 1 else 90

        def refusal_reason(self, payload, schema, metadata):
            calls = int(metadata.get("decision_calls_used") or 0)
            tokens = int(metadata.get("decision_tokens_used") or 0)
            needed = self.estimate(payload, schema)
            if calls >= self.max_calls:
                return "call_allowance"
            if tokens + needed > self.max_tokens:
                return "token_allowance"
            if needed > self.token_budget:
                return "request_budget"
            return None

        async def decide(self, payload, schema, metadata):
            metadata["decision_calls_used"] = int(metadata.get("decision_calls_used") or 0) + 1
            metadata["decision_tokens_used"] = int(metadata.get("decision_tokens_used") or 0) + self.estimate(payload, schema)
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            return {"raised_code_ids": ids, "rationale": "kept"}, {"status": "validated"}

    codes = [_code(item, "KEEP this code") for item in ("c-0", "c-1", "c-2", "c-3")]
    state = _selection_state(codes)
    full = consolidation_allowance_reserve(
        ["c-0", "c-1", "c-2", "c-3"],
        lambda ids: 40 if len(list(ids)) <= 1 else 90,
        per_request=50,
    )
    assert full["consolidation_calls_reserved"] == 8
    assert full["consolidation_tokens_reserved"] == 320
    reviewer = _Meter()
    metadata = {
        "decision_calls_used": 1,
        "decision_tokens_used": 50,
        "consolidation_calls_reserved": full["consolidation_calls_reserved"],
        "consolidation_tokens_reserved": full["consolidation_tokens_reserved"],
    }
    progress = {
        "raised_ids": ["c-0", "c-1", "c-2", "c-3"],
        "reviewed_ids": ["c-0", "c-1", "c-2", "c-3"],
        "chunks": [["c-0"], ["c-1"], ["c-2"], ["c-3"]],
        "compaction": {},
    }
    run_sync(
        _consolidate_selection_chunks(
            SimpleNamespace(reviewer=reviewer), state, progress, metadata, [], [],
        )
    )
    assert metadata["consolidation_calls_charged"] == 4
    assert metadata["consolidation_tokens_charged"] == 160
    assert metadata["consolidation_calls_charged"] <= metadata["consolidation_calls_reserved"]
    assert metadata["decision_calls_used"] == 1
    assert metadata["decision_tokens_used"] == 50
    assert reviewer.max_calls == 4
    assert reviewer.max_tokens == 200
    assert reviewer.max_calls - metadata["decision_calls_used"] == 3
    assert reviewer.max_tokens - metadata["decision_tokens_used"] == 150

    def _run(place, *, consolidate: bool):
        return focused_code(
            _focused_source([
                _code("c-0", "KEEP this code"),
                _code("c-1", "DROP this code"),
                _code("c-2", "DROP this code"),
                _code("c-3", "DROP this code"),
            ]),
            model=FakeAxialCoder(),
            decision_model=_ScriptedSelector(consolidate_to=["c-0"]),
            decision_mode="llm",
            seed=3,
            batch_size=1,
            interrupt=False,
            max_focused_review_calls=0,
            focused_neighbor_limit=0,
            focused_outside_limit=0,
            focused_negative_case_limit=0,
            max_decision_calls=4,
            max_decision_tokens=200000,
            decision_token_budget=1000,
            selection_consolidation=consolidate,
            output_dir=place,
        )

    on = _run(tmp_path / "on", consolidate=True)
    off = _run(tmp_path / "off", consolidate=False)
    assert on.metadata["consolidation_reserve_separate"] is True
    assert on.metadata["consolidation_calls_reserved"] > 1
    assert on.metadata["consolidation_tokens_reserved"] > 0
    assert on.metadata["calls_available"] == off.metadata["calls_available"]
    assert on.metadata["tokens_required"] == off.metadata["tokens_required"]
    assert on.metadata["decision_call_allowance"] == off.metadata["decision_call_allowance"]
    assert on.metadata["decision_token_allowance"] == off.metadata["decision_token_allowance"]
    assert on.metadata["decision_calls_used"] == off.metadata["decision_calls_used"]
    assert on.metadata["decision_tokens_used"] == off.metadata["decision_tokens_used"]
    assert on.metadata["consolidation_calls_charged"] >= 1
    assert on.metadata["consolidation_calls_charged"] <= on.metadata["consolidation_calls_reserved"]
    assert off.metadata["consolidation_calls_reserved"] == 0
    assert off.metadata["consolidation_tokens_reserved"] == 0


def _row(statement: str, eligible: bool | None, *, to_id: str = "crew") -> dict:
    row = {"from_id": "tools", "to_id": to_id, "statement": statement}
    if eligible is not None:
        row["downstream_eligible"] = eligible
    return row


def test_eligibility_matching_ignores_row_order_and_keeps_valid_flips():
    def compared(flags):
        return compare_pipeline_runs(
            [
                {"raised_code_ids": ["c-a"], "relationships": [_row("shared", flag) for flag in flags]},
                {"raised_code_ids": ["c-a"], "relationships": [_row("shared", True)]},
            ]
        )

    forward = compared([True, False])
    reverse = compared([False, True])
    assert forward == reverse
    unknown = forward["pairwise"][0]["eligibility_changes"]
    assert unknown["matched"] == []
    assert unknown["unknown_eligibility"][0]["ambiguous"] is True
    assert unknown["unknown_eligibility"][0]["left"] is None

    agreed = compared([True, True])
    agreed_again = compared([True])
    assert agreed["pairwise"][0]["eligibility_changes"]["matched"] == []
    assert agreed["pairwise"][0]["eligibility_changes"]["unknown_eligibility"] == []
    assert agreed_again["pairwise"][0]["eligibility_changes"]["matched"] == []

    flip = compare_pipeline_runs(
        [
            {
                "raised_code_ids": ["c-a"],
                "relationships": [_row("shared", False, to_id="crew"), _row("other", None, to_id="bench")],
            },
            {
                "raised_code_ids": ["c-a"],
                "relationships": [_row("shared", True, to_id="crew"), _row("other", True, to_id="bench")],
            },
            {"raised_code_ids": ["c-a"], "relationships": []},
            {"raised_code_ids": ["c-a"], "relationships": []},
        ]
    )
    assert any(item["eligibility_changes"] is None for item in flip["pairwise"])
    summary = flip["summary"]["eligibility_changes"]
    assert summary
    assert any(item["from_id"] == "tools" and item["to_id"] == "crew" and item["left"] is False for item in summary)
    paired = next(
        item["eligibility_changes"]
        for item in flip["pairwise"]
        if item["eligibility_changes"] and item["eligibility_changes"]["unknown_eligibility"]
    )
    assert paired["matched"]
    assert paired["unknown_eligibility"]
    assert all(item in summary or item["to_id"] != "crew" for item in paired["matched"])
