"""0.9.31 review fixes. Fakes and synthetic fixtures only. No network calls.

These checks are processing coverage. They are not selection validity.
Human-adjudicated examples are not in this package.
"""

from __future__ import annotations

from types import SimpleNamespace

from automated_grounded_theory.async_support import run_sync
from automated_grounded_theory.focused import _consolidate_selection_chunks, _selection_export
from automated_grounded_theory.relationship_tiers import apply_evidence_tier
from tests.test_v0924 import _code, _selection_state
from tests.test_v0926 import _book, _both, _passages

_LINK = "tools for the shift lead to coworkers on the crew"


def _tier(quote: str, statement: str | None = None) -> dict:
    passages = _passages(quote, ["r1", "r2"])
    row = {
        "relationship_id": "rel-cond",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": quote,
        "statement": _LINK if statement is None else statement,
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


def _ids(candidates) -> list[str]:
    return [row["code_id"] for row in candidates]


def _outcome(candidates, code_id: str) -> dict:
    matches = [row for row in candidates if row["code_id"] == code_id]
    assert len(matches) == 1, code_id
    return matches[0]


def test_condition_markers_stay_qualified_and_the_statement_must_keep_them():
    paraphrases = [
        f"{_LINK} if it rains",
        f"{_LINK} unless it rains",
        f"{_LINK} provided that it rains",
        f"The reason {_LINK} if it rains is simple",
        f"Part of why {_LINK} if the night is long",
        f"{_LINK} providing that it rains",
        f"{_LINK} as long as it rains",
        f"{_LINK} so long as it rains",
        f"{_LINK} only when it rains",
        f"{_LINK} in case it rains",
        f"{_LINK} assuming it rains",
        f"{_LINK} on condition that it rains",
        f"{_LINK} except when it rains",
        f"{_LINK} except if it rains",
    ]
    for quote in paraphrases:
        row = _tier(quote)
        link = row["participant_link"]
        assert row["evidence_tier"] != 3, quote
        assert link["status"] == "hypothetical", quote
        assert link["authorizes_tier_3"] is False, quote
        assert link["condition"], quote
        assert "participant_link_conditional" in link["qualifications"], quote
        assert row["condition"] == link["condition"], quote
        assert row["condition_retained"] is False, quote
        assert "condition_not_in_statement" in link["qualifications"], quote

    reason = _tier(f"The reason {_LINK} if it rains is simple")
    assert reason["condition"] == "if it rains"
    unless = _tier(f"{_LINK} unless it rains")
    assert unless["condition"] == "unless it rains"
    provided = _tier(f"{_LINK} provided that it rains")
    assert provided["condition"] == "provided that it rains"

    retained = _tier(f"{_LINK} if it rains", statement=f"{_LINK} if it rains")
    assert retained["condition_retained"] is True
    assert retained["evidence_tier"] != 3
    assert retained["participant_link"]["authorizes_tier_3"] is False
    assert "condition_not_in_statement" not in retained["participant_link"]["qualifications"]

    embedded = _tier(f"The question of if {_LINK}")
    assert embedded["evidence_tier"] == 3
    assert embedded["participant_link"]["status"] == "validated"
    assert embedded["participant_link"]["authorizes_tier_3"] is True
    assert embedded["condition"] is None

    aside = _tier(f"{_LINK}, if that makes sense")
    assert aside["evidence_tier"] == 3
    assert aside["condition"] is None


def _run(reviewer, progress, metadata, state):
    warnings: list[str] = []
    run_sync(_consolidate_selection_chunks(
        SimpleNamespace(reviewer=reviewer), state, progress, metadata, [], warnings,
    ))
    return warnings


def test_an_oversized_candidate_cannot_disappear():
    class _Reviewer:
        token_budget = 50
        max_calls = 8
        max_tokens = 1000
        payloads: list[list[str]] = []

        def estimate(self, payload, schema):
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            if "big" in ids:
                return 80
            return 20

        def refusal_reason(self, payload, schema, metadata):
            if self.estimate(payload, schema) > self.token_budget:
                return "request_budget"
            return None

        async def decide(self, payload, schema, metadata):
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            self.payloads.append(ids)
            return {"raised_code_ids": ids, "rationale": "kept the request"}, {"status": "validated"}

    state = _selection_state([_code("small", "KEEP"), _code("big", "KEEP")])
    progress = {
        "raised_ids": ["small", "big"],
        "reviewed_ids": ["small", "big"],
        "chunks": [["small"], ["big"]],
        "compaction": {},
    }
    metadata: dict = {"consolidation_calls_reserved": 4, "consolidation_tokens_reserved": 400}
    reviewer = _Reviewer()
    _run(reviewer, progress, metadata, state)
    candidates = metadata["selection_consolidation_candidates"]
    assert _ids(candidates) == ["big", "small"]
    assert _outcome(candidates, "small") == {"code_id": "small", "outcome": "reviewed-kept", "unresolved": False}
    assert _outcome(candidates, "big")["outcome"] == "unreviewed-oversized"
    assert _outcome(candidates, "big")["unresolved"] is True
    assert metadata["selection_consolidation_status"] == "unresolved"
    assert metadata["selection_consolidation_reason"] == "unreviewed_oversized"
    assert metadata["selection_consolidation_error"] is None
    assert metadata["selection_consolidation"] is None
    assert metadata.get("selection_consolidation_skipped") is None
    assert progress["consolidation_closed"] is False
    assert progress["raised_ids"] == ["small", "big"]
    assert metadata["selection_consolidation_unresolved_ids"] == ["big"]
    assert reviewer.payloads == [["small"]]
    exported = _selection_export({"metadata": metadata})
    assert _ids(exported["selection_consolidation_candidates"]) == ["big", "small"]
    assert exported["selection_consolidation_unresolved_ids"] == ["big"]


def test_consolidation_usage_accumulates_across_attempts():
    class _Charge:
        token_budget = 6000
        max_calls = 8
        max_tokens = 5000
        calls = 0

        def estimate(self, payload, schema):
            return 20

        def refusal_reason(self, payload, schema, metadata):
            return None

        async def decide(self, payload, schema, metadata):
            self.calls += 1
            metadata["decision_calls_used"] = int(metadata.get("decision_calls_used") or 0) + 1
            metadata["decision_tokens_used"] = int(metadata.get("decision_tokens_used") or 0) + 20
            metadata["decision_stop"] = "review_error"
            return None, {"status": "error"}

    state = _selection_state([_code("c-0", "KEEP"), _code("c-1", "KEEP")])
    progress = {
        "raised_ids": ["c-0", "c-1"],
        "reviewed_ids": ["c-0", "c-1"],
        "chunks": [["c-0", "c-1"]],
        "compaction": {},
    }
    reviewer = _Charge()
    metadata = {
        "decision_calls_used": 2,
        "decision_tokens_used": 30,
        "consolidation_calls_reserved": 4,
        "consolidation_tokens_reserved": 400,
    }
    _run(reviewer, progress, metadata, state)
    _run(reviewer, progress, metadata, state)
    assert reviewer.calls == 2
    assert metadata["consolidation_calls_charged"] == 2
    assert metadata["consolidation_tokens_charged"] == 40
    assert metadata["consolidation_attempt_calls"] == 1
    assert metadata["consolidation_attempt_tokens"] == 20
    assert [item["calls"] for item in metadata["consolidation_attempts"]] == [1, 1]
    assert [item["tokens"] for item in metadata["consolidation_attempts"]] == [20, 20]
    assert metadata["decision_calls_used"] == 2
    assert metadata["decision_tokens_used"] == 30
    assert metadata["selection_consolidation_status"] == "failed"

    metadata["consolidation_calls_reserved"] = 2
    _run(reviewer, progress, metadata, state)
    assert reviewer.calls == 2
    assert metadata["consolidation_calls_charged"] == 2
    assert metadata["consolidation_tokens_charged"] == 40
    assert metadata["consolidation_attempts"][-1]["calls"] == 0
    assert metadata["consolidation_attempts"][-1]["tokens"] == 0
    assert metadata["consolidation_attempts"][-1]["reason"] == "call_allowance"
    assert metadata["decision_calls_used"] == 2


def test_validated_groups_are_reused_and_synthesis_can_change_representation():
    class _Reviewer:
        token_budget = 50
        max_calls = 8
        max_tokens = 5000
        payloads: list[list[str]] = []

        def estimate(self, payload, schema):
            rows = payload.get("codes") or []
            count = len(rows)
            if any(int(row.get("evidence_compaction") or 0) >= 3 for row in rows):
                return 60
            return 40 if count <= 1 else 90

        def refusal_reason(self, payload, schema, metadata):
            if self.estimate(payload, schema) > self.token_budget:
                return "request_budget"
            return None

        async def decide(self, payload, schema, metadata):
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            self.payloads.append(ids)
            metadata["decision_calls_used"] = int(metadata.get("decision_calls_used") or 0) + 1
            metadata["decision_tokens_used"] = int(metadata.get("decision_tokens_used") or 0) + 20
            return {"raised_code_ids": ids, "rationale": "kept"}, {"status": "validated"}

    codes = [_code(item, "KEEP") for item in ("c-0", "c-1", "c-2")]
    state = _selection_state(codes)
    progress = {
        "raised_ids": ["c-0", "c-1", "c-2"],
        "reviewed_ids": ["c-0", "c-1", "c-2"],
        "chunks": [["c-0"], ["c-1"], ["c-2"]],
        "compaction": {},
    }
    metadata = {
        "decision_calls_used": 1,
        "decision_tokens_used": 10,
        "consolidation_calls_reserved": 8,
        "consolidation_tokens_reserved": 800,
    }
    reviewer = _Reviewer()
    _run(reviewer, progress, metadata, state)
    assert metadata["selection_consolidation_status"] == "unresolved-synthesis"
    assert metadata["selection_consolidation_reason"] == "synthesis_cannot_fit"
    assert metadata["selection_consolidation_error"] is None
    assert progress["consolidation_closed"] is False
    assert progress["raised_ids"] == ["c-0", "c-1", "c-2"]
    assert reviewer.payloads == [["c-0"], ["c-1"], ["c-2"]]
    assert len(progress["consolidation_validated_groups"]) == 3
    candidates = metadata["selection_consolidation_candidates"]
    assert _ids(candidates) == ["c-0", "c-1", "c-2"]
    assert {row["outcome"] for row in candidates} == {"unresolved-synthesis"}
    assert all(row["unresolved"] is True for row in candidates)
    assert metadata["consolidation_calls_charged"] == 3

    before = len(reviewer.payloads)
    _run(reviewer, progress, metadata, state)
    assert reviewer.payloads[before:] == []
    assert metadata["consolidation_attempts"][-1]["calls"] == 0
    assert metadata["consolidation_calls_charged"] == 3
    assert metadata["selection_consolidation_status"] == "unresolved-synthesis"
    assert progress["raised_ids"] == ["c-0", "c-1", "c-2"]

    reviewer.token_budget = 80
    _run(reviewer, progress, metadata, state)
    assert reviewer.payloads[before:] == [["c-0", "c-1", "c-2"]]
    assert metadata["selection_consolidation_status"] == "completed"
    assert metadata["selection_consolidation_error"] is None
    assert progress["consolidation_closed"] is True
    assert progress["raised_ids"] == ["c-0", "c-1", "c-2"]
    assert {row["outcome"] for row in metadata["selection_consolidation_candidates"]} == {"reviewed-kept"}
    assert metadata["selection_consolidation_unresolved_ids"] == []
    assert metadata["consolidation_calls_charged"] == 4


def test_a_transient_failure_does_not_resend_a_validated_group():
    class _FailOnce:
        token_budget = 50
        max_calls = 8
        max_tokens = 5000
        payloads: list[list[str]] = []
        failed = False

        def estimate(self, payload, schema):
            count = len(payload.get("codes") or [])
            return 40 if count <= 1 else 90

        def refusal_reason(self, payload, schema, metadata):
            if self.estimate(payload, schema) > self.token_budget:
                return "request_budget"
            return None

        async def decide(self, payload, schema, metadata):
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            self.payloads.append(ids)
            if "c-1" in ids and not self.failed:
                self.failed = True
                raise RuntimeError("scripted transient failure")
            return {"raised_code_ids": ids, "rationale": "kept"}, {"status": "validated"}

    state = _selection_state([_code(item, "KEEP") for item in ("c-0", "c-1")])
    progress = {
        "raised_ids": ["c-0", "c-1"],
        "reviewed_ids": ["c-0", "c-1"],
        "chunks": [["c-0"], ["c-1"]],
        "compaction": {},
    }
    metadata = {"consolidation_calls_reserved": 4, "consolidation_tokens_reserved": 400}
    reviewer = _FailOnce()
    _run(reviewer, progress, metadata, state)
    assert metadata["selection_consolidation_status"] == "failed"
    assert "RuntimeError" in metadata["selection_consolidation_error"]
    assert progress["consolidation_closed"] is False
    assert [row["code_ids"] for row in progress["consolidation_validated_groups"]] == [["c-0"]]
    sent = list(reviewer.payloads)
    _run(reviewer, progress, metadata, state)
    assert reviewer.payloads[len(sent):] == [["c-1"]]
    assert progress["raised_ids"] == ["c-0", "c-1"]


def test_a_fitted_summary_can_merge_without_dropping_a_candidate():
    class _Merge:
        token_budget = 50
        max_calls = 8
        max_tokens = 5000

        def estimate(self, payload, schema):
            rows = payload.get("codes") or []
            if any(int(row.get("evidence_compaction") or 0) >= 3 for row in rows):
                return 30
            return 40 if len(rows) <= 1 else 90

        def refusal_reason(self, payload, schema, metadata):
            if self.estimate(payload, schema) > self.token_budget:
                return "request_budget"
            return None

        async def decide(self, payload, schema, metadata):
            ids = [str(row.get("code_id") or "") for row in payload.get("codes") or []]
            if len(ids) > 1:
                return {"raised_code_ids": ["c-0"], "rationale": "merged"}, {"status": "validated"}
            return {"raised_code_ids": ids, "rationale": "kept"}, {"status": "validated"}

    state = _selection_state([_code("c-0", "KEEP"), _code("c-1", "KEEP")])
    progress = {
        "raised_ids": ["c-0", "c-1"],
        "reviewed_ids": ["c-0", "c-1"],
        "chunks": [["c-0"], ["c-1"]],
        "compaction": {},
    }
    metadata = {"consolidation_calls_reserved": 4, "consolidation_tokens_reserved": 400}
    _run(_Merge(), progress, metadata, state)
    candidates = metadata["selection_consolidation_candidates"]
    assert _ids(candidates) == ["c-0", "c-1"]
    assert _outcome(candidates, "c-0")["outcome"] == "reviewed-kept"
    assert _outcome(candidates, "c-1")["outcome"] == "reviewed-merged"
    assert all(row["unresolved"] is False for row in candidates)
    assert metadata["selection_consolidation_status"] == "completed"
    assert metadata["selection_consolidation_unresolved_ids"] == []
    assert progress["raised_ids"] == ["c-0"]
    assert progress["consolidation_closed"] is True
