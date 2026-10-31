"""0.9.38 resume stops. Fakes and synthetic fixtures only. No network calls.

These checks reproduce a saved core stop that no longer matches the bracket.
They are processing coverage. They do not establish selection validity.
"""

from __future__ import annotations

import json

from automated_grounded_theory.axial import export_axial_state, resume_core_selection
from automated_grounded_theory.decision_review import (
    CHUNKING_CHECKPOINT_MESSAGE,
    CORE_PAIR_EXCEEDS_REQUEST_BUDGET,
    CORE_RESUME_STOP_RULE,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.version import __version__
from tests.test_v0935 import ScriptedCore, _core_bodies, _run_core
from tests.test_v0936 import _match_records
from tests.test_v0937 import _run, _wide_source

_FINGERPRINT_KEYS = (
    "payload_fingerprint",
    "policy_fingerprint",
    "chunk_index",
    "chunk_count",
    "round_role",
    "reuse",
    "estimated_input_tokens",
)


def test_version_names_the_resume_stop_rule():
    assert __version__ == "0.9.41"
    assert POLICY_VERSION == "0.9.41"
    assert CORE_RESUME_STOP_RULE == "rederive_inherited_stops_v1"
    assert "0.9.37" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.38" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.39" in CHUNKING_CHECKPOINT_MESSAGE


def _partial_wide_export(tmp_path):
    """Two validated opening abstentions, then a v42-shaped checkpoint."""
    first, first_model = _run(
        _wide_source(),
        ["abstain"] * 6,
        max_decision_calls=4,
        max_core_review_calls=2,
        max_core_review_tokens=10**6,
        max_axial_review_tokens=10**6,
        axial_token_budget=6000,
        allow_partial_core_review=True,
    )
    assert first_model.core_seen == 2
    assert first.metadata["core_outcome"] == "selection_unresolved"
    export_axial_state(
        {
            "metadata": first.metadata,
            "warnings": list(first.warnings),
            "prompt_log": list(first.prompt_log),
            "relationships": [],
            "core_candidates": list(first.core_candidates),
            "saturation": {},
            "coverage": {},
        },
        tmp_path,
    )
    run_path = tmp_path / "selective_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metadata = run["metadata"]
    progress = metadata["decision_chunk_state"]
    progress["stop_reasons"] = [CORE_PAIR_EXCEEDS_REQUEST_BUDGET]
    progress["open"] = True
    progress["provisional"] = True
    metadata["decision_stop"] = CORE_PAIR_EXCEEDS_REQUEST_BUDGET
    metadata["core_comparison_stop"] = CORE_PAIR_EXCEEDS_REQUEST_BUDGET
    reused = []
    for round_row in progress["core_bracket"]["rounds"]:
        for record in round_row.get("match_records") or []:
            reused.append([str(item) for item in record.get("match") or []])
            for key in _FINGERPRINT_KEYS:
                record.pop(key, None)
    assert len(reused) == 2
    run_path.write_text(json.dumps(run), encoding="utf-8")
    return reused


def test_a_stale_pair_stop_resumes_every_opening_match_without_resending(tmp_path):
    reused = _partial_wide_export(tmp_path)
    model = ScriptedCore(core_steps=["abstain"] * 8)
    resumed = resume_core_selection(
        tmp_path,
        model,
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=4,
        max_core_review_calls=20,
        max_core_review_tokens=10**6,
        max_axial_review_tokens=10**6,
        axial_token_budget=6000,
        batch_size=1,
    )
    records = _match_records(resumed)
    opening = [
        [str(item) for item in record.get("match") or []]
        for record in records
        if str(record.get("round_role") or "opening") == "opening"
    ]
    # The two stored matches stay, and the four that were never sent are sent once.
    assert len(opening) == 6
    assert opening[:2] == reused
    assert [record.get("match") for record in records[:2]] == reused
    assert resumed.metadata["core_matches_reused"] >= 2
    assert resumed.metadata["core_reuse_invalidated"] == ""
    assert all(isinstance(record.get("chunk_index"), int) for record in records[:2])
    assert all(isinstance(record.get("chunk_count"), int) for record in records[:2])
    assert model.core_seen == 4
    sent_ids = [body["review_chunk"]["code_ids"] for body in _core_bodies(resumed)]
    assert sent_ids[:2] == reused
    assert sent_ids[2:] == [record.get("match") for record in records[2:]]
    assert len(sent_ids) == len(records)
    for stale in reused:
        assert sent_ids.count(stale) == 1

    preflight = resumed.metadata["core_bracket_preflight"]
    assert preflight["unavoidable_opening_calls"] == 6
    assert len(preflight["opening_match_tokens"]) == preflight["unavoidable_opening_calls"]
    assert all(int(item) <= 6000 for item in preflight["opening_match_tokens"])
    assert preflight["opening_request_too_large"] is False
    assert int(preflight["call_shortfall"] or 0) == 0
    assert int(preflight["token_shortfall"] or 0) == 0
    assert len(records) == preflight["unavoidable_opening_calls"]
    assert sum(len(record.get("match") or []) for record in records) == 54
    assert resumed.metadata["core_candidates_eligible"] == 54
    assert resumed.metadata["core_candidates_compared"] == 54
    assert resumed.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 6
    assert resumed.metadata["core_outcome"] == "no_core_supported"
    assert CORE_PAIR_EXCEEDS_REQUEST_BUDGET not in (resumed.metadata.get("decision_chunk_state") or {}).get(
        "stop_reasons", []
    )
    assert resumed.metadata.get("decision_stop") != CORE_PAIR_EXCEEDS_REQUEST_BUDGET
    assert resumed.metadata["core_resume_stop_rule"] == CORE_RESUME_STOP_RULE


def test_a_pair_that_still_does_not_fit_stops_again_after_the_saved_reason_is_cleared(tmp_path):
    from tests.test_v0934 import _core_source, _payload_estimate
    from automated_grounded_theory.axial import AxialDecisionReviewer
    from tests.test_v080 import FakeAxialCoder

    source = _core_source("packing " * 40, 3)
    from automated_grounded_theory.axial import _score_candidates

    candidates = _score_candidates(source)
    probe = AxialDecisionReviewer(FakeAxialCoder(), max_calls=8, max_tokens=10**6, token_budget=10**6)
    one = _payload_estimate(probe, candidates[:1], ["c00"], whole_set=True, eligible=1)
    budget = max(1, one - 1)
    first, first_model = _run(
        source,
        ["abstain"] * 4,
        max_decision_calls=8,
        max_axial_review_tokens=10**6,
        axial_token_budget=budget,
    )
    assert first_model.core_seen == 0
    assert first.metadata["decision_stop"] == CORE_PAIR_EXCEEDS_REQUEST_BUDGET
    export_axial_state(
        {
            "metadata": first.metadata,
            "warnings": list(first.warnings),
            "prompt_log": list(first.prompt_log),
            "relationships": [],
            "core_candidates": list(first.core_candidates),
            "saturation": {},
            "coverage": {},
        },
        tmp_path,
    )
    model = ScriptedCore(core_steps=["abstain"] * 4)
    resumed = resume_core_selection(
        tmp_path,
        model,
        seed=3,
        decision_mode="llm",
        interrupt=False,
        include_memos=False,
        include_storyline=False,
        max_decision_calls=8,
        max_axial_review_tokens=10**6,
        axial_token_budget=budget,
        batch_size=1,
    )
    assert model.core_seen == 0
    assert resumed.metadata["decision_stop"] == CORE_PAIR_EXCEEDS_REQUEST_BUDGET
    assert resumed.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 0
    assert resumed.metadata["core_candidates_compared"] == 0
    assert resumed.metadata["core_outcome"] == "selection_unresolved"


def test_a_failed_core_call_is_not_a_compared_seat():
    failed = _run_core(["parse_failure"])
    assert failed.metadata["decision_stop"] == "review_error"
    assert failed.metadata["core_candidates_compared"] == 0
    assert failed.metadata["decision_reviews"]["axial_core_review"]["chunks_sent"] == 1
