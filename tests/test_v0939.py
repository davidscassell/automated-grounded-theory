"""0.9.39 core reuse accounting. Fakes and synthetic fixtures only. No network calls.

These checks reproduce a multi-step resume of an export written before match
fingerprints. They are processing coverage. They do not establish selection validity.
"""

from __future__ import annotations

import json

from automated_grounded_theory.axial import export_axial_state, resume_core_selection
from automated_grounded_theory.decision_review import (
    CHUNKING_CHECKPOINT_MESSAGE,
    CORE_PROVENANCE_REUSED_CHECKPOINT,
    CORE_PROVENANCE_REUSED_EXPORT,
    CORE_PROVENANCE_SENT_THIS_RUN,
    CORE_REUSE_ACCOUNTING_RULE,
    CORE_UPSTREAM_EVIDENCE_RULE,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.version import __version__
from tests.test_v0934 import _core_source
from tests.test_v0935 import ScriptedCore, _run_core
from tests.test_v0936 import _match_records
from tests.test_v0937 import _run
from tests.test_v0938 import _partial_wide_export


def test_version_names_the_reuse_accounting_rules():
    assert __version__ == "0.9.40"
    assert POLICY_VERSION == "0.9.40"
    assert CORE_REUSE_ACCOUNTING_RULE == "loaded_unsent_and_final_usage_v1"
    assert CORE_UPSTREAM_EVIDENCE_RULE == "note_thin_axial_evidence_v1"
    assert "0.9.38" in CHUNKING_CHECKPOINT_MESSAGE
    assert "0.9.39" in CHUNKING_CHECKPOINT_MESSAGE


def _resume(tmp_path):
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
    return resumed, model


def test_a_multi_step_resume_counts_only_loaded_matches_and_closes_usage(tmp_path):
    _partial_wide_export(tmp_path)
    resumed, model = _resume(tmp_path)
    records = _match_records(resumed)
    provenance = [str(record.get("provenance") or "") for record in records]
    assert provenance == [CORE_PROVENANCE_REUSED_EXPORT] * 2 + [CORE_PROVENANCE_SENT_THIS_RUN] * 4
    assert [str(record.get("reuse") or "") for record in records] == provenance
    assert resumed.metadata["core_matches_reused"] == 2
    assert resumed.metadata["core_unfingerprinted_reused"] == 2
    assert model.core_seen == 4
    assert resumed.metadata["core_outcome"] == "no_core_supported"
    usage = resumed.metadata["core_usage"]
    assert usage["prior_calls"] == 2
    assert usage["this_run_calls"] == 4
    assert usage["calls"] == 6
    assert usage["prior_tokens_estimated"] > 0
    assert usage["this_run_tokens_estimated"] > 0
    assert usage["prior_tokens_estimated"] + usage["this_run_tokens_estimated"] == usage["tokens_estimated"]
    progress = resumed.metadata["decision_chunk_state"]
    assert usage["calls"] == progress["calls"]
    assert usage["tokens_estimated"] == progress["tokens_estimated"]
    step = next(
        item
        for item in resumed.metadata["decision_budget_steps"]
        if item.get("step") == "axial_core_review" and item.get("allowance") == 20
    )
    assert step["starting_usage"] == usage["prior_calls"]
    assert step["calls_made"] == usage["this_run_calls"]
    assert step["starting_tokens"] == usage["prior_tokens_estimated"]
    assert step["tokens_used"] == usage["this_run_tokens_estimated"]
    assert resumed.metadata["core_thin_upstream_evidence"] is True
    assert "outcome is unchanged" in resumed.metadata["core_upstream_evidence_note"]
    assert resumed.metadata["core_reuse_accounting_rule"] == CORE_REUSE_ACCOUNTING_RULE


def test_a_fingerprinted_export_is_reused_checkpoint_not_unfingerprinted(tmp_path):
    first, first_model = _run(
        _core_source("packing " * 40, 54),
        ["abstain"] * 6,
        max_decision_calls=4,
        max_core_review_calls=2,
        max_core_review_tokens=10**6,
        max_axial_review_tokens=10**6,
        axial_token_budget=6000,
        allow_partial_core_review=True,
    )
    assert first_model.core_seen == 2
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
    stored = json.loads((tmp_path / "selective_run.json").read_text(encoding="utf-8"))
    saved = stored["metadata"]["decision_chunk_state"]["core_bracket"]["rounds"][0]["match_records"]
    assert all(record.get("payload_fingerprint") and record.get("policy_fingerprint") for record in saved)
    resumed, model = _resume(tmp_path)
    records = _match_records(resumed)
    assert [record.get("provenance") for record in records[:2]] == [CORE_PROVENANCE_REUSED_CHECKPOINT] * 2
    assert all(record.get("provenance") == CORE_PROVENANCE_SENT_THIS_RUN for record in records[2:])
    assert resumed.metadata["core_matches_reused"] == 2
    assert resumed.metadata["core_unfingerprinted_reused"] == 0
    assert model.core_seen == 4
    usage = resumed.metadata["core_usage"]
    assert usage["prior_calls"] == 2
    assert usage["this_run_calls"] == 4
    assert usage["calls"] == 6


def test_thin_upstream_evidence_is_noted_only_when_the_graph_is_empty():
    bare = _run_core(["abstain", "abstain"])
    assert bare.metadata["core_outcome"] == "no_core_supported"
    assert bare.metadata["core_selection_complete"] is True
    assert bare.metadata["core_thin_upstream_evidence"] is True
    assert bare.metadata["core_upstream_evidence_note"]
    assert bare.metadata["model_core_id"] == ""
    usage = bare.metadata["core_usage"]
    assert usage["prior_calls"] == 0
    assert usage["this_run_calls"] == usage["calls"]
    assert usage["calls"] == 2
    assert [row.get("provenance") for row in _match_records(bare)] == [CORE_PROVENANCE_SENT_THIS_RUN] * 2

    source = _core_source("tools", 3)
    source["relationships"] = [
        {
            "relationship_id": "rel-1",
            "from_id": "c00",
            "to_id": "c01",
            "passage_id": "p1",
            "statement": "Tools relate to the next code in this synthetic note.",
            "evidence_tier": 2,
            "disposition": "proposed",
        }
    ]
    held, _model = _run(
        source,
        ["abstain", "abstain"],
        max_decision_calls=8,
        max_axial_review_tokens=10**6,
        axial_token_budget=10**6,
    )
    assert held.metadata["core_outcome"] == "no_core_supported"
    assert held.metadata["core_thin_upstream_evidence"] is None
    assert held.metadata["core_upstream_evidence_note"] is None
    assert any(int(row.get("pattern_supported_centrality") or 0) > 0 for row in held.core_candidates)
