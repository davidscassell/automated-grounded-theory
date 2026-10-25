"""0.9.19 review fixes. Fakes only. No network calls."""

from __future__ import annotations

import copy

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, axial_code
from automated_grounded_theory.axial import (
    AxialCheckpointMismatch,
    _evidence_source_hash,
    _source_hash,
    _source_mapping,
)
from automated_grounded_theory.records import POLICY_VERSION
from tests.test_v080 import FakeAxialCoder, _corpus


def test_version_is_0919():
    assert __version__ == "0.9.25"
    assert POLICY_VERSION == "0.9.25"


def _passages(*, attribution: bool) -> list[dict]:
    shared = "The tools for the shift sit beside coworkers on the crew tonight"
    linked = "tools for the shift lead to coworkers on the crew because the night ran long"
    text = f"{shared} {linked}" if attribution else shared
    return [
        {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": text},
        {"passage_id": "p2", "doc_id": "d2", "respondent_id": "r2", "text": text},
        {
            "passage_id": "p3",
            "doc_id": "d3",
            "respondent_id": "r3",
            "text": "The schedule changed and nobody mentioned either side.",
        },
    ]


def _relationship(*, attribution: bool) -> dict:
    if attribution:
        quotes = [
            {
                "passage_id": "p1",
                "quote": "tools for the shift lead to coworkers on the crew because the night ran long",
                "endpoint": "both",
            },
            {
                "passage_id": "p2",
                "quote": "tools for the shift lead to coworkers on the crew because the night ran long",
                "endpoint": "both",
            },
        ]
    else:
        quotes = [
            {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
            {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        ]
    return {
        "relationship_id": "rel-0001",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": "",
        "quote_from": "tools for the shift sit",
        "quote_to": "coworkers on the crew tonight",
        "statement": "tools for the shift may be related to coworkers on the crew",
        "disposition": "proposed",
        "flags": [],
        "evidence_condition": "anchored",
        "paradigm_role": "unspecified",
        "counterexamples": [
            {
                "passage_id": "p3",
                "judgment": "contradicts",
                "valid_test": True,
                "synthetic": False,
                "quote": "The schedule changed",
            }
        ],
        "case_quotes": quotes,
    }


def _source(*, memo: dict | None = None, attribution: bool = False) -> dict:
    return {
        "codebook": {
            "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
            "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
        },
        "passages": _passages(attribution=attribution),
        "segments": [
            {"segment_id": "s1", "code_id": "tools", "passage_id": "p1", "quote": "tools for the shift sit"},
            {"segment_id": "s2", "code_id": "crew", "passage_id": "p1", "quote": "coworkers on the crew tonight"},
            {"segment_id": "s3", "code_id": "tools", "passage_id": "p2", "quote": "tools for the shift sit"},
            {"segment_id": "s4", "code_id": "crew", "passage_id": "p2", "quote": "coworkers on the crew tonight"},
        ],
        "relationships": [_relationship(attribution=attribution)],
        "memos": [] if memo is None else [memo],
        "properties": [],
        "dimensions": [],
    }


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


def test_an_irrelevant_memo_does_not_become_participant_attribution():
    """The two-case pattern stays tier 2. A formatting memo does not raise it."""
    quiet = _row(_source())
    assert quiet["evidence_tier"] == 2
    assert quiet["allowed_wording"] == "co-occurs in 2 respondents"
    assert quiet["attribution_source"] is None
    noisy = _row(
        _source(
            memo={
                "relationship_id": "rel-0001",
                "text": "I mention tools because this memo is a formatting example.",
            }
        )
    )
    assert noisy["evidence_tier"] == 2
    assert noisy["allowed_wording"] == "co-occurs in 2 respondents"
    assert noisy["attribution_source"] is None
    assert "participants attribute" not in str(noisy["allowed_wording"])
    named = _row(
        _source(
            memo={
                "relationship_id": "rel-0001",
                "text": (
                    "I mention tools for the shift and coworkers on the crew "
                    "because this memo is a formatting example."
                ),
            }
        )
    )
    assert named["evidence_tier"] == 2
    assert named["allowed_wording"] == "co-occurs in 2 respondents"
    assert named["attribution_source"] is None


def test_a_participant_quote_still_reaches_tier_3():
    row = _row(_source(attribution=True))
    assert row["evidence_tier"] == 3
    assert row["evidence_label"] == "attribution"
    assert row["allowed_wording"] == "participants attribute X to Y"
    assert row["attribution_source"] == "participant"


def test_an_analyst_interpretation_is_labeled_and_not_downstream_eligible():
    """0.9.20 keeps the analyst reading on its own component.

    The row wording stays the co-occurrence pattern. The analyst component
    uses analyst wording, stays a hypothesis, and does not support a core.
    The pattern component stays downstream eligible.
    """
    row = _row(
        _source(
            memo={
                "relationship_id": "rel-0001",
                "actor": "human",
                "text": (
                    "The analyst interprets tools for the shift as related to "
                    "coworkers on the crew."
                ),
            }
        )
    )
    analyst = row["claims"]["analyst_interpretation"]
    assert row["evidence_tier"] == 2
    assert row["allowed_wording"] == "co-occurs in 2 respondents"
    assert row["attribution_source"] is None
    assert analyst["allowed_wording"] == "analyst interprets X as related to Y"
    assert analyst["present"] is True
    assert analyst["downstream_eligible"] is False
    assert analyst["supports_core"] is False
    assert analyst["evidence_use"] == "hypothesis_exploration"
    assert "participants attribute" not in row["allowed_wording"]
    assert "participants attribute" not in analyst["allowed_wording"]
    assert "analyst_interpretation" in row["flags"]
    assert row["needs_review"] is True
    assert row["downstream_eligible"] is True
    assert row["evidence_use"] == "supported_evidence"
    assert row["core_support_component"] == "pattern"


def test_llm_mode_cannot_use_its_own_memo_as_participant_attribution():
    row = _row(
        _source(
            memo={
                "memo_id": "memo-0001",
                "relationship_id": "rel-0001",
                "kind": "model_memo",
                "actor": "model",
                "text": (
                    "The analyst interprets tools for the shift as related to "
                    "coworkers on the crew because participants attribute the link."
                ),
            }
        )
    )
    assert row["evidence_tier"] == 2
    assert row["allowed_wording"] == "co-occurs in 2 respondents"
    assert row["attribution_source"] is None
    assert "participants attribute" not in str(row["allowed_wording"])
    assert "analyst_interpretation" not in row["flags"]


def _identified() -> dict:
    source = _corpus()
    for passage in source["passages"]:
        passage["respondent_id"] = str(passage["doc_id"])
    return source


def _pause(source: dict, saver: InMemorySaver, thread_id: str):
    return axial_code(
        source,
        FakeAxialCoder(),
        seed=3,
        max_axial_review_calls=2,
        batch_size=1,
        checkpointer=saver,
        thread_id=thread_id,
        interrupt=True,
    )


def _resume(source: dict, saver: InMemorySaver, thread_id: str, **extra):
    return axial_code(
        source,
        FakeAxialCoder(),
        seed=3,
        max_axial_review_calls=2,
        batch_size=1,
        checkpointer=saver,
        thread_id=thread_id,
        interrupt=True,
        resuming=True,
        resume={"leave_relations": True},
        **extra,
    )


def _rewrite_saved_source_hash(saver: InMemorySaver, thread_id: str, source_hash: str) -> None:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    saved = saver.get_tuple(config)
    assert saved is not None
    version = saved.checkpoint["channel_versions"]["metadata"]
    namespace = saved.config["configurable"].get("checkpoint_ns", "")
    key = (thread_id, namespace, "metadata", version)
    type_tag, blob = saver.blobs[key]
    metadata = saver.serde.loads_typed((type_tag, blob))
    metadata["procedure"]["source_hash"] = source_hash
    saver.blobs[key] = saver.serde.dumps_typed(metadata)


def test_an_unchanged_axial_source_resumes():
    saver = InMemorySaver()
    source = _identified()
    paused = _pause(copy.deepcopy(source), saver, "same")
    assert paused.interrupted
    resumed = _resume(copy.deepcopy(source), saver, "same")
    assert resumed.metadata["procedure"]["source_hash"] == _evidence_source_hash(_source_mapping(source))


def test_changed_axial_evidence_is_rejected_on_resume():
    source = _identified()

    def mismatch(mutate, thread_id: str) -> None:
        saver = InMemorySaver()
        _pause(copy.deepcopy(source), saver, thread_id)
        changed = copy.deepcopy(source)
        mutate(changed)
        with pytest.raises(AxialCheckpointMismatch, match="source_hash"):
            _resume(changed, saver, thread_id, allow_changes=True)

    mismatch(lambda item: item["codebook"]["c-prep"].__setitem__("definition", "a different definition"), "definition")
    mismatch(lambda item: item["segments"].clear(), "segments")
    mismatch(lambda item: item["passages"][0].__setitem__("respondent_id", "r-other"), "case")


def test_a_0918_checkpoint_cannot_resume_without_the_evidence_fingerprint():
    saver = InMemorySaver()
    source = _identified()
    _pause(copy.deepcopy(source), saver, "legacy")
    legacy = _source_hash(_source_mapping(copy.deepcopy(source)))
    _rewrite_saved_source_hash(saver, "legacy", legacy)
    with pytest.raises(AxialCheckpointMismatch, match="fingerprinted"):
        _resume(copy.deepcopy(source), saver, "legacy", allow_changes=True)
