"""0.9.9 regressions. Fake models only. No live evaluation."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import __version__, theoretical_sample
from automated_grounded_theory.gate_evaluation import (
    GateEvaluationError,
    export_gate_relationships,
    load_adjudications,
    score_gate_adjudications,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.sampling import (
    _number_directions,
    _question_key,
    _target_key,
    normalize_sampling_ledger,
    rank_sampling_directions,
    retrieve_sampling_passages,
    stable_question_key,
)
from automated_grounded_theory.saturation import stopping_criteria_report
from tests.test_v080 import _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs, _payload


def test_version_is_099():
    assert __version__ == "0.9.29"
    assert POLICY_VERSION == "0.9.29"


class _SameRelevantNote(FakeSamplingCoder):
    """Every reread returns the same relevant judgment and does not name a new question."""

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "ModelSamplingSelection":
                direction = str((body.get("directions") or [{}])[0].get("direction_id") or "")
                groups = body.get("passage_proposals") or [{}]
                proposals = (groups[0] or {}).get("proposals") or [{}]
                passage = str((proposals[0] or {}).get("passage_id") or "")
                parsed = schema(
                    selections=[{"direction_id": direction, "passage_id": passage}] if direction and passage else [],
                    rationale="The same selection rule, with no new question.",
                )
            elif name == "SamplingRereadNote":
                parsed = schema(
                    passage_id=str(body.get("passage_id") or ""),
                    note="This evidence is not irrelevant to the packing question.",
                    judgment="relevant",
                    quote=str(body.get("text") or "")[:48],
                )
            elif name == "ModelSaturationJudgment":
                parsed = schema(
                    assessment="not_saturated",
                    rationale="The window is interpretable and a sampling gap is still open.",
                )
            else:
                parsed = schema()
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def test_repeated_relevant_notes_do_not_mint_new_directions():
    source = _corpus()
    rounds = []
    for _ in range(4):
        result = theoretical_sample(
            source,
            _SameRelevantNote(),
            decision_mode="llm",
            pursuit="reread",
            interrupt=False,
            quiet_rounds=2,
            **_kwargs(max_sampling_review_calls=4, max_decision_calls=4),
        )
        rounds.append(result)
        source = _corpus()
        source["metadata"] = {"sampling_ledger": result.sampling_ledger}
    ledger = rounds[-1].sampling_ledger
    assert len(ledger) == 4
    follow_ups = [
        key
        for row in ledger
        for key in (row.get("direction_keys") or [])
        if "reread_follow_up" in str(key)
    ]
    assert follow_ups == []
    introduced = []
    seen = set(ledger[0]["direction_keys"])
    for row in ledger[1:]:
        introduced.extend(key for key in row["new_direction_keys"] if key not in seen)
        assert row["new_direction_keys"] == []
        assert row["analytic_new_direction_keys"] == []
    assert len(introduced) <= 1
    pairs = []
    for row in ledger:
        for note in row.get("reread_notes") or []:
            pairs.append((note["question_key"], note["passage_id"]))
    assert len(pairs) == len(set(pairs))
    assert pairs
    quiet = next(
        item
        for item in rounds[-1].stopping_criteria["criteria"]
        if item["criterion"] == "sampling_rounds_without_new_directions"
    )
    assert quiet["status"] == "met"
    assert rounds[-1].saturation["judgment"] is None


def test_nested_follow_up_keys_normalize_and_a_distinct_question_does_not_nest():
    assert stable_question_key("reread_follow_up:reread_follow_up:thin_category:c-prep") == "thin_category:c-prep"
    legacy = [
        {
            "direction_keys": ["thin_category:c-prep"],
            "new_direction_keys": ["thin_category:c-prep"],
            "analytic_new_direction_keys": ["thin_category:c-prep"],
            "reread_unsent_pairs": [],
        },
        {
            "direction_keys": ["thin_category:c-prep", "reread_follow_up:thin_category:c-prep"],
            "new_direction_keys": ["reread_follow_up:thin_category:c-prep"],
            "analytic_new_direction_keys": ["reread_follow_up:thin_category:c-prep"],
            "reread_notes": [
                {
                    "direction_key": "thin_category:c-prep",
                    "judgment": "relevant",
                    "note": "A property showed up in this passage.",
                    "passage_id": "p-low",
                }
            ],
            "closed_direction_keys": ["reread_follow_up:thin_category:c-other"],
            "reread_unsent_pairs": [],
        },
        {
            "direction_keys": ["reread_follow_up:reread_follow_up:thin_category:c-prep"],
            "new_direction_keys": ["reread_follow_up:reread_follow_up:thin_category:c-prep"],
            "analytic_new_direction_keys": ["reread_follow_up:reread_follow_up:thin_category:c-prep"],
            "reread_unsent_pairs": [],
            "reread_notes": [
                {
                    "direction_key": "reread_follow_up:thin_category:c-prep",
                    "judgment": "relevant",
                    "note": "This evidence is not irrelevant to the packing question.",
                    "passage_id": "p-low",
                }
            ],
        },
    ]
    normalized = normalize_sampling_ledger(legacy)
    assert normalized[1]["new_direction_keys"] == []
    assert normalized[2]["direction_keys"] == ["thin_category:c-prep"]
    assert normalized[2]["analytic_new_direction_keys"] == []
    assert normalized[2]["reread_notes"][0]["question_key"] == "thin_category:c-prep"
    assert normalized[2]["reread_notes"][0]["recorded_direction_key"] == "reread_follow_up:thin_category:c-prep"
    assert normalized[1]["completed_pairs"] == [{"question_key": "thin_category:c-prep", "passage_id": "p-low"}]
    assert normalized[1]["closed_direction_keys"] == ["thin_category:c-other"]
    quiet = stopping_criteria_report(
        {"n_documents": 4},
        normalized[1:],
        quiet_rounds=2,
    )
    quiet_row = next(item for item in quiet["criteria"] if item["criterion"] == "sampling_rounds_without_new_directions")
    assert quiet_row["status"] == "met"
    held = _corpus()
    held["metadata"] = {"sampling_ledger": normalized}
    ranked = rank_sampling_directions(held)
    assert all(item["kind"] != "reread_follow_up" for item in ranked)
    numbered = _number_directions(ranked, round_index=4)
    groups = retrieve_sampling_passages(numbered, {**held, "sampling_ledger": normalized})
    prep = next(row for row in numbered if _question_key(row) == "thin_category:c-prep")
    offered = next(group for group in groups if group["direction_id"] == prep["direction_id"])
    assert "p-low" not in [row["passage_id"] for row in offered["proposals"]]

    prose = {
        "direction_key": "thin_category:c-prep",
        "judgment": "relevant",
        "note": "A new question about c-return should be opened.",
        "passage_id": "p2",
    }
    unknown = {**prose, "follow_up_kind": "thin_category", "follow_up_target_id": "not-a-code"}
    prose_state = _corpus()
    prose_state["metadata"] = {"sampling_ledger": [{"reread_notes": [prose, unknown]}]}
    prose_keys = {_target_key(item) for item in rank_sampling_directions(prose_state)}
    assert prose_keys == {_target_key(item) for item in rank_sampling_directions(_corpus())}

    distinct_note = {
        "direction_key": "thin_category:c-prep",
        "judgment": "relevant",
        "note": "The wording here must not decide the question.",
        "passage_id": "p2",
        "follow_up_kind": "partial_relationship",
        "follow_up_target_id": "rel-packed",
    }
    distinct_state = _corpus()
    distinct_state["relationships"] = [
        {
            "relationship_id": "rel-packed",
            "from_id": "c-prep",
            "to_id": "c-return",
            "statement": "Preparing to leave comes before the return to work.",
            "disposition": "accepted",
        }
    ]
    distinct_state["metadata"] = {"sampling_ledger": [{"reread_notes": [distinct_note]}]}
    once = rank_sampling_directions(distinct_state)
    assert sum(1 for item in once if _question_key(item) == "partial_relationship:rel-packed") == 1
    assert all("reread_follow_up" not in _question_key(item) for item in once)
    again_state = dict(distinct_state)
    again_state["metadata"] = {
        "sampling_ledger": [
            {
                "direction_keys": ["partial_relationship:rel-packed"],
                "reread_notes": [distinct_note],
            }
        ]
    }
    again = rank_sampling_directions(again_state)
    assert sum(1 for item in again if _question_key(item) == "partial_relationship:rel-packed") == 1


def _adjudication(**overrides):
    row = {
        "relationship_id": "rel-1",
        "rules": ["stem_only"],
        "verdict": "supported",
        "adjudicator": "example-adjudicator",
        "timestamp": "2026-01-01T00:00:00Z",
        "reason": "Example reason, not a study finding.",
    }
    row.update(overrides)
    return row


def test_gate_export_labels_rules_and_leaves_undefined_metrics_null(tmp_path):
    relationships = [
        {
            "relationship_id": "rel-accepted",
            "disposition": "accepted",
            "flags": ["wording_exceeds_tier"],
            "endpoint_anchoring": [{"endpoint": "from", "anchored": False, "reason": "stem_only"}],
        },
        {
            "relationship_id": "rel-withheld",
            "disposition": "provisional",
            "flags": ["acceptance_withheld", "weak_stem_anchor"],
            "counterexamples": [
                {"reclassification_reason": "absence_is_not_contradiction", "synthetic": False, "judgment": "qualifies"}
            ],
        },
        {"relationship_id": "rel-proposed", "disposition": "proposed"},
        {"relationship_id": "rel-synthetic", "disposition": "accepted", "synthetic": True, "flags": ["weak_stem_anchor"]},
    ]
    original = json.dumps(relationships)
    exported = export_gate_relationships(relationships)
    assert json.dumps(relationships) == original
    assert [row["relationship_id"] for row in exported["accepted"]] == ["rel-accepted"]
    assert exported["accepted"][0]["rules"] == ["consequence_wording", "stem_only"]
    withheld = exported["withheld"][0]
    assert withheld["relationship_id"] == "rel-withheld"
    assert "acceptance_withheld" in withheld["rules"]
    assert "stem_only" in withheld["rules"]
    assert "absence_reclassified" in withheld["rules"]
    assert [row["relationship_id"] for row in exported["proposed"]] == ["rel-proposed"]
    assert all(row["relationship_id"] != "rel-proposed" for row in exported["accepted"] + exported["withheld"])
    assert all(row["relationship_id"] != "rel-synthetic" for row in exported["accepted"] + exported["withheld"] + exported["proposed"])

    empty = score_gate_adjudications(exported, [])
    assert empty["accept_precision"] is None
    assert empty["withheld_supported_recall"] is None
    assert empty["by_rule"]["stem_only"]["accept_precision"] is None
    assert empty["by_rule"]["consequence_wording"]["withheld_supported_recall"] is None

    adjudications = [
        _adjudication(relationship_id="rel-accepted", verdict="supported", rules=["stem_only", "consequence_wording"]),
        _adjudication(
            relationship_id="rel-withheld",
            verdict="supported",
            rules=["stem_only", "absence_reclassified"],
            reason="Example of a supported row the gate withheld.",
        ),
    ]
    path = tmp_path / "adjudications.json"
    path.write_text(json.dumps({"adjudications": adjudications}), encoding="utf-8")
    loaded = load_adjudications(path)
    assert loaded[0]["adjudicator"] == "example-adjudicator"
    scored = score_gate_adjudications(exported, loaded)
    assert scored["accept_precision"] == 1.0
    assert scored["withheld_supported_recall"] == 0.5
    assert scored["by_rule"]["stem_only"]["accept_precision"] == 1.0
    assert scored["by_rule"]["stem_only"]["withheld_supported_recall"] == 0.5
    assert scored["by_rule"]["absence_reclassified"]["accept_precision"] is None
    assert scored["by_rule"]["absence_reclassified"]["withheld_supported_recall"] == 1.0
    assert scored["unmatched_adjudication_ids"] == []

    missing = score_gate_adjudications(
        exported,
        [_adjudication(relationship_id="rel-missing", verdict="unsupported")],
    )
    assert missing["unmatched_adjudication_ids"] == ["rel-missing"]
    assert missing["accept_precision"] is None
    try:
        load_adjudications([{"relationship_id": "rel-1", "verdict": "maybe", "adjudicator": "example", "timestamp": "t", "reason": "r"}])
    except GateEvaluationError:
        pass
    else:
        raise AssertionError("an unknown verdict should be refused")
