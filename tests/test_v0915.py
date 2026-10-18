"""0.9.15 regressions. Fake models only. No live evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from automated_grounded_theory import axial_code, selective_code
from automated_grounded_theory.axial import AxialProposalBatch, AxialRelationshipDraft, export_axial_state
from automated_grounded_theory.graph_grounding import graph_supported
from tests.test_v080 import QUOTE, FakeAxialCoder, _corpus, _payload


_COMMON = dict(
    seed=3,
    max_axial_review_calls=2,
    axial_share_propose=1,
    axial_share_counterexample=0,
    batch_size=1,
    interrupt=False,
)


def _texts(messages) -> str:
    parts = []
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


class _RepairCoder(FakeAxialCoder):
    """First proposal is raw JSON with a blank statement. The repair is one call."""

    outcome: str = "verbatim"

    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            name = getattr(schema, "__name__", "")
            if name != "AxialProposalBatch":
                raise AssertionError(name)
            self.call_count += 1
            body = _payload(messages)
            pair = (body.get("pairs") or [{}])[0]
            from_id = str(pair.get("from_id") or "c-prep")
            to_id = str(pair.get("to_id") or "c-return")
            if "REPAIR THIS OUTPUT" not in _texts(messages):
                raw = AIMessage(
                    content=json.dumps(
                        {
                            "relationships": [
                                {
                                    "from_id": from_id,
                                    "to_id": to_id,
                                    "statement": "   ",
                                    "passage_id": "p1",
                                    "quote": QUOTE,
                                }
                            ]
                        }
                    )
                )
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("blank statement")}
            if self.outcome == "still_blank":
                raw = AIMessage(
                    content=json.dumps(
                        {
                            "relationships": [
                                {
                                    "from_id": from_id,
                                    "to_id": to_id,
                                    "statement": "",
                                    "passage_id": "p1",
                                    "quote": QUOTE,
                                }
                            ]
                        }
                    )
                )
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("blank statement")}
            if self.outcome == "explicit_none":
                parsed = AxialProposalBatch(relationships=[], nothing_to_propose=True)
            elif self.outcome == "verbatim":
                parsed = AxialProposalBatch(
                    relationships=[
                        AxialRelationshipDraft(
                            from_id=from_id,
                            to_id=to_id,
                            statement=QUOTE,
                            passage_id="p1",
                            quote=QUOTE,
                        )
                    ]
                )
            elif self.outcome == "stitched":
                parsed = AxialProposalBatch(
                    relationships=[
                        AxialRelationshipDraft(
                            from_id=from_id,
                            to_id=to_id,
                            statement="Packing may be related to returning.",
                            passage_id="p1",
                            quote_from="Packing the suitcase took all night",
                            quote_to="The return to work after leave",
                        )
                    ]
                )
            elif self.outcome == "unanchored":
                parsed = AxialProposalBatch(
                    relationships=[
                        AxialRelationshipDraft(
                            from_id=from_id,
                            to_id=to_id,
                            statement="The schedule may be related to the meeting.",
                            passage_id="p1",
                            quote="this quotation is not in the passage",
                        )
                    ]
                )
            else:
                parsed = AxialProposalBatch(
                    relationships=[
                        AxialRelationshipDraft(
                            from_id=from_id,
                            to_id=to_id,
                            statement="Packing may be related to the trip.",
                            passage_id="p1",
                            quote=QUOTE,
                        )
                    ]
                )
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def _repair_row(result):
    rows = [row for row in result.prompt_log if row.get("kind") == "schema_repair"]
    assert len(rows) == 1
    blob = json.dumps(rows[0]["messages"])
    assert "REPAIR THIS OUTPUT" in blob
    assert "Packing the suitcase took all night before the trip." in blob
    return rows[0]


def test_statement_schema_rejects_blanks_and_one_repair_stays_grounded():
    """A blank statement fails the schema. One repair is logged against the original evidence.

    A recovered statement that is only a quote is not a relationship. A recovered
    claim still passes the passage and stitching checks.
    """
    with pytest.raises(ValidationError):
        AxialRelationshipDraft()
    with pytest.raises(ValidationError):
        AxialRelationshipDraft(statement="   ", from_id="c-prep", to_id="c-return")
    empty = AxialProposalBatch(nothing_to_propose=True, relationships=[])
    assert empty.nothing_to_propose is True
    assert empty.relationships == []

    verbatim = _RepairCoder(outcome="verbatim")
    quoted = axial_code(_corpus(), verbatim, **_COMMON)
    assert verbatim.call_count == 2
    assert quoted.relationships == []
    assert quoted.metadata["proposal_diagnostics"]["schema_repairs"] == 1
    assert quoted.metadata["proposal_diagnostics"]["rejection_reasons"]["verbatim_quote_statement"] == 1
    _repair_row(quoted)

    grounded = _RepairCoder(outcome="grounded")
    kept = axial_code(_corpus(), grounded, **_COMMON)
    assert grounded.call_count == 2
    assert len(kept.relationships) == 1
    assert kept.relationships[0]["statement"] == "Packing may be related to the trip."
    assert kept.relationships[0]["endpoint_anchoring"]
    assert "verbatim_quote_statement" not in kept.metadata["proposal_diagnostics"]["rejection_reasons"]

    stitched = _RepairCoder(outcome="stitched")
    dropped = axial_code(_corpus(), stitched, **_COMMON)
    assert stitched.call_count == 2
    assert dropped.relationships == []
    assert dropped.metadata["proposal_diagnostics"]["rejection_reasons"]["stitched_unrelated_passages"] == 1

    loose = _RepairCoder(outcome="unanchored")
    held = axial_code(_corpus(), loose, **_COMMON)
    assert len(held.relationships) == 1
    assert held.relationships[0]["evidence_condition"] == "unanchored"
    assert held.relationships[0]["downstream_eligible"] is False

    explicit = _RepairCoder(outcome="explicit_none")
    none = axial_code(_corpus(), explicit, **_COMMON)
    assert explicit.call_count == 2
    assert none.relationships == []
    assert none.metadata["proposal_diagnostics"]["explicit_none"] == 1

    failed = _RepairCoder(outcome="still_blank")
    unparsed = axial_code(_corpus(), failed, **_COMMON)
    assert failed.call_count == 2
    assert unparsed.relationships == []
    assert unparsed.metadata["proposal_diagnostics"]["schema_repairs"] == 1
    assert unparsed.metadata.get("axial_stop") == "schema_unparsed"
    repair_rows = [row for row in unparsed.prompt_log if row.get("kind") == "schema_repair"]
    assert len(repair_rows) == 1
    assert repair_rows[0]["status"] == "schema_unparsed"


def test_weak_stem_cannot_reach_supported_evidence_consumers(tmp_path: Path):
    """A same-passage stem is hypothesis exploration, even if eligibility was set true.

    Core graph support ignores that stale flag. Export rewrites it. A memo and
    a storyline still receive the row as a hypothesis, not as supported evidence.
    A tier-2 pattern without a weak stem still supports the graph. Causal
    wording on that pattern withholds eligibility and does not remove graph support.
    """
    quote = "the people in my shift. management not so much"
    corpus = {
        "codebook": {
            "c-shift": {"code_id": "c-shift", "code": "shift", "definition": "shift"},
            "c-mgmt": {"code_id": "c-mgmt", "code": "management", "definition": "management"},
        },
        "passages": [{"passage_id": "p-shift", "doc_id": "d1", "respondent_id": "r1", "text": quote}],
        "segments": [
            {"segment_id": "s1", "code_id": "c-shift", "passage_id": "p-shift"},
            {"segment_id": "s2", "code_id": "c-mgmt", "passage_id": "p-shift"},
        ],
    }

    class _Stem(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or "c-shift"),
                        to_id=str(pair.get("to_id") or "c-mgmt"),
                        paradigm_role="context",
                        statement="Shift may be related to management.",
                        passage_id="p-shift",
                        quote=quote,
                    )
                ]
            )

    coded = axial_code(
        corpus,
        _Stem(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    row = coded.relationships[0]
    assert row["evidence_scope"] == "same_passage_stem_only"
    assert "weak_stem_anchor" in row["flags"]
    assert row["evidence_use"] == "hypothesis_exploration"
    assert row["downstream_eligible"] is False

    poisoned = dict(row)
    poisoned["evidence_tier"] = 2
    poisoned["evidence_label"] = "pattern_supported"
    poisoned["evidence_status"] = "pattern_supported"
    poisoned["evidence_condition"] = "anchored"
    poisoned["disposition"] = "accepted"
    poisoned["downstream_eligible"] = True
    assert graph_supported(poisoned) is False

    paths = export_axial_state(
        {"relationships": [poisoned], "metadata": {"stage": "axial"}, "codebook": corpus["codebook"]},
        tmp_path,
    )
    saved = json.loads(Path(paths["category_relationships.json"]).read_text(encoding="utf-8"))
    assert saved["relationships"][0]["downstream_eligible"] is False
    assert saved["relationships"][0]["evidence_use"] == "hypothesis_exploration"

    coded.relationships = [poisoned]
    coded.codebook = [
        {"code_id": "c-shift", "code": "shift", "definition": "shift"},
        {"code_id": "c-mgmt", "code": "management", "definition": "management"},
    ]
    coded.passages = list(corpus["passages"])
    coded.segments = list(corpus["segments"])
    chosen = selective_code(
        coded,
        FakeAxialCoder(core_id="c-shift"),
        decision_mode="llm",
        include_memos=True,
        include_storyline=True,
        interrupt=False,
        max_selective_review_calls=4,
    )
    assert chosen.metadata["core_unsupported_by_graph"] is True
    assert chosen.relationships[0]["downstream_eligible"] is False
    assert chosen.relationships[0]["evidence_use"] == "hypothesis_exploration"
    memo_prompt = next(item for item in chosen.prompt_log if item.get("kind") == "theoretical_memo")
    memo_body = json.loads(memo_prompt["messages"][-1]["content"])
    sent_ids = [item.get("relationship_id") for item in memo_body["relationships"]]
    assert row["relationship_id"] in sent_ids
    sent = next(item for item in memo_body["relationships"] if item.get("relationship_id") == row["relationship_id"])
    assert sent["downstream_eligible"] is False
    assert sent["evidence_use"] == "hypothesis_exploration"
    assert chosen.metadata["graph_snapshots"]["storyline"]["relationship_ids"] == [row["relationship_id"]]
    assert chosen.metadata["storyline_core_emphasis"] is False

    strong = {
        "relationship_id": "rel-strong",
        "from_id": "a",
        "to_id": "b",
        "statement": "The weekly meeting may be related to the written agenda.",
        "evidence_tier": 2,
        "evidence_label": "pattern_supported",
        "evidence_condition": "anchored",
        "disposition": "accepted",
        "downstream_eligible": False,
        "endpoint_anchoring": [
            {"endpoint": "from", "anchored": True, "reason": ""},
            {"endpoint": "to", "anchored": True, "reason": ""},
        ],
        "flags": [],
    }
    assert graph_supported(strong) is True
    causal = dict(strong)
    causal["relationship_id"] = "rel-causal"
    causal["flags"] = ["causal_wording_withheld", "wording_exceeds_tier"]
    assert graph_supported(causal) is True
    causal_paths = export_axial_state(
        {"relationships": [strong, causal], "metadata": {"stage": "axial"}, "codebook": {}},
        tmp_path / "strong",
    )
    exported = json.loads(Path(causal_paths["category_relationships.json"]).read_text(encoding="utf-8"))
    by_id = {item["relationship_id"]: item for item in exported["relationships"]}
    assert by_id["rel-strong"]["evidence_use"] == "supported_evidence"
    assert by_id["rel-strong"]["downstream_eligible"] is True
    assert by_id["rel-causal"]["evidence_use"] == "withheld"
    assert by_id["rel-causal"]["downstream_eligible"] is False


def test_association_may_stay_unspecified_and_role_evidence_is_not_statement_wording():
    """A causal role below tier 3 is still withheld. The sentence is a separate flag.

    unspecified is allowed when the statement is only an association.
    """
    from automated_grounded_theory.axial import _KIND_PROMPTS
    from automated_grounded_theory.gate_evaluation import export_gate_relationships
    from automated_grounded_theory.relationship_tiers import role_evidence_overreach, wording_overreach

    prompt = _KIND_PROMPTS["axial_proposal"]
    assert "only an association" in prompt
    assert "unspecified is not the default" not in prompt
    assert wording_overreach("The teams met weekly.", 2, role="consequence") is True
    assert wording_overreach("leads to", 2) is True
    assert role_evidence_overreach(2, "consequence") is True
    assert role_evidence_overreach(3, "consequence") is False

    text = "The tools for the shift sit beside coworkers on the crew tonight"
    corpus = {
        "codebook": {
            "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
            "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
        },
        "passages": [{"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": text}],
        "segments": [
            {"segment_id": "s1", "code_id": "tools", "passage_id": "p1"},
            {"segment_id": "s2", "code_id": "crew", "passage_id": "p1"},
        ],
    }

    class _Roles(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            from_id = str(pair.get("from_id") or "tools")
            to_id = str(pair.get("to_id") or "crew")
            shared = dict(
                from_id=from_id,
                to_id=to_id,
                passage_id="p1",
                quote_from="tools for the shift sit",
                quote_to="coworkers on the crew tonight",
            )
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        paradigm_role="unspecified",
                        statement="The tools may be related to the crew.",
                        **shared,
                    ),
                    AxialRelationshipDraft(
                        paradigm_role="consequence",
                        statement="The teams met weekly.",
                        **shared,
                    ),
                    AxialRelationshipDraft(
                        paradigm_role="context",
                        statement="The tools lead to the crew.",
                        **shared,
                    ),
                ]
            )

    result = axial_code(
        corpus,
        _Roles(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    by_statement = {row["statement"]: row for row in result.relationships}
    association = by_statement["The tools may be related to the crew."]
    assert association["paradigm_role"] == "unspecified"
    assert "paradigm_role_unspecified" not in association["flags"]
    assert "role_evidence_exceeds_tier" not in association["flags"]
    assert "wording_exceeds_tier" not in association["flags"]
    mild = by_statement["The teams met weekly."]
    assert "role_evidence_exceeds_tier" in mild["flags"]
    assert "wording_exceeds_tier" not in mild["flags"]
    assert "causal_wording_withheld" in mild["flags"]
    assert mild["downstream_eligible"] is False
    causal = by_statement["The tools lead to the crew."]
    assert "wording_exceeds_tier" in causal["flags"]
    assert "role_evidence_exceeds_tier" not in causal["flags"]
    assert "causal_wording_withheld" in causal["flags"]
    gate = export_gate_relationships(result.relationships)
    rules = {
        item["relationship_id"]: set(item.get("rules") or [])
        for bucket in ("accepted", "withheld", "proposed")
        for item in gate[bucket]
    }
    assert "role_evidence" in rules[mild["relationship_id"]]
    assert "consequence_wording" not in rules[mild["relationship_id"]]
    assert "consequence_wording" in rules[causal["relationship_id"]]
    assert "role_evidence" not in rules[causal["relationship_id"]]
