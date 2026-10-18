"""0.9.15 regressions. Fake models only. No live evaluation."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from automated_grounded_theory import axial_code
from automated_grounded_theory.axial import AxialProposalBatch, AxialRelationshipDraft
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
