"""0.9.16 review fixes. Fakes only. No network calls."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import axial_code, theoretical_sample
from automated_grounded_theory.axial import (
    AxialCaseQuote,
    AxialProposalBatch,
    AxialRelationDecision,
    AxialRelationDecisions,
    AxialRelationshipDraft,
)
from automated_grounded_theory.backends import InstallationMismatch, installation_record
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v090 import FakeSamplingCoder, _payload


def _axial(**overrides):
    fields = dict(
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    fields.update(overrides)
    return fields


class _Dist:
    def __init__(self, version: str, root: Path, direct_url: str | None = None):
        self.version = version
        self._root = root
        self._direct = direct_url

    def locate_file(self, rel: str):
        return self._root / rel

    def read_text(self, name: str) -> str:
        if name == "direct_url.json" and self._direct is not None:
            return self._direct
        raise FileNotFoundError(name)


def test_editable_install_is_checked_before_a_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An editable checkout is consistent from another working directory.

    Metadata may live in site-packages while the import comes from the
    checkout. A different distribution version still fails before the coder runs.
    """
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        coder = FakeAxialCoder()
        result = axial_code(_corpus(), coder, **_axial())
    finally:
        os.chdir(previous)
    assert coder.call_count >= 1
    provenance = result.metadata["provenance"]
    assert provenance["module_version"] == provenance["distribution_version"] == __version__
    assert provenance["layout"] in {"editable", "distribution"}
    assert Path(provenance["module_path"]).is_file()

    imported = Path(provenance["module_path"])
    project = imported.parents[1]
    site = tmp_path / "site-packages"
    site.mkdir()
    direct = json.dumps({"dir_info": {"editable": True}, "url": project.as_uri()})
    monkeypatch.setattr(
        "automated_grounded_theory.backends.importlib.metadata.distribution",
        lambda name: _Dist(__version__, site, direct),
    )
    record = installation_record()
    assert record["layout"] == "editable"
    assert Path(record["source_root"]) == project
    editable = axial_code(_corpus(), FakeAxialCoder(), **_axial())
    assert editable.metadata["provenance"]["layout"] == "editable"

    other = tmp_path / "other-checkout"
    other.mkdir()
    named = json.dumps({"dir_info": {"editable": True}, "url": other.as_uri()})
    monkeypatch.setattr(
        "automated_grounded_theory.backends.importlib.metadata.distribution",
        lambda name: _Dist(__version__, site, named),
    )
    with pytest.raises(InstallationMismatch, match="editable metadata"):
        axial_code(_corpus(), FakeAxialCoder(), **_axial())

    mismatch = FakeAxialCoder()
    monkeypatch.setattr(
        "automated_grounded_theory.backends.importlib.metadata.distribution",
        lambda name: _Dist("9.9.9", site, direct),
    )
    with pytest.raises(InstallationMismatch, match="9.9.9"):
        axial_code(_corpus(), mismatch, **_axial())
    assert mismatch.call_count == 0


def _sampling_visit_allowance(calls: int) -> int:
    """The same transitions ``_sampling_recursion_limit`` counts."""
    return 5 + 1 + (calls * 3) + 1 + 3


class _EveryPassage(FakeSamplingCoder):
    """Select every proposal on the thin category, then reread each one."""

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "ModelSamplingSelection":
                selections = []
                for row in body.get("directions") or []:
                    if str(row.get("kind") or "") != "thin_category":
                        continue
                    direction_id = str(row.get("direction_id") or "")
                    for group in body.get("passage_proposals") or []:
                        if str(group.get("direction_id") or "") != direction_id:
                            continue
                        for proposal in group.get("proposals") or []:
                            selections.append(
                                {
                                    "direction_id": direction_id,
                                    "passage_id": str(proposal.get("passage_id") or ""),
                                }
                            )
                parsed = schema(
                    selections=selections,
                    rationale="Every retrieved passage for the thin category is reread.",
                )
            elif name == "SamplingRereadNote":
                passage_id = str(body.get("passage_id") or "")
                text = str(body.get("text") or "")
                parsed = schema(
                    passage_id=passage_id,
                    judgment="relevant",
                    note="This passage bears on the thin category.",
                    quote=text[:48],
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
            from langchain_core.messages import AIMessage

            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_call)


def test_default_recursion_limit_finishes_thirty_two_rereads():
    """Thirty-two rereads need a visit to arm each pair and a visit to send it.

    A limit of the call ceiling plus a fixed overhead stops before the last
    requests. The default limit is the graph's own transitions.
    """
    stem = "zephyrcrate"
    passages = [
        {
            "passage_id": f"extra-{index:02d}",
            "doc_id": f"d-{index:02d}",
            "respondent_id": f"r-{index:02d}",
            "text": f"Another {stem} account that was not coded yet.",
        }
        for index in range(32)
    ]
    corpus = {
        "codebook": {
            "c-pack": {
                "code_id": "c-pack",
                "code": stem,
                "definition": f"A {stem} that still needs another document.",
            }
        },
        "passages": [
            {
                "passage_id": "coded",
                "doc_id": "d-coded",
                "respondent_id": "r-coded",
                "text": f"The coded {stem} is already evidence.",
            },
            *passages,
        ],
        "segments": [{"segment_id": "s-coded", "code_id": "c-pack", "passage_id": "coded"}],
        "properties": [],
        "dimensions": [],
        "relationships": [],
        "memos": [],
    }
    calls = 32
    result = theoretical_sample(
        corpus,
        _EveryPassage(),
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        seed=3,
        evidence_limit=32,
        category_documents=2,
        max_sampling_review_calls=calls,
        max_decision_calls=4,
        sampling_token_budget=200000,
        max_sampling_review_tokens=200000,
        sampling_request_timeout=30,
    )
    assert result.metadata.get("stop") != "recursion_limit"
    assert result.metadata["performance"]["recursion_limit"] >= _sampling_visit_allowance(calls)
    completed = {row["passage_id"] for row in result.reread_notes}
    assert completed == {row["passage_id"] for row in passages}
    assert len(result.reread_notes) == calls


_SHARED = "The tools for the shift sit beside coworkers on the crew tonight"
_CAUSAL = "The tools for the shift lead to coworkers on the crew."
_ASSOCIATION = "The tools for the shift sit beside coworkers on the crew."


def _tier2_corpus() -> dict:
    return {
        "codebook": {
            "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
            "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
        },
        "passages": [
            {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": _SHARED},
            {"passage_id": "p2", "doc_id": "d2", "respondent_id": "r2", "text": _SHARED},
        ],
        "segments": [
            {"segment_id": "s1", "code_id": "tools", "passage_id": "p1", "quote": "tools for the shift sit"},
            {"segment_id": "s2", "code_id": "crew", "passage_id": "p1", "quote": "coworkers on the crew tonight"},
        ],
        "properties": [],
        "dimensions": [],
        "memos": [],
        "relationships": [],
    }


def _case_quotes() -> list[AxialCaseQuote]:
    return [
        AxialCaseQuote(passage_id="p1", quote="tools for the shift sit", endpoint="from"),
        AxialCaseQuote(passage_id="p1", quote="coworkers on the crew tonight", endpoint="to"),
        AxialCaseQuote(passage_id="p2", quote="tools for the shift sit", endpoint="from"),
        AxialCaseQuote(passage_id="p2", quote="coworkers on the crew tonight", endpoint="to"),
    ]


class _Tier2Coder(FakeAxialCoder):
    """One anchored tier-2 proposal. ``review_role`` is what the reviewer returns."""

    proposal_role: str = "consequence"
    proposal_statement: str = _CAUSAL
    review_role: str = "context"

    def proposals(self, body: dict) -> AxialProposalBatch:
        del body
        return AxialProposalBatch(
            relationships=[
                AxialRelationshipDraft(
                    from_id="tools",
                    to_id="crew",
                    paradigm_role=self.proposal_role,  # type: ignore[arg-type]
                    statement=self.proposal_statement,
                    passage_id="p1",
                    quote="tools for the shift sit",
                    quote_from="tools for the shift sit",
                    quote_to="coworkers on the crew tonight",
                    case_quotes=_case_quotes(),
                )
            ]
        )

    def relation_review(self, body: dict):
        return AxialRelationDecisions(
            relationships=[
                AxialRelationDecision(
                    relationship_id=str(row.get("relationship_id") or ""),
                    disposition="accepted",
                    paradigm_role=self.review_role,  # type: ignore[arg-type]
                    rationale="The two cases support this role.",
                )
                for row in body.get("relationships") or []
            ]
        )


def test_human_correction_clears_causal_withholding():
    """A human can clear role and wording flags that the proposal earned.

    The model sentence stays on the row. Validation reads the corrected
    sentence and the corrected role, and keeps the old flags in the history.
    """
    saver = InMemorySaver()
    common = _axial(interrupt=True)
    paused = axial_code(_tier2_corpus(), _Tier2Coder(), checkpointer=saver, thread_id="clear-role", **common)
    proposed = paused.relationships[0]
    assert proposed["evidence_tier"] == 2
    assert proposed["model_statement"] == _CAUSAL
    for flag in ("wording_exceeds_tier", "role_evidence_exceeds_tier", "causal_wording_withheld"):
        assert flag in proposed["flags"]
    assert proposed["downstream_eligible"] is False

    done = axial_code(
        _tier2_corpus(),
        _Tier2Coder(),
        checkpointer=saver,
        thread_id="clear-role",
        resuming=True,
        resume={
            "relationships": [
                {
                    "relationship_id": proposed["relationship_id"],
                    "action": "accept",
                    "paradigm_role": "unspecified",
                    "statement": _ASSOCIATION,
                }
            ]
        },
        **_axial(interrupt=False),
    )
    corrected = done.relationships[0]
    assert corrected["human_paradigm_role"] == "unspecified"
    assert corrected["human_statement"] == _ASSOCIATION
    assert corrected["model_statement"] == _CAUSAL
    assert corrected["evidence_tier"] == 2
    for flag in ("wording_exceeds_tier", "role_evidence_exceeds_tier", "causal_wording_withheld"):
        assert flag not in corrected["flags"]
    assert corrected["downstream_eligible"] is True
    history = corrected["validation_revision_history"]
    assert history
    assert set(history[0]["flags"]) >= {"role_evidence_exceeds_tier", "causal_wording_withheld"}
    assert history[0]["role"] == "consequence"
    assert "lead to" in str(history[0]["statement"])


def test_reviewer_role_overrides_until_a_human_sets_one():
    """An LLM reviewer's role is the working role. It is not only a proposal.

    Actor fields keep the original proposal, the reviewer, and the human.
    Validation, eligibility, and validated_decision read the working role.
    A human role still wins once it is set.
    """
    reviewed = axial_code(
        _tier2_corpus(),
        _Tier2Coder(proposal_role="unspecified", proposal_statement=_ASSOCIATION, review_role="consequence"),
        decision_mode="llm",
        max_decision_calls=2,
        **_axial(),
    )
    row = reviewed.relationships[0]
    assert row["evidence_tier"] == 2
    assert row["proposal_paradigm_role"] == "unspecified"
    assert row["model_paradigm_role"] == "consequence"
    assert row["reviewer_paradigm_role"] == "consequence"
    assert row["human_paradigm_role"] is None
    assert row["paradigm_role"] == "consequence"
    assert "role_evidence_exceeds_tier" in row["flags"]
    assert "causal_wording_withheld" in row["flags"]
    assert "wording_exceeds_tier" not in row["flags"]
    assert row["downstream_eligible"] is False
    recorded = reviewed.metadata["validated_decision"]["relationships"][0]
    assert recorded["paradigm_role"] == "consequence"
    assert recorded["proposal_paradigm_role"] == "unspecified"
    assert recorded["reviewer_paradigm_role"] == "consequence"
    assert recorded["human_paradigm_role"] is None


def _message_text(messages) -> str:
    parts = []
    for message in messages:
        content = getattr(message, "content", "")
        parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(parts)


class _RawBatch(FakeAxialCoder):
    """Return raw JSON for the first proposal. Later calls use the parent fixture."""

    payload: dict = {}

    def with_structured_output(self, schema, **kwargs):
        parent = FakeAxialCoder.with_structured_output(self, schema, **kwargs)

        def _call(messages):
            name = getattr(schema, "__name__", "")
            if name == "AxialProposalBatch" and "REPAIR THIS OUTPUT" not in _message_text(messages):
                self.call_count += 1
                from langchain_core.messages import AIMessage

                raw = AIMessage(
                    content=json.dumps(self.payload),
                    usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                )
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("batch failed")}
            return parent.invoke(messages)

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_call)


def test_mixed_batch_keeps_valid_relationships_and_logs_the_rest():
    """One invalid row does not discard the valid rows or spend a repair.

    A repair can replace the whole batch, including rows that already parsed.
    A batch with nothing valid still uses that one repair. Each dropped row
    keeps its index, reason, error, and draft.
    """
    kept_statement = "Packing the suitcase before departure sits with the return to work."
    mixed = _RawBatch(
        payload={
            "relationships": [
                {
                    "from_id": "c-prep",
                    "to_id": "c-return",
                    "paradigm_role": "unspecified",
                    "statement": kept_statement,
                    "passage_id": "p1",
                    "quote": "Packing the suitcase took all night",
                },
                {
                    "from_id": "c-prep",
                    "to_id": "c-return",
                    "paradigm_role": "context",
                    "statement": "   ",
                    "passage_id": "p1",
                    "quote": "Packing the suitcase took all night",
                },
                {
                    "from_id": "c-prep",
                    "to_id": "c-return",
                    "paradigm_role": "not-a-role",
                    "statement": "This role is not in the schema.",
                    "passage_id": "p1",
                    "quote": "Packing the suitcase took all night",
                },
            ]
        }
    )
    result = axial_code(_corpus(), mixed, **_axial())
    assert [row["statement"] for row in result.relationships] == [kept_statement]
    diagnostics = result.metadata["proposal_diagnostics"]
    assert int(diagnostics.get("schema_repairs") or 0) == 0
    assert diagnostics["partial_batch"] is True
    assert diagnostics["rejection_reasons"]["blank_statement"] == 1
    assert diagnostics["rejection_reasons"]["invalid_relationship"] == 1
    dropped = diagnostics["dropped_relationships"]
    assert [item["index"] for item in dropped] == [1, 2]
    assert [item["reason"] for item in dropped] == ["blank_statement", "invalid_relationship"]
    assert "blank statement" in dropped[0]["error"].casefold()
    assert dropped[0]["draft"]["statement"] == "   "
    assert dropped[1]["draft"]["paradigm_role"] == "not-a-role"
    assert mixed.call_count == 1

    blank = _RawBatch(
        payload={
            "relationships": [
                {
                    "from_id": "c-prep",
                    "to_id": "c-return",
                    "statement": "   ",
                    "passage_id": "p1",
                    "quote": "Packing the suitcase took all night",
                }
            ]
        }
    )
    repaired = axial_code(_corpus(), blank, **_axial(max_axial_review_calls=2))
    assert int(repaired.metadata["proposal_diagnostics"].get("schema_repairs") or 0) == 1
    assert blank.call_count == 2
    assert repaired.relationships
