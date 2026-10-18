"""0.9.0 theoretical sampling and an explicit saturation decision. No network calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import (
    __version__,
    axial_code,
    get_graph,
    stopping_criteria_report,
    theoretical_sample,
)
from automated_grounded_theory.axial import AxialError
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.sampling import GLASER_CODING_FAMILIES, SamplingError
from tests.test_v080 import FakeAxialCoder, _corpus


def test_version_is_090():
    assert __version__ == "0.9.20"
    assert POLICY_VERSION == "0.9.20"


def _payload(messages) -> dict:
    for message in reversed(list(messages)):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.startswith("{"):
            return json.loads(content)
    return {}


class FakeSamplingCoder(BaseChatModel):
    """Offline sampling responses. ``mode`` selects the fixture."""

    call_count: int = 0
    mode: str = "grounded"
    judgment_assessment: str = "not_saturated"

    @property
    def _llm_type(self) -> str:
        return "fake-sampling"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "SamplingRereadNote" and self.mode == "timeout":
                raise TimeoutError("timed out")
            if name == "ModelSamplingSelection" and self.mode == "unknown":
                parsed = schema(
                    direction_ids=["dir-nope"],
                    passage_ids=["missing-passage"],
                    rationale="These ids are not in the request.",
                )
            elif name == "ModelSamplingSelection":
                direction = str((body.get("directions") or [{}])[0].get("direction_id") or "")
                groups = body.get("passage_proposals") or [{}]
                proposals = (groups[0] or {}).get("proposals") or [{}]
                passage = str((proposals[0] or {}).get("passage_id") or "")
                parsed = schema(
                    direction_ids=[direction] if direction else [],
                    passage_ids=[passage] if passage else [],
                    rationale="The audit names this gap, and the passage is already in the corpus.",
                )
            elif name == "SamplingRereadNote":
                quote = "this sentence was never written" if self.mode == "unanchored" else str(body.get("text") or "")[:48]
                parsed = schema(passage_id=str(body.get("passage_id") or ""), note="A note about this passage.", quote=quote)
            elif name == "ModelSaturationJudgment":
                parsed = schema(
                    assessment=self.judgment_assessment,
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


def _carried_corpus():
    source = _corpus()
    source["metadata"] = {
        "provider_budget": {
            "calls_used": 2,
            "tokens_estimated": 100,
            "max_calls": None,
            "max_tokens": 24000,
            "stages": {},
            "observed_raw": 0,
            "observed_input": 0,
        }
    }
    return source


def _kwargs(**overrides):
    fields = dict(
        seed=3,
        batch_size=1,
        sampling_request_timeout=30,
        max_sampling_review_calls=4,
        max_decision_calls=4,
    )
    fields.update(overrides)
    return fields


def test_sampling_graph_is_separate_and_axial_lens_is_unchanged():
    sampling = get_graph("sampling")
    assert "rank_directions" in sampling.nodes
    assert "saturation_decision" in sampling.nodes
    assert "propose_relations" not in sampling.nodes
    with pytest.raises(AxialError, match="coding_lens"):
        axial_code(_corpus(), FakeAxialCoder(), coding_lens="glaser", interrupt=False, seed=1)
    assert "conditions" in GLASER_CODING_FAMILIES


def test_directions_are_grounded_and_glaser_family_is_only_a_label():
    result = theoretical_sample(_corpus(), FakeSamplingCoder(), coding_family="glaser", interrupt=False, **_kwargs())
    kinds = {row["kind"] for row in result.directions}
    assert "thin_category" in kinds
    assert "no_negative_case" in kinds
    prep = next(row for row in result.directions if row["kind"] == "thin_category" and row["target_id"] == "c-prep")
    assert "c-prep" in prep["why"]
    assert prep["audit"]["document_ids"] == ["d1"]
    assert prep["coding_family"] == "conditions"
    assert "Selection criteria" in prep["sampling_query"]
    assert result.saturation["judgment"] is None
    assert result.metadata["saturation_judgment"] is None
    assert result.metadata.get("saturation_decision") is None


def test_retrieval_proposes_corpus_passages_and_marks_an_empty_corpus():
    result = theoretical_sample(_corpus(), interrupt=False, **_kwargs())
    prep = next(group for group in result.passage_proposals if group["target_id"] == "c-prep")
    proposed = [row["passage_id"] for row in prep["proposals"]]
    assert "p-low" in proposed
    assert "p1" not in proposed
    assert "p-none" not in proposed
    assert prep["new_data_collection"] == "not_required"
    empty = {
        "codebook": {"c-z": {"code_id": "c-z", "code": "zzzz unique", "definition": "zzzz unique"}},
        "passages": [{"passage_id": "p-other", "doc_id": "d9", "text": "The meeting format was virtual."}],
        "segments": [],
        "properties": [],
        "dimensions": [],
        "relationships": [],
        "memos": [],
    }
    missed = theoretical_sample(empty, interrupt=False, **_kwargs())
    assert missed.passage_proposals
    assert all(group["corpus_status"] == "no_suitable_unsampled_passages" for group in missed.passage_proposals)
    assert missed.metadata["new_data_collection"] == "researcher_action"
    assert "did not collect" in missed.passage_proposals[0]["note"]


def test_human_pause_resume_keeps_the_carried_budget(tmp_path: Path):
    saver = InMemorySaver()
    coder = FakeSamplingCoder()
    common = _kwargs(pursuit="reread", output_dir=tmp_path)
    paused = theoretical_sample(
        _carried_corpus(),
        coder,
        checkpointer=saver,
        thread_id="sample-1",
        **common,
    )
    assert paused.interrupted
    assert paused.interrupt_payload["kind"] == "theoretical_sampling"
    assert paused.metadata["provider_budget"]["calls_used"] == 2
    # 0.9.10 baselines carried token estimates. calls_used stays. This stage's
    # tokens_estimated starts at 0 so upstream tokens are not charged.
    assert paused.metadata["sampling_token_baseline"]["tokens_estimated_carried"] == 100
    assert paused.metadata["provider_budget"]["tokens_estimated"] == 0
    direction_id = paused.directions[0]["direction_id"]
    mid = theoretical_sample(
        _carried_corpus(),
        coder,
        checkpointer=saver,
        thread_id="sample-1",
        resuming=True,
        resume={
            "directions": [
                {
                    "direction_id": direction_id,
                    "action": "edit",
                    "sampling_query": "Seek another suitcase passage that is not already evidence.",
                    "sampled_passage_ids": ["p-low"],
                }
            ]
        },
        **common,
    )
    assert mid.interrupted
    assert mid.interrupt_payload["kind"] == "saturation_decision"
    assert mid.interrupt_payload["sampling_ledger"]
    assert mid.interrupt_payload["stopping_criteria"]["declares_saturation"] is False
    budget = mid.metadata["provider_budget"]
    assert budget["calls_used"] == 3
    assert budget["tokens_estimated"] > 0
    assert mid.sampling_ledger[-1]["sampled_passage_ids"] == ["p-low"]
    assert mid.reread_notes
    assert mid.reread_notes[0]["quote"]
    assert "quote_unanchored" not in mid.reread_notes[0]["flags"]
    done = theoretical_sample(
        _carried_corpus(),
        coder,
        checkpointer=saver,
        thread_id="sample-1",
        resuming=True,
        resume={"decision": "not_saturated", "rationale": "The thin categories are still thin."},
        **common,
    )
    assert done.interrupted is False
    decision = done.metadata["saturation_decision"]
    assert decision["decision"] == "not_saturated"
    assert decision["actor"] == "human"
    assert "T" in decision["timestamp"]
    assert done.saturation["judgment"] is None
    assert done.metadata["saturation_judgment"] is None
    assert done.metadata.get("model_saturation_judgment") is None
    assert done.metadata["provider_budget"]["calls_used"] == 3
    exported = json.loads((tmp_path / "theoretical_sampling.json").read_text(encoding="utf-8"))
    assert exported["judgment"] is None
    assert exported["next_round"]["passage_ids"] == ["p-low"]
    assert exported["sampling_ledger"]


def test_empty_resume_is_rejected():
    saver = InMemorySaver()
    theoretical_sample(_corpus(), checkpointer=saver, thread_id="empty", **_kwargs())
    with pytest.raises(SamplingError, match="empty object"):
        theoretical_sample(_corpus(), checkpointer=saver, thread_id="empty", resuming=True, resume={}, **_kwargs())


def test_llm_selection_is_labeled_and_does_not_write_human_fields():
    coder = FakeSamplingCoder(mode="unknown")
    unknown = theoretical_sample(_corpus(), coder, decision_mode="llm", interrupt=False, **_kwargs())
    assert unknown.metadata["model_unknown_direction_ids"] == ["dir-nope"]
    assert unknown.metadata["model_unknown_passage_ids"] == ["missing-passage"]
    assert unknown.metadata["model_direction_ids"] == []
    assert unknown.metadata["model_passage_ids"] == []
    assert unknown.metadata.get("researcher_sampling") is None
    assert unknown.metadata.get("saturation_decision") is None
    judgment = unknown.metadata["model_saturation_judgment"]
    assert judgment["status"] == "not_interpretable"
    assert judgment["assessment"] is None
    assert judgment["model_judgment"] is True
    assert judgment["does_not_set_judgment"] is True
    assert unknown.saturation["judgment"] is None
    assert unknown.saturation.get("researcher_saturation") in (None, unknown.metadata.get("researcher_saturation"))
    grounded = theoretical_sample(_corpus(), FakeSamplingCoder(), decision_mode="llm", interrupt=False, **_kwargs())
    assert grounded.metadata["model_direction_ids"]
    assert grounded.metadata["model_passage_ids"]
    assert grounded.metadata.get("saturation_decision") is None
    assert grounded.metadata["model_saturation_judgment"]["assessment"] is None


def test_llm_judgment_uses_the_window_and_stays_out_of_human_fields():
    source = _corpus()
    source["saturation"] = {
        "n_documents": 3,
        "saturation_window": "interpretable",
        "window_new_code_rate": 0.25,
        "window_new_property_rate": 0.0,
        "window_new_relationship_rate": None,
        "code_frequency_shift": 0.4,
        "judgment": None,
    }
    source["metadata"] = {
        "sampling_ledger": [
            {
                "round_index": 1,
                "direction_keys": ["thin_category:c-prep"],
                "new_direction_keys": ["thin_category:c-prep"],
                "indicators": {"window_new_code_rate": 0.5},
            }
        ]
    }
    result = theoretical_sample(source, FakeSamplingCoder(), decision_mode="llm", interrupt=False, **_kwargs())
    judgment = result.metadata["model_saturation_judgment"]
    assert judgment["status"] == "recorded"
    assert judgment["assessment"] == "not_saturated"
    assert judgment["actor"] == "llm"
    assert result.metadata.get("saturation_decision") is None
    assert result.metadata.get("researcher_saturation") is None
    assert result.saturation["judgment"] is None
    assert "model_saturation_judgment" not in (result.metadata.get("researcher_sampling") or {})
    assert len(result.sampling_ledger) == 2
    assert result.sampling_ledger[-1]["constant_comparison"]["compared"] is True
    assert result.sampling_ledger[-1]["constant_comparison"]["window_new_code_rate_delta"] == -0.25
    assert result.sampling_ledger[-1]["constant_comparison"]["window_new_relationship_rate_delta"] is None
    assert result.sampling_ledger[-1]["model_saturation_judgment_status"] == "recorded"
    assert result.sampling_ledger[-1]["judgment"] is None
    new_codes = next(row for row in result.stopping_criteria["criteria"] if row["criterion"] == "new_codes")
    relationships = next(row for row in result.stopping_criteria["criteria"] if row["criterion"] == "new_relationships")
    assert new_codes["status"] == "unmet"
    assert relationships["status"] == "undefined"
    assert result.stopping_criteria["declares_saturation"] is False
    assert result.stopping_criteria["judgment"] is None


def test_stopping_criteria_name_thresholds_without_a_saturation_claim():
    saturation = {
        "n_documents": 3,
        "window_new_code_rate": 0.0,
        "window_new_property_rate": 0.0,
        "window_new_relationship_rate": 0.0,
        "code_frequency_shift": 0.0,
    }
    quiet = stopping_criteria_report(
        saturation,
        [
            {"new_direction_keys": [], "reread_unsent_pairs": []},
            {"new_direction_keys": [], "reread_unsent_pairs": []},
        ],
        core_id="c-prep",
        relationships=[{"from_id": "c-prep", "to_id": "c-return", "statement": "A link.", "disposition": "accepted"}],
    )
    by_name = {row["criterion"]: row for row in quiet["criteria"]}
    assert by_name["new_codes"]["status"] == "met"
    assert by_name["new_codes"]["threshold"]["window_rate_at_most"] == 0.0
    assert by_name["code_frequency_shift"]["status"] == "met"
    assert by_name["sampling_rounds_without_new_directions"]["status"] == "met"
    assert by_name["core_category_coverage"]["status"] == "met"
    assert quiet["declares_saturation"] is False
    short = stopping_criteria_report({"n_documents": 2, "window_new_code_rate": 0.0}, [])
    short_codes = next(row for row in short["criteria"] if row["criterion"] == "new_codes")
    quiet_row = next(row for row in short["criteria"] if row["criterion"] == "sampling_rounds_without_new_directions")
    core = next(row for row in short["criteria"] if row["criterion"] == "core_category_coverage")
    assert short_codes["status"] == "undefined"
    assert quiet_row["status"] == "undefined"
    assert core["status"] == "undefined"
    unsupported = stopping_criteria_report(saturation, [], core_id="c-missing", relationships=[])
    core_row = next(row for row in unsupported["criteria"] if row["criterion"] == "core_category_coverage")
    assert core_row["status"] == "unmet"


def test_timeout_keeps_the_round_and_the_spent_allowance(tmp_path: Path):
    coder = FakeSamplingCoder(mode="timeout")
    result = theoretical_sample(
        _corpus(),
        coder,
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        output_dir=tmp_path,
        **_kwargs(),
    )
    assert result.metadata["status"] == "incomplete"
    assert result.metadata["stop"] == "timeout"
    assert result.directions
    assert result.sampling_ledger
    assert result.sampling_ledger[0]["round_index"] == 1
    assert result.metadata["provider_budget"]["calls_used"] >= 2
    assert result.saturation["judgment"] is None
    exported = json.loads((tmp_path / "theoretical_sampling.json").read_text(encoding="utf-8"))
    assert exported["sampling_ledger"]
    assert exported["judgment"] is None


def test_unanchored_reread_quote_is_not_stored():
    result = theoretical_sample(
        _corpus(),
        FakeSamplingCoder(mode="unanchored"),
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        **_kwargs(),
    )
    assert result.reread_notes
    assert result.reread_notes[0]["quote"] == ""
    assert "quote_unanchored" in result.reread_notes[0]["flags"]
    assert result.metadata["reread_notes_do_not_change_saturation_indicators"] is False
    assert result.sampling_ledger[-1]["reread_notes"]


def test_later_round_appends_the_ledger():
    first = theoretical_sample(_corpus(), interrupt=False, **_kwargs())
    second = theoretical_sample(first, interrupt=False, **_kwargs())
    assert len(first.sampling_ledger) == 1
    assert len(second.sampling_ledger) == 2
    assert second.sampling_ledger[0]["round_index"] == 1
    assert second.sampling_ledger[1]["round_index"] == 2
    assert second.sampling_ledger[1]["constant_comparison"]["compared"] is True
    assert second.sampling_ledger[0]["constant_comparison"]["window_new_code_rate_delta"] is None
