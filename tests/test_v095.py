"""0.9.5 review fixes. No network and no paid model calls."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from automated_grounded_theory import __version__, theoretical_sample
from automated_grounded_theory.records import POLICY_VERSION
from tests.test_v080 import _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs, _payload


def test_version_is_095():
    assert __version__ == "0.9.35"
    assert POLICY_VERSION == "0.9.35"


class _SecondRereadTimeout(FakeSamplingCoder):
    rereads: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "SamplingRereadNote":
                self.rereads += 1
                if self.rereads == 2:
                    raise TimeoutError("timed out")
                parsed = schema(
                    passage_id=str(body.get("passage_id") or ""),
                    note="A note about this passage.",
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


def test_a_later_reread_timeout_keeps_the_first_note_and_resume_skips_it():
    saver = InMemorySaver()
    coder = _SecondRereadTimeout()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=4)
    paused = theoretical_sample(_corpus(), coder, checkpointer=saver, thread_id="reread-timeout", **common)
    ids = [str(row["direction_id"]) for row in paused.directions[:3]]
    assert len(ids) >= 3
    passage = "p-low"
    assigned = {
        "directions": [
            {"direction_id": direction_id, "action": "accept", "sampled_passage_ids": [passage]}
            for direction_id in ids
        ]
    }
    partial = theoretical_sample(
        _corpus(),
        coder,
        checkpointer=saver,
        thread_id="reread-timeout",
        resuming=True,
        resume=assigned,
        **common,
    )
    assert partial.metadata["status"] == "incomplete"
    assert partial.metadata["stop"] == "timeout"
    assert partial.resumable is True
    assert len(partial.reread_notes) == 1
    assert partial.reread_notes[0]["direction_id"] == ids[0]
    assert partial.reread_notes[0]["passage_id"] == passage
    assert partial.metadata["provider_budget"]["calls_used"] == 2
    assert {(row["direction_id"], row["passage_id"]) for row in partial.metadata["reread_completed_pairs"]} == {(ids[0], passage)}
    assert {(row["direction_id"], row["passage_id"]) for row in partial.metadata["reread_failed_pairs"]} == {(ids[1], passage)}
    assert {(row["direction_id"], row["passage_id"]) for row in partial.metadata["reread_unattempted_pairs"]} == {(ids[2], passage)}
    validated = [
        row for row in partial.prompt_log if row.get("kind") == "sampling_reread" and row.get("status") == "validated"
    ]
    timed_out = [
        row for row in partial.prompt_log if row.get("kind") == "sampling_reread" and row.get("status") == "timeout"
    ]
    assert len(validated) == 1
    assert len(timed_out) == 1
    assert ids[0] in json.dumps(validated)
    assert "usage" not in timed_out[0]
    assert any(row.get("status") == "timeout" for row in partial.prompt_log)
    spent = partial.metadata["provider_budget"]["calls_used"]
    resumed = theoretical_sample(
        _corpus(),
        coder,
        checkpointer=saver,
        thread_id="reread-timeout",
        resuming=True,
        resume={"continue": True},
        **common,
    )
    notes = [row for row in resumed.reread_notes if row["passage_id"] == passage]
    assert {row["direction_id"] for row in notes} == set(ids)
    assert len(notes) == 3
    assert [row["direction_id"] for row in notes].count(ids[0]) == 1
    assert resumed.metadata["provider_budget"]["calls_used"] == spent + 2
    assert coder.rereads == 4


class _SelectionCoder(FakeSamplingCoder):
    """Returns an explicit selection. Flat lists follow ``selection``."""

    selection: str = "pairs"
    chosen: dict = Field(default_factory=dict)

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _ids(body):
            directions = [str(row.get("direction_id") or "") for row in (body.get("directions") or []) if row.get("direction_id")]
            passages = []
            for group in body.get("passage_proposals") or []:
                for row in group.get("proposals") or []:
                    passage_id = str((row or {}).get("passage_id") or "")
                    if passage_id and passage_id not in passages:
                        passages.append(passage_id)
            return directions, passages

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "ModelSamplingSelection":
                directions, passages = _ids(body)
                d0, d1 = directions[0], directions[1]
                p0 = passages[0]
                p1 = passages[1] if len(passages) > 1 else passages[0]
                self.chosen = {"d0": d0, "d1": d1, "p0": p0, "p1": p1}
                if self.selection == "pairs":
                    parsed = schema(
                        selections=[{"direction_id": d0, "passage_id": p0}],
                        rationale="The pair names the reread. The flat lists were not sent.",
                    )
                elif self.selection == "inconsistent":
                    parsed = schema(
                        selections=[{"direction_id": d0, "passage_id": p0}],
                        direction_ids=[d1],
                        passage_ids=[p1],
                        rationale="The flat lists name a different direction.",
                    )
                elif self.selection == "unknown":
                    parsed = schema(
                        selections=[{"direction_id": "no-such-direction", "passage_id": "no-such-passage"}],
                        rationale="These ids are not in the corpus.",
                    )
                else:
                    parsed = schema(
                        selections=[
                            {"direction_id": d0, "passage_id": p0},
                            {"direction_id": "no-such-direction", "passage_id": p0},
                        ],
                        rationale="One pair is in the corpus and one direction is not.",
                    )
            elif name == "SamplingRereadNote":
                parsed = schema(
                    passage_id=str(body.get("passage_id") or ""),
                    note="A note about this passage.",
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


def test_pair_only_selections_are_reread_and_the_stage_completes():
    coder = _SelectionCoder(selection="pairs")
    result = theoretical_sample(
        _corpus(),
        coder,
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        **_kwargs(),
    )
    choice = coder.chosen
    assert result.metadata["status"] == "complete"
    assert result.metadata["model_sampling_pairs"] == [{"direction_id": choice["d0"], "passage_id": choice["p0"]}]
    assert result.metadata["model_direction_ids"] == [choice["d0"]]
    assert result.metadata["model_passage_ids"] == [choice["p0"]]
    assert result.metadata["model_rejected_selections"] == []
    assert result.metadata["model_flat_list_mismatch"] is None
    assert not any("not paired" in str(row) for row in result.warnings)
    notes = [row for row in result.reread_notes if row["passage_id"] == choice["p0"]]
    assert len(notes) == 1
    assert notes[0]["direction_id"] == choice["d0"]


def test_inconsistent_flat_lists_keep_the_pairs_and_stay_incomplete():
    coder = _SelectionCoder(selection="inconsistent")
    result = theoretical_sample(
        _corpus(),
        coder,
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        **_kwargs(),
    )
    choice = coder.chosen
    assert choice["d0"] != choice["d1"]
    assert result.metadata["status"] != "complete"
    assert result.metadata["model_sampling_status"] == "incomplete"
    assert result.metadata["model_sampling_pairs"] == [{"direction_id": choice["d0"], "passage_id": choice["p0"]}]
    assert result.metadata["model_direction_ids"] == [choice["d0"]]
    assert result.metadata["model_flat_list_mismatch"]["wins"] == "selections"
    assert choice["d1"] in result.metadata["model_flat_list_mismatch"]["direction_ids"]
    assert any("flat id lists" in str(row) for row in result.warnings)
    notes = [row["direction_id"] for row in result.reread_notes]
    assert notes == [choice["d0"]]
    assert choice["d1"] not in notes


def test_unknown_selection_ids_are_rejected_and_the_stage_is_incomplete():
    unknown = _SelectionCoder(selection="unknown")
    rejected = theoretical_sample(
        _corpus(),
        unknown,
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        **_kwargs(),
    )
    assert rejected.metadata["status"] != "complete"
    assert rejected.metadata["model_sampling_status"] == "incomplete"
    assert rejected.reread_notes == []
    rows = rejected.metadata["model_rejected_selections"]
    assert len(rows) == 1
    assert rows[0]["direction_id"] == "no-such-direction"
    assert rows[0]["passage_id"] == "no-such-passage"
    assert "unknown_direction" in rows[0]["reasons"]
    assert "unknown_passage" in rows[0]["reasons"]
    assert any("rejected" in str(row) for row in rejected.warnings)

    mixed_coder = _SelectionCoder(selection="mixed")
    mixed = theoretical_sample(
        _corpus(),
        mixed_coder,
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        **_kwargs(),
    )
    choice = mixed_coder.chosen
    assert mixed.metadata["status"] != "complete"
    kept = mixed.metadata["model_sampling_pairs"]
    assert kept == [{"direction_id": choice["d0"], "passage_id": choice["p0"]}]
    unknown_rows = mixed.metadata["model_rejected_selections"]
    assert len(unknown_rows) == 1
    assert unknown_rows[0]["direction_id"] == "no-such-direction"
    assert "unknown_direction" in unknown_rows[0]["reasons"]
    assert [row["direction_id"] for row in mixed.reread_notes] == [choice["d0"]]


class _RepairTimeout(FakeSamplingCoder):
    """First selection does not parse. The repair times out. The next repair parses."""

    selection_calls: int = 0

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "ModelSamplingSelection":
                self.selection_calls += 1
                if self.selection_calls == 1:
                    raw = AIMessage(
                        content="not json",
                        usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                    )
                    return {"raw": raw, "parsed": None, "parsing_error": "malformed"}
                if self.selection_calls == 2:
                    raise TimeoutError("repair timed out")
                direction = str((body.get("directions") or [{}])[0].get("direction_id") or "")
                groups = body.get("passage_proposals") or [{}]
                proposals = (groups[0] or {}).get("proposals") or [{}]
                passage = str((proposals[0] or {}).get("passage_id") or "")
                parsed = schema(
                    selections=[{"direction_id": direction, "passage_id": passage}],
                    rationale="The repaired selection names one pair.",
                )
            elif name == "ModelSaturationJudgment":
                parsed = schema(
                    assessment="not_saturated",
                    rationale="The window is interpretable and a sampling gap is still open.",
                )
            elif name == "SamplingRereadNote":
                parsed = schema(passage_id=str(body.get("passage_id") or ""), note="A note.", quote="")
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


def test_a_selection_repair_timeout_keeps_the_first_response():
    saver = InMemorySaver()
    coder = _RepairTimeout()
    common = _kwargs(max_decision_calls=6)
    partial = theoretical_sample(
        _corpus(),
        coder,
        decision_mode="llm",
        pursuit="brief_only",
        interrupt=False,
        checkpointer=saver,
        thread_id="repair-timeout",
        **common,
    )
    assert partial.metadata["status"] == "incomplete"
    assert partial.metadata["stop"] == "timeout"
    assert partial.resumable is True
    assert partial.metadata["selection_repair_pending"] is True
    assert coder.selection_calls == 2
    assert partial.metadata["provider_budget"]["calls_used"] == 2
    failed = [row for row in partial.prompt_log if row.get("status") == "parse_failed"]
    assert len(failed) == 1
    assert any(int((row.get("usage") or {}).get("total_tokens") or 0) > 0 for row in partial.metadata["sampling_attempts"])
    timeouts = [row for row in partial.prompt_log if row.get("status") == "timeout"]
    assert timeouts
    assert all(row.get("kind") == "provider" for row in timeouts)
    assert partial.metadata["reread_interpretations"]["reread_expenditure_calls"] == 0
    resumed = theoretical_sample(
        _corpus(),
        coder,
        decision_mode="llm",
        pursuit="brief_only",
        interrupt=False,
        checkpointer=saver,
        thread_id="repair-timeout",
        resuming=True,
        resume={"continue": True},
        **common,
    )
    assert coder.selection_calls == 3
    assert resumed.metadata["provider_budget"]["calls_used"] == 3
    assert resumed.metadata["selection_repair_pending"] is False
    assert resumed.metadata["model_sampling_pairs"]
    assert len([row for row in resumed.prompt_log if row.get("status") == "parse_failed"]) == 1
    assert resumed.metadata["status"] == "complete"
