"""0.8.3 checkpoint budgets, stage ceilings, and stage status. No network."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, axial_code, focused_code, selective_code
from automated_grounded_theory.axial import AxialProposalBatch, _excerpt_to_budget
from automated_grounded_theory.records import POLICY_VERSION
from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _memos, _segments_for_raised, _source, _sweep
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v082 import _RepairCoder, _install_estimates


def test_version_is_083():
    assert __version__ == "0.9.1"
    assert POLICY_VERSION == "0.9.1"


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _dump_saver(saver) -> bytes:
    """Checkpoint bytes a new process can load. The saver object itself does not pickle."""
    return pickle.dumps(
        {
            "storage": _plain(saver.storage),
            "writes": _plain(saver.writes),
            "blobs": _plain(saver.blobs),
        }
    )


def _load_saver(blob: bytes):
    data = pickle.loads(blob)
    fresh = InMemorySaver()
    fresh.storage = defaultdict(
        lambda: defaultdict(dict),
        {key: defaultdict(dict, value) for key, value in data["storage"].items()},
    )
    fresh.writes = defaultdict(dict, {key: dict(value) for key, value in data["writes"].items()})
    fresh.blobs = defaultdict(dict, data["blobs"])
    return fresh


def _sized_estimate(messages, schema, tokenizer=None):
    name = getattr(schema, "__name__", "")
    if name == "AxialProposalBatch":
        return 600
    if name == "AxialCounterexampleBatch":
        return 500
    return 40


def _finish_axial(saver, thread: str, **kwargs):
    """Resume a run that is waiting on the relationship checkpoint."""
    result = axial_code(
        _corpus(),
        FakeAxialCoder(),
        resuming=True,
        resume={"accept": True},
        checkpointer=saver,
        thread_id=thread,
        interrupt=True,
        **kwargs,
    )
    for _ in range(3):
        if not result.interrupted:
            return result
        result = axial_code(
            _corpus(),
            FakeAxialCoder(),
            resuming=True,
            resume=_resume_value(result),
            checkpointer=saver,
            thread_id=thread,
            interrupt=True,
            **kwargs,
        )
    return result


def _resume_value(result) -> dict:
    kind = (result.interrupt_payload or {}).get("kind")
    if kind == "axial_relations":
        return {"accept": True}
    if kind == "thematic_saturation":
        return {"leave_saturation": True}
    if kind == "selective_core":
        return {"leave_core": True}
    raise AssertionError(kind)


def _view(result) -> dict:
    budget = result.metadata.get("provider_budget") or {}
    validated = [
        int(row.get("estimated_input_tokens") or 0)
        for row in result.prompt_log
        if row.get("status") == "validated"
    ]
    stages = {
        name: int((row or {}).get("calls_used") or 0)
        for name, row in (budget.get("stages") or {}).items()
    }
    return {
        "tokens": int(budget.get("tokens_estimated") or 0),
        "calls": int(budget.get("calls_used") or 0),
        "stages": stages,
        "validated": sum(validated),
        "status": result.metadata.get("status"),
        "stop": result.metadata.get("stop"),
    }


def test_pause_and_uninterrupted_keep_the_same_ceiling(monkeypatch, tmp_path):
    _install_estimates(monkeypatch, _sized_estimate)
    common = dict(
        seed=3,
        max_axial_review_calls=4,
        axial_share_propose=0.5,
        axial_share_counterexample=0.5,
        max_axial_review_tokens=1000,
    )
    straight = axial_code(_corpus(), FakeAxialCoder(), interrupt=False, **common)
    saver = InMemorySaver()
    paused = axial_code(
        _corpus(),
        FakeAxialCoder(),
        interrupt=True,
        checkpointer=saver,
        thread_id="budget",
        **common,
    )
    assert paused.interrupt_payload["kind"] == "axial_relations"
    assert paused.metadata["provider_budget"]["tokens_estimated"] == 600
    blob_path = tmp_path / "checkpoint.pkl"
    blob_path.write_bytes(_dump_saver(saver))
    resumed = _finish_axial(_load_saver(blob_path.read_bytes()), "budget", **common)
    assert _view(straight) == _view(resumed)
    assert _view(resumed)["tokens"] == 600
    assert _view(resumed)["stages"]["coding"] == 1
    assert _view(resumed)["status"] == "budget_limited"
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "resume_fresh.py"
    script.write_text(
        f"""
import json, sys
sys.path.insert(0, {str(root)!r})
import automated_grounded_theory.axial as axial
import automated_grounded_theory.batching as batching
from tests.test_v083 import _finish_axial, _load_saver, _sized_estimate, _view

axial.request_input_tokens = _sized_estimate
batching.request_input_tokens = _sized_estimate
saver = _load_saver(open({str(blob_path)!r}, "rb").read())
result = _finish_axial(saver, "budget", seed=3, max_axial_review_calls=4, axial_share_propose=0.5, axial_share_counterexample=0.5, max_axial_review_tokens=1000)
print(json.dumps(_view(result)))
"""
    )
    completed = subprocess.run([sys.executable, str(script)], cwd=str(root), capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == _view(straight)


def test_pause_matches_when_the_allowance_covers_both_stages(monkeypatch, tmp_path):
    _install_estimates(monkeypatch, _sized_estimate)
    common = dict(
        seed=3,
        max_axial_review_calls=4,
        axial_share_propose=0.5,
        axial_share_counterexample=0.5,
        max_axial_review_tokens=2000,
    )
    straight = axial_code(_corpus(), FakeAxialCoder(), interrupt=False, **common)
    saver = InMemorySaver()
    axial_code(_corpus(), FakeAxialCoder(), interrupt=True, checkpointer=saver, thread_id="wide", **common)
    resumed = _finish_axial(_load_saver(_dump_saver(saver)), "wide", **common)
    assert _view(straight) == _view(resumed)
    assert _view(resumed)["tokens"] == 1600
    assert _view(resumed)["stages"]["coding"] == 3
    assert _view(resumed)["status"] == "complete"


def test_review_mode_does_not_change_the_coding_ceiling():
    common = dict(
        seed=3,
        axial_share_propose=1,
        axial_share_counterexample=0,
        interrupt=False,
        max_axial_review_tokens=24000,
    )
    for ceiling in (1, 2):
        human = _RepairCoder(fail_proposals=1)
        llm = _RepairCoder(fail_proposals=1)
        human_result = axial_code(
            _corpus(), human, decision_mode="human", max_axial_review_calls=ceiling, **common
        )
        llm_result = axial_code(
            _corpus(),
            llm,
            decision_mode="llm",
            max_axial_review_calls=ceiling,
            max_decision_calls=4,
            **common,
        )
        assert human_result.metadata["provider_budget"]["stages"]["coding"]["calls_used"] == ceiling
        assert llm_result.metadata["provider_budget"]["stages"]["coding"]["calls_used"] == ceiling
        assert human.call_count == ceiling
        if ceiling == 1:
            assert llm.call_count == ceiling + 1
            assert llm_result.metadata["provider_budget"]["stages"]["review"]["calls_used"] == 1
        else:
            assert llm.call_count == ceiling + 2
            assert llm_result.metadata["provider_budget"]["stages"]["review"]["calls_used"] == 2


def test_selective_omission_and_parse_failure_are_not_complete(tmp_path):
    omitted = selective_code(
        _corpus(),
        FakeAxialCoder(),
        include_storyline=True,
        max_axial_review_tokens=0,
        interrupt=False,
        output_dir=tmp_path / "zero",
    )
    assert omitted.metadata["status"] == "budget_limited"
    assert omitted.metadata["complete"] is False
    assert omitted.metadata["stage_outcomes"]["storyline"]["status"] == "budget_limited"
    assert omitted.metadata["stage_outcomes"]["storyline"]["reason"] == "token_allowance"
    exported = json.loads((tmp_path / "zero" / "selective_run.json").read_text(encoding="utf-8"))
    assert exported["metadata"]["stage_outcomes"]["storyline"]["required"] is True

    coder = _BrokenStory()
    failed = selective_code(
        _corpus(),
        coder,
        include_storyline=True,
        include_memos=False,
        interrupt=False,
        output_dir=tmp_path / "bad",
    )
    assert coder.call_count == 2
    assert failed.metadata["status"] == "incomplete"
    assert failed.metadata["stage_outcomes"]["storyline"]["status"] == "incomplete"
    assert failed.metadata["stage_outcomes"]["storyline"]["reason"] == "schema_unparsed"
    usage = failed.metadata["provider_token_usage"]
    assert usage["input_tokens"] == 90
    assert usage["reported_calls"] == 2
    logged = [row for row in failed.prompt_log if row.get("kind") in {"storyline", "schema_repair"}]
    inputs = sorted(int((row.get("usage") or {}).get("input_tokens") or 0) for row in logged if row.get("usage"))
    assert inputs == [30, 60]
    assert any(row.get("raw_response") for row in logged)


def test_omitted_llm_review_is_budget_limited(monkeypatch):
    def estimate(messages, schema, tokenizer=None):
        name = getattr(schema, "__name__", "")
        if name == "AxialProposalBatch":
            return 1402
        if name in {"AxialRelationDecisions", "ModelSaturationDecision", "ModelCoreDecision"}:
            return 1166
        return 30

    _install_estimates(monkeypatch, estimate)
    result = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=3,
        decision_mode="llm",
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        max_decision_calls=4,
        max_axial_review_tokens=1500,
        interrupt=False,
    )
    assert result.metadata["decision_stop"] == "token_allowance"
    assert result.metadata["status"] == "budget_limited"
    assert result.metadata["complete"] is False
    assert result.metadata["stage_outcomes"]["decision_review"]["status"] == "budget_limited"
    assert result.metadata["stage_outcomes"]["proposals"]["status"] == "complete"


def test_storyline_failure_after_successful_proposals_is_incomplete(tmp_path):
    result = axial_code(
        _corpus(),
        _BrokenStory(),
        seed=3,
        also_selective=True,
        include_storyline=True,
        include_memos=False,
        interrupt=False,
        output_dir=tmp_path,
    )
    assert result.metadata["stage_outcomes"]["proposals"]["status"] == "complete"
    assert result.metadata["stage_outcomes"]["storyline"]["status"] == "incomplete"
    assert result.metadata["status"] == "incomplete"
    assert result.metadata["complete"] is False
    exported = json.loads((tmp_path / "selective_run.json").read_text(encoding="utf-8"))
    assert exported["metadata"]["stage_outcomes"]["proposals"]["status"] == "complete"
    assert exported["metadata"]["status"] == "incomplete"


def test_failed_repair_keeps_both_usages():
    result = selective_code(
        _corpus(),
        _BrokenStory(),
        include_storyline=True,
        include_memos=False,
        interrupt=False,
    )
    repair = [row for row in result.prompt_log if row.get("kind") == "schema_repair"]
    assert repair
    assert int(repair[-1]["usage"]["input_tokens"]) == 60
    assert "repair_raw_response" in json.dumps(repair[-1])
    assert result.metadata["provider_token_usage"]["input_tokens"] == 90
    assert result.metadata["provider_token_usage"]["reported_calls"] == 2


def test_evidence_free_proposal_is_deferred():
    source = _corpus()
    source["segments"] = []
    result = axial_code(
        source,
        FakeAxialCoder(),
        seed=3,
        axial_token_budget=1,
        max_axial_review_calls=2,
        interrupt=False,
    )
    assert result.relationships == []
    assert result.metadata["missing_category_evidence"]
    assert any(row.get("reason") == "missing_category_evidence" for row in result.prompt_log)
    assert result.coverage["missing_category_evidence"]
    estimate, excerpted = _excerpt_to_budget({"passages": [], "pairs": []}, AxialProposalBatch, 1)
    assert excerpted is False
    assert estimate >= 0


def test_focused_pause_keeps_the_recorded_allowance(tmp_path):
    source = _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep())
    kwargs = dict(
        seed=1,
        pins=["c-rare"],
        focused_min_score=1.0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        max_focused_review_calls=1,
    )
    straight = focused_code(source, model=FakeFocusedCoder(), interrupt=False, **kwargs)
    saver = InMemorySaver()
    resumed = focused_code(
        source,
        model=FakeFocusedCoder(),
        interrupt=True,
        checkpointer=saver,
        thread_id="focused-budget",
        **kwargs,
    )
    for _ in range(4):
        if not resumed.interrupted:
            break
        kind = (resumed.interrupt_payload or {}).get("kind")
        resume = {"accept": True} if kind == "focused_selection" else {"leave_claims": True}
        saver = _load_saver(_dump_saver(saver))
        resumed = focused_code(
            source,
            model=FakeFocusedCoder(),
            resuming=True,
            resume=resume,
            interrupt=True,
            checkpointer=saver,
            thread_id="focused-budget",
            **kwargs,
        )
    assert resumed.metadata["provider_budget"]["calls_used"] == straight.metadata["provider_budget"]["calls_used"]
    assert resumed.metadata["provider_budget"]["tokens_estimated"] == straight.metadata["provider_budget"]["tokens_estimated"]


class _BrokenStory(FakeAxialCoder):
    """Storyline responses fail to parse. The repair reports a larger usage."""

    story_calls: int = 0

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))
        parent = super().with_structured_output(schema, **kwargs)
        if getattr(schema, "__name__", "") != "StorylineBatch":
            return parent

        def _call(messages):
            self.story_calls += 1
            self.call_count += 1
            tokens = 30 if self.story_calls == 1 else 60
            raw = AIMessage(
                content=f"story-{self.story_calls}",
                usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
            )
            if include_raw:
                return {"raw": raw, "parsed": None, "parsing_error": "malformed"}
            return None

        return RunnableLambda(_call)

