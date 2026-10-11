"""Live-run fixes: deadlines, sweep decisions, theme caps, memos, in-vivo prompts.

No provider and no API key. Timeouts are raised by the fake model or by a
per-call deadline around a sleeper.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from automated_grounded_theory import load_run, open_code
from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.backends import resolve_chat_model
from automated_grounded_theory.codebook import commit_passage, finalize_candidate_decisions
from automated_grounded_theory.graph import _append_reading_notes
from automated_grounded_theory.prompts import CODING_SYSTEM, message_text
from automated_grounded_theory.results import OpenCodeResult
from automated_grounded_theory.schemas import ThemeCoding, ThemeProposal
from automated_grounded_theory.themes import ThemeError, build_themes, collapse_themes
from automated_grounded_theory.validation import code_key, substantive_memo
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder
from tests.fake_themer import FakeThemer
from tests.test_themes import _result


class SweepTimeout(FakeCoder):
    """Codes passages, then the sweep hits a deadline."""

    def sweep(self, text: str):
        raise TimeoutError("deadline exceeded")


class TimeoutThemer(FakeThemer):
    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            self.call_count += 1
            raise TimeoutError("deadline exceeded")

        return RunnableLambda(_call)


class MergeTimeout(BaseChatModel):
    """Groups two near-duplicate themes, then the rename pass times out."""

    call_count: int = 0
    seen: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "merge-timeout"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))
        schema_name = getattr(schema, "__name__", "")

        def _call(messages):
            self.call_count += 1
            self.seen.append(schema_name)
            if schema_name == "ThemeMergePlan":
                raise TimeoutError("deadline exceeded")
            text = "\n".join(message_text(message) for message in messages)
            labels = re.findall(r"^CODE_LABEL: (.+)$", text, re.M)
            features: list[str] = []
            relationships: list[str] = []
            other: list[str] = []
            for label in labels:
                folded = label.casefold()
                if "pack" in folded:
                    features.append(label)
                elif "cry" in folded:
                    relationships.append(label)
                else:
                    other.append(label)
            themes = []
            if features:
                themes.append(
                    ThemeProposal(
                        name="Valuing workplace features",
                        definition="People value workplace features such as pay tools and the building.",
                        properties=["features"],
                        codes=features,
                    )
                )
            if relationships:
                themes.append(
                    ThemeProposal(
                        name="Valuing workplace relationships",
                        definition="People value workplace relationships such as pay tools and the building.",
                        properties=["relationships"],
                        codes=relationships,
                    )
                )
            if other:
                themes.append(
                    ThemeProposal(
                        name="Starting a separate job",
                        definition="A hiring incident with no shared workplace wording.",
                        properties=["hiring"],
                        codes=other,
                    )
                )
            coding = ThemeCoding(themes=themes, tensions=[], negative_cases=[])
            if not include_raw:
                return coding
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            )
            return {"raw": raw, "parsed": coding, "parsing_error": None}

        return RunnableLambda(_call)


class _Sleeper:
    async def ainvoke(self, payload):
        await asyncio.sleep(5)
        return "done"


class _Flaky:
    def __init__(self, exc: BaseException):
        self.exc = exc
        self.calls = 0

    async def ainvoke(self, payload):
        self.calls += 1
        if self.calls == 1:
            raise self.exc
        return "ok"


def _limiter(**kwargs) -> CallLimiter:
    defaults = dict(max_retries=1, backoff=0, max_backoff=0, request_timeout=0)
    defaults.update(kwargs)
    return CallLimiter(1, **defaults)


def test_timeout_and_connection_retry_without_aimd():
    limiter = _limiter()
    flaky = _Flaky(TimeoutError("timed out"))
    assert asyncio.run(limiter.ainvoke(flaky, {"call": "timeout"})) == "ok"
    assert flaky.calls == 2
    assert limiter.timeout_retries >= 1
    assert limiter.limit == 1

    limiter = _limiter()
    flaky = _Flaky(ConnectionError("connection reset"))
    assert asyncio.run(limiter.ainvoke(flaky, {"call": "connection"})) == "ok"
    assert limiter.timeout_retries >= 1
    assert limiter.rate_limit_retries == 0
    assert limiter.limit == 1


def test_per_call_deadline_fails_fast():
    import time

    limiter = _limiter(max_retries=0, request_timeout=0.05)
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        asyncio.run(limiter.ainvoke(_Sleeper(), {"call": "slow"}))
    assert time.perf_counter() - started < 2


def test_resolve_chat_model_sets_timeout(monkeypatch):
    captured: dict = {}

    def fake_init(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return FakeCoder()

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init)
    model = resolve_chat_model("xai:grok-4")
    assert isinstance(model, FakeCoder)
    assert captured["kwargs"]["timeout"] == 120

    resolve_chat_model("xai:grok-4", model_kwargs={"timeout": 5}, timeout=120)
    assert captured["kwargs"]["timeout"] == 5


def test_every_candidate_is_accepted_or_rejected():
    applied = {"accepted_merges": [{"source": "packing bags up", "target": "packing bags"}], "rejected_merges": []}
    finalize_candidate_decisions(
        applied,
        candidates=[
            {"source": "packing bags up", "target": "packing bags"},
            {"source": "paying more", "target": "asking for pay"},
            {"source": "zeta", "target": "alpha"},
        ],
        rejections=[{"source": "paying more", "target": "asking for pay", "rationale": "Different incidents."}],
        skipped=[{"source": "staffing up", "target": "short staffed", "rationale": "chunk died"}],
    )
    reasons = {
        frozenset((item["source"], item["target"])): item["reason"]
        for item in applied["rejected_merges"]
    }
    assert reasons[frozenset(("paying more", "asking for pay"))] == "model_rejected"
    assert reasons[frozenset(("staffing up", "short staffed"))] == "chunk_skipped"
    assert reasons[frozenset(("zeta", "alpha"))] == "model_omitted"
    assert "packing bags" not in {item.get("source") for item in applied["rejected_merges"]}


def test_sweep_chunk_timeout_is_skipped_and_logged():
    result = open_code(
        CORPUS,
        CONTEXT,
        model=SweepTimeout(),
        seed=7,
        sweep_every=2,
        pipeline=False,
        batch_size=1,
        concurrency=1,
        max_retries=0,
        retry_backoff=0,
        retry_max_backoff=0,
    )
    assert any("skipped after a timeout" in warning for warning in result.warnings)
    assert result.sweep_log
    for record in result.sweep_log:
        decided = set()
        for item in list(record["accepted_merges"]) + list(record["rejected_merges"]):
            pair = frozenset((code_key(str(item.get("source") or "")), code_key(str(item.get("target") or ""))))
            decided.add(pair)
        for candidate in record["candidates"]:
            pair = frozenset(
                (code_key(str(candidate.get("source") or "")), code_key(str(candidate.get("target") or "")))
            )
            assert pair in decided
        if record["candidates"]:
            assert any(item.get("reason") == "chunk_skipped" for item in record["rejected_merges"])


def test_blank_sweep_model_uses_the_coder():
    coder = FakeCoder()
    result = open_code(
        CORPUS[:1],
        CONTEXT,
        model=coder,
        sweep_model="  ",
        sweep_every=25,
        pipeline=False,
        max_retries=0,
        retry_backoff=0,
    )
    assert result.metadata["sweep_model"] == result.metadata["model"]
    import inspect

    assert inspect.signature(open_code).parameters["sweep_chunk"].default == 20


def test_load_run_then_build_themes(tmp_path: Path):
    coded = open_code(
        CORPUS,
        CONTEXT,
        model=FakeCoder(),
        output_dir=tmp_path,
        seed=1,
        sweep_every=25,
        pipeline=False,
        max_retries=0,
        retry_backoff=0,
    )
    loaded = load_run(tmp_path)
    assert [row["code"] for row in loaded.codebook] == [row["code"] for row in coded.codebook]
    assert loaded.saturation == coded.saturation
    report = json.loads((tmp_path / "saturation.json").read_text(encoding="utf-8"))
    assert isinstance(report, dict)
    assert "points" in report

    themed = build_themes(tmp_path, CONTEXT, model=FakeThemer(), chart=False, max_retries=0, retry_backoff=0)
    assert themed.themes
    labels = [code for theme in themed.themes for code in theme["codes"]]
    assert sorted(labels) == sorted(row["code"] for row in coded.codebook)

    (tmp_path / "run.json").unlink()
    again = OpenCodeResult.load(tmp_path)
    assert [row["code"] for row in again.codebook] == [row["code"] for row in coded.codebook]
    assert again.saturation == coded.saturation


def test_theme_timeout_becomes_ungrouped_incidents():
    model = TimeoutThemer()
    result = build_themes(
        _result(),
        CONTEXT,
        model=model,
        chart=False,
        validation_retries=0,
        max_retries=0,
        retry_backoff=0,
        retry_max_backoff=0,
    )
    assert any("Ungrouped incidents" in warning for warning in result.warnings)
    names = [theme["name"] for theme in result.themes]
    assert names == ["Ungrouped incidents"]
    assert sorted(code for theme in result.themes for code in theme["codes"]) == [
        "crying at the station",
        "packing bags",
        "starting a new job",
    ]


def test_theme_validation_failure_still_raises():
    model = FakeThemer(drop_last="always")
    with pytest.raises(ThemeError):
        build_themes(
            _result(),
            CONTEXT,
            model=model,
            chart=False,
            validation_retries=0,
            max_retries=0,
            retry_backoff=0,
        )


def test_collapse_caps_themes_and_merges_near_duplicates():
    words = [
        "apple",
        "bridge",
        "comet",
        "drum",
        "eagle",
        "forest",
        "glacier",
        "harbor",
        "island",
        "jungle",
        "kettle",
        "lemon",
    ]
    themes = [
        {
            "name": "Valuing workplace features",
            "definition": "People value workplace features such as pay tools and the building.",
            "properties": [],
            "codes": ["code a"],
        },
        {
            "name": "Valuing workplace relationships",
            "definition": "People value workplace relationships such as pay tools and the building.",
            "properties": [],
            "codes": ["code b"],
        },
    ]
    for word in words:
        themes.append({"name": word, "definition": word, "properties": [], "codes": [word]})
    merged, mapping, changed = collapse_themes(themes, max_themes=13, cap=True)
    assert changed
    assert len(merged) == 13
    assert mapping["Valuing workplace features"] == mapping["Valuing workplace relationships"]
    survivor = mapping["Valuing workplace features"]
    assert survivor == "Valuing workplace features"
    kept = next(theme for theme in merged if theme["name"] == survivor)
    assert sorted(kept["codes"]) == ["code a", "code b"]

    capped, _, capped_changed = collapse_themes(themes, max_themes=12, cap=True)
    assert capped_changed
    assert len(capped) <= 12

    uncapped, _, also_changed = collapse_themes(themes, max_themes=12, cap=False)
    assert also_changed
    assert len(uncapped) == 13

    embedded, embed_map, embed_changed = collapse_themes(
        [
            {"name": "alpha", "definition": "alpha", "properties": [], "codes": ["a"]},
            {"name": "bravo", "definition": "bravo", "properties": [], "codes": ["b"]},
        ],
        max_themes=12,
        vectors=[[1.0, 0.0], [1.0, 0.0]],
        cap=False,
    )
    assert embed_changed
    assert embed_map["alpha"] == embed_map["bravo"]
    assert len(embedded) == 1


def test_merge_timeout_keeps_deterministic_grouping():
    model = MergeTimeout()
    result = build_themes(
        _result(),
        CONTEXT,
        model=model,
        chart=False,
        validation_retries=0,
        max_retries=0,
        retry_backoff=0,
        retry_max_backoff=0,
    )
    assert model.call_count == 2
    assert "ThemeMergePlan" in model.seen
    assert any("Theme merge pass timed out" in warning for warning in result.warnings)
    grouped = {theme["name"]: theme["codes"] for theme in result.themes}
    survivor = next(name for name, codes in grouped.items() if "packing bags" in codes)
    assert "crying at the station" in grouped[survivor]
    assert survivor == "Valuing workplace features"
    assert any("starting a new job" in codes for codes in grouped.values())


def test_unchanged_themes_do_not_call_the_merge_model():
    model = FakeThemer()
    build_themes(_result(), CONTEXT, model=model, chart=False, max_retries=0, retry_backoff=0)
    assert model.call_count == 1


def test_in_vivo_prompt_requires_a_vivid_phrase():
    assert "Pay. Pay. Pay." in CODING_SYSTEM
    assert "feels like a number" in CODING_SYSTEM
    assert "staying for the team" in CODING_SYSTEM
    assert "in_vivo" in CODING_SYSTEM


def test_empty_document_memo_is_omitted():
    assert not substantive_memo("No property, dimension, or comparison was recorded.")
    assert substantive_memo("Compared with packing bags, this incident adds a time pressure.")

    memos: list = []
    commit_passage(
        {},
        [],
        memos,
        {},
        {},
        {"doc_id": "d1", "passage_id": "p1"},
        {"accepted": [], "memo": "No property, dimension, or comparison was recorded."},
    )
    assert memos == []

    notes: list = []
    _append_reading_notes(
        notes,
        {},
        [{"passage_index": 0}],
        {"passages": [{"passage_id": "p1", "doc_id": "d1", "doc_name": "Ada"}]},
    )
    assert [item["kind"] for item in notes] == ["reflexive"]
    assert all("No property" not in item["text"] for item in notes)
