"""Structured decisions, confidence gates, and merge guards."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.async_support import ainvoke_structured
from automated_grounded_theory.codebook import (
    apply_sweep,
    interpret_coding,
    remember_pair_decisions,
    replay_accepted_pairs,
    sync_counts,
)
from automated_grounded_theory.evaluate import _confidence_distribution
from automated_grounded_theory.export import export_result
from automated_grounded_theory.results import OpenCodeResult
from automated_grounded_theory.schemas import PassageCoding
from automated_grounded_theory.validation import label_restates_context


class _Limiter:
    async def ainvoke(self, runnable, messages):
        return runnable.invoke(messages)


class _RepairModel:
    call_count = 0

    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            self.call_count += 1
            text = "\n".join(getattr(message, "content", str(message)) for message in messages)
            if "REPAIR THIS OUTPUT" in text:
                parsed = PassageCoding(passage_id="p1", codes=[], memo="")
                return {"raw": AIMessage(content="{}"), "parsed": parsed, "parsing_error": None}
            return {"raw": AIMessage(content="this is not json"), "parsed": None, "parsing_error": "nope"}

        return RunnableLambda(_call)


def test_repair_retries_once():
    model = _RepairModel()

    async def run():
        return await ainvoke_structured(
            model,
            PassageCoding,
            [HumanMessage(content="{}")],
            _Limiter(),
        )

    (payload, _usage), messages = asyncio.run(run())
    assert payload["passage_id"] == "p1"
    assert model.call_count == 2
    assert "REPAIR THIS OUTPUT" in messages[-1].content


def test_unknown_code_id_is_rejected():
    passage = {"passage_id": "p1", "text": "I packed my suitcase.", "start_offset": 0, "metadata": {}}
    coding = {
        "passage_id": "p1",
        "codes": [
            {
                "action": "reuse",
                "code": "packing bags",
                "code_id": "not-a-real-id",
                "quote": "I packed my suitcase.",
                "confidence": 0.9,
            }
        ],
    }
    interpreted = interpret_coding(
        coding,
        passage,
        {"packing bags": {"code": "packing bags", "code_id": "abc", "key": "packing bags"}},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
    )
    assert interpreted["accepted"] == []
    assert interpreted["rejections"][0]["reason"] == "unknown_code_id"


def _kept(label: str) -> dict:
    return {
        "code_key": label,
        "code": label,
        "quote": label,
        "passage_id": label,
        "doc_id": "d",
    }


def test_low_confidence_merge_is_not_applied():
    book = {
        "packing bags": {"code": "packing bags", "key": "packing bags", "definition": "own", "history": []},
        "packing bags up": {"code": "packing bags up", "key": "packing bags up", "definition": "other", "history": []},
    }
    segments = [_kept("packing bags"), _kept("packing bags up")]
    low = apply_sweep(
        book,
        segments,
        merges=[{"source": "packing bags up", "target": "packing bags", "confidence": 0.1, "rationale": "same"}],
        candidates=[],
        min_merge_confidence=0.5,
    )
    assert low["accepted_merges"] == []
    assert low["review"][0]["reason"] == "below_merge_confidence"
    assert set(low["codebook"]) == {"packing bags", "packing bags up"}
    high = apply_sweep(
        book,
        segments,
        merges=[{
            "source": "packing bags up",
            "target": "packing bags",
            "confidence": 1.0,
            "rationale": "same",
            "same_action": True,
            "same_object": True,
            "contrast_checked": True,
            "material_contrast": False,
            "evidence_same": True,
        }],
        candidates=[],
        min_merge_confidence=0.5,
    )
    assert len(high["accepted_merges"]) == 1
    assert high["codebook"][next(iter(high["codebook"]))]["definition"] in {"own", "other"}


def test_opposite_valence_does_not_block_a_shared_category():
    book = {
        "praising direct team": {
            "code": "praising direct team",
            "key": "praising direct team",
            "definition": "praise",
            "valence": "positive",
            "history": [],
        },
        "noting manager overload": {
            "code": "noting manager overload",
            "key": "noting manager overload",
            "definition": "strain",
            "valence": "negative",
            "history": [],
        },
    }
    applied = apply_sweep(
        book,
        [_kept("praising direct team"), _kept("noting manager overload")],
        merges=[
            {
                "source": "noting manager overload",
                "target": "praising direct team",
                "confidence": 0.99,
                "rationale": "both about the team",
            }
        ],
        candidates=[],
    )
    assert applied["accepted_merges"] == []
    assert applied["rejected_merges"][0]["reason"] == "aspects_differ"
    assert applied["rejected_merges"][0]["source_valence"] == "negative"
    assert applied["rejected_merges"][0]["target_valence"] == "positive"
    assert "opposite" not in applied["rejected_merges"][0]["reason"]


def test_recreated_code_reapplies_the_accepted_merge():
    book = {
        "wanting better pay": {
            "code": "wanting better pay",
            "key": "wanting better pay",
            "definition": "Describing pay as below what the work is worth.",
            "history": [],
        },
        "wanting more pay": {
            "code": "wanting more pay",
            "key": "wanting more pay",
            "definition": "Incidents of wanting more pay.",
            "history": [],
        },
    }
    first = apply_sweep(
        book,
        [_kept("wanting better pay"), _kept("wanting more pay")],
        merges=[{
            "source": "wanting more pay",
            "target": "wanting better pay",
            "confidence": 0.9,
            "rationale": "same pay",
            "same_action": True,
            "same_object": True,
            "contrast_checked": True,
            "material_contrast": False,
            "evidence_same": True,
        }],
        candidates=[],
    )
    assert set(first["codebook"]) == {"wanting better pay"}
    cache = remember_pair_decisions(
        [],
        book,
        [{**first["accepted_merges"][0], "decision": "accept"}],
    )
    recreated = dict(first["codebook"])
    # The accept was recorded against the pre-merge book, which had no examples.
    # Replay stands only while that evidence is still the evidence that was judged.
    survivor = dict(recreated["wanting better pay"])
    survivor["examples"] = []
    recreated["wanting better pay"] = survivor
    recreated["wanting more pay"] = {
        "code": "wanting more pay",
        "key": "wanting more pay",
        "definition": "Incidents of wanting more pay.",
        "examples": [],
        "history": [],
    }
    replay = replay_accepted_pairs(cache, recreated)
    assert len(replay) == 1
    second = apply_sweep(
        recreated,
        list(first["segments"]) + [_kept("wanting more pay")],
        merges=replay,
        candidates=[],
        frozen_pairs=set(),
    )
    assert set(second["codebook"]) == {"wanting better pay"}
    rejected = remember_pair_decisions(
        [],
        book,
        [{**first["accepted_merges"][0], "decision": "reject"}],
    )
    assert replay_accepted_pairs(rejected, recreated) == []


def test_retired_definition_does_not_replace_the_survivor():
    book = {
        "packing bags": {"code": "packing bags", "key": "packing bags", "definition": "Survivor wording.", "history": []},
        "packing bags up": {
            "code": "packing bags up",
            "key": "packing bags up",
            "definition": "Retired wording.",
            "history": [],
        },
    }
    applied = apply_sweep(
        book,
        [_kept("packing bags"), _kept("packing bags up")],
        merges=[{
            "source": "packing bags up",
            "target": "packing bags",
            "confidence": 1,
            "rationale": "same",
            "same_action": True,
            "same_object": True,
            "contrast_checked": True,
            "material_contrast": False,
            "evidence_same": True,
        }],
        candidates=[],
        definitions=[{"code": "packing bags up", "definition": "Retired wording pasted onto the survivor."}],
    )
    survivor = next(iter(applied["codebook"].values()))
    assert survivor["definition"] == "Survivor wording."
    assert not any(item["reason"] == "definition_not_survivor" for item in applied["review"])


def test_in_vivo_below_threshold_clears_the_flag():
    passage = {"passage_id": "p1", "text": "Pay. Pay. Pay.", "start_offset": 0, "metadata": {}}
    interpreted = interpret_coding(
        {
            "codes": [
                {
                    "action": "new",
                    "code": "Pay. Pay. Pay.",
                    "quote": "Pay. Pay. Pay.",
                    "in_vivo": True,
                    "in_vivo_confidence": 0.2,
                    "confidence": 0.9,
                    "definition": "Repeating the word pay.",
                }
            ]
        },
        passage,
        {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
        review_below=0.5,
    )
    assert interpreted["accepted"][0]["in_vivo"] is False
    assert "in_vivo_below_review_threshold" in interpreted["accepted"][0]["review_reasons"]


def test_confidence_mean_and_distribution():
    book = sync_counts(
        {"packing bags": {"code": "packing bags", "key": "packing bags"}},
        [
            {"code_key": "packing bags", "code": "packing bags", "confidence": 0.4, "quote": "a", "passage_id": "p1"},
            {"code_key": "packing bags", "code": "packing bags", "confidence": 0.8, "quote": "b", "passage_id": "p2"},
        ],
    )
    assert book["packing bags"]["confidence_min"] == 0.4
    assert book["packing bags"]["confidence_mean"] == 0.6
    report = _confidence_distribution([0.4, 0.8], 0.5)
    assert report["n"] == 2
    assert report["below_review"] == 1
    assert report["min"] == 0.4


def test_context_restatement_and_taguette_columns(tmp_path):
    assert label_restates_context("leaving home", {"leaving", "home", "students"})
    assert not label_restates_context("packing bags", {"leaving", "home", "students"})
    result = OpenCodeResult(
        codebook=[{"code": "packing bags", "definition": "Bags.", "frequency": 1, "document_ids": ["a"]}],
        segments=[],
        memos=[],
        passages=[],
        saturation=[],
        saturation_report={"judgment": None, "points": []},
        validation_log=[],
        sweep_log=[],
        prompt_log=[],
        metadata={},
    )
    paths = export_result(result, tmp_path)
    header = open(paths["taguette_codebook.csv"], encoding="utf-8").readline().strip()
    assert header == "tag,description,number of highlights,number of documents"
