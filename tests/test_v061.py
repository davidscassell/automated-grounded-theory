"""0.6.1: semantic duplicates, alias gate, batches, idle slots, and the shared loop."""

import asyncio
import json
from pathlib import Path

from automated_grounded_theory import open_code
from automated_grounded_theory.async_support import CallLimiter, run_sync
from automated_grounded_theory.codebook import (
    apply_post_merge,
    apply_sweep,
    aspects_differ,
    classify_comparison,
    mark_later_splits,
)
from automated_grounded_theory.records import same_round_duplicates
from automated_grounded_theory.themes import presentation_themes
from automated_grounded_theory.validation import code_key
from benchmarks.compare_versions import load_context
from tests.corpus import CONTEXT
from tests.fake_llm import FakeCoder

_ALIAS = {
    "same_action": True,
    "same_object": True,
    "contrast_checked": True,
    "material_contrast": False,
    "evidence_same": True,
    "comparison_op": "alias_duplicate",
}


def _book(label, definition, snapshot):
    return {
        "code": label,
        "key": label,
        "definition": definition,
        "valence": "neutral",
        "history": [],
        "origin_snapshot_id": snapshot,
    }


def test_alias_requires_the_same_concept_and_a_contrast_check():
    expense = "noting Embase.com expense"
    comparison = "liking Embase.com result comparison"
    book = {
        code_key(expense): _book(expense, "Noting what Embase.com costs.", "s1"),
        code_key(comparison): _book(comparison, "Liking how Embase.com compares results.", "s1"),
    }
    assert aspects_differ(expense, comparison)
    mixed = apply_sweep(
        book,
        [],
        merges=[{"source": expense, "target": comparison, "confidence": 0.95, "rationale": "both mention Embase", **_ALIAS}],
        candidates=[],
    )
    assert mixed["accepted_merges"] == []
    assert mixed["rejected_merges"][0]["reason"] == "aspects_differ"
    assert mixed["rejected_merges"][0]["comparison_op"] == "keep_distinct"

    unchecked = apply_sweep(
        {
            "packing bags": _book("packing bags", "Bagging possessions.", "s1"),
            "packing bags up": _book("packing bags up", "Bagging possessions.", "s1"),
        },
        [],
        merges=[{
            "source": "packing bags up",
            "target": "packing bags",
            "confidence": 0.95,
            "rationale": "same bags",
            "same_action": True,
            "same_object": True,
        }],
        candidates=[],
    )
    assert unchecked["rejected_merges"][0]["reason"] == "contrast_unchecked"
    assert unchecked["rejected_merges"][0]["comparison_op"] == "defer"

    evidence = apply_sweep(
        {
            "packing bags": _book("packing bags", "Bagging possessions.", "s1"),
            "packing bags up": _book("packing bags up", "Bagging possessions.", "s1"),
        },
        [
            {"code": "packing bags", "code_key": "packing bags", "quote": "I packed my suitcase.", "passage_id": "a"},
            {"code": "packing bags up", "code_key": "packing bags up", "quote": "Packing bags up with boxes.", "passage_id": "b"},
        ],
        merges=[{
            "source": "packing bags up",
            "target": "packing bags",
            "confidence": 0.95,
            "rationale": "same bags",
            **_ALIAS,
            "quote_fits": [{"quote": "Packing bags up with boxes.", "fits": False}],
        }],
        candidates=[],
    )
    assert evidence["accepted_merges"] == []
    assert evidence["rejected_merges"][0]["reason"] == "mixed_evidence"
    decision, reason = classify_comparison({"source": "packing bags", "target": "packing bags up", "trivial": True}, {})
    assert (decision, reason) == ("keep_distinct", "not_spelling_variant")
    variant, variant_reason = classify_comparison(
        {"source": "packing bags", "target": "packing bag", "trivial": True},
        {},
    )
    assert (variant, variant_reason) == ("alias_duplicate", "trivial_pair")


def test_same_round_duplicates_follow_meaning_and_the_audit_records_a_later_split():
    book = {
        "packing bags": _book("packing bags", "Bagging possessions before leaving.", "s00003"),
        "packing bags up": _book("packing bags up", "Bagging possessions before leaving.", "s00003"),
        "crying at the station": _book("crying at the station", "Crying while a train leaves.", "s00004"),
    }
    segments = [
        {"code": "packing bags", "code_key": "packing bags", "quote": "I packed my suitcase.", "passage_id": "a", "doc_id": "d"},
        {"code": "packing bags up", "code_key": "packing bags up", "quote": "Packing bags up with boxes.", "passage_id": "b", "doc_id": "e"},
    ]
    applied = apply_sweep(
        book,
        segments,
        merges=[{"source": "packing bags up", "target": "packing bags", "confidence": 0.95, "rationale": "same bags", **_ALIAS}],
        candidates=[],
    )
    audit = [row for row in applied["lost_distinction_candidates"] if row.get("kind") == "accepted_alias"]
    assert audit
    assert audit[0]["source_definition"]
    assert audit[0]["target_definition"]
    assert audit[0]["source_quotes"]
    assert audit[0]["rationale"] == "same bags"
    assert audit[0]["later_split"] is False
    pairs = same_round_duplicates(
        book,
        [{**applied["accepted_merges"][0], "same_action": True, "same_object": True}],
    )
    assert pairs[0]["snapshot_id"] == "s00003"
    assert pairs[0]["judgment"] == "alias_duplicate"
    other_round = same_round_duplicates(
        book,
        [{"source": "packing bags", "target": "crying at the station", "same_action": True, "same_object": True, "decision": "reject"}],
    )
    assert other_round == []
    book2, rows, ops, stats = apply_post_merge(
        applied["codebook"],
        applied["segments"],
        [{
            **applied["accepted_merges"][0],
            "quote_fits": [{
                "quote": "Packing bags up with boxes.",
                "fits": False,
                "split_to": "packing bags up",
                "rationale": "This quote is the retired code.",
            }],
        }],
        book,
        segments,
    )
    assert stats["merges_later_split"] == 1
    assert mark_later_splits(audit, stats["split_merges"]) == 1
    assert audit[0]["later_split"] is True
    assert any(row["code"] == "packing bags up" for row in rows)
    assert ops


def test_backoff_releases_the_slot_and_retry_after_is_capped():
    limiter = CallLimiter(1, max_retries=1, backoff=0.2, max_backoff=0.2, request_timeout=5)

    class Once:
        def __init__(self):
            self.failed = False

        async def ainvoke(self, payload):
            if payload == "flaky" and not self.failed:
                self.failed = True
                raise TimeoutError("timed out")
            return payload

    order = []
    runnable = Once()

    async def call(payload):
        result = await limiter.ainvoke(runnable, payload)
        order.append(payload)
        return result

    async def main():
        first = asyncio.create_task(call("flaky"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(call("other"))
        await asyncio.gather(first, second)

    asyncio.run(main())
    assert order.index("other") < order.index("flaky")
    report = limiter.activity_report()
    sleeps = [event for event in report["events"] if event.get("kind") == "idle_sleep"]
    assert sleeps
    assert sleeps[0]["seconds"] <= 0.2
    assert "idle_gaps" in report
    assert "events" in report

    class Later:
        retry_after = 1000

    assert limiter._wait(0, Later()) <= 0.2


def test_batch_size_groups_one_incident_documents_into_rounds(tmp_path):
    corpus = [{"id": f"r{index}", "text": f"I spent the morning packing suitcase {index}."} for index in range(8)]
    result = open_code(
        corpus,
        CONTEXT,
        model=FakeCoder(),
        seed=3,
        sweep_every=100,
        batch_size=4,
        concurrency=1,
        pipeline=False,
        output_dir=tmp_path,
    )
    metrics = result.metadata["metrics"]
    assert metrics["reconciliation_calls"] == 2
    assert [row["size"] for row in metrics["rounds"]] == [4, 4]
    assert len({doc for row in metrics["rounds"] for doc in row["doc_ids"]}) == 8
    assert (tmp_path / "duplicate_rounds.json").is_file()
    assert (tmp_path / "lost_distinction_candidates.json").is_file()
    activity = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))["metrics"]["activity"]
    assert "events" in activity
    assert "idle_gaps" in activity


def test_run_sync_reuses_one_loop_and_does_not_close_a_shared_client():
    class Client:
        def __init__(self):
            self.closed = False
            self.loop_ids = []

        async def aclose(self):
            self.closed = True

        async def ping(self):
            self.loop_ids.append(id(asyncio.get_running_loop()))
            return "ok"

    client = Client()

    async def ping():
        return await client.ping()

    assert run_sync(ping()) == "ok"
    assert run_sync(ping()) == "ok"
    assert client.loop_ids[0] == client.loop_ids[1]
    assert client.closed is False


def test_presentation_themes_fold_repeated_topics_without_a_quota():
    folded = presentation_themes(
        [
            {"name": "wanting better pay", "definition": "Pay.", "codes": ["a"]},
            {"name": "recognizing pay", "definition": "Pay again.", "codes": ["b"]},
            {"name": "naming a small perk", "definition": "A perk.", "codes": ["c"]},
            {"name": "valuing perks", "definition": "Perks.", "codes": ["d"]},
            {"name": "building reputation", "definition": "Reputation.", "codes": ["e"]},
        ]
    )
    assert len(folded) == 3
    pay = next(theme for theme in folded if "pay" in theme["name"])
    assert set(pay["codes"]) == {"a", "b"}
    assert len(pay["folded_from"]) == 2
    assert all(theme["presentation"] for theme in folded)


def test_compare_versions_loads_research_context(tmp_path: Path):
    notes = tmp_path / "context.md"
    notes.write_text("Employees at one hospital.", encoding="utf-8")
    loaded = load_context(str(notes), "How do people leave?")
    assert loaded["research_question"] == "How do people leave?"
    assert "hospital" in loaded["notes"]
    blob = tmp_path / "context.json"
    blob.write_text(json.dumps({"research_question": "What happened?", "population": "nurses"}), encoding="utf-8")
    assert load_context(str(blob), "")["population"] == "nurses"
