"""0.6.2: in-batch comparison, interleaved rounds, and a longer sweep deadline."""

import asyncio
import json

from automated_grounded_theory import open_code
from automated_grounded_theory.async_support import CallLimiter, sweep_deadline
from automated_grounded_theory.codebook import plan_in_batch_aliases
from automated_grounded_theory.segment import interleave_document_queue, next_round_count
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder

_ALIAS = {
    "same_action": True,
    "same_object": True,
    "same_stance": True,
    "contrast_checked": True,
    "material_contrast": False,
    "evidence_same": True,
    "comparison_op": "alias_duplicate",
    "confidence": 0.95,
}


def test_in_batch_comparison_aliases_one_concept_and_keeps_another():
    expense = "noting Embase.com expense"
    comparison = "liking Embase.com result comparison"
    plan = plan_in_batch_aliases(
        [
            {"code": expense, "definition": "Noting what Embase.com costs.", "quotes": ["Embase.com is expensive."]},
            {"code": comparison, "definition": "Liking how Embase.com compares results.", "quotes": ["I like the comparison."]},
            {"code": "packing bags", "definition": "Bagging possessions.", "quotes": ["I packed my suitcase."]},
            {"code": "packing bags up", "definition": "Bagging possessions.", "quotes": ["Packing bags up with boxes."]},
        ],
        merges=[
            {"source": expense, "target": comparison, "rationale": "both mention Embase", **_ALIAS},
            {"source": "packing bags up", "target": "packing bags", "rationale": "same bags", "rewritten_definition": "Bagging possessions.", **_ALIAS},
        ],
    )
    assert plan["rewrites"] == {} or "embase" not in " ".join(plan["rewrites"].values()).casefold()
    assert all(pair["source"] != expense or pair["applied"] is False for pair in plan["pairs"])
    assert plan["rewrites"]["packing bags up"] == "packing bags"
    assert any(pair["applied"] and pair["scope"] == "in_batch" for pair in plan["pairs"])


def test_fake_round_aliases_packing_once_and_keeps_the_other_codes(tmp_path):
    kwargs = dict(
        seed=7,
        sweep_every=100,
        batch_size=4,
        concurrency=1,
        pipeline=False,
        review_mode="document",
        output_dir=tmp_path,
    )
    model = FakeCoder()
    result = open_code(CORPUS, CONTEXT, model=model, **kwargs)
    labels = {entry["code"] for entry in result.codebook}
    packing = [entry for entry in result.codebook if "packing" in entry["code"]]
    assert len(packing) == 1
    assert packing[0]["frequency"] == 2
    assert "crying at the station" in labels
    assert "starting a new job" in labels
    assert sum(item["kind"] == "in_batch" for item in result.prompt_log) == 1
    exported = json.loads((tmp_path / "duplicate_rounds.json").read_text(encoding="utf-8"))
    applied = [
        pair
        for row in exported["in_batch"]
        for pair in row["pairs"]
        if pair.get("applied")
    ]
    assert applied
    assert all("packing" in str(pair["source"]) and "packing" in str(pair["target"]) for pair in applied)
    assert exported["cross_snapshot"] == [] or all(
        "packing" not in str(pair) for row in exported["cross_snapshot"] for pair in row.get("pairs") or []
    )
    again = open_code(CORPUS, CONTEXT, model=FakeCoder(), seed=7, sweep_every=100, batch_size=4, concurrency=1, pipeline=False)
    assert [row["passage_ids"] for row in result.metadata["metrics"]["rounds"]] == [
        row["passage_ids"] for row in again.metadata["metrics"]["rounds"]
    ]


def _sentences(doc_id: str, count: int) -> dict:
    text = " ".join(f"{doc_id} mentions item {index}." for index in range(count))
    return {"id": doc_id, "text": text}


def test_rounds_interleave_documents_and_repeat(tmp_path):
    corpus = [_sentences("ann", 3), _sentences("bo", 3), _sentences("cy", 3)]
    kwargs = dict(seed=4, unit="sentence", sweep_every=100, batch_size=2, concurrency=1, pipeline=False, review_mode="document")
    first = open_code(corpus, CONTEXT, model=FakeCoder(), output_dir=tmp_path, **kwargs)
    second = open_code(corpus, CONTEXT, model=FakeCoder(), **kwargs)
    rounds = first.metadata["metrics"]["rounds"]
    assert [row["passage_ids"] for row in rounds] == [
        row["passage_ids"] for row in second.metadata["metrics"]["rounds"]
    ]
    by_id = {item["passage_id"]: item for item in first.passages}
    seen = {doc: [] for doc in ("ann", "bo", "cy")}
    for row in rounds:
        passages = [by_id[passage_id] for passage_id in row["passage_ids"]]
        docs = [item["doc_id"] for item in passages]
        assert len(docs) == len(set(docs))
        for item in passages:
            seen[item["doc_id"]].append(item["position"])
    for positions in seen.values():
        assert positions == sorted(positions)
        assert len(positions) == 3

    solo = open_code(
        [_sentences("only", 5)],
        CONTEXT,
        model=FakeCoder(),
        seed=1,
        unit="sentence",
        sweep_every=100,
        batch_size=2,
        concurrency=1,
        pipeline=False,
    )
    sizes = [row["size"] for row in solo.metadata["metrics"]["rounds"]]
    assert sizes == [2, 2, 1]
    positions = []
    by_id = {item["passage_id"]: item for item in solo.passages}
    for row in solo.metadata["metrics"]["rounds"]:
        assert {by_id[passage_id]["doc_id"] for passage_id in row["passage_ids"]} == {"only"}
        positions.extend(by_id[passage_id]["position"] for passage_id in row["passage_ids"])
    assert positions == [0, 1, 2, 3, 4]


def test_interleave_queue_never_pairs_adjacent_passages_while_documents_remain():
    passages = []
    queue = []
    for doc_index, doc_id in enumerate(("a", "b", "c")):
        for position in range(3):
            queue.append(len(passages))
            passages.append({"doc_id": doc_id, "position": position})
    ordered = interleave_document_queue(passages, queue, 4)
    cursor = 0
    while cursor < len(ordered):
        count = next_round_count(passages, ordered[cursor:], 4)
        chunk = ordered[cursor : cursor + count]
        docs = [passages[index]["doc_id"] for index in chunk]
        assert len(docs) == len(set(docs))
        cursor += count
    assert [passages[index]["position"] for index in ordered if passages[index]["doc_id"] == "a"] == [0, 1, 2]


def test_sweep_deadline_is_longer_than_coding_and_can_be_passed_per_call():
    assert sweep_deadline(120, 300) == 300
    assert sweep_deadline(400, 300) == 400
    assert sweep_deadline(0, 300) == 0

    class Slow:
        async def ainvoke(self, payload):
            await asyncio.sleep(0.2)
            return payload

    limiter = CallLimiter(1, max_retries=0, request_timeout=30)

    async def run():
        await limiter.ainvoke(Slow(), "x", timeout=0.05)

    try:
        asyncio.run(run())
    except TimeoutError:
        return
    raise AssertionError("a per-call timeout did not cancel the call")
