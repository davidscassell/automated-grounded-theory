"""Async execution, document review order, and deterministic reconciliation."""

import asyncio

import pytest

from automated_grounded_theory import aopen_code, open_code
from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.clustering import adjusted_rand_index, union_roots
from automated_grounded_theory.evaluate import assignment_ari
from automated_grounded_theory.graph import reconcile_sweep_parts
from automated_grounded_theory.themes import reconcile_theme_proposals
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder


class RateLimitError(Exception):
    pass


class FlakyRateCoder(FakeCoder):
    failures: int = 0

    def with_structured_output(self, schema, **kwargs):
        parent = super().with_structured_output(schema, **kwargs)

        async def _call(messages):
            if self.failures < 1:
                self.failures += 1
                raise RateLimitError("rate limit")
            return parent.invoke(messages)

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_call)


def test_sync_wrapper_runs_inside_a_loop():
    async def _inner():
        return open_code(
            CORPUS,
            CONTEXT,
            model=FakeCoder(),
            seed=7,
            sweep_every=25,
            batch_size=4,
            concurrency=1,
            review_mode="document",
            pipeline=False,
        )

    result = asyncio.run(_inner())
    assert len(result.segments) == 4
    assert {item["doc_id"] for item in result.passages} == {"alex", "blair", "casey", "drew"}


def test_rate_limit_retries_then_succeeds():
    async def _inner():
        limiter = CallLimiter(1, max_retries=2, backoff=0)
        coder = FlakyRateCoder()
        result = await aopen_code(
            CORPUS,
            CONTEXT,
            model=coder,
            seed=1,
            sweep_every=25,
            batch_size=4,
            concurrency=1,
            review_mode="wave",
            pipeline=False,
            limiter=limiter,
            max_retries=2,
            retry_backoff=0,
        )
        return result, coder, limiter

    result, coder, limiter = asyncio.run(_inner())
    assert coder.failures == 1
    assert limiter.rate_limit_retries >= 1
    assert len(result.segments) == 4


def test_reconcile_sweep_is_order_independent():
    early = {
        "chunk_index": 0,
        "merges": [{"source": "alpha", "target": "beta", "rationale": "early"}],
        "definitions": [{"code": "alpha", "definition": "short"}],
        "overloaded_codes": [],
        "relation_hints": [],
        "integrative_memo": "first",
        "reflexive_memo": "",
    }
    late = {
        "chunk_index": 1,
        "merges": [{"source": "beta", "target": "alpha", "rationale": "late"}],
        "definitions": [{"code": "alpha", "definition": "a longer definition wins"}],
        "overloaded_codes": [],
        "relation_hints": [],
        "integrative_memo": "second",
        "reflexive_memo": "",
    }
    forward = reconcile_sweep_parts([late, early])
    backward = reconcile_sweep_parts([early, late])
    assert forward == backward
    assert forward["merges"] == [early["merges"][0]]
    assert forward["definitions"] == [{"code": "alpha", "definition": "a longer definition wins"}]


def test_reconcile_themes_keeps_the_earlier_chunk():
    first = (1, {"themes": [{"name": "A", "codes": ["packing bags"]}], "tensions": [], "negative_cases": []})
    second = (0, {"themes": [{"name": "B", "codes": ["packing bags"]}], "tensions": [], "negative_cases": []})
    joined = reconcile_theme_proposals([first, second])
    assert [theme["name"] for theme in joined["themes"]] == ["B", "A"]


def test_document_mode_reviews_every_passage_in_seeded_order():
    first = open_code(CORPUS, CONTEXT, model=FakeCoder(), seed=7, sweep_every=25, pipeline=False)
    second = open_code(CORPUS, CONTEXT, model=FakeCoder(), seed=7, sweep_every=25, pipeline=False)
    third = open_code(CORPUS, CONTEXT, model=FakeCoder(), seed=4, sweep_every=25, pipeline=False)
    assert [item["passage_id"] for item in first.passages] == [item["passage_id"] for item in second.passages]
    assert [item["doc_id"] for item in first.passages] != [item["doc_id"] for item in third.passages]
    assert {item["doc_id"] for item in first.passages} == {"alex", "blair", "casey", "drew"}
    assert len(first.segments) == 4
    assert any(memo["kind"] == "document" for memo in first.memos)
    assert any(memo["kind"] == "reflexive" for memo in first.memos)
    assert first.metadata["review_mode"] == "document"


def test_wave_mode_can_show_an_empty_codebook_to_two_batches():
    coder = FakeCoder()
    result = open_code(
        CORPUS,
        CONTEXT,
        model=coder,
        seed=7,
        sweep_every=25,
        batch_size=1,
        concurrency=2,
        review_mode="wave",
        pipeline=False,
    )
    code_prompts = [item for item in result.prompt_log if item["kind"] == "code_passage"]
    assert len(code_prompts) == 4
    import json

    assert sum("codebook is empty" in json.dumps(item["messages"]) for item in code_prompts) == 2
    assert result.metadata["review_mode"] == "wave"


def test_union_find_root_and_ari_are_deterministic():
    forward = union_roots([("b", "a"), ("c", "b")])
    backward = union_roots([("c", "b"), ("a", "b")])
    assert forward == backward
    assert forward["c"] == "a"
    assert adjusted_rand_index(["a", "a", "b"], ["a", "a", "b"]) == 1.0
    assert adjusted_rand_index(["a", "a", "b"], ["a", "b", "b"]) < 1.0
    left = {"segments": [{"passage_id": "p1", "code": "packing"}, {"passage_id": "p2", "code": "crying"}]}
    right = {"segments": [{"passage_id": "p2", "code": "crying"}, {"passage_id": "p1", "code": "packing"}]}
    assert assignment_ari(left, right) == 1.0


def test_same_seed_document_runs_match():
    kwargs = dict(seed=11, sweep_every=2, review_mode="document", pipeline=True)
    one = open_code(CORPUS, CONTEXT, model=FakeCoder(), **kwargs)
    two = open_code(CORPUS, CONTEXT, model=FakeCoder(), **kwargs)
    assert [entry["code"] for entry in one.codebook] == [entry["code"] for entry in two.codebook]
    assert [item["quote"] for item in one.segments] == [item["quote"] for item in two.segments]
    packing = [entry for entry in one.codebook if "packing" in entry["code"]]
    assert len(packing) == 1
    assert packing[0]["definition"].startswith("Bagging")
