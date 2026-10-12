#!/usr/bin/env python3
"""Compare sequential, document, and wave review on a fake model with latency.

No API key. The coder sleeps on each call so wall time reflects concurrency
rather than a provider. Quality proxies are the final code count, accepted
merges, and near-duplicate pairs (label Jaccard >= 0.5) left after synthesis.

    python scripts/benchmark_review_modes.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import open_code
from automated_grounded_theory.retrieval import jaccard
from tests.fake_llm import FakeCoder


class SlowCoder(FakeCoder):
    """FakeCoder with a fixed delay on every structured call."""

    delay: float = 0.05

    def with_structured_output(self, schema, **kwargs):
        parent = super().with_structured_output(schema, **kwargs)

        async def _call(messages):
            await asyncio.sleep(self.delay)
            return parent.invoke(messages)

        return RunnableLambda(_call)


def corpus(documents: int = 6, passages: int = 4) -> list[dict]:
    """Repeated incidents so synthesis has duplicates to merge."""
    stems = [
        "I spent the morning packing my suitcase before the train.",
        "Packing bags up with boxes took all night and I was not done.",
        "I was crying at the station when the train doors closed.",
        "They hired me on a Friday and the new job started immediately.",
    ]
    rows = []
    for doc_index in range(documents):
        paragraphs = [stems[offset % len(stems)] for offset in range(passages)]
        rows.append({"id": f"doc{doc_index:02d}", "name": f"Doc {doc_index}", "text": "\n\n".join(paragraphs)})
    return rows


def _duplicates(codebook: list[dict]) -> int:
    labels = [str(entry.get("code") or "") for entry in codebook]
    found = 0
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if jaccard(labels[left], labels[right]) >= 0.5:
                found += 1
    return found


def _merges(result) -> int:
    return sum(len(entry.get("accepted_merges") or []) for entry in result.sweep_log)


def run_mode(name: str, *, delay: float, **kwargs) -> dict:
    coder = SlowCoder()
    coder.delay = delay
    started = time.perf_counter()
    result = open_code(corpus(), "How do people leave?", model=coder, seed=7, **kwargs)
    elapsed = time.perf_counter() - started
    passages = len(result.passages)
    return {
        "mode": name,
        "seconds": round(elapsed, 3),
        "calls": coder.call_count,
        "passages": passages,
        "passages_per_second": round(passages / elapsed, 2) if elapsed else 0,
        "codes": len(result.codebook),
        "merges": _merges(result),
        "near_duplicate_pairs": _duplicates(result.codebook),
        "review_mode": result.metadata.get("review_mode"),
        "pipeline": kwargs.get("pipeline"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds of simulated latency per model call")
    args = parser.parse_args()
    context_kwargs = dict(sweep_every=8, temperature=0, unit="paragraph")
    rows = [
        run_mode(
            "sequential",
            delay=args.delay,
            review_mode="document",
            pipeline=False,
            batch_size=1,
            concurrency=1,
            **context_kwargs,
        ),
        run_mode(
            "document",
            delay=args.delay,
            review_mode="document",
            pipeline=True,
            batch_size=4,
            concurrency=4,
            **context_kwargs,
        ),
        run_mode(
            "wave",
            delay=args.delay,
            review_mode="wave",
            pipeline=True,
            batch_size=4,
            concurrency=4,
            **context_kwargs,
        ),
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
