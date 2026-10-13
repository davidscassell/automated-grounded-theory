#!/usr/bin/env python3
"""Compare sequential, document, and wave review on a fake model with latency.

No API key. The coder sleeps on each call so wall time reflects concurrency
rather than a provider. The embedding-latency row also sleeps on each
embedding call and ranks passages through the query API, so query concurrency
is part of the timing. Quality proxies are the final code count, accepted
merges, and near-duplicate pairs (label Jaccard >= 0.5) left after synthesis.

    python scripts/benchmark_review_modes.py
    python scripts/benchmark_review_modes.py --embedding-delay 0.03
    python scripts/benchmark_review_modes.py --unique-passages
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from langchain_core.embeddings import Embeddings
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


class SlowEmbeddings(Embeddings):
    """Embeddings that sleep. Queries use the async query API, one text at a time."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.model = "benchmark-slow-embeddings"
        self.dimensions = 4
        self.queries: list[str] = []
        self.documents: list[str] = []
        self.in_flight = 0
        self.peak = 0

    def embed_documents(self, texts):
        time.sleep(self.delay)
        self.documents.extend(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        time.sleep(self.delay)
        self.queries.append(text)
        return [0.0, 1.0, 0.0, 0.0]

    async def aembed_documents(self, texts):
        await asyncio.sleep(self.delay)
        self.documents.extend(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def aembed_query(self, text):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.queries.append(text)
        try:
            await asyncio.sleep(self.delay)
            return [0.0, 1.0, 0.0, 0.0]
        finally:
            self.in_flight -= 1


def corpus(documents: int = 6, passages: int = 4, *, unique: bool = False) -> list[dict]:
    """Repeated incidents so synthesis has duplicates to merge.

    ``unique=True`` gives every passage its own text. The default reuses four
    stems, so an embedding row mostly hits the query cache.
    """
    stems = [
        "I spent the morning packing my suitcase before the train.",
        "Packing bags up with boxes took all night and I was not done.",
        "I was crying at the station when the train doors closed.",
        "They hired me on a Friday and the new job started immediately.",
    ]
    rows = []
    for doc_index in range(documents):
        paragraphs = []
        for offset in range(passages):
            text = stems[offset % len(stems)]
            if unique:
                text = f"{text} Document {doc_index} passage {offset}."
            paragraphs.append(text)
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


def run_mode(
    name: str,
    *,
    delay: float,
    embeddings=None,
    documents: int = 6,
    passages: int = 4,
    rows: list | None = None,
    **kwargs,
) -> dict:
    coder = SlowCoder()
    coder.delay = delay
    started = time.perf_counter()
    result = open_code(
        rows if rows is not None else corpus(documents, passages),
        "How do people leave?",
        model=coder,
        seed=7,
        embeddings=embeddings,
        **kwargs,
    )
    elapsed = time.perf_counter() - started
    passage_count = len(result.passages)
    row = {
        "mode": name,
        "seconds": round(elapsed, 3),
        "calls": coder.call_count,
        "passages": passage_count,
        "passages_per_second": round(passage_count / elapsed, 2) if elapsed else 0,
        "codes": len(result.codebook),
        "merges": _merges(result),
        "near_duplicate_pairs": _duplicates(result.codebook),
        "review_mode": result.metadata.get("review_mode"),
        "pipeline": kwargs.get("pipeline"),
    }
    if embeddings is not None:
        row["embedding_queries"] = len(getattr(embeddings, "queries", []) or [])
        row["embedding_query_peak"] = int(getattr(embeddings, "peak", 0) or 0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds of simulated latency per model call")
    parser.add_argument(
        "--embedding-delay",
        type=float,
        default=None,
        help="Seconds of simulated latency per embedding call. Defaults to --delay. Pass 0 to skip that row.",
    )
    parser.add_argument(
        "--unique-passages",
        action="store_true",
        help="Embedding row: a distinct query text per passage, instead of four reused stems.",
    )
    args = parser.parse_args()
    embedding_delay = args.delay if args.embedding_delay is None else args.embedding_delay
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
    if embedding_delay > 0:
        embedding = run_mode(
            "embedding",
            delay=args.delay,
            embeddings=SlowEmbeddings(embedding_delay),
            rows=corpus(unique=True) if args.unique_passages else None,
            review_mode="wave",
            pipeline=True,
            batch_size=4,
            concurrency=4,
            **context_kwargs,
        )
        if args.unique_passages:
            embedding["unique_passages"] = True
        rows.append(embedding)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
