"""Compare v0.5.4, 0.6.0 at batch size 1, and 0.6.0 at batch size 4.

Offline runs use the repository's fake coder and the synthetic corpus. A live
model is optional. v0.5.4 is the git tag of the merged 0.5.4 release, checked
out into a temporary tree so this process keeps importing 0.6.0.

    python benchmarks/compare_versions.py --offline
    python benchmarks/compare_versions.py --model xai:grok-4 --corpus path.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _progress(message: str) -> None:
    print(message, flush=True)


def _run_current(batch_size: int, *, model: str, corpus, context: dict, seed: int) -> dict:
    from automated_grounded_theory import open_code
    from automated_grounded_theory.version import __version__

    if model == "fake":
        from tests.fake_llm import FakeCoder

        chat = FakeCoder()
    else:
        chat = model
    result = open_code(
        corpus,
        context,
        model=chat,
        seed=seed,
        batch_size=batch_size,
        concurrency=1 if batch_size == 1 else 4,
        sweep_every=25,
        review_mode="document",
    )
    return {
        "version": __version__,
        "batch_size": batch_size,
        "codes": len(result.codebook),
        "segments": len(result.segments),
        "passages": len(result.passages),
        "calls": getattr(chat, "call_count", None),
        "metrics": (result.metadata or {}).get("metrics") or {},
        "stage_seconds": (result.metadata or {}).get("stage_seconds") or {},
    }


def _run_tag(tag: str, *, corpus, context: dict, seed: int) -> dict:
    destination = Path(tempfile.mkdtemp(prefix="agt-v054-"))
    archive = subprocess.run(
        ["git", "archive", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive.stdout, check=True)
    program = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from automated_grounded_theory import open_code
from automated_grounded_theory.version import __version__
from tests.fake_llm import FakeCoder
payload = json.loads(sys.argv[2])
model = FakeCoder()
result = open_code(
    payload["corpus"],
    payload["context"],
    model=model,
    seed=payload["seed"],
    batch_size=8,
    concurrency=2,
    sweep_every=25,
)
print(json.dumps({
    "version": __version__,
    "batch_size": 8,
    "codes": len(result.codebook),
    "segments": len(result.segments),
    "passages": len(result.passages),
    "calls": model.call_count,
}))
"""
    payload = json.dumps({"corpus": corpus, "context": context, "seed": seed})
    completed = subprocess.run(
        [sys.executable, "-u", "-c", program, str(destination), payload],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Compare v0.5.4 with 0.6.0 batch 1 and batch 4.")
    parser.add_argument("--offline", action="store_true", help="Use the fake coder and the built-in corpus.")
    parser.add_argument("--model", default="", help="provider:model-name for a live run of 0.6.0 only.")
    parser.add_argument("--corpus", default="", help="CSV or JSON corpus for a live run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="v0.5.4")
    args = parser.parse_args()
    if args.offline:
        from tests.corpus import CONTEXT, CORPUS

        corpus, context, model = CORPUS, CONTEXT, "fake"
    elif args.model and args.corpus:
        corpus = args.corpus
        context = {"research_question": "What is happening in these documents?"}
        model = args.model
    else:
        parser.error("Pass --offline, or --model and --corpus.")
    rows = []
    if model == "fake":
        _progress(f"Running {args.tag}")
        rows.append(_run_tag(args.tag, corpus=corpus, context=context, seed=args.seed))
    for batch in (1, 4):
        _progress(f"Running 0.6.0 batch_size={batch}")
        rows.append(_run_current(batch, model=model, corpus=corpus, context=context, seed=args.seed))
    _progress(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
