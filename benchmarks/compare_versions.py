"""Compare a git tag with the current package at batch size 1 and batch size 4.

The tag is checked out in a git worktree so this process keeps importing the
current package. A research context is a JSON object or a markdown notes file.
Every live model call is appended to ``calls.jsonl``.

    python benchmarks/compare_versions.py --offline
    python benchmarks/compare_versions.py --model xai:grok-4 --corpus docs.json --context context.md --tag v0.5.4
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _progress(message: str) -> None:
    print(message, flush=True)


def load_context(path: str | None, question: str) -> dict:
    """JSON object, or markdown/text stored as notes under ``research_question``."""
    if not path:
        return {"research_question": question or "What is happening in these documents?"}
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.casefold() == ".json":
        payload = json.loads(text)
        if isinstance(payload, str):
            return {"research_question": payload}
        if not isinstance(payload, dict):
            raise ValueError("A JSON context must be an object or a research-question string.")
        return payload
    return {"research_question": question or "See notes", "notes": text}


def load_corpus(path: str):
    """JSON list, ``{"corpus", "context"}``, or a CSV with an id and a text column."""
    place = Path(path)
    if place.suffix.casefold() == ".json":
        payload = json.loads(place.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "corpus" in payload:
            return payload["corpus"], payload.get("context") or payload.get("ctx")
        return payload, None
    rows = list(csv.DictReader(place.open(encoding="utf-8")))
    documents = []
    for index, row in enumerate(rows):
        text = row.get("text") or row.get("comment") or row.get("document") or ""
        doc_id = row.get("id") or row.get("doc_id") or row.get("respondent_id") or f"d{index + 1}"
        documents.append({"id": str(doc_id), "text": text, "metadata": {key: value for key, value in row.items() if key not in {"text", "comment", "document"}}})
    return documents, None


class CallLog:
    """Append one JSON line per chat-model call. Safe to share across threads."""

    def __init__(self, path: Path, phase: str) -> None:
        from langchain_core.callbacks import BaseCallbackHandler

        self.path = path
        self.phase = phase
        self._lock = threading.Lock()
        self._starts: dict = {}
        parent = self

        class _Handler(BaseCallbackHandler):
            def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
                parent._starts[run_id] = time.time()

            def on_llm_end(self, response, *, run_id, **kwargs):
                parent._write(run_id, ok=True, response=response)

            def on_llm_error(self, error, *, run_id, **kwargs):
                parent._write(run_id, ok=False, error=error)

        self.handler = _Handler()

    def _write(self, run_id, *, ok: bool, response=None, error=None) -> None:
        started = self._starts.pop(run_id, time.time())
        usage = {}
        if response is not None:
            try:
                usage = response.generations[0][0].message.usage_metadata or {}
            except Exception:
                usage = {}
        row = {
            "phase": self.phase,
            "sec": round(time.time() - started, 2),
            "in": usage.get("input_tokens", 0),
            "out": usage.get("output_tokens", 0),
            "ok": ok,
        }
        if error is not None:
            row["err"] = type(error).__name__
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    def write_prompt_log(self, prompt_log) -> None:
        """Record calls that never reached a LangChain callback, such as the fake coder."""
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                for item in prompt_log or []:
                    usage = item.get("usage") or {}
                    handle.write(
                        json.dumps(
                            {
                                "phase": self.phase,
                                "kind": item.get("kind"),
                                "sec": None,
                                "in": usage.get("input_tokens", 0),
                                "out": usage.get("output_tokens", 0),
                                "ok": True,
                                "source": "prompt_log",
                            }
                        )
                        + "\n"
                    )


def _run_current(batch_size: int, *, model: str, corpus, context: dict, seed: int, log_path: Path | None) -> dict:
    from automated_grounded_theory import open_code
    from automated_grounded_theory.version import __version__

    logger = CallLog(log_path, "open_code") if log_path else None
    if model == "fake":
        from tests.fake_llm import FakeCoder

        chat = FakeCoder()
        model_kwargs = None
    else:
        chat = model
        model_kwargs = {"callbacks": [logger.handler]} if logger else None
    result = open_code(
        corpus,
        context,
        model=chat,
        seed=seed,
        batch_size=batch_size,
        concurrency=1 if batch_size == 1 else 4,
        sweep_every=25,
        review_mode="document",
        model_kwargs=model_kwargs,
    )
    if logger is not None and model == "fake":
        logger.write_prompt_log(result.prompt_log)
    metrics = dict((result.metadata or {}).get("metrics") or {})
    metrics.pop("coding_seconds", None)
    return {
        "version": __version__,
        "batch_size": batch_size,
        "codes": len(result.codebook),
        "segments": len(result.segments),
        "passages": len(result.passages),
        "calls": getattr(chat, "call_count", None),
        "rounds": len(metrics.get("rounds") or []),
        "reconciliation_calls": metrics.get("reconciliation_calls"),
        "duplicates_before_reconciliation": metrics.get("duplicates_before_reconciliation"),
        "metrics": metrics,
        "stage_seconds": (result.metadata or {}).get("stage_seconds") or {},
        "activity": metrics.get("activity") or {},
    }


def _materialize_tag(tag: str) -> tuple[Path, str]:
    parent = Path(tempfile.mkdtemp(prefix=f"agt-{tag}-"))
    destination = parent / "tree"
    completed = subprocess.run(
        ["git", "worktree", "add", "--detach", str(destination), tag],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return destination, "worktree"
    archive = subprocess.run(["git", "archive", tag], cwd=ROOT, check=True, capture_output=True)
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive.stdout, check=True)
    return destination, "archive"


def _remove_tree(destination: Path, kind: str) -> None:
    if kind == "worktree":
        subprocess.run(["git", "worktree", "remove", "--force", str(destination)], cwd=ROOT, check=False)
    else:
        import shutil

        shutil.rmtree(destination.parent, ignore_errors=True)


_CHILD = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from automated_grounded_theory import open_code
from automated_grounded_theory.version import __version__
payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
log_path = Path(sys.argv[3])
if payload["model"] == "fake":
    from tests.fake_llm import FakeCoder
    chat = FakeCoder()
else:
    chat = payload["model"]
sys.path.insert(0, sys.argv[4])
from benchmarks.compare_versions import CallLog
logger = CallLog(log_path, "open_code")
model_kwargs = None if payload["model"] == "fake" else {"callbacks": [logger.handler]}
kwargs = dict(model=chat, seed=payload["seed"], model_kwargs=model_kwargs)
if str(__version__).startswith("0.6"):
    batch = int(payload["batch_size"])
    kwargs.update(batch_size=batch, concurrency=1 if batch == 1 else 4, review_mode="document")
result = open_code(payload["corpus"], payload["context"], **kwargs)
if payload["model"] == "fake":
    logger.write_prompt_log(result.prompt_log)
metrics = dict((result.metadata or {}).get("metrics") or {})
print(json.dumps({
    "version": __version__,
    "batch_size": kwargs.get("batch_size"),
    "codes": len(result.codebook),
    "segments": len(result.segments),
    "passages": len(result.passages),
    "calls": getattr(chat, "call_count", None),
    "reconciliation_calls": metrics.get("reconciliation_calls"),
    "duplicates_before_reconciliation": metrics.get("duplicates_before_reconciliation"),
}))
"""


def _run_tag(tag: str, *, corpus, context: dict, seed: int, model: str, log_path: Path | None) -> dict:
    destination, kind = _materialize_tag(tag)
    payload_path = destination / "payload.json"
    payload_path.write_text(
        json.dumps({"corpus": corpus, "context": context, "seed": seed, "model": model, "batch_size": 8}),
        encoding="utf-8",
    )
    log = log_path or (destination / "calls.jsonl")
    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-c", _CHILD, str(destination), str(payload_path), str(log), str(ROOT)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        _remove_tree(destination, kind)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Compare a git tag with the current package at batch 1 and batch 4.")
    parser.add_argument("--offline", action="store_true", help="Use the fake coder and the built-in corpus.")
    parser.add_argument("--model", default="", help="provider:model-name for a live run.")
    parser.add_argument("--corpus", default="", help="CSV or JSON corpus. JSON may be {corpus, context}.")
    parser.add_argument("--context", default="", help="JSON object or a markdown notes file.")
    parser.add_argument("--question", default="", help="Research question used when --context is not JSON.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="v0.5.4")
    parser.add_argument("--log-dir", default="", help="Directory for calls.jsonl. One file per run.")
    args = parser.parse_args()
    context_from_corpus = None
    if args.offline:
        from tests.corpus import CONTEXT, CORPUS

        corpus, context, model = CORPUS, CONTEXT, "fake"
    elif args.model and args.corpus:
        corpus, context_from_corpus = load_corpus(args.corpus)
        context = load_context(args.context or None, args.question)
        if context_from_corpus and not args.context:
            context = context_from_corpus
        model = args.model
    else:
        parser.error("Pass --offline, or --model and --corpus.")
    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if model == "fake" or args.tag:
        _progress(f"Running {args.tag}")
        tag_log = (log_dir / f"{args.tag}.jsonl") if log_dir else None
        rows.append(_run_tag(args.tag, corpus=corpus, context=context, seed=args.seed, model=model, log_path=tag_log))
    for batch in (1, 4):
        _progress(f"Running current batch_size={batch}")
        current_log = (log_dir / f"current-batch-{batch}.jsonl") if log_dir else None
        rows.append(_run_current(batch, model=model, corpus=corpus, context=context, seed=args.seed, log_path=current_log))
    _progress(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
