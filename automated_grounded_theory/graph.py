"""LangGraph pipeline: segment, code every passage, sweep, export.

Validation is its own node and does not call a model. The coding and sweep
nodes do. Checkpointing is attached by the caller so a run can resume after
an interrupt or a crash.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from automated_grounded_theory.codebook import (
    apply_sweep,
    commit_passage,
    empty_counters,
    interpret_coding,
    union_codebooks,
)
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.corpus import LoadedDocument
from automated_grounded_theory.export import export_result
from automated_grounded_theory.prompts import (
    CODING_SYSTEM,
    SWEEP_SYSTEM,
    build_coding_messages,
    build_sweep_messages,
    serialize_messages,
    validation_feedback,
)
from automated_grounded_theory.results import result_from_state
from automated_grounded_theory.retrieval import CodeRetriever, merge_candidates, select_prompt_codes
from automated_grounded_theory.saturation import saturation_point
from automated_grounded_theory.schemas import PassageCoding, SweepCoding
from automated_grounded_theory.segment import plan_coding

logger = logging.getLogger("automated_grounded_theory")


class CodingState(TypedDict, total=False):
    documents: list[dict]
    passages: list[dict]
    queue: list[int]
    next_index: int
    parallel: bool
    branch_queues: dict[str, list[int]]
    branch_codebooks: dict[str, dict]
    codebook: dict
    segments: list[dict]
    memos: list[dict]
    highlight_ids: dict[str, str]
    counters: dict[str, int]
    pending: list[dict]
    saturation: list[dict]
    validation_log: list[dict]
    sweep_log: list[dict]
    prompt_log: list[dict]
    new_codes_since_sweep: int
    passages_since_sweep: int
    passages_coded: int
    metadata: dict
    warnings: list[str]
    export_paths: dict[str, str]
    finished: bool


@dataclass
class RunConfig:
    coder: BaseChatModel
    sweeper: BaseChatModel
    retriever: CodeRetriever
    embeddings: Any
    context: ResearchContext
    prompt_cache: bool
    top_k_codes: int = 12
    frequent_codes: int = 8
    merge_threshold: float = 0.82
    lexical_merge_threshold: float = 0.5
    max_codes_per_passage: int = 6
    neighbor_chars: int = 400
    long_passage_chars: int = 2000
    validation_retries: int = 1
    require_gerund: bool = False
    candidate_limit: int = 30
    model_name: str = ""
    sweep_model_name: str = ""
    prompt_log_path: str | None = None
    output_dir: str | None = None
    prompt_records: list = field(default_factory=list)
    _log_lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        if self._log_lock is None:
            self._log_lock = threading.Lock()


class CodingError(RuntimeError):
    """The model response could not be read as coding output."""


def initial_state(documents: list[LoadedDocument], metadata: dict) -> CodingState:
    return {
        "documents": [
            {"doc_id": doc.doc_id, "name": doc.name, "text": doc.text, "source": doc.source}
            for doc in documents
        ],
        "passages": [],
        "queue": [],
        "next_index": 0,
        "parallel": bool(metadata.get("parallel_documents")),
        "branch_queues": {},
        "branch_codebooks": {},
        "codebook": {},
        "segments": [],
        "memos": [],
        "highlight_ids": {},
        "counters": empty_counters(),
        "pending": [],
        "saturation": [],
        "validation_log": [],
        "sweep_log": [],
        "prompt_log": [],
        "new_codes_since_sweep": 0,
        "passages_since_sweep": 0,
        "passages_coded": 0,
        "metadata": metadata,
        "warnings": [],
        "export_paths": {},
        "finished": False,
    }


def build_graph(runtime: RunConfig):
    """Return an uncompiled graph. The caller attaches a checkpointer."""
    builder = StateGraph(CodingState)
    builder.add_node("segment_and_shuffle", _segment_node)
    builder.add_node("code_passage", _code_node(runtime))
    builder.add_node("validate_passage", _validate_node(runtime))
    builder.add_node("sweep", _sweep_node(runtime))
    builder.add_node("export", _export_node(runtime))
    builder.add_edge(START, "segment_and_shuffle")
    builder.add_conditional_edges(
        "segment_and_shuffle",
        _route_after_segment,
        {"code_passage": "code_passage", "export": "export"},
    )
    builder.add_edge("code_passage", "validate_passage")
    builder.add_conditional_edges(
        "validate_passage",
        _route_after_validate,
        {"code_passage": "code_passage", "sweep": "sweep", "export": "export"},
    )
    builder.add_conditional_edges(
        "sweep",
        _route_after_sweep,
        {"code_passage": "code_passage", "export": "export"},
    )
    builder.add_edge("export", END)
    return builder


def compile_graph(runtime: RunConfig, checkpointer, *, interrupt_before_sweep: bool = False):
    builder = build_graph(runtime)
    options = {}
    if interrupt_before_sweep:
        options["interrupt_before"] = ["sweep"]
    return builder.compile(checkpointer=checkpointer, **options)


def _route_after_segment(state: CodingState) -> str:
    if not state.get("passages"):
        return "export"
    return "code_passage"


def _remaining(state: CodingState) -> bool:
    if state.get("parallel"):
        queues = state.get("branch_queues") or {}
        return any(queues.values())
    return int(state.get("next_index") or 0) < len(state.get("queue") or [])


def _route_after_validate(state: CodingState) -> str:
    if not state.get("passages"):
        return "export"
    sweep_every = int(state.get("metadata", {}).get("sweep_every") or 1)
    due = int(state.get("passages_since_sweep") or 0) >= sweep_every
    remaining = _remaining(state)
    if due or not remaining:
        if int(state.get("passages_since_sweep") or 0) == 0:
            return "export"
        return "sweep"
    return "code_passage"


def _route_after_sweep(state: CodingState) -> str:
    if _remaining(state):
        return "code_passage"
    return "export"


def _segment_node(state: CodingState) -> dict:
    metadata = state["metadata"]
    documents = [
        LoadedDocument(
            doc_id=item["doc_id"],
            name=item["name"],
            text=item["text"],
            source=item.get("source") or "",
        )
        for item in state["documents"]
    ]
    # plan_coding loads via load_corpus, which would re-wrap mappings.
    # Documents are already loaded, so call the planner on mappings.
    plan = plan_coding(
        [{"id": doc.doc_id, "name": doc.name, "text": doc.text, "source": doc.source} for doc in documents],
        unit=metadata["unit"],
        seed=metadata["seed"],
        parallel_documents=bool(metadata.get("parallel_documents")),
    )
    branch_codebooks = {doc_id: {} for doc_id in plan.branch_queues} if plan.parallel else {}
    return {
        "passages": plan.passages,
        "queue": plan.queue,
        "next_index": 0,
        "parallel": plan.parallel,
        "branch_queues": plan.branch_queues,
        "branch_codebooks": branch_codebooks,
        "warnings": list(state.get("warnings") or []) + list(plan.warnings),
        "metadata": {**metadata, "seed": plan.seed, "passage_count": len(plan.passages)},
    }


def _code_node(runtime: RunConfig):
    async def code_passage(state: CodingState) -> dict:
        jobs = _jobs(state)
        if not jobs:
            return {"pending": []}

        async def one(doc_id: str | None, index: int) -> dict:
            passage = state["passages"][index]
            book = _book_for(state, doc_id)
            coding, attempt_errors = await _code_with_retries(runtime, passage, book)
            return {
                "passage_index": index,
                "doc_id": doc_id,
                "coding": coding,
                "attempt_errors": attempt_errors,
            }

        pending = list(await asyncio.gather(*(one(doc_id, index) for doc_id, index in jobs)))
        return {"pending": pending}

    return code_passage


def _jobs(state: CodingState) -> list[tuple[str | None, int]]:
    if state.get("parallel"):
        jobs = []
        for doc_id in sorted(state.get("branch_queues") or {}):
            queue = state["branch_queues"][doc_id]
            if queue:
                jobs.append((doc_id, queue[0]))
        return jobs
    index = int(state.get("next_index") or 0)
    queue = state.get("queue") or []
    if index >= len(queue):
        return []
    return [(None, queue[index])]


def _book_for(state: CodingState, doc_id: str | None) -> dict:
    if state.get("parallel") and doc_id is not None:
        return state.get("branch_codebooks", {}).get(doc_id) or {}
    return state.get("codebook") or {}


async def _code_with_retries(runtime: RunConfig, passage: dict, codebook: dict) -> tuple[dict, list]:
    feedback = None
    attempt_errors: list[list[dict]] = []
    coding: dict = {}
    attempts = max(0, runtime.validation_retries) + 1
    for attempt in range(attempts):
        coding = await _invoke_coding(runtime, passage, codebook, feedback)
        interpreted = interpret_coding(
            coding,
            passage,
            codebook,
            max_codes=runtime.max_codes_per_passage,
            require_gerund=runtime.require_gerund,
            long_passage_chars=runtime.long_passage_chars,
        )
        quote_failures = [item for item in interpreted["rejections"] if item.get("reason") == "quote_not_verbatim"]
        if not quote_failures or attempt == attempts - 1:
            break
        attempt_errors.append(quote_failures)
        feedback = validation_feedback(quote_failures)
    return coding, attempt_errors


async def _invoke_coding(runtime: RunConfig, passage: dict, codebook: dict, feedback: str | None) -> dict:
    ranked = runtime.retriever.rank(str(passage.get("text") or ""), list(codebook.values()), k=runtime.top_k_codes)
    view, omitted = select_prompt_codes(list(codebook.values()), ranked, frequent_limit=runtime.frequent_codes)
    messages = build_coding_messages(
        context=runtime.context,
        codebook_view=view,
        omitted_codes=omitted,
        passage=passage,
        neighbor_chars=runtime.neighbor_chars,
        feedback=feedback,
        prompt_cache=runtime.prompt_cache,
    )
    parsed = await _ainvoke_structured(
        runtime.coder,
        PassageCoding,
        messages,
        kind="code_passage",
        model_name=runtime.model_name,
        passage_id=str(passage.get("passage_id")),
        runtime=runtime,
    )
    return parsed


def _validate_node(runtime: RunConfig):
    def validate_passage(state: CodingState) -> dict:
        pending = list(state.get("pending") or [])
        if not pending:
            return {}
        codebook = dict(state.get("codebook") or {})
        segments = list(state.get("segments") or [])
        memos = list(state.get("memos") or [])
        highlights = dict(state.get("highlight_ids") or {})
        counters = dict(state.get("counters") or empty_counters())
        validation_log = list(state.get("validation_log") or [])
        prompt_log = _take_prompts(runtime, state)
        branch_queues = {key: list(value) for key, value in (state.get("branch_queues") or {}).items()}
        branch_books = {key: dict(value) for key, value in (state.get("branch_codebooks") or {}).items()}
        next_index = int(state.get("next_index") or 0)
        new_codes = int(state.get("new_codes_since_sweep") or 0)
        passages_since = int(state.get("passages_since_sweep") or 0)
        passages_coded = int(state.get("passages_coded") or 0)
        global_keys = set(codebook)
        if state.get("parallel"):
            global_keys = set()
            for book in branch_books.values():
                global_keys.update(book)

        # Apply in a stable order.
        pending_sorted = sorted(pending, key=lambda item: (item.get("doc_id") or "", item.get("passage_index") or 0))
        for item in pending_sorted:
            passage = state["passages"][item["passage_index"]]
            doc_id = item.get("doc_id")
            book = branch_books.get(doc_id, {}) if state.get("parallel") and doc_id else codebook
            interpreted = interpret_coding(
                item["coding"],
                passage,
                book,
                max_codes=runtime.max_codes_per_passage,
                require_gerund=runtime.require_gerund,
                long_passage_chars=runtime.long_passage_chars,
            )
            for earlier in item.get("attempt_errors") or []:
                validation_log.append(
                    {
                        "passage_id": passage.get("passage_id"),
                        "doc_id": passage.get("doc_id"),
                        "stage": "retry",
                        "rejections": earlier,
                    }
                )
            if interpreted["rejections"]:
                validation_log.append(
                    {
                        "passage_id": passage.get("passage_id"),
                        "doc_id": passage.get("doc_id"),
                        "stage": "final",
                        "rejections": interpreted["rejections"],
                    }
                )
            committed = commit_passage(
                book,
                segments,
                memos,
                highlights,
                counters,
                passage,
                interpreted,
                existing_keys=global_keys,
            )
            segments = committed["segments"]
            memos = committed["memos"]
            highlights = committed["highlight_ids"]
            counters = committed["counters"]
            new_codes += committed["new_codes"]
            global_keys.update(committed["codebook"])
            if state.get("parallel") and doc_id:
                branch_books[doc_id] = _resync(
                    committed["codebook"],
                    [segment for segment in segments if segment.get("doc_id") == doc_id],
                )
                queue = branch_queues.get(doc_id) or []
                if queue and queue[0] == item["passage_index"]:
                    branch_queues[doc_id] = queue[1:]
            else:
                codebook = committed["codebook"]
                next_index += 1
            passages_since += 1
            passages_coded += 1

        if state.get("parallel"):
            codebook = union_codebooks(list(branch_books.values()))
            codebook = _resync(codebook, segments)

        return {
            "codebook": codebook,
            "segments": segments,
            "memos": memos,
            "highlight_ids": highlights,
            "counters": counters,
            "validation_log": validation_log,
            "prompt_log": prompt_log,
            "branch_queues": branch_queues,
            "branch_codebooks": branch_books,
            "next_index": next_index,
            "new_codes_since_sweep": new_codes,
            "passages_since_sweep": passages_since,
            "passages_coded": passages_coded,
            "pending": [],
        }

    return validate_passage


def _resync(codebook: dict, segments: list[dict]) -> dict:
    from automated_grounded_theory.codebook import sync_counts

    return sync_counts(codebook, segments)


def _sweep_node(runtime: RunConfig):
    async def sweep(state: CodingState) -> dict:
        codebook = dict(state.get("codebook") or {})
        if state.get("parallel") and state.get("branch_codebooks"):
            codebook = union_codebooks(list(state["branch_codebooks"].values()))
            codebook = _resync(codebook, state.get("segments") or [])
        passages_since = int(state.get("passages_since_sweep") or 0)
        new_codes = int(state.get("new_codes_since_sweep") or 0)
        counters = dict(state.get("counters") or empty_counters())
        sweep_index = counters.get("sweep", 0) + 1
        counters["sweep"] = sweep_index
        memos = list(state.get("memos") or [])
        sweep_log = list(state.get("sweep_log") or [])
        segments = list(state.get("segments") or [])

        def _point(book: dict) -> dict:
            return saturation_point(
                batch=sweep_index,
                passages_in_batch=passages_since,
                new_codes=new_codes,
                cumulative_passages=int(state.get("passages_coded") or 0),
                cumulative_codes=len(book),
            )

        if not codebook:
            memo_id = f"m{counters.get('memo', 0) + 1:05d}"
            counters["memo"] = counters.get("memo", 0) + 1
            memos.append(
                {
                    "memo_id": memo_id,
                    "kind": "sweep",
                    "doc_id": None,
                    "passage_id": None,
                    "sweep_index": sweep_index,
                    "text": "Sweep skipped: the codebook is empty.",
                }
            )
            return _sweep_update(
                state,
                codebook=codebook,
                segments=segments,
                memos=memos,
                counters=counters,
                saturation=list(state.get("saturation") or []) + [_point(codebook)],
                sweep_log=sweep_log
                + [{"sweep_index": sweep_index, "candidates": [], "accepted_merges": [], "rejected_merges": []}],
                prompt_log=_take_prompts(runtime, state),
            )

        codes = list(codebook.values())
        codes.sort(key=lambda entry: (-int(entry.get("frequency") or 0), str(entry.get("code") or "").casefold()))
        truncated = codes[:80]
        candidates = merge_candidates(
            truncated,
            embeddings=runtime.embeddings,
            embedding_threshold=runtime.merge_threshold,
            lexical_threshold=runtime.lexical_merge_threshold,
            limit=runtime.candidate_limit,
        )
        messages = build_sweep_messages(
            context=runtime.context,
            codes=truncated,
            candidates=candidates,
            prompt_cache=runtime.prompt_cache,
        )
        parsed = await _ainvoke_structured(
            runtime.sweeper,
            SweepCoding,
            messages,
            kind="sweep",
            model_name=runtime.sweep_model_name,
            passage_id=f"sweep-{sweep_index}",
            runtime=runtime,
        )
        prompt_log = _take_prompts(runtime, state)
        applied = apply_sweep(
            codebook,
            segments,
            merges=list(parsed.get("merges") or []),
            candidates=candidates,
            definitions=list(parsed.get("definitions") or []),
            overloaded=list(parsed.get("overloaded_codes") or []),
            relation_hints=list(parsed.get("relation_hints") or []),
        )
        for merge in applied["accepted_merges"]:
            logger.info("merged code %s into %s", merge.get("source"), merge.get("target"))
        integrative = str(parsed.get("integrative_memo") or "").strip()
        reflexive = str(parsed.get("reflexive_memo") or "").strip()
        overloaded = [str(item) for item in (parsed.get("overloaded_codes") or [])]
        if overloaded:
            integrative = (
                integrative
                + "\n\nOverloaded codes were flagged and not split, because earlier segments "
                "cannot be reassigned without rereading: "
                + ", ".join(overloaded)
            ).strip()
        memo_text = integrative or "Sweep compared the codebook and made no further note."
        memo_id = f"m{counters.get('memo', 0) + 1:05d}"
        counters["memo"] = counters.get("memo", 0) + 1
        memos.append(
            {
                "memo_id": memo_id,
                "kind": "sweep",
                "doc_id": None,
                "passage_id": None,
                "sweep_index": sweep_index,
                "text": memo_text,
            }
        )
        if reflexive:
            memo_id = f"m{counters.get('memo', 0) + 1:05d}"
            counters["memo"] = counters.get("memo", 0) + 1
            memos.append(
                {
                    "memo_id": memo_id,
                    "kind": "reflexive",
                    "doc_id": None,
                    "passage_id": None,
                    "sweep_index": sweep_index,
                    "text": reflexive,
                }
            )
        record = {
            "sweep_index": sweep_index,
            "candidate_method": candidates[0]["method"] if candidates else ("embedding" if runtime.embeddings else "lexical"),
            "candidates": candidates,
            "accepted_merges": applied["accepted_merges"],
            "rejected_merges": applied["rejected_merges"],
            "merge_map": applied["merge_map"],
            "relation_hints": applied["relation_hints"],
            "overloaded_codes": overloaded,
            "codes_sent": len(truncated),
        }
        branch_books = state.get("branch_codebooks") or {}
        if state.get("parallel"):
            branch_books = {doc_id: json.loads(json.dumps(applied["codebook"])) for doc_id in branch_books}
        return _sweep_update(
            state,
            codebook=applied["codebook"],
            segments=applied["segments"],
            memos=memos,
            counters=counters,
            saturation=list(state.get("saturation") or []) + [_point(applied["codebook"])],
            sweep_log=sweep_log + [record],
            branch_codebooks=branch_books,
            prompt_log=prompt_log,
        )

    return sweep


def _sweep_update(
    state: CodingState,
    *,
    codebook: dict,
    segments: list,
    memos: list,
    counters: dict,
    saturation: list,
    sweep_log: list,
    branch_codebooks: dict | None = None,
    prompt_log: list | None = None,
) -> dict:
    return {
        "codebook": codebook,
        "segments": segments,
        "memos": memos,
        "counters": counters,
        "saturation": saturation,
        "sweep_log": sweep_log,
        "prompt_log": prompt_log if prompt_log is not None else list(state.get("prompt_log") or []),
        "new_codes_since_sweep": 0,
        "passages_since_sweep": 0,
        "branch_codebooks": branch_codebooks if branch_codebooks is not None else state.get("branch_codebooks") or {},
    }


def _export_node(runtime: RunConfig):
    def export_node(state: CodingState) -> dict:
        provisional = result_from_state(state, interrupted=False, thread_id=str(state.get("metadata", {}).get("thread_id") or ""), resumable=False)
        paths: dict[str, str] = {}
        if runtime.output_dir:
            paths = export_result(provisional, runtime.output_dir)
        return {"export_paths": paths, "finished": True}

    return export_node


async def _ainvoke_structured(
    model: BaseChatModel,
    schema,
    messages: list,
    *,
    kind: str,
    model_name: str,
    passage_id: str,
    runtime: RunConfig,
) -> dict:
    try:
        structured = model.with_structured_output(schema)
        result = await structured.ainvoke(messages)
    except Exception as exc:
        raise CodingError(f"{kind} failed for {passage_id}: {exc}") from exc
    if result is None:
        raise CodingError(f"{kind} for {passage_id} returned no structured output.")
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    elif isinstance(result, dict):
        parsed = result.get("parsed", result)
        if hasattr(parsed, "model_dump"):
            payload = parsed.model_dump()
        else:
            payload = schema.model_validate(parsed).model_dump()
    else:
        raise CodingError(f"{kind} for {passage_id} returned {type(result).__name__}, not {schema.__name__}.")
    entry = {
        "kind": kind,
        "passage_id": passage_id,
        "model": model_name,
        "messages": serialize_messages(messages),
        "output": payload,
    }
    _append_prompt_log(runtime, entry)
    return payload


def _take_prompts(runtime: RunConfig, state: CodingState) -> list:
    assert runtime._log_lock is not None
    with runtime._log_lock:
        fresh = list(runtime.prompt_records)
        runtime.prompt_records.clear()
    return list(state.get("prompt_log") or []) + fresh


def _append_prompt_log(runtime: RunConfig, entry: dict) -> None:
    assert runtime._log_lock is not None
    line = json.dumps(entry, ensure_ascii=False)
    with runtime._log_lock:
        runtime.prompt_records.append(entry)
        if not runtime.prompt_log_path:
            return
        path = Path(runtime.prompt_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def prompt_templates() -> dict[str, str]:
    return {"coding_system": CODING_SYSTEM, "sweep_system": SWEEP_SYSTEM}
