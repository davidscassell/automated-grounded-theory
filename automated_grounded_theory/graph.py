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
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from automated_grounded_theory.async_support import ainvoke_structured, is_transient
from automated_grounded_theory.batching import pack_indices
from automated_grounded_theory.codebook import (
    agreement_confidence,
    apply_post_merge,
    apply_sweep,
    commit_passage,
    empty_counters,
    frozen_pair_keys,
    interpret_coding,
    remember_pair_decisions,
    replay_accepted_pairs,
    trivial_merges,
    union_codebooks,
)
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.corpus import LoadedDocument
from automated_grounded_theory.export import export_result
from automated_grounded_theory.prompts import (
    CODING_SYSTEM,
    SWEEP_SYSTEM,
    build_coding_messages,
    build_merge_check_messages,
    build_rewrite_messages,
    build_sweep_messages,
    serialize_messages,
    validation_feedback,
)
from automated_grounded_theory.results import result_from_state
from automated_grounded_theory.retrieval import (
    CodeRetriever,
    EmbeddingCodeRetriever,
    LexicalCodeRetriever,
    aembed_texts,
    code_text,
    comparison_order,
    hint_pairs,
    select_prompt_codes,
)
from automated_grounded_theory.validation import code_key, context_term_set, label_restates_context
from automated_grounded_theory.saturation import saturation_point
from automated_grounded_theory.schemas import (
    BatchCoding,
    DefinitionRewriteBatch,
    LeakCheckBatch,
    MergeCheckBatch,
    QuoteFitBatch,
    SweepCoding,
)
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
    ops: list[dict]
    review: list[dict]
    pair_decisions: list[dict]
    consolidated: bool
    stage_seconds: dict


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
    merge_threshold: float = 0.65
    lexical_merge_threshold: float = 0.5
    max_codes_per_passage: int = 6
    neighbor_chars: int = 400
    long_passage_chars: int = 2000
    validation_retries: int = 1
    require_gerund: bool = False
    candidate_limit: int = 30
    batch_size: int = 8
    concurrency: int = 2
    sweep_chunk: int = 20
    embed_batch_size: int = 64
    review_mode: str = "document"
    review_below: float = 0.5
    min_merge_confidence: float = 0.5
    max_merges_per_sweep: int = 12
    check_model: Any = None
    pipeline: bool = True
    token_budget: int = 4000
    model_name: str = ""
    sweep_model_name: str = ""
    prompt_log_path: str | None = None
    output_dir: str | None = None
    limiter: Any = None
    prompt_records: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    _log_lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        if self._log_lock is None:
            self._log_lock = threading.Lock()
        if self.limiter is None:
            from automated_grounded_theory.async_support import CallLimiter

            self.limiter = CallLimiter(self.concurrency)


class CodingError(RuntimeError):
    """The model response could not be read as coding output."""


def initial_state(documents: list[LoadedDocument], metadata: dict) -> CodingState:
    return {
        "documents": [
            {
                "doc_id": doc.doc_id,
                "name": doc.name,
                "text": doc.text,
                "source": doc.source,
                "metadata": dict(doc.metadata),
            }
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
        "ops": [],
        "review": [],
        "pair_decisions": [],
        "consolidated": False,
        "stage_seconds": {},
    }


def build_graph(runtime: RunConfig):
    """Return an uncompiled graph. The caller attaches a checkpointer."""
    builder = StateGraph(CodingState)
    builder.add_node("segment_and_shuffle", _segment_node)
    builder.add_node("code_passage", _code_node(runtime))
    builder.add_node("validate_passage", _validate_node(runtime))
    builder.add_node("sweep", _sweep_node(runtime))
    builder.add_node("consolidate", _sweep_node(runtime, consolidation=True))
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
        {"code_passage": "code_passage", "consolidate": "consolidate", "export": "export"},
    )
    builder.add_edge("consolidate", "export")
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
    if not state.get("consolidated") and len(state.get("codebook") or {}) >= 2:
        return "consolidate"
    return "export"


def _segment_node(state: CodingState) -> dict:
    metadata = state["metadata"]
    documents = [
        LoadedDocument(
            doc_id=item["doc_id"],
            name=item["name"],
            text=item["text"],
            source=item.get("source") or "",
            metadata=dict(item.get("metadata") or {}),
        )
        for item in state["documents"]
    ]
    # plan_coding loads via load_corpus, which would re-wrap mappings.
    # Documents are already loaded, so call the planner on mappings.
    plan = plan_coding(
        [
            {
                "id": doc.doc_id,
                "name": doc.name,
                "text": doc.text,
                "source": doc.source,
                "metadata": doc.metadata,
            }
            for doc in documents
        ],
        unit=metadata["unit"],
        seed=metadata["seed"],
        parallel_documents=bool(metadata.get("parallel_documents")),
        review_mode=str(metadata.get("review_mode") or "document"),
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
        started = time.perf_counter()
        jobs = _wave(state, runtime)
        if not jobs:
            update: dict = {"pending": []}
        else:
            rows = await _pending_rows(runtime, state, jobs)
            update = {"pending": rows}
        timing = dict(state.get("stage_seconds") or {})
        timing["coding"] = round(float(timing.get("coding") or 0) + time.perf_counter() - started, 4)
        update["stage_seconds"] = timing
        return update

    return code_passage


def _wave(
    state: CodingState,
    runtime: RunConfig,
    *,
    upcoming: bool = False,
) -> list[tuple[str | None, list[int]]]:
    """One coding step against a frozen codebook snapshot.

    Document mode (the default) takes the next document only, in reading
    order, and packs its passages into token-bounded batches. Those batches
    share the snapshot; the codebook is committed after the document.

    Wave mode may mix passages from several documents. It is opt-in because
    that weakens constant comparison.

    The step stops at the next sweep. ``upcoming`` plans the step that will
    run during that sweep, so the sweep window is treated as already reset.
    """
    metadata = state.get("metadata") or {}
    batch_size = max(1, int(metadata.get("batch_size") or runtime.batch_size))
    concurrency = max(1, int(metadata.get("concurrency") or runtime.concurrency))
    sweep_every = max(1, int(metadata.get("sweep_every") or 1))
    since = 0 if upcoming else int(state.get("passages_since_sweep") or 0)
    room = sweep_every - since
    if room <= 0:
        return []
    token_budget = max(1, int(metadata.get("token_budget") or runtime.token_budget))
    review_mode = str(metadata.get("review_mode") or runtime.review_mode or "document")

    if state.get("parallel"):
        jobs: list[tuple[str | None, list[int]]] = []
        used = 0
        for doc_id in sorted(state.get("branch_queues") or {}):
            if len(jobs) >= concurrency or used >= room:
                break
            queue = list(state["branch_queues"][doc_id])
            if not queue:
                continue
            take = min(batch_size, len(queue), room - used)
            if take <= 0:
                break
            jobs.append((doc_id, queue[:take]))
            used += take
        return jobs

    index = int(state.get("next_index") or 0)
    queue = list(state.get("queue") or [])
    if index >= len(queue):
        return []
    if review_mode == "document":
        # Stay inside one document. ``batch_size`` 1 commits after every passage,
        # so the next passage is compared with codes just earned. A larger batch
        # codes several passages of this document against one frozen snapshot.
        doc_id = state["passages"][queue[index]].get("doc_id")
        cap = min(room, batch_size * concurrency)
        indices = []
        for passage_index in queue[index:]:
            if state["passages"][passage_index].get("doc_id") != doc_id:
                break
            if len(indices) >= cap:
                break
            indices.append(passage_index)
    else:
        take = min(room, batch_size * concurrency, len(queue) - index)
        indices = queue[index : index + take]
    texts = [str(state["passages"][item].get("text") or "") for item in range(len(state["passages"]))]
    packed = pack_indices(indices, texts, batch_size=batch_size, token_budget=token_budget)
    return [(None, batch) for batch in packed if batch]


async def _pending_rows(runtime: RunConfig, state: CodingState, jobs: list[tuple[str | None, list[int]]]) -> list[dict]:
    """Code every batch in ``jobs`` against the codebook frozen on ``state``."""

    async def one(doc_id: str | None, indices: list[int]) -> list[dict]:
        passages = [state["passages"][index] for index in indices]
        book = _book_for(state, doc_id)
        coded = await _code_batch(runtime, passages, book)
        rows = []
        for index in indices:
            passage = state["passages"][index]
            coding, attempt_errors = coded[str(passage.get("passage_id"))]
            rows.append(
                {
                    "passage_index": index,
                    "doc_id": doc_id,
                    "coding": coding,
                    "attempt_errors": attempt_errors,
                }
            )
        return rows

    groups = await asyncio.gather(*(one(doc_id, indices) for doc_id, indices in jobs))
    return [row for group in groups for row in group]


def _book_for(state: CodingState, doc_id: str | None) -> dict:
    if state.get("parallel") and doc_id is not None:
        return state.get("branch_codebooks", {}).get(doc_id) or {}
    return state.get("codebook") or {}


async def _code_batch(runtime: RunConfig, passages: list[dict], codebook: dict) -> dict[str, tuple[dict, list]]:
    """Code a batch. Quote failures are retried on their own, not with the batch."""
    pending = list(passages)
    feedback: dict[str, str] = {}
    errors: dict[str, list] = {str(passage.get("passage_id")): [] for passage in passages}
    done: dict[str, tuple[dict, list]] = {}
    attempts = max(0, runtime.validation_retries) + 1
    for attempt in range(attempts):
        if not pending:
            break
        aligned = await _invoke_coding(runtime, pending, codebook, feedback or None)
        still_pending = []
        feedback = {}
        for passage, coding in zip(pending, aligned):
            passage_id = str(passage.get("passage_id"))
            interpreted = interpret_coding(
                coding,
                passage,
                codebook,
                max_codes=runtime.max_codes_per_passage,
                require_gerund=runtime.require_gerund,
                long_passage_chars=runtime.long_passage_chars,
                review_below=runtime.review_below,
                context_terms=context_term_set(runtime.context),
            )
            quote_failures = [
                item for item in interpreted["rejections"] if item.get("reason") == "quote_not_verbatim"
            ]
            restated = [
                {"code": item.get("code"), "reason": "context_restatement", "quote": item.get("quote")}
                for item in interpreted["accepted"]
                if "context_restatement" in (item.get("review_reasons") or [])
            ]
            retryable = quote_failures + restated
            if retryable and attempt < attempts - 1:
                errors[passage_id].append(retryable)
                still_pending.append(passage)
                feedback[passage_id] = validation_feedback(retryable)
            else:
                done[passage_id] = (coding, errors[passage_id])
        pending = still_pending
    if runtime.check_model is not None:
        await _annotate_leaks(runtime, done)
    for passage in passages:
        passage_id = str(passage.get("passage_id"))
        done.setdefault(passage_id, ({"passage_id": passage_id, "codes": [], "memo": ""}, errors[passage_id]))
    return done


async def _invoke_coding(
    runtime: RunConfig,
    passages: list[dict],
    codebook: dict,
    feedback: dict[str, str] | None,
) -> list[dict]:
    ranked: list[str] = []
    seen: set[str] = set()
    entries = list(codebook.values())
    if isinstance(runtime.retriever, EmbeddingCodeRetriever):
        try:
            per_passage = await runtime.retriever.arank_many(
                [str(passage.get("text") or "") for passage in passages],
                entries,
                k=runtime.top_k_codes,
                limiter=runtime.limiter,
                batch_size=runtime.embed_batch_size,
            )
        except Exception as exc:
            if not is_transient(exc):
                raise
            runtime.notes.append(
                f"Code retrieval embeddings failed ({exc}). Nearest codes used label overlap instead."
            )
            lexical = LexicalCodeRetriever()
            per_passage = [
                lexical.rank(str(passage.get("text") or ""), entries, k=runtime.top_k_codes)
                for passage in passages
            ]
    else:
        per_passage = [
            runtime.retriever.rank(str(passage.get("text") or ""), entries, k=runtime.top_k_codes)
            for passage in passages
        ]
    for labels in per_passage:
        for label in labels:
            key = label.casefold()
            if key not in seen:
                seen.add(key)
                ranked.append(label)
    view, omitted = select_prompt_codes(entries, ranked, frequent_limit=runtime.frequent_codes)
    messages = build_coding_messages(
        context=runtime.context,
        codebook_view=view,
        omitted_codes=omitted,
        passages=passages,
        neighbor_chars=runtime.neighbor_chars,
        feedback=feedback,
        prompt_cache=runtime.prompt_cache,
        nearest=per_passage,
    )
    passage_ids = ",".join(str(passage.get("passage_id")) for passage in passages)
    parsed = await _ainvoke_structured(
        runtime.coder,
        BatchCoding,
        messages,
        kind="code_passage",
        model_name=runtime.model_name,
        passage_id=passage_ids,
        runtime=runtime,
    )
    return _align_batch(parsed, passages)


def _align_batch(payload: dict, passages: list[dict]) -> list[dict]:
    items = list(payload.get("passages") or [])
    by_id = {str(item.get("passage_id")): item for item in items if item.get("passage_id")}
    aligned = []
    for index, passage in enumerate(passages):
        passage_id = str(passage.get("passage_id") or "")
        item = by_id.get(passage_id)
        if item is None and index < len(items):
            item = items[index]
        if item is None:
            item = {"passage_id": passage_id, "codes": [], "memo": "", "low_confidence": True}
        aligned.append(item)
    return aligned


def _validate_node(runtime: RunConfig):
    def validate_passage(state: CodingState, *, pending: list | None = None, take_prompts: bool = True) -> dict:
        if pending is None:
            pending = list(state.get("pending") or [])
        if not pending:
            return {}
        codebook = dict(state.get("codebook") or {})
        segments = list(state.get("segments") or [])
        memos = list(state.get("memos") or [])
        highlights = dict(state.get("highlight_ids") or {})
        counters = dict(state.get("counters") or empty_counters())
        validation_log = list(state.get("validation_log") or [])
        ops = list(state.get("ops") or [])
        review = list(state.get("review") or [])
        prompt_log = _take_prompts(runtime, state) if take_prompts else list(state.get("prompt_log") or [])
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

        # Commit in seeded coding order so earlier passages in a wave land first.
        pending_sorted = sorted(
            pending,
            key=lambda item: state["passages"][item["passage_index"]].get("coding_order", item["passage_index"]),
        )
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
                review_below=runtime.review_below,
                context_terms=context_term_set(runtime.context),
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
            ops.extend(committed.get("ops") or [])
            review.extend(item for item in (committed.get("ops") or []) if item.get("status") == "review")
            new_codes += committed["new_codes"]
            global_keys.update(committed["codebook"])
            if state.get("parallel") and doc_id:
                branch_books[doc_id] = _resync(
                    committed["codebook"],
                    [segment for segment in segments if segment.get("doc_id") == doc_id],
                )
                branch_queues[doc_id] = [
                    index for index in (branch_queues.get(doc_id) or []) if index != item["passage_index"]
                ]
            else:
                codebook = committed["codebook"]
                next_index += 1
            passages_since += 1
            passages_coded += 1

        _append_reading_notes(memos, counters, pending_sorted, state)
        notes = list(runtime.notes)
        runtime.notes.clear()

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
            "ops": ops,
            "review": review,
            "warnings": list(state.get("warnings") or []) + notes,
        }

    runtime._commit = validate_passage
    return validate_passage


def _append_reading_notes(memos: list, counters: dict, pending_sorted: list, state: CodingState) -> None:
    """One comparison memo and one reflexivity note per document in this commit.

    The comparison memo quotes the incident memos the model wrote. It does not
    call the model again. The reflexivity note records that this document was
    read in seeded order and was not skipped.
    """
    groups: list[tuple[str, str, list]] = []
    for item in pending_sorted:
        passage = state["passages"][item["passage_index"]]
        doc_id = str(passage.get("doc_id") or "")
        if not groups or groups[-1][0] != doc_id:
            groups.append((doc_id, str(passage.get("doc_name") or doc_id), []))
        groups[-1][2].append(passage)
    for doc_id, doc_name, passages in groups:
        passage_ids = {str(passage.get("passage_id") or "") for passage in passages}
        from automated_grounded_theory.validation import substantive_memo

        comparisons = [
            str(memo.get("text") or "").strip()
            for memo in memos
            if memo.get("kind") == "passage"
            and str(memo.get("passage_id") or "") in passage_ids
            and substantive_memo(str(memo.get("text") or ""))
        ]
        listed = ", ".join(str(passage.get("passage_id") or "") for passage in passages)
        if comparisons:
            memo_id = f"m{counters.get('memo', 0) + 1:05d}"
            counters["memo"] = counters.get("memo", 0) + 1
            memos.append(
                {
                    "memo_id": memo_id,
                    "kind": "document",
                    "doc_id": doc_id,
                    "passage_id": None,
                    "sweep_index": None,
                    "text": f"While reading {doc_name} ({listed}):\n" + "\n\n".join(comparisons),
                }
            )
        memo_id = f"m{counters.get('memo', 0) + 1:05d}"
        counters["memo"] = counters.get("memo", 0) + 1
        memos.append(
            {
                "memo_id": memo_id,
                "kind": "reflexive",
                "doc_id": doc_id,
                "passage_id": None,
                "sweep_index": None,
                "text": (
                    f"Read {doc_name} in seeded document order before later documents. "
                    f"Passages reviewed: {listed}. "
                    "The research context sensitized attention and was not used as a code list."
                ),
            }
        )


def _resync(codebook: dict, segments: list[dict]) -> dict:
    from automated_grounded_theory.codebook import sync_counts

    return sync_counts(codebook, segments)


def _sweep_node(runtime: RunConfig, *, consolidation: bool = False):
    async def sweep(state: CodingState) -> dict:
        started = time.perf_counter()
        extras = {"rewrites": 0.0}

        def finish(update: dict) -> dict:
            timing = dict(state.get("stage_seconds") or {})
            elapsed = time.perf_counter() - started
            timing["sweeps"] = round(float(timing.get("sweeps") or 0) + elapsed, 4)
            timing["rewrites"] = round(float(timing.get("rewrites") or 0) + extras["rewrites"], 4)
            if consolidation:
                timing["consolidation"] = round(float(timing.get("consolidation") or 0) + elapsed, 4)
            update["stage_seconds"] = timing
            update["consolidated"] = True if consolidation else bool(state.get("consolidated"))
            return update
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

        prior_points = list(state.get("saturation") or [])
        prior_codes = 0
        if prior_points:
            prior_codes = int(
                prior_points[-1].get("cumulative_codes_post_merge", prior_points[-1].get("cumulative_codes")) or 0
            )

        def _point(book: dict, *, pre_len: int | None = None) -> dict:
            post_len = len(book)
            return saturation_point(
                batch=sweep_index,
                passages_in_batch=passages_since,
                new_codes=new_codes,
                cumulative_passages=int(state.get("passages_coded") or 0),
                cumulative_codes=post_len,
                cumulative_codes_pre_merge=post_len if pre_len is None else pre_len,
                new_codes_post_merge=post_len - prior_codes,
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
            return finish(
                _sweep_update(
                    state,
                    codebook=codebook,
                    segments=segments,
                    memos=memos,
                    counters=counters,
                    saturation=list(state.get("saturation") or []) + [_point(codebook)],
                    sweep_log=sweep_log
                    + [{"sweep_index": sweep_index, "candidates": [], "accepted_merges": [], "rejected_merges": []}],
                    prompt_log=_take_prompts(runtime, state),
                    warnings=list(state.get("warnings") or []),
                )
            )

        code_task = None
        if runtime.pipeline and _remaining(state):
            jobs = _wave(state, runtime, upcoming=True)
            if jobs:
                code_task = asyncio.create_task(_pending_rows(runtime, state, jobs))
        codes = comparison_order(list(codebook.values()))
        frozen = frozen_pair_keys(list(state.get("pair_decisions") or []), codebook)
        try:
            parsed, candidates, chunk_count = await _sweep_chunks(
                runtime, codes, sweep_index, frozen=frozen, consolidation=consolidation
            )
            overlap_pending: list = []
            if code_task is not None:
                overlap_pending = await code_task
                code_task = None
        finally:
            if code_task is not None and not code_task.done():
                code_task.cancel()
        overlap_state = None
        if overlap_pending and getattr(runtime, "_commit", None):
            overlap_state = runtime._commit(state, pending=overlap_pending, take_prompts=False)
            codebook = overlap_state["codebook"]
            segments = overlap_state["segments"]
            memos = overlap_state["memos"]
            counters = dict(overlap_state["counters"])
            counters["sweep"] = sweep_index
        prompt_log = _take_prompts(runtime, state)
        from automated_grounded_theory.codebook import finalize_candidate_decisions

        pre_merge_book = codebook
        pre_len = len(codebook)
        cache = list(state.get("pair_decisions") or [])
        frozen = frozen_pair_keys(cache, codebook)
        replay = replay_accepted_pairs(cache, codebook)
        replay_pairs = {
            frozenset((code_key(str(item.get("source") or "")), code_key(str(item.get("target") or ""))))
            for item in replay
        }
        def _stamp(merge) -> dict:
            row = dict(merge)
            agreed = agreement_confidence(row)
            row["model_confidence"] = row.get("confidence", 1.0)
            if agreed is not None:
                row["agreement_confidence"] = agreed
            elif row.get("agreement_confidence") is None:
                row["agreement_confidence"] = row["model_confidence"]
            source_key = code_key(str(row.get("source") or ""))
            target_key = code_key(str(row.get("target") or ""))
            if not row.get("source_valence"):
                row["source_valence"] = str(codebook.get(source_key, {}).get("valence") or "")
            if not row.get("target_valence"):
                row["target_valence"] = str(codebook.get(target_key, {}).get("valence") or "")
            if not str(row.get("rationale") or "").strip():
                row["rationale"] = (
                    "The sweep judged this pair with the same-action, same-object, same-stance rubric."
                )
            return row

        proposed = []
        already = []
        seen_pairs: set[frozenset[str]] = set()
        for merge in parsed.get("merges") or []:
            pair = frozenset(
                (code_key(str(merge.get("source") or "")), code_key(str(merge.get("target") or "")))
            )
            if pair in replay_pairs:
                continue
            if pair in frozen:
                already.append(
                    {
                        **dict(merge),
                        "reason": "already_decided",
                        "decision": "reject",
                        "confidence": merge.get("confidence", 1.0),
                        "rationale": merge.get("rationale")
                        or "This pair was already decided in this run, and neither definition has changed in substance.",
                    }
                )
            else:
                proposed.append(_stamp(merge))
                seen_pairs.add(pair)
        for merge in trivial_merges(list(codebook.values()), frozen):
            pair = frozenset((code_key(str(merge.get("source") or "")), code_key(str(merge.get("target") or ""))))
            if pair in seen_pairs or pair in replay_pairs:
                continue
            proposed.append(merge)
            seen_pairs.add(pair)
        pre_segments = list(segments)
        rewrite_started = time.perf_counter()
        applied = apply_sweep(
            codebook,
            segments,
            merges=list(proposed) + [_stamp(item) for item in replay],
            candidates=candidates,
            definitions=list(parsed.get("definitions") or []),
            overloaded=list(parsed.get("overloaded_codes") or []),
            relation_hints=list(parsed.get("relation_hints") or []),
            min_merge_confidence=runtime.min_merge_confidence,
            review_below=runtime.review_below,
            frozen_pairs=frozen - replay_pairs,
            max_merges_per_sweep=runtime.max_merges_per_sweep,
        )
        applied["rejected_merges"] = list(already) + list(applied.get("rejected_merges") or [])
        book, rows, post_ops, quote_stats = apply_post_merge(
            applied["codebook"],
            applied["segments"],
            applied["accepted_merges"],
            pre_merge_book,
            pre_segments,
        )
        applied["codebook"] = book
        applied["segments"] = rows
        applied.setdefault("ops", []).extend(post_ops)
        applied.setdefault("review", []).extend(op for op in post_ops if op.get("status") == "review")
        if runtime.check_model is not None and applied["accepted_merges"] and quote_stats["quotes_checked"] == 0:
            checked = await _check_quotes(runtime, applied["codebook"], applied["segments"], applied["accepted_merges"], sweep_index)
            if checked:
                book, rows, more_ops, extra = apply_post_merge(
                    applied["codebook"],
                    applied["segments"],
                    checked,
                    pre_merge_book,
                    pre_segments,
                )
                applied["codebook"] = book
                applied["segments"] = rows
                applied.setdefault("ops", []).extend(more_ops)
                quote_stats["quotes_checked"] += extra["quotes_checked"]
                quote_stats["quotes_split"] += extra["quotes_split"]
        extras["rewrites"] = time.perf_counter() - rewrite_started
        finalize_candidate_decisions(
            applied,
            candidates,
            rejections=list(parsed.get("rejections") or []),
            skipped=list(parsed.get("skipped_candidates") or []),
        )
        warnings = list(state.get("warnings") or []) + list(parsed.get("warnings") or []) + list(runtime.notes)
        runtime.notes.clear()
        ignored = list(applied.get("ignored_decisions") or [])
        if ignored:
            warnings.append(
                f"Ignored {len(ignored)} sweep decisions that were not offered candidate pairs."
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
            "candidate_method": (
                "lexical+embedding" if runtime.embeddings is not None and not any(
                    "label overlap" in warning for warning in warnings
                )
                else "lexical"
            ),
            "candidates": candidates,
            "accepted_merges": applied["accepted_merges"],
            "rejected_merges": applied["rejected_merges"],
            "ignored_decisions": list(applied.get("ignored_decisions") or []),
            "candidate_decisions": list(applied.get("candidate_decisions") or []),
            "merge_map": applied["merge_map"],
            "relation_hints": applied["relation_hints"],
            "overloaded_codes": overloaded,
            "codes_sent": len(codes),
            "chunks": chunk_count,
            "review": list(applied.get("review") or []),
            "consolidation": consolidation,
            "quotes_checked": quote_stats["quotes_checked"],
            "quotes_split": quote_stats["quotes_split"],
        }
        decision_rows = []
        for item in applied["accepted_merges"]:
            decision_rows.append({**item, "decision": "accept"})
        for item in applied["rejected_merges"]:
            if item.get("reason") in {"unknown_or_same", "unknown_definition", "not_offered", "already_decided", "merge_cap"}:
                continue
            decision_rows.append({**item, "decision": "reject"})
        pair_decisions = remember_pair_decisions(
            list(state.get("pair_decisions") or []),
            pre_merge_book,
            decision_rows,
        )
        carried = overlap_state if overlap_state is not None else state
        ops = list(carried.get("ops") or [])
        review = list(carried.get("review") or [])
        for op in applied.get("ops") or []:
            op = dict(op)
            if not op.get("op_id"):
                counters["op"] = int(counters.get("op", 0)) + 1
                op["op_id"] = f"o{counters['op']:05d}"
            ops.append(op)
            if op.get("status") == "review":
                review.append(op)
        branch_books = state.get("branch_codebooks") or {}
        if state.get("parallel"):
            branch_books = {doc_id: deepcopy(applied["codebook"]) for doc_id in branch_books}
        update = _sweep_update(
            state,
            codebook=applied["codebook"],
            segments=applied["segments"],
            memos=memos,
            counters=counters,
            saturation=list(state.get("saturation") or []) + [_point(applied["codebook"], pre_len=pre_len)],
            sweep_log=sweep_log + [record],
            branch_codebooks=branch_books,
            prompt_log=prompt_log,
            warnings=warnings,
            ops=ops,
            review=review,
            pair_decisions=pair_decisions,
        )
        if overlap_state is not None:
            coded_before = int(state.get("passages_since_sweep") or 0)
            new_before = int(state.get("new_codes_since_sweep") or 0)
            update["next_index"] = overlap_state["next_index"]
            update["highlight_ids"] = overlap_state["highlight_ids"]
            update["validation_log"] = overlap_state["validation_log"]
            update["branch_queues"] = overlap_state["branch_queues"]
            update["passages_coded"] = overlap_state["passages_coded"]
            update["passages_since_sweep"] = int(overlap_state["passages_since_sweep"]) - coded_before
            update["new_codes_since_sweep"] = int(overlap_state["new_codes_since_sweep"]) - new_before
            if not state.get("parallel"):
                update["branch_codebooks"] = overlap_state["branch_codebooks"]
        return finish(update)

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
    warnings: list | None = None,
    ops: list | None = None,
    review: list | None = None,
    pair_decisions: list | None = None,
) -> dict:
    return {
        "codebook": codebook,
        "segments": segments,
        "memos": memos,
        "counters": counters,
        "saturation": saturation,
        "sweep_log": sweep_log,
        "warnings": list(state.get("warnings") or []) if warnings is None else warnings,
        "prompt_log": prompt_log if prompt_log is not None else list(state.get("prompt_log") or []),
        "new_codes_since_sweep": 0,
        "passages_since_sweep": 0,
        "branch_codebooks": branch_codebooks if branch_codebooks is not None else state.get("branch_codebooks") or {},
        "ops": list(state.get("ops") or []) if ops is None else ops,
        "review": list(state.get("review") or []) if review is None else review,
        "pair_decisions": list(state.get("pair_decisions") or []) if pair_decisions is None else pair_decisions,
    }


def _export_node(runtime: RunConfig):
    def export_node(state: CodingState) -> dict:
        provisional = result_from_state(state, interrupted=False, thread_id=str(state.get("metadata", {}).get("thread_id") or ""), resumable=False)
        paths: dict[str, str] = {}
        if runtime.output_dir:
            paths = export_result(provisional, runtime.output_dir)
        return {"export_paths": paths, "finished": True}

    return export_node


def iter_sweep_chunks(codes: list, limit: int) -> list[list]:
    """Split a codebook into sweep requests. Every code is sent; none are dropped."""
    size = max(1, limit)
    if len(codes) <= size:
        return [list(codes)]
    return [list(codes[start : start + size]) for start in range(0, len(codes), size)]


def reconcile_sweep_parts(parts: list[dict]) -> dict:
    """Merge per-chunk sweep results. Chunk index wins over completion order.

    A pair proposed by two chunks is kept once, from the earlier chunk. When
    those chunks disagree about which label survives, the earlier chunk's
    direction is kept. Definitions keep the longer text, then the earlier chunk.
    """
    ordered = sorted(parts, key=lambda part: int(part.get("chunk_index") or 0))
    merges: list[dict] = []
    seen_pairs: set[frozenset[str]] = set()
    for part in ordered:
        for merge in part.get("merges") or []:
            source = code_key(str(merge.get("source") or ""))
            target = code_key(str(merge.get("target") or ""))
            if not source or not target or source == target:
                continue
            pair = frozenset((source, target))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            merges.append(dict(merge))
    definitions: dict[str, dict] = {}
    for part in ordered:
        chunk_index = int(part.get("chunk_index") or 0)
        for update in part.get("definitions") or []:
            key = code_key(str(update.get("code") or ""))
            text = str(update.get("definition") or "").strip()
            if not key or not text:
                continue
            current = definitions.get(key)
            if current is None or len(text) > len(current["definition"]) or (
                len(text) == len(current["definition"]) and chunk_index < current["chunk_index"]
            ):
                definitions[key] = {
                    "code": update.get("code"),
                    "code_id": update.get("code_id") or "",
                    "definition": text,
                    "confidence": update.get("confidence", 1.0),
                    "chunk_index": chunk_index,
                }
    overloaded: list[str] = []
    seen_over: set[str] = set()
    for part in ordered:
        for label in part.get("overloaded_codes") or []:
            key = code_key(str(label))
            if key and key not in seen_over:
                seen_over.add(key)
                overloaded.append(str(label))
    rejections: list[dict] = []
    seen_rejections: set[frozenset[str]] = set()
    for part in ordered:
        for rejection in part.get("rejections") or []:
            source = code_key(str(rejection.get("source") or ""))
            target = code_key(str(rejection.get("target") or ""))
            if not source or not target or source == target:
                continue
            pair = frozenset((source, target))
            if pair in seen_pairs or pair in seen_rejections:
                continue
            seen_rejections.add(pair)
            rejections.append(dict(rejection))
    hints: list[dict] = []
    seen_hints: set[tuple] = set()
    for part in ordered:
        for hint in part.get("relation_hints") or []:
            marker = (
                code_key(str(hint.get("source") or "")),
                code_key(str(hint.get("target") or "")),
                str(hint.get("relation") or ""),
                str(hint.get("note") or ""),
            )
            if marker in seen_hints or not marker[0] or not marker[1]:
                continue
            seen_hints.add(marker)
            hints.append(dict(hint))
    return {
        "merges": merges,
        "definitions": [
            {
                "code": item["code"],
                "code_id": item.get("code_id") or "",
                "definition": item["definition"],
                "confidence": item.get("confidence", 1.0),
            }
            for item in sorted(definitions.values(), key=lambda item: code_key(str(item["code"] or "")))
        ],
        "overloaded_codes": overloaded,
        "relation_hints": hints,
        "rejections": rejections,
        "integrative_memo": _join_unique(str(part.get("integrative_memo") or "") for part in ordered),
        "reflexive_memo": _join_unique(str(part.get("reflexive_memo") or "") for part in ordered),
    }


def _join_unique(texts) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for text in texts:
        cleaned = str(text or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            kept.append(cleaned)
    return "\n\n".join(kept)


def _quotes_for(segments: list, key: str, limit: int = 6) -> list[str]:
    quotes = []
    for segment in segments:
        if (segment.get("code_key") or code_key(str(segment.get("code") or ""))) != key:
            continue
        quote = str(segment.get("quote") or "").strip()
        if quote and quote not in quotes:
            quotes.append(quote)
        if len(quotes) >= limit:
            break
    return quotes


async def _annotate_leaks(runtime: RunConfig, done: dict) -> None:
    """Ask ``check_model`` whether flagged labels really restate the context."""
    if runtime.check_model is None:
        return
    from automated_grounded_theory.prompts import build_leak_messages

    labels = []
    for coding, _errors in done.values():
        for proposal in coding.get("codes") or []:
            label = str(proposal.get("code") or "")
            if label and label_restates_context(label, context_term_set(runtime.context)):
                labels.append(proposal)
    if not labels:
        return
    messages = build_leak_messages(context=runtime.context, labels=labels, prompt_cache=runtime.prompt_cache)
    payload = await _ainvoke_structured(
        runtime.check_model,
        LeakCheckBatch,
        messages,
        kind="context_leak",
        model_name="check",
        passage_id="context-leak",
        runtime=runtime,
    )
    by_label = {str(item.get("label") or "").casefold(): item for item in payload.get("checks") or []}
    for proposal in labels:
        check = by_label.get(str(proposal.get("code") or "").casefold())
        if not check:
            continue
        proposal["context_leak_rationale"] = check.get("rationale") or ""
        if not check.get("leaks", True):
            proposal["context_leak_cleared"] = True
        elif check.get("reworded_label"):
            proposal["context_leak_reword"] = check.get("reworded_label")


async def _check_quotes(runtime: RunConfig, codebook: dict, segments: list, accepted: list, sweep_index: int) -> list:
    """Optional cheap model: which member quotes do not fit the merged code."""
    from automated_grounded_theory.prompts import build_quote_fit_messages

    rows = []
    for merge in accepted:
        source = code_key(str(merge.get("source") or ""))
        target = code_key(str(merge.get("target") or ""))
        survivor = source if source in codebook else target if target in codebook else ""
        if not survivor:
            continue
        quotes = _quotes_for(segments, survivor)
        if not quotes:
            continue
        rows.append({"code_id": codebook[survivor].get("code_id") or "", "label": codebook[survivor].get("code"), "quotes": quotes, "merge": merge})
    if not rows:
        return []
    messages = build_quote_fit_messages(context=runtime.context, codes=rows, prompt_cache=runtime.prompt_cache)
    payload = await _ainvoke_structured(
        runtime.check_model,
        QuoteFitBatch,
        messages,
        kind="quote_fit",
        model_name="check",
        passage_id=f"sweep-{sweep_index}-quotes",
        runtime=runtime,
    )
    fits_by_code: dict[str, list] = {}
    for check in payload.get("checks") or []:
        fits_by_code.setdefault(str(check.get("code_id") or ""), []).append(check)
    stamped = []
    for row in rows:
        fits = fits_by_code.get(str(row["code_id"]) or "")
        if not fits:
            continue
        merge = dict(row["merge"])
        merge["quote_fits"] = fits
        stamped.append(merge)
    return stamped


async def _verify_proposed_merges(
    runtime: RunConfig,
    codebook: dict,
    merges: list,
    sweep_index: int,
) -> tuple[list, list]:
    """Second structured look at each proposed merge. Opposite stances are not sent."""
    from automated_grounded_theory.codebook import opposite_valence

    eligible = []
    rejected = []
    for merge in merges:
        source = code_key(str(merge.get("source") or ""))
        target = code_key(str(merge.get("target") or ""))
        if source in codebook and target in codebook and opposite_valence(
            codebook[source].get("valence"), codebook[target].get("valence")
        ):
            rejected.append({**dict(merge), "reason": "opposite_valence", "confidence": merge.get("confidence", 1.0)})
            continue
        eligible.append(merge)
    if not eligible:
        return [], rejected
    messages = build_merge_check_messages(
        context=runtime.context,
        codebook=codebook,
        merges=eligible,
        prompt_cache=runtime.prompt_cache,
    )
    payload = await _ainvoke_structured(
        runtime.sweeper,
        MergeCheckBatch,
        messages,
        kind="merge_verify",
        model_name=runtime.sweep_model_name,
        passage_id=f"sweep-{sweep_index}-verify",
        runtime=runtime,
    )
    by_pair = {}
    for check in payload.get("checks") or []:
        pair = frozenset(
            (
                code_key(str(check.get("source") or "")),
                code_key(str(check.get("target") or "")),
            )
        )
        if len(pair) == 2:
            by_pair[pair] = check
    kept = []
    for merge in eligible:
        pair = frozenset((code_key(str(merge.get("source") or "")), code_key(str(merge.get("target") or ""))))
        check = by_pair.get(pair)
        if check is None or str(check.get("action") or "") != "accept":
            rejected.append(
                {
                    **dict(merge),
                    "reason": "verify_rejected" if check else "verify_omitted",
                    "confidence": 1.0 if check is None else check.get("confidence", 1.0),
                    "rationale": "" if check is None else check.get("rationale") or merge.get("rationale") or "",
                }
            )
            continue
        confirmed = dict(merge)
        confirmed["confidence"] = check.get("confidence", merge.get("confidence", 1.0))
        confirmed["rationale"] = check.get("rationale") or merge.get("rationale") or ""
        kept.append(confirmed)
    return kept, rejected


async def _rewrite_survivor_definitions(
    runtime: RunConfig,
    codebook: dict,
    segments: list,
    accepted: list,
    pre_merge_book: dict,
    sweep_index: int,
) -> tuple[dict, list]:
    """Keep each survivor's label. Rewrite its definition from the quotes it now holds."""
    from automated_grounded_theory.codebook import _code_id

    groups: dict[str, list[str]] = {}
    for merge in accepted:
        source = code_key(str(merge.get("source") or ""))
        target = code_key(str(merge.get("target") or ""))
        survivor = source if source in codebook else target if target in codebook else ""
        retired = target if survivor == source else source
        if not survivor:
            continue
        label = str(pre_merge_book.get(retired, {}).get("code") or merge.get("source") or merge.get("target") or "")
        groups.setdefault(survivor, [])
        if label and label not in groups[survivor] and label != codebook[survivor].get("code"):
            groups[survivor].append(label)
    if not groups:
        return codebook, []
    rows = []
    for key, retired in groups.items():
        entry = codebook[key]
        rows.append(
            {
                "code_id": entry.get("code_id") or _code_id(key),
                "code": entry.get("code"),
                "definition": entry.get("definition") or "",
                "retired_labels": retired,
                "quotes": _quotes_for(segments, key),
            }
        )
    messages = build_rewrite_messages(context=runtime.context, codes=rows, prompt_cache=runtime.prompt_cache)
    try:
        payload = await _ainvoke_structured(
            runtime.sweeper,
            DefinitionRewriteBatch,
            messages,
            kind="definition_rewrite",
            model_name=runtime.sweep_model_name,
            passage_id=f"sweep-{sweep_index}-rewrite",
            runtime=runtime,
        )
    except CodingError as exc:
        if not is_transient(exc):
            raise
        runtime.notes.append(f"Definition rewrite after sweep {sweep_index} was skipped: {exc}")
        return codebook, []
    book = {key: dict(entry) for key, entry in codebook.items()}
    by_id = {str(entry.get("code_id") or _code_id(key)): key for key, entry in book.items()}
    ops = []
    for rewrite in payload.get("rewrites") or []:
        code_id = str(rewrite.get("code_id") or "")
        confidence = float(rewrite.get("confidence") or 0)
        key = by_id.get(code_id) or code_key(str(rewrite.get("code") or ""))
        definition = str(rewrite.get("definition") or "").strip()
        if key not in groups or key not in book or not definition:
            ops.append(
                {
                    "op_id": "",
                    "op": "refine_definition",
                    "status": "rejected",
                    "confidence": confidence,
                    "ids": {"code_id": code_id},
                    "reason": "unknown_code_id",
                }
            )
            continue
        if confidence < runtime.review_below:
            ops.append(
                {
                    "op_id": "",
                    "op": "refine_definition",
                    "status": "review",
                    "confidence": confidence,
                    "ids": {"code_id": book[key].get("code_id") or code_id},
                    "reason": "below_review_threshold",
                }
            )
            continue
        book[key]["definition"] = definition
        book[key].setdefault("history", []).append({"event": "definition_rewritten", "definition": definition})
        ops.append(
            {
                "op_id": "",
                "op": "refine_definition",
                "status": "applied",
                "confidence": confidence,
                "ids": {"code_id": book[key].get("code_id") or code_id},
                "reason": "",
            }
        )
    return book, ops


async def _sweep_chunks(
    runtime: RunConfig,
    codes: list,
    sweep_index: int,
    *,
    frozen: set[frozenset[str]] | None = None,
    consolidation: bool = False,
) -> tuple[dict, list, int]:
    """Detect candidates, confirm merges, and refine definitions per chunk, concurrently.

    Chunks share one ``abatch``. Their merge maps are reconciled by chunk index,
    not by which chunk returned first.
    """
    chunks = iter_sweep_chunks(codes, runtime.sweep_chunk)
    vectors = None
    embed_note = ""
    if runtime.embeddings is not None:
        try:
            vectors = await aembed_texts(
                runtime.embeddings,
                [code_text(entry) for entry in codes],
                batch_size=runtime.embed_batch_size,
                limiter=runtime.limiter,
            )
        except Exception as exc:
            if not is_transient(exc):
                raise
            vectors = None
            embed_note = (
                f"Sweep embeddings failed ({exc}). Merge candidates used label overlap instead."
            )

    hints = await asyncio.to_thread(
        hint_pairs,
        codes,
        vectors,
        embedding_threshold=runtime.merge_threshold,
        lexical_threshold=runtime.lexical_merge_threshold,
        limit=runtime.candidate_limit,
    )
    if frozen:
        hints = [
            hint
            for hint in hints
            if frozenset(
                (
                    code_key(str(hint.get("source") or "")),
                    code_key(str(hint.get("target") or "")),
                )
            )
            not in frozen
        ]
    for index, hint in enumerate(hints, start=1):
        hint["pair_id"] = f"s{sweep_index}p{index:04d}"
    key_chunk: dict[str, int] = {}
    for number, chunk in enumerate(chunks, start=1):
        for entry in chunk:
            key_chunk[code_key(str(entry.get("code") or ""))] = number
    per_chunk_candidates: list[list] = [[] for _ in chunks]
    for hint in hints:
        owners = {
            key_chunk.get(code_key(str(hint.get("source") or ""))),
            key_chunk.get(code_key(str(hint.get("target") or ""))),
        }
        owners.discard(None)
        for number in sorted(owners):
            per_chunk_candidates[number - 1].append(hint)
    candidates = hints
    warnings: list[str] = [embed_note] if embed_note else []
    if len(codes) >= 2 and not hints:
        warnings.append(
            f"Sweep {sweep_index} found no similarity hints among {len(codes)} codes. "
            "The model still reviews the codebook and may propose duplicate merges."
        )
    skipped_candidates: list[dict] = []

    async def confirm(index: int, chunk: list) -> tuple[int, dict, str | None]:
        passage_id = f"sweep-{sweep_index}" if len(chunks) == 1 else f"sweep-{sweep_index}-chunk-{index:04d}"
        messages = build_sweep_messages(
            context=runtime.context,
            codes=chunk,
            candidates=per_chunk_candidates[index - 1],
            prompt_cache=runtime.prompt_cache,
            all_codes=codes if len(chunks) > 1 else None,
            chunk_index=index,
            chunk_count=len(chunks),
            consolidation=consolidation,
        )
        try:
            payload = await _ainvoke_structured(
                runtime.sweeper,
                SweepCoding,
                messages,
                kind="sweep",
                model_name=runtime.sweep_model_name,
                passage_id=passage_id,
                runtime=runtime,
            )
            return index, payload, None
        except CodingError as exc:
            if not is_transient(exc):
                raise
            note = (
                f"Sweep {sweep_index} chunk {index} of {len(chunks)} was skipped after a "
                f"timeout or connection error: {exc}"
            )
            return index, {
                "merges": [],
                "rejections": [],
                "definitions": [],
                "overloaded_codes": [],
                "relation_hints": [],
                "integrative_memo": "",
                "reflexive_memo": "",
            }, note

    confirmed = await asyncio.gather(*(confirm(index, chunk) for index, chunk in enumerate(chunks, start=1)))
    confirmed.sort(key=lambda item: item[0])
    parts = []
    for index, parsed, note in confirmed:
        if note:
            warnings.append(note)
            for candidate in per_chunk_candidates[index - 1]:
                skipped_candidates.append({**candidate, "rationale": note})
        parts.append(
            {
                "chunk_index": index,
                "merges": list(parsed.get("merges") or []),
                "rejections": list(parsed.get("rejections") or []),
                "definitions": list(parsed.get("definitions") or []),
                "overloaded_codes": list(parsed.get("overloaded_codes") or []),
                "relation_hints": list(parsed.get("relation_hints") or []),
                "integrative_memo": parsed.get("integrative_memo") or "",
                "reflexive_memo": parsed.get("reflexive_memo") or "",
            }
        )
    reconciled = reconcile_sweep_parts(parts)
    reconciled["skipped_candidates"] = skipped_candidates
    reconciled["warnings"] = warnings
    return reconciled, candidates, len(chunks)


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
        (payload, usage), used = await ainvoke_structured(model, schema, messages, runtime.limiter)
    except Exception as exc:
        raise CodingError(f"{kind} failed for {passage_id}: {exc}") from exc
    _append_prompt_log(
        runtime,
        {
            "kind": kind,
            "passage_id": passage_id,
            "model": model_name,
            "messages": serialize_messages(used),
            "output": payload,
            "usage": usage,
        },
    )
    return payload


async def _abatch_structured(
    model: BaseChatModel,
    schema,
    message_lists: list,
    *,
    kind: str,
    model_name: str,
    passage_ids: list[str],
    runtime: RunConfig,
) -> list[dict]:
    """One ``abatch`` for every message list. Logs are written in input order."""
    if not message_lists:
        return []
    from automated_grounded_theory.async_support import parse_structured, structured_runnable

    runnable = structured_runnable(model, schema)
    try:
        results = await runtime.limiter.abatch(runnable, message_lists)
    except Exception as exc:
        label = passage_ids[0] if passage_ids else kind
        raise CodingError(f"{kind} failed for {label}: {exc}") from exc
    payloads = []
    for messages, result, passage_id in zip(message_lists, results, passage_ids):
        try:
            payload, usage = parse_structured(result, schema)
            used = messages
        except ValueError:
            try:
                (payload, usage), used = await ainvoke_structured(model, schema, messages, runtime.limiter)
            except Exception as exc:
                raise CodingError(f"{kind} failed for {passage_id}: {exc}") from exc
        _append_prompt_log(
            runtime,
            {
                "kind": kind,
                "passage_id": passage_id,
                "model": model_name,
                "messages": serialize_messages(used),
                "output": payload,
                "usage": usage,
            },
        )
        payloads.append(payload)
    return payloads


def _take_prompts(runtime: RunConfig, state: CodingState) -> list:
    assert runtime._log_lock is not None
    with runtime._log_lock:
        fresh = list(runtime.prompt_records)
        runtime.prompt_records.clear()
    return list(state.get("prompt_log") or []) + fresh


def _append_prompt_log(runtime: RunConfig, entry: dict) -> None:
    assert runtime._log_lock is not None
    line = json.dumps(entry, ensure_ascii=False, default=repr)
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
