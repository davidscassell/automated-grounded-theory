"""LangGraph pipeline: segment, code every passage, sweep, export.

Validation is its own node and does not call a model. The coding and sweep
nodes do. Checkpointing is attached by the caller so a run can resume after
an interrupt or a crash.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from automated_grounded_theory.records import (
    POLICY_VERSION,
    evidence_revision,
    freeze_snapshot,
    percentile,
    proposal_buffer_reducer,
    proposal_id,
    select_committable,
)

from automated_grounded_theory.async_support import (
    OversizedRequest,
    ainvoke_structured,
    is_transient,
    oversized_request_message,
    sweep_deadline,
)
from automated_grounded_theory.batching import (
    count_message_tokens,
    pack_indices,
    repair_allowance_tokens,
    request_input_tokens,
    reserved_output_tokens,
    resolve_tokenizer,
    schema_token_count,
)
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
    lexical_comparison_proposals,
    trivial_merges,
    union_codebooks,
)
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.corpus import LoadedDocument
from automated_grounded_theory.export import export_result
from automated_grounded_theory.prompts import (
    CODING_SYSTEM,
    SWEEP_SYSTEM,
    build_candidate_discovery_messages,
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
    code_needs_review,
    code_text,
    comparison_order,
    full_audit_due,
    hint_pairs,
    review_fingerprint,
    select_cross_chunk,
    select_prompt_codes,
    shared_codebook_vectors,
)
from automated_grounded_theory.validation import code_key, context_term_set, label_restates_context
from automated_grounded_theory.saturation import saturation_point
from automated_grounded_theory.schemas import (
    BatchCoding,
    CandidateDiscovery,
    DefinitionRewriteBatch,
    LeakCheckBatch,
    MergeCheckBatch,
    QuoteFitBatch,
    SweepCoding,
)
from automated_grounded_theory.segment import interleave_document_queue, next_round_count, plan_coding

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
    snapshot_id: str
    snapshot_version: int
    definition_revision: str
    evidence_revision: str
    policy_version: str
    batch_id: str
    round_jobs: list
    proposal_buffer: Annotated[dict, proposal_buffer_reducer]
    committed_ids: list
    metrics: dict
    lost_distinction_candidates: list


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
    batch_size: int = 1
    concurrency: int = 2
    passages_per_request: int = 1
    max_concurrent_requests: int = 2
    passages_per_codebook_update: int = 1
    sweep_chunk: int = 20
    embed_batch_size: int = 64
    full_audit_every: int = 4
    cross_chunk_limit: int = 8
    review_mode: str = "document"
    review_below: float = 0.5
    min_merge_confidence: float = 0.5
    max_merges_per_sweep: int = 12
    sweep_timeout: float = 300.0
    check_model: Any = None
    pipeline: bool = True
    token_budget: int = 6000
    model_name: str = ""
    sweep_model_name: str = ""
    prompt_log_path: str | None = None
    output_dir: str | None = None
    limiter: Any = None
    prompt_records: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    oversized_requests: list = field(default_factory=list)
    _rankings: dict = field(default_factory=dict)
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
        "snapshot_id": "s00000",
        "snapshot_version": 0,
        "definition_revision": "",
        "evidence_revision": "",
        "policy_version": POLICY_VERSION,
        "batch_id": "",
        "round_jobs": [],
        "proposal_buffer": {"items": [], "duplicates": 0, "conflicts": []},
        "committed_ids": [],
        "metrics": {},
        "lost_distinction_candidates": [],
    }


def build_graph(runtime: RunConfig):
    """Return an uncompiled graph. Coding rounds fan out with :class:`Send`.

    Workers read one frozen snapshot. ``commit`` is the only writer.
    """
    _validate_node(runtime)
    builder = StateGraph(CodingState)
    builder.add_node("segment_and_shuffle", _segment_node)
    builder.add_node("prepare_round", _prepare_round(runtime))
    builder.add_node("code_worker", _code_worker(runtime), input_schema=WorkerInput)
    builder.add_node("commit", _commit_node(runtime))
    builder.add_node("sweep", _sweep_node(runtime))
    builder.add_node("consolidate", _sweep_node(runtime, consolidation=True))
    builder.add_node("export", _export_node(runtime))
    builder.add_edge(START, "segment_and_shuffle")
    builder.add_conditional_edges(
        "segment_and_shuffle",
        _route_after_segment,
        {"prepare_round": "prepare_round", "export": "export"},
    )
    builder.add_conditional_edges(
        "prepare_round",
        _route_round,
        ["code_worker", "sweep", "consolidate", "export"],
    )
    builder.add_edge("code_worker", "commit")
    builder.add_conditional_edges(
        "commit",
        _route_after_commit,
        {"prepare_round": "prepare_round", "sweep": "sweep", "consolidate": "consolidate", "export": "export"},
    )
    builder.add_conditional_edges(
        "sweep",
        _route_after_sweep,
        {"prepare_round": "prepare_round", "consolidate": "consolidate", "export": "export"},
    )
    builder.add_edge("consolidate", "export")
    builder.add_edge("export", END)
    return builder


class WorkerInput(TypedDict, total=False):
    """Payload for one Send worker. These keys are not parent-state channels."""

    worker_job_id: str
    worker_batch_id: str
    worker_snapshot_id: str
    worker_snapshot_version: int
    worker_definition_revision: str
    worker_evidence_revision: str
    worker_policy_version: str
    worker_doc_id: str | None
    worker_indices: list
    worker_passages: list
    worker_slices: list
    worker_warning: str
    worker_codebook: dict
    worker_documents: dict


class _IdleModel(BaseChatModel):
    """Placeholder so :func:`get_graph` can draw a graph without a provider."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    @property
    def _llm_type(self) -> str:
        return "idle"


def get_graph(kind: str = "open_code"):
    """Return the LangGraph for ``open_code`` or ``themes``.

    Both graphs fan out with ``Send``. ``draw_mermaid_png`` posts the diagram
    to ``mermaid.ink``. ``draw_ascii`` needs the ``viz`` extra (``grandalf``).
    ``draw_png`` needs optional ``pygraphviz`` and a Graphviz install.
    """
    name = str(kind or "open_code").strip().casefold()
    if name == "open_code":
        from automated_grounded_theory.context import ResearchContext
        from automated_grounded_theory.retrieval import LexicalCodeRetriever

        runtime = RunConfig(
            coder=_IdleModel(),
            sweeper=_IdleModel(),
            retriever=LexicalCodeRetriever(),
            embeddings=None,
            context=ResearchContext(research_question="What is happening in this text?"),
            prompt_cache=False,
        )
        return build_graph(runtime)
    if name == "themes":
        from automated_grounded_theory.themes import build_theme_graph

        return build_theme_graph()
    raise ValueError("kind must be 'open_code' or 'themes'.")


def compile_graph(runtime: RunConfig, checkpointer, *, interrupt_before_sweep: bool = False):
    builder = build_graph(runtime)
    options = {}
    if interrupt_before_sweep:
        options["interrupt_before"] = ["sweep"]
    return builder.compile(checkpointer=checkpointer, **options)


def _route_after_segment(state: CodingState) -> str:
    if not state.get("passages"):
        return "export"
    return "prepare_round"


def _route_after_commit(state: CodingState) -> str:
    destination = _route_after_validate(state)
    if destination == "code_passage":
        return "prepare_round"
    return destination


def _route_round(state: CodingState):
    """Fan out one Send per batch, or leave the round when nothing is queued."""
    jobs = list(state.get("round_jobs") or [])
    if not jobs:
        destination = _route_after_validate(state)
        if destination == "code_passage":
            return "export"
        return destination
    documents = _document_texts(state)
    sends = []
    for job in jobs:
        doc_id = job.get("doc_id")
        indices = list(job.get("indices") or [])
        sends.append(
            Send(
                "code_worker",
                {
                    "worker_job_id": job.get("job_id"),
                    "worker_batch_id": state.get("batch_id") or state.get("snapshot_id"),
                    "worker_snapshot_id": state.get("snapshot_id"),
                    "worker_snapshot_version": state.get("snapshot_version"),
                    "worker_definition_revision": state.get("definition_revision"),
                    "worker_evidence_revision": state.get("evidence_revision"),
                    "worker_policy_version": state.get("policy_version") or POLICY_VERSION,
                    "worker_doc_id": doc_id,
                    "worker_indices": indices,
                    "worker_slices": job.get("slices"),
                    "worker_warning": str(job.get("warning") or ""),
                    "worker_passages": [state["passages"][index] for index in indices],
                    "worker_codebook": deepcopy(_book_for(state, doc_id)),
                    "worker_documents": documents,
                },
            )
        )
    return sends


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
        return "prepare_round"
    if not state.get("consolidated") and len(state.get("codebook") or {}) >= 2:
        return "consolidate"
    return "export"


def _prepare_round(runtime: RunConfig):
    async def prepare_round(state: CodingState) -> dict:
        version = int(state.get("snapshot_version") or 0) + 1
        frozen = freeze_snapshot(state, version=version)
        jobs = await _wave(state, runtime)
        packaged = []
        warnings = list(state.get("warnings") or [])
        for index, (doc_id, indices, meta) in enumerate(jobs):
            warning = str((meta or {}).get("warning") or "")
            if warning and warning not in warnings:
                warnings.append(warning)
            packaged.append(
                {
                    "doc_id": doc_id,
                    "indices": list(indices),
                    "slices": (meta or {}).get("slices"),
                    "warning": warning,
                    "job_id": f"{frozen['snapshot_id']}-j{index:03d}",
                }
            )
        return {
            **frozen,
            "batch_id": frozen["snapshot_id"],
            "round_jobs": packaged,
            "proposal_buffer": {"__reset__": True},
            "warnings": warnings,
        }

    return prepare_round


def _code_worker(runtime: RunConfig):
    async def code_worker(state: CodingState) -> dict:
        started = time.perf_counter()
        snapshot_id = str(state.get("worker_snapshot_id") or "")
        passages = list(state.get("worker_passages") or [])
        indices = list(state.get("worker_indices") or [])
        runtime.limiter.cache_context = {
            "snapshot_id": snapshot_id,
            "definition_revision": state.get("worker_definition_revision") or "",
            "evidence_revision": state.get("worker_evidence_revision") or "",
            "policy_version": state.get("worker_policy_version") or POLICY_VERSION,
        }
        book = dict(state.get("worker_codebook") or {})
        bucket: list[dict] = []
        push = getattr(runtime.limiter, "push_accounting", None)
        token = push(bucket) if push is not None else None
        try:
            coded = await _code_indexed(
                runtime,
                passages,
                list(range(len(passages))),
                state.get("worker_slices"),
                book,
                documents=dict(state.get("worker_documents") or {}),
            )
        finally:
            pop = getattr(runtime.limiter, "pop_accounting", None)
            if pop is not None and token is not None:
                pop(token)
        ended = time.perf_counter()
        latencies = [float(row["latency"]) for row in bucket if row.get("latency") is not None]
        queues = [float(row.get("queue_wait") or 0) for row in bucket]
        items = []
        for index, passage in zip(indices, passages):
            passage_key = str(passage.get("passage_id"))
            coding, attempt_errors = coded[passage_key]
            items.append(
                {
                    "proposal_id": proposal_id(snapshot_id, passage_key),
                    "job_id": state.get("worker_job_id"),
                    "batch_id": state.get("worker_batch_id"),
                    "snapshot_id": snapshot_id,
                    "snapshot_version": state.get("worker_snapshot_version"),
                    "passage_index": index,
                    "doc_id": state.get("worker_doc_id"),
                    "coding": coding,
                    "attempt_errors": attempt_errors,
                    "wall_start": started,
                    "wall_end": ended,
                    "request_latencies": latencies,
                    "queue_waits": queues,
                }
            )
        return {"proposal_buffer": {"items": items, "duplicates": 0, "conflicts": []}}

    return code_worker


def union_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    """Seconds covered by ``intervals``, counting overlap once."""
    ordered = sorted((float(start), float(end)) for start, end in intervals if end >= start)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + (end - start)


def fold_coding_times(proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Wall time, per-request latency, and queue wait for one commit.

    A batch duration is stored on every passage that shared the call. It is
    counted once, keyed by ``job_id``. Overlapping batches contribute the
    union of their intervals to wall time. Latency samples are the provider
    calls, not a copy of that wall time.
    """
    intervals: list[tuple[float, float]] = []
    latencies: list[float] = []
    queue = 0.0
    seen: set[str] = set()
    for index, item in enumerate(proposals):
        job = str(item.get("job_id") or "")
        if not job:
            job = f"row-{item.get('passage_index', index)}"
        if job in seen:
            continue
        seen.add(job)
        start = item.get("wall_start")
        end = item.get("wall_end")
        if start is not None and end is not None:
            intervals.append((float(start), float(end)))
        for sample in item.get("request_latencies") or []:
            if sample is None:
                continue
            latencies.append(float(sample))
        for sample in item.get("queue_waits") or []:
            queue += float(sample or 0)
    return {
        "wall_seconds": round(union_seconds(intervals), 4),
        "request_latencies": [round(sample, 4) for sample in latencies],
        "queue_wait_seconds": round(queue, 4),
    }


def _commit_node(runtime: RunConfig):
    async def commit(state: CodingState) -> dict:
        started = time.perf_counter()
        buffer = state.get("proposal_buffer") or {}
        fresh, stale, _skipped = select_committable(
            list(buffer.get("items") or []),
            snapshot_id=str(state.get("snapshot_id") or ""),
            committed_ids=list(state.get("committed_ids") or []),
        )
        pending = [
            {
                "passage_index": item["passage_index"],
                "doc_id": item.get("doc_id"),
                "coding": item.get("coding") or {},
                "attempt_errors": item.get("attempt_errors") or [],
            }
            for item in fresh
        ]
        in_batch_pairs: list = []
        if pending:
            pending, in_batch_pairs = await reconcile_in_batch(runtime, state, pending)
        update = runtime._commit(state, pending=pending) if pending else {}
        if not isinstance(update, dict):
            update = {}
        committed = list(state.get("committed_ids") or [])
        for item in fresh:
            pid = str(item.get("proposal_id") or "")
            if pid and pid not in committed:
                committed.append(pid)
        review = list(update.get("review") or state.get("review") or [])
        for item in stale:
            review.append(
                {
                    "op": "commit",
                    "status": "rejected",
                    "reason": "stale_snapshot",
                    "proposal_id": item.get("proposal_id"),
                    "snapshot_id": item.get("snapshot_id"),
                }
            )
        metrics = dict(state.get("metrics") or {})
        metrics["exact_proposal_collisions"] = int(metrics.get("exact_proposal_collisions") or 0) + int(
            buffer.get("duplicates") or 0
        )
        rounds = list(metrics.get("rounds") or [])
        rounds.append(
            {
                "snapshot_id": str(state.get("snapshot_id") or ""),
                "passage_ids": [
                    str(state["passages"][item["passage_index"]].get("passage_id") or "") for item in fresh
                ],
                "doc_ids": [str(item.get("doc_id") or state["passages"][item["passage_index"]].get("doc_id") or "") for item in fresh],
                "size": len(fresh),
            }
        )
        metrics["rounds"] = rounds
        metrics["reconciliation_calls"] = int(metrics.get("reconciliation_calls") or 0) + 1
        metrics["reconciliation_seconds"] = round(
            float(metrics.get("reconciliation_seconds") or 0) + time.perf_counter() - started,
            4,
        )
        metrics["stale_proposals"] = int(metrics.get("stale_proposals") or 0) + len(stale)
        metrics["conflicts"] = list(buffer.get("conflicts") or [])
        if in_batch_pairs:
            in_batch_rounds = list(metrics.get("in_batch_rounds") or [])
            in_batch_rounds.append(
                {"snapshot_id": str(state.get("snapshot_id") or ""), "pairs": in_batch_pairs}
            )
            metrics["in_batch_rounds"] = in_batch_rounds
        folded = fold_coding_times(list(fresh) + list(stale))
        samples = list(metrics.get("coding_request_latency_seconds") or metrics.get("coding_seconds") or [])
        samples.extend(folded["request_latencies"])
        metrics["coding_seconds"] = samples
        metrics["coding_request_latency_seconds"] = samples
        metrics["coding_p50"] = percentile(samples, 50)
        metrics["coding_p95"] = percentile(samples, 95)
        metrics["coding_queue_wait_seconds"] = round(
            float(metrics.get("coding_queue_wait_seconds") or 0) + folded["queue_wait_seconds"],
            4,
        )
        metrics["coding_wall_seconds"] = round(
            float(metrics.get("coding_wall_seconds") or 0) + folded["wall_seconds"],
            4,
        )
        timing = dict(update.get("stage_seconds") or state.get("stage_seconds") or {})
        timing["coding"] = metrics["coding_wall_seconds"]
        update["stage_seconds"] = timing
        limiter = runtime.limiter
        metrics["transport_retries"] = int(getattr(limiter, "timeout_retries", 0) or 0) + int(
            getattr(limiter, "rate_limit_retries", 0) or 0
        )
        metrics["schema_repairs"] = int(getattr(limiter, "schema_repairs", 0) or 0)
        metrics["cache_hits"] = int(getattr(limiter, "cache_hits", 0) or 0)
        update["review"] = review
        update["committed_ids"] = committed
        update["proposal_buffer"] = {"__reset__": True}
        update["round_jobs"] = []
        update["metrics"] = metrics
        update["evidence_revision"] = evidence_revision(update.get("segments") or state.get("segments") or [])
        if stale and "codebook" not in update:
            update["codebook"] = state.get("codebook") or {}
        return update

    return commit


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
    queue = list(plan.queue)
    per_request = max(1, int(metadata.get("passages_per_request") or metadata.get("batch_size") or 1))
    commit_every = max(1, int(metadata.get("passages_per_codebook_update") or per_request))
    review_mode = str(metadata.get("review_mode") or "document")
    if not plan.parallel and review_mode == "document" and commit_every > 1:
        queue = interleave_document_queue(plan.passages, queue, commit_every)
        for order, index in enumerate(queue):
            plan.passages[index]["coding_order"] = order
    return {
        "passages": plan.passages,
        "queue": queue,
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
        jobs = await _wave(state, runtime)
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


def _ranking_snapshot(entries: Sequence[Mapping[str, Any]]) -> str:
    payload = "\0".join(code_text(entry) for entry in entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ranking_key(snapshot: str, text: str, k: int) -> tuple:
    return (snapshot, text, int(k))


async def _pack_jobs(
    state: CodingState,
    runtime: RunConfig,
    indices: Sequence[int],
    token_budget: int,
    per_request: int,
    doc_id: str | None,
) -> list[tuple]:
    """Pack passage indexes so each request, plus reserved output, fits the budget.

    Full passages are ranked once before packing. A passage that does not fit
    is split on whitespace to the end. Candidate slices are not ranked one by
    one, and the packer is not restarted. The slices that packing actually
    keeps are ranked together afterwards.
    """
    if not indices:
        return []
    entries = list(_book_for(state, doc_id).values())
    passage_texts = [str(state["passages"][index].get("text") or "") for index in indices]
    await _ensure_rankings(runtime, passage_texts, entries)
    texts = [str(passage.get("text") or "") for passage in state["passages"]]
    names = [str(passage.get("passage_id") or "") for passage in state["passages"]]
    reserve = reserved_output_tokens(1, max_codes_per_passage=runtime.max_codes_per_passage)
    measure = _coding_measure(runtime, state, doc_id, token_budget)
    batches, _warnings = pack_indices(
        indices,
        texts,
        batch_size=per_request,
        token_budget=token_budget,
        names=names,
        tokenizer=resolve_tokenizer(runtime.coder),
        reserved_output=reserve,
        measure=measure,
    )
    await _ensure_rankings(runtime, _finalized_slice_texts(batches, texts), entries)
    jobs = []
    for batch in batches:
        if not batch.indices:
            continue
        slices = [list(span) for span in batch.slices] if batch.slices else None
        jobs.append((doc_id, list(batch.indices), {"slices": slices, "warning": batch.warning}))
    return jobs


def _finalized_slice_texts(batches: Sequence, texts: Sequence[str]) -> list[str]:
    """Passage slices packing kept. Probes from the search are not included."""
    found: list[str] = []
    for batch in batches:
        if not batch.slices or len(batch.indices) != 1:
            continue
        index = batch.indices[0]
        source = texts[index] if 0 <= index < len(texts) else ""
        for span in batch.slices:
            start, end = int(span[0]), int(span[1])
            found.append(source[start:end])
    return found


def _embedding_fallback_note(exc: BaseException) -> str:
    return f"Code retrieval embeddings failed ({exc}). Nearest codes used label overlap instead."


def _remember_note(runtime: RunConfig, note: str) -> None:
    if note and note not in runtime.notes:
        runtime.notes.append(note)


def _cached_labels(runtime: RunConfig, snapshot: str, text: str, k: int) -> list[str] | None:
    cached = runtime._rankings.get(_ranking_key(snapshot, text, k))
    if cached is None:
        return None
    return list(cached)


async def _rank_texts(
    runtime: RunConfig,
    texts: Sequence[str],
    entries: Sequence[Mapping[str, Any]],
    k: int,
) -> list[list[str]]:
    """Rank ``texts`` without blocking the event loop on an embedding client."""
    if k <= 0 or not entries:
        return [[] for _ in texts]
    if isinstance(runtime.retriever, EmbeddingCodeRetriever):
        try:
            ranked = await runtime.retriever.arank_many(
                list(texts),
                entries,
                k=k,
                limiter=runtime.limiter,
                batch_size=runtime.embed_batch_size,
            )
        except Exception as exc:
            if not is_transient(exc):
                raise
            _remember_note(runtime, _embedding_fallback_note(exc))
            lexical = LexicalCodeRetriever()
            ranked = [lexical.rank(text, entries, k=k) for text in texts]
    else:
        rank = getattr(runtime.retriever, "rank", None)
        if rank is None:
            return [[] for _ in texts]
        ranked = [rank(text, entries, k=k) or [] for text in texts]
    return [[str(label) for label in labels if str(label)] for labels in ranked]


async def _ensure_rankings(
    runtime: RunConfig,
    texts: Sequence[str],
    entries: Sequence[Mapping[str, Any]],
) -> None:
    """Retrieve each missing passage once for this codebook snapshot.

    Planning and coding share ``runtime._rankings``. A later call with the
    same passage text and the same code texts does not embed again.
    """
    k = max(0, int(runtime.top_k_codes))
    snapshot = _ranking_snapshot(entries)
    missing: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        if _cached_labels(runtime, snapshot, text, k) is None:
            missing.append(text)
    if not missing:
        return
    ranked = await _rank_texts(runtime, missing, entries, k)
    for text, labels in zip(missing, ranked):
        runtime._rankings[_ranking_key(snapshot, text, k)] = list(labels)


def _union_labels(per_passage: Sequence[Sequence[str]]) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for labels in per_passage:
        for label in labels:
            key = label.casefold()
            if key and key not in seen:
                seen.add(key)
                ranked.append(label)
    return ranked


def _scoped_feedback(
    feedback: Mapping[str, str] | None,
    passages: Sequence[Mapping[str, Any]],
) -> dict[str, str] | None:
    """Feedback for these passages only. Other passages are a different request."""
    if not feedback:
        return None
    ids = {str(passage.get("passage_id") or "") for passage in passages}
    scoped = {
        str(key): str(value)
        for key, value in feedback.items()
        if str(key) in ids and str(value or "").strip()
    }
    return scoped or None


def _coding_messages_for(
    runtime: RunConfig,
    passages: Sequence[Mapping[str, Any]],
    per_passage: Sequence[Sequence[str]],
    feedback: Mapping[str, str] | None,
    view: Sequence[Mapping[str, Any]],
    omitted: int,
) -> list:
    return build_coding_messages(
        context=runtime.context,
        codebook_view=view,
        omitted_codes=omitted,
        passages=passages,
        neighbor_chars=runtime.neighbor_chars,
        feedback=_scoped_feedback(feedback, passages),
        prompt_cache=runtime.prompt_cache,
        nearest=per_passage,
    )


def _coding_request_cost(runtime: RunConfig, messages: Sequence, n_passages: int) -> int:
    """Assembled input, schema, repair headroom, and reserved output."""
    tokenizer = resolve_tokenizer(runtime.coder)
    reserved = reserved_output_tokens(n_passages, max_codes_per_passage=runtime.max_codes_per_passage)
    return request_input_tokens(messages, BatchCoding, tokenizer) + repair_allowance_tokens(BatchCoding, tokenizer) + reserved


def _view_for(
    entries: Sequence[Mapping[str, Any]],
    per_passage: Sequence[Sequence[str]],
    frequent_codes: int,
) -> tuple[list[dict], int]:
    return select_prompt_codes(entries, _union_labels(per_passage), frequent_limit=frequent_codes)


def _feedback_chars(feedback: Mapping[str, str] | None) -> int:
    if not feedback:
        return 0
    return sum(len(str(value or "")) for value in feedback.values())


def _halve_feedback(feedback: Mapping[str, str] | None) -> dict[str, str]:
    """Drop or halve each note. An empty result means nothing more can be cut."""
    if not feedback:
        return {}
    shrunk: dict[str, str] = {}
    for key, value in feedback.items():
        text = str(value or "")
        if len(text) <= 160:
            continue
        shrunk[str(key)] = text[: max(160, len(text) // 2)]
    return shrunk


def _fit_coding_messages(
    runtime: RunConfig,
    passages: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    per_passage: Sequence[Sequence[str]],
    feedback: Mapping[str, str] | None,
    budget: int,
) -> list:
    """Assemble the real request, then drop codes and shorten feedback until it fits.

    An empty codebook view is not success. Validation feedback can still leave
    the request over ``budget``. The caller sends those messages, logs a
    warning, and records an ``oversized_request`` flag.
    """
    view, omitted = _view_for(entries, per_passage, runtime.frequent_codes)
    full = len(view)
    feedback_now = dict(_scoped_feedback(feedback, passages) or {})
    original_feedback = _feedback_chars(feedback_now)
    trimmed_codes = False
    for _ in range(max(full, 1) + 40):
        extra = full - len(view)
        messages = _coding_messages_for(runtime, passages, per_passage, feedback_now or None, view, omitted + extra)
        if _coding_request_cost(runtime, messages, len(passages)) <= budget:
            if trimmed_codes:
                note = (
                    f"Trimmed the coding context from {full} codes to {len(view)} "
                    f"to fit token_budget ({budget})."
                )
                _remember_note(runtime, note)
            if _feedback_chars(feedback_now) < original_feedback:
                _remember_note(runtime, f"Shortened validation feedback to fit token_budget ({budget}).")
            return messages
        # Feedback is what a validation retry adds. Shorten it before dropping
        # codes that already fit on the attempt without that note.
        nxt = _halve_feedback(feedback_now)
        if _feedback_chars(nxt) < _feedback_chars(feedback_now):
            feedback_now = nxt
            continue
        if view:
            view = view[:-1]
            trimmed_codes = True
            continue
        return messages
    return messages


def _passage_groups(
    runtime: RunConfig,
    passages: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    per_passage: Sequence[Sequence[str]],
    feedback: Mapping[str, str] | None,
    budget: int,
) -> list[list[int]]:
    """Split a batch whose real retrieved union does not fit."""
    groups: list[list[int]] = []
    current: list[int] = []

    def cost(indexes: Sequence[int]) -> int:
        subset = [passages[index] for index in indexes]
        ranks = [per_passage[index] for index in indexes]
        view, omitted = _view_for(entries, ranks, runtime.frequent_codes)
        messages = _coding_messages_for(runtime, subset, ranks, feedback, view, omitted)
        return _coding_request_cost(runtime, messages, len(subset))

    for index in range(len(passages)):
        if current and cost([*current, index]) > budget:
            groups.append(current)
            current = [index]
        else:
            current.append(index)
    if current:
        groups.append(current)
    return groups


def _input_cost(messages: Sequence, tokenizer) -> int:
    """Messages, JSON schema, and the short repair note. Reserved output is separate."""
    return (
        count_message_tokens(messages, tokenizer)
        + schema_token_count(BatchCoding, tokenizer)
        + repair_allowance_tokens(BatchCoding, tokenizer)
    )


def _coding_measure(runtime: RunConfig, state: CodingState, doc_id: str | None, token_budget: int):
    """Token count of the request these passages would actually send.

    Neighbors are retrieved for the passages in the candidate batch and
    unioned, then the messages are built and the schema is counted. The old
    estimate used only ``top_k_codes + frequent_codes`` of the longest entries
    and missed both the union across passages and the schema.

    A batch of several passages returns that full cost, so the packer splits
    the batch when the union does not fit. A single passage trims retrieved
    codes down to the budget before the packer splits the passage text. The
    text is split only when it still does not fit with an empty codebook view.
    """
    tokenizer = resolve_tokenizer(runtime.coder)
    entries = list(_book_for(state, doc_id).values())
    snapshot = _ranking_snapshot(entries)
    k = max(0, int(runtime.top_k_codes))
    reserve = reserved_output_tokens(1, max_codes_per_passage=runtime.max_codes_per_passage)
    budget = max(1, int(token_budget))

    def labels_for(text: str) -> list[str]:
        cached = _cached_labels(runtime, snapshot, text, k)
        if cached is None:
            # A candidate slice has no ranking yet. One passage trims retrieved
            # codes until the request fits, so the slice fits exactly when it
            # fits with nothing retrieved. The empty ranking lets the splitter
            # finish. ``_pack_jobs`` ranks the slices that were kept.
            return []
        return cached

    def cost_for(
        passages: Sequence[Mapping[str, Any]],
        per_passage: Sequence[Sequence[str]],
        view: Sequence[Mapping[str, Any]],
        omitted: int,
    ) -> int:
        messages = _coding_messages_for(runtime, passages, per_passage, None, view, omitted)
        return _input_cost(messages, tokenizer)

    def measure(batch: Sequence[int], overrides: Mapping[int, str] | None = None) -> int:
        passages = []
        per_passage: list[list[str]] = []
        for index in batch:
            passage = dict(state["passages"][index])
            if overrides and index in overrides:
                passage["text"] = overrides[index]
            passages.append(passage)
            per_passage.append(labels_for(str(passage.get("text") or "")))
        view, omitted = _view_for(entries, per_passage, runtime.frequent_codes)
        cost = cost_for(passages, per_passage, view, omitted)
        if len(batch) != 1:
            return cost
        # pack_indices adds one reserved-output block on top of this return.
        ceiling = budget - reserve
        full = len(view)
        while cost > ceiling and view:
            view = view[:-1]
            cost = cost_for(passages, per_passage, view, omitted + (full - len(view)))
        return cost

    return measure


def _sliced_passages(passage: Mapping[str, Any], spans: Sequence[Sequence[int]]) -> list[dict]:
    text = str(passage.get("text") or "")
    clones = []
    for span in spans:
        start, end = int(span[0]), int(span[1])
        clone = dict(passage)
        clone["text"] = text[start:end]
        clones.append(clone)
    return clones


def _merge_codings(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict:
    """Union of two slice codings. The commit anchors quotes on the full passage."""
    merged = dict(left)
    seen = {
        (str(item.get("code") or ""), str(item.get("quote") or ""))
        for item in (left.get("codes") or [])
    }
    codes = [dict(item) for item in (left.get("codes") or [])]
    for item in right.get("codes") or []:
        marker = (str(item.get("code") or ""), str(item.get("quote") or ""))
        if marker in seen:
            continue
        seen.add(marker)
        codes.append(dict(item))
    merged["codes"] = codes
    memos = []
    for memo in (left.get("memo"), right.get("memo")):
        text = str(memo or "").strip()
        if text and text not in memos:
            memos.append(text)
    merged["memo"] = "\n\n".join(memos)
    return merged


async def _code_indexed(
    runtime: RunConfig,
    passages: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    slices: Sequence[Sequence[int]] | None,
    book: dict,
    documents: dict[str, str] | None,
) -> dict[str, tuple[dict, list]]:
    """Code ``indices``. An oversized passage's slices are coded, then merged."""
    selected = [passages[index] for index in indices]
    if slices and len(selected) == 1:
        base = dict(selected[0])
        passage_id = str(base.get("passage_id"))
        merged: dict | None = None
        errors: list = []
        for clone in _sliced_passages(base, slices):
            coded = await _code_batch(runtime, [clone], book, documents=documents)
            coding, attempt_errors = coded[passage_id]
            errors.extend(attempt_errors)
            merged = coding if merged is None else _merge_codings(merged, coding)
        empty = {"passage_id": passage_id, "codes": [], "memo": ""}
        return {passage_id: (merged or empty, errors)}
    return await _code_batch(runtime, selected, book, documents=documents)


def _coding_schedule(metadata: Mapping[str, Any], runtime: RunConfig) -> tuple[int, int, int]:
    """Request size, commit cadence, and in-flight cap.

    ``max_concurrent_requests`` does not widen the frozen codebook. Only
    ``passages_per_codebook_update`` decides how many passages share a snapshot.
    """
    per_request = max(
        1,
        int(metadata.get("passages_per_request") or metadata.get("batch_size") or runtime.batch_size or 1),
    )
    # Missing commit cadence follows the request size. It does not follow
    # concurrency. The runtime default of 1 must not override a larger batch.
    if metadata.get("passages_per_codebook_update") is not None:
        commit_every = max(1, int(metadata.get("passages_per_codebook_update") or 1))
    else:
        commit_every = per_request
    concurrent = max(
        1,
        int(metadata.get("max_concurrent_requests") or metadata.get("concurrency") or runtime.concurrency or 1),
    )
    return per_request, commit_every, concurrent


async def _wave(
    state: CodingState,
    runtime: RunConfig,
    *,
    upcoming: bool = False,
) -> list[tuple[str | None, list[int]]]:
    """One coding step against a frozen codebook snapshot.

    The step covers ``passages_per_codebook_update`` passages, packed into
    requests of ``passages_per_request``. Those passages share the snapshot.
    The codebook is committed when the step finishes, so the next step sees
    the codes. ``max_concurrent_requests`` only limits how many of those
    requests are in flight. It does not add passages to the step.

    Document mode with a commit cadence of 1 stays inside the current document
    and takes one passage. A larger cadence round-robins across documents.
    Wave mode may mix passages from several documents. It is opt-in because
    that weakens constant comparison.

    The step stops at the next sweep. ``upcoming`` plans the step that will
    run during that sweep, so the sweep window is treated as already reset.
    """
    metadata = state.get("metadata") or {}
    per_request, commit_every, concurrent = _coding_schedule(metadata, runtime)
    sweep_every = max(1, int(metadata.get("sweep_every") or 1))
    since = 0 if upcoming else int(state.get("passages_since_sweep") or 0)
    room = sweep_every - since
    if room <= 0:
        return []
    token_budget = max(1, int(metadata.get("token_budget") or runtime.token_budget))
    review_mode = str(metadata.get("review_mode") or runtime.review_mode or "document")
    window = min(room, commit_every)

    if state.get("parallel"):
        jobs: list[tuple] = []
        used = 0
        documents = 0
        for doc_id in sorted(state.get("branch_queues") or {}):
            if documents >= concurrent or used >= room:
                break
            queue = list(state["branch_queues"][doc_id])
            if not queue:
                continue
            take = min(window, len(queue), room - used)
            if take <= 0:
                break
            indices = queue[:take]
            jobs.extend(await _pack_jobs(state, runtime, indices, token_budget, per_request, doc_id))
            used += take
            documents += 1
        return jobs

    index = int(state.get("next_index") or 0)
    queue = list(state.get("queue") or [])
    if index >= len(queue):
        return []
    if review_mode == "document" and commit_every > 1:
        # The queue is already round-robin across documents. One round takes
        # at most one passage from each document, unless only one document
        # is left. The round width is the commit cadence, not the concurrency.
        count = next_round_count(state["passages"], queue[index:], commit_every)
        indices = queue[index : index + min(room, count)]
    elif review_mode == "document":
        # Commit after each passage (or each commit cadence) inside this document.
        # Later passages are a later step and see the codes just committed.
        doc_id = state["passages"][queue[index]].get("doc_id")
        indices = []
        for passage_index in queue[index:]:
            if len(indices) >= window:
                break
            if state["passages"][passage_index].get("doc_id") != doc_id:
                break
            indices.append(passage_index)
    else:
        take = min(room, commit_every, len(queue) - index)
        indices = queue[index : index + take]
    return await _pack_jobs(state, runtime, indices, token_budget, per_request, None)


async def _pending_rows(runtime: RunConfig, state: CodingState, jobs: list[tuple]) -> list[dict]:
    """Code every batch in ``jobs`` against the codebook frozen on ``state``."""

    async def one(doc_id: str | None, indices: list[int], meta: dict | None) -> list[dict]:
        warning = str((meta or {}).get("warning") or "")
        if warning and warning not in runtime.notes:
            runtime.notes.append(warning)
        book = _book_for(state, doc_id)
        coded = await _code_indexed(
            runtime,
            state["passages"],
            indices,
            (meta or {}).get("slices"),
            book,
            documents=_document_texts(state),
        )
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

    normalized = []
    for job in jobs:
        doc_id, indices = job[0], job[1]
        meta = job[2] if len(job) > 2 else None
        normalized.append((doc_id, indices, meta))
    groups = await asyncio.gather(*(one(doc_id, indices, meta) for doc_id, indices, meta in normalized))
    return [row for group in groups for row in group]


def _new_round_proposals(state: CodingState, pending: list[dict]) -> list[dict]:
    """New labels in this round, with the quotes that proposed them."""
    known: set[str] = set()
    for entry in (state.get("codebook") or {}).values():
        key = code_key(str(entry.get("code") or entry.get("key") or ""))
        if key:
            known.add(key)
    for book in (state.get("branch_codebooks") or {}).values():
        for entry in (book or {}).values():
            key = code_key(str(entry.get("code") or entry.get("key") or ""))
            if key:
                known.add(key)
    found: dict[str, dict] = {}
    for item in pending:
        coding = item.get("coding") or {}
        if hasattr(coding, "model_dump"):
            coding = coding.model_dump()
        passage = state["passages"][item["passage_index"]]
        for proposal in coding.get("codes") or []:
            label = str(proposal.get("code") or "").strip()
            key = code_key(label)
            if not key or key in known:
                continue
            row = found.setdefault(
                key,
                {
                    "code": label,
                    "definition": str(proposal.get("definition") or ""),
                    "valence": str(proposal.get("valence") or ""),
                    "quotes": [],
                    "passage_id": str(passage.get("passage_id") or ""),
                },
            )
            if not row["definition"]:
                row["definition"] = str(proposal.get("definition") or "")
            quote = str(proposal.get("quote") or "").strip()
            if quote and quote not in row["quotes"]:
                row["quotes"].append(quote)
    return list(found.values())


def _apply_label_rewrites(pending: list[dict], rewrites: dict[str, str], definitions: dict[str, str]) -> None:
    """Point retired proposals at the survivor label before the commit writes them."""
    if not rewrites and not definitions:
        return
    for item in pending:
        coding = item.get("coding") or {}
        if hasattr(coding, "model_dump"):
            coding = coding.model_dump()
        else:
            coding = dict(coding)
        codes = []
        seen: set[str] = set()
        for proposal in coding.get("codes") or []:
            row = dict(proposal)
            key = code_key(str(row.get("code") or ""))
            if key in rewrites:
                row["code"] = rewrites[key]
                row["action"] = "new"
                row["code_id"] = ""
                key = code_key(str(row.get("code") or ""))
            if key in definitions:
                row["definition"] = definitions[key]
            if key in seen:
                continue
            seen.add(key)
            codes.append(row)
        coding["codes"] = codes
        item["coding"] = coding


async def reconcile_in_batch(runtime: RunConfig, state: CodingState, pending: list[dict]) -> tuple[list[dict], list[dict]]:
    """One comparison of this round's new codes, then rewrite labels that alias.

    A failed comparison leaves the proposals unchanged. Distinct concepts stay
    distinct. The pair list is for ``duplicate_rounds.json`` and is not the
    cross-snapshot duplicate count.
    """
    from automated_grounded_theory.codebook import plan_in_batch_aliases
    from automated_grounded_theory.prompts import build_in_batch_messages
    from automated_grounded_theory.schemas import InBatchComparison

    proposals = _new_round_proposals(state, pending)
    if len(proposals) < 2:
        return pending, []
    messages = build_in_batch_messages(
        context=runtime.context,
        codes=proposals,
        prompt_cache=runtime.prompt_cache,
    )
    try:
        payload = await _ainvoke_structured(
            runtime.coder,
            InBatchComparison,
            messages,
            kind="in_batch",
            model_name=runtime.model_name,
            passage_id=str(state.get("snapshot_id") or "round"),
            runtime=runtime,
        )
    except CodingError as exc:
        runtime.notes.append(f"In-batch comparison was skipped: {exc}")
        return pending, []
    plan = plan_in_batch_aliases(
        proposals,
        list(payload.get("merges") or []),
        min_merge_confidence=runtime.min_merge_confidence,
    )
    _apply_label_rewrites(pending, plan["rewrites"], plan["definitions"])
    return pending, list(plan["pairs"])


def _document_texts(state: CodingState) -> dict[str, str]:
    return {str(doc.get("doc_id")): str(doc.get("text") or "") for doc in state.get("documents") or []}


def _coding_payload(coding: Any) -> dict:
    if hasattr(coding, "model_dump"):
        return coding.model_dump()
    return dict(coding)


def _preserve_flagged_codes(previous: dict, current: Any, keys: set[str]) -> dict:
    """Keep a code the retry dropped after a quote mismatch.

    The retry may improve the quote. It must not erase the code.
    """
    payload = _coding_payload(current)
    if not keys:
        return payload
    codes = list(payload.get("codes") or [])
    present = {code_key(str(item.get("code") or "")) for item in codes}
    for item in previous.get("codes") or []:
        key = code_key(str(item.get("code") or ""))
        if key in keys and key not in present:
            codes.append(dict(item))
            present.add(key)
    payload["codes"] = codes
    return payload


def _book_for(state: CodingState, doc_id: str | None) -> dict:
    if state.get("parallel") and doc_id is not None:
        return state.get("branch_codebooks", {}).get(doc_id) or {}
    return state.get("codebook") or {}


async def _code_batch(
    runtime: RunConfig,
    passages: list[dict],
    codebook: dict,
    documents: dict[str, str] | None = None,
) -> dict[str, tuple[dict, list]]:
    """Code a batch. Weak quotes and bad labels are retried once, then kept and flagged."""
    pending = list(passages)
    feedback: dict[str, str] = {}
    errors: dict[str, list] = {str(passage.get("passage_id")): [] for passage in passages}
    done: dict[str, tuple[dict, list]] = {}
    earlier_coding: dict[str, dict] = {}
    flagged_keys: dict[str, set[str]] = {}
    sibling_keys: dict[str, set[str]] = {}
    label_hold_keys: dict[str, set[str]] = {}
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
                document_text=(documents or {}).get(str(passage.get("doc_id") or ""), ""),
            )
            coding = _coding_payload(coding)
            coding["codes"] = list(interpreted.get("expanded_codes") or coding.get("codes") or [])
            quote_failures = list(interpreted.get("quote_retries") or [])
            label_failures = list(interpreted.get("label_retries") or [])
            restated = [
                {"code": item.get("code"), "reason": "context_restatement", "quote": item.get("quote")}
                for item in interpreted["accepted"]
                if "context_restatement" in (item.get("review_reasons") or [])
            ]
            label_keyset = {
                code_key(str(item.get("source_code", item.get("code")) or "")) for item in label_failures
            }
            quote_keyset = {
                code_key(str(item.get("code") or ""))
                for item in quote_failures
                if item.get("code") and code_key(str(item.get("code") or "")) not in label_keyset
            }
            sibling_keyset = {
                code_key(str(item.get("code") or ""))
                for item in coding.get("codes") or []
                if item.get("_joined_part") and code_key(str(item.get("code") or "")) not in label_keyset
            }
            retryable = quote_failures + restated + label_failures
            if retryable and attempt < attempts - 1:
                errors[passage_id].append(retryable)
                earlier_coding[passage_id] = coding
                flagged_keys[passage_id] = quote_keyset
                sibling_keys[passage_id] = sibling_keyset
                label_hold_keys[passage_id] = label_keyset
                still_pending.append(passage)
                feedback[passage_id] = validation_feedback(retryable)
            else:
                if passage_id in earlier_coding:
                    returned = list(coding.get("codes") or [])
                    coding = _preserve_flagged_codes(
                        earlier_coding[passage_id],
                        coding,
                        flagged_keys.get(passage_id) or set(),
                    )
                    coding = _preserve_flagged_codes(
                        earlier_coding[passage_id],
                        coding,
                        sibling_keys.get(passage_id) or set(),
                    )
                    if not returned:
                        coding = _preserve_flagged_codes(
                            earlier_coding[passage_id],
                            coding,
                            label_hold_keys.get(passage_id) or set(),
                        )
                done[passage_id] = (coding, errors[passage_id])
        pending = still_pending
    if runtime.check_model is not None:
        await _annotate_leaks(runtime, done)
    for passage in passages:
        passage_id = str(passage.get("passage_id"))
        done.setdefault(passage_id, ({"passage_id": passage_id, "codes": [], "memo": ""}, errors[passage_id]))
    return done


async def _rank_passages(
    runtime: RunConfig,
    passages: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    """Rankings for these passages. A planning pass for this snapshot is reused."""
    texts = [str(passage.get("text") or "") for passage in passages]
    await _ensure_rankings(runtime, texts, entries)
    snapshot = _ranking_snapshot(entries)
    k = max(0, int(runtime.top_k_codes))
    return [list(_cached_labels(runtime, snapshot, text, k) or []) for text in texts]


def _passage_ids(passages: Sequence[Mapping[str, Any]] | Sequence[str]) -> list[str]:
    ids: list[str] = []
    for item in passages:
        if isinstance(item, str):
            ids.extend(part for part in item.split(",") if part)
        else:
            passage_id = str(item.get("passage_id") or "")
            if passage_id:
                ids.append(passage_id)
    return ids


def _record_oversized(
    runtime: RunConfig,
    passages: Sequence[Mapping[str, Any]] | Sequence[str],
    estimated: int,
    budget: int,
    *,
    kind: str,
    log: bool = True,
) -> None:
    """Warn and store one ``oversized_request`` flag per passage. The call is still sent."""
    ids = _passage_ids(passages) or ["unknown"]
    note = oversized_request_message(kind, estimated, budget, ",".join(ids))
    _remember_note(runtime, note)
    if log:
        logger.warning("%s", note)
    for passage_id in ids:
        row = {
            "flag": "oversized_request",
            "passage_id": passage_id,
            "estimated": int(estimated),
            "budget": int(budget),
            "kind": kind,
        }
        if row not in runtime.oversized_requests:
            runtime.oversized_requests.append(row)


def _attach_oversized(
    runtime: RunConfig,
    state: Mapping[str, Any],
    review: list,
) -> tuple[dict | None, list]:
    """Copy flags into run metadata and the review list. Coding is not dropped."""
    rows = [dict(item) for item in runtime.oversized_requests]
    runtime.oversized_requests.clear()
    if not rows:
        return None, review
    metadata = dict(state.get("metadata") or {})
    stored = [dict(item) for item in (metadata.get("oversized_requests") or [])]
    review = list(review)
    for row in rows:
        stored.append(row)
        review.append(
            {
                "status": "review",
                "reason": "oversized_request",
                "flag": "oversized_request",
                "passage_id": row["passage_id"],
                "estimated": row["estimated"],
                "budget": row["budget"],
                "kind": row.get("kind") or "coding",
            }
        )
    metadata["oversized_requests"] = stored
    return metadata, review


def _on_schema_overflow(runtime: RunConfig, passage_id: str):
    def _record(flag: OversizedRequest) -> None:
        _record_oversized(
            runtime,
            [passage_id] if passage_id else [],
            flag.estimated,
            flag.budget,
            kind="schema_repair",
            log=False,
        )

    return _record


async def _invoke_coding(
    runtime: RunConfig,
    passages: list[dict],
    codebook: dict,
    feedback: dict[str, str] | None,
) -> list[dict]:
    """Retrieve, measure the real request, then repack or trim to the budget."""
    entries = list(codebook.values())
    per_passage = await _rank_passages(runtime, passages, entries)
    budget = max(1, int(runtime.token_budget))
    groups = _passage_groups(runtime, passages, entries, per_passage, feedback, budget)
    if len(groups) > 1:
        note = (
            f"Repacked {len(passages)} passages into {len(groups)} coding requests "
            f"to fit token_budget ({budget})."
        )
        if note not in runtime.notes:
            runtime.notes.append(note)
    aligned: list[dict] = []
    for group in groups:
        subset = [passages[index] for index in group]
        ranks = [per_passage[index] for index in group]
        messages = _fit_coding_messages(runtime, subset, entries, ranks, feedback, budget)
        passage_ids = ",".join(str(passage.get("passage_id")) for passage in subset)
        cost = _coding_request_cost(runtime, messages, len(subset))
        if cost > budget:
            _record_oversized(runtime, subset, cost, budget, kind="coding")
        parsed = await _ainvoke_structured(
            runtime.coder,
            BatchCoding,
            messages,
            kind="code_passage",
            model_name=runtime.model_name,
            passage_id=passage_ids,
            runtime=runtime,
            reserved_output=reserved_output_tokens(
                len(subset),
                max_codes_per_passage=runtime.max_codes_per_passage,
            ),
            token_budget=budget,
        )
        aligned.extend(_align_batch(parsed, subset))
    return aligned


def _align_batch(payload: dict, passages: list[dict]) -> list[dict]:
    """One output slot per passage. A missing slot is a failure, not an empty coding."""
    items = list(payload.get("passages") or [])
    by_id = {str(item.get("passage_id")): item for item in items if item.get("passage_id")}
    aligned = []
    for index, passage in enumerate(passages):
        passage_id = str(passage.get("passage_id") or "")
        item = by_id.get(passage_id)
        if item is None and index < len(items) and not str(items[index].get("passage_id") or ""):
            item = dict(items[index])
            item["passage_id"] = passage_id
        failed = False
        if item is None:
            failed = True
            item = {
                "passage_id": passage_id,
                "codes": [],
                "memo": "",
                "low_confidence": True,
                "coverage_status": "failed",
            }
        else:
            item = dict(item)
        if failed:
            item["coverage_status"] = "failed"
        elif item.get("codes"):
            item["coverage_status"] = item.get("coverage_status") or "coded"
        else:
            item["coverage_status"] = item.get("coverage_status") or "context_only"
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
            coding = item.get("coding") or {}
            if hasattr(coding, "model_dump"):
                coding = coding.model_dump()
            interpreted = interpret_coding(
                coding,
                passage,
                book,
                max_codes=runtime.max_codes_per_passage,
                require_gerund=runtime.require_gerund,
                long_passage_chars=runtime.long_passage_chars,
                review_below=runtime.review_below,
                context_terms=context_term_set(runtime.context),
                document_text=_document_texts(state).get(str(passage.get("doc_id") or ""), ""),
                passages=state.get("passages") or [],
            )
            kept_quotes = [
                {
                    "code": row.get("code"),
                    "reason": f"quote_{row.get('quote_status')}",
                    "quote": row.get("model_quote") or row.get("quote"),
                    "quote_status": row.get("quote_status"),
                    "quote_similarity": row.get("quote_similarity"),
                    "kept": True,
                }
                for row in interpreted["accepted"]
                if row.get("quote_status") in {"possibly_modified", "unanchored"}
            ]
            kept_labels = [
                {
                    "code": row.get("code"),
                    "reason": reason,
                    "quote": row.get("quote"),
                    "kept": True,
                }
                for row in interpreted["accepted"]
                for reason in (row.get("review_reasons") or [])
                if reason in {"empty_code", "code_too_long", "bundled_concepts"}
            ]
            if kept_quotes or kept_labels:
                validation_log.append(
                    {
                        "passage_id": passage.get("passage_id"),
                        "doc_id": passage.get("doc_id"),
                        "stage": "kept",
                        "rejections": kept_quotes + kept_labels,
                    }
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
            coverage = str(coding.get("coverage_status") or "")
            coded_here = [
                row
                for row in interpreted["accepted"]
                if str(row.get("attributed_passage_id") or passage.get("passage_id") or "")
                == str(passage.get("passage_id") or "")
            ]
            if coverage == "failed":
                passage["coverage_status"] = "failed"
            elif coded_here:
                passage["coverage_status"] = "coded"
            else:
                passage["coverage_status"] = "context_only"
            passages_by_id = {
                str(row.get("passage_id") or ""): row for row in (state.get("passages") or [])
            }
            for row in interpreted["accepted"]:
                attributed = str(row.get("attributed_passage_id") or "")
                other = passages_by_id.get(attributed)
                if other is None or other is passage:
                    continue
                if other.get("coverage_status") in {"failed", "excluded"}:
                    continue
                other["coverage_status"] = "coded"
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
                origin_snapshot_id=str(state.get("snapshot_id") or ""),
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
        metadata, review = _attach_oversized(runtime, state, review)

        if state.get("parallel"):
            codebook = union_codebooks(list(branch_books.values()))
            codebook = _resync(codebook, segments)

        returned = {
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
            "passages": state.get("passages") or [],
        }
        if metadata is not None:
            returned["metadata"] = metadata
        return returned

    runtime._commit = validate_passage
    return validate_passage


def _append_reading_notes(memos: list, counters: dict, pending_sorted: list, state: CodingState) -> None:
    """One comparison memo and one provenance note per document in this commit.

    The comparison memo quotes the incident memos the model wrote. It does not
    call the model again. The provenance note records reading order only. It
    is not a reflexivity memo. A reflexivity memo is the one the sweep model
    writes when it has something to say about the analyst's position.
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
                "kind": "provenance",
                "doc_id": doc_id,
                "passage_id": None,
                "sweep_index": None,
                "text": (
                    f"Provenance: read {doc_name} in seeded document order before later documents. "
                    f"Passages reviewed: {listed}. "
                    "This note records reading order. It is not a reflexivity memo."
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
        runtime.limiter.cache_context = {
            "snapshot_id": state.get("snapshot_id") or "",
            "definition_revision": state.get("definition_revision") or "",
            "evidence_revision": state.get("evidence_revision") or "",
            "policy_version": POLICY_VERSION,
        }

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
        # Overlap needs a free slot. At concurrency 1 the coding task takes the
        # only slot and the sweep waits with nothing of its own in flight.
        if runtime.pipeline and _remaining(state) and int(runtime.concurrency) > 1:
            jobs = await _wave(state, runtime, upcoming=True)
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
        overlap_pairs: list = []
        if overlap_pending and getattr(runtime, "_commit", None):
            overlap_pending, overlap_pairs = await reconcile_in_batch(runtime, state, overlap_pending)
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
                    "The sweep judged this pair with the same-action, same-object rubric. "
                    "Stance is context, not a merge veto."
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
                        or "This pair was already decided in this run, and neither definition nor its evidence has changed.",
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
        for key, fingerprint in (parsed.get("reviewed_revisions") or {}).items():
            if key in applied["codebook"]:
                applied["codebook"][key]["reviewed_revision"] = fingerprint
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
                quote_stats["merges_later_split"] = int(quote_stats.get("merges_later_split") or 0) + int(
                    extra.get("merges_later_split") or 0
                )
                quote_stats["split_merges"] = list(quote_stats.get("split_merges") or []) + list(
                    extra.get("split_merges") or []
                )
        extras["rewrites"] = time.perf_counter() - rewrite_started
        finalize_candidate_decisions(
            applied,
            candidates,
            rejections=list(parsed.get("rejections") or []),
            skipped=list(parsed.get("skipped_candidates") or []),
        )
        base_warnings = list(state.get("warnings") or [])
        if overlap_state is not None:
            for warning in overlap_state.get("warnings") or []:
                if warning not in base_warnings:
                    base_warnings.append(warning)
        warnings = base_warnings + list(parsed.get("warnings") or []) + list(runtime.notes)
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
            "codes_sent": int(parsed.get("codes_sent", len(codes))),
            "audit": str(parsed.get("audit") or "full"),
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
        oversized_metadata, review = _attach_oversized(runtime, carried, review)
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
        from automated_grounded_theory.codebook import mark_later_splits
        from automated_grounded_theory.records import accumulate_duplicate_rounds, duplicate_pair_count, same_round_duplicates

        audit = list(state.get("lost_distinction_candidates") or []) + list(
            applied.get("lost_distinction_candidates") or []
        )
        split_merges = list(quote_stats.get("split_merges") or [])
        mark_later_splits(audit, split_merges)
        update["lost_distinction_candidates"] = audit
        judgments = list(applied.get("accepted_merges") or []) + list(applied.get("rejected_merges") or []) + list(proposed)
        pairs = same_round_duplicates(pre_merge_book, judgments)
        metrics = dict(state.get("metrics") or {})
        if overlap_pairs:
            in_batch_rounds = list(metrics.get("in_batch_rounds") or [])
            in_batch_rounds.append(
                {"snapshot_id": str(state.get("snapshot_id") or ""), "pairs": overlap_pairs}
            )
            metrics["in_batch_rounds"] = in_batch_rounds
        rounds = accumulate_duplicate_rounds(list(metrics.get("duplicate_rounds") or []), pairs)
        metrics["duplicate_rounds"] = rounds
        metrics["duplicates_before_reconciliation"] = duplicate_pair_count(rounds)
        metrics["lost_distinction_candidates"] = len(audit)
        metrics["merges_later_split"] = int(metrics.get("merges_later_split") or 0) + int(
            quote_stats.get("merges_later_split") or 0
        )
        update["metrics"] = metrics
        if oversized_metadata is not None:
            update["metadata"] = oversized_metadata
        elif overlap_state is not None and overlap_state.get("metadata"):
            update["metadata"] = overlap_state["metadata"]
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
        metrics = dict(state.get("metrics") or {})
        report = runtime.limiter.activity_report()
        metrics["activity"] = report
        metrics["max_in_flight"] = report.get("max_in_flight")
        metrics["request_latency_seconds"] = list(report.get("request_latency_seconds") or [])
        metrics["queue_wait_seconds"] = report.get("queue_wait_seconds")
        state = {**state, "metrics": metrics}
        provisional = result_from_state(state, interrupted=False, thread_id=str(state.get("metadata", {}).get("thread_id") or ""), resumable=False)
        paths: dict[str, str] = {}
        if runtime.output_dir:
            paths = export_result(provisional, runtime.output_dir)
        return {"export_paths": paths, "finished": True, "metrics": metrics}

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
    """Second structured look at each proposed merge. Stance is context, not a veto."""
    eligible = []
    rejected = []
    for merge in merges:
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


async def _label_discovery_pairs(runtime: RunConfig, codes: list, sweep_index: int) -> list[dict]:
    """One compact label-only pass. It does not repeat definitions on every chunk."""
    messages = build_candidate_discovery_messages(
        context=runtime.context,
        codes=codes,
        prompt_cache=runtime.prompt_cache,
    )
    payload = await _ainvoke_structured(
        runtime.sweeper,
        CandidateDiscovery,
        messages,
        kind="candidate_discovery",
        model_name=runtime.sweep_model_name,
        passage_id=f"sweep-{sweep_index}-discovery",
        runtime=runtime,
    )
    known = {code_key(str(entry.get("code") or "")): str(entry.get("code") or "") for entry in codes}
    pairs = []
    for item in payload.get("pairs") or []:
        source = known.get(code_key(str(item.get("source") or "")))
        target = known.get(code_key(str(item.get("target") or "")))
        if not source or not target or code_key(source) == code_key(target):
            continue
        pairs.append({"source": source, "target": target, "score": 1.0, "method": "discovery"})
    return pairs


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
    audit_all = full_audit_due(
        sweep_index,
        consolidation=consolidation,
        full_audit_every=runtime.full_audit_every,
    )
    audit = "full" if audit_all else "incremental"
    due = list(codes) if audit_all else [entry for entry in codes if code_needs_review(entry)]
    if not due:
        return (
            {
                "merges": [],
                "rejections": [],
                "definitions": [],
                "overloaded_codes": [],
                "relation_hints": [],
                "integrative_memo": "",
                "reflexive_memo": "",
                "skipped_candidates": [],
                "warnings": [
                    f"Sweep {sweep_index} skipped the model. No code was new or materially changed."
                ],
                "reviewed_revisions": {},
                "audit": audit,
                "codes_sent": 0,
            },
            [],
            0,
        )
    chunks = iter_sweep_chunks(due, runtime.sweep_chunk)
    vectors = None
    embed_note = ""
    if runtime.embeddings is not None:
        try:
            vectors = await shared_codebook_vectors(
                runtime.embeddings,
                codes,
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

    due_keys = {code_key(str(entry.get("code") or "")) for entry in due}
    due_indexes = [
        index
        for index, entry in enumerate(codes)
        if code_key(str(entry.get("code") or "")) in due_keys
    ]
    hints = await asyncio.to_thread(
        hint_pairs,
        codes,
        vectors,
        embedding_threshold=runtime.merge_threshold,
        lexical_threshold=runtime.lexical_merge_threshold,
        limit=runtime.candidate_limit,
        chunk_indexes=None if audit_all else due_indexes,
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
    # Jaccard at the proposal threshold is a hint, not an auto-merge. Pairs the
    # lexical threshold already returned stay as they are. This adds any
    # near-duplicate the cap or a higher threshold would have dropped.
    offered = {
        frozenset((code_key(str(hint.get("source") or "")), code_key(str(hint.get("target") or ""))))
        for hint in hints
    }
    for proposal in lexical_comparison_proposals(codes):
        source_key = code_key(str(proposal.get("source") or ""))
        target_key = code_key(str(proposal.get("target") or ""))
        pair = frozenset((source_key, target_key))
        if not pair or pair in offered or (frozen and pair in frozen):
            continue
        if not audit_all and source_key not in due_keys and target_key not in due_keys:
            continue
        hints.append(proposal)
        offered.add(pair)
    warnings: list[str] = [embed_note] if embed_note else []
    if len(chunks) > 1:
        try:
            discovered = await _label_discovery_pairs(runtime, codes, sweep_index)
        except CodingError as exc:
            if not is_transient(exc):
                raise
            discovered = []
            warnings.append(
                f"Sweep {sweep_index} candidate discovery was skipped after a timeout or connection error: {exc}"
            )
        for proposal in discovered:
            source_key = code_key(str(proposal.get("source") or ""))
            target_key = code_key(str(proposal.get("target") or ""))
            pair = frozenset((source_key, target_key))
            if not pair or pair in offered or (frozen and pair in frozen):
                continue
            hints.append(proposal)
            offered.add(pair)
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
    if len(codes) >= 2 and not hints:
        warnings.append(
            f"Sweep {sweep_index} found no similarity hints among {len(codes)} codes. "
            "The model still reviews the codebook and may propose duplicate merges."
        )
    skipped_candidates: list[dict] = []
    calls: list[dict] = []
    span = max(len(chunks), 1)
    for owner, chunk in enumerate(chunks, start=1):
        planned, leftover = _plan_chunk_requests(
            codes,
            chunk,
            vectors,
            per_chunk_candidates[owner - 1],
            limit=runtime.cross_chunk_limit,
            embedding_threshold=runtime.merge_threshold,
            lexical_threshold=runtime.lexical_merge_threshold,
        )
        if leftover:
            warnings.append(
                f"Sweep {sweep_index} could not show evidence for {len(leftover)} hinted pairs "
                f"in chunk {owner}. They were not sent for a decision."
            )
        for part, item in enumerate(planned):
            order = owner if item["primary"] else owner + part * span * 1000
            calls.append({**item, "owner": owner, "part": part, "order": order, "chunk": chunk})

    async def confirm(call: dict) -> tuple:
        owner = int(call["owner"])
        if call["primary"]:
            passage_id = (
                f"sweep-{sweep_index}" if len(chunks) == 1 else f"sweep-{sweep_index}-chunk-{owner:04d}"
            )
        else:
            passage_id = f"sweep-{sweep_index}-chunk-{owner:04d}-compare-{int(call['part']):04d}"
        messages = build_sweep_messages(
            context=runtime.context,
            codes=call["local"],
            candidates=call["hints"],
            prompt_cache=runtime.prompt_cache,
            all_codes=codes,
            cross_chunk=call["cross"],
            chunk_index=owner,
            chunk_count=len(chunks),
            consolidation=consolidation,
            audit=audit,
        )
        empty = {
            "merges": [],
            "rejections": [],
            "definitions": [],
            "overloaded_codes": [],
            "relation_hints": [],
            "integrative_memo": "",
            "reflexive_memo": "",
        }
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
            return call["order"], payload, None, call
        except CodingError as exc:
            if not is_transient(exc):
                raise
            label = "chunk" if call["primary"] else "comparison"
            note = (
                f"Sweep {sweep_index} {label} {owner} of {len(chunks)} was skipped after a "
                f"timeout or connection error: {exc}"
            )
            return call["order"], empty, note, call

    confirmed = await asyncio.gather(*(confirm(call) for call in calls))
    confirmed.sort(key=lambda item: item[0])
    parts = []
    reviewed: dict[str, str] = {}
    for index, parsed, note, call in confirmed:
        if note:
            warnings.append(note)
            for candidate in call["hints"]:
                skipped_candidates.append({**candidate, "rationale": note})
        elif call["primary"]:
            for entry in call["chunk"]:
                key = code_key(str(entry.get("code") or ""))
                if key:
                    reviewed[key] = review_fingerprint(entry)
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
    reconciled["reviewed_revisions"] = reviewed
    reconciled["audit"] = audit
    reconciled["codes_sent"] = len(due)
    return reconciled, candidates, len(chunks)


def _evidence_keys(entries: Sequence[Mapping[str, Any]]) -> set[str]:
    keys = set()
    for entry in entries:
        key = code_key(str(entry.get("code") or ""))
        if key:
            keys.add(key)
    return keys


def _partition_hints(hints: Sequence[Mapping[str, Any]], evidence: set[str]) -> tuple[list, list]:
    """Hints whose both codes are in ``evidence``, and the ones that are not."""
    ready: list = []
    deferred: list = []
    for hint in hints:
        source = code_key(str(hint.get("source") or ""))
        target = code_key(str(hint.get("target") or ""))
        if source and target and source in evidence and target in evidence:
            ready.append(hint)
        else:
            deferred.append(hint)
    return ready, deferred


def _locals_for(chunk: Sequence[Mapping[str, Any]], hints: Sequence[Mapping[str, Any]]) -> list:
    """Chunk codes that are an endpoint of ``hints``. The follow-up stays bounded."""
    needed: set[str] = set()
    for hint in hints:
        needed.add(code_key(str(hint.get("source") or "")))
        needed.add(code_key(str(hint.get("target") or "")))
    chosen = [entry for entry in chunk if code_key(str(entry.get("code") or "")) in needed]
    return chosen or list(chunk)


def _plan_chunk_requests(
    codes: list,
    chunk: list,
    vectors,
    hints: list,
    *,
    limit: int,
    embedding_threshold: float,
    lexical_threshold: float,
) -> tuple[list[dict], list]:
    """One sweep of ``chunk``, then further bounded comparisons for deferred hints.

    A hint is sent only when both codes' evidence is in that request. Partners
    that do not fit in ``limit`` are queued onto the next request. Hints that
    still have nowhere to go (a limit of zero, or a label that is not in the
    book) are returned as leftovers and are not sent.
    """
    remaining = list(hints)
    planned: list[dict] = []
    primary = True
    while True:
        cross = _cross_chunk_for(
            codes,
            chunk,
            vectors,
            remaining,
            limit=limit,
            embedding_threshold=embedding_threshold,
            lexical_threshold=lexical_threshold,
        )
        evidence = _evidence_keys(chunk) | _evidence_keys(cross)
        ready, deferred = _partition_hints(remaining, evidence)
        local = list(chunk) if primary else _locals_for(chunk, ready)
        # Follow-ups send only the local endpoints. Re-check against that set
        # so a hint is never asked for without both definitions in the request.
        sent = _evidence_keys(local) | _evidence_keys(cross)
        ready, held = _partition_hints(ready, sent)
        deferred = held + deferred
        if primary:
            planned.append({"local": local, "cross": list(cross), "hints": ready, "primary": True})
            primary = False
        elif ready:
            planned.append({"local": local, "cross": list(cross), "hints": ready, "primary": False})
        else:
            return planned, deferred
        if not deferred:
            return planned, []
        if len(deferred) >= len(remaining):
            return planned, deferred
        remaining = deferred


def _cross_chunk_for(
    codes: list,
    chunk: list,
    vectors,
    hints: list,
    *,
    limit: int,
    embedding_threshold: float,
    lexical_threshold: float,
) -> list:
    """Full evidence for hinted partners outside the chunk, then other similar codes."""
    cap = max(0, int(limit))
    if cap == 0:
        return []
    by_key = {}
    for entry in codes:
        key = code_key(str(entry.get("code") or ""))
        if key:
            by_key[key] = entry
    in_chunk = {code_key(str(entry.get("code") or "")) for entry in chunk}
    selected: list = []
    seen: set[str] = set()

    def add(key: str) -> None:
        if key in in_chunk or key in seen or key not in by_key or len(selected) >= cap:
            return
        seen.add(key)
        selected.append(by_key[key])

    for hint in hints:
        add(code_key(str(hint.get("source") or "")))
        add(code_key(str(hint.get("target") or "")))
    if len(selected) < cap:
        for entry in select_cross_chunk(
            codes,
            chunk,
            vectors,
            limit=cap,
            embedding_threshold=embedding_threshold,
            lexical_threshold=lexical_threshold,
        ):
            add(code_key(str(entry.get("code") or "")))
    return selected


async def _ainvoke_structured(
    model: BaseChatModel,
    schema,
    messages: list,
    *,
    kind: str,
    model_name: str,
    passage_id: str,
    runtime: RunConfig,
    reserved_output: int | None = None,
    token_budget: int | None = None,
) -> dict:
    timeout = None
    if kind in {"sweep", "merge_verify", "definition_rewrite", "quote_fit", "candidate_discovery"}:
        timeout = sweep_deadline(runtime.limiter.request_timeout, runtime.sweep_timeout)
    try:
        (payload, usage), used = await ainvoke_structured(
            model,
            schema,
            messages,
            runtime.limiter,
            timeout=timeout,
            reserved_output=reserved_output,
            token_budget=token_budget,
            tokenizer=resolve_tokenizer(model),
            passage_id=passage_id,
            on_overflow=_on_schema_overflow(runtime, passage_id),
        )
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
    timeout = None
    if kind in {"sweep", "merge_verify", "definition_rewrite", "quote_fit", "candidate_discovery"}:
        timeout = sweep_deadline(runtime.limiter.request_timeout, runtime.sweep_timeout)
    payloads = []
    for messages, result, passage_id in zip(message_lists, results, passage_ids):
        try:
            payload, usage = parse_structured(result, schema)
            used = messages
        except ValueError:
            # The failed response is already in the limiter cache. Replaying it
            # through ainvoke_structured repairs once and reports that usage
            # plus the repaired response. Adding it here would count it twice.
            try:
                (payload, usage), used = await ainvoke_structured(
                    model, schema, messages, runtime.limiter, timeout=timeout
                )
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
