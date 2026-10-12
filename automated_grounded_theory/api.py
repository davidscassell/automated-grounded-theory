"""Public entry points for first-round open coding."""

from __future__ import annotations

import asyncio
import uuid
import warnings
from pathlib import Path
from typing import Any, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from automated_grounded_theory.async_support import CallLimiter, run_sync
from automated_grounded_theory.backends import (
    describe_model,
    prompt_cache_supported,
    redact,
    resolve_chat_model,
    resolve_embeddings,
)
from automated_grounded_theory.context import ResearchContext, parse_context
from automated_grounded_theory.corpus import LoadedDocument, load_corpus
from automated_grounded_theory.export import export_result
from automated_grounded_theory.graph import CodingError, RunConfig, compile_graph, initial_state
from automated_grounded_theory.prompts import CODING_SYSTEM, SWEEP_SYSTEM
from automated_grounded_theory.results import OpenCodeResult, result_from_state
from automated_grounded_theory.retrieval import (
    CodeRetriever,
    EmbeddingCodeRetriever,
    LexicalCodeRetriever,
)
from automated_grounded_theory.segment import UNITS, CodingPlan, plan_coding
from automated_grounded_theory.stability import StabilityReport, compare_runs
from automated_grounded_theory.version import __version__

__all__ = [
    "CodingError",
    "CodingPlan",
    "OpenCodeResult",
    "ResearchContext",
    "StabilityReport",
    "amulti_seed_stability",
    "aopen_code",
    "compare_runs",
    "multi_seed_stability",
    "open_code",
    "plan_coding",
    "resolve_coding_schedule",
    "strict_sequential_settings",
]


def strict_sequential_settings() -> dict[str, Any]:
    """One passage at a time, committed before the next passage is coded.

    ``review_mode`` is ``document``. One passage is sent per request, one
    request is in flight, and the codebook is committed after every passage.
    Pipeline overlap is off, and documents are not coded in parallel. A later
    passage therefore sees codes committed from earlier passages.

    Pass the dict as keyword arguments to :func:`open_code`, or set
    ``procedure="strict_sequential"``.
    """
    return {
        "procedure": "strict_sequential",
        "review_mode": "document",
        "passages_per_request": 1,
        "max_concurrent_requests": 1,
        "passages_per_codebook_update": 1,
        "batch_size": 1,
        "concurrency": 1,
        "pipeline": False,
        "parallel_documents": False,
    }


def resolve_coding_schedule(
    *,
    batch_size: int = 1,
    concurrency: int = 4,
    passages_per_request: int | None = None,
    max_concurrent_requests: int | None = None,
    passages_per_codebook_update: int | None = None,
    procedure: str | None = None,
    review_mode: str = "document",
    pipeline: bool = True,
    parallel_documents: bool = False,
) -> dict[str, Any]:
    """Split the old batch and concurrency knobs into procedure and performance.

    ``passages_per_request`` is how many passages share one model call.
    ``max_concurrent_requests`` caps in-flight calls and does not change which
    passages share a codebook. ``passages_per_codebook_update`` is how many
    passages are coded before that codebook is committed.

    ``batch_size`` is the deprecated alias of ``passages_per_request``.
    ``concurrency`` is the deprecated alias of ``max_concurrent_requests``.
    When ``batch_size`` is 1, concurrency used to code several passages of one
    document against one frozen codebook. That no longer happens. Set
    ``passages_per_codebook_update`` to ask for that wider window on purpose.
    """
    if procedure not in (None, "", "strict_sequential"):
        raise ValueError("procedure must be 'strict_sequential' or omitted.")
    if procedure == "strict_sequential":
        if review_mode != "document":
            raise ValueError("strict_sequential uses review_mode='document'.")
        if parallel_documents:
            raise ValueError("strict_sequential does not code documents in parallel.")
        if batch_size != 1:
            raise ValueError("strict_sequential sends one passage per request. Do not set batch_size.")
        if concurrency not in (1, 4):
            raise ValueError(
                "strict_sequential uses one request at a time. "
                "Do not set concurrency; use strict_sequential_settings()."
            )
        if passages_per_request not in (None, 1) or max_concurrent_requests not in (None, 1):
            raise ValueError("strict_sequential uses one passage per request and one request in flight.")
        if passages_per_codebook_update not in (None, 1):
            raise ValueError("strict_sequential commits the codebook after every passage.")
        return strict_sequential_settings()

    per_request = batch_size if passages_per_request is None else passages_per_request
    concurrent = concurrency if max_concurrent_requests is None else max_concurrent_requests
    if passages_per_request is not None and batch_size not in (1, passages_per_request):
        warnings.warn(
            "batch_size is ignored because passages_per_request is set.",
            DeprecationWarning,
            stacklevel=3,
        )
    if (
        passages_per_request is None
        and max_concurrent_requests is None
        and passages_per_codebook_update is None
        and batch_size == 1
        and concurrency > 1
    ):
        warnings.warn(
            "concurrency no longer codes several passages against one frozen codebook. "
            "It only caps in-flight requests (max_concurrent_requests). "
            "batch_size is the number of passages per request (passages_per_request). "
            "Set passages_per_codebook_update to code more than one passage before the codebook is committed. "
            "strict_sequential_settings() commits after every passage.",
            DeprecationWarning,
            stacklevel=3,
        )
    commit_every = per_request if passages_per_codebook_update is None else passages_per_codebook_update
    return {
        "procedure": procedure or "",
        "review_mode": review_mode,
        "passages_per_request": int(per_request),
        "max_concurrent_requests": int(concurrent),
        "passages_per_codebook_update": int(commit_every),
        "batch_size": int(per_request),
        "concurrency": int(concurrent),
        "pipeline": pipeline,
        "parallel_documents": parallel_documents,
    }


def open_code(
    corpus: Any,
    context: ResearchContext | dict | str,
    *,
    model: BaseChatModel | str,
    seed: int | None = None,
    sweep_every: int = 25,
    sweep_model: BaseChatModel | str | None = None,
    unit: str = "incident",
    embeddings: Embeddings | str | None = None,
    api_key: str | None = None,
    embeddings_api_key: str | None = None,
    temperature: float = 0.0,
    sweep_temperature: float | None = None,
    parallel_documents: bool = False,
    batch_size: int = 1,
    concurrency: int = 4,
    passages_per_request: int | None = None,
    max_concurrent_requests: int | None = None,
    passages_per_codebook_update: int | None = None,
    procedure: str | None = None,
    top_k_codes: int = 12,
    frequent_codes: int = 8,
    merge_threshold: float = 0.65,
    lexical_merge_threshold: float = 0.5,
    checkpoint_path: str | Path | None = None,
    thread_id: str | None = None,
    resume: bool = False,
    interrupt_before_sweep: bool = False,
    output_dir: str | Path | None = None,
    prompt_log_path: str | Path | None = None,
    code_retriever: CodeRetriever | None = None,
    llm_seed: int | None = None,
    model_kwargs: dict | None = None,
    sweep_model_kwargs: dict | None = None,
    embeddings_kwargs: dict | None = None,
    long_passage_chars: int = 2000,
    max_codes_per_passage: int = 6,
    neighbor_chars: int = 400,
    validation_retries: int = 1,
    require_gerund: bool = False,
    candidate_limit: int = 30,
    review_mode: str = "document",
    review_below: float = 0.5,
    min_merge_confidence: float = 0.5,
    max_merges_per_sweep: int = 12,
    check_model: BaseChatModel | str | None = None,
    pipeline: bool = True,
    token_budget: int = 4000,
    prompt_cache: bool | None = None,
    recursion_limit: int | None = None,
    checkpointer: Any = None,
    sweep_chunk: int = 20,
    embed_batch_size: int = 64,
    max_retries: int = 4,
    retry_backoff: float = 0.5,
    retry_max_backoff: float = 8.0,
    request_timeout: float = 120.0,
    sweep_timeout: float = 300.0,
) -> OpenCodeResult:
    """Run first-round grounded-theory open coding over ``corpus``.

    ``model`` is either a ``provider:model-name`` string for
    :func:`langchain.chat_models.init_chat_model` or a ``BaseChatModel`` you
    built yourself. The same applies to ``sweep_model`` and ``embeddings``.
    No provider is the default. ``api_key`` is forwarded only when a model is
    created from a string; otherwise the provider SDK reads its own
    environment variable.

    ``context`` guides attention (research question, population, setting,
    sensitizing concepts). It is rejected if it contains preset codes.

    The saturation curve is reported and is not a stopping decision.
    ``passages_per_request`` passages share one model call (``batch_size`` is
    the same setting). ``max_concurrent_requests`` (``concurrency``) caps
    in-flight model calls and does not change how many passages share a
    codebook. ``passages_per_codebook_update`` is how many passages are coded
    before that codebook is committed. ``procedure="strict_sequential"`` sends
    one passage per request, allows one request, commits after every passage,
    and turns pipeline overlap and parallel documents off. A wave stops at the
    next sweep, then the codebook is updated. Sweep chunks (candidates, merge
    confirmation, definition updates) run together under the in-flight cap and
    are reconciled in chunk order.
    ``parallel_documents`` codes each document against its own codebook and
    merges those codebooks at each sweep. It is off by default because that
    delays constant comparison across documents.

    ``review_mode='document'`` (the default) reads every document in seeded
    order and every passage of that document in reading order, then updates
    the codebook. ``review_mode='wave'`` may code passages from several
    documents in one step. That is faster and weakens constant comparison.

    When ``pipeline`` is true, synthesis of the passages just coded runs while
    the next document or wave is coded against the pre-merge snapshot. The
    merge map is applied afterwards, in a fixed order. ``sweep_model`` defaults
    to ``model``. Sweep chunks default to 20 codes, sorted alphabetically by
    label and then by definition. The sweep model reads each chunk and proposes
    duplicate merges with a reason. Token overlap, and embeddings when you pass
    them, are optional hints. Merging does not require an embeddings model.
    Every hinted pair and every model-proposed merge is logged. A sweep chunk
    that still times out after retries is skipped with a warning instead of
    failing the run.

    Models built from a ``provider:model`` string use ``request_timeout``
    (120 seconds). Coding calls use that deadline. Sweep calls use
    ``sweep_timeout`` (300 seconds) when that is longer. Rate limits, timeouts,
    and connection errors retry with full-jitter backoff and honor
    ``Retry-After``. This wrapper works inside a running event loop.
    """
    return run_sync(
        aopen_code(
            corpus,
            context,
            model=model,
            seed=seed,
            sweep_every=sweep_every,
            sweep_model=sweep_model,
            unit=unit,
            embeddings=embeddings,
            api_key=api_key,
            embeddings_api_key=embeddings_api_key,
            temperature=temperature,
            sweep_temperature=sweep_temperature,
            parallel_documents=parallel_documents,
            batch_size=batch_size,
            concurrency=concurrency,
            passages_per_request=passages_per_request,
            max_concurrent_requests=max_concurrent_requests,
            passages_per_codebook_update=passages_per_codebook_update,
            procedure=procedure,
            top_k_codes=top_k_codes,
            frequent_codes=frequent_codes,
            merge_threshold=merge_threshold,
            lexical_merge_threshold=lexical_merge_threshold,
            checkpoint_path=checkpoint_path,
            thread_id=thread_id,
            resume=resume,
            interrupt_before_sweep=interrupt_before_sweep,
            output_dir=output_dir,
            prompt_log_path=prompt_log_path,
            code_retriever=code_retriever,
            llm_seed=llm_seed,
            model_kwargs=model_kwargs,
            sweep_model_kwargs=sweep_model_kwargs,
            embeddings_kwargs=embeddings_kwargs,
            long_passage_chars=long_passage_chars,
            max_codes_per_passage=max_codes_per_passage,
            neighbor_chars=neighbor_chars,
            validation_retries=validation_retries,
            require_gerund=require_gerund,
            review_mode=review_mode,
            review_below=review_below,
            min_merge_confidence=min_merge_confidence,
            max_merges_per_sweep=max_merges_per_sweep,
            check_model=check_model,
            pipeline=pipeline,
            token_budget=token_budget,
            candidate_limit=candidate_limit,
            prompt_cache=prompt_cache,
            recursion_limit=recursion_limit,
            checkpointer=checkpointer,
            sweep_chunk=sweep_chunk,
            embed_batch_size=embed_batch_size,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            request_timeout=request_timeout,
            sweep_timeout=sweep_timeout,
        )
    )


async def aopen_code(
    corpus: Any,
    context: ResearchContext | dict | str,
    *,
    model: BaseChatModel | str,
    seed: int | None = None,
    sweep_every: int = 25,
    sweep_model: BaseChatModel | str | None = None,
    unit: str = "incident",
    embeddings: Embeddings | str | None = None,
    api_key: str | None = None,
    embeddings_api_key: str | None = None,
    temperature: float = 0.0,
    sweep_temperature: float | None = None,
    parallel_documents: bool = False,
    batch_size: int = 1,
    concurrency: int = 4,
    passages_per_request: int | None = None,
    max_concurrent_requests: int | None = None,
    passages_per_codebook_update: int | None = None,
    procedure: str | None = None,
    top_k_codes: int = 12,
    frequent_codes: int = 8,
    merge_threshold: float = 0.65,
    lexical_merge_threshold: float = 0.5,
    checkpoint_path: str | Path | None = None,
    thread_id: str | None = None,
    resume: bool = False,
    interrupt_before_sweep: bool = False,
    output_dir: str | Path | None = None,
    prompt_log_path: str | Path | None = None,
    code_retriever: CodeRetriever | None = None,
    llm_seed: int | None = None,
    model_kwargs: dict | None = None,
    sweep_model_kwargs: dict | None = None,
    embeddings_kwargs: dict | None = None,
    long_passage_chars: int = 2000,
    max_codes_per_passage: int = 6,
    neighbor_chars: int = 400,
    validation_retries: int = 1,
    require_gerund: bool = False,
    candidate_limit: int = 30,
    review_mode: str = "document",
    review_below: float = 0.5,
    min_merge_confidence: float = 0.5,
    max_merges_per_sweep: int = 12,
    check_model: BaseChatModel | str | None = None,
    pipeline: bool = True,
    token_budget: int = 4000,
    prompt_cache: bool | None = None,
    recursion_limit: int | None = None,
    checkpointer: Any = None,
    sweep_chunk: int = 20,
    embed_batch_size: int = 64,
    max_retries: int = 4,
    retry_backoff: float = 0.5,
    retry_max_backoff: float = 8.0,
    request_timeout: float = 120.0,
    sweep_timeout: float = 300.0,
    limiter: CallLimiter | None = None,
) -> OpenCodeResult:
    """Async form of :func:`open_code`."""
    schedule = resolve_coding_schedule(
        batch_size=batch_size,
        concurrency=concurrency,
        passages_per_request=passages_per_request,
        max_concurrent_requests=max_concurrent_requests,
        passages_per_codebook_update=passages_per_codebook_update,
        procedure=procedure,
        review_mode=review_mode,
        pipeline=pipeline,
        parallel_documents=parallel_documents,
    )
    review_mode = schedule["review_mode"]
    pipeline = schedule["pipeline"]
    parallel_documents = schedule["parallel_documents"]
    batch_size = schedule["batch_size"]
    concurrency = schedule["concurrency"]
    passages_per_request = schedule["passages_per_request"]
    max_concurrent_requests = schedule["max_concurrent_requests"]
    passages_per_codebook_update = schedule["passages_per_codebook_update"]
    procedure = schedule["procedure"] or None
    _validate_params(
        unit=unit,
        sweep_every=sweep_every,
        top_k_codes=top_k_codes,
        max_codes_per_passage=max_codes_per_passage,
        validation_retries=validation_retries,
        batch_size=batch_size,
        concurrency=concurrency,
        passages_per_request=passages_per_request,
        max_concurrent_requests=max_concurrent_requests,
        passages_per_codebook_update=passages_per_codebook_update,
        sweep_chunk=sweep_chunk,
        embed_batch_size=embed_batch_size,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        review_mode=review_mode,
        token_budget=token_budget,
        review_below=review_below,
        min_merge_confidence=min_merge_confidence,
        request_timeout=request_timeout,
        sweep_timeout=sweep_timeout,
    )
    if max_merges_per_sweep < 1:
        raise ValueError("max_merges_per_sweep must be at least 1.")
    parsed_context = parse_context(context)
    documents = load_corpus(corpus)
    coder = resolve_chat_model(
        model,
        api_key=api_key,
        temperature=temperature,
        llm_seed=llm_seed,
        model_kwargs=model_kwargs,
        timeout=request_timeout,
    )
    if sweep_model is None or (isinstance(sweep_model, str) and not sweep_model.strip()):
        sweeper_source = model
        sweeper = coder
    else:
        sweeper_source = sweep_model
        sweeper = resolve_chat_model(
            sweep_model,
            api_key=api_key,
            temperature=temperature if sweep_temperature is None else sweep_temperature,
            llm_seed=llm_seed,
            model_kwargs=sweep_model_kwargs,
            timeout=request_timeout,
        )
    checker = None
    checker_name = ""
    if check_model is not None and not (isinstance(check_model, str) and not str(check_model).strip()):
        checker = resolve_chat_model(
            check_model,
            api_key=api_key,
            temperature=0.0,
            llm_seed=llm_seed,
            timeout=request_timeout,
        )
        checker_name = describe_model(check_model if isinstance(check_model, str) else checker)
    embed_key = api_key if embeddings_api_key is None else embeddings_api_key
    embedding_model = resolve_embeddings(
        embeddings,
        api_key=embed_key,
        embeddings_kwargs=embeddings_kwargs,
        timeout=request_timeout,
    )
    retriever = code_retriever
    if retriever is None:
        retriever = EmbeddingCodeRetriever(embedding_model) if embedding_model is not None else LexicalCodeRetriever()

    model_name = describe_model(model if isinstance(model, str) else coder)
    sweep_name = describe_model(sweeper_source if isinstance(sweeper_source, str) else sweeper)
    if isinstance(embeddings, str):
        embeddings_name = embeddings
    elif embedding_model is None:
        embeddings_name = "lexical"
    else:
        embeddings_name = type(embedding_model).__name__

    cache = prompt_cache
    if cache is None:
        cache = prompt_cache_supported(coder, model_name) or prompt_cache_supported(sweeper, sweep_name)

    if thread_id is None:
        thread_id = "default" if (checkpoint_path or checkpointer is not None) else uuid.uuid4().hex
    if resume and checkpoint_path is None and checkpointer is None:
        raise ValueError("resume=True requires checkpoint_path or checkpointer.")

    metadata = _metadata(
        version=__version__,
        seed=seed,
        unit=unit,
        sweep_every=sweep_every,
        model_name=model_name,
        sweep_name=sweep_name,
        embeddings_name=embeddings_name,
        temperature=temperature,
        sweep_temperature=temperature if sweep_temperature is None else sweep_temperature,
        parallel_documents=parallel_documents,
        batch_size=batch_size,
        concurrency=concurrency,
        passages_per_request=passages_per_request,
        max_concurrent_requests=max_concurrent_requests,
        passages_per_codebook_update=passages_per_codebook_update,
        procedure=procedure,
        review_mode=review_mode,
        review_below=review_below,
        min_merge_confidence=min_merge_confidence,
        max_merges_per_sweep=max_merges_per_sweep,
        check_model=checker_name,
        pipeline=pipeline,
        token_budget=token_budget,
        prompt_cache=cache,
        api_key_provided=api_key is not None,
        thread_id=thread_id,
        llm_seed=llm_seed,
        model_kwargs=model_kwargs,
        sweep_model_kwargs=sweep_model_kwargs,
        top_k_codes=top_k_codes,
        frequent_codes=frequent_codes,
        merge_threshold=merge_threshold,
        lexical_merge_threshold=lexical_merge_threshold,
        long_passage_chars=long_passage_chars,
        max_codes_per_passage=max_codes_per_passage,
        neighbor_chars=neighbor_chars,
        validation_retries=validation_retries,
        require_gerund=require_gerund,
        retriever=type(retriever).__name__,
        context=parsed_context,
    )
    runtime = RunConfig(
        coder=coder,
        sweeper=sweeper,
        retriever=retriever,
        embeddings=embedding_model,
        context=parsed_context,
        prompt_cache=cache,
        top_k_codes=top_k_codes,
        frequent_codes=frequent_codes,
        merge_threshold=merge_threshold,
        lexical_merge_threshold=lexical_merge_threshold,
        max_codes_per_passage=max_codes_per_passage,
        neighbor_chars=neighbor_chars,
        long_passage_chars=long_passage_chars,
        validation_retries=validation_retries,
        require_gerund=require_gerund,
        review_mode=review_mode,
        review_below=review_below,
        min_merge_confidence=min_merge_confidence,
        max_merges_per_sweep=max_merges_per_sweep,
        check_model=checker,
        pipeline=pipeline,
        token_budget=token_budget,
        candidate_limit=candidate_limit,
        batch_size=batch_size,
        concurrency=concurrency,
        passages_per_request=passages_per_request or batch_size,
        max_concurrent_requests=max_concurrent_requests or concurrency,
        passages_per_codebook_update=passages_per_codebook_update or batch_size,
        sweep_chunk=sweep_chunk,
        embed_batch_size=embed_batch_size,
        model_name=model_name,
        sweep_model_name=sweep_name,
        prompt_log_path=str(prompt_log_path) if prompt_log_path else None,
        output_dir=str(output_dir) if output_dir else None,
        sweep_timeout=sweep_timeout,
        limiter=limiter
        or CallLimiter(
            concurrency,
            max_retries=max_retries,
            backoff=retry_backoff,
            max_backoff=retry_max_backoff,
            request_timeout=request_timeout,
        ),
    )
    limit = recursion_limit if recursion_limit is not None else 250_000
    resumable = checkpoint_path is not None or checkpointer is not None

    if checkpoint_path is not None:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise ImportError(
                "checkpoint_path requires the sqlite checkpointer. "
                "Install the 'checkpoint' extra (langgraph-checkpoint-sqlite)."
            ) from exc
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            return await _execute(
                saver,
                runtime,
                documents,
                metadata,
                thread_id=thread_id,
                resume=resume,
                interrupt_before_sweep=interrupt_before_sweep,
                recursion_limit=limit,
                output_dir=output_dir,
                resumable=True,
            )
    saver = checkpointer if checkpointer is not None else _memory_saver()
    return await _execute(
        saver,
        runtime,
        documents,
        metadata,
        thread_id=thread_id,
        resume=resume,
        interrupt_before_sweep=interrupt_before_sweep,
        recursion_limit=limit,
        output_dir=output_dir,
        resumable=resumable,
    )


async def _execute(
    saver,
    runtime: RunConfig,
    documents: list[LoadedDocument],
    metadata: dict,
    *,
    thread_id: str,
    resume: bool,
    interrupt_before_sweep: bool,
    recursion_limit: int,
    output_dir: str | Path | None,
    resumable: bool,
) -> OpenCodeResult:
    graph = compile_graph(runtime, saver, interrupt_before_sweep=interrupt_before_sweep)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    snapshot = await graph.aget_state(config)
    has_state = bool(getattr(snapshot, "values", None))
    if resume:
        if not has_state:
            raise RuntimeError(
                f"No checkpoint was found for thread {thread_id!r}. Start a run before passing resume=True."
            )
        if snapshot.next:
            state = await graph.ainvoke(None, config)
        else:
            state = snapshot.values
    else:
        if has_state:
            raise RuntimeError(
                f"Checkpoint thread {thread_id!r} already has a run. "
                "Pass resume=True to continue it, or a new thread_id to start another."
            )
        state = await graph.ainvoke(initial_state(documents, metadata), config)
    snapshot = await graph.aget_state(config)
    interrupted = bool(snapshot.next)
    if interrupted:
        state = snapshot.values
    result = result_from_state(
        state,
        interrupted=interrupted,
        thread_id=thread_id,
        resumable=resumable and interrupted,
    )
    if output_dir and (interrupted or not result.export_paths):
        result.export_paths = export_result(result, output_dir)
    return result


def multi_seed_stability(
    corpus: Any,
    context: ResearchContext | dict | str,
    seeds: Sequence[int],
    **open_code_kwargs: Any,
) -> StabilityReport:
    """Run open coding once per seed, concurrently, and compare the codebooks.

    The report describes label overlap. It does not pick a winning seed.
    Seeds share one concurrency cap. Results are ordered by the ``seeds`` list,
    not by which seed finishes first. This wrapper works inside a running event loop.
    """
    return run_sync(amulti_seed_stability(corpus, context, seeds, **open_code_kwargs))


async def amulti_seed_stability(
    corpus: Any,
    context: ResearchContext | dict | str,
    seeds: Sequence[int],
    **open_code_kwargs: Any,
) -> StabilityReport:
    """Async form of :func:`multi_seed_stability`."""
    seed_list = [int(seed) for seed in seeds]
    if len(seed_list) < 2:
        raise ValueError("multi_seed_stability needs at least two seeds.")
    if "seed" in open_code_kwargs:
        raise ValueError("Pass seeds to multi_seed_stability, not seed=.")
        limiter = open_code_kwargs.pop("limiter", None)
    if limiter is None:
        limiter = CallLimiter(
            int(open_code_kwargs.get("max_concurrent_requests") or open_code_kwargs.get("concurrency", 4)),
            max_retries=int(open_code_kwargs.get("max_retries", 4)),
            backoff=float(open_code_kwargs.get("retry_backoff", 0.5)),
            max_backoff=float(open_code_kwargs.get("retry_max_backoff", 8.0)),
            request_timeout=float(open_code_kwargs.get("request_timeout", 120.0)),
        )
    base_thread = open_code_kwargs.get("thread_id")
    base_output = open_code_kwargs.get("output_dir")

    async def one(seed: int):
        kwargs = dict(open_code_kwargs)
        if base_thread:
            kwargs["thread_id"] = f"{base_thread}-seed-{seed}"
        elif kwargs.get("checkpoint_path") or kwargs.get("checkpointer") is not None:
            kwargs["thread_id"] = f"seed-{seed}"
        if base_output:
            kwargs["output_dir"] = str(Path(base_output) / f"seed-{seed}")
        if kwargs.get("checkpoint_path"):
            path = Path(kwargs["checkpoint_path"])
            kwargs["checkpoint_path"] = str(path.with_name(f"{path.stem}-seed-{seed}{path.suffix}"))
        return await aopen_code(corpus, context, seed=seed, limiter=limiter, **kwargs)

    runs = await asyncio.gather(*(one(seed) for seed in seed_list))
    report = compare_runs(runs)
    report.runs = list(runs)
    return report


def _validate_params(
    *,
    unit: str,
    sweep_every: int,
    top_k_codes: int,
    max_codes_per_passage: int,
    validation_retries: int,
    batch_size: int,
    concurrency: int,
    passages_per_request: int = 1,
    max_concurrent_requests: int = 1,
    passages_per_codebook_update: int = 1,
    sweep_chunk: int,
    embed_batch_size: int,
    max_retries: int,
    retry_backoff: float,
    review_mode: str,
    token_budget: int,
    request_timeout: float,
    sweep_timeout: float = 300.0,
    review_below: float = 0.5,
    min_merge_confidence: float = 0.5,
) -> None:
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {', '.join(UNITS)}. Got {unit!r}.")
    if sweep_every < 1:
        raise ValueError("sweep_every must be at least 1.")
    if batch_size < 1 or concurrency < 1:
        raise ValueError("batch_size and concurrency must be at least 1.")
    if passages_per_request < 1 or max_concurrent_requests < 1 or passages_per_codebook_update < 1:
        raise ValueError(
            "passages_per_request, max_concurrent_requests, and passages_per_codebook_update must be at least 1."
        )
    if sweep_chunk < 1 or embed_batch_size < 1:
        raise ValueError("sweep_chunk and embed_batch_size must be at least 1.")
    if max_retries < 0 or retry_backoff < 0:
        raise ValueError("max_retries and retry_backoff must be zero or more.")
    if request_timeout < 0 or sweep_timeout < 0:
        raise ValueError("request_timeout and sweep_timeout must be zero or more. Zero waits with no per-call deadline.")
    if review_mode not in {"document", "wave"}:
        raise ValueError("review_mode must be 'document' or 'wave'.")
    if not 0 <= review_below <= 1 or not 0 <= min_merge_confidence <= 1:
        raise ValueError("review_below and min_merge_confidence must be between 0 and 1.")
    if token_budget < 1:
        raise ValueError("token_budget must be at least 1.")
    if top_k_codes < 0 or max_codes_per_passage < 1 or validation_retries < 0:
        raise ValueError("top_k_codes, max_codes_per_passage, and validation_retries are out of range.")


def _metadata(**kwargs: Any) -> dict:
    context: ResearchContext = kwargs["context"]
    return {
        "package_version": kwargs["version"],
        "protocol": "open-coding-1",
        "seed": kwargs["seed"],
        "unit": kwargs["unit"],
        "sweep_every": kwargs["sweep_every"],
        "model": kwargs["model_name"],
        "sweep_model": kwargs["sweep_name"],
        "embeddings": kwargs["embeddings_name"],
        "temperature": kwargs["temperature"],
        "sweep_temperature": kwargs["sweep_temperature"],
        "parallel_documents": kwargs["parallel_documents"],
        "batch_size": kwargs["batch_size"],
        "concurrency": kwargs["concurrency"],
        "passages_per_request": kwargs.get("passages_per_request", kwargs["batch_size"]),
        "max_concurrent_requests": kwargs.get("max_concurrent_requests", kwargs["concurrency"]),
        "passages_per_codebook_update": kwargs.get("passages_per_codebook_update", kwargs["batch_size"]),
        "procedure": kwargs.get("procedure") or "",
        "review_mode": kwargs["review_mode"],
        "review_below": kwargs.get("review_below", 0.5),
        "min_merge_confidence": kwargs.get("min_merge_confidence", 0.5),
        "max_merges_per_sweep": kwargs.get("max_merges_per_sweep", 12),
        "check_model": kwargs.get("check_model") or "",
        "pipeline": kwargs["pipeline"],
        "token_budget": kwargs["token_budget"],
        "prompt_cache": kwargs["prompt_cache"],
        "api_key_provided": kwargs["api_key_provided"],
        "thread_id": kwargs["thread_id"],
        "llm_seed": kwargs["llm_seed"],
        "model_kwargs": redact(kwargs["model_kwargs"]),
        "sweep_model_kwargs": redact(kwargs["sweep_model_kwargs"]),
        "prompts": {"coding_system": CODING_SYSTEM, "sweep_system": SWEEP_SYSTEM},
        "params": {
            "top_k_codes": kwargs["top_k_codes"],
            "frequent_codes": kwargs["frequent_codes"],
            "merge_threshold": kwargs["merge_threshold"],
            "lexical_merge_threshold": kwargs["lexical_merge_threshold"],
            "review_below": kwargs.get("review_below", 0.5),
            "min_merge_confidence": kwargs.get("min_merge_confidence", 0.5),
            "long_passage_chars": kwargs["long_passage_chars"],
            "max_codes_per_passage": kwargs["max_codes_per_passage"],
            "neighbor_chars": kwargs["neighbor_chars"],
            "validation_retries": kwargs["validation_retries"],
            "require_gerund": kwargs["require_gerund"],
            "retriever": kwargs["retriever"],
            "parallel_documents": kwargs["parallel_documents"],
            "batch_size": kwargs["batch_size"],
            "concurrency": kwargs["concurrency"],
            "passages_per_request": kwargs.get("passages_per_request", kwargs["batch_size"]),
            "max_concurrent_requests": kwargs.get("max_concurrent_requests", kwargs["concurrency"]),
            "passages_per_codebook_update": kwargs.get("passages_per_codebook_update", kwargs["batch_size"]),
        },
        "context": {
            "research_question": context.research_question,
            "population": context.population,
            "setting": context.setting,
            "sensitizing_concepts": list(context.sensitizing_concepts),
            "notes": context.notes,
        },
    }


def _memory_saver():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
