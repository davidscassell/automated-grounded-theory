"""Public entry points for first-round open coding."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

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
    "aopen_code",
    "compare_runs",
    "multi_seed_stability",
    "open_code",
    "plan_coding",
]


def open_code(
    corpus: Any,
    context: ResearchContext | dict | str,
    *,
    model: BaseChatModel | str,
    seed: int | None = None,
    sweep_every: int = 25,
    sweep_model: BaseChatModel | str | None = None,
    unit: str = "paragraph",
    embeddings: Embeddings | str | None = None,
    api_key: str | None = None,
    embeddings_api_key: str | None = None,
    temperature: float = 0.0,
    sweep_temperature: float | None = None,
    parallel_documents: bool = False,
    top_k_codes: int = 12,
    frequent_codes: int = 8,
    merge_threshold: float = 0.82,
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
    prompt_cache: bool | None = None,
    recursion_limit: int | None = None,
    checkpointer: Any = None,
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
    ``parallel_documents`` codes each document against its own codebook and
    merges those codebooks at each sweep. It is off by default because that
    delays constant comparison across documents.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
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
                candidate_limit=candidate_limit,
                prompt_cache=prompt_cache,
                recursion_limit=recursion_limit,
                checkpointer=checkpointer,
            )
        )
    raise RuntimeError("open_code() cannot run inside a running event loop. Await aopen_code() instead.")


async def aopen_code(
    corpus: Any,
    context: ResearchContext | dict | str,
    *,
    model: BaseChatModel | str,
    seed: int | None = None,
    sweep_every: int = 25,
    sweep_model: BaseChatModel | str | None = None,
    unit: str = "paragraph",
    embeddings: Embeddings | str | None = None,
    api_key: str | None = None,
    embeddings_api_key: str | None = None,
    temperature: float = 0.0,
    sweep_temperature: float | None = None,
    parallel_documents: bool = False,
    top_k_codes: int = 12,
    frequent_codes: int = 8,
    merge_threshold: float = 0.82,
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
    prompt_cache: bool | None = None,
    recursion_limit: int | None = None,
    checkpointer: Any = None,
) -> OpenCodeResult:
    """Async form of :func:`open_code`."""
    _validate_params(
        unit=unit,
        sweep_every=sweep_every,
        top_k_codes=top_k_codes,
        max_codes_per_passage=max_codes_per_passage,
        validation_retries=validation_retries,
    )
    parsed_context = parse_context(context)
    documents = load_corpus(corpus)
    coder = resolve_chat_model(
        model,
        api_key=api_key,
        temperature=temperature,
        llm_seed=llm_seed,
        model_kwargs=model_kwargs,
    )
    sweeper_source = model if sweep_model is None else sweep_model
    if sweep_model is None:
        sweeper = coder
    else:
        sweeper = resolve_chat_model(
            sweep_model,
            api_key=api_key,
            temperature=temperature if sweep_temperature is None else sweep_temperature,
            llm_seed=llm_seed,
            model_kwargs=sweep_model_kwargs,
        )
    embed_key = api_key if embeddings_api_key is None else embeddings_api_key
    embedding_model = resolve_embeddings(embeddings, api_key=embed_key, embeddings_kwargs=embeddings_kwargs)
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
        candidate_limit=candidate_limit,
        model_name=model_name,
        sweep_model_name=sweep_name,
        prompt_log_path=str(prompt_log_path) if prompt_log_path else None,
        output_dir=str(output_dir) if output_dir else None,
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
    """Run :func:`open_code` once per seed and compare the codebooks.

    The report describes label overlap. It does not pick a winning seed.
    Checkpoint thread ids and output directories are suffixed with the seed
    so the runs do not overwrite one another.
    """
    if len(list(seeds)) < 2:
        raise ValueError("multi_seed_stability needs at least two seeds.")
    if "seed" in open_code_kwargs:
        raise ValueError("Pass seeds to multi_seed_stability, not seed=.")
    runs = []
    base_thread = open_code_kwargs.get("thread_id")
    base_output = open_code_kwargs.get("output_dir")
    for seed in seeds:
        kwargs = dict(open_code_kwargs)
        if base_thread:
            kwargs["thread_id"] = f"{base_thread}-seed-{seed}"
        elif kwargs.get("checkpoint_path") or kwargs.get("checkpointer") is not None:
            kwargs["thread_id"] = f"seed-{seed}"
        if base_output:
            kwargs["output_dir"] = str(Path(base_output) / f"seed-{seed}")
        runs.append(open_code(corpus, context, seed=int(seed), **kwargs))
    report = compare_runs(runs)
    report.runs = runs
    return report


def _validate_params(*, unit: str, sweep_every: int, top_k_codes: int, max_codes_per_passage: int, validation_retries: int) -> None:
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {', '.join(UNITS)}. Got {unit!r}.")
    if sweep_every < 1:
        raise ValueError("sweep_every must be at least 1.")
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
            "long_passage_chars": kwargs["long_passage_chars"],
            "max_codes_per_passage": kwargs["max_codes_per_passage"],
            "neighbor_chars": kwargs["neighbor_chars"],
            "validation_retries": kwargs["validation_retries"],
            "require_gerund": kwargs["require_gerund"],
            "retriever": kwargs["retriever"],
            "parallel_documents": kwargs["parallel_documents"],
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
