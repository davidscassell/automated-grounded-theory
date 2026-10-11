"""Focused coding: group open codes into themes, then count without a model.

The LangGraph here only proposes and checks assignments. Prevalence, metadata
breakdowns, representative quotes, and uncoded passages are computed from the
open-coding result afterwards. Quotes are kept only when they occur in the passage.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from automated_grounded_theory.async_support import (
    CallLimiter,
    is_transient,
    parse_structured,
    run_sync,
    structured_runnable,
)
from automated_grounded_theory.backends import (
    describe_model,
    prompt_cache_supported,
    resolve_chat_model,
    resolve_embeddings,
)
from automated_grounded_theory.context import ResearchContext, parse_context
from automated_grounded_theory.export import _write_csv, _write_json, taguette_tag
from automated_grounded_theory.prompts import build_theme_messages, serialize_messages
from automated_grounded_theory.results import OpenCodeResult
from automated_grounded_theory.retrieval import aembed_texts, cosine, jaccard
from automated_grounded_theory.schemas import ThemeCoding, ThemeMergePlan
from automated_grounded_theory.validation import code_key, display_label, locate_quote
from automated_grounded_theory.version import __version__

# Metadata keys treated as a person id when the caller does not pass respondent_key.
_RESPONDENT_FIELDS = ("respondent_id", "respondent", "participant_id", "participant")


class ThemeError(RuntimeError):
    """Focused coding did not assign every open code to one existing theme."""


class ThemeGraphState(TypedDict, total=False):
    attempt: int
    max_attempts: int
    feedback: str
    proposal: dict | None
    proposals: dict
    accepted: bool
    assigned: list
    missing: list
    unknown: list
    duplicates: list
    prompt_log: list
    timed_out: bool
    chunk_warnings: list
    themes: list
    tensions: list
    negative_cases: list
    uncoded: list
    totals: dict


@dataclass
class ThemeResult:
    """Themes built from one open-coding result.

    Prevalence fields are computed from segments and passages. The model does
    not supply the counts or the representative quotes.
    """

    themes: list[dict]
    tensions: list[dict]
    negative_cases: list[dict]
    uncoded: list[dict]
    totals: dict
    prompt_log: list[dict]
    metadata: dict
    warnings: list[str] = field(default_factory=list)
    export_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "themes": self.themes,
            "tensions": self.tensions,
            "negative_cases": self.negative_cases,
            "uncoded": self.uncoded,
            "totals": self.totals,
            "prompt_log": self.prompt_log,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "export_paths": self.export_paths,
        }


def order_codes(codes: Sequence[Mapping[str, Any]], seed: int | None) -> list[dict]:
    """Frequency order, or a seeded shuffle of that same list.

    The shuffle only changes what the model sees. Quote selection does not use it.
    """
    ordered = [dict(entry) for entry in codes if code_key(str(entry.get("code") or ""))]
    ordered.sort(key=lambda entry: (-int(entry.get("frequency") or 0), code_key(str(entry.get("code") or ""))))
    if seed is not None:
        random.Random(seed).shuffle(ordered)
    return ordered


def respondent_of(
    metadata: Mapping[str, Any] | None,
    doc_id: Any,
    respondent_key: str | None,
) -> str:
    """Person id for prevalence.

    An explicit ``respondent_key`` wins. Otherwise the first populated field
    among respondent / participant ids is used. With neither, the document id
    is the respondent, so respondents and documents match.
    """
    meta = metadata or {}
    if respondent_key:
        value = meta.get(respondent_key)
        if value not in (None, ""):
            return str(value)
        return str(doc_id or "")
    for key in _RESPONDENT_FIELDS:
        value = meta.get(key)
        if value not in (None, ""):
            return str(value)
    return str(doc_id or "")


def theme_tag_path(theme_name: str, code_label: str) -> str:
    """Taguette hierarchy: ``Theme\\Code``.

    Taguette's tag path uses a backslash between parent and child
    (see ``interesting\\places`` in its tests and ``.qdc`` export). Backslashes
    inside either label are replaced so they do not add another level.
    """
    from automated_grounded_theory.export import TAGUETTE_SEPARATOR

    return f"{taguette_tag(theme_name)}{TAGUETTE_SEPARATOR}{taguette_tag(code_label)}"


def assign_codes(
    proposal: Mapping[str, Any],
    codebook: Sequence[Mapping[str, Any]],
) -> tuple[list[dict], dict[str, list[str]]]:
    """Keep the first assignment of each real code. Drop unknown names.

    Returns the themes that still have codes, plus missing, unknown, and
    duplicate labels. Missing is empty only when every open code was named.
    """
    by_key = {}
    for entry in codebook:
        key = code_key(str(entry.get("code") or ""))
        if key and key not in by_key:
            by_key[key] = entry
    seen: dict[str, str] = {}
    unknown: list[str] = []
    duplicates: list[str] = []
    raw_themes: list[dict] = []
    for raw in proposal.get("themes") or []:
        name = display_label(str(raw.get("name") or ""))
        codes: list[str] = []
        for label in raw.get("codes") or []:
            key = code_key(str(label))
            if not key or key not in by_key:
                if str(label).strip():
                    unknown.append(str(label).strip())
                continue
            if key in seen:
                duplicates.append(by_key[key]["code"])
                continue
            seen[key] = name or key
            codes.append(str(by_key[key]["code"]))
        raw_themes.append(
            {
                "name": name,
                "definition": str(raw.get("definition") or "").strip(),
                "properties": _clean_list(raw.get("properties")),
                "codes": codes,
            }
        )
    merged: list[dict] = []
    index: dict[str, dict] = {}
    for theme in raw_themes:
        key = code_key(theme["name"])
        if not key:
            for label in theme["codes"]:
                seen.pop(code_key(label), None)
            continue
        if key in index:
            index[key]["codes"].extend(theme["codes"])
            if not index[key]["definition"]:
                index[key]["definition"] = theme["definition"]
            index[key]["properties"] = _clean_list(index[key]["properties"] + theme["properties"])
        else:
            index[key] = theme
            merged.append(theme)
    for theme in merged:
        theme["codes"] = _unique_labels(theme["codes"])
    merged = [theme for theme in merged if theme["codes"]]
    missing = [str(entry["code"]) for key, entry in by_key.items() if key not in seen]
    return merged, {"missing": missing, "unknown": unknown, "duplicates": duplicates}


def assignment_feedback(report: Mapping[str, list[str]]) -> str:
    lines = [
        "VALIDATION FAILED",
        "Assign every open code to exactly one theme. Copy each CODE_LABEL exactly. Do not invent codes.",
    ]
    if report.get("missing"):
        lines.append("These codes were not assigned:")
        lines.extend(f"- {label}" for label in report["missing"])
    if report.get("unknown"):
        lines.append("These names are not in the codebook and were dropped:")
        lines.extend(f"- {label}" for label in report["unknown"])
    if report.get("duplicates"):
        lines.append("These codes were assigned more than once. Only the first assignment was kept:")
        lines.extend(f"- {label}" for label in report["duplicates"])
    return "\n".join(lines)


def compute_themes(
    assigned: Sequence[Mapping[str, Any]],
    proposal: Mapping[str, Any],
    *,
    codebook: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    respondent_key: str | None,
    quotes_per_theme: int,
    min_quote_chars: int,
    with_quotes: bool = True,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict, list[str]]:
    """Prevalence, breakdowns, quotes, tensions, and uncoded passages.

    No model is called. A passage counts when one of its segments carries a
    member code. A document counts when one of its passages does.
    """
    warnings: list[str] = []
    passages_by_id = {str(item.get("passage_id")): item for item in passages if item.get("passage_id")}
    definitions = {
        code_key(str(entry.get("code") or "")): str(entry.get("definition") or "")
        for entry in codebook
    }
    coded_ids = {str(segment.get("passage_id")) for segment in segments if segment.get("passage_id")}
    totals = _totals(passages, respondent_key)
    themes: list[dict] = []
    assigned_keys: set[str] = set()
    for theme in assigned:
        built = _one_theme(
            theme,
            segments=segments,
            passages_by_id=passages_by_id,
            definitions=definitions,
            totals=totals,
            respondent_key=respondent_key,
            quotes_per_theme=quotes_per_theme,
            min_quote_chars=min_quote_chars,
            with_quotes=with_quotes,
        )
        assigned_keys.update(code_key(label) for label in built["codes"])
        themes.append(built)
    themes.sort(key=lambda item: (-item["n_documents"], -item["n_passages"], item["name"].casefold()))
    orphan = sorted(
        {
            str(segment.get("code"))
            for segment in segments
            if segment.get("code") and code_key(str(segment.get("code"))) not in assigned_keys
        }
    )
    if orphan:
        warnings.append(
            "Segments whose codes are not in the codebook were left out of every theme: "
            + "; ".join(orphan)
        )
    tensions = _tensions(proposal.get("tensions") or [], themes, passages_by_id, warnings)
    negatives = _negatives(
        proposal.get("negative_cases") or [],
        themes,
        passages_by_id,
        assigned_keys,
        warnings,
    )
    uncoded = _uncoded(passages, coded_ids, respondent_key)
    return themes, tensions, negatives, uncoded, totals, warnings


def themes_markdown(result: ThemeResult) -> str:
    totals = result.totals
    lines = [
        "# Themes",
        "",
        "Focused coding grouped the open codes into themes. Each code is assigned to exactly one theme. "
        "Counts are computed from the coded segments, not by the model. A passage counts toward a theme "
        "when any of its segments carries a member code. A document counts when any of its passages does. "
        "Respondents come from metadata when a respondent field is present, and otherwise each document is "
        "its own respondent. Quotes are coded segments checked as verbatim substrings of the passage.",
        "",
    ]
    if not result.themes:
        lines.append("_No themes. The codebook had no open codes to group._")
        lines.append("")
    for theme in result.themes:
        lines.extend(_theme_markdown(theme, result.metadata.get("min_quote_chars", 15)))
    lines.append("## Tensions and negative cases")
    lines.append("")
    if not result.tensions and not result.negative_cases:
        lines.append("_None proposed._")
        lines.append("")
    for item in result.tensions:
        joined = " / ".join(item.get("themes") or []) or "Themes"
        lines.append(f"- **{joined}.** {item.get('note', '').strip()}".rstrip())
        if item.get("quote"):
            where = item.get("passage_id") or ""
            suffix = f" ({where})" if where else ""
            lines.append(f'  - "{item["quote"]}"{suffix}')
    for item in result.negative_cases:
        code = f", code: {item['code']}" if item.get("code") else ""
        lines.append(f"- **{item.get('theme', 'Theme')}** (negative case{code}). {item.get('note', '').strip()}".rstrip())
        if item.get("quote"):
            where = item.get("passage_id") or ""
            suffix = f" ({where})" if where else ""
            lines.append(f'  - "{item["quote"]}"{suffix}')
    if result.tensions or result.negative_cases:
        lines.append("")
    n_passages = int(totals.get("n_passages") or 0)
    lines.append(f"## Uncoded passages ({len(result.uncoded)} of {n_passages})")
    lines.append("")
    lines.append("Passages with no segment, including ones skipped before coding (empty or non-answers).")
    lines.append("")
    if not result.uncoded:
        lines.append("_Every passage has at least one segment._")
        lines.append("")
    for passage in result.uncoded:
        text = _clip(str(passage.get("text") or "").replace("\n", " "), 500)
        lines.append(f'- {passage.get("passage_id")} ({passage.get("doc_id")}): "{text}"')
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "Each open code sits in one theme, so a passage whose codes fall in two themes counts in both. "
        "Passage and document percentages can sum to more than 100. Respondent percentages overlap when "
        "one person contributed to more than one theme."
    )
    lines.append("")
    lines.append(
        f"Up to {result.metadata.get('quotes_per_theme', 3)} quotes are kept per theme, "
        "chosen by code frequency and then by length. "
        f"A quote shorter than {result.metadata.get('min_quote_chars', 15)} characters is skipped, "
        "and only one quote is kept per respondent."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_themes(result: ThemeResult, output_dir: str | Path, *, chart: bool = True) -> dict[str, str]:
    """Write themes.md, themes.csv, themes.json, and a Taguette codebook.

    The prevalence chart is written when matplotlib imports. A missing
    matplotlib is a warning, not a failed export.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def place(name: str) -> Path:
        path = directory / name
        paths[name] = str(path)
        return path

    payload = {
        "themes": result.themes,
        "tensions": result.tensions,
        "negative_cases": result.negative_cases,
        "uncoded": result.uncoded,
        "totals": result.totals,
        "warnings": result.warnings,
        "metadata": result.metadata,
    }
    _write_json(place("themes.json"), payload)
    place("themes.md").write_text(themes_markdown(result), encoding="utf-8")
    _write_csv(
        place("themes.csv"),
        [
            "theme",
            "definition",
            "codes",
            "n_passages",
            "pct_passages",
            "n_documents",
            "pct_documents",
            "n_comments",
            "pct_comments",
            "n_respondents",
            "pct_respondents",
        ],
        [
            [
                theme["name"],
                theme["definition"],
                "; ".join(theme["codes"]),
                theme["n_passages"],
                theme["pct_passages"],
                theme["n_documents"],
                theme["pct_documents"],
                theme["n_comments"],
                theme["pct_comments"],
                theme["n_respondents"],
                theme["pct_respondents"],
            ]
            for theme in result.themes
        ],
    )
    _write_taguette(place, result)
    _write_json(place("prompt_log.json"), result.prompt_log)
    if chart and result.themes:
        try:
            plt = _load_pyplot()
        except ImportError:
            result.warnings.append(
                "matplotlib is not installed, so theme_prevalence.png was not written. "
                "Install the plots extra to draw it."
            )
        else:
            _write_chart(place("theme_prevalence.png"), result.themes, plt)
    result.export_paths.update(paths)
    return paths


def build_themes(
    open_code_result: OpenCodeResult | str | Path,
    context: ResearchContext | dict | str,
    *,
    model: BaseChatModel | str,
    n_themes: int | None = None,
    seed: int | None = None,
    respondent_key: str | None = None,
    output_dir: str | Path | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    llm_seed: int | None = None,
    model_kwargs: dict | None = None,
    prompt_cache: bool | None = None,
    chart: bool = True,
    quotes_per_theme: int = 3,
    min_quote_chars: int = 15,
    validation_retries: int = 1,
    prompt_log_path: str | Path | None = None,
    concurrency: int = 2,
    theme_chunk: int = 40,
    max_retries: int = 4,
    retry_backoff: float = 0.5,
    retry_max_backoff: float = 8.0,
    max_themes: int = 12,
    request_timeout: float = 120.0,
    embeddings: Any = None,
) -> ThemeResult:
    """Group the codes from ``open_code`` into themes.

    ``model`` is a ``provider:model-name`` string or a ``BaseChatModel``.
    ``n_themes`` asks for that many themes. Leave it unset and the model is asked
    for about 6 to ``max_themes`` (default 12), then near-duplicates are folded.
    ``open_code_result`` may be an :class:`OpenCodeResult` or a directory written
    by ``open_code`` (see :func:`load_run`). ``seed`` shuffles the codebook order
    in the prompt. It does not pick quotes.

    The codebook is split into chunks of ``theme_chunk``. Those chunks are
    proposed concurrently (``concurrency`` caps in-flight calls), then reconciled
    in chunk order so every open code lands in one theme. Quote checks for each
    theme also run concurrently. Counts and breakdowns are computed from the
    segments. This wrapper works inside a running event loop.
    """
    return run_sync(
        abuild_themes(
            open_code_result,
            context,
            model=model,
            n_themes=n_themes,
            seed=seed,
            respondent_key=respondent_key,
            output_dir=output_dir,
            api_key=api_key,
            temperature=temperature,
            llm_seed=llm_seed,
            model_kwargs=model_kwargs,
            prompt_cache=prompt_cache,
            chart=chart,
            quotes_per_theme=quotes_per_theme,
            min_quote_chars=min_quote_chars,
            validation_retries=validation_retries,
            prompt_log_path=prompt_log_path,
            concurrency=concurrency,
            theme_chunk=theme_chunk,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            max_themes=max_themes,
            request_timeout=request_timeout,
            embeddings=embeddings,
        )
    )


async def abuild_themes(
    open_code_result: OpenCodeResult | str | Path,
    context: ResearchContext | dict | str,
    *,
    model: BaseChatModel | str,
    n_themes: int | None = None,
    seed: int | None = None,
    respondent_key: str | None = None,
    output_dir: str | Path | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    llm_seed: int | None = None,
    model_kwargs: dict | None = None,
    prompt_cache: bool | None = None,
    chart: bool = True,
    quotes_per_theme: int = 3,
    min_quote_chars: int = 15,
    validation_retries: int = 1,
    prompt_log_path: str | Path | None = None,
    concurrency: int = 2,
    theme_chunk: int = 40,
    max_retries: int = 4,
    retry_backoff: float = 0.5,
    retry_max_backoff: float = 8.0,
    max_themes: int = 12,
    request_timeout: float = 120.0,
    embeddings: Any = None,
    limiter: CallLimiter | None = None,
) -> ThemeResult:
    """Async form of :func:`build_themes`."""
    _validate_theme_params(
        n_themes=n_themes,
        quotes_per_theme=quotes_per_theme,
        min_quote_chars=min_quote_chars,
        validation_retries=validation_retries,
        concurrency=concurrency,
        theme_chunk=theme_chunk,
        max_retries=max_retries,
        max_themes=max_themes,
    )
    research = parse_context(context)
    source = _coerce_result(open_code_result)
    codes = order_codes(source.codebook, seed)
    warnings: list[str] = []
    if getattr(source, "interrupted", False):
        warnings.append("The open-coding run was interrupted. Themes cover only the codes committed so far.")

    if not codes:
        themed = _empty_result(
            source,
            research,
            model_name=model if isinstance(model, str) else describe_model(model),
            n_themes=n_themes,
            seed=seed,
            respondent_key=respondent_key,
            quotes_per_theme=quotes_per_theme,
            min_quote_chars=min_quote_chars,
            validation_retries=validation_retries,
            api_key_provided=api_key is not None,
            prompt_cache=False,
            warnings=warnings + ["No open codes to group."],
        )
        return _finish(themed, output_dir, chart, prompt_log_path)

    coder = resolve_chat_model(
        model,
        api_key=api_key,
        temperature=temperature,
        llm_seed=llm_seed,
        model_kwargs=model_kwargs,
        timeout=request_timeout,
    )
    model_name = describe_model(coder if not isinstance(model, str) else model)
    cache = prompt_cache_supported(coder, model_name) if prompt_cache is None else prompt_cache
    if limiter is None:
        limiter = CallLimiter(
            concurrency,
            max_retries=max_retries,
            backoff=retry_backoff,
            max_backoff=retry_max_backoff,
            request_timeout=request_timeout,
        )
    chunks = _theme_chunks(codes, theme_chunk)
    graph = _compile_theme_graph(
        coder,
        codes,
        chunks,
        research,
        n_themes=n_themes if len(chunks) == 1 else None,
        max_themes=max_themes,
        prompt_cache=cache,
        model_name=model_name,
        limiter=limiter,
    )
    final = await graph.ainvoke(
        {
            "attempt": 0,
            "max_attempts": validation_retries + 1,
            "feedback": "",
            "proposals": {},
            "accepted": False,
            "assigned": [],
            "missing": [],
            "unknown": [],
            "duplicates": [],
            "prompt_log": [],
            "timed_out": False,
            "chunk_warnings": [],
        }
    )
    warnings.extend(final.get("chunk_warnings") or [])
    proposal = dict(final.get("proposal") or {})
    assigned = list(final.get("assigned") or [])
    if not final.get("accepted"):
        missing = list(final.get("missing") or [])
        if final.get("timed_out"):
            warnings.append(
                "Some theme chunks timed out. Codes the model did not group are in 'Ungrouped incidents'."
            )
            themes_raw = list(proposal.get("themes") or [])
            if missing:
                themes_raw.append(
                    {
                        "name": "Ungrouped incidents",
                        "definition": "Open codes the model did not group before the deadline.",
                        "properties": [],
                        "codes": missing,
                    }
                )
            proposal["themes"] = themes_raw
            assigned, report = assign_codes(proposal, codes)
            if report["missing"]:
                detail = "; ".join(report["missing"])
                raise ThemeError(f"Focused coding left open codes unassigned: {detail}.")
        else:
            detail = "; ".join(missing) if missing else "the model returned no usable themes"
            retry = ""
            if validation_retries:
                retry = " The model was shown the missing labels and still did not cover them."
            raise ThemeError(f"Focused coding left open codes unassigned: {detail}.{retry}")
    if final.get("unknown"):
        warnings.append("Dropped names that are not open codes: " + "; ".join(final["unknown"]))
    if final.get("duplicates"):
        warnings.append(
            "Some codes were named in more than one theme. The first assignment was kept: "
            + "; ".join(final["duplicates"])
        )
    prompt_log = list(final.get("prompt_log") or [])
    assigned, proposal = await _fold_near_duplicates(
        assigned,
        proposal,
        coder=coder,
        context=research,
        limiter=limiter,
        max_themes=max_themes,
        cap=n_themes is None,
        embeddings=embeddings,
        api_key=api_key,
        request_timeout=request_timeout,
        prompt_cache=cache,
        model_name=model_name,
        warnings=warnings,
        prompt_log=prompt_log,
    )
    themes, tensions, negatives, uncoded, totals, extra = compute_themes(
        assigned,
        proposal,
        codebook=codes,
        segments=source.segments,
        passages=source.passages,
        respondent_key=respondent_key,
        quotes_per_theme=quotes_per_theme,
        min_quote_chars=min_quote_chars,
        with_quotes=False,
    )
    await _fill_quotes(
        themes,
        segments=source.segments,
        passages=source.passages,
        respondent_key=respondent_key,
        quotes_per_theme=quotes_per_theme,
        min_quote_chars=min_quote_chars,
        limiter=limiter,
    )
    warnings.extend(extra)
    if n_themes is not None and len(themes) != n_themes:
        warnings.append(
            f"Requested {n_themes} themes; {len(themes)} non-empty themes were returned. "
            "Every open code is still assigned."
        )
    themed = ThemeResult(
        themes=themes,
        tensions=tensions,
        negative_cases=negatives,
        uncoded=uncoded,
        totals=totals,
        prompt_log=prompt_log,
        metadata=_metadata(
            model_name=model_name,
            n_themes=n_themes,
            max_themes=max_themes,
            seed=seed,
            respondent_key=respondent_key,
            quotes_per_theme=quotes_per_theme,
            min_quote_chars=min_quote_chars,
            validation_retries=validation_retries,
            api_key_provided=api_key is not None,
            prompt_cache=cache,
            calls=len(prompt_log),
            question=research.research_question,
            concurrency=concurrency,
            theme_chunk=theme_chunk,
        ),
        warnings=warnings,
    )
    return _finish(themed, output_dir, chart, prompt_log_path)


def _theme_chunks(codes: Sequence[Mapping[str, Any]], size: int) -> list[list[dict]]:
    width = max(1, size)
    rows = [dict(entry) for entry in codes]
    if len(rows) <= width:
        return [rows]
    return [rows[start : start + width] for start in range(0, len(rows), width)]


def reconcile_theme_proposals(parts: Sequence[tuple[int, Mapping[str, Any]]]) -> dict:
    """Join chunk proposals in chunk-index order. Earlier chunks keep a code."""
    themes: list = []
    tensions: list = []
    negatives: list = []
    for _index, payload in sorted(parts, key=lambda item: item[0]):
        themes.extend(payload.get("themes") or [])
        tensions.extend(payload.get("tensions") or [])
        negatives.extend(payload.get("negative_cases") or [])
    return {"themes": themes, "tensions": tensions, "negative_cases": negatives}


def collapse_themes(
    themes: Sequence[Mapping[str, Any]],
    *,
    max_themes: int,
    vectors: Sequence[Sequence[float]] | None = None,
    near_threshold: float = 0.5,
    cap: bool = True,
) -> tuple[list[dict], dict[str, str], bool]:
    """Fold near-duplicate themes, then closest pairs until ``max_themes``.

    Similarity is cosine when ``vectors`` is aligned with ``themes``, otherwise
    token overlap of the name and definition. The survivor is the theme with
    more codes, then the lexicographically smaller name. ``cap`` is false when
    the caller asked for an exact ``n_themes``: near-duplicates still fold, and
    the list is not forced down to ``max_themes``.
    """
    rows = [dict(theme) for theme in themes if str(theme.get("name") or "").strip()]
    count = len(rows)
    if count == 0:
        return [], {}, False
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    def score(left: int, right: int) -> float:
        if vectors is not None and left < len(vectors) and right < len(vectors):
            return cosine(vectors[left], vectors[right])
        return jaccard(_theme_text(rows[left]), _theme_text(rows[right]))

    pairs = [(score(left, right), left, right) for left in range(count) for right in range(left + 1, count)]
    for value, left, right in sorted(pairs, key=lambda item: (-item[0], item[1], item[2])):
        if value >= near_threshold:
            union(left, right)

    def cluster_count() -> int:
        return len({find(index) for index in range(count)})

    if cap and max_themes >= 1:
        while cluster_count() > max_themes:
            best: tuple[tuple[float, int, int], int, int] | None = None
            roots: dict[int, list[int]] = {}
            for index in range(count):
                roots.setdefault(find(index), []).append(index)
            root_ids = sorted(roots)
            for offset, root_left in enumerate(root_ids):
                for root_right in root_ids[offset + 1 :]:
                    for left in roots[root_left]:
                        for right in roots[root_right]:
                            value = score(left, right)
                            ordered = (left, right) if left < right else (right, left)
                            key = (-value, ordered[0], ordered[1])
                            if best is None or key < best[0]:
                                best = (key, left, right)
            if best is None:
                break
            union(best[1], best[2])

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    merged: list[dict] = []
    mapping: dict[str, str] = {}
    changed = False
    for root in sorted(groups, key=lambda item: min(groups[item])):
        members = [rows[index] for index in groups[root]]
        if len(members) > 1:
            changed = True
        bundled = _bundle_themes(members)
        merged.append(bundled)
        for theme in members:
            mapping[str(theme.get("name") or "")] = bundled["name"]
    return merged, mapping, changed


def apply_theme_merge_plan(
    themes: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    max_themes: int,
    cap: bool,
) -> tuple[list[dict], dict[str, str]] | None:
    """Apply a merge plan that names every input theme exactly once.

    Returns None when a name is missing, repeated, unknown, or the plan has
    more groups than ``max_themes`` while ``cap`` is set.
    """
    groups = list(plan.get("themes") or [])
    if cap and len(groups) > max_themes:
        return None
    by_name = {str(theme.get("name") or ""): theme for theme in themes}
    seen: list[str] = []
    rebuilt: list[dict] = []
    rename: dict[str, str] = {}
    for group in groups:
        members = []
        for label in group.get("members") or []:
            name = str(label)
            if name not in by_name:
                return None
            members.append(by_name[name])
            seen.append(name)
        if not members:
            return None
        fallback = _bundle_themes(members)
        name = display_label(str(group.get("name") or "")) or fallback["name"]
        definition = str(group.get("definition") or "").strip() or fallback["definition"]
        bundled = _bundle_themes(members, name=name, definition=definition)
        rebuilt.append(bundled)
        for member in members:
            rename[str(member.get("name") or "")] = bundled["name"]
    if sorted(seen) != sorted(by_name) or len(seen) != len(set(seen)):
        return None
    return rebuilt, rename


async def _fold_near_duplicates(
    assigned: list[dict],
    proposal: dict,
    *,
    coder: BaseChatModel,
    context: ResearchContext,
    limiter: CallLimiter,
    max_themes: int,
    cap: bool,
    embeddings: Any,
    api_key: str | None,
    request_timeout: float,
    prompt_cache: bool,
    model_name: str,
    warnings: list[str],
    prompt_log: list,
) -> tuple[list[dict], dict]:
    """Collapse near-duplicate themes, then ask the model to rename that grouping.

    The model is called only when membership changed. A timeout or an invalid
    plan keeps the deterministic names.
    """
    if not assigned:
        return assigned, proposal
    vectors = await _theme_vectors(
        assigned,
        embeddings=embeddings,
        api_key=api_key,
        request_timeout=request_timeout,
        limiter=limiter,
        warnings=warnings,
    )
    collapsed, mapping, changed = collapse_themes(
        assigned,
        max_themes=max_themes,
        vectors=vectors,
        cap=cap,
    )
    if not changed:
        return assigned, proposal
    current = collapsed
    try:
        payload, usage, messages = await _propose_theme_merge(
            coder,
            current,
            context=context,
            limiter=limiter,
            max_themes=max_themes,
            cap=cap,
            prompt_cache=prompt_cache,
        )
    except Exception as exc:
        if is_transient(exc):
            warnings.append("Theme merge pass timed out. Near-duplicate themes kept the wording grouping.")
        else:
            warnings.append(
                f"Theme merge pass failed ({exc}). Near-duplicate themes kept the wording grouping."
            )
    else:
        prompt_log.append(
            {
                "kind": "theme_merge",
                "model": model_name,
                "messages": serialize_messages(messages),
                "output": payload,
                "usage": usage,
            }
        )
        applied = apply_theme_merge_plan(current, payload, max_themes=max_themes, cap=cap)
        if applied is None:
            warnings.append(
                "Theme merge pass did not cover every theme once. Near-duplicate themes kept the wording grouping."
            )
        else:
            current, rename = applied
            mapping = {old: rename.get(new, new) for old, new in mapping.items()}
    return current, _remap_theme_names(proposal, mapping)


async def _theme_vectors(
    themes: Sequence[Mapping[str, Any]],
    *,
    embeddings: Any,
    api_key: str | None,
    request_timeout: float,
    limiter: CallLimiter,
    warnings: list[str],
) -> list[list[float]] | None:
    if embeddings is None:
        return None
    embedder = resolve_embeddings(embeddings, api_key=api_key, timeout=request_timeout)
    if embedder is None:
        return None
    texts = [_theme_text(theme) for theme in themes]
    try:
        return await aembed_texts(embedder, texts, batch_size=64, limiter=limiter)
    except Exception as exc:
        if not is_transient(exc):
            raise
        warnings.append(f"Theme embeddings failed ({exc}). Near-duplicate themes were compared by wording.")
        return None


async def _propose_theme_merge(
    model: BaseChatModel,
    themes: Sequence[Mapping[str, Any]],
    *,
    context: ResearchContext,
    limiter: CallLimiter,
    max_themes: int,
    cap: bool,
    prompt_cache: bool,
) -> tuple[dict, dict, list]:
    runnable = structured_runnable(model, ThemeMergePlan)
    lines = [
        "These themes overlap. Group them so each incident is named once.",
        "Every THEME_NAME below must appear in exactly one group's members list. Copy the name exactly.",
        "Do not drop a theme name and do not invent open codes.",
    ]
    if cap:
        lines.append(f"Use at most {max_themes} groups.")
    lines.append("")
    for theme in themes:
        lines.append(f"THEME_NAME: {theme.get('name') or ''}")
        definition = str(theme.get("definition") or "").strip()
        if definition:
            lines.append(f"DEFINITION: {definition}")
        codes = "; ".join(str(label) for label in theme.get("codes") or [])
        if codes:
            lines.append(f"CODES: {codes}")
        lines.append("")
    ceiling = f" At most {max_themes} groups." if cap else ""
    from automated_grounded_theory.prompts import _system_message, render_context

    system = (
        "You are reconciling near-duplicate themes after focused coding. "
        "Members are theme names, not open codes. "
        f"Every THEME_NAME appears in exactly one group.{ceiling}\n\n"
        + render_context(context)
    )
    messages = [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content="\n".join(lines))]
    result = await limiter.ainvoke(runnable, messages)
    payload, usage = parse_structured(result, ThemeMergePlan)
    return payload, usage, messages


def _remap_theme_names(proposal: Mapping[str, Any], mapping: Mapping[str, str]) -> dict:
    def renamed(label: Any) -> str:
        text = str(label or "")
        return str(mapping.get(text, text))

    tensions = []
    for item in proposal.get("tensions") or []:
        copy = dict(item)
        copy["themes"] = [renamed(label) for label in item.get("themes") or []]
        tensions.append(copy)
    negatives = []
    for item in proposal.get("negative_cases") or []:
        copy = dict(item)
        if copy.get("theme"):
            copy["theme"] = renamed(copy["theme"])
        negatives.append(copy)
    return {**dict(proposal), "tensions": tensions, "negative_cases": negatives}


def _theme_text(theme: Mapping[str, Any]) -> str:
    return f"{theme.get('name') or ''} {theme.get('definition') or ''}".strip()


def _bundle_themes(
    members: Sequence[Mapping[str, Any]],
    *,
    name: str | None = None,
    definition: str | None = None,
) -> dict:
    survivor = min(
        members,
        key=lambda theme: (-len(list(theme.get("codes") or [])), str(theme.get("name") or "").casefold()),
    )
    codes: list[str] = []
    seen: set[str] = set()
    for theme in members:
        for label in theme.get("codes") or []:
            key = code_key(str(label))
            if key and key not in seen:
                seen.add(key)
                codes.append(str(label))
    codes.sort(key=code_key)
    properties = _clean_list([prop for theme in members for prop in (theme.get("properties") or [])])
    kept_definition = definition if definition is not None else str(survivor.get("definition") or "").strip()
    if not kept_definition:
        kept_definition = max((str(theme.get("definition") or "").strip() for theme in members), key=len, default="")
    return {
        "name": name or str(survivor.get("name") or ""),
        "definition": kept_definition,
        "properties": properties,
        "codes": codes,
    }


def _compile_theme_graph(
    model: BaseChatModel,
    codes: Sequence[Mapping[str, Any]],
    chunks: Sequence[Sequence[Mapping[str, Any]]],
    context: ResearchContext,
    *,
    n_themes: int | None,
    max_themes: int,
    prompt_cache: bool,
    model_name: str,
    limiter: CallLimiter,
):
    runnable = structured_runnable(model, ThemeCoding)

    async def propose(state: ThemeGraphState) -> dict:
        proposals = dict(state.get("proposals") or {})
        missing = list(state.get("missing") or [])
        indexes = _chunks_to_retry(chunks, missing, first=not proposals)
        timed_out = bool(state.get("timed_out"))
        chunk_warnings: list[str] = []

        async def one(index: int):
            messages = build_theme_messages(
                context=context,
                codes=chunks[index],
                n_themes=n_themes,
                feedback=str(state.get("feedback") or ""),
                prompt_cache=prompt_cache,
                chunk_index=index + 1,
                chunk_count=len(chunks),
                max_themes=max_themes,
            )
            try:
                result = await limiter.ainvoke(runnable, messages)
            except Exception as exc:
                if is_transient(exc):
                    return index, messages, None, exc
                raise ThemeError(f"Focused coding failed on chunk {index + 1}: {exc}") from exc
            return index, messages, result, None

        outcomes = sorted(await asyncio.gather(*(one(index) for index in indexes)), key=lambda item: item[0])
        log = list(state.get("prompt_log") or [])
        for index, messages, result, exc in outcomes:
            if exc is not None:
                timed_out = True
                chunk_warnings.append(
                    f"Theme chunk {index + 1} of {len(chunks)} timed out or lost its connection. "
                    "Those codes were not grouped in this attempt."
                )
                proposals[index] = {"themes": [], "tensions": [], "negative_cases": []}
                log.append(
                    {
                        "kind": "themes",
                        "model": model_name,
                        "chunk_index": index + 1,
                        "messages": serialize_messages(messages),
                        "output": proposals[index],
                        "error": str(exc),
                    }
                )
                continue
            try:
                payload, usage = parse_structured(result, ThemeCoding)
            except ValueError as exc:
                raise ThemeError(f"Focused coding failed on chunk {index + 1}: {exc}") from exc
            proposals[index] = payload
            log.append(
                {
                    "kind": "themes",
                    "model": model_name,
                    "chunk_index": index + 1,
                    "messages": serialize_messages(messages),
                    "output": payload,
                    "usage": usage,
                }
            )
        log.sort(key=lambda item: int(item.get("chunk_index") or 0))
        return {
            "proposals": proposals,
            "attempt": int(state.get("attempt") or 0) + 1,
            "prompt_log": log,
            "timed_out": timed_out,
            "chunk_warnings": chunk_warnings,
        }

    def validate(state: ThemeGraphState) -> dict:
        parts = [(int(index), payload) for index, payload in (state.get("proposals") or {}).items()]
        proposal = reconcile_theme_proposals(parts)
        assigned, report = assign_codes(proposal, codes)
        accepted = not report["missing"]
        return {
            "accepted": accepted,
            "proposal": proposal,
            "assigned": assigned,
            "missing": report["missing"],
            "unknown": report["unknown"],
            "duplicates": report["duplicates"],
            "feedback": "" if accepted else assignment_feedback(report),
        }

    def route(state: ThemeGraphState) -> str:
        if state.get("accepted"):
            return "done"
        if int(state.get("attempt") or 0) >= int(state.get("max_attempts") or 1):
            return "done"
        return "propose"

    builder = StateGraph(ThemeGraphState)
    builder.add_node("propose", propose)
    builder.add_node("validate", validate)
    builder.add_edge(START, "propose")
    builder.add_edge("propose", "validate")
    builder.add_conditional_edges("validate", route, {"propose": "propose", "done": END})
    return builder.compile()


def _chunks_to_retry(chunks: Sequence[Sequence[Mapping[str, Any]]], missing: Sequence[str], *, first: bool) -> list[int]:
    if first or not missing:
        return list(range(len(chunks)))
    needed = {code_key(label) for label in missing}
    chosen = []
    for index, chunk in enumerate(chunks):
        keys = {code_key(str(entry.get("code") or "")) for entry in chunk}
        if keys & needed:
            chosen.append(index)
    return chosen or list(range(len(chunks)))


async def _fill_quotes(
    themes: list[dict],
    *,
    segments: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    respondent_key: str | None,
    quotes_per_theme: int,
    min_quote_chars: int,
    limiter: CallLimiter,
) -> None:
    """Validate representative quotes for every theme at once. Theme order is fixed."""
    if not themes:
        return
    passages_by_id = {str(item.get("passage_id")): item for item in passages if item.get("passage_id")}
    prepared = []
    for theme in themes:
        member_keys = {code_key(label) for label in theme.get("codes") or []}
        member_segments = [
            segment
            for segment in segments
            if code_key(str(segment.get("code") or "")) in member_keys
        ]
        counts = Counter(code_key(str(segment.get("code") or "")) for segment in member_segments)
        prepared.append((member_segments, counts))

    async def one(index: int) -> list[dict]:
        member_segments, counts = prepared[index]
        return await limiter.call(
            _select_quotes,
            member_segments,
            passages_by_id,
            counts,
            respondent_key=respondent_key,
            limit=quotes_per_theme,
            min_chars=min_quote_chars,
        )

    chosen = await asyncio.gather(*(one(index) for index in range(len(themes))))
    for theme, quotes in zip(themes, chosen):
        theme["quotes"] = quotes


def _one_theme(
    theme: Mapping[str, Any],
    *,
    segments: Sequence[Mapping[str, Any]],
    passages_by_id: Mapping[str, Mapping[str, Any]],
    definitions: Mapping[str, str],
    totals: Mapping[str, int],
    respondent_key: str | None,
    quotes_per_theme: int,
    min_quote_chars: int,
    with_quotes: bool = True,
) -> dict:
    member_keys = {code_key(label) for label in theme.get("codes") or []}
    member_segments = [
        segment
        for segment in segments
        if code_key(str(segment.get("code") or "")) in member_keys
    ]
    passage_ids: list[str] = []
    seen_passages: set[str] = set()
    for segment in member_segments:
        passage_id = str(segment.get("passage_id") or "")
        if passage_id in passages_by_id and passage_id not in seen_passages:
            seen_passages.add(passage_id)
            passage_ids.append(passage_id)
    doc_ids = []
    seen_docs: set[str] = set()
    respondents = []
    seen_people: set[str] = set()
    for passage_id in passage_ids:
        passage = passages_by_id[passage_id]
        doc_id = str(passage.get("doc_id") or "")
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            doc_ids.append(doc_id)
        person = respondent_of(passage.get("metadata"), doc_id, respondent_key)
        if person not in seen_people:
            seen_people.add(person)
            respondents.append(person)
    counts = Counter(code_key(str(segment.get("code") or "")) for segment in member_segments)
    documents_for_code: dict[str, set[str]] = defaultdict(set)
    for segment in member_segments:
        documents_for_code[code_key(str(segment.get("code") or ""))].add(str(segment.get("doc_id") or ""))
    labels = list(theme.get("codes") or [])
    labels.sort(key=lambda label: (-counts[code_key(label)], label.casefold()))
    code_counts = [
        {
            "code": label,
            "definition": definitions.get(code_key(label), ""),
            "segments": counts[code_key(label)],
            "documents": len(documents_for_code[code_key(label)]),
        }
        for label in labels
    ]
    n_passages = len(passage_ids)
    n_documents = len(doc_ids)
    n_respondents = len(respondents)
    return {
        "name": theme.get("name") or "",
        "definition": theme.get("definition") or "",
        "properties": list(theme.get("properties") or []),
        "codes": labels,
        "code_counts": code_counts,
        "n_passages": n_passages,
        "pct_passages": _pct(n_passages, totals["n_passages"]),
        "n_documents": n_documents,
        "pct_documents": _pct(n_documents, totals["n_documents"]),
        "n_comments": n_documents,
        "pct_comments": _pct(n_documents, totals["n_documents"]),
        "n_respondents": n_respondents,
        "pct_respondents": _pct(n_respondents, totals["n_respondents"]),
        "breakdowns": _breakdowns(passage_ids, passages_by_id),
        "quotes": (
            _select_quotes(
                member_segments,
                passages_by_id,
                counts,
                respondent_key=respondent_key,
                limit=quotes_per_theme,
                min_chars=min_quote_chars,
            )
            if with_quotes
            else []
        ),
    }


def _breakdowns(
    passage_ids: Sequence[str],
    passages_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict]]:
    """Document counts for every metadata field on passages in the theme."""
    docs_for_value: dict[tuple[str, str], set[str]] = defaultdict(set)
    for passage_id in passage_ids:
        passage = passages_by_id[passage_id]
        doc_id = str(passage.get("doc_id") or "")
        metadata = passage.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            continue
        for key, value in metadata.items():
            text = _meta_value(value)
            if not text:
                continue
            docs_for_value[(str(key), text)].add(doc_id)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for (key, value), docs in docs_for_value.items():
        grouped[key].append({"value": value, "n": len(docs)})
    return {
        key: sorted(rows, key=lambda row: (-row["n"], row["value"].casefold()))
        for key, rows in sorted(grouped.items())
    }


def _select_quotes(
    segments: Sequence[Mapping[str, Any]],
    passages_by_id: Mapping[str, Mapping[str, Any]],
    counts: Counter,
    *,
    respondent_key: str | None,
    limit: int,
    min_chars: int,
) -> list[dict]:
    """Up to ``limit`` verified quotes: frequent codes first, then longer quotes.

    One quote per respondent. The stored quote is the passage slice.
    """
    ranked = sorted(
        segments,
        key=lambda segment: (
            -counts[code_key(str(segment.get("code") or ""))],
            -len(str(segment.get("quote") or "")),
            str(segment.get("passage_id") or ""),
            str(segment.get("segment_id") or ""),
        ),
    )
    seen: set[str] = set()
    chosen: list[dict] = []
    for segment in ranked:
        passage_id = str(segment.get("passage_id") or "")
        passage = passages_by_id.get(passage_id)
        if not passage:
            continue
        span = locate_quote(str(segment.get("quote") or ""), str(passage.get("text") or ""))
        if span is None:
            continue
        text = str(passage.get("text") or "")
        verbatim = text[span[0] : span[1]]
        if len(verbatim.strip()) < min_chars:
            continue
        doc_id = str(passage.get("doc_id") or segment.get("doc_id") or "")
        person = respondent_of(passage.get("metadata"), doc_id, respondent_key)
        if person in seen:
            continue
        seen.add(person)
        chosen.append(
            {
                "quote": verbatim,
                "doc_id": doc_id,
                "passage_id": passage_id,
                "code": segment.get("code") or "",
                "respondent": person,
            }
        )
        if len(chosen) >= limit:
            break
    return chosen


def _tensions(
    raw_items: Sequence[Mapping[str, Any]],
    themes: Sequence[Mapping[str, Any]],
    passages_by_id: Mapping[str, Mapping[str, Any]],
    warnings: list[str],
) -> list[dict]:
    known = {code_key(str(theme.get("name") or "")): theme["name"] for theme in themes}
    kept: list[dict] = []
    dropped = 0
    for raw in raw_items:
        names = []
        for label in raw.get("themes") or []:
            key = code_key(str(label))
            if key in known and known[key] not in names:
                names.append(known[key])
        note = str(raw.get("note") or "").strip()
        quote, passage_id = _verified_optional_quote(raw, passages_by_id)
        if raw.get("quote") and not quote:
            dropped += 1
        if not names or not (note or quote):
            continue
        kept.append({"themes": names, "note": note, "quote": quote, "passage_id": passage_id})
    if dropped:
        warnings.append(
            f"{dropped} tension quote(s) were dropped because they were not verbatim substrings of a passage."
        )
    return kept


def _negatives(
    raw_items: Sequence[Mapping[str, Any]],
    themes: Sequence[Mapping[str, Any]],
    passages_by_id: Mapping[str, Mapping[str, Any]],
    assigned_keys: set[str],
    warnings: list[str],
) -> list[dict]:
    known = {code_key(str(theme.get("name") or "")): theme["name"] for theme in themes}
    kept: list[dict] = []
    dropped = 0
    unknown_codes = 0
    for raw in raw_items:
        theme_name = known.get(code_key(str(raw.get("theme") or "")))
        if not theme_name:
            continue
        note = str(raw.get("note") or "").strip()
        quote, passage_id = _verified_optional_quote(raw, passages_by_id)
        if raw.get("quote") and not quote:
            dropped += 1
        code = str(raw.get("code") or "").strip()
        if code and code_key(code) not in assigned_keys:
            unknown_codes += 1
            code = ""
        if not (note or quote):
            continue
        kept.append(
            {"theme": theme_name, "note": note, "code": code, "quote": quote, "passage_id": passage_id}
        )
    if dropped:
        warnings.append(
            f"{dropped} negative-case quote(s) were dropped because they were not verbatim substrings of a passage."
        )
    if unknown_codes:
        warnings.append(f"{unknown_codes} negative-case code(s) were not open codes and were cleared.")
    return kept


def _verified_optional_quote(
    raw: Mapping[str, Any],
    passages_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    quote = str(raw.get("quote") or "").strip()
    if not quote:
        return "", ""
    hinted = str(raw.get("passage_id") or "")
    order: list[str] = []
    if hinted and hinted in passages_by_id:
        order.append(hinted)
    order.extend(pid for pid in passages_by_id if pid not in order)
    for passage_id in order:
        text = str(passages_by_id[passage_id].get("text") or "")
        span = locate_quote(quote, text)
        if span is not None:
            return text[span[0] : span[1]], passage_id
    return "", ""


def _uncoded(
    passages: Sequence[Mapping[str, Any]],
    coded_ids: set[str],
    respondent_key: str | None,
) -> list[dict]:
    rows = []
    for passage in passages:
        passage_id = str(passage.get("passage_id") or "")
        if passage_id in coded_ids:
            continue
        doc_id = passage.get("doc_id")
        rows.append(
            {
                "passage_id": passage_id,
                "doc_id": doc_id,
                "text": passage.get("text") or "",
                "skipped": bool(passage.get("skipped")),
                "respondent": respondent_of(passage.get("metadata"), doc_id, respondent_key),
            }
        )
    return rows


def _totals(passages: Sequence[Mapping[str, Any]], respondent_key: str | None) -> dict[str, int]:
    documents = []
    people = []
    seen_docs: set[str] = set()
    seen_people: set[str] = set()
    for passage in passages:
        doc_id = str(passage.get("doc_id") or "")
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            documents.append(doc_id)
        person = respondent_of(passage.get("metadata"), doc_id, respondent_key)
        if person not in seen_people:
            seen_people.add(person)
            people.append(person)
    return {
        "n_passages": len(passages),
        "n_documents": len(documents),
        "n_respondents": len(people),
        "n_codes": 0,
    }


def _theme_markdown(theme: Mapping[str, Any], min_quote_chars: int) -> list[str]:
    lines = [
        f"## {theme['name']}",
        "",
        f"*Definition:* {theme.get('definition') or ''}",
        "",
    ]
    properties = theme.get("properties") or []
    if properties:
        lines.append("*Properties:* " + "; ".join(properties))
        lines.append("")
    lines.append(
        "**Prevalence:** "
        f"{_count_phrase(theme['n_passages'], 'passage')}"
        f" ({theme['pct_passages']}%), "
        f"{_count_phrase(theme['n_documents'], 'document')}"
        f" ({theme['pct_documents']}%), "
        f"{_count_phrase(theme['n_respondents'], 'respondent')}"
        f" ({theme['pct_respondents']}%)"
    )
    lines.append("")
    breakdowns = theme.get("breakdowns") or {}
    if breakdowns:
        for key, rows in breakdowns.items():
            body = ", ".join(f"{row['value']} {row['n']}" for row in rows)
            lines.append(f"- By {key}: {body}")
        lines.append("")
    counts = theme.get("code_counts") or []
    rendered = "; ".join(f"{item['code']} ({item['segments']})" for item in counts)
    lines.append(f"**Member codes ({len(counts)}), segment counts:** {rendered}")
    lines.append("")
    lines.append("**Representative quotes:**")
    quotes = theme.get("quotes") or []
    if not quotes:
        lines.append(f"_No verified quote of at least {min_quote_chars} characters._")
    for quote in quotes:
        lines.append(
            f'- "{quote["quote"]}" — {quote.get("respondent")} '
            f'({quote.get("doc_id")}; code: {quote.get("code")})'
        )
    lines.append("")
    return lines


def _write_taguette(place, result: ThemeResult) -> None:
    """Codebook CSV Taguette can import. Each code is ``Theme\\Code``.

    A parent row for the theme is written first so the prefix exists. Taguette's
    importer keeps the path string; highlight filtering treats a backslash path
    as a sub-tag of that prefix. Taguette does not import highlights.
    """
    from automated_grounded_theory.export import TAGUETTE_SEPARATOR

    used: set[str] = set()
    rows = []
    for theme in result.themes:
        parent = _unique_path(taguette_tag(str(theme.get("name") or "")), used)
        member_rows = []
        highlights = 0
        documents = 0
        for item in theme.get("code_counts") or []:
            path = _unique_path(theme_tag_path(theme["name"], item["code"]), used)
            description = theme.get("definition") or ""
            if item.get("definition"):
                description = (description + "\n\n" + item["definition"]).strip()
            member_rows.append([path, description, item["segments"], item["documents"]])
            highlights += int(item.get("segments") or 0)
            documents += int(item.get("documents") or 0)
        if TAGUETTE_SEPARATOR in parent:
            parent = parent.replace(TAGUETTE_SEPARATOR, "·")
        rows.append([parent, theme.get("definition") or "", highlights, documents])
        rows.extend(member_rows)
    _write_csv(
        place("taguette_themes.csv"),
        ["tag", "description", "number of highlights", "number of documents"],
        rows,
    )
    from automated_grounded_theory.export import codebook_qdc

    place("taguette_themes.qdc").write_text(
        codebook_qdc(
            [{"code": row[0], "definition": row[1]} for row in rows],
            origin=f"automated_grounded_theory {__version__}",
            preserve_separator=True,
        ),
        encoding="utf-8",
    )


def _write_chart(path: Path, themes: Sequence[Mapping[str, Any]], plt) -> None:
    ordered = list(reversed(list(themes)))
    labels = [str(theme["name"]) for theme in ordered]
    figure, axes = plt.subplots(figsize=(9, max(3.0, 0.45 * len(labels) + 1)))
    axes.barh(labels, [theme["pct_passages"] for theme in ordered], label="% of passages")
    axes.barh(
        labels,
        [theme["pct_respondents"] for theme in ordered],
        height=0.35,
        label="% of respondents",
    )
    axes.set_xlabel("%")
    axes.set_title("Theme prevalence")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(
    result: ThemeResult,
    output_dir: str | Path | None,
    chart: bool,
    prompt_log_path: str | Path | None,
) -> ThemeResult:
    result.totals["n_codes"] = sum(len(theme["codes"]) for theme in result.themes)
    if output_dir is not None:
        export_themes(result, output_dir, chart=chart)
    if prompt_log_path is not None:
        path = Path(prompt_log_path)
        if output_dir is None or path.resolve() != (Path(output_dir) / "prompt_log.json").resolve():
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(path, result.prompt_log)
            result.export_paths["prompt_log"] = str(path)
    return result


def _empty_result(
    source: OpenCodeResult,
    research: ResearchContext,
    *,
    model_name: str,
    n_themes: int | None,
    seed: int | None,
    respondent_key: str | None,
    quotes_per_theme: int,
    min_quote_chars: int,
    validation_retries: int,
    api_key_provided: bool,
    prompt_cache: bool,
    warnings: list[str],
) -> ThemeResult:
    totals = _totals(source.passages, respondent_key)
    totals["n_codes"] = 0
    return ThemeResult(
        themes=[],
        tensions=[],
        negative_cases=[],
        uncoded=_uncoded(source.passages, set(), respondent_key),
        totals=totals,
        prompt_log=[],
        metadata=_metadata(
            model_name=model_name,
            n_themes=n_themes,
            seed=seed,
            respondent_key=respondent_key,
            quotes_per_theme=quotes_per_theme,
            min_quote_chars=min_quote_chars,
            validation_retries=validation_retries,
            api_key_provided=api_key_provided,
            prompt_cache=prompt_cache,
            calls=0,
            question=research.research_question,
        ),
        warnings=warnings,
    )


def _metadata(**kwargs: Any) -> dict:
    return {
        "protocol": "focused-coding-1",
        "version": __version__,
        "model": kwargs["model_name"],
        "n_themes_requested": kwargs["n_themes"],
        "max_themes": kwargs.get("max_themes", 12),
        "seed": kwargs["seed"],
        "respondent_key": kwargs["respondent_key"],
        "quotes_per_theme": kwargs["quotes_per_theme"],
        "min_quote_chars": kwargs["min_quote_chars"],
        "validation_retries": kwargs["validation_retries"],
        "api_key_provided": kwargs["api_key_provided"],
        "prompt_cache": kwargs["prompt_cache"],
        "calls": kwargs["calls"],
        "research_question": kwargs["question"],
        "concurrency": kwargs.get("concurrency"),
        "theme_chunk": kwargs.get("theme_chunk"),
    }


def _coerce_result(source: OpenCodeResult | str | Path) -> OpenCodeResult:
    if isinstance(source, OpenCodeResult):
        return source
    if isinstance(source, (str, Path)):
        return OpenCodeResult.load(source)
    raise TypeError(
        "build_themes expects the OpenCodeResult returned by open_code, or a directory "
        f"from output_dir. Got {type(source).__name__}."
    )


def _validate_theme_params(
    *,
    n_themes: int | None,
    quotes_per_theme: int,
    min_quote_chars: int,
    validation_retries: int,
    concurrency: int,
    theme_chunk: int,
    max_retries: int,
    max_themes: int,
) -> None:
    if n_themes is not None and n_themes < 1:
        raise ValueError("n_themes must be at least 1, or None to let the model choose.")
    if max_themes < 1:
        raise ValueError("max_themes must be at least 1.")
    if quotes_per_theme < 1:
        raise ValueError("quotes_per_theme must be at least 1.")
    if min_quote_chars < 0:
        raise ValueError("min_quote_chars must be zero or more.")
    if validation_retries < 0 or max_retries < 0:
        raise ValueError("validation_retries and max_retries must be zero or more.")
    if concurrency < 1 or theme_chunk < 1:
        raise ValueError("concurrency and theme_chunk must be at least 1.")


def _pct(count: int, total: int) -> float:
    if not total:
        return 0.0
    return round(100 * count / total, 1)


def _meta_value(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if str(item).strip())
        return ""
    return str(value).strip()


def _clean_list(values: Any) -> list[str]:
    if not values:
        return []
    cleaned = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned


def _unique_labels(labels: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for label in labels:
        key = code_key(label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(label)
    return unique


def _unique_path(path: str, used: set[str]) -> str:
    if path not in used:
        used.add(path)
        return path
    number = 2
    while f"{path} ({number})" in used:
        number += 1
    unique = f"{path} ({number})"
    used.add(unique)
    return unique


def _count_phrase(count: int, noun: str) -> str:
    word = noun if count == 1 else noun + "s"
    return f"{count} {word}"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
