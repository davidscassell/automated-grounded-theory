"""Score an open-coding run.

``evaluate_run`` reads a result object or an ``output_dir``. Chen et al.
coverage, overlap, novelty, and divergence compare two or more codebooks.
By default those four use lexical clusters (token Jaccard at least 0.5).
Pass ``model`` and the same scores are computed from concurrent same-code
judgments instead. Without a model, nothing here calls a provider.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import difflib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from automated_grounded_theory.codebook import negation_differs, spelling_variant
from automated_grounded_theory.async_support import (
    CallLimiter,
    parse_structured,
    run_sync,
    structured_runnable,
)
from automated_grounded_theory.backends import resolve_chat_model
from automated_grounded_theory.prompts import build_judge_messages
from automated_grounded_theory.retrieval import jaccard
from automated_grounded_theory.schemas import SameCodeJudgment
from automated_grounded_theory.clustering import adjusted_rand_index
from automated_grounded_theory.validation import code_key

_STOP = set(
    "a an the of and or to in on for with by as is are be from that this their its "
    "employee employees comments response what do you like most should we improve any other".split()
)
_MEMO_CUES = re.compile(
    r"\b(suggest|indicat|tension|contrast|process|pattern|compar|relat|underl|implies|"
    r"may reflect|mechanism|trade-?off|ambivalen|strateg|condition|consequen|why)\w*",
    re.IGNORECASE,
)
_CHEN_NOTE = (
    "not computed: coverage/overlap/novelty/divergence need >=2 codebooks "
    "(e.g., multi-seed or human) on the same data"
)
_CLUSTER_JACCARD = 0.5


def evaluate_run(
    source: Any,
    context: str | None = None,
    log_path: str | Path | None = None,
    other_sources: Sequence[Any] | None = None,
    *,
    model=None,
    concurrency: int = 4,
    max_retries: int = 4,
    retry_backoff: float = 0.5,
    retry_max_backoff: float = 8.0,
    api_key: str | None = None,
    temperature: float = 0.0,
    judge_limit: int = 64,
    llm_seed: int | None = None,
) -> dict:
    """Metrics for one run, plus Chen et al. metrics when more codebooks are passed.

    ``source`` is an :class:`~automated_grounded_theory.results.OpenCodeResult`
    or a directory written by ``output_dir``. ``context`` is the research-context
    text, or a path to a file of that text, used only for the leakage check.
    ``log_path`` is optional JSON (one object per line, or a single object) with
    a ``seconds`` field for wall-clock time. ``other_sources`` are further runs
    or directories on the same corpus.

    ``model``, when set, judges near-duplicate codes concurrently. Chen metrics
    then use those judgments. Omit ``model`` to keep the lexical clusters.
    This wrapper works inside a running event loop.
    """
    return run_sync(
        aevaluate(
            source,
            context,
            log_path,
            other_sources,
            model=model,
            concurrency=concurrency,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            api_key=api_key,
            temperature=temperature,
            judge_limit=judge_limit,
            llm_seed=llm_seed,
        )
    )


async def aevaluate(
    source: Any,
    context: str | None = None,
    log_path: str | Path | None = None,
    other_sources: Sequence[Any] | None = None,
    *,
    model=None,
    concurrency: int = 4,
    max_retries: int = 4,
    retry_backoff: float = 0.5,
    retry_max_backoff: float = 8.0,
    api_key: str | None = None,
    temperature: float = 0.0,
    judge_limit: int = 64,
    llm_seed: int | None = None,
) -> dict:
    """Async form of :func:`evaluate_run`."""
    if concurrency < 1 or judge_limit < 1 or max_retries < 0:
        raise ValueError("concurrency and judge_limit must be at least 1, and max_retries at least 0.")
    import time

    started = time.perf_counter()
    run = _load_run(source)
    context_text = _context_text(context, run)
    metrics = _score(run, context_text)
    if log_path:
        seconds = _wall_clock(log_path)
        if seconds is not None:
            calls = metrics["llm_calls_logged"] or 1
            passages = metrics["passages"] or 1
            metrics["wall_clock_s"] = seconds
            metrics["s_per_llm_call"] = round(seconds / calls, 1)
            metrics["s_per_passage"] = round(seconds / passages, 1)
    books = [_codes(run)]
    if other_sources:
        books.extend(_codes(_load_run(item)) for item in other_sources)
    if other_sources:
        others = [_load_run(item) for item in other_sources]
        metrics["adjusted_rand_index"] = round(
            sum(assignment_ari(run, other) for other in others) / len(others),
            4,
        )
        agreements = [aligned_assignment_agreement(run, other) for other in others]
        metrics["aligned_assignment_agreement"] = round(
            sum(item["mean_jaccard"] for item in agreements) / len(agreements),
            4,
        )
        metrics["aligned_assignment_alpha"] = round(
            sum(item["krippendorff_alpha"] for item in agreements) / len(agreements),
            4,
        )
    if model is None:
        metrics["chen_metrics"] = chen_metrics(books) if len(books) >= 2 else _CHEN_NOTE
        metrics.setdefault("stage_seconds", {})["eval"] = round(time.perf_counter() - started, 4)
        return metrics
    coder = resolve_chat_model(model, api_key=api_key, temperature=temperature, llm_seed=llm_seed)
    limiter = CallLimiter(concurrency, max_retries=max_retries, backoff=retry_backoff, max_backoff=retry_max_backoff)
    tasks = []
    if len(books) >= 2:
        tasks.append(_judge_pairs(coder, books, limiter, judge_limit, same_coder=False))
    tasks.append(_judge_pairs(coder, books, limiter, judge_limit, same_coder=True))
    judged = await asyncio.gather(*tasks)
    cursor = 0
    if len(books) >= 2:
        cross, cross_rows = judged[cursor]
        cursor += 1
        metrics["chen_metrics"] = chen_metrics(books, links=cross)
    else:
        cross_rows = []
        metrics["chen_metrics"] = _CHEN_NOTE
    _within, within_rows = judged[cursor]
    metrics["llm_judgments"] = {
        "cross_codebook": cross_rows,
        "within_codebook": within_rows,
        "same_cross": sum(1 for row in cross_rows if row["same"]),
        "same_within": sum(1 for row in within_rows if row["same"]),
    }
    metrics.setdefault("stage_seconds", {})["eval"] = round(time.perf_counter() - started, 4)
    return metrics


def chen_metrics(
    codebooks: Sequence[Sequence[Mapping[str, Any]] | Mapping[str, Any]],
    links: Sequence[tuple[str, str]] | None = None,
) -> dict:
    """Coverage, overlap, novelty, and divergence across codebooks.

    Clusters are connected components of codes whose labels (or label plus
    definition) have token Jaccard of at least 0.5. Weights follow Chen et al.
    (arXiv:2411.12142): each code is weighted by how many coders reach it, and
    each coder is down-weighted as their codebook grows. Divergence is the
    square root of Jensen-Shannon divergence between a coder and the others.
    """
    coders = [_code_rows(book) for book in codebooks]
    if len(coders) < 2:
        raise ValueError("chen_metrics needs at least two codebooks.")
    entries: list[tuple[int, str, str]] = []
    for coder_index, rows in enumerate(coders):
        for row in rows:
            entries.append((coder_index, row["code"], row["definition"]))
    if not entries:
        empty = {"coverage": 0.0, "overlap": 0.0, "novelty": 0.0, "divergence": 0.0, "codes": 0}
        return {
            "method": _chen_method(links),
            "per_coder": [dict(empty) for _ in coders],
            "mean_coverage": 0.0,
            "mean_overlap": 0.0,
            "mean_novelty": 0.0,
            "mean_divergence": 0.0,
        }

    parent = list(range(len(entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    linked = _link_keys(links)
    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            left_label, left_definition = entries[left][1], entries[left][2]
            right_label, right_definition = entries[right][1], entries[right][2]
            if linked is not None:
                left_key = code_key(left_label)
                right_key = code_key(right_label)
                if left_key == right_key or frozenset((left_key, right_key)) in linked:
                    union(left, right)
                continue
            score = jaccard(left_label, right_label)
            if score < _CLUSTER_JACCARD:
                score = jaccard(f"{left_label} {left_definition}", f"{right_label} {right_definition}")
            if score >= _CLUSTER_JACCARD:
                union(left, right)

    clusters: dict[int, list[int]] = collections.defaultdict(list)
    for index in range(len(entries)):
        clusters[find(index)].append(index)
    cluster_ids = list(clusters)
    membership = []
    for coder_index, _rows in enumerate(coders):
        present = set()
        for cluster_id, members in clusters.items():
            if any(entries[member][0] == coder_index for member in members):
                present.add(cluster_id)
        membership.append(present)

    sizes = [len(rows) for rows in coders]
    median_size = statistics.median(sizes) if sizes else 0
    weights = []
    for size in sizes:
        sized = max(float(size), float(median_size), 2.0)
        weights.append(1.0 / math.log(sized))

    observations = []
    for present in membership:
        observations.append([1.0 if cluster_id in present else 0.0 for cluster_id in cluster_ids])

    scores = []
    for cluster_index in range(len(cluster_ids)):
        total = 0.0
        for coder_index, obs in enumerate(observations):
            total += obs[cluster_index] * weights[coder_index]
        scores.append(total)
    score_total = sum(scores) or 1.0

    novel = []
    for cluster_index, cluster_id in enumerate(cluster_ids):
        owners = sum(1 for present in membership if cluster_id in present)
        novel.append(owners == 1)
    novel_score_total = sum(score for score, is_novel in zip(scores, novel) if is_novel)

    per_coder = []
    for coder_index, obs in enumerate(observations):
        coverage = sum(obs_value * score for obs_value, score in zip(obs, scores)) / score_total
        without = [max(0.0, score - obs_value * weights[coder_index]) for obs_value, score in zip(obs, scores)]
        without_total = sum(without)
        overlap = (
            sum(obs_value * score for obs_value, score in zip(obs, without)) / without_total if without_total else 0.0
        )
        if novel_score_total:
            novelty = sum(
                obs_value * score for obs_value, score, is_novel in zip(obs, scores, novel) if is_novel
            ) / novel_score_total
        else:
            novelty = 0.0
        baseline = without
        divergence = math.sqrt(_jsd(baseline, obs))
        per_coder.append(
            {
                "coder": coder_index,
                "codes": sizes[coder_index],
                "coverage": round(coverage, 4),
                "overlap": round(overlap, 4),
                "novelty": round(novelty, 4),
                "divergence": round(divergence, 4),
            }
        )
    return {
        "method": _chen_method(links),
        "per_coder": per_coder,
        "mean_coverage": round(statistics.mean(item["coverage"] for item in per_coder), 4),
        "mean_overlap": round(statistics.mean(item["overlap"] for item in per_coder), 4),
        "mean_novelty": round(statistics.mean(item["novelty"] for item in per_coder), 4),
        "mean_divergence": round(statistics.mean(item["divergence"] for item in per_coder), 4),
    }


def assignment_ari(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Adjusted Rand index of passage→code assignments.

    Passages are paired by ``passage_id``. Several codes on one passage are
    joined in sorted order so the labeling does not depend on segment order.
    This is a stability score across seeds. It is not a claim of saturation.

    The index compares partitions, not labels. Two runs that give every
    passage its own different code score 1.0, because no two passages share a
    cluster in either run. Use :func:`aligned_assignment_agreement` when the
    question is whether the same codes were applied to the same passages.
    """
    left_labels = _passage_labels(left)
    right_labels = _passage_labels(right)
    keys = sorted(set(left_labels) | set(right_labels))
    if not keys:
        return 1.0
    return adjusted_rand_index(
        [left_labels.get(key, "") for key in keys],
        [right_labels.get(key, "") for key in keys],
    )


def _run_segments(run: Any) -> list[Mapping[str, Any]]:
    if isinstance(run, Mapping):
        return [item for item in (run.get("segments") or []) if isinstance(item, Mapping)]
    return [item for item in (getattr(run, "segments", []) or []) if isinstance(item, Mapping)]


def _passage_code_sets(run: Any) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for segment in _run_segments(run):
        passage_id = str(segment.get("passage_id") or "")
        code = str(segment.get("code") or "")
        if passage_id and code:
            grouped.setdefault(passage_id, set()).add(code)
    return grouped


def _set_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def nominal_krippendorff_alpha(units: Sequence[Sequence[Any]]) -> float:
    """Nominal Krippendorff's alpha. ``None`` in a unit is a missing rating.

    Returns 1 when there is no disagreement to explain, including a single
    category with perfect agreement.
    """
    observed: list[list[Any]] = []
    for unit in units:
        ratings = [value for value in unit if value is not None]
        if len(ratings) >= 2:
            observed.append(ratings)
    if len(observed) < 2:
        return 1.0
    categories = sorted({value for ratings in observed for value in ratings}, key=str)
    index = {category: position for position, category in enumerate(categories)}
    size = len(categories)
    coincidence = [[0.0 for _ in range(size)] for _ in range(size)]
    for ratings in observed:
        count = len(ratings)
        for left_index, left in enumerate(ratings):
            for right_index, right in enumerate(ratings):
                if left_index == right_index:
                    continue
                coincidence[index[left]][index[right]] += 1.0 / (count - 1)
    total = sum(sum(row) for row in coincidence)
    if total <= 1:
        return 1.0
    observed_disagreement = 0.0
    for left in range(size):
        for right in range(size):
            if left != right:
                observed_disagreement += coincidence[left][right]
    observed_disagreement /= total
    marginal = [sum(coincidence[left][right] for right in range(size)) for left in range(size)]
    expected = 0.0
    for left in range(size):
        for right in range(size):
            if left != right:
                expected += marginal[left] * marginal[right]
    expected /= total * (total - 1)
    if expected == 0:
        return 1.0
    return 1.0 - (observed_disagreement / expected)


def _labels_match(left: str, right: str, *, align_threshold: float | None) -> bool:
    """Exact key or spelling variant. A negation or polarity flip does not match.

    ``align_threshold`` is an explicit Jaccard fallback. It still refuses a
    polarity difference. The default does not use token overlap.
    """
    left_key = code_key(left)
    right_key = code_key(right)
    if left_key and left_key == right_key:
        return True
    if negation_differs(left, right):
        return False
    if spelling_variant(left, right):
        return True
    if align_threshold is None:
        return False
    return jaccard(left, right) >= float(align_threshold)


def aligned_assignment_agreement(
    left: Any,
    right: Any,
    *,
    align_threshold: float | None = None,
    mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Per-passage code-set agreement after aligning codes by label.

    The default alignment is exact (case and whitespace) or a spelling variant
    (punctuation, plural, gerund). ``trusting managers to share information``
    and ``not trusting managers to share information`` do not align: token
    Jaccard is high, and the polarity is opposite.

    ``mapping`` is an explicit reviewed equivalence, ``{label: other label}``.
    Those pairs align even when the default would keep them apart. Pass
    ``align_threshold`` to also pair labels at that token Jaccard, still
    without pairing a polarity flip.

    Each passage then contributes the Jaccard of its two code sets, using
    those aligned identities. Unmatched codes stay distinct.

    ``krippendorff_alpha`` is nominal alpha on the aligned passage-by-code
    matrix: each cell is whether that coder applied that code to that passage.
    Cells where neither coder applied the code are omitted.

    Unlike :func:`assignment_ari`, different singleton codes on every passage
    score low here. Adjusted Rand only asks whether passages were grouped
    together, so an all-singleton partition scores 1 even when the codes share
    no labels.
    """
    left_sets = _passage_code_sets(left)
    right_sets = _passage_code_sets(right)
    left_codes = {code for labels in left_sets.values() for code in labels}
    right_codes = {code for labels in right_sets.values() for code in labels}
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            nxt = parent[key]
            parent[key] = root
            key = nxt
        return root

    def union(left_key: str, right_key: str) -> None:
        left_root = find(left_key)
        right_root = find(right_key)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for left_code in left_codes:
        for right_code in right_codes:
            if _labels_match(left_code, right_code, align_threshold=align_threshold):
                left_key = code_key(left_code)
                right_key = code_key(right_code)
                if left_key and right_key:
                    union(left_key, right_key)
    if mapping:
        for raw_left, raw_right in mapping.items():
            left_key = code_key(str(raw_left))
            right_key = code_key(str(raw_right))
            if left_key and right_key:
                union(left_key, right_key)

    def canon(label: str) -> str:
        key = code_key(label)
        if not key:
            return ""
        return find(key)

    left_roots = {canon(code) for code in left_codes}
    aligned_roots: set[str] = set()
    for code in right_codes:
        root = canon(code)
        if root and root in left_roots:
            aligned_roots.add(root)

    passages = sorted(set(left_sets) | set(right_sets))
    scores: list[float] = []
    units: list[tuple[int, int]] = []
    for passage_id in passages:
        left_canon = {canon(label) for label in left_sets.get(passage_id, set())}
        right_canon = {canon(label) for label in right_sets.get(passage_id, set())}
        left_canon.discard("")
        right_canon.discard("")
        scores.append(_set_jaccard(left_canon, right_canon))
        for code in left_canon | right_canon:
            units.append((1 if code in left_canon else 0, 1 if code in right_canon else 0))
    mean = sum(scores) / len(scores) if scores else 1.0
    return {
        "mean_jaccard": round(mean, 4),
        "krippendorff_alpha": round(nominal_krippendorff_alpha(units), 4),
        "passages": len(passages),
        "aligned_codes": len(aligned_roots),
    }


def _passage_labels(run: Any) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for segment in _run_segments(run):
        passage_id = str(segment.get("passage_id") or "")
        if not passage_id:
            continue
        grouped.setdefault(passage_id, []).append(str(segment.get("code") or ""))
    return {passage_id: "|".join(sorted(labels)) for passage_id, labels in grouped.items()}


def _chen_method(links: Sequence[tuple[str, str]] | None) -> str:
    if links is None:
        return "lexical Jaccard >= 0.5 clusters; not the paper's model merge"
    return "llm same-code judgments, reconciled in pair order"


def _link_keys(links: Sequence[tuple[str, str]] | None) -> set[frozenset[str]] | None:
    if links is None:
        return None
    found: set[frozenset[str]] = set()
    for left, right in links:
        left_key = code_key(str(left))
        right_key = code_key(str(right))
        if left_key and right_key and left_key != right_key:
            found.add(frozenset((left_key, right_key)))
    return found


def _candidate_pairs(codebooks: Sequence, limit: int, *, same_coder: bool) -> list[tuple[float, dict, dict]]:
    """Near-duplicate pairs, highest overlap first, so the cap is deterministic."""
    rows: list[tuple[int, dict]] = []
    for coder_index, book in enumerate(codebooks):
        for row in _code_rows(book):
            rows.append((coder_index, row))
    found: list[tuple[float, dict, dict]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if (rows[left][0] == rows[right][0]) != same_coder:
                continue
            left_row, right_row = rows[left][1], rows[right][1]
            score = jaccard(left_row["code"], right_row["code"])
            if score < _CLUSTER_JACCARD:
                score = max(
                    score,
                    jaccard(
                        f"{left_row['code']} {left_row['definition']}",
                        f"{right_row['code']} {right_row['definition']}",
                    ),
                )
            if score < 0.2:
                continue
            found.append((score, left_row, right_row))
    found.sort(
        key=lambda item: (
            -item[0],
            item[1]["code"].casefold(),
            item[2]["code"].casefold(),
            item[1]["definition"].casefold(),
            item[2]["definition"].casefold(),
        )
    )
    return found[:limit]


async def _judge_pairs(model, codebooks, limiter: CallLimiter, limit: int, *, same_coder: bool):
    pairs = _candidate_pairs(codebooks, limit, same_coder=same_coder)
    if not pairs:
        return [], []
    runnable = structured_runnable(model, SameCodeJudgment)
    results = await limiter.abatch(
        runnable,
        [build_judge_messages(left, right) for _score, left, right in pairs],
    )
    links: list[tuple[str, str]] = []
    rows = []
    for (_score, left, right), result in zip(pairs, results):
        payload, _usage = parse_structured(result, SameCodeJudgment)
        same = bool(payload.get("same"))
        rows.append(
            {
                "left": left["code"],
                "right": right["code"],
                "same": same,
                "rationale": payload.get("rationale") or "",
            }
        )
        if same:
            links.append((left["code"], right["code"]))
    return links, rows


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``agt-evaluate RUN_DIR [CONTEXT] [LOG] [--also RUN_DIR ...]``."""
    parser = argparse.ArgumentParser(
        prog="agt-evaluate",
        description="Score an automated_grounded_theory output directory. Does not call a model.",
    )
    parser.add_argument("run_dir", help="Directory written by open_code(..., output_dir=...)")
    parser.add_argument("context", nargs="?", help="Research context text file, for the leakage check")
    parser.add_argument("log", nargs="?", help="JSON log line with a seconds field, for wall-clock metrics")
    parser.add_argument(
        "--also",
        nargs="*",
        default=None,
        help="Other output directories on the same corpus, for Chen et al. metrics",
    )
    parser.add_argument("-o", "--output", help="Also write the JSON to this path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    metrics = evaluate_run(args.run_dir, context=args.context, log_path=args.log, other_sources=args.also)
    text = json.dumps(metrics, indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _score(run: Mapping[str, Any], context_text: str) -> dict:
    codebook = list(run.get("codebook") or [])
    segments = list(run.get("segments") or [])
    passages = list(run.get("passages") or [])
    memos = list(run.get("memos") or [])
    sweeps = list(run.get("sweep_log") or [])
    prompts = list(run.get("prompt_log") or [])
    saturation = _saturation_points(run.get("saturation"))
    passage_text = {_passage_id(passage): str(passage.get("text") or "") for passage in passages}
    metrics: dict[str, Any] = {
        "passages": len(passages),
        "skipped_passages": sum(1 for passage in passages if passage.get("skipped")),
        "total_codes": len(codebook),
        "segments": len(segments),
    }
    counts = collections.Counter(segment.get("passage_id") for segment in segments)
    per_passage = [counts.get(passage_id, 0) for passage_id in passage_text] or [0]
    metrics["codes_per_passage_mean"] = round(statistics.mean(per_passage), 3) if passage_text else 0.0
    metrics["codes_per_passage_median"] = statistics.median(per_passage) if passage_text else 0.0
    metrics["coverage_pct"] = round(100 * sum(count > 0 for count in per_passage) / len(per_passage), 1) if passage_text else 0.0
    if segments:
        verbatim = sum(
            str(segment.get("quote") or "") in passage_text.get(segment.get("passage_id"), "") for segment in segments
        )
        metrics["verbatim_quote_pct"] = round(100 * verbatim / len(segments), 1)
    else:
        metrics["verbatim_quote_pct"] = None
    metrics.update(_quote_anchor_rates(segments))
    eligible = [passage for passage in passages if not passage.get("skipped")]
    uncoded = [
        passage
        for passage in eligible
        if counts.get(_passage_id(passage), 0) == 0 and passage.get("coverage_status") != "failed"
    ]
    failed = [passage for passage in passages if passage.get("coverage_status") == "failed"]
    metrics["uncoded_share"] = round(len(uncoded) / len(eligible), 4) if eligible else 0.0
    metrics["failed_slot_share"] = round(len(failed) / len(passages), 4) if passages else 0.0
    metrics["excluded_passages"] = sum(1 for passage in passages if passage.get("coverage_status") == "excluded" or passage.get("skipped"))
    labels = [str(entry.get("code") or "") for entry in codebook if entry.get("code")]
    if labels:
        metrics["gerund_first_word_pct"] = round(
            100 * sum(label.split()[0].lower().endswith("ing") for label in labels if label.split()) / len(labels),
            1,
        )
        metrics["in_vivo_pct"] = round(100 * sum(bool(entry.get("in_vivo")) for entry in codebook) / len(codebook), 1)
        metrics["mean_label_words"] = round(statistics.mean(len(label.split()) for label in labels), 2)
        metrics["singleton_codes_pct"] = round(
            100 * sum(int(entry.get("frequency") or 0) == 1 for entry in codebook) / len(codebook),
            1,
        )
        metrics["code_reuse_pct"] = _code_reuse_pct(segments)
        metrics["code_reuse_target"] = (
            "Share of segments that reuse a code already applied earlier. "
            "Prefer reuse or refinement over a new singleton. "
            "Nearest existing codes are listed in the coding prompt for that comparison."
        )
    else:
        metrics["gerund_first_word_pct"] = None
        metrics["in_vivo_pct"] = None
        metrics["mean_label_words"] = None
        metrics["singleton_codes_pct"] = None
        metrics["code_reuse_pct"] = None
        metrics["code_reuse_target"] = (
            "Share of segments that reuse a code already applied earlier. "
            "Prefer reuse or refinement over a new singleton."
        )
    pairs = _redundant_pairs(codebook)
    metrics["redundant_pairs_count"] = len(pairs)
    metrics["redundant_pairs_top"] = pairs[:15]
    metrics["redundancy_threshold"] = "label TF-IDF cosine>0.5 or difflib ratio>0.75"
    metrics["saturation"] = [
        (point.get("batch"), point.get("new_codes"), point.get("new_codes_per_passage")) for point in saturation
    ]
    below = [
        point.get("batch")
        for point in saturation
        if point.get("new_codes_per_passage") is not None and float(point["new_codes_per_passage"]) < 0.10
    ]
    metrics["batch_new_rate_below_10pct"] = below[0] if below else None
    below_post = [
        point.get("batch")
        for point in saturation
        if point.get("new_codes_per_passage_post_merge") is not None
        and float(point["new_codes_per_passage_post_merge"]) < 0.10
    ]
    metrics["batch_post_merge_rate_below_10pct"] = below_post[0] if below_post else None
    metrics["saturation_post_merge"] = [
        (
            point.get("batch"),
            point.get("new_codes_post_merge"),
            point.get("new_codes_per_passage_post_merge"),
        )
        for point in saturation
    ]
    review_below = 0.5
    metadata = run.get("metadata") or {}
    params = metadata.get("params") if isinstance(metadata, Mapping) else None
    if isinstance(params, Mapping) and params.get("review_below") is not None:
        review_below = float(params["review_below"])
    elif isinstance(metadata, Mapping) and metadata.get("review_below") is not None:
        review_below = float(metadata["review_below"])
    segment_confidence = _confidence_values(segment.get("confidence") for segment in segments)
    merge_confidence = []
    for sweep in sweeps:
        for row in list(sweep.get("accepted_merges") or []) + list(sweep.get("rejected_merges") or []) + list(
            sweep.get("candidate_decisions") or []
        ):
            if row.get("confidence") is not None:
                merge_confidence.append(row.get("confidence"))
    metrics["confidence"] = _confidence_distribution(segment_confidence, review_below)
    metrics["confidence_merges"] = _confidence_distribution(_confidence_values(merge_confidence), review_below)
    metrics["review_below"] = review_below
    metrics["consistency"] = _consistency(run)
    metrics["definition_not_survivor"] = metrics["consistency"]["definition_not_survivor"]
    metrics["sweeps"] = len(sweeps)
    metrics["merge_candidates"] = sum(len(item.get("candidates") or []) for item in sweeps)
    metrics["merges_accepted"] = sum(len(item.get("accepted_merges") or []) for item in sweeps)
    metrics["overloaded_flags"] = sum(len(item.get("overloaded_codes") or []) for item in sweeps)
    passage_memos = [memo for memo in memos if memo.get("kind") == "passage"]
    lengths = [len(str(memo.get("text") or "").split()) for memo in passage_memos]
    metrics["passage_memos"] = len(passage_memos)
    metrics["memo_words_mean"] = round(statistics.mean(lengths), 1) if lengths else None
    metrics["memo_words_median"] = statistics.median(lengths) if lengths else None
    if passage_memos:
        analytic = sum(len(_MEMO_CUES.findall(str(memo.get("text") or ""))) >= 2 for memo in passage_memos)
        metrics["analytic_memo_pct_heuristic"] = round(100 * analytic / len(passage_memos), 1)
    else:
        metrics["analytic_memo_pct_heuristic"] = None
    metrics["context_leakage_pct"] = _leakage(labels, context_text)
    metrics["context_leakage_def"] = ">=50% of non-verb label tokens appear in the research context"
    chars = sum(len(json.dumps(prompt.get("messages", ""), default=repr)) for prompt in prompts)
    metrics["llm_calls_logged"] = len(prompts)
    metrics["llm_calls_by_kind"] = dict(collections.Counter(prompt.get("kind") for prompt in prompts))
    metrics["prompt_tokens_est_total"] = chars // 4
    metrics["prompt_tokens_est_per_passage"] = round(chars / 4 / len(passages)) if passages else 0
    usage_input = sum(int((prompt.get("usage") or {}).get("input_tokens") or 0) for prompt in prompts)
    usage_output = sum(int((prompt.get("usage") or {}).get("output_tokens") or 0) for prompt in prompts)
    usage_total = sum(int((prompt.get("usage") or {}).get("total_tokens") or 0) for prompt in prompts)
    metrics["usage_input_tokens"] = usage_input
    metrics["usage_output_tokens"] = usage_output
    metrics["usage_total_tokens"] = usage_total
    if usage_total and passages:
        metrics["tokens_per_passage"] = round(usage_total / len(passages), 1)
        metrics["tokens_note"] = "usage tokens recorded on prompt_log entries"
    else:
        metrics["tokens_per_passage"] = metrics["prompt_tokens_est_per_passage"]
        metrics["tokens_note"] = "estimate: prompt chars/4; prompt_log usage was zero"
    return metrics


def _confidence_values(values) -> list[float]:
    scores = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            scores.append(number)
    return scores


def _quote_anchor_rates(segments: Sequence[Mapping[str, Any]]) -> dict:
    """Shares of quote_status values. Flagged means possibly_modified or unanchored."""
    empty = {
        "quote_status_counts": {},
        "quote_flag_rate": None,
        "quote_possibly_modified_rate": None,
        "quote_unanchored_rate": None,
        "quote_fuzzy_match_rate": None,
        "quote_exact_rate": None,
        "quote_normalized_match_rate": None,
    }
    if not segments:
        return empty
    counts = collections.Counter(str(segment.get("quote_status") or "unspecified") for segment in segments)
    total = len(segments)

    def rate(status: str) -> float:
        return round(counts.get(status, 0) / total, 4)

    flagged = counts.get("possibly_modified", 0) + counts.get("unanchored", 0)
    return {
        "quote_status_counts": dict(sorted(counts.items())),
        "quote_flag_rate": round(flagged / total, 4),
        "quote_possibly_modified_rate": rate("possibly_modified"),
        "quote_unanchored_rate": rate("unanchored"),
        "quote_fuzzy_match_rate": rate("fuzzy_match"),
        "quote_exact_rate": rate("exact"),
        "quote_normalized_match_rate": rate("normalized_match"),
    }


def _consistency(run: Mapping[str, Any]) -> dict:
    """Post-merge consistency: retired-code definitions should not pile up on the review list."""
    review = list(run.get("review") or [])
    checked = 0
    split = 0
    for sweep in run.get("sweep_log") or []:
        review.extend(sweep.get("review") or [])
        checked += int(sweep.get("quotes_checked") or 0)
        split += int(sweep.get("quotes_split") or 0)
    survivor = sum(1 for item in review if str(item.get("reason") or "") == "definition_not_survivor")
    return {
        "definition_not_survivor": survivor,
        "quotes_checked": checked,
        "quotes_split": split,
        "quote_fit_rate": round(1.0 - (split / checked), 4) if checked else None,
    }


def _confidence_distribution(values: Sequence[float], review_below: float) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = ordered[(len(ordered) - 1) // 2]
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "min": min(values),
        "max": max(values),
        "p50": middle,
        "below_review": sum(value < review_below for value in values),
    }


def _redundant_pairs(codebook: Sequence[Mapping[str, Any]]) -> list[list]:
    labels = [str(entry.get("code") or "") for entry in codebook]
    if len(labels) < 2:
        return []

    def tokens(text: str) -> list[str]:
        return [word for word in re.findall(r"[a-z]+", text.lower()) if word not in _STOP]

    docs = [tokens(label + " " + label) for label in labels]
    document_frequency: collections.Counter = collections.Counter(word for doc in docs for word in set(doc))
    count = len(docs)
    vectors = []
    for doc in docs:
        term_frequency = collections.Counter(doc)
        raw = {word: term_frequency[word] * math.log((1 + count) / (1 + document_frequency[word]) + 1) for word in term_frequency}
        norm = math.sqrt(sum(value * value for value in raw.values())) or 1
        vectors.append({word: value / norm for word, value in raw.items()})
    pairs = []
    for left in range(count):
        for right in range(left + 1, count):
            cosine = sum(value * vectors[right].get(word, 0) for word, value in vectors[left].items())
            ratio = difflib.SequenceMatcher(None, labels[left], labels[right]).ratio()
            if cosine > 0.5 or ratio > 0.75:
                pairs.append([labels[left], labels[right], round(cosine, 2), round(ratio, 2)])
    pairs.sort(key=lambda item: -max(item[2], item[3]))
    return pairs


def _leakage(labels: Sequence[str], context_text: str) -> float | None:
    if not labels:
        return None

    def tokens(text: str) -> list[str]:
        return [word for word in re.findall(r"[a-z]+", text.lower()) if word not in _STOP]

    context_words = set(tokens(context_text)) - {"s"}
    if not context_words:
        return 0.0
    ratios = []
    for label in labels:
        words = tokens(label)[1:]
        ratios.append(sum(word in context_words for word in words) / max(1, len(words)))
    return round(100 * sum(ratio >= 0.5 for ratio in ratios) / len(ratios), 1)


def _code_reuse_pct(segments: Sequence[Mapping[str, Any]]) -> float | None:
    """Percent of segments whose code was already used on an earlier segment."""
    if not segments:
        return None
    ordered = sorted(
        segments,
        key=lambda item: (
            int(item.get("coding_order") or 0),
            str(item.get("segment_id") or ""),
        ),
    )
    seen: set[str] = set()
    reused = 0
    counted = 0
    for segment in ordered:
        key = code_key(str(segment.get("code") or ""))
        if not key:
            continue
        counted += 1
        if key in seen:
            reused += 1
        else:
            seen.add(key)
    if not counted:
        return None
    return round(100 * reused / counted, 1)


def _saturation_points(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        points = payload.get("points")
        if isinstance(points, list):
            return points
        return []
    if isinstance(payload, list):
        return payload
    return []


def _load_run(source: Any) -> dict:
    if isinstance(source, (str, Path)):
        directory = Path(source)
        if not directory.is_dir():
            raise FileNotFoundError(f"No output directory at {directory}")

        def load(name: str, default: Any) -> Any:
            path = directory / name
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))

        saturation = load("saturation.json", [])
        return {
            "codebook": load("codebook.json", []),
            "segments": load("coded_segments.json", []),
            "passages": load("passages.json", []),
            "memos": load("memos.json", []),
            "sweep_log": load("sweep_log.json", []),
            "prompt_log": load("prompt_log.json", []),
            "saturation": saturation,
            "metadata": load("run_metadata.json", {}),
        }
    if isinstance(source, Mapping):
        return dict(source)
    saturation = getattr(source, "saturation_report", None)
    if saturation is None:
        saturation = getattr(source, "saturation", [])
    return {
        "codebook": list(getattr(source, "codebook", []) or []),
        "segments": list(getattr(source, "segments", []) or []),
        "passages": list(getattr(source, "passages", []) or []),
        "memos": list(getattr(source, "memos", []) or []),
        "sweep_log": list(getattr(source, "sweep_log", []) or []),
        "prompt_log": list(getattr(source, "prompt_log", []) or []),
        "saturation": saturation if isinstance(saturation, dict) else list(getattr(source, "saturation", []) or []),
        "metadata": dict(getattr(source, "metadata", {}) or {}),
    }


def _codes(run: Mapping[str, Any]) -> list[dict]:
    return list(run.get("codebook") or [])


def _code_rows(book: Any) -> list[dict]:
    rows = book
    if isinstance(book, Mapping) and "codebook" in book:
        rows = book["codebook"]
    found = []
    for entry in rows or []:
        if isinstance(entry, str):
            label, definition = entry, ""
        else:
            label = str(entry.get("code") or "")
            definition = str(entry.get("definition") or "")
        if label:
            found.append({"code": label, "definition": definition, "key": code_key(label)})
    return found


def _context_text(context: str | None, run: Mapping[str, Any]) -> str:
    if context:
        path = Path(context)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return context
    stored = (run.get("metadata") or {}).get("context") or {}
    if isinstance(stored, str):
        return stored
    if not isinstance(stored, Mapping):
        return ""
    parts = [
        str(stored.get("research_question") or ""),
        str(stored.get("population") or ""),
        str(stored.get("setting") or ""),
        " ".join(stored.get("sensitizing_concepts") or []),
        str(stored.get("notes") or ""),
    ]
    return "\n".join(part for part in parts if part)


def _wall_clock(log_path: str | Path) -> float | None:
    path = Path(log_path)
    if not path.is_file():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    objects = []
    for line in lines:
        if line.startswith("{") or line.startswith("["):
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not objects and path.suffix == ".json":
        try:
            objects.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return None
    for payload in reversed(objects):
        if isinstance(payload, dict) and "seconds" in payload:
            return float(payload["seconds"])
    return None


def _passage_id(passage: Mapping[str, Any]) -> str:
    return str(passage.get("passage_id") or passage.get("id") or "")


def _jsd(left: Sequence[float], right: Sequence[float]) -> float:
    def normalize(values: Sequence[float]) -> list[float]:
        cleaned = [max(0.0, float(value)) for value in values]
        total = sum(cleaned)
        if total <= 0:
            if not cleaned:
                return []
            share = 1.0 / len(cleaned)
            return [share for _ in cleaned]
        return [value / total for value in cleaned]

    def kl(source: Sequence[float], target: Sequence[float]) -> float:
        total = 0.0
        for origin, destination in zip(source, target):
            if origin > 0 and destination > 0:
                total += origin * math.log(origin / destination)
        return total

    left_n = normalize(left)
    right_n = normalize(right)
    if not left_n:
        return 0.0
    mixture = [(a + b) / 2 for a, b in zip(left_n, right_n)]
    return 0.5 * kl(left_n, mixture) + 0.5 * kl(right_n, mixture)


if __name__ == "__main__":
    raise SystemExit(main())
