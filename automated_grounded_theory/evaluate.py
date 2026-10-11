"""Score an open-coding run without calling a model.

``evaluate_run`` reads a result object or an ``output_dir``. Chen et al.
coverage, overlap, novelty, and divergence compare two or more codebooks.
Those four use lexical clusters (token Jaccard at least 0.5) as the
aggregated code space. The paper's merge step calls a model; this module
does not, and it does not need an API key.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from automated_grounded_theory.retrieval import jaccard
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
) -> dict:
    """Metrics for one run, plus Chen et al. metrics when more codebooks are passed.

    ``source`` is an :class:`~automated_grounded_theory.results.OpenCodeResult`
    or a directory written by ``output_dir``. ``context`` is the research-context
    text, or a path to a file of that text, used only for the leakage check.
    ``log_path`` is optional JSON (one object per line, or a single object) with
    a ``seconds`` field for wall-clock time. ``other_sources`` are further runs
    or directories on the same corpus.
    """
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
    if len(books) >= 2:
        metrics["chen_metrics"] = chen_metrics(books)
    else:
        metrics["chen_metrics"] = _CHEN_NOTE
    return metrics


def chen_metrics(codebooks: Sequence[Sequence[Mapping[str, Any]] | Mapping[str, Any]]) -> dict:
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
            "method": "lexical Jaccard >= 0.5 clusters; not the paper's model merge",
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

    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            left_label, left_definition = entries[left][1], entries[left][2]
            right_label, right_definition = entries[right][1], entries[right][2]
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
        "method": "lexical Jaccard >= 0.5 clusters; not the paper's model merge",
        "per_coder": per_coder,
        "mean_coverage": round(statistics.mean(item["coverage"] for item in per_coder), 4),
        "mean_overlap": round(statistics.mean(item["overlap"] for item in per_coder), 4),
        "mean_novelty": round(statistics.mean(item["novelty"] for item in per_coder), 4),
        "mean_divergence": round(statistics.mean(item["divergence"] for item in per_coder), 4),
    }


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
    else:
        metrics["gerund_first_word_pct"] = None
        metrics["in_vivo_pct"] = None
        metrics["mean_label_words"] = None
        metrics["singleton_codes_pct"] = None
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
