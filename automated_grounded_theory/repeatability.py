"""Repeat a frozen reviewer or a full pipeline, then compare the runs.

Neither helper calls a paid model by itself. The caller supplies the reviewer
or the pipeline callable. Temperature 0 is configuration control, not a
reproducibility guarantee: two runs can still differ.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping, Sequence

from automated_grounded_theory.async_support import run_sync
from automated_grounded_theory.decision_review import RAISED_VARIANCE_RULE


def statement_hash(text: str) -> str:
    """Hash of a whitespace-normalized statement. Empty text still hashes."""
    normalized = " ".join(str(text or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _jaccard(left: set, right: set) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _as_run(run: Any) -> Mapping[str, Any]:
    if isinstance(run, Mapping):
        return run
    data: dict[str, Any] = {}
    for key in ("relationships", "raised_code_ids", "metadata", "prompt_log"):
        if hasattr(run, key):
            data[key] = getattr(run, key)
    return data


def _raised_ids(run: Mapping[str, Any]) -> list[str] | None:
    if "raised_code_ids" in run:
        return [str(item) for item in (run.get("raised_code_ids") or [])]
    metadata = run.get("metadata") if isinstance(run.get("metadata"), Mapping) else {}
    if "raised_code_ids" in metadata:
        return [str(item) for item in (metadata.get("raised_code_ids") or [])]
    return None


def _relationships(run: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    rows = run.get("relationships")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    return [item for item in rows if isinstance(item, Mapping)]


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("from_id") or ""), str(row.get("to_id") or ""))


def _statement_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (*_pair_key(row), statement_hash(str(row.get("statement") or "")))


def _eligibility(row: Mapping[str, Any]) -> bool | None:
    value = row.get("downstream_eligible")
    if isinstance(value, bool):
        return value
    return None


def repeat_frozen_review(reviewer: Any, payload: Mapping[str, Any], schema: Any, repetitions: int) -> dict[str, Any]:
    """Rerun one reviewer on the same payload. Upstream state is not rebuilt."""
    count = max(0, int(repetitions))
    decisions = []
    for index in range(count):
        metadata: dict[str, Any] = {}
        parsed, audit = run_sync(reviewer.decide(dict(payload), schema, metadata))
        digest = None if not isinstance(audit, Mapping) else audit.get("prompt_hash")
        decisions.append(
            {
                "index": index,
                "decision": parsed,
                "status": None if not isinstance(audit, Mapping) else audit.get("status"),
                "prompt_hash": digest,
                "stop": metadata.get("decision_stop"),
            }
        )
    hashes = {item.get("prompt_hash") for item in decisions}
    prompt = decisions[0]["prompt_hash"] if decisions else None
    return {
        "experiment": "frozen_reviewer",
        "repetitions": count,
        "prompt_hash": prompt,
        "prompt_hash_stable": bool(decisions) and len(hashes) == 1 and prompt is not None,
        "decisions": decisions,
    }


def repeat_pipeline(run: Callable[[], Any], repetitions: int) -> dict[str, Any]:
    """Call the pipeline callable once per repetition. Upstream differences propagate."""
    count = max(0, int(repetitions))
    return {
        "experiment": "full_pipeline",
        "repetitions": count,
        "runs": [run() for _ in range(count)],
    }


def _pair_report(left: Any, right: Any, left_index: int, right_index: int) -> dict[str, Any]:
    left_run = _as_run(left)
    right_run = _as_run(right)
    left_ids = _raised_ids(left_run)
    right_ids = _raised_ids(right_run)
    selected = None
    if left_ids is not None and right_ids is not None:
        selected = _jaccard(set(left_ids), set(right_ids))
    left_rows = _relationships(left_run)
    right_rows = _relationships(right_run)
    directed = None
    statements = None
    eligibility = None
    if left_rows is not None and right_rows is not None:
        directed = _jaccard({_pair_key(row) for row in left_rows}, {_pair_key(row) for row in right_rows})
        statements = _jaccard(
            {_statement_key(row) for row in left_rows},
            {_statement_key(row) for row in right_rows},
        )
        eligibility = _eligibility_changes(left_rows, right_rows)
    count_left = None if left_ids is None else len(left_ids)
    count_right = None if right_ids is None else len(right_ids)
    return {
        "left": left_index,
        "right": right_index,
        "selected_code_jaccard": selected,
        "raised_code_count_left": count_left,
        "raised_code_count_right": count_right,
        "raised_code_count_diff": None if count_left is None or count_right is None else abs(count_left - count_right),
        "raised_code_ids_only_left": None
        if left_ids is None or right_ids is None
        else sorted(set(left_ids) - set(right_ids)),
        "raised_code_ids_only_right": None
        if left_ids is None or right_ids is None
        else sorted(set(right_ids) - set(left_ids)),
        "directed_pair_jaccard": directed,
        "statement_jaccard": statements,
        "eligibility_changes": eligibility,
    }


def _eligibility_row(
    key: tuple[str, str, str],
    left: bool | None,
    right: bool | None,
    *,
    ambiguous: bool = False,
) -> dict[str, Any]:
    row = {
        "from_id": key[0],
        "to_id": key[1],
        "statement_hash": key[2],
        "left": left,
        "right": right,
    }
    if ambiguous:
        row["ambiguous"] = True
    return row


def _consolidate_eligibility(rows: Sequence[Mapping[str, Any]]) -> tuple[bool | None, bool]:
    """One eligibility for duplicate rows, independent of their order.

    Agreement on one boolean is that value. Disagreement, or a missing boolean
    beside another row, is ambiguous. All-missing rows stay unknown and are
    not marked ambiguous.
    """
    values = [_eligibility(row) for row in rows]
    booleans = [item for item in values if isinstance(item, bool)]
    if any(item is None for item in values) and booleans:
        return None, True
    if len(set(booleans)) > 1:
        return None, True
    if not booleans:
        return None, False
    return booleans[0], False


def _eligibility_map(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str, str], bool | None], set[tuple[str, str, str]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_statement_key(row), []).append(row)
    resolved: dict[tuple[str, str, str], bool | None] = {}
    ambiguous: set[tuple[str, str, str]] = set()
    for key, group in grouped.items():
        value, flagged = _consolidate_eligibility(group)
        resolved[key] = value
        if flagged:
            ambiguous.add(key)
    return resolved, ambiguous


def _eligibility_changes(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Matched eligibility flips, plus additions, removals, and unknown rows.

    Duplicate rows for one statement and directed pair are combined without
    regard to order. When they disagree, the statement is ambiguous and is
    not a flip. A statement missing from one run does not hide a flip on a
    statement that both runs have. A missing boolean is unknown. It is not
    treated as a flip and it does not discard the rest of the comparison.
    ``None`` means there are no relationship rows to compare.
    """
    left, left_ambiguous = _eligibility_map(left_rows)
    right, right_ambiguous = _eligibility_map(right_rows)
    if not left and not right:
        return None
    matched = []
    unknown = []
    for key in sorted(set(left) & set(right)):
        left_value = left[key]
        right_value = right[key]
        ambiguous = key in left_ambiguous or key in right_ambiguous
        if ambiguous or left_value is None or right_value is None:
            unknown.append(_eligibility_row(key, left_value, right_value, ambiguous=ambiguous))
        elif left_value != right_value:
            matched.append(_eligibility_row(key, left_value, right_value))
    additions = [
        _eligibility_row(key, None, right[key], ambiguous=key in right_ambiguous)
        for key in sorted(set(right) - set(left))
    ]
    removals = [
        _eligibility_row(key, left[key], None, ambiguous=key in left_ambiguous)
        for key in sorted(set(left) - set(right))
    ]
    return {
        "matched": matched,
        "additions": additions,
        "removals": removals,
        "unknown_eligibility": unknown,
    }


def _mean(values: Sequence[float | None]) -> float | None:
    defined = [item for item in values if item is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _raised_count_report(runs: Sequence[Any]) -> dict[str, Any]:
    """Run-to-run raised-code counts. A missing list stays null.

    Jaccard can stay high when the counts differ. The counts are recorded
    beside it. They are null when any run has no ``raised_code_ids``.
    """
    lists = [_raised_ids(_as_run(run)) for run in runs]
    if any(item is None for item in lists):
        return {
            "raised_variance_rule": RAISED_VARIANCE_RULE,
            "raised_code_counts": None,
            "raised_code_count_min": None,
            "raised_code_count_max": None,
            "raised_code_count_range": None,
            "raised_code_counts_vary": None,
            "raised_code_ids_not_in_every_run": None,
        }
    counts = [len(item or []) for item in lists]
    present = [set(item or []) for item in lists]
    common = set.intersection(*present) if present else set()
    union = set.union(*present) if present else set()
    return {
        "raised_variance_rule": RAISED_VARIANCE_RULE,
        "raised_code_counts": counts,
        "raised_code_count_min": min(counts) if counts else None,
        "raised_code_count_max": max(counts) if counts else None,
        "raised_code_count_range": (max(counts) - min(counts)) if counts else None,
        "raised_code_counts_vary": len(set(counts)) > 1 if counts else None,
        "raised_code_ids_not_in_every_run": sorted(union - common),
    }


def compare_pipeline_runs(runs: Sequence[Any]) -> dict[str, Any]:
    """Pairwise overlap of two or more runs. Undefined comparisons stay null.

    Selected-code overlap is Jaccard of ``raised_code_ids``. Relationship
    overlap is Jaccard of directed pairs, and of directed pair plus statement
    hash. Eligibility changes compare matched statements. Additions, removals, and
    unknown eligibility are separate lists. The summary lists every matched
    flip from a comparison that has relationship rows. A comparison with no
    rows, or an unknown row inside another comparison, does not drop those
    flips. The field is null only when no comparison has relationship rows.
    """
    items = list(runs or [])
    pairwise = []
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            pairwise.append(_pair_report(items[left_index], items[right_index], left_index, right_index))
    defined = [item["eligibility_changes"] for item in pairwise if item["eligibility_changes"] is not None]
    variance = _raised_count_report(items)
    return {
        "run_count": len(items),
        "pairwise": pairwise,
        "summary": {
            "selected_code_jaccard": _mean([item["selected_code_jaccard"] for item in pairwise]),
            "directed_pair_jaccard": _mean([item["directed_pair_jaccard"] for item in pairwise]),
            "statement_jaccard": _mean([item["statement_jaccard"] for item in pairwise]),
            "eligibility_changes": None
            if not defined
            else [change for report in defined for change in report["matched"]],
            **variance,
        },
    }
