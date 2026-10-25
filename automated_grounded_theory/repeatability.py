"""Repeat a frozen reviewer or a full pipeline, then compare the runs.

Neither helper calls a paid model by itself. The caller supplies the reviewer
or the pipeline callable. Temperature 0 is configuration control, not a
reproducibility guarantee: two runs can still differ.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping, Sequence

from automated_grounded_theory.async_support import run_sync


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
    return {
        "left": left_index,
        "right": right_index,
        "selected_code_jaccard": selected,
        "directed_pair_jaccard": directed,
        "statement_jaccard": statements,
        "eligibility_changes": eligibility,
    }


def _eligibility_row(key: tuple[str, str, str], left: bool | None, right: bool | None) -> dict[str, Any]:
    return {
        "from_id": key[0],
        "to_id": key[1],
        "statement_hash": key[2],
        "left": left,
        "right": right,
    }


def _eligibility_values(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[bool | None]]:
    grouped: dict[tuple[str, str, str], list[bool | None]] = {}
    for row in rows:
        grouped.setdefault(_statement_key(row), []).append(_eligibility(row))
    return grouped


def _distinct_eligibility(values: Sequence[bool | None]) -> list[bool | None]:
    rank = {False: 0, True: 1, None: 2}
    unique: list[bool | None] = []
    for item in values:
        if item not in unique:
            unique.append(item)
    return sorted(unique, key=lambda item: rank.get(item, 3))


def _conflict_row(
    key: tuple[str, str, str],
    left_values: Sequence[bool | None],
    right_values: Sequence[bool | None],
) -> dict[str, Any]:
    return {
        "from_id": key[0],
        "to_id": key[1],
        "statement_hash": key[2],
        "left_values": list(left_values),
        "right_values": list(right_values),
    }


def _eligibility_changes(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Matched eligibility flips, plus additions, removals, and unknown rows.

    A statement missing from one run does not hide a flip on a statement that
    both runs have. A missing boolean is unknown. It is not treated as a flip
    and it does not discard the rest of the comparison. Duplicate statement
    keys keep every distinct eligibility value. Conflicting values are reported
    and are not resolved by row order. ``None`` means there are no relationship
    rows to compare.
    """
    left = _eligibility_values(left_rows)
    right = _eligibility_values(right_rows)
    if not left and not right:
        return None
    matched = []
    unknown = []
    conflicting = []
    for key in sorted(set(left) & set(right)):
        left_values = _distinct_eligibility(left[key])
        right_values = _distinct_eligibility(right[key])
        if len(left_values) > 1 or len(right_values) > 1:
            conflicting.append(_conflict_row(key, left_values, right_values))
            continue
        left_value = left_values[0]
        right_value = right_values[0]
        if left_value is None or right_value is None:
            unknown.append(_eligibility_row(key, left_value, right_value))
        elif left_value != right_value:
            matched.append(_eligibility_row(key, left_value, right_value))
    additions = []
    for key in sorted(set(right) - set(left)):
        values = _distinct_eligibility(right[key])
        if len(values) > 1:
            conflicting.append(_conflict_row(key, [], values))
        else:
            additions.append(_eligibility_row(key, None, values[0]))
    removals = []
    for key in sorted(set(left) - set(right)):
        values = _distinct_eligibility(left[key])
        if len(values) > 1:
            conflicting.append(_conflict_row(key, values, []))
        else:
            removals.append(_eligibility_row(key, values[0], None))
    return {
        "matched": matched,
        "additions": additions,
        "removals": removals,
        "unknown_eligibility": unknown,
        "conflicting_duplicates": conflicting,
    }


def _combine_eligibility(reports: Sequence[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Aggregate comparisons that have relationship rows.

    A pairwise report of ``None`` is unavailable. It is counted and it does
    not erase a flip from a comparison that does have rows. The result is null
    only when every comparison is unavailable.
    """
    available = [item for item in reports if isinstance(item, Mapping)]
    if not available:
        return None
    combined: dict[str, Any] = {
        "matched": [],
        "additions": [],
        "removals": [],
        "unknown_eligibility": [],
        "conflicting_duplicates": [],
        "comparisons_attempted": len(reports),
        "comparisons_available": len(available),
        "comparisons_unavailable": len(reports) - len(available),
    }
    for item in available:
        for key in ("matched", "additions", "removals", "unknown_eligibility", "conflicting_duplicates"):
            combined[key].extend(item.get(key) or [])
    return combined


def _mean(values: Sequence[float | None]) -> float | None:
    defined = [item for item in values if item is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def compare_pipeline_runs(runs: Sequence[Any]) -> dict[str, Any]:
    """Pairwise overlap of two or more runs. Undefined comparisons stay null.

    Selected-code overlap is Jaccard of ``raised_code_ids``. Relationship
    overlap is Jaccard of directed pairs, and of directed pair plus statement
    hash. Eligibility changes compare matched statements. Additions, removals,
    unknown eligibility, and conflicting duplicates are separate lists. A run
    with no relationship list is an unavailable comparison. The summary keeps
    the valid comparisons and reports unavailable comparisons against the
    number attempted. The field is null only when every comparison is unavailable.
    """
    items = list(runs or [])
    pairwise = []
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            pairwise.append(_pair_report(items[left_index], items[right_index], left_index, right_index))
    return {
        "run_count": len(items),
        "pairwise": pairwise,
        "summary": {
            "selected_code_jaccard": _mean([item["selected_code_jaccard"] for item in pairwise]),
            "directed_pair_jaccard": _mean([item["directed_pair_jaccard"] for item in pairwise]),
            "statement_jaccard": _mean([item["statement_jaccard"] for item in pairwise]),
            "eligibility_changes": _combine_eligibility(
                [item["eligibility_changes"] for item in pairwise]
            ),
        },
    }
