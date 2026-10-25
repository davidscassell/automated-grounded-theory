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


def _eligibility_changes(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Matched eligibility flips, plus additions, removals, and unknown rows.

    A statement missing from one run does not hide a flip on a statement that
    both runs have. A missing boolean is unknown. It is not treated as a flip
    and it does not discard the rest of the comparison. ``None`` means there
    are no relationship rows to compare.
    """
    left = {_statement_key(row): _eligibility(row) for row in left_rows}
    right = {_statement_key(row): _eligibility(row) for row in right_rows}
    if not left and not right:
        return None
    matched = []
    unknown = []
    for key in sorted(set(left) & set(right)):
        left_value = left[key]
        right_value = right[key]
        if left_value is None or right_value is None:
            unknown.append(_eligibility_row(key, left_value, right_value))
        elif left_value != right_value:
            matched.append(_eligibility_row(key, left_value, right_value))
    additions = [
        _eligibility_row(key, None, right[key])
        for key in sorted(set(right) - set(left))
    ]
    removals = [
        _eligibility_row(key, left[key], None)
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


def compare_pipeline_runs(runs: Sequence[Any]) -> dict[str, Any]:
    """Pairwise overlap of two or more runs. Undefined comparisons stay null.

    Selected-code overlap is Jaccard of ``raised_code_ids``. Relationship
    overlap is Jaccard of directed pairs, and of directed pair plus statement
    hash. Eligibility changes compare matched statements. Additions, removals, and
    unknown eligibility are separate lists. The field is null when there are
    no relationship rows to compare.
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
            "eligibility_changes": None
            if not pairwise or any(item["eligibility_changes"] is None for item in pairwise)
            else [change for item in pairwise for change in item["eligibility_changes"]["matched"]],
        },
    }
