"""Core selection and storyline checks against the relationship graph.

A core is graph-supported only when it is an endpoint of a relationship at
tier 2 or higher. A proposed link or a tier-1 hypothesis is shown to the
chooser and does not clear ``core_unsupported_by_graph``. Coverage shares are
null when the graph has no relationships. ``core_stability`` is null below
``CORE_STABILITY_MIN_N`` runs that name a core. That minimum is a design
choice for this package.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from automated_grounded_theory.relationship_tiers import case_id_of, weak_stem_relationship

CORE_STABILITY_MIN_N = 2
CORE_STABILITY_MIN_N_NOTE = (
    "A design choice for this package. It is not a quotation from the method literature. "
    "One run cannot show whether the core was stable."
)
_STATUS_TIER = {
    "pattern_supported": 2,
    "attribution": 3,
    "verified": 4,
}
_QUALIFIERS = (
    "however",
    "except",
    "unless",
    "not always",
    "only when",
    "condition",
    "contradict",
    "does not",
    "qualified",
    "variation",
)


def tier_of(row: Mapping[str, Any]) -> int | None:
    """Numeric tier, or the status label when the number was not stored."""
    if not isinstance(row, Mapping):
        return None
    if row.get("evidence_tier") is not None:
        try:
            return int(row.get("evidence_tier"))
        except (TypeError, ValueError):
            return None
    status = str(row.get("evidence_status") or row.get("evidence_label") or "")
    return _STATUS_TIER.get(status)


def _rejected(row: Mapping[str, Any]) -> bool:
    for key in ("disposition", "human_disposition", "model_disposition"):
        if str(row.get(key) or "").strip().casefold() in {"rejected", "merged"}:
            return True
    return False


def _usable(row: Mapping[str, Any]) -> bool:
    if not isinstance(row, Mapping) or row.get("synthetic"):
        return False
    if str(row.get("passage_id") or "") == "synthetic-counterexample":
        return False
    if _rejected(row):
        return False
    if "statement" in row and not str(row.get("statement") or "").strip():
        return False
    return True


def graph_supported(row: Mapping[str, Any]) -> bool:
    """Tier 2 or higher, excluding a weak-stem hypothesis and an analyst reading.

    ``downstream_eligible`` is not read. A row that was marked eligible and
    also carries ``weak_stem_anchor`` does not support a core. An analyst
    interpretation does not support a core either.
    """
    tier = tier_of(row)
    if not (_usable(row) and tier is not None and tier >= 2):
        return False
    flags = {str(item) for item in (row.get("flags") or [])}
    if "analyst_interpretation" in flags or str(row.get("attribution_source") or "") == "analyst_interpretation":
        return False
    return not weak_stem_relationship(row)


def contradicted(row: Mapping[str, Any]) -> bool:
    if "relationship_contradicted" in set(row.get("flags") or []):
        return True
    for case in row.get("counterexamples") or []:
        if not isinstance(case, Mapping) or case.get("synthetic"):
            continue
        if case.get("valid_test") and str(case.get("judgment") or "") == "contradicts":
            return True
    return False


def qualified(row: Mapping[str, Any]) -> bool:
    if "relationship_qualified" in set(row.get("flags") or []):
        return True
    if row.get("variation_note") or row.get("condition_note"):
        return True
    for case in row.get("counterexamples") or []:
        if not isinstance(case, Mapping) or case.get("synthetic"):
            continue
        if case.get("valid_test") and str(case.get("judgment") or "") == "qualifies":
            return True
    return False


def has_qualifier(text: str) -> bool:
    folded = f" {str(text or '').casefold()} "
    return any(token in folded for token in _QUALIFIERS)


def graph_snapshot(relationships: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Relationship ids and tiers a stage actually saw."""
    ids: list[str] = []
    tiers: dict[str, int | None] = {}
    for row in relationships:
        if not _usable(row):
            continue
        relationship_id = str(row.get("relationship_id") or "")
        if not relationship_id or relationship_id in tiers:
            continue
        ids.append(relationship_id)
        tiers[relationship_id] = tier_of(row)
    return {
        "relationship_ids": ids,
        "tiers": tiers,
        "note": "The relationship ids and tiers this stage read. Later edits do not rewrite it.",
    }


def named_core_id(run: Any) -> str:
    if isinstance(run, Mapping):
        metadata = run.get("metadata") if isinstance(run.get("metadata"), Mapping) else {}
        for key in ("researcher_core_id", "model_core_id", "core_id"):
            value = str((metadata or {}).get(key) or run.get(key) or "").strip()
            if value:
                return value
        return ""
    metadata = getattr(run, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("researcher_core_id", "model_core_id", "core_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    return ""


def core_agreement(runs: Sequence[Any], *, minimum_n: int = CORE_STABILITY_MIN_N) -> dict[str, Any]:
    """Pairwise agreement of named cores. Null below ``minimum_n`` named cores."""
    records = list(runs)
    named = [named_core_id(run) for run in records]
    named = [item for item in named if item]
    floor = max(2, int(minimum_n or CORE_STABILITY_MIN_N))
    if len(named) < floor:
        score = None
    else:
        scores = []
        for index, left in enumerate(named):
            for right in named[index + 1 :]:
                scores.append(1.0 if left == right else 0.0)
        score = sum(scores) / len(scores) if scores else None
    return {
        "name": "core_stability",
        "n": len(named),
        "runs": len(records),
        "cores": named,
        "named_cores": len(named),
        "score": score,
        "minimum_n": floor,
        "minimum_n_note": CORE_STABILITY_MIN_N_NOTE,
        "note": (
            "Pairwise agreement of the named cores in cores. "
            "n is the length of that list, including this run when it names a core. "
            "The score is null when fewer than minimum_n cores are named. "
            "Disagreeing cores score 0. They are not averaged into one core."
        ),
    }


def _categories(codebook: Mapping[str, Any] | Sequence[Any] | None) -> list[str]:
    if isinstance(codebook, Mapping):
        return [str(key) for key in codebook.keys()]
    ids = []
    for row in codebook or []:
        if isinstance(row, Mapping) and row.get("code_id"):
            ids.append(str(row.get("code_id")))
    return ids


def integration_report(
    core_id: str,
    codebook: Mapping[str, Any] | Sequence[Any] | None,
    relationships: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Who connects to the core, who is orphaned, and which links are contradicted.

    Category and respondent coverage are null when no relationship is in the
    graph. A zero share is used only when relationships exist and the core
    links none of them.
    """
    categories = _categories(codebook)
    rows = [row for row in relationships if _usable(row)]
    contradicted_ids = [str(row.get("relationship_id") or "") for row in rows if contradicted(row) and row.get("relationship_id")]
    neighbors: dict[str, set[str]] = {code_id: set() for code_id in categories}
    for row in rows:
        left = str(row.get("from_id") or "")
        right = str(row.get("to_id") or "")
        if not left or not right:
            continue
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    if not rows:
        return {
            "core_id": core_id or "",
            "direct_category_ids": [],
            "path_category_ids": [],
            "orphan_category_ids": sorted(categories),
            "contradicted_relationship_ids": [],
            "core_category_coverage": None,
            "core_respondent_coverage": None,
            "graph_snapshot": graph_snapshot(relationships),
            "note": "No relationships are in the graph. Coverage is undefined.",
        }
    if not core_id:
        orphans = sorted(code_id for code_id in categories if code_id not in {item for item, links in neighbors.items() if links})
        return {
            "core_id": "",
            "direct_category_ids": [],
            "path_category_ids": [],
            "orphan_category_ids": orphans,
            "contradicted_relationship_ids": contradicted_ids,
            "core_category_coverage": None,
            "core_respondent_coverage": None,
            "graph_snapshot": graph_snapshot(relationships),
            "note": "No core is selected. Coverage of a core is undefined.",
        }
    direct = sorted(neighbors.get(core_id, set()))
    seen = {core_id}
    frontier = list(direct)
    path: list[str] = []
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if current not in direct and current != core_id:
            path.append(current)
        for nxt in sorted(neighbors.get(current, set())):
            if nxt not in seen:
                frontier.append(nxt)
    linked = set(direct) | set(path)
    if core_id in categories or core_id in neighbors:
        linked_with_core = set(linked)
        if core_id in categories:
            linked_with_core.add(core_id)
        denominator = len(categories) if categories else len(neighbors)
        category_share = round(len(linked_with_core & set(categories or neighbors)) / denominator, 4) if denominator else None
    else:
        category_share = 0.0 if categories else None
    orphans = sorted(code_id for code_id in categories if code_id != core_id and code_id not in linked)
    passage_rows: list[Mapping[str, Any]]
    if isinstance(passages, Mapping):
        passage_rows = [row for row in passages.values() if isinstance(row, Mapping)]
    else:
        passage_rows = [row for row in (passages or []) if isinstance(row, Mapping)]
    by_passage = {str(row.get("passage_id") or ""): row for row in passage_rows if row.get("passage_id")}
    respondents = {case_id_of(row) for row in passage_rows if case_id_of(row)}
    connected_ids = set(linked) | ({core_id} if core_id else set())
    linked_respondents: set[str] = set()
    for row in rows:
        ends = {str(row.get("from_id") or ""), str(row.get("to_id") or "")}
        if not ends & connected_ids:
            continue
        for key in ("passage_id",):
            passage = by_passage.get(str(row.get(key) or ""))
            if passage is not None and case_id_of(passage):
                linked_respondents.add(case_id_of(passage))
        for case in row.get("supporting_cases") or []:
            if isinstance(case, Mapping) and case.get("case_id"):
                linked_respondents.add(str(case.get("case_id")))
    if not respondents:
        respondent_share = None
    else:
        respondent_share = round(len(linked_respondents & respondents) / len(respondents), 4)
    return {
        "core_id": core_id,
        "direct_category_ids": direct,
        "path_category_ids": sorted(path),
        "orphan_category_ids": orphans,
        "contradicted_relationship_ids": contradicted_ids,
        "core_category_coverage": category_share,
        "core_respondent_coverage": respondent_share,
        "graph_snapshot": graph_snapshot(relationships),
        "note": (
            "core_category_coverage is the share of categories linked to the core, including the core. "
            "core_respondent_coverage is the share of respondents cited by those links. "
            "Both are null when the graph has no relationships."
        ),
    }
