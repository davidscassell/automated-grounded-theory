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

from automated_grounded_theory.batching import estimate_tokens
from automated_grounded_theory.relationship_tiers import case_id_of, weak_stem_relationship

DEFAULT_CORE_EVIDENCE_TOKEN_CAP = 80

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
    """Tier 2 or higher, excluding a weak-stem hypothesis.

    The row and the pattern component use one rule. When the row has claim
    components, a withheld row is not graph-supported, even when an older
    pattern component still says it can support a core. A row with no claim
    components does not trust a stale ``downstream_eligible`` flag: the tier
    and the weak-stem check decide. Only the pattern component supports a
    core. An analyst interpretation does not. A 0.9.19 export that still
    stores the analyst reading as a row flag, and has no ``claims``, stays
    unsupported until it is reassessed.
    """
    claims = row.get("claims") if isinstance(row.get("claims"), Mapping) else None
    if claims is not None and row.get("downstream_eligible") is not True:
        return False
    tier = tier_of(row)
    if not (_usable(row) and tier is not None and tier >= 2):
        return False
    if weak_stem_relationship(row):
        return False
    if claims is not None:
        pattern = claims.get("pattern") if isinstance(claims.get("pattern"), Mapping) else {}
        return bool(pattern.get("supports_core")) and str(row.get("core_support_component") or "") == "pattern"
    flags = {str(item) for item in (row.get("flags") or [])}
    if "analyst_interpretation" in flags or str(row.get("attribution_source") or "") == "analyst_interpretation":
        return False
    return True


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
    support: dict[str, str] = {}
    for row in relationships:
        if not _usable(row):
            continue
        relationship_id = str(row.get("relationship_id") or "")
        if not relationship_id or relationship_id in tiers:
            continue
        ids.append(relationship_id)
        tiers[relationship_id] = tier_of(row)
        component = str(row.get("core_support_component") or "")
        if component:
            support[relationship_id] = component
    return {
        "relationship_ids": ids,
        "tiers": tiers,
        "support_components": support,
        "note": (
            "The relationship ids and tiers this stage read. "
            "support_components names the claim that supports a core, when one does. "
            "Later edits do not rewrite it."
        ),
    }


def _summary_text(
    definition: str,
    statements: Sequence[str],
    variation: str,
    counterevidence: str,
) -> str:
    return " ".join(
        piece
        for piece in (
            definition,
            "; ".join(item for item in statements if item),
            variation,
            counterevidence,
        )
        if piece
    )


def _fit_piece(text: str, token_cap: int, prefix: str) -> str:
    """The longest prefix of ``text`` whose summary still fits ``token_cap``."""
    raw = " ".join(str(text or "").split())
    if not raw or token_cap < 1:
        return ""
    joiner = " " if prefix else ""
    if estimate_tokens(prefix + joiner + raw) <= token_cap:
        return raw
    lo = 0
    hi = len(raw)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = raw[:mid].rstrip()
        if not candidate:
            hi = mid - 1
            continue
        trial = prefix + joiner + candidate
        if estimate_tokens(trial) <= token_cap:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def candidate_evidence_summary(
    *,
    definition: str,
    relationships: Sequence[Mapping[str, Any]],
    token_cap: int,
) -> dict[str, Any]:
    """Compact evidence for one core candidate, truncated to ``token_cap``.

    The summary keeps a category definition, up to two relationship
    statements, one variation note, and one counterexample quote. ``tokens``
    is the heuristic for that text. ``improves_selection`` stays null: this
    package has not measured whether the extra text changes a choice.
    """
    cap = max(1, int(token_cap or 1))
    statements: list[str] = []
    variation = ""
    counter = ""
    for row in relationships:
        if not isinstance(row, Mapping) or row.get("synthetic"):
            continue
        if str(row.get("disposition") or "").strip().casefold() == "rejected":
            continue
        statement = str(row.get("presented_statement") or row.get("statement") or "").strip()
        if statement and statement not in statements and len(statements) < 2:
            statements.append(statement)
        if not variation:
            variation = str(row.get("variation_note") or row.get("condition_note") or "").strip()
        if counter:
            continue
        for case in list(row.get("counterexamples") or []) + list(row.get("negative_cases") or []):
            if not isinstance(case, Mapping) or case.get("synthetic"):
                continue
            quote = str(case.get("quote") or case.get("text") or "").strip()
            if not quote:
                continue
            judgment = str(case.get("judgment") or "")
            if judgment == "contradicts" or not counter:
                counter = quote
            if judgment == "contradicts":
                break
    full = _summary_text(str(definition or "").strip(), statements, variation, counter)
    slots = []
    if statements:
        slots.append("statements")
    if variation:
        slots.append("variation")
    if counter:
        slots.append("counter")
    if str(definition or "").strip():
        slots.append("definition")
    weights = {"statements": 3, "variation": 2, "counter": 2, "definition": 4}
    budgets: dict[str, int] = {}
    used = 0
    # Statements, variation, and counterevidence are reserved before the definition
    # so a long definition cannot consume the whole cap.
    ordered_slots = [name for name in ("statements", "variation", "counter", "definition") if name in slots]
    total_weight = sum(weights[name] for name in ordered_slots) or 1
    for index, name in enumerate(ordered_slots):
        if index == len(ordered_slots) - 1:
            budgets[name] = max(0, cap - used)
            continue
        share = max(1, (cap * weights[name]) // total_weight)
        remaining_slots = len(ordered_slots) - index - 1
        if used + share + remaining_slots > cap:
            share = max(0, cap - used - remaining_slots)
        budgets[name] = share
        used += share
    statement_budget = int(budgets.get("statements") or 0)
    kept: list[str] = []
    statement_prefix = ""
    for statement in statements:
        if statement_budget < 1:
            break
        piece = _fit_piece(statement, statement_budget, statement_prefix)
        if not piece:
            continue
        joiner = "; " if statement_prefix else ""
        trial = statement_prefix + joiner + piece
        if estimate_tokens(trial) > statement_budget:
            continue
        kept.append(piece)
        statement_prefix = trial
    variation_out = _fit_piece(variation, int(budgets.get("variation") or 0), "")
    counter_out = _fit_piece(counter, int(budgets.get("counter") or 0), "")
    definition_out = _fit_piece(str(definition or ""), int(budgets.get("definition") or 0), "")
    text = _summary_text(definition_out, kept, variation_out, counter_out)
    while estimate_tokens(text) > cap:
        if definition_out:
            definition_out = _fit_piece(definition_out, max(0, estimate_tokens(definition_out) - 1), "")
        elif counter_out:
            counter_out = _fit_piece(counter_out, max(0, estimate_tokens(counter_out) - 1), "")
        elif variation_out:
            variation_out = _fit_piece(variation_out, max(0, estimate_tokens(variation_out) - 1), "")
        elif kept:
            kept.pop()
        else:
            break
        text = _summary_text(definition_out, kept, variation_out, counter_out)
    return {
        "definition": definition_out,
        "representative_statements": kept,
        "key_variation": variation_out,
        "counterevidence": counter_out,
        "source_definition": " ".join(str(definition or "").split()),
        "source_statements": list(statements),
        "source_variation": variation,
        "source_counterevidence": counter,
        "tokens": estimate_tokens(text),
        "token_cap": cap,
        "truncated": text != full,
        "improves_selection": None,
    }


def evidence_cap_for_match(
    match_count: int,
    *,
    per_request: int,
    per_step: int,
    configured: int = DEFAULT_CORE_EVIDENCE_TOKEN_CAP,
    richer: bool = False,
) -> int:
    """Per-candidate evidence cap for one match, from that match's budget.

    The original eligible-set size is not used. A match of 12 is not priced
    as if every eligible candidate were in the request. ``richer`` is the
    insufficient-evidence retry: a larger share of the same match budget.
    """
    count = max(1, int(match_count or 1))
    ceiling = max(1, int(configured or DEFAULT_CORE_EVIDENCE_TOKEN_CAP))
    limits = []
    for value in (per_request, per_step):
        try:
            number = int(value or 0)
        except (TypeError, ValueError):
            number = 0
        if number > 0:
            limits.append(number)
    allowance = min(limits) if limits else 0
    if allowance <= 0:
        return ceiling * (4 if richer else 1)
    share = 2 if richer else 4
    room = max(count, allowance // share)
    cap = max(1, room // count)
    if richer:
        return max(ceiling, min(cap, ceiling * 4))
    return min(cap, ceiling)


def refit_evidence_summary(summary: Mapping[str, Any] | None, token_cap: int) -> dict[str, Any] | None:
    """Fit a stored summary to a match cap, from the original source text.

    A definition, a representative statement, and counterevidence stay in the
    summary when the source has them and the cap can hold one token of each.
    """
    if not isinstance(summary, Mapping):
        return None
    definition = str(summary.get("source_definition") or summary.get("definition") or "")
    statements = [
        str(item)
        for item in (summary.get("source_statements") or summary.get("representative_statements") or [])
        if str(item)
    ]
    variation = str(summary.get("source_variation") or summary.get("key_variation") or "")
    counter = str(summary.get("source_counterevidence") or summary.get("counterevidence") or "")
    relationships = []
    if statements or variation or counter:
        relationships.append(
            {
                "statement": statements[0] if statements else "",
                "variation_note": variation,
                "counterexamples": [{"quote": counter, "judgment": "contradicts"}] if counter else [],
            }
        )
        if len(statements) > 1:
            relationships.append({"statement": statements[1]})
    fitted = candidate_evidence_summary(
        definition=definition,
        relationships=relationships,
        token_cap=max(1, int(token_cap or 1)),
    )
    # Keep the original source so a later retry can fit a larger cap.
    fitted["source_definition"] = definition
    fitted["source_statements"] = statements
    fitted["source_variation"] = variation
    fitted["source_counterevidence"] = counter
    present = []
    if definition:
        present.append("definition")
    if statements:
        present.append("statement")
    if counter:
        present.append("counterevidence")
    cap = max(1, int(token_cap or 1))
    if cap >= len(present) >= 1:
        if "definition" in present and not fitted["definition"]:
            fitted["definition"] = _fit_piece(definition, 1, "")
        if "statement" in present and not fitted["representative_statements"]:
            piece = _fit_piece(statements[0], 1, "")
            fitted["representative_statements"] = [piece] if piece else []
        if "counterevidence" in present and not fitted["counterevidence"]:
            fitted["counterevidence"] = _fit_piece(counter, 1, "")
        text = _summary_text(
            fitted["definition"],
            fitted["representative_statements"],
            fitted["key_variation"],
            fitted["counterevidence"],
        )
        while estimate_tokens(text) > cap and fitted["key_variation"]:
            fitted["key_variation"] = _fit_piece(fitted["key_variation"], max(0, estimate_tokens(fitted["key_variation"]) - 1), "")
            text = _summary_text(
                fitted["definition"],
                fitted["representative_statements"],
                fitted["key_variation"],
                fitted["counterevidence"],
            )
        fitted["tokens"] = estimate_tokens(text)
        fitted["truncated"] = text != _summary_text(definition, statements[:2], variation, counter)
    return fitted


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
