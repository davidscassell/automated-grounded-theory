"""Export relationships for human adjudication of accept and withhold gates.

The gates themselves are unchanged. This module only reads the flags,
endpoint reasons, and counterexample reclassifications already stored on a
relationship, then scores a separate adjudication file. A metric with no
adjudicated rows stays null.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

_ACCEPTED = frozenset({"accepted", "verified"})
_WITHHELD = frozenset({"rejected", "provisional", "merged"})
_VERDICTS = frozenset({"supported", "unsupported"})

# Flags and reasons already written by the axial gates. An unlisted flag is
# reported under its own name so a later gate is still visible to a reader.
_FLAG_RULES = {
    "weak_stem_anchor": "stem_only",
    "wording_exceeds_tier": "consequence_wording",
    "role_evidence_exceeds_tier": "role_evidence",
    "reviewer_causal_claim_mismatch": "consequence_wording",
    "memo_wording_exceeds_tier": "consequence_wording",
    "storyline_wording_exceeds_tier": "consequence_wording",
    "acceptance_withheld": "acceptance_withheld",
    "endpoint_evidence_partial": "endpoint_unanchored",
    "unanchored_quote": "unanchored_quote",
    "axial_test_invalid": "axial_test_invalid",
    "rationale_indicates_irrelevance": "rationale_indicates_irrelevance",
    "model_cannot_verify": "model_cannot_verify",
}
_ANCHOR_RULES = {
    "stem_only": "stem_only",
    "generic_stem_only": "generic_stem_only",
    "missing_quote": "missing_quote",
    "unknown_category": "unknown_category",
    "missing_category_label": "missing_category_label",
}
_ABSENCE_RULE = "absence_reclassified"
# Flags that block a row from downstream use. Disposition can still be accepted.
_WITHHOLDING_FLAGS = (
    "causal_wording_withheld",
    "acceptance_withheld",
    "verification_withheld",
    "relationship_rejected",
    "relationship_merged",
)


class GateEvaluationError(ValueError):
    """An adjudication file is missing a required field or uses an unknown verdict."""


def _rules_for(row: Mapping[str, Any]) -> list[str]:
    found: list[str] = []

    def _add(rule: str) -> None:
        if rule and rule not in found:
            found.append(rule)

    for flag in row.get("flags") or []:
        name = str(flag or "")
        if not name:
            continue
        _add(_FLAG_RULES.get(name, name))
    for anchor in row.get("endpoint_anchoring") or []:
        if not isinstance(anchor, Mapping):
            continue
        reason = str(anchor.get("reason") or "")
        if reason:
            _add(_ANCHOR_RULES.get(reason, reason))
    for case in row.get("counterexamples") or []:
        if not isinstance(case, Mapping) or case.get("synthetic"):
            continue
        reason = str(case.get("reclassification_reason") or "")
        if reason == "absence_is_not_contradiction":
            _add(_ABSENCE_RULE)
        elif str(case.get("reason") or "") == "absence_is_not_contradiction":
            _add(_ABSENCE_RULE)
    return found


def _outcome(row: Mapping[str, Any], rules: Sequence[str]) -> str | None:
    if row.get("synthetic"):
        return None
    disposition = str(row.get("disposition") or "").strip().casefold()
    if disposition in _ACCEPTED:
        return "accepted"
    if disposition in _WITHHELD or "acceptance_withheld" in set(rules):
        return "withheld"
    if disposition in {"", "proposed", "revised"}:
        return "proposed"
    return "proposed"


def _downstream_eligible(row: Mapping[str, Any]) -> bool | None:
    """Copy a boolean eligibility. A missing or non-boolean value stays unknown."""
    if "downstream_eligible" not in row:
        return None
    value = row.get("downstream_eligible")
    if isinstance(value, bool):
        return value
    return None


def _withholding_flags(row: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    for flag in row.get("flags") or []:
        name = str(flag or "")
        if name in _WITHHOLDING_FLAGS and name not in found:
            found.append(name)
    return found


def export_gate_relationships(relationships: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """List accepted, withheld, and still-proposed relationships.

    A synthetic row is omitted. A row that has not been accepted or withheld
    is listed under ``proposed`` and is not scored as either. ``disposition``
    is the reviewer disposition. ``downstream_eligible`` is copied only when
    the row stores a boolean, and the withholding flags are copied beside it.
    The input rows are not modified.
    """
    accepted: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    proposed: list[dict[str, Any]] = []
    for row in relationships:
        if not isinstance(row, Mapping):
            continue
        rules = _rules_for(row)
        outcome = _outcome(row, rules)
        if outcome is None:
            continue
        item = {
            "relationship_id": str(row.get("relationship_id") or ""),
            "disposition": str(row.get("disposition") or ""),
            "rules": list(rules),
            "withholding_flags": _withholding_flags(row),
        }
        eligible = _downstream_eligible(row)
        if eligible is not None:
            item["downstream_eligible"] = eligible
        if outcome == "accepted":
            accepted.append(item)
        elif outcome == "withheld":
            withheld.append(item)
        else:
            proposed.append(item)
    return {"accepted": accepted, "withheld": withheld, "proposed": proposed}


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise GateEvaluationError(f"An adjudication needs a non-empty {field}.")
    return value


def _one_adjudication(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise GateEvaluationError("Each adjudication must be an object.")
    verdict = _require_text(row, "verdict").casefold()
    if verdict not in _VERDICTS:
        raise GateEvaluationError("verdict must be 'supported' or 'unsupported'.")
    rules = [str(item) for item in (row.get("rules") or []) if str(item or "").strip()]
    item = {
        "relationship_id": _require_text(row, "relationship_id"),
        "rules": rules,
        "verdict": verdict,
        "adjudicator": _require_text(row, "adjudicator"),
        "timestamp": _require_text(row, "timestamp"),
        "reason": _require_text(row, "reason"),
    }
    # Optional notes a reviewer file may carry. Scoring reads the export, not these.
    eligible = _downstream_eligible(row)
    if eligible is not None:
        item["downstream_eligible"] = eligible
    flags = row.get("withholding_flags")
    if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)):
        item["withholding_flags"] = [str(flag) for flag in flags if str(flag or "").strip()]
    return item


def load_adjudications(source: str | Path | Sequence[Any] | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read adjudication objects from a list or a JSON file.

    A file may be a list, or an object with an ``adjudications`` list.
    This does not create verdicts.
    """
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = source
    if isinstance(payload, Mapping):
        rows = payload.get("adjudications")
    else:
        rows = payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise GateEvaluationError("Adjudications must be a list.")
    return [_one_adjudication(row) for row in rows]


def _blank_counts() -> dict[str, Any]:
    return {
        "accept_precision": None,
        "withheld_supported_recall": None,
        "downstream_precision": None,
        "downstream_withheld_supported_recall": None,
        "accepted_adjudicated": 0,
        "accepted_supported": 0,
        "withheld_adjudicated": 0,
        "withheld_supported": 0,
        "supported_adjudicated": 0,
        "downstream_eligible_adjudicated": 0,
        "downstream_eligible_supported": 0,
        "downstream_withheld_supported": 0,
        "downstream_adjudicated": 0,
        "downstream_eligibility_known_adjudicated": 0,
        "downstream_supported_known": 0,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _fill(counts: dict[str, Any]) -> dict[str, Any]:
    accepted = int(counts["accepted_adjudicated"])
    supported = int(counts["supported_adjudicated"])
    eligible = int(counts["downstream_eligible_adjudicated"])
    adjudicated = int(counts["downstream_adjudicated"])
    known_rows = int(counts["downstream_eligibility_known_adjudicated"])
    supported_known = int(counts["downstream_supported_known"])
    eligible_supported = int(counts["downstream_eligible_supported"])
    withheld_supported = int(counts["downstream_withheld_supported"])
    counts["accept_precision"] = _rate(int(counts["accepted_supported"]), accepted)
    counts["withheld_supported_recall"] = _rate(int(counts["withheld_supported"]), supported)
    # Full-population rates stay null unless every row in that population has a
    # boolean downstream_eligible. A known-only rate is the same fraction on
    # the rows whose eligibility is known, and it carries that denominator.
    precision_identified = adjudicated > 0 and known_rows == adjudicated
    counts["downstream_precision"] = _rate(eligible_supported, eligible) if precision_identified else None
    counts["downstream_precision_known"] = _rate(eligible_supported, eligible)
    counts["downstream_precision_known_denominator"] = eligible
    counts["downstream_precision_coverage"] = _rate(known_rows, adjudicated)
    recall_identified = supported > 0 and supported_known == supported
    counts["downstream_withheld_supported_recall"] = (
        _rate(withheld_supported, supported) if recall_identified else None
    )
    counts["downstream_withheld_supported_recall_known"] = _rate(withheld_supported, supported_known)
    counts["downstream_withheld_supported_recall_known_denominator"] = supported_known
    counts["downstream_withheld_supported_recall_coverage"] = _rate(supported_known, supported)
    return counts


def _public_counts(counts: dict[str, Any]) -> dict[str, Any]:
    return _fill(counts)


def score_gate_adjudications(
    exported: Mapping[str, Sequence[Mapping[str, Any]]],
    adjudications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Precision of accepts, and the share of human-supported rows that were withheld.

    ``accept_precision`` is supported accepts over adjudicated accepts.
    ``withheld_supported_recall`` is withheld-and-supported over every
    adjudicated supported relationship. Those two rates follow disposition.

    ``downstream_precision`` and ``downstream_withheld_supported_recall`` are
    full-population rates. Each stays null unless every row in its population
    has a boolean ``downstream_eligible``. A missing field is unknown, not
    eligible and not withheld. An unknown row in the denominator would make
    the fraction a bound, so that unidentified rate is null rather than a
    point estimate.

    The precision population is every adjudicated row, because an unknown row
    might be eligible. ``downstream_precision`` is supported rows among rows
    whose export says ``downstream_eligible`` is true, and only when
    eligibility is known for every adjudicated row. ``downstream_precision_known``
    is that same fraction among rows already known to be eligible. Its
    denominator is ``downstream_precision_known_denominator``.
    ``downstream_precision_coverage`` is adjudicated rows with known
    eligibility over adjudicated rows.

    The recall population is adjudicated supported rows.
    ``downstream_withheld_supported_recall`` is supported rows whose export
    says ``downstream_eligible`` is false, over every adjudicated supported
    row, and only when every one of those rows has known eligibility. An
    unsupported row does not identify it.
    ``downstream_withheld_supported_recall_known`` uses only the supported
    rows whose eligibility is known. Its denominator is
    ``downstream_withheld_supported_recall_known_denominator``.
    ``downstream_withheld_supported_recall_coverage`` is those known supported
    rows over every adjudicated supported row.

    A zero denominator stays null, including a known-only rate with no known
    rows. Coverage is null when its own denominator is zero. An export that
    does not carry ``downstream_eligible`` leaves the rates null. Per-rule
    figures use the same rule. A rule that fired on no exported row is
    omitted. Optional fields on an adjudication are not the source of these
    rates.
    """
    by_outcome: dict[str, dict[str, Any]] = {}
    for outcome in ("accepted", "withheld"):
        for row in exported.get(outcome) or []:
            if not isinstance(row, Mapping):
                continue
            relationship_id = str(row.get("relationship_id") or "")
            if not relationship_id:
                continue
            known = "downstream_eligible" in row and isinstance(row.get("downstream_eligible"), bool)
            by_outcome[relationship_id] = {
                "outcome": outcome,
                "rules": [str(item) for item in (row.get("rules") or []) if str(item or "")],
                "downstream_known": known,
                "downstream_eligible": row.get("downstream_eligible") if known else None,
            }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in adjudications:
        item = _one_adjudication(row)
        grouped.setdefault(item["relationship_id"], []).append(item)
    duplicates = sorted(key for key, items in grouped.items() if len(items) > 1)
    overall = _blank_counts()
    per_rule: dict[str, dict[str, Any]] = {}
    for exported_row in by_outcome.values():
        for rule in exported_row["rules"]:
            _rule_bucket(per_rule, rule)
    unmatched: list[str] = []
    for relationship_id, items in grouped.items():
        if len(items) != 1:
            continue
        item = items[0]
        exported_row = by_outcome.get(relationship_id)
        if exported_row is None:
            unmatched.append(relationship_id)
            continue
        supported = item["verdict"] == "supported"
        outcome = exported_row["outcome"]
        rules = list(exported_row["rules"])
        known = bool(exported_row.get("downstream_known"))
        eligible = known and exported_row.get("downstream_eligible") is True
        blocked = known and exported_row.get("downstream_eligible") is False
        buckets = [overall] + [_rule_bucket(per_rule, rule) for rule in rules]
        for bucket in buckets:
            bucket["downstream_adjudicated"] += 1
            if known:
                bucket["downstream_eligibility_known_adjudicated"] += 1
            if supported:
                bucket["supported_adjudicated"] += 1
                if known:
                    bucket["downstream_supported_known"] += 1
                if blocked:
                    bucket["downstream_withheld_supported"] += 1
            if eligible:
                bucket["downstream_eligible_adjudicated"] += 1
                if supported:
                    bucket["downstream_eligible_supported"] += 1
            if outcome == "accepted":
                bucket["accepted_adjudicated"] += 1
                if supported:
                    bucket["accepted_supported"] += 1
            elif outcome == "withheld":
                bucket["withheld_adjudicated"] += 1
                if supported:
                    bucket["withheld_supported"] += 1
    return {
        **_public_counts(overall),
        "by_rule": {rule: _public_counts(counts) for rule, counts in sorted(per_rule.items())},
        "unmatched_adjudication_ids": unmatched,
        "duplicate_relationship_ids": duplicates,
    }


def _rule_bucket(per_rule: dict[str, dict[str, Any]], rule: str) -> dict[str, Any]:
    bucket = per_rule.get(rule)
    if bucket is None:
        bucket = _blank_counts()
        per_rule[rule] = bucket
    return bucket
