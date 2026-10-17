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


def export_gate_relationships(relationships: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """List accepted, withheld, and still-proposed relationships.

    A synthetic row is omitted. A row that has not been accepted or withheld
    is listed under ``proposed`` and is not scored as either. The input rows
    are not modified.
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
        }
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
    return {
        "relationship_id": _require_text(row, "relationship_id"),
        "rules": rules,
        "verdict": verdict,
        "adjudicator": _require_text(row, "adjudicator"),
        "timestamp": _require_text(row, "timestamp"),
        "reason": _require_text(row, "reason"),
    }


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
        "accepted_adjudicated": 0,
        "accepted_supported": 0,
        "withheld_adjudicated": 0,
        "withheld_supported": 0,
        "supported_adjudicated": 0,
    }


def _fill(counts: dict[str, Any]) -> dict[str, Any]:
    accepted = int(counts["accepted_adjudicated"])
    supported = int(counts["supported_adjudicated"])
    counts["accept_precision"] = None if accepted == 0 else round(int(counts["accepted_supported"]) / accepted, 4)
    counts["withheld_supported_recall"] = (
        None if supported == 0 else round(int(counts["withheld_supported"]) / supported, 4)
    )
    return counts


def score_gate_adjudications(
    exported: Mapping[str, Sequence[Mapping[str, Any]]],
    adjudications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Precision of accepts, and the share of human-supported rows that were withheld.

    ``accept_precision`` is supported accepts over adjudicated accepts.
    ``withheld_supported_recall`` is withheld-and-supported over every
    adjudicated supported relationship. Either value is null when its
    denominator is zero. Per-rule figures use the same rule and stay null
    on the same condition. A rule that fired on no exported row is omitted.
    """
    by_outcome: dict[str, dict[str, Any]] = {}
    for outcome in ("accepted", "withheld"):
        for row in exported.get(outcome) or []:
            if not isinstance(row, Mapping):
                continue
            relationship_id = str(row.get("relationship_id") or "")
            if not relationship_id:
                continue
            by_outcome[relationship_id] = {
                "outcome": outcome,
                "rules": [str(item) for item in (row.get("rules") or []) if str(item or "")],
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
        buckets = [overall] + [_rule_bucket(per_rule, rule) for rule in rules]
        for bucket in buckets:
            if supported:
                bucket["supported_adjudicated"] += 1
            if outcome == "accepted":
                bucket["accepted_adjudicated"] += 1
                if supported:
                    bucket["accepted_supported"] += 1
            elif outcome == "withheld":
                bucket["withheld_adjudicated"] += 1
                if supported:
                    bucket["withheld_supported"] += 1
    return {
        **_fill(overall),
        "by_rule": {rule: _fill(counts) for rule, counts in sorted(per_rule.items())},
        "unmatched_adjudication_ids": unmatched,
        "duplicate_relationship_ids": duplicates,
    }


def _rule_bucket(per_rule: dict[str, dict[str, Any]], rule: str) -> dict[str, Any]:
    bucket = per_rule.get(rule)
    if bucket is None:
        bucket = _blank_counts()
        per_rule[rule] = bucket
    return bucket
