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


# Flag and rule names a reader can tally. These are not a partition: one row
# can carry several, and an eligible row can carry an inspect flag.
_WITHHELD_REASON_KEYS = (
    "shared_quote_inspect",
    "frequency_qualified",
    "frequency_negation",
    "omitted_oversized",
    "participant_denies_link",
    "negation_does_not_reject_link",
)
_INSPECT_FLAG_KEYS = (
    "shared_quote_inspect",
    "frequency_qualified",
    "frequency_negation",
    "negation_does_not_reject_link",
)
# Reporting convention for ``primary_withholding_cause``. The first applicable
# name wins, so the primary counts are mutually exclusive. Substantive limits
# outrank a reviewer decision. ``reviewer_policy_rejection`` is a policy-only
# rejection of tier-1 evidence, not an independent judgment of correctness.
# ``other_withheld`` is the residual so an explained row is never left uncounted.
# ``withholding_reasons`` is the overlapping diagnostic list and does not sum
# to the explained rows. The primary counts do.
WITHHOLDING_CAUSES = (
    "endpoint_unreviewed_by_focus",
    "endpoint_not_selected_by_focus",
    "participant_denies_link",
    "partial_endpoint_evidence",
    "wording_withheld",
    "insufficient_cases",
    "omitted_oversized",
    "llm_reviewer_rejected",
    "reviewer_policy_rejection",
    "incomplete_condition",
    "hypothetical_link",
    "no_review_decision",
    "other_withheld",
)
PRIMARY_WITHHOLDING_CAUSE = (
    "primary_withholding_cause is a reporting convention. "
    "The first applicable cause in WITHHOLDING_CAUSES wins. "
    "withholding_cause_counts are mutually exclusive and sum to the explained rows. "
    "withholding_reason_counts overlap and do not sum to that total. "
    "An explained row is disposition-withheld, downstream_eligible false, "
    "unreviewed at an endpoint, or still proposed with a known case or tier shortfall. "
    "A proposed row with enough cases and unknown eligibility is not explained."
)
_WORDING_CAUSE = frozenset({
    "causal_wording_withheld",
    "wording_exceeds_tier",
    "role_evidence_exceeds_tier",
    "consequence_wording",
    "role_evidence",
})
_PARTIAL_CAUSE = frozenset({
    "endpoint_evidence_partial",
    "endpoint_unanchored",
    "unanchored_quote",
    "missing_quote",
})


def _tier_number(row: Mapping[str, Any]) -> int | None:
    tier = row.get("evidence_tier")
    if isinstance(tier, bool) or not isinstance(tier, int):
        return None
    return tier


def _reviewer_rejected(row: Mapping[str, Any]) -> bool:
    disposition = str(row.get("disposition") or "").strip().casefold()
    model_disposition = str(row.get("model_disposition") or "").strip().casefold()
    reviewer_disposition = str(row.get("reviewer_disposition") or "").strip().casefold()
    rejected = "rejected" in {disposition, model_disposition, reviewer_disposition}
    decided = (
        str(row.get("review_actor") or "") == "llm"
        or row.get("reviewer_decided") is True
        or model_disposition == "rejected"
        or reviewer_disposition == "rejected"
    )
    return rejected and decided


def _cause_names(row: Mapping[str, Any], rules: Sequence[str] | None) -> set[str]:
    flags = {str(item) for item in (row.get("flags") or []) if str(item)}
    return flags | {str(item) for item in (rules or []) if str(item)}


def endpoint_review_status(row: Mapping[str, Any]) -> str:
    """Whether focused review limited the endpoints of this row.

    ``unreviewed`` is a flag, an eligibility reason, or a limited selection
    whose endpoint is missing from the reviewed ids. ``reviewed`` is a limited
    selection whose endpoints were both reviewed. Anything else is
    ``not_limited``.
    """
    names = _cause_names(row, ())
    reason = str(row.get("eligibility_reason") or "")
    focus = row.get("focus_review") if isinstance(row.get("focus_review"), Mapping) else {}
    limits = bool(focus.get("limits_eligibility"))
    reviewed = {str(item) for item in (focus.get("reviewed_code_ids") or []) if str(item)}
    endpoints = [str(row.get("from_id") or ""), str(row.get("to_id") or "")]
    endpoints = [item for item in endpoints if item]
    flagged = "endpoint_unreviewed_by_focus" in names or reason == "endpoint_unreviewed_by_focus"
    if limits and endpoints and reviewed and all(item in reviewed for item in endpoints):
        return "reviewed"
    if flagged or (limits and any(item not in reviewed for item in endpoints)):
        return "unreviewed"
    return "not_limited"


def _pattern_minimum(row: Mapping[str, Any]) -> int:
    minimum = row.get("pattern_min_cases")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        return 2
    return minimum


def _known_case_count(row: Mapping[str, Any]) -> int | None:
    """Supporting-case count when the row records one. A missing list stays unknown."""
    counted = row.get("cases_with_both_endpoints")
    if isinstance(counted, int) and not isinstance(counted, bool):
        return counted
    cases = row.get("supporting_cases")
    if isinstance(cases, list):
        return len([item for item in cases if isinstance(item, Mapping)])
    return None


def _case_count_below_minimum(row: Mapping[str, Any]) -> bool:
    count = _known_case_count(row)
    if count is None:
        return False
    return count < _pattern_minimum(row)


def _proposed_case_or_tier_short(row: Mapping[str, Any]) -> bool:
    """A proposed row whose cases or tier are known to be below the pattern bar."""
    if _case_count_below_minimum(row):
        return True
    tier = _tier_number(row)
    return tier is not None and tier < 2


def _row_explained(row: Mapping[str, Any], rules: Sequence[str]) -> bool:
    """True when this row is ineligible and the cause counts must include it.

    Withheld rows and rows whose ``downstream_eligible`` is false are explained.
    So is an unreviewed endpoint, and a proposed row with a known case or tier
    shortfall, even when eligibility was never stamped false. A proposed row
    that still has enough cases and unknown eligibility is not explained.
    ``downstream_eligible`` true is not explained unless the disposition is withheld.
    """
    outcome = _outcome(row, rules)
    if outcome is None:
        return False
    if outcome == "withheld":
        return True
    eligible = _downstream_eligible(row)
    if eligible is False:
        return True
    if eligible is True:
        return False
    if endpoint_review_status(row) == "unreviewed":
        return True
    if outcome == "proposed" and _proposed_case_or_tier_short(row):
        return True
    return False


def withholding_reasons(row: Mapping[str, Any], rules: Sequence[str] | None = None) -> list[str]:
    """Every applicable diagnostic reason. Empty when the row is not explained.

    A policy-only rejection of evidence below tier 2 is
    ``reviewer_policy_rejection`` and is not also ``llm_reviewer_rejected``.
    A tier-2 rejection is ``llm_reviewer_rejected``. Inspect flags are not reasons.
    """
    names = _cause_names(row, rules)
    if not _row_explained(row, rules or ()):
        return []
    found: list[str] = []

    def add(name: str) -> None:
        if name not in found:
            found.append(name)

    if (
        "endpoint_unreviewed_by_focus" in names
        or str(row.get("eligibility_reason") or "") == "endpoint_unreviewed_by_focus"
    ):
        add("endpoint_unreviewed_by_focus")
    if (
        "endpoint_not_selected_by_focus" in names
        or str(row.get("eligibility_reason") or "") == "endpoint_not_selected_by_focus"
    ):
        add("endpoint_not_selected_by_focus")
    if "participant_denies_link" in names:
        add("participant_denies_link")
    ends = [item for item in (row.get("endpoint_anchoring") or []) if isinstance(item, Mapping)]
    unanchored = bool(ends) and any(not item.get("anchored") for item in ends)
    if names & _PARTIAL_CAUSE or unanchored or "acceptance_withheld" in names:
        add("partial_endpoint_evidence")
    if names & _WORDING_CAUSE:
        add("wording_withheld")
    tier_n = _tier_number(row)
    if tier_n is None or tier_n < 2 or _case_count_below_minimum(row):
        add("insufficient_cases")
    if "omitted_oversized" in names:
        add("omitted_oversized")
    if _reviewer_rejected(row):
        if tier_n is not None and tier_n >= 2:
            add("llm_reviewer_rejected")
        else:
            add("reviewer_policy_rejection")
    if "condition_incomplete" in names or row.get("condition_incomplete") is True:
        add("incomplete_condition")
    if "participant_link_hypothetical" in names or "hypothetical_link" in names:
        add("hypothetical_link")
    if "no_review_decision" in names or str(row.get("eligibility_reason") or "") == "no_review_decision":
        add("no_review_decision")
    ordered = [name for name in WITHHOLDING_CAUSES if name in found and name != "other_withheld"]
    if not ordered:
        ordered = ["other_withheld"]
    return ordered


def primary_withholding_cause(row: Mapping[str, Any], rules: Sequence[str] | None = None) -> str | None:
    """The documented primary cause, or null when the row is not explained.

    See ``PRIMARY_WITHHOLDING_CAUSE``. The overlapping list is
    ``withholding_reasons``.
    """
    reasons = withholding_reasons(row, rules)
    if not reasons:
        return None
    return reasons[0]


def relationship_assessment(row: Mapping[str, Any]) -> dict[str, Any]:
    """One assessment for the gate export and the run export.

    ``evidence_tier`` is the stored tier. Tier 2 can coexist with
    ineligibility. ``reviewer_disposition`` is copied and is not invented from
    ``disposition``. ``downstream_eligible`` is copied only when it is a boolean.
    """
    rules = _rules_for(row)
    reasons = withholding_reasons(row, rules)
    primary = reasons[0] if reasons else None
    reviewer = row.get("reviewer_disposition")
    return {
        "evidence_tier": _tier_number(row),
        "reviewer_disposition": None if reviewer in (None, "") else reviewer,
        "endpoint_review_status": endpoint_review_status(row),
        "downstream_eligible": _downstream_eligible(row),
        "withholding_reasons": reasons,
        "primary_withholding_cause": primary,
        "withholding_cause": primary,
    }


def stamp_relationship_assessment(row: dict[str, Any]) -> dict[str, Any]:
    """Write :func:`relationship_assessment` onto a row. Returns that assessment.

    The fields are copied onto the row, and the same object is stored under
    ``relationship_assessment``.
    """
    assessment = relationship_assessment(row)
    row.update(assessment)
    row["relationship_assessment"] = dict(assessment)
    return assessment


def export_gate_relationships(relationships: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """List accepted, withheld, and still-proposed relationships.

    A synthetic row is omitted. ``accepted`` follows disposition
    (``accepted`` or ``verified``). ``accepted_downstream`` is the accepted
    rows whose ``downstream_eligible`` is true. An ineligible row stays in
    ``accepted`` when that is its disposition, and it is not an accepted
    downstream result. ``withheld_reason_counts`` tallies withholding flags
    and rules. Inspect flags such as ``shared_quote_inspect`` are not reasons.
    ``withholding_cause_counts`` is one primary cause per explained row.
    An explained row is disposition-withheld, ``downstream_eligible`` false,
    unreviewed at an endpoint, or still proposed with a known case or tier
    shortfall. Those counts sum to the explained rows.
    ``withholding_reason_counts`` overlaps and does not sum to that total.
    ``inspect_flag_counts`` is separate. The maps are null when there is
    nothing to count, and zeros when rows were counted but none are explained.
    The input rows are not modified. Fields come from
    :func:`relationship_assessment`.
    """
    accepted: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    proposed: list[dict[str, Any]] = []
    counted: list[tuple[str, set[str]]] = []
    explained: list[dict[str, Any]] = []
    for row in relationships:
        if not isinstance(row, Mapping):
            continue
        rules = _rules_for(row)
        outcome = _outcome(row, rules)
        if outcome is None:
            continue
        assessment = relationship_assessment(row)
        item = {
            "relationship_id": str(row.get("relationship_id") or ""),
            "disposition": str(row.get("disposition") or ""),
            "rules": list(rules),
            "withholding_flags": _withholding_flags(row),
            **assessment,
            "relationship_assessment": dict(assessment),
        }
        flags = {str(flag) for flag in (row.get("flags") or []) if str(flag)}
        counted.append((outcome, flags | set(rules)))
        if assessment.get("primary_withholding_cause"):
            explained.append(item)
        if outcome == "accepted":
            accepted.append(item)
        elif outcome == "withheld":
            withheld.append(item)
        else:
            proposed.append(item)
    accepted_downstream = [item for item in accepted if item.get("downstream_eligible") is True]
    if not counted:
        withheld_reason_counts = None
        inspect_flag_counts = None
        withholding_cause_counts = None
        withholding_reason_counts = None
    else:
        withheld_reason_counts = {name: 0 for name in _WITHHELD_REASON_KEYS}
        inspect_flag_counts = {name: 0 for name in _INSPECT_FLAG_KEYS}
        withholding_cause_counts = {name: 0 for name in WITHHOLDING_CAUSES}
        withholding_reason_counts = {name: 0 for name in WITHHOLDING_CAUSES}
        for outcome, names in counted:
            if outcome == "withheld":
                for name in _WITHHELD_REASON_KEYS:
                    if name in names and name not in _INSPECT_FLAG_KEYS:
                        withheld_reason_counts[name] += 1
            for name in _INSPECT_FLAG_KEYS:
                if name in names:
                    inspect_flag_counts[name] += 1
        for item in explained:
            cause = str(item.get("primary_withholding_cause") or "other_withheld")
            withholding_cause_counts[cause] = withholding_cause_counts.get(cause, 0) + 1
            for reason in item.get("withholding_reasons") or []:
                withholding_reason_counts[str(reason)] = withholding_reason_counts.get(str(reason), 0) + 1
    return {
        "accepted": accepted,
        "accepted_downstream": accepted_downstream,
        "withheld": withheld,
        "proposed": proposed,
        "explained_rows": len(explained) if counted else None,
        "primary_withholding_convention": PRIMARY_WITHHOLDING_CAUSE if counted else None,
        "withheld_reason_counts": withheld_reason_counts,
        "withholding_cause_counts": withholding_cause_counts,
        "withholding_reason_counts": withholding_reason_counts,
        "inspect_flag_counts": inspect_flag_counts,
    }


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
