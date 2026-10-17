"""New-codes-per-batch curve.

The curve is a report. It is not a decision that theoretical saturation has
been reached. Charmaz treats saturation as a claim the researcher has to earn
at the level of categories, not as a stopping rule a counter can trip.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def saturation_point(
    *,
    batch: int,
    passages_in_batch: int,
    new_codes: int,
    cumulative_passages: int,
    cumulative_codes: int,
    cumulative_codes_pre_merge: int | None = None,
    new_codes_post_merge: int | None = None,
) -> dict:
    """One window of the saturation curve.

    ``new_codes`` is the count before merges. ``new_codes_post_merge`` is the
    net change in codebook size after merges, against the previous window.
    ``cumulative_codes`` is the codebook size after merges.
    """
    rate = (new_codes / passages_in_batch) if passages_in_batch else 0.0
    pre_total = cumulative_codes if cumulative_codes_pre_merge is None else cumulative_codes_pre_merge
    post_new = new_codes if new_codes_post_merge is None else new_codes_post_merge
    post_rate = (post_new / passages_in_batch) if passages_in_batch else 0.0
    return {
        "batch": batch,
        "passages_in_batch": passages_in_batch,
        "new_codes": new_codes,
        "new_codes_per_passage": rate,
        "new_codes_pre_merge": new_codes,
        "new_codes_per_passage_pre_merge": rate,
        "new_codes_post_merge": post_new,
        "new_codes_per_passage_post_merge": post_rate,
        "new_codes_pre_alias": new_codes,
        "new_codes_post_alias": post_new,
        "new_codes_per_passage_pre_alias": rate,
        "new_codes_per_passage_post_alias": post_rate,
        "cumulative_passages": cumulative_passages,
        "cumulative_codes": cumulative_codes,
        "cumulative_codes_pre_merge": pre_total,
        "cumulative_codes_post_merge": cumulative_codes,
    }


def _normalized_code_distribution(frequencies: Mapping[str, int]) -> dict[str, float] | None:
    """Shares of this document's assignment counts. An empty document is undefined."""
    total = sum(max(0, int(value)) for value in frequencies.values())
    if total <= 0:
        return None
    return {str(key): max(0, int(value)) / total for key, value in frequencies.items()}


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Half the L1 distance between two distributions on the same code ids."""
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


SATURATION_MIN_DOCUMENTS = 3
CODE_FREQUENCY_SHIFT_LABEL = (
    "Total variation between the last two documents' code mixes. "
    "0 is the same mix. 1 is maximal change. This is not a stabilization score."
)


def thematic_saturation_report(
    windows: Sequence[Mapping],
    *,
    property_stability_report: Mapping | None = None,
    relationship_stability_report: Mapping | None = None,
) -> dict:
    """Indicators a researcher can read. This function does not set ``judgment``.

    Each window is one document. Batch size stays 1, so a rate is the count of
    new codes, properties, or relationships first evidenced in that document.
    ``code_frequency_shift`` is the total variation between the last two
    documents' normalized assignment distributions. 0 is the same mix and 1 is
    maximal change. It is not a stabilization score. ``code_frequency_stabilization``
    stores that same number for older readers. The figure describes the
    difference between those documents. It does not establish theoretical
    saturation, and it is separate from a human confirmation. It is undefined
    with fewer than two documents, or when either of those documents has no
    assignments. A 0.8.0 report stored presence bits and is not comparable.
    Cross-run stability needs the caller to pass a report from more than one
    run. Missing figures stay null.
    """
    rows = []
    previous: dict[str, float] | None = None
    have_previous = False
    for index, window in enumerate(windows):
        item = dict(window)
        frequencies = {
            str(key): int(value)
            for key, value in dict(item.get("code_frequencies") or {}).items()
        }
        distribution = _normalized_code_distribution(frequencies)
        shift = None
        if have_previous:
            if previous is not None and distribution is not None:
                shift = _total_variation(previous, distribution)
        previous = distribution
        have_previous = True
        new_codes = int(item.get("new_codes") or 0)
        new_properties = int(item.get("new_properties") or 0)
        new_relationships = int(item.get("new_relationships") or 0)
        rows.append(
            {
                "batch": index + 1,
                "document_id": str(item.get("document_id") or ""),
                "passages_in_batch": 1,
                "new_codes": new_codes,
                "new_properties": new_properties,
                "new_relationships": new_relationships,
                "new_code_rate": new_codes,
                "new_property_rate": new_properties,
                "new_relationship_rate": new_relationships,
                "code_frequency_shift": shift,
                "cumulative_codes": int(item.get("cumulative_codes") or 0),
            }
        )
    too_few = len(rows) < 2
    empty_comparison = (not too_few) and rows[-1]["code_frequency_shift"] is None
    property_level = None if not property_stability_report else property_stability_report.get("name_level")
    relationship_level = None if not relationship_stability_report else relationship_stability_report.get("role_level")
    notes = [
        "These figures are a saturation curve, not a verdict. judgment stays null. "
        "A researcher may record a decision separately. A model assessment is a model judgment "
        "and does not set judgment.",
        "code_frequency_shift is the total variation between consecutive documents' "
        "normalized code-assignment distributions. 0 is the same mix. 1 is maximal change. "
        "This is not a stabilization score. A low figure does not establish theoretical saturation "
        "and is not a human confirmation. "
        "code_frequency_stabilization stores that same total variation for older readers. "
        "A 0.8.0 report stored presence bits (1 once a code had been seen) and is not comparable.",
    ]
    n_documents = len(rows)
    window = "not_interpretable" if n_documents < SATURATION_MIN_DOCUMENTS else "interpretable"
    window_new_codes = sum(int(row["new_codes"]) for row in rows)
    window_new_relationships = sum(int(row["new_relationships"]) for row in rows)
    window_definition = (
        f"Window-wide rates divide the sum of new codes or relationships across {n_documents} document(s) "
        "by that document count. The last-document rate is only the newest document and is not the window rate."
    )
    if not rows:
        notes.append("Saturation rates are undefined: there are no documents.")
    elif too_few:
        notes.append("Code-frequency shift is undefined: fewer than two documents.")
    elif empty_comparison:
        notes.append("Code-frequency shift is undefined: a document has no code assignments.")
    if rows:
        notes.append(
            "last_document_new_code_rate and last_document_new_relationship_rate are the newest document only. "
            "window_new_code_rate and window_new_relationship_rate are the sums divided by n_documents. "
            "Do not read the last-document rate as the window rate."
        )
    if n_documents < SATURATION_MIN_DOCUMENTS:
        notes.append(
            f"The saturation window is {n_documents} document(s), below {SATURATION_MIN_DOCUMENTS}. "
            "That window is not interpretable. new_code_rate on it is not a saturation claim."
        )
    if property_stability_report is None or property_stability_report.get("name_level") is None:
        notes.append("Property stability across passes is undefined until two runs name properties.")
    if relationship_stability_report is None or relationship_stability_report.get("role_level") is None:
        notes.append("Relationship stability across passes is undefined until two runs share relationships.")
    return {
        "model": "theoretical_saturation_indicators",
        "batch_size": 1,
        "points": rows,
        "new_code_rate": None if not rows else rows[-1]["new_code_rate"],
        "new_property_rate": None if not rows else rows[-1]["new_property_rate"],
        "new_relationship_rate": None if not rows else rows[-1]["new_relationship_rate"],
        "last_document_new_code_rate": None if not rows else rows[-1]["new_code_rate"],
        "last_document_new_property_rate": None if not rows else rows[-1]["new_property_rate"],
        "last_document_new_relationship_rate": None if not rows else rows[-1]["new_relationship_rate"],
        "window_new_code_rate": None if n_documents == 0 else round(window_new_codes / n_documents, 4),
        "window_new_relationship_rate": None if n_documents == 0 else round(window_new_relationships / n_documents, 4),
        "window_definition": window_definition,
        "code_frequency_comparison": "total_variation",
        "code_frequency_shift": None if too_few or empty_comparison else rows[-1]["code_frequency_shift"],
        "code_frequency_shift_label": CODE_FREQUENCY_SHIFT_LABEL,
        "code_frequency_stabilization": None if too_few or empty_comparison else rows[-1]["code_frequency_shift"],
        "property_stability": None if property_stability_report is None else dict(property_stability_report),
        "relationship_stability": None if relationship_stability_report is None else dict(relationship_stability_report),
        "property_name_level": property_level,
        "relationship_role_level": relationship_level,
        "n_documents": n_documents,
        "saturation_window": window,
        "judgment": None,
        "researcher_saturation": None,
        "model_saturation": None,
        "note": " ".join(notes),
    }


STOPPING_QUIET_ROUNDS = 2
LOW_PAIR_COVERAGE_MIN_PLANNED = 10
LOW_PAIR_COVERAGE_RATE = 0.05
LOW_PAIR_COVERAGE_MIN_RELATIONSHIPS = 2
_REJECTED_RELATION_DISPOSITIONS = frozenset({"rejected", "merged"})


def pair_coverage_reading(
    planned: int | None,
    completed: int | None,
    relationships_found: int | None,
) -> dict:
    """Whether the pairs actually examined can support a saturation reading.

    Below 10 planned pairs the criterion stays undefined. At or above that
    bar, coverage under 0.05 or fewer than 2 relationships is unmet. A missing
    planned count stays undefined. The rate is not filled in as zero.
    """
    threshold = {
        "min_planned": LOW_PAIR_COVERAGE_MIN_PLANNED,
        "max_low_rate": LOW_PAIR_COVERAGE_RATE,
        "min_relationships": LOW_PAIR_COVERAGE_MIN_RELATIONSHIPS,
    }
    found = None if relationships_found is None else int(relationships_found)
    if planned is None:
        return {
            "status": "undefined",
            "low": False,
            "pair_coverage": None,
            "pairs_planned": None,
            "pairs_completed": None if completed is None else int(completed),
            "relationships_found": found,
            "caveat": "",
            "threshold": threshold,
        }
    planned_n = int(planned)
    completed_n = 0 if completed is None else int(completed)
    if planned_n <= 0:
        return {
            "status": "undefined",
            "low": False,
            "pair_coverage": None,
            "pairs_planned": planned_n,
            "pairs_completed": completed_n,
            "relationships_found": found,
            "caveat": "",
            "threshold": threshold,
        }
    rate = round(completed_n / planned_n, 4)
    low_rate = planned_n >= LOW_PAIR_COVERAGE_MIN_PLANNED and rate < LOW_PAIR_COVERAGE_RATE
    thin = (
        planned_n >= LOW_PAIR_COVERAGE_MIN_PLANNED
        and found is not None
        and found < LOW_PAIR_COVERAGE_MIN_RELATIONSHIPS
    )
    low = low_rate or thin
    if planned_n < LOW_PAIR_COVERAGE_MIN_PLANNED:
        status = "undefined"
    elif low:
        status = "unmet"
    else:
        status = "met"
    caveat = ""
    if low:
        found_text = "undefined" if found is None else str(found)
        caveat = (
            f"{completed_n} of {planned_n} pairs were examined (coverage {rate}). "
            f"Relationships found: {found_text}. "
            "That coverage is not interpretable as sufficient_for_this_question."
        )
    return {
        "status": status,
        "low": low,
        "pair_coverage": rate,
        "pairs_planned": planned_n,
        "pairs_completed": completed_n,
        "relationships_found": found,
        "caveat": caveat,
        "threshold": threshold,
    }


def gate_model_saturation(record: Mapping | None, reading: Mapping | None) -> dict:
    """Keep a low-coverage sufficiency claim from standing as a judgment.

    The model's words stay on ``model_said``. ``sufficient_for_this_question``
    is withheld. Other assessments stay, with the caveat required. ``judgment``
    is not a field this function sets.
    """
    row = dict(record or {})
    if not reading or not reading.get("low"):
        row.setdefault("status", "recorded")
        return row
    said = row.get("assessment")
    row["model_said"] = said
    row["status"] = "not_interpretable"
    row["coverage_caveat"] = str(reading.get("caveat") or "")
    row["pair_coverage"] = reading.get("pair_coverage")
    row["pairs_planned"] = reading.get("pairs_planned")
    row["pairs_completed"] = reading.get("pairs_completed")
    row["relationships_found"] = reading.get("relationships_found")
    if said == "sufficient_for_this_question":
        row["assessment"] = None
        row["withheld_reason"] = "low_pair_coverage"
    return row


def _window_rate(report: Mapping, key: str, point_key: str) -> float | None:
    """A rate already on the report, or the mean of document points. Missing stays missing."""
    if key in report and report.get(key) is not None:
        return float(report[key])
    points = [row for row in (report.get("points") or []) if isinstance(row, Mapping)]
    if not points or any(point_key not in row or row.get(point_key) is None for row in points):
        return None
    total = sum(int(row.get(point_key) or 0) for row in points)
    return round(total / len(points), 4)


def _rate_criterion(
    name: str,
    observed: float | None,
    *,
    n_documents: int | None,
    min_documents: int,
) -> dict:
    threshold = {"window_rate_at_most": 0.0, "min_documents": min_documents}
    if n_documents is None or int(n_documents) < min_documents or observed is None:
        status = "undefined"
    elif float(observed) <= 0.0:
        status = "met"
    else:
        status = "unmet"
    return {
        "criterion": name,
        "status": status,
        "threshold": threshold,
        "observed": observed,
        "n_documents": n_documents,
    }


def _core_has_endpoint(core_id: str, relationships: Sequence[Mapping] | None) -> bool:
    for row in relationships or []:
        if not isinstance(row, Mapping) or row.get("synthetic"):
            continue
        if "statement" in row and not str(row.get("statement") or "").strip():
            continue
        if str(row.get("disposition") or "") in _REJECTED_RELATION_DISPOSITIONS:
            continue
        if core_id and core_id in {str(row.get("from_id") or ""), str(row.get("to_id") or "")}:
            return True
    return False


def stopping_criteria_report(
    saturation: Mapping | None,
    ledger: Sequence[Mapping] | None = None,
    *,
    core_id: str = "",
    relationships: Sequence[Mapping] | None = None,
    quiet_rounds: int = STOPPING_QUIET_ROUNDS,
    min_documents: int = SATURATION_MIN_DOCUMENTS,
) -> dict:
    """Which stopping criteria are met, unmet, or undefined.

    The report does not decide that a theory is saturated. ``judgment`` stays
    null. A rate that was not computed stays undefined rather than zero.
    """
    report = dict(saturation or {})
    points = [row for row in (report.get("points") or []) if isinstance(row, Mapping)]
    n_documents = report.get("n_documents")
    if n_documents is None:
        n_documents = len(points) if points else None
    else:
        n_documents = int(n_documents)
    rounds = [row for row in (ledger or []) if isinstance(row, Mapping)]
    quiet = max(1, int(quiet_rounds))
    criteria = [
        _rate_criterion(
            "new_codes",
            _window_rate(report, "window_new_code_rate", "new_codes"),
            n_documents=n_documents,
            min_documents=min_documents,
        ),
        _rate_criterion(
            "new_properties",
            _window_rate(report, "window_new_property_rate", "new_properties"),
            n_documents=n_documents,
            min_documents=min_documents,
        ),
        _rate_criterion(
            "new_relationships",
            _window_rate(report, "window_new_relationship_rate", "new_relationships"),
            n_documents=n_documents,
            min_documents=min_documents,
        ),
    ]
    shift = report.get("code_frequency_shift")
    shift_threshold = {"total_variation_at_most": 0.0, "min_documents": 2}
    if shift is None or n_documents is None or int(n_documents) < 2:
        shift_status = "undefined"
    elif float(shift) <= 0.0:
        shift_status = "met"
    else:
        shift_status = "unmet"
    criteria.append(
        {
            "criterion": "code_frequency_shift",
            "status": shift_status,
            "threshold": shift_threshold,
            "observed": None if shift is None else float(shift),
            "n_documents": n_documents,
            "label": CODE_FREQUENCY_SHIFT_LABEL,
        }
    )
    if len(rounds) < quiet:
        quiet_status = "undefined"
        quiet_observed = None
    else:
        recent = rounds[-quiet:]
        introduced = [
            list(
                row.get("analytic_new_direction_keys")
                if "analytic_new_direction_keys" in row
                else row.get("new_direction_keys") or []
            )
            for row in recent
        ]
        quiet_observed = introduced
        quiet_status = "met" if all(not row for row in introduced) else "unmet"
    criteria.append(
        {
            "criterion": "sampling_rounds_without_new_directions",
            "status": quiet_status,
            "threshold": {"quiet_rounds": quiet},
            "observed": quiet_observed,
            "sampling_rounds": len(rounds),
        }
    )
    chosen = str(core_id or "")
    if not chosen:
        core_status = "undefined"
        core_supported = None
    else:
        core_supported = _core_has_endpoint(chosen, relationships)
        core_status = "met" if core_supported else "unmet"
    criteria.append(
        {
            "criterion": "core_category_coverage",
            "status": core_status,
            "threshold": {"requires_relationship_endpoint": True},
            "observed": None if not chosen else {"core_id": chosen, "endpoint": core_supported},
        }
    )
    reading = pair_coverage_reading(
        report.get("pairs_planned") if "pairs_planned" in report else None,
        report.get("pairs_completed") if "pairs_planned" in report else None,
        report.get("relationships_found") if "relationships_found" in report else None,
    )
    criteria.append(
        {
            "criterion": "pair_coverage",
            "status": reading["status"],
            "threshold": reading["threshold"],
            "observed": {
                "pairs_planned": reading.get("pairs_planned"),
                "pairs_completed": reading.get("pairs_completed"),
                "pair_coverage": reading.get("pair_coverage"),
                "relationships_found": reading.get("relationships_found"),
            },
            "low": reading["low"],
            "caveat": reading.get("caveat") or None,
        }
    )
    if reading["low"]:
        for row in criteria:
            if row.get("criterion") == "new_relationships" and row.get("status") == "met":
                row["status"] = "undefined"
                row["undefined_reason"] = "low_pair_coverage"
                row["coverage_caveat"] = reading.get("caveat") or ""
    return {
        "model": "stopping_criteria",
        "declares_saturation": False,
        "judgment": None,
        "criteria": criteria,
        "thresholds_note": (
            "A criterion is met only at the threshold written on that row. "
            "met does not mean the theory is saturated. "
            "undefined stays undefined when the window, the rate, the core, or the sampling ledger cannot support the comparison."
        ),
        "note": (
            "This report states which stopping criteria are met, unmet, or undefined. "
            "It does not declare the theory saturated. judgment stays null. "
            "A researcher decision and a model judgment are separate records."
        ),
    }


def saturation_report(points: Sequence[Mapping]) -> dict:
    rows = [dict(point) for point in points]
    latest = rows[-1] if rows else None
    return {
        "window": "passages between sweeps",
        "points": rows,
        "latest_new_codes": None if latest is None else latest["new_codes"],
        "latest_new_codes_per_passage": None if latest is None else latest["new_codes_per_passage"],
        "cumulative_codes": 0 if latest is None else latest["cumulative_codes"],
        "cumulative_passages": 0 if latest is None else latest["cumulative_passages"],
        "judgment": None,
        "note": (
            "New codes per batch are reported so a researcher can judge saturation. "
            "This package does not decide that categories are saturated."
        ),
    }
