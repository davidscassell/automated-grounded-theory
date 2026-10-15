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
    if not rows:
        notes.append("Saturation rates are undefined: there are no documents.")
    elif too_few:
        notes.append("Code-frequency shift is undefined: fewer than two documents.")
    elif empty_comparison:
        notes.append("Code-frequency shift is undefined: a document has no code assignments.")
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
        "code_frequency_comparison": "total_variation",
        "code_frequency_shift": None if too_few or empty_comparison else rows[-1]["code_frequency_shift"],
        "code_frequency_shift_label": CODE_FREQUENCY_SHIFT_LABEL,
        "code_frequency_stabilization": None if too_few or empty_comparison else rows[-1]["code_frequency_shift"],
        "property_stability": None if property_stability_report is None else dict(property_stability_report),
        "relationship_stability": None if relationship_stability_report is None else dict(relationship_stability_report),
        "property_name_level": property_level,
        "relationship_role_level": relationship_level,
        "judgment": None,
        "researcher_saturation": None,
        "model_saturation": None,
        "note": " ".join(notes),
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
