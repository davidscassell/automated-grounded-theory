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
) -> dict:
    rate = (new_codes / passages_in_batch) if passages_in_batch else 0.0
    return {
        "batch": batch,
        "passages_in_batch": passages_in_batch,
        "new_codes": new_codes,
        "new_codes_per_passage": rate,
        "cumulative_passages": cumulative_passages,
        "cumulative_codes": cumulative_codes,
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
