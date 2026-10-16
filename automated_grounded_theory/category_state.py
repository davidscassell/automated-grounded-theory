"""Versioned category-development state shared by later stages.

Focused coding writes this state. Relations are optional. A later stage reads
the file it is given and does not require an axial graph to have run.
Integration must tolerate ``relations`` being null.

Every proposal keeps three separate fields:

- ``disposition``: ``proposed``, ``accepted``, ``rejected``, or ``superseded``
- ``evidence_condition``: ``anchored``, ``disputed``, ``disputed_pending_review``, or ``insufficient``
- ``downstream_eligible``: true only when the disposition is ``accepted``,
  the evidence condition is ``anchored``, the row is not marked
  ``needs_review``, human ``claim_support`` is not ``contradicted``, and the
  row is not a conflicting property merge

``claim_support`` is a researcher review. It stays ``undefined`` until a person
sets ``supported``, ``partially_supported``, ``contradicted``, or
``indeterminate``. A contradicted claim is excluded downstream. Citation
anchoring is a different measurement: whether the quote located a span. It is
not claim support. An LLM reviewer writes ``model_claim_support`` and does not
set ``claim_support``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

REREAD_PURPOSES = ("develop", "compete", "outside")
# claim_compete is reserved up front and run after properties exist. It is not
# an incident purpose and it is not interleaved into the first queue.
COVERAGE_PURPOSES = ("develop", "compete", "outside", "claim_compete")
DISPOSITIONS = ("proposed", "accepted", "rejected", "superseded")
EVIDENCE_CONDITIONS = ("anchored", "disputed", "disputed_pending_review", "insufficient")
CLAIM_SUPPORTS = (
    "supported",
    "partially_supported",
    "contradicted",
    "indeterminate",
    "undefined",
)
ANALYTIC_FEATURES = ("memo", "in_vivo", "comparison")
ANCHOR_STATUSES = frozenset({"exact", "normalized_match", "fuzzy_match"})
STATE_VERSION = 1
# Memo count is one analytic-weight feature. A long memo raises it. That can
# reward verbosity rather than a distinction the researcher would pin.
MEMO_COUNT_NOTE = (
    "Substantive memo count can reward a verbose memo. "
    "Ablate the 'memo' feature to see the selection without that count."
)

DEFAULT_SHARES = {
    "develop": 0.25,
    "compete": 0.25,
    "outside": 0.25,
    "claim_compete": 0.25,
}
PROPERTY_STABILITY_NOTE = (
    "Category-level stability is the Jaccard overlap of categories that received a property. "
    "Name-level stability is the Jaccard overlap of (category_id, property_key) pairs. "
    "property_key sorts the content words of the name and drops function words, "
    "so 'timing of departure' and 'timing during departure' match 'departure timing'. "
    "A low name-level score means those remaining words differ. The score is reported as measured. "
    "A low score with many properties in each run means the runs named different properties. "
    "category_counts, name_counts, and properties_per_run are those set sizes. "
    "A single run leaves both scores empty. property_stability() compares two or more runs. "
    "This is not claim support."
)
_PROPERTY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "these",
        "those",
        "it",
        "as",
        "when",
        "during",
        "about",
        "into",
        "over",
        "than",
        "their",
        "its",
    }
)


def category_revision_id(entry: Mapping[str, Any]) -> str:
    """Stable id for one revision of a category. Label and definition are included."""
    raw = "|".join(
        (
            str(entry.get("code_id") or ""),
            str(entry.get("revision") or 1),
            str(entry.get("code") or ""),
            str(entry.get("definition") or ""),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evidence_revision_id(
    *,
    passage_id: str,
    document_id: str,
    quote: str,
    quote_status: str,
) -> str:
    """Stable id for the quote as cited. A different span is a different id."""
    raw = "|".join((passage_id, document_id, quote_status, quote))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalize_shares(shares: Mapping[str, float] | None) -> dict[str, float]:
    """Return develop, compete, outside, and claim_compete shares that sum to 1.

    A three-purpose mapping that already sums to 1 keeps ``claim_compete`` at 0,
    so an older caller does not lose its split. The default reserves a quarter
    for the claim round.
    """
    if shares is None:
        incoming = dict(DEFAULT_SHARES)
    else:
        incoming = {str(key): value for key, value in dict(shares).items()}
        unknown = set(incoming) - set(COVERAGE_PURPOSES)
        if unknown:
            raise ValueError(f"Unknown reread purpose: {sorted(unknown)}")
        queue_total = sum(float(incoming.get(purpose, 0) or 0) for purpose in REREAD_PURPOSES)
        if "claim_compete" not in incoming and abs(queue_total - 1.0) <= 1e-6:
            incoming["claim_compete"] = 0.0
    unknown = set(incoming) - set(COVERAGE_PURPOSES)
    if unknown:
        raise ValueError(f"Unknown reread purpose: {sorted(unknown)}")
    values = {purpose: float(incoming.get(purpose, 0) or 0) for purpose in COVERAGE_PURPOSES}
    if any(value < 0 for value in values.values()):
        raise ValueError("Focused reread shares must be zero or more.")
    total = sum(values.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Focused reread shares must sum to 1.")
    return values


def property_key(name: str) -> str:
    """Stable key for a property name. Word order and light plurals do not change it."""
    tokens = []
    for raw in str(name or "").casefold().replace("-", " ").replace("_", " ").split():
        cleaned = "".join(character for character in raw if character.isalnum() or character == "'")
        if not cleaned or cleaned in _PROPERTY_STOPWORDS:
            continue
        if cleaned.endswith("ies") and len(cleaned) > 4:
            cleaned = cleaned[:-3] + "y"
        elif cleaned.endswith("s") and not cleaned.endswith("ss") and len(cleaned) > 3:
            cleaned = cleaned[:-1]
        tokens.append(cleaned)
    return " ".join(sorted(tokens))


def _property_citation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_id": row.get("entry_id"),
        "name": row.get("name"),
        "document_id": row.get("document_id"),
        "passage_id": row.get("passage_id"),
        "quote": row.get("quote"),
        "quote_status": row.get("quote_status"),
    }


def _norm_statement(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _property_constituent(row: Mapping[str, Any]) -> dict[str, Any]:
    """Identity of one row folded into a merged property."""
    return {
        "entry_id": row.get("entry_id"),
        "name": row.get("name"),
        "text": row.get("text"),
        "pole": row.get("pole"),
        "document_id": row.get("document_id"),
        "passage_id": row.get("passage_id"),
        "quote": row.get("quote"),
    }


def _statements_conflict(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Same property key, different stated text or pole. Names may still match."""
    for key in ("text", "pole"):
        a = _norm_statement(left.get(key))
        b = _norm_statement(right.get(key))
        if a and b and a != b:
            return True
    return False


def merge_properties(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse properties that share a category and a normalized name.

    The first row keeps its text. Later rows add a citation and a constituent.
    A different non-empty text or pole is a ``property_merge_conflict``: the
    merged row is flagged, keeps every constituent, and is not downstream
    eligible. Dimensions are not merged. A blank name is not merged with
    another blank name. A human ``contradicted`` label on either row excludes
    the merged property downstream.
    """
    merged: list[dict[str, Any]] = []
    index: dict[tuple[str, str], int] = {}
    for row in records:
        item = dict(row)
        if str(item.get("kind") or "property") != "property":
            merged.append(item)
            continue
        key = property_key(str(item.get("name") or ""))
        item["property_key"] = key
        slot = (str(item.get("category_id") or ""), key)
        if not key or slot not in index:
            item["citations"] = list(item.get("citations") or []) or [_property_citation(item)]
            names = [str(name) for name in (item.get("merged_names") or []) if str(name)]
            own = str(item.get("name") or "")
            if own and own not in names:
                names.append(own)
            item["merged_names"] = names
            if str(item.get("claim_support") or "") == "contradicted":
                item["downstream_eligible"] = False
            if key:
                index[slot] = len(merged)
            merged.append(item)
            continue
        target = merged[index[slot]]
        citations = list(target.get("citations") or [])
        citations.extend(item.get("citations") or [_property_citation(item)])
        target["citations"] = citations
        names = list(target.get("merged_names") or [])
        for name in list(item.get("merged_names") or []) + [str(item.get("name") or "")]:
            if name and name not in names:
                names.append(name)
        target["merged_names"] = names
        constituents = list(target.get("constituents") or [])
        if not constituents:
            constituents.append(_property_constituent(target))
        constituents.append(_property_constituent(item))
        target["constituents"] = constituents
        flags = list(target.get("flags") or [])
        for flag in item.get("flags") or []:
            if flag not in flags:
                flags.append(flag)
        conflict = (
            _statements_conflict(target, item)
            or bool(item.get("property_merge_conflict"))
            or bool(target.get("property_merge_conflict"))
            or "property_merge_conflict" in flags
        )
        if conflict and "property_merge_conflict" not in flags:
            flags.append("property_merge_conflict")
        target["flags"] = flags
        if conflict:
            target["property_merge_conflict"] = True
            target["needs_review"] = True
            target["downstream_eligible"] = False
        if item.get("needs_review"):
            target["needs_review"] = True
            target["downstream_eligible"] = False
        if str(item.get("claim_support") or "") == "contradicted" or str(target.get("claim_support") or "") == "contradicted":
            target["claim_support"] = "contradicted"
            target["downstream_eligible"] = False
    return merged


def _property_rows(run: Any) -> list[Mapping[str, Any]]:
    if isinstance(run, (list, tuple)):
        return [row for row in run if isinstance(row, Mapping)]
    if isinstance(run, Mapping):
        rows = run.get("properties")
        if rows is None and "category_development" in run:
            rows = (run.get("category_development") or {}).get("properties")
        return list(rows or [])
    rows = getattr(run, "properties", None)
    return list(rows or [])


def _mean_jaccard(sets: Sequence[set], *, skip_empty_pairs: bool = False) -> float | None:
    if len(sets) < 2:
        return None
    scores = []
    for left_index in range(len(sets)):
        for right in sets[left_index + 1 :]:
            left = sets[left_index]
            union = left | right
            if skip_empty_pairs and not union:
                continue
            scores.append(1.0 if not union else len(left & right) / len(union))
    if not scores:
        return None
    return round(sum(scores) / len(scores), 6)


def property_stability(runs: Sequence[Any]) -> dict[str, Any]:
    """Category-level and name-level property overlap across runs, usually seeds.

    Category-level compares the set of categories that have a property.
    Name-level compares ``(category_id, property_key)``. Both are mean pairwise
    Jaccard. One run has no pair, so both scores are empty.
    """
    category_sets = []
    name_sets = []
    for run in runs:
        rows = [row for row in _property_rows(run) if str(row.get("kind") or "property") == "property"]
        category_sets.append({str(row.get("category_id") or "") for row in rows if row.get("category_id")})
        name_sets.append(
            {
                (str(row.get("category_id") or ""), str(row.get("property_key") or property_key(str(row.get("name") or ""))))
                for row in rows
                if row.get("property_key") or property_key(str(row.get("name") or ""))
            }
        )
    name_level = _mean_jaccard(name_sets)
    name_counts = [len(item) for item in name_sets]
    volume_note = ""
    if name_level is not None and name_level < 0.5 and name_counts and min(name_counts) >= 5:
        volume_note = (
            "Name-level overlap is low while each run named many properties. "
            "The runs used different property names. The score is reported and is not claim support."
        )
    return {
        "name": "property_stability",
        "runs": len(list(runs)),
        "category_level": _mean_jaccard(category_sets),
        "name_level": name_level,
        "category_counts": [len(item) for item in category_sets],
        "name_counts": name_counts,
        "properties_per_run": [
            len([row for row in _property_rows(run) if str(row.get("kind") or "property") == "property"])
            for run in runs
        ],
        "volume_note": volume_note,
        "note": PROPERTY_STABILITY_NOTE,
    }


PROPOSAL_STABILITY_NOTE = (
    "Code-level stability is the Jaccard overlap of codes that received a revise or drop proposal. "
    "Key-level stability compares (code_id, passage_id, property_key(definition)). "
    "property_key folds wording. An empty definition, which a drop often carries, matches a revise "
    "of the same code on the same passage, so revise and drop of one code are comparable. "
    "When those kinds differ, or the folded definitions differ, the pair is listed in cross_seed_conflicts. "
    "The overlap score does not hide that disagreement, and neither proposal is applied. "
    "A single run leaves both scores empty. proposal_stability() compares two or more runs. "
    "A pair of runs with no proposals is not scored. If no pair has a proposal, both scores are undefined. "
    "Pass a run, a run folder, an export, segments, or proposal rows. "
    "This is not claim support."
)
_PROPOSAL_KINDS = frozenset({"focused_revise_proposal", "focused_drop_proposal", "revise", "drop"})


def _proposal_kind(row: Mapping[str, Any]) -> str:
    kind = str(row.get("kind") or row.get("action") or "")
    if kind in {"revise", "focused_revise_proposal"}:
        return "revise"
    if kind in {"drop", "focused_drop_proposal"}:
        return "drop"
    return kind


def _proposal_fold(row: Mapping[str, Any]) -> str:
    return property_key(str(row.get("definition") or ""))


def _proposal_record(row: Mapping[str, Any]) -> bool:
    return str(row.get("kind") or row.get("action") or "") in _PROPOSAL_KINDS


def _segment_like(row: Mapping[str, Any]) -> bool:
    if any(key in row for key in ("segment_id", "focused_revise_proposal", "focused_drop_proposal")):
        return True
    return "code_id" in row and "kind" not in row and "action" not in row


def _proposals_from_export(place: Path) -> list[Mapping[str, Any]]:
    """Revise and drop rows from a run folder or one export file."""
    if place.is_dir():
        segments_file = place / "coded_segments.json"
        run_file = place / "focused_run.json"
        if not segments_file.exists() and not run_file.exists():
            raise ValueError(
                f"No focused export in {place}. Expected coded_segments.json or focused_run.json."
            )
        rows: list[Mapping[str, Any]] = []
        if segments_file.exists():
            payload = json.loads(segments_file.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows.extend(_proposals_from_segments(payload))
        if not rows and run_file.exists():
            payload = json.loads(run_file.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                rows.extend(_proposal_rows(payload))
        return rows
    if place.is_file():
        payload = json.loads(place.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return _proposal_rows(payload)
        if isinstance(payload, Mapping):
            return _proposal_rows(payload)
    raise ValueError(f"proposal_stability cannot read {place}.")


def _proposal_rows_from_sequence(items: Sequence[Any]) -> list[Mapping[str, Any]]:
    """Proposal rows, or the proposals nested on a segment list.

    A bare segment list is not a set of proposals. Scoring every segment's
    ``code_id`` made two unrelated runs look identical.
    """
    rows = list(items)
    if not rows:
        return []
    if all(isinstance(item, (str, Path)) for item in rows):
        found: list[Mapping[str, Any]] = []
        for item in rows:
            found.extend(_proposals_from_export(Path(item)))
        return found
    mappings = [item for item in rows if isinstance(item, Mapping)]
    if len(mappings) != len(rows):
        raise TypeError(
            "proposal_stability expected runs, a run folder, an export, segments, or proposal rows. "
            f"The list contained {type(rows[0]).__name__}."
        )
    if all(_proposal_record(item) for item in mappings):
        return [dict(item) for item in mappings]
    if all(_segment_like(item) for item in mappings):
        return _proposals_from_segments(mappings)
    raise TypeError(
        "proposal_stability could not tell proposal rows from segments. "
        "Pass runs, a run folder, coded_segments.json, or rows whose kind is "
        "focused_revise_proposal or focused_drop_proposal."
    )


def _proposal_rows(run: Any) -> list[Mapping[str, Any]]:
    """Revise and drop proposals from a run, a folder, a segment list, or proposal rows."""
    if isinstance(run, (str, Path)):
        return _proposals_from_export(Path(run))
    if isinstance(run, (list, tuple)):
        return _proposal_rows_from_sequence(run)
    if isinstance(run, Mapping):
        consolidated = run.get("proposal_consolidation")
        if consolidated is None and isinstance(run.get("metadata"), Mapping):
            consolidated = run["metadata"].get("proposal_consolidation")
        if isinstance(consolidated, list):
            rows = []
            for group in consolidated:
                if not isinstance(group, Mapping):
                    continue
                for item in group.get("proposals") or []:
                    if isinstance(item, Mapping):
                        rows.append({**item, "code_id": item.get("code_id") or group.get("code_id")})
            if rows:
                return rows
        segments = run.get("segments")
        if isinstance(segments, list):
            return _proposals_from_segments(segments)
    segments = getattr(run, "segments", None)
    if isinstance(segments, list):
        return _proposals_from_segments(segments)
    consolidated = getattr(run, "metadata", None)
    if isinstance(consolidated, Mapping):
        return _proposal_rows(consolidated)
    raise TypeError(
        "proposal_stability expected a run, a run folder, an export, segments, or proposal rows. "
        f"Got {type(run).__name__}."
    )


def _proposals_from_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        code_id = str(segment.get("code_id") or "")
        passage_id = str(segment.get("passage_id") or "")
        for kind in ("focused_revise_proposal", "focused_drop_proposal"):
            payload = segment.get(kind)
            if isinstance(payload, Mapping):
                rows.append({"code_id": code_id, "passage_id": passage_id, "kind": kind, **dict(payload)})
    return rows


def _comparable_proposal_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("code_id") or ""),
        str(row.get("passage_id") or ""),
        _proposal_fold(row),
    )


def _proposal_keys_match(left: tuple[str, str, str], right: tuple[str, str, str]) -> bool:
    """Same code and passage. An empty definition matches the other side's fold."""
    if left[0] != right[0] or left[1] != right[1]:
        return False
    return not left[2] or not right[2] or left[2] == right[2]


def _comparable_key_jaccard(sets: Sequence[set[tuple[str, str, str]]]) -> float | None:
    if len(sets) < 2:
        return None
    scores = []
    for left_index in range(len(sets)):
        left = sets[left_index]
        for right in sets[left_index + 1 :]:
            if not left and not right:
                continue
            used: set[int] = set()
            intersection = 0
            for item in left:
                for index, other in enumerate(tuple(right)):
                    if index in used:
                        continue
                    if _proposal_keys_match(item, other):
                        used.add(index)
                        intersection += 1
                        break
            union = len(left) + len(right) - intersection
            scores.append(1.0 if union == 0 else intersection / union)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 6)


def _cross_seed_conflicts(grouped_rows: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Revise versus drop, or different folded definitions, for one code on one passage."""
    grouped: dict[tuple[str, str], list[tuple[int, Mapping[str, Any]]]] = {}
    for run_index, rows in enumerate(grouped_rows):
        for row in rows:
            code_id = str(row.get("code_id") or "")
            if not code_id:
                continue
            grouped.setdefault((code_id, str(row.get("passage_id") or "")), []).append((run_index, row))
    conflicts = []
    for (code_id, passage_id), items in grouped.items():
        runs_present = {index for index, _row in items}
        if len(runs_present) < 2:
            continue
        kinds = sorted({_proposal_kind(row) for _index, row in items if _proposal_kind(row)})
        folds = sorted({_proposal_fold(row) for _index, row in items if _proposal_fold(row)})
        if len(kinds) <= 1 and len(folds) <= 1:
            continue
        conflicts.append(
            {
                "code_id": code_id,
                "passage_id": passage_id,
                "kinds": kinds,
                "property_keys": folds,
                "definitions": sorted(
                    {str(row.get("definition") or "") for _index, row in items if str(row.get("definition") or "")}
                ),
                "runs": sorted(runs_present),
            }
        )
    return conflicts


def proposal_stability(runs: Sequence[Any]) -> dict[str, Any]:
    """Overlap of revise and drop proposals across runs.

    One run has no pair, so both scores are empty. Two runs that both have no
    proposals are undefined, not 1. Key-level overlap treats a revise and a drop
    of the same code on the same passage as the same target. ``cross_seed_conflicts``
    lists that disagreement. It does not apply a proposal and it is not claim support.
    """
    code_sets = []
    key_sets: list[set[tuple[str, str, str]]] = []
    grouped_rows = []
    for run in runs:
        rows = _proposal_rows(run)
        grouped_rows.append(rows)
        code_sets.append({str(row.get("code_id") or "") for row in rows if row.get("code_id")})
        key_sets.append({_comparable_proposal_key(row) for row in rows if row.get("code_id")})
    return {
        "name": "proposal_stability",
        "runs": len(list(runs)),
        "code_level": _mean_jaccard(code_sets, skip_empty_pairs=True),
        "key_level": _comparable_key_jaccard(key_sets),
        "cross_seed_conflicts": _cross_seed_conflicts(grouped_rows),
        "note": PROPOSAL_STABILITY_NOTE,
    }


def property_stability_report(
    raw: Sequence[Mapping[str, Any]],
    merged: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """What one run can say before another seed is compared."""
    rows = [row for row in merged if str(row.get("kind") or "property") == "property"]
    return {
        "name": "property_stability",
        "runs": 1,
        "category_level": None,
        "name_level": None,
        "raw_properties": len(list(raw)),
        "merged_properties": len(rows),
        "categories": sorted({str(row.get("category_id") or "") for row in rows if row.get("category_id")}),
        "property_keys": sorted(
            {str(row.get("property_key") or "") for row in rows if row.get("property_key")}
        ),
        "note": PROPERTY_STABILITY_NOTE,
    }


def normalize_analytic_features(features: Sequence[str] | None) -> tuple[str, ...]:
    """Features that enter analytic weight. None keeps memo, in-vivo, and comparison.

    An empty sequence turns analytic weight off. Memo count can reward verbosity.
    """
    if features is None:
        return ANALYTIC_FEATURES
    unknown = [name for name in features if name not in ANALYTIC_FEATURES]
    if unknown:
        raise ValueError(
            "focused analytic features must be drawn from "
            + ", ".join(ANALYTIC_FEATURES)
            + f". Unknown: {unknown}"
        )
    chosen = set(features)
    return tuple(name for name in ANALYTIC_FEATURES if name in chosen)


def evidence_condition_for(*, quote_status: str, flags: Sequence[str], supported: bool) -> str:
    """Anchored, disputed, or insufficient. This is not a researcher verdict."""
    held = set(flags or [])
    located = quote_status in ANCHOR_STATUSES and "unsupported" not in held
    disputed = bool(held & {"evidence_outside_passage", "possibly_modified", "attributed_passage"})
    if quote_status == "possibly_modified":
        disputed = True
    if located and not disputed and supported:
        return "anchored"
    if located or disputed:
        return "disputed"
    return "insufficient"


def proposal_record_fields(
    *,
    disposition: str = "proposed",
    evidence_condition: str,
    claim_support: str = "undefined",
    needs_review: bool = False,
) -> dict[str, Any]:
    """Disposition, evidence condition, and whether a downstream view may use the row."""
    if disposition not in DISPOSITIONS:
        raise ValueError(f"disposition must be one of {DISPOSITIONS}")
    if evidence_condition not in EVIDENCE_CONDITIONS:
        raise ValueError(f"evidence_condition must be one of {EVIDENCE_CONDITIONS}")
    if claim_support not in CLAIM_SUPPORTS:
        raise ValueError(f"claim_support must be one of {CLAIM_SUPPORTS}")
    eligible = (
        disposition == "accepted"
        and evidence_condition == "anchored"
        and not needs_review
        and claim_support != "contradicted"
    )
    return {
        "disposition": disposition,
        "evidence_condition": evidence_condition,
        "downstream_eligible": eligible,
        "claim_support": claim_support,
    }


def citation_anchored(record: Mapping[str, Any]) -> bool:
    """True when the quote located a span. Tentative and unanchored quotes are not included."""
    return str(record.get("quote_status") or "") in ANCHOR_STATUSES


def citation_anchoring_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Share of claims whose quotes anchor. This is not whether the claim is true."""
    rows = list(records)
    anchored = [row for row in rows if citation_anchored(row)]
    tentative = [
        row
        for row in rows
        if str(row.get("quote_status") or "") in {"possibly_modified", "unanchored"}
        or "unsupported" in (row.get("flags") or [])
    ]
    total = len(rows)
    return {
        "name": "citation_anchoring_rate",
        "anchored": len(anchored),
        "total": total,
        "rate": None if total == 0 else round(len(anchored) / total, 6),
        "tentative_or_unanchored": len(tentative),
        "note": (
            "Citation anchoring is whether a quote locates a span. "
            "It is not claim support and it is not the truth of a relationship."
        ),
    }


def claim_support_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Researcher labels. Unreviewed claims are reported as undefined."""
    counts = {label: 0 for label in CLAIM_SUPPORTS}
    for row in records:
        label = str(row.get("claim_support") or "undefined")
        if label not in counts:
            label = "undefined"
        counts[label] += 1
    reviewed = sum(counts[label] for label in CLAIM_SUPPORTS if label != "undefined")
    return {
        "name": "claim_support",
        "counts": counts,
        "reviewed": reviewed,
        "undefined": counts["undefined"],
        "note": "Undefined until a researcher reviews the claim. The model does not set this field.",
    }


def view_notes() -> dict[str, str]:
    """Which records each export uses. All rows are stored."""
    return {
        "category_development.json": "all records, including rejected, superseded, and insufficient",
        "focused_claims.csv": "all records",
        "downstream_claims.json": (
            "downstream_eligible only: disposition accepted, evidence anchored, "
            "not needs_review, claim_support not contradicted, and not a property_merge_conflict"
        ),
        "downstream_eligible": (
            "accepted and anchored, not marked needs_review, "
            "claim_support not contradicted, and not a property_merge_conflict"
        ),
    }


def interleave_queue(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    shares: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Merge purpose queues so a prefix of the allowance covers each purpose.

    Empty purposes drop out and their share is redistributed. The same groups
    and shares always produce the same order.
    """
    pending = {purpose: [dict(item) for item in groups.get(purpose) or []] for purpose in REREAD_PURPOSES}
    active = [purpose for purpose in REREAD_PURPOSES if pending[purpose]]
    if not active:
        return []
    total = sum(float(shares[purpose]) for purpose in active) or 1.0
    weight = {purpose: float(shares[purpose]) / total for purpose in active}
    debt = {purpose: 0.0 for purpose in active}
    ordered: list[dict[str, Any]] = []
    while any(pending[purpose] for purpose in active):
        for purpose in active:
            debt[purpose] += weight[purpose]
        choice = max(
            (purpose for purpose in active if pending[purpose]),
            key=lambda purpose: (debt[purpose], -active.index(purpose)),
        )
        debt[choice] -= 1.0
        ordered.append(pending[choice].pop(0))
    return ordered


def purpose_coverage(
    queue: Sequence[Mapping[str, Any]],
    reread_ids: Sequence[str],
    calls: Mapping[str, int],
    tokens: Mapping[str, int],
    shares: Mapping[str, float],
) -> dict[str, Any]:
    """Queued, reread, and unsent passages, with calls and tokens, per purpose."""
    read = {str(item) for item in reread_ids}
    report: dict[str, Any] = {}
    for purpose in COVERAGE_PURPOSES:
        ids = [str(item.get("passage_id") or "") for item in queue if item.get("purpose") == purpose]
        reread = [passage_id for passage_id in ids if passage_id in read]
        report[purpose] = {
            "share": float(shares.get(purpose, 0)),
            "queued": len(ids),
            "passage_ids": ids,
            "reread": reread,
            "not_reread": [passage_id for passage_id in ids if passage_id not in read],
            "calls": int(calls.get(purpose) or 0),
            "tokens": int(tokens.get(purpose) or 0),
        }
    return report


def _pole_token(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _text_has_pole(text: str, pole: str) -> bool:
    token = _pole_token(pole)
    if not token:
        return False
    return token in " ".join(str(text or "").casefold().split())


def negative_case_candidates(
    passages: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    *,
    codebook: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Passages that can contradict one claim.

    The search is per claim. It includes passages that carry the claim's own
    category, for example the same code with an opposite consequence or another
    pole of the same dimension. It is not limited to passages outside a
    storyline or to passages that were already flagged.
    """
    by_passage = {str(item.get("passage_id") or ""): item for item in passages if item.get("passage_id")}
    assigned: dict[str, set[str]] = {}
    for segment in segments:
        code_id = str(segment.get("code_id") or "")
        passage_id = str(segment.get("passage_id") or "")
        if code_id and passage_id:
            assigned.setdefault(code_id, set()).add(passage_id)
    poles_by_dimension: dict[str, set[str]] = {}
    for entry in codebook or []:
        for dimension in entry.get("dimensions") or []:
            name = _pole_token(str(dimension.get("name") or ""))
            pole = _pole_token(str(dimension.get("value") or dimension.get("pole") or ""))
            if name and pole:
                poles_by_dimension.setdefault(name, set()).add(pole)
    for claim in claims:
        name = _pole_token(str(claim.get("name") or ""))
        pole = _pole_token(str(claim.get("pole") or ""))
        if name and pole:
            poles_by_dimension.setdefault(name, set()).add(pole)
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    claim_rows = [dict(item) for item in claims]
    for claim in claim_rows:
        claim_id = str(claim.get("entry_id") or claim.get("claim_id") or "")
        category_id = str(claim.get("category_id") or "")
        pole = _pole_token(str(claim.get("pole") or ""))
        name = _pole_token(str(claim.get("name") or ""))
        support = {str(claim.get("passage_id") or "")} - {""}
        support |= assigned.get(category_id, set())
        other_poles = set(poles_by_dimension.get(name, set())) - ({pole} if pole else set())
        for sibling in claim_rows:
            if sibling is claim:
                continue
            if str(sibling.get("category_id") or "") != category_id:
                continue
            if _pole_token(str(sibling.get("name") or "")) != name:
                continue
            sibling_pole = _pole_token(str(sibling.get("pole") or sibling.get("text") or ""))
            if sibling_pole and sibling_pole != pole:
                other_poles.add(sibling_pole)
                sibling_passage = str(sibling.get("passage_id") or "")
                if sibling_passage:
                    _add_hit(
                        hits,
                        seen,
                        passage_id=sibling_passage,
                        claim_id=claim_id,
                        category_id=category_id,
                        reason="within_category",
                        detail="same category, different consequence",
                    )
        for passage_id in sorted(support):
            passage = by_passage.get(passage_id) or {}
            text = str(passage.get("text") or "")
            if pole and any(_text_has_pole(text, other) for other in other_poles):
                _add_hit(
                    hits,
                    seen,
                    passage_id=passage_id,
                    claim_id=claim_id,
                    category_id=category_id,
                    reason="within_category",
                    detail="same category, opposite pole in the passage",
                )
        for passage_id, passage in by_passage.items():
            if passage_id in support:
                continue
            text = str(passage.get("text") or "")
            if pole and (_text_has_pole(text, pole) or any(_text_has_pole(text, other) for other in other_poles)):
                _add_hit(
                    hits,
                    seen,
                    passage_id=passage_id,
                    claim_id=claim_id,
                    category_id=category_id,
                    reason="outside_support",
                    detail="passage is outside the claim's supporting assignments",
                )
    return hits


def _add_hit(
    hits: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    *,
    passage_id: str,
    claim_id: str,
    category_id: str,
    reason: str,
    detail: str,
) -> None:
    if not passage_id:
        return
    key = (passage_id, claim_id, reason)
    if key in seen:
        return
    seen.add(key)
    hits.append(
        {
            "passage_id": passage_id,
            "claim_id": claim_id,
            "category_id": category_id,
            "reason": reason,
            "detail": detail,
            "preflagged": False,
        }
    )


def apply_category_edits(
    codebook: Mapping[str, Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    properties: Sequence[Mapping[str, Any]],
    dimensions: Sequence[Mapping[str, Any]],
    edits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Revise categories and mark dependents needs_review. History stays on the row."""
    book = {str(key): dict(value) for key, value in codebook.items()}
    by_id = {}
    for key, entry in book.items():
        code_id = str(entry.get("code_id") or key)
        by_id[code_id] = key
    segment_rows = [dict(item) for item in segments]
    property_rows = [dict(item) for item in properties]
    dimension_rows = [dict(item) for item in dimensions]
    for edit in edits:
        code_id = str(edit.get("code_id") or "")
        key = by_id.get(code_id)
        if key is None:
            raise KeyError(f"Category edit names an unknown code: {code_id}")
        entry = book[key]
        previous = str(entry.get("category_revision_id") or category_revision_id(entry))
        history = list(entry.get("revision_history") or [])
        history.append(
            {
                "category_revision_id": previous,
                "code": entry.get("code"),
                "definition": entry.get("definition"),
                "revision": int(entry.get("revision") or 1),
            }
        )
        if str(edit.get("code") or "").strip():
            entry["code"] = str(edit.get("code"))
        if "definition" in edit and edit.get("definition") is not None:
            entry["definition"] = str(edit.get("definition"))
        entry["revision"] = int(entry.get("revision") or 1) + 1
        entry["revision_history"] = history
        entry["category_revision_id"] = category_revision_id(entry)
        for segment in segment_rows:
            if str(segment.get("code_id") or "") != code_id:
                continue
            reasons = list(segment.get("review_reasons") or [])
            if "category_revised" not in reasons:
                reasons.append("category_revised")
            segment["review_reasons"] = reasons
            segment["needs_review"] = True
        for claim in property_rows + dimension_rows:
            if str(claim.get("category_id") or "") != code_id:
                continue
            flags = list(claim.get("flags") or [])
            if "category_revised" not in flags:
                flags.append("category_revised")
            claim["flags"] = flags
            claim["needs_review"] = True
            claim["previous_category_revision_id"] = claim.get("category_revision_id")
            claim["downstream_eligible"] = False
    return {
        "codebook": book,
        "segments": segment_rows,
        "properties": property_rows,
        "dimensions": dimension_rows,
    }


def apply_claim_review(
    properties: Sequence[Mapping[str, Any]],
    dimensions: Sequence[Mapping[str, Any]],
    decision: Any,
) -> dict[str, list]:
    """Researcher disposition and claim-support labels. {} leaves both as they are."""
    property_rows = [dict(item) for item in properties]
    dimension_rows = [dict(item) for item in dimensions]
    # An empty object cannot be passed as Command(resume=...) : LangGraph treats
    # every dict whose keys are interrupt ids, and an empty dict matches that.
    if (
        decision in (None, True)
        or (isinstance(decision, Mapping) and decision.get("leave_claims") is True and "claim_dispositions" not in decision and "claim_reviews" not in decision and "accept_claims" not in decision and "category_edits" not in decision)
        or decision == {"accept": True}
    ):
        return {"properties": property_rows, "dimensions": dimension_rows}
    if not isinstance(decision, Mapping):
        raise ValueError("Claim review resume must be an object.")
    rows = {str(item.get("entry_id") or ""): item for item in property_rows + dimension_rows}
    if decision.get("accept_claims") is True:
        for item in property_rows + dimension_rows:
            if item.get("evidence_condition") == "anchored" and item.get("disposition") == "proposed":
                item["disposition"] = "accepted"
                item.update(
                    proposal_record_fields(
                        disposition="accepted",
                        evidence_condition="anchored",
                        claim_support=str(item.get("claim_support") or "undefined"),
                        needs_review=bool(item.get("needs_review") or item.get("property_merge_conflict")),
                    )
                )
                if item.get("property_merge_conflict") or "property_merge_conflict" in (item.get("flags") or []):
                    item["downstream_eligible"] = False
    for update in decision.get("claim_dispositions") or []:
        item = rows.get(str(update.get("entry_id") or ""))
        if item is None:
            raise KeyError(f"Unknown claim entry_id: {update.get('entry_id')}")
        disposition = str(update.get("disposition") or "")
        item.update(
            proposal_record_fields(
                disposition=disposition,
                evidence_condition=str(item.get("evidence_condition") or "insufficient"),
                claim_support=str(item.get("claim_support") or "undefined"),
                needs_review=bool(item.get("needs_review") or item.get("property_merge_conflict")),
            )
        )
        if item.get("property_merge_conflict") or "property_merge_conflict" in (item.get("flags") or []):
            item["downstream_eligible"] = False
        if str(item.get("claim_support") or "") == "contradicted":
            item["downstream_eligible"] = False
    for update in decision.get("claim_reviews") or []:
        item = rows.get(str(update.get("entry_id") or ""))
        if item is None:
            raise KeyError(f"Unknown claim entry_id: {update.get('entry_id')}")
        label = str(update.get("claim_support") or "")
        if label not in CLAIM_SUPPORTS or label == "":
            raise ValueError(f"claim_support must be one of {CLAIM_SUPPORTS}")
        item["claim_support"] = label
        item["review_actor"] = "human"
        history = list(item.get("review_history") or [])
        history.append({"actor": "human", "model": None, "claim_support": label})
        item["review_history"] = history
        if label == "contradicted" or item.get("property_merge_conflict") or "property_merge_conflict" in (item.get("flags") or []):
            item["downstream_eligible"] = False
        else:
            item.update(
                proposal_record_fields(
                    disposition=str(item.get("disposition") or "proposed"),
                    evidence_condition=str(item.get("evidence_condition") or "insufficient"),
                    claim_support=label,
                    needs_review=bool(item.get("needs_review")),
                )
            )
    return {"properties": property_rows, "dimensions": dimension_rows}


def load_category_development(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read a category-development state. Relations may be absent. No axial load."""
    if not isinstance(payload, Mapping):
        raise TypeError("category development state must be an object")
    relations = payload["relations"] if "relations" in payload else None
    properties = [dict(item) for item in (payload.get("properties") or [])]
    dimensions = [dict(item) for item in (payload.get("dimensions") or [])]
    return {
        "state_version": int(payload.get("state_version") or STATE_VERSION),
        "stage": str(payload.get("stage") or "focused"),
        "relations": relations,
        "relations_required": False,
        "properties": properties,
        "dimensions": dimensions,
        "revision_cycles": [dict(item) for item in (payload.get("revision_cycles") or [])],
        "views": dict(payload.get("views") or view_notes()),
        "citation_anchoring": citation_anchoring_report(properties + dimensions),
        "claim_support": claim_support_report(properties + dimensions),
    }


def record_revision_cycle(
    development: Mapping[str, Any],
    *,
    cycle: str,
    reason: str,
    passage_ids: Sequence[str] = (),
    claim_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Append a cycle that returns to evidence. Existing claim text stays."""
    allowed = {"return_to_evidence", "revise_claim", "researcher_edit"}
    if cycle not in allowed:
        raise ValueError(f"revision cycle must be one of {sorted(allowed)}")
    updated = {
        "state_version": int(development.get("state_version") or STATE_VERSION),
        "stage": development.get("stage") or "focused",
        "relations": development.get("relations", None),
        "relations_required": False,
        "properties": [dict(item) for item in (development.get("properties") or [])],
        "dimensions": [dict(item) for item in (development.get("dimensions") or [])],
        "revision_cycles": [dict(item) for item in (development.get("revision_cycles") or [])],
        "views": dict(development.get("views") or view_notes()),
    }
    updated["revision_cycles"].append(
        {
            "cycle": cycle,
            "reason": reason,
            "passage_ids": [str(item) for item in passage_ids],
            "claim_ids": [str(item) for item in claim_ids],
        }
    )
    if cycle == "return_to_evidence":
        wanted = {str(item) for item in claim_ids}
        for bucket in ("properties", "dimensions"):
            for row in updated[bucket]:
                if wanted and str(row.get("entry_id") or "") not in wanted:
                    continue
                flags = list(row.get("flags") or [])
                if "return_to_evidence" not in flags:
                    flags.append("return_to_evidence")
                row["flags"] = flags
                row["needs_review"] = True
                row["downstream_eligible"] = False
    return updated


def plan_linked_run(
    *,
    parent_run_id: str,
    parent_output_dir: str,
    additional_passages: Sequence[Mapping[str, Any]],
    sampling_brief: str,
    new_run_id: str,
) -> dict[str, Any]:
    """Plan a new run that adds data. The original run is referenced and not rewritten."""
    return {
        "run_id": new_run_id,
        "kind": "linked_run",
        "parent_run_id": parent_run_id,
        "parent_output_dir": parent_output_dir,
        "preserves_parent": True,
        "mutates_parent": False,
        "sampling_brief": sampling_brief,
        "additional_passage_ids": [str(item.get("passage_id") or "") for item in additional_passages],
        "additional_passages": [dict(item) for item in additional_passages],
        "relations_required": False,
        "category_development_state_version": STATE_VERSION,
        "note": (
            "The original run stays as stored. This plan is the route for a later run "
            "whose sampling brief names data the original corpus does not contain."
        ),
    }


def code_id_sidecar(codes: Sequence[Mapping[str, Any]], tags: Sequence[str] | None = None) -> dict[str, Any]:
    """Map Taguette tag text to stable code ids. The Taguette CSV itself stays label-only."""
    rows = []
    for index, entry in enumerate(codes):
        tag = str(entry.get("tag") or (tags[index] if tags and index < len(tags) else entry.get("code") or ""))
        rows.append(
            {
                "code_id": str(entry.get("code_id") or ""),
                "tag": tag,
                "category_revision_id": str(entry.get("category_revision_id") or category_revision_id(entry)),
            }
        )
    return {"version": 1, "codes": rows}


def classify_codebook_roundtrip(
    sidecar: Mapping[str, Any],
    imported_tags: Sequence[str],
    approved_mapping: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Tell a rename apart from a removal plus an addition.

    A tag that disappears is kept and flagged until an approved mapping says it
    was renamed or removed. A new tag is not treated as that rename unless the
    mapping says so.
    """
    previous = [dict(item) for item in (sidecar.get("codes") or [])]
    imported = [str(item) for item in imported_tags]
    imported_set = set(imported)
    mapping = dict(approved_mapping or {})
    renames = {str(item.get("code_id") or ""): str(item.get("tag") or "") for item in (mapping.get("renames") or [])}
    removals = {str(item) for item in (mapping.get("removals") or [])}
    additions = [dict(item) for item in (mapping.get("additions") or [])]
    actions: list[dict[str, Any]] = []
    explained: set[str] = set()
    for row in previous:
        code_id = str(row.get("code_id") or "")
        tag = str(row.get("tag") or "")
        if code_id in renames:
            new_tag = renames[code_id]
            actions.append(
                {
                    "action": "rename",
                    "code_id": code_id,
                    "old_tag": tag,
                    "tag": new_tag,
                    "category_revision_id": row.get("category_revision_id"),
                }
            )
            explained.add(new_tag)
            continue
        if code_id in removals:
            actions.append(
                {
                    "action": "remove_approved",
                    "code_id": code_id,
                    "tag": tag,
                    "kept": True,
                }
            )
            continue
        if tag in imported_set:
            actions.append({"action": "unchanged", "code_id": code_id, "tag": tag})
            explained.add(tag)
            continue
        actions.append(
            {
                "action": "remove_kept",
                "code_id": code_id,
                "tag": tag,
                "kept": True,
                "flag": "researcher_removed",
                "note": "Missing from the imported file. Not deleted. A rename needs an approved mapping.",
            }
        )
    for tag in imported:
        if tag in explained:
            continue
        addition = next((item for item in additions if str(item.get("tag") or "") == tag), None)
        if addition is not None:
            actions.append(
                {
                    "action": "add",
                    "tag": tag,
                    "code_id": str(addition.get("code_id") or ""),
                }
            )
            continue
        actions.append(
            {
                "action": "ambiguous",
                "tag": tag,
                "note": "New tag without an approved addition. Not applied as a rename of a missing code.",
            }
        )
    return {"actions": actions}


def read_incidents_csv(path: str | Path) -> list[dict[str, str]]:
    """Researcher-selected incidents. Columns are passage_id and purpose."""
    incidents = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            incidents.append(
                {
                    "passage_id": str(row.get("passage_id") or "").strip(),
                    "purpose": str(row.get("purpose") or "").strip(),
                }
            )
    return incidents


def parse_incidents(decision: Any, known_ids: set[str]) -> list[dict[str, str]]:
    """Incidents from a resume object and, when named, a CSV. Unknown ids raise."""
    if not isinstance(decision, Mapping):
        return []
    rows: list[dict[str, str]] = []
    csv_path = decision.get("incidents_csv")
    if csv_path:
        rows.extend(read_incidents_csv(str(csv_path)))
    for item in decision.get("incidents") or []:
        if isinstance(item, Mapping):
            rows.append(
                {
                    "passage_id": str(item.get("passage_id") or "").strip(),
                    "purpose": str(item.get("purpose") or "").strip(),
                }
            )
    parsed = []
    for item in rows:
        passage_id = item["passage_id"]
        purpose = item["purpose"]
        if purpose not in REREAD_PURPOSES:
            raise ValueError(
                f"Incident purpose must be one of {REREAD_PURPOSES}. Got {purpose!r} for {passage_id}."
            )
        if passage_id not in known_ids:
            raise ValueError(f"Incident passage is not in the corpus: {passage_id}")
        parsed.append({"passage_id": passage_id, "purpose": purpose, "source": "researcher"})
    return parsed


def plan_relation_pairs(
    category_ids: Sequence[str],
    *,
    examined: Sequence[Sequence[str]] = (),
    round_size: int = 8,
    obvious_share: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Choose the next axial pairs and report the pairs this round does not examine.

    Categories are taken in the order given, which is a salience order. Adjacent
    ids are the obvious band. Other pairs are the less-obvious band. A round
    mixes both. Fifty categories have 1,225 unordered pairs; one allowance does
    not examine them. Allowances are ceilings, not measured costs.
    """
    if round_size < 0:
        raise ValueError("round_size must be zero or more.")
    if not 0 <= obvious_share <= 1:
        raise ValueError("obvious_share must be between 0 and 1.")
    ids = [str(item) for item in category_ids]
    seen_ids: list[str] = []
    for item in ids:
        if item not in seen_ids:
            seen_ids.append(item)
    pairs = [(left, right) for index, left in enumerate(seen_ids) for right in seen_ids[index + 1 :]]
    position = {item: index for index, item in enumerate(seen_ids)}
    obvious = [(left, right) for left, right in pairs if abs(position[left] - position[right]) == 1]
    less = [(left, right) for left, right in pairs if abs(position[left] - position[right]) != 1]
    rng = random.Random(int(seed))
    less = list(less)
    rng.shuffle(less)
    examined_keys = {tuple(sorted((str(pair[0]), str(pair[1])))) for pair in examined}
    def pending(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return [pair for pair in rows if tuple(sorted(pair)) not in examined_keys]
    obvious_pending = pending(obvious)
    less_pending = pending(less)
    if obvious_pending:
        offset = int(seed) % len(obvious_pending)
        if offset:
            obvious_pending = obvious_pending[offset:] + obvious_pending[:offset]
    take_obvious = int(round(round_size * obvious_share))
    take_less = round_size - take_obvious
    chosen_obvious = obvious_pending[:take_obvious]
    chosen_less = less_pending[:take_less]
    short_obvious = take_obvious - len(chosen_obvious)
    short_less = take_less - len(chosen_less)
    if short_obvious > 0:
        chosen_less = less_pending[: take_less + short_obvious]
    if short_less > 0:
        chosen_obvious = obvious_pending[: take_obvious + short_less]
    chosen = [("obvious", pair) for pair in chosen_obvious] + [("less_obvious", pair) for pair in chosen_less]
    chosen_keys = {tuple(sorted(pair)) for _band, pair in chosen}
    unexamined = [pair for pair in pairs if tuple(sorted(pair)) not in examined_keys | chosen_keys]
    examined_count = len(examined_keys | chosen_keys)
    return {
        "pair_count": len(pairs),
        "examined_count": examined_count,
        "pair_coverage": None if not pairs else round(examined_count / len(pairs), 4),
        "pair_coverage_statement": f"{examined_count} of {len(pairs)} unordered pairs examined",
        "round": [
            {"from_id": left, "to_id": right, "band": band}
            for band, (left, right) in chosen
        ],
        "unexamined_pair_count": len(unexamined),
        "unexamined_pairs": [{"from_id": left, "to_id": right} for left, right in unexamined],
        "allowance_note": "Call and token allowances are ceilings, not measured costs.",
    }


def dumps_state(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
