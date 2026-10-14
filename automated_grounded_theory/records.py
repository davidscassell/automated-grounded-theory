"""Versioned evidence records, proposal reduction, and the commit owner.

Workers propose against a frozen snapshot. Only :func:`commit_round` writes
the next snapshot, and it ignores a proposal whose snapshot id is stale.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Mapping, Sequence

POLICY_VERSION = "0.7.6"

COMPARISON_OPS = (
    "alias_duplicate",
    "refine_definition",
    "split_code",
    "link_to_category",
    "record_dimension",
    "relate_categories",
    "keep_distinct",
    "defer",
)

# relate_categories is accepted by the schema and stored. It does not build a theory.
SCHEMA_ONLY_OPS = frozenset({"relate_categories"})

UNCERTAINTY_FLAGS = (
    "ambiguous_span",
    "insufficient_context",
    "competing_code",
    "unsupported_inference",
)

COVERAGE_STATUSES = ("coded", "context_only", "excluded", "failed", "pending")


def definition_revision(codebook: Mapping[str, Mapping[str, Any]] | None) -> str:
    """Hash of code ids, revisions, and definitions. Frequency is not included."""
    rows = []
    for key in sorted(codebook or {}):
        entry = codebook[key]
        rows.append(
            "|".join(
                (
                    str(entry.get("code_id") or key),
                    str(entry.get("revision") or 1),
                    str(entry.get("definition") or ""),
                )
            )
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def evidence_revision(segments: Sequence[Mapping[str, Any]] | None) -> str:
    """Hash of assignment coordinates. A new span changes it; a count does not."""
    rows = []
    for segment in segments or []:
        rows.append(
            "|".join(
                (
                    str(segment.get("segment_id") or ""),
                    str(segment.get("doc_id") or ""),
                    str(segment.get("doc_start")),
                    str(segment.get("doc_end")),
                    str(segment.get("code_id") or ""),
                )
            )
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def freeze_snapshot(state: Mapping[str, Any], *, version: int) -> dict[str, Any]:
    """Immutable coordinates for one coding round. Models are not stored."""
    codebook = state.get("codebook") or {}
    segments = state.get("segments") or []
    return {
        "snapshot_id": f"s{int(version):05d}",
        "snapshot_version": int(version),
        "definition_revision": definition_revision(codebook),
        "evidence_revision": evidence_revision(segments),
        "policy_version": POLICY_VERSION,
    }


def proposal_id(snapshot_id: str, passage_id: str) -> str:
    return f"{snapshot_id}:{passage_id}"


def _reset(value: Any) -> bool:
    if isinstance(value, Mapping) and value.get("__reset__"):
        return True
    return False


def _items_of(value: Any) -> list[dict]:
    if _reset(value) or value is None:
        return []
    if isinstance(value, Mapping):
        raw = value.get("items") or []
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    return [dict(item) for item in raw if isinstance(item, Mapping) and not item.get("__reset__")]


def _same_payload(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    def dump(item: Mapping[str, Any]) -> str:
        coding = item.get("coding")
        if hasattr(coding, "model_dump"):
            coding = coding.model_dump()
        body = {
            "passage_index": item.get("passage_index"),
            "doc_id": item.get("doc_id"),
            "coding": coding,
        }
        return json.dumps(body, sort_keys=True, default=str)

    return dump(left) == dump(right)


def proposal_buffer_reducer(left: Any, right: Any) -> dict[str, Any]:
    """Dedupe by proposal id, drop conflicts, sort, and honor a real reset.

    An empty list does not clear the buffer. Only ``{"__reset__": True}`` does.
    """
    if _reset(right):
        return {"items": [], "duplicates": 0, "conflicts": []}
    base = left if isinstance(left, Mapping) else {}
    seen: dict[str, dict] = {}
    blocked: set[str] = set(str(item) for item in (base.get("conflicts") or []))
    duplicates = int(base.get("duplicates") or 0)
    conflicts = [str(item) for item in (base.get("conflicts") or [])]
    for item in _items_of(base) + _items_of(right):
        pid = str(item.get("proposal_id") or "")
        if not pid or pid in blocked:
            if pid and pid in blocked:
                duplicates += 1
            continue
        previous = seen.get(pid)
        if previous is None:
            seen[pid] = item
            continue
        duplicates += 1
        if not _same_payload(previous, item):
            blocked.add(pid)
            conflicts.append(pid)
            seen.pop(pid, None)
    ordered = sorted(
        seen.values(),
        key=lambda item: (
            str(item.get("snapshot_id") or ""),
            int(item.get("passage_index") or 0),
            str(item.get("proposal_id") or ""),
        ),
    )
    return {"items": ordered, "duplicates": duplicates, "conflicts": conflicts}


def select_committable(
    proposals: Sequence[Mapping[str, Any]],
    *,
    snapshot_id: str,
    committed_ids: Sequence[str] | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Split fresh proposals from stale ones. Already committed ids are skipped."""
    done = set(committed_ids or [])
    fresh: list[dict] = []
    stale: list[dict] = []
    skipped: list[str] = []
    for item in proposals:
        pid = str(item.get("proposal_id") or "")
        if pid and pid in done:
            skipped.append(pid)
            continue
        if str(item.get("snapshot_id") or "") != str(snapshot_id):
            stale.append(dict(item))
            continue
        fresh.append(dict(item))
    return fresh, stale, skipped


def _judged_same_concept(item: Mapping[str, Any]) -> bool:
    """True when a sweep accepted an alias or said the pair is one concept."""
    if str(item.get("comparison_op") or "") == "alias_duplicate" and str(item.get("decision") or "") == "accept":
        return True
    return item.get("same_action") is True and item.get("same_object") is True


def same_round_duplicates(
    codebook: Mapping[str, Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
) -> list[dict]:
    """Pairs from one snapshot that an alias or a same-concept judgment later joins.

    Label identity is not enough. Two proposals count when they were created
    against the same frozen snapshot and a later comparison says they are one
    concept, including when that comparison is then kept distinct.
    """
    from automated_grounded_theory.validation import code_key

    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in judgments:
        if not _judged_same_concept(item):
            continue
        source = code_key(str(item.get("source") or ""))
        target = code_key(str(item.get("target") or ""))
        if not source or not target or source == target:
            continue
        left = codebook.get(source) or {}
        right = codebook.get(target) or {}
        snapshot = str(left.get("origin_snapshot_id") or "")
        if not snapshot or snapshot != str(right.get("origin_snapshot_id") or ""):
            continue
        key = tuple(sorted((source, target)))
        if key in seen:
            continue
        seen.add(key)
        found.append(
            {
                "snapshot_id": snapshot,
                "source": str(left.get("code") or item.get("source") or source),
                "target": str(right.get("code") or item.get("target") or target),
                "judgment": "alias_duplicate"
                if str(item.get("decision") or "") == "accept"
                else "same_concept",
                "rationale": str(item.get("rationale") or ""),
            }
        )
    return found


def accumulate_duplicate_rounds(existing: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Group same-round duplicate pairs by snapshot. A pair is stored once."""
    grouped: dict[str, dict] = {}
    for row in existing:
        snapshot = str(row.get("snapshot_id") or "")
        if not snapshot:
            continue
        grouped[snapshot] = {"snapshot_id": snapshot, "pairs": [dict(item) for item in row.get("pairs") or []]}
    for pair in pairs:
        snapshot = str(pair.get("snapshot_id") or "")
        if not snapshot:
            continue
        bucket = grouped.setdefault(snapshot, {"snapshot_id": snapshot, "pairs": []})
        known = {tuple(sorted((str(item.get("source")), str(item.get("target"))))) for item in bucket["pairs"]}
        key = tuple(sorted((str(pair.get("source")), str(pair.get("target")))))
        if key in known:
            continue
        bucket["pairs"].append(dict(pair))
    return [grouped[key] for key in sorted(grouped)]


def duplicate_pair_count(rounds: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(row.get("pairs") or []) for row in rounds)


def component_aliases_are_direct(members: Sequence[str], accepted_pairs: set[frozenset[str]]) -> bool:
    """A merge of three or more codes needs every pair judged, not a chain."""
    names = sorted({str(item) for item in members if item})
    if len(names) <= 2:
        return True
    for left, right in combinations(names, 2):
        if frozenset((left, right)) not in accepted_pairs:
            return False
    return True


def apply_comparisons(
    codebook: dict[str, dict],
    segments: list[dict],
    operations: Sequence[Mapping[str, Any]],
    *,
    snapshot_id: str = "",
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any]:
    """Apply one comparison list. Alias merges are checked per component.

    ``relate_categories`` is stored and does not create categories.
    Valence is recorded and does not block any operation.
    """
    from automated_grounded_theory.codebook import apply_sweep

    book = {key: dict(value) for key, value in codebook.items()}
    rows = [dict(segment) for segment in segments]
    applied: list[dict] = []
    recorded: list[dict] = []
    deferred: list[dict] = []
    rejected: list[dict] = []
    lost: list[dict] = []
    aliases: list[dict] = []
    for raw in operations:
        op = str(raw.get("op") or raw.get("operation") or "").strip()
        if op not in COMPARISON_OPS:
            rejected.append({**dict(raw), "reason": "unknown_operation"})
            continue
        record = {
            "op": op,
            "snapshot_id": snapshot_id,
            "policy_version": policy_version,
            "reason": str(raw.get("reason") or raw.get("rationale") or ""),
            "source": raw.get("source") or raw.get("source_id") or "",
            "target": raw.get("target") or raw.get("target_id") or "",
            "valence": raw.get("valence") or raw.get("source_valence") or "",
            "status": "applied",
        }
        if op == "defer":
            record["status"] = "deferred"
            deferred.append(record)
            continue
        if op in SCHEMA_ONLY_OPS:
            record["status"] = "recorded"
            record["relation"] = str(raw.get("relation") or "")
            recorded.append(record)
            continue
        if op == "alias_duplicate":
            aliases.append(dict(raw))
            continue
        if op == "keep_distinct":
            record["status"] = "applied"
            _note(book, raw, "distinct_from", raw.get("target") or raw.get("other") or "")
            applied.append(record)
            continue
        if op == "link_to_category":
            category = str(raw.get("category") or raw.get("category_id") or "shared")
            for label in _labels(raw):
                key = _key(book, label)
                if key is None:
                    continue
                links = list(book[key].get("category_links") or [])
                if category not in links:
                    links.append(category)
                book[key]["category_links"] = links
            record["category"] = category
            applied.append(record)
            continue
        if op == "record_dimension":
            key = _key(book, raw.get("code") or raw.get("source") or "")
            if key is None:
                rejected.append({**record, "reason": "unknown_code"})
                continue
            dimensions = list(book[key].get("dimensions") or [])
            dimensions.append(
                {
                    "name": str(raw.get("dimension") or raw.get("name") or "stance"),
                    "value": str(raw.get("value") or raw.get("valence") or ""),
                    "span_ids": list(raw.get("span_ids") or []),
                }
            )
            book[key]["dimensions"] = dimensions
            applied.append(record)
            continue
        if op == "refine_definition":
            key = _key(book, raw.get("code") or raw.get("source") or "")
            if key is None:
                rejected.append({**record, "reason": "unknown_code"})
                continue
            book[key]["definition"] = str(raw.get("definition") or book[key].get("definition") or "")
            book[key]["revision"] = int(book[key].get("revision") or 1) + 1
            if raw.get("inclusion") is not None:
                book[key]["inclusion"] = str(raw.get("inclusion") or "")
            if raw.get("exclusion") is not None:
                book[key]["exclusion"] = str(raw.get("exclusion") or "")
            applied.append(record)
            continue
        if op == "split_code":
            key = _key(book, raw.get("code") or raw.get("source") or "")
            if key is None:
                rejected.append({**record, "reason": "unknown_code"})
                continue
            book[key]["lifecycle"] = "split"
            book[key]["revision"] = int(book[key].get("revision") or 1) + 1
            book[key]["split_rationale"] = str(raw.get("reason") or raw.get("rationale") or "")
            applied.append(record)
            continue
        rejected.append({**record, "reason": "unhandled_operation"})

    if aliases:
        sweep = apply_sweep(
            book,
            rows,
            merges=[
                {
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "source_id": item.get("source_id") or "",
                    "target_id": item.get("target_id") or "",
                    "confidence": item.get("confidence", 1.0),
                    "rationale": item.get("reason") or item.get("rationale") or "alias_duplicate",
                    "comparison_op": "alias_duplicate",
                }
                for item in aliases
            ],
            candidates=[],
            max_merges_per_sweep=max(1, len(aliases)),
        )
        book = sweep["codebook"]
        rows = sweep["segments"]
        applied.extend(sweep.get("accepted_merges") or [])
        rejected.extend(sweep.get("rejected_merges") or [])
        lost.extend(sweep.get("lost_distinction_candidates") or [])
    return {
        "codebook": book,
        "segments": rows,
        "applied": applied,
        "recorded": recorded,
        "deferred": deferred,
        "rejected": rejected,
        "lost_distinction_candidates": lost,
        "policy_version": policy_version,
    }


def _labels(raw: Mapping[str, Any]) -> list[str]:
    labels = list(raw.get("codes") or [])
    for name in ("source", "target", "code"):
        if raw.get(name):
            labels.append(raw.get(name))
    return [str(item) for item in labels if str(item).strip()]


def _key(book: Mapping[str, Mapping[str, Any]], label: Any) -> str | None:
    from automated_grounded_theory.validation import code_key

    key = code_key(str(label or ""))
    if key in book:
        return key
    for entry_key, entry in book.items():
        if str(entry.get("code_id") or "") == str(label):
            return entry_key
    return None


def _note(book: dict[str, dict], raw: Mapping[str, Any], field: str, value: Any) -> None:
    key = _key(book, raw.get("source") or raw.get("code") or "")
    if key is None or not value:
        return
    current = list(book[key].get(field) or [])
    text = str(value)
    if text not in current:
        current.append(text)
    book[key][field] = current


def materialize_records(
    *,
    documents: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    codebook: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> dict[str, list]:
    """Build Document, Incident, EvidenceSpan, and CodeVersion rows from a run."""
    docs = []
    for item in documents:
        text = str(item.get("text") or "")
        docs.append(
            {
                "doc_id": str(item.get("doc_id") or item.get("id") or ""),
                "respondent_id": str((item.get("metadata") or {}).get("respondent_id") or item.get("doc_id") or ""),
                "question": str(item.get("question") or (item.get("metadata") or {}).get("question") or ""),
                "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata": dict(item.get("metadata") or {}),
                "order": item.get("order"),
            }
        )
    incidents = []
    for passage in passages:
        incidents.append(
            {
                "incident_id": str(passage.get("passage_id") or ""),
                "doc_id": str(passage.get("doc_id") or ""),
                "question": str(passage.get("question") or ""),
                "start": passage.get("start_offset"),
                "end": passage.get("end_offset"),
                "text_sha256": hashlib.sha256(str(passage.get("text") or "").encode("utf-8")).hexdigest(),
                "coverage_status": passage.get("coverage_status") or ("excluded" if passage.get("skipped") else "pending"),
                "exclusion_reason": passage.get("exclusion_reason") or "",
                "nonsubstantive": bool(passage.get("nonsubstantive")),
                "nonsubstantive_kind": str(passage.get("nonsubstantive_kind") or ""),
            }
        )
    spans = []
    for segment in segments:
        spans.append(
            {
                "span_id": str(segment.get("highlight_id") or segment.get("segment_id") or ""),
                "doc_id": str(segment.get("doc_id") or ""),
                "incident_id": str(segment.get("passage_id") or ""),
                "start": segment.get("doc_start"),
                "end": segment.get("doc_end"),
                "code_id": str(segment.get("code_id") or ""),
                "quote_status": segment.get("quote_status") or "",
                "uncertainty_flags": list(segment.get("uncertainty_flags") or []),
            }
        )
    versions = []
    entries: list[Mapping[str, Any]]
    if isinstance(codebook, Mapping):
        entries = list(codebook.values())
    else:
        entries = list(codebook)
    for entry in entries:
        versions.append(
            {
                "code_id": str(entry.get("code_id") or ""),
                "revision": int(entry.get("revision") or 1),
                "label": str(entry.get("code") or ""),
                "definition": str(entry.get("definition") or ""),
                "inclusion": str(entry.get("inclusion") or ""),
                "exclusion": str(entry.get("exclusion") or ""),
                "anchor_span_ids": list(entry.get("anchor_span_ids") or []),
                "contrast_span_ids": list(entry.get("contrast_span_ids") or []),
                "lifecycle": str(entry.get("lifecycle") or "active"),
                "category_links": list(entry.get("category_links") or []),
            }
        )
    return {
        "documents": docs,
        "incidents": incidents,
        "evidence_spans": spans,
        "code_versions": versions,
    }


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 4)
