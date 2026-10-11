"""Apply passage coding and sweep decisions to the running codebook.

Counts are rebuilt from segments after every commit so a merge cannot
double-count applications.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from automated_grounded_theory.validation import (
    assess_label,
    code_key,
    locate_quote,
    review_passage_flags,
)

_ACTION_ALIASES = {
    "reuse": "reuse",
    "refine": "refine",
    "new": "new",
    "create": "new",
    "add": "new",
    "update": "refine",
}


def empty_counters() -> dict[str, int]:
    return {"segment": 0, "memo": 0, "highlight": 0, "sweep": 0}


def _next(counters: dict[str, int], name: str, prefix: str) -> str:
    counters[name] = int(counters.get(name, 0)) + 1
    return f"{prefix}{counters[name]:05d}"


def interpret_coding(
    coding: Mapping[str, Any] | Any,
    passage: Mapping[str, Any],
    codebook: Mapping[str, Mapping[str, Any]],
    *,
    max_codes: int,
    require_gerund: bool,
    long_passage_chars: int,
) -> dict[str, Any]:
    """Validate one model response. Does not mutate the codebook."""
    payload = coding.model_dump() if hasattr(coding, "model_dump") else dict(coding)
    proposals = list(payload.get("codes") or [])
    rejections: list[dict] = []
    accepted: list[dict] = []
    seen_keys: set[str] = set()
    low_confidence = bool(payload.get("low_confidence"))
    passage_text = str(passage.get("text") or "")

    if len(proposals) > max_codes:
        for extra in proposals[max_codes:]:
            rejections.append(
                {
                    "code": extra.get("code", ""),
                    "reason": "over_max_codes",
                    "quote": extra.get("quote", ""),
                }
            )
        proposals = proposals[:max_codes]

    known = {code_key(str(entry.get("code") or entry.get("key") or "")): entry for entry in codebook.values()}

    for proposal in proposals:
        label = str(proposal.get("code") or "")
        quote = str(proposal.get("quote") or "")
        span = locate_quote(quote, passage_text)
        if span is None:
            rejections.append({"code": label, "reason": "quote_not_verbatim", "quote": quote})
            continue
        start, end = span
        verbatim = passage_text[start:end]
        raw_action = str(proposal.get("action") or "").strip().casefold()
        action = _ACTION_ALIASES.get(raw_action)
        provisional_key = code_key(label)
        if action is None:
            action = "reuse" if provisional_key in known or provisional_key in seen_keys else "new"
        exists = provisional_key in known or provisional_key in seen_keys
        if action == "new" and exists:
            action = "reuse"
        if action in {"reuse", "refine"} and not exists:
            action = "new"

        in_vivo = bool(proposal.get("in_vivo"))
        ok, reasons, fields = assess_label(
            label,
            in_vivo=in_vivo,
            passage=passage_text,
            require_gerund=require_gerund,
        )
        if not ok:
            rejections.append({"code": label, "reason": reasons[0] if reasons else "rejected", "quote": verbatim})
            continue

        refined = proposal.get("refined_label")
        refined_label = str(refined).strip() if refined else ""
        if action == "refine" and refined_label:
            refined_ok, refined_reasons, refined_fields = assess_label(
                refined_label,
                in_vivo=in_vivo,
                passage=passage_text,
                require_gerund=require_gerund,
            )
            if not refined_ok:
                rejections.append(
                    {
                        "code": refined_label,
                        "reason": refined_reasons[0] if refined_reasons else "rejected",
                        "quote": verbatim,
                    }
                )
                continue
            fields = refined_fields
            reasons = list(reasons) + [reason for reason in refined_reasons if reason not in reasons]

        try:
            confidence = float(proposal.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
            reasons.append("confidence_unparsed")
        confidence = min(1.0, max(0.0, confidence))
        review = review_passage_flags(
            passage_text,
            quote=verbatim,
            long_passage_chars=long_passage_chars,
            low_confidence=low_confidence,
            confidence=confidence,
        )
        review.extend(reasons)
        definition = str(proposal.get("definition") or "").strip()
        if action == "new" and not definition:
            definition = str(proposal.get("rationale") or "").strip() or "Definition not yet specified."
            review.append("missing_definition")

        key = fields["key"]
        if key in seen_keys and action == "new":
            action = "reuse"
        seen_keys.add(key)
        accepted.append(
            {
                "action": action,
                "key": key,
                "code": fields["code"],
                "previous_key": provisional_key if action == "refine" and key != provisional_key else None,
                "definition": definition,
                "quote": verbatim,
                "local_start": start,
                "local_end": end,
                "doc_start": int(passage["start_offset"]) + start,
                "doc_end": int(passage["start_offset"]) + end,
                "in_vivo": fields["in_vivo"],
                "gerund_ok": fields["gerund_ok"],
                "rationale": str(proposal.get("rationale") or ""),
                "confidence": confidence,
                "needs_review": bool(review),
                "review_reasons": review,
            }
        )

    if not accepted and len(passage_text) >= long_passage_chars:
        rejections.append({"code": "", "reason": "uncoded_long_passage", "quote": ""})

    return {
        "accepted": accepted,
        "rejections": rejections,
        "memo": str(payload.get("memo") or "").strip(),
        "what_is_happening": str(payload.get("what_is_happening") or "").strip(),
        "low_confidence": low_confidence,
    }


def commit_passage(
    codebook: dict[str, dict],
    segments: list[dict],
    memos: list[dict],
    highlight_ids: dict[str, str],
    counters: dict[str, int],
    passage: Mapping[str, Any],
    interpreted: Mapping[str, Any],
    *,
    existing_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Append segments for accepted codes. ``existing_keys`` tracks globally new labels."""
    book = deepcopy(codebook)
    new_segments = [dict(segment) for segment in segments]
    new_memos = list(memos)
    highlights = dict(highlight_ids)
    counters = dict(counters)
    known_before = set(existing_keys) if existing_keys is not None else set(book)
    new_codes = 0

    what = interpreted.get("what_is_happening") or ""
    memo_body = interpreted.get("memo") or ""
    text = "\n\n".join(part for part in (what.strip(), memo_body.strip()) if part).strip()
    if not text:
        text = "No analytic memo was returned for this passage."
    memo_id = _next(counters, "memo", "m")
    new_memos.append(
        {
            "memo_id": memo_id,
            "kind": "passage",
            "doc_id": passage.get("doc_id"),
            "passage_id": passage.get("passage_id"),
            "sweep_index": None,
            "text": text,
        }
    )

    for item in interpreted.get("accepted") or []:
        key = item["key"]
        previous = item.get("previous_key")
        if previous and previous in book and previous != key:
            book = _rename_code(book, new_segments, previous, key, item["code"])
        if key not in book:
            if key not in known_before:
                new_codes += 1
                known_before.add(key)
            book[key] = {
                "code": item["code"],
                "key": key,
                "definition": item["definition"],
                "in_vivo": item["in_vivo"],
                "gerund_ok": item["gerund_ok"],
                "frequency": 0,
                "document_ids": [],
                "examples": [],
                "first_seen": passage.get("passage_id"),
                "history": [
                    {
                        "event": "created",
                        "passage_id": passage.get("passage_id"),
                        "definition": item["definition"],
                        "action": item["action"],
                    }
                ],
                "flagged_overloaded": False,
                "relation_hints": [],
            }
        elif item["action"] == "refine" and item["definition"]:
            book[key]["definition"] = item["definition"]
            book[key]["history"].append(
                {
                    "event": "refined",
                    "passage_id": passage.get("passage_id"),
                    "definition": item["definition"],
                }
            )
            book[key]["in_vivo"] = bool(book[key].get("in_vivo") or item["in_vivo"])
            book[key]["gerund_ok"] = item["gerund_ok"]
        elif item["in_vivo"]:
            book[key]["in_vivo"] = True

        span_key = f"{passage.get('doc_id')}:{item['doc_start']}:{item['doc_end']}"
        if span_key not in highlights:
            highlights[span_key] = _next(counters, "highlight", "h")
        segment = {
            "segment_id": _next(counters, "segment", "s"),
            "highlight_id": highlights[span_key],
            "passage_id": passage.get("passage_id"),
            "doc_id": passage.get("doc_id"),
            "doc_name": passage.get("doc_name"),
            "code": book[key]["code"],
            "code_key": key,
            "quote": item["quote"],
            "doc_start": item["doc_start"],
            "doc_end": item["doc_end"],
            "in_vivo": item["in_vivo"],
            "action": item["action"],
            "rationale": item["rationale"],
            "confidence": item["confidence"],
            "needs_review": item["needs_review"],
            "review_reasons": list(item["review_reasons"]),
            "memo_id": memo_id,
        }
        new_segments.append(segment)

    book = sync_counts(book, new_segments)
    return {
        "codebook": book,
        "segments": new_segments,
        "memos": new_memos,
        "highlight_ids": highlights,
        "counters": counters,
        "new_codes": new_codes,
    }


def _rename_code(codebook: dict[str, dict], segments: list[dict], old: str, new: str, display: str) -> dict[str, dict]:
    book = codebook
    if old not in book:
        return book
    entry = book.pop(old)
    entry["key"] = new
    entry["code"] = display
    entry.setdefault("history", []).append({"event": "renamed", "from": old, "to": new})
    if new in book:
        survivor = book[new]
        survivor.setdefault("history", []).extend(entry.get("history") or [])
        survivor["history"].append({"event": "merged_from", "code": entry.get("code"), "key": old})
        if entry.get("definition") and not survivor.get("definition"):
            survivor["definition"] = entry["definition"]
    else:
        book[new] = entry
    for segment in segments:
        if segment.get("code_key") == old:
            segment["code_key"] = new
            segment["code"] = book[new]["code"] if new in book else display
    return book


def sync_counts(codebook: dict[str, dict], segments: list[dict]) -> dict[str, dict]:
    """Rebuild frequency, documents, and examples from ``segments``."""
    grouped: dict[str, list[dict]] = {}
    for segment in segments:
        key = segment.get("code_key") or code_key(str(segment.get("code") or ""))
        grouped.setdefault(key, []).append(segment)
    synced: dict[str, dict] = {}
    for key, entry in codebook.items():
        applications = grouped.get(key) or []
        if not applications:
            continue
        updated = deepcopy(entry)
        updated["key"] = key
        updated["frequency"] = len(applications)
        updated["document_ids"] = sorted({str(item.get("doc_id")) for item in applications if item.get("doc_id")})
        examples = []
        seen_quotes: set[tuple] = set()
        for item in applications:
            marker = (item.get("passage_id"), item.get("quote"))
            if marker in seen_quotes:
                continue
            seen_quotes.add(marker)
            examples.append(
                {
                    "quote": item.get("quote"),
                    "doc_id": item.get("doc_id"),
                    "passage_id": item.get("passage_id"),
                    "doc_start": item.get("doc_start"),
                    "doc_end": item.get("doc_end"),
                }
            )
            if len(examples) >= 8:
                break
        updated["examples"] = examples
        if applications and not updated.get("first_seen"):
            updated["first_seen"] = applications[0].get("passage_id")
        label = str(updated.get("code") or "")
        if updated.get("in_vivo") and not any(
            code_key(label) and label.casefold() in str(item.get("quote") or "").casefold() for item in examples
        ):
            # Keep the flag only when some stored quote still contains the label.
            from automated_grounded_theory.validation import in_vivo_supported

            updated["in_vivo"] = any(in_vivo_supported(label, str(item.get("quote") or "")) for item in examples)
        synced[key] = updated
    return synced


def union_codebooks(books: list[dict[str, dict]]) -> dict[str, dict]:
    """Combine per-document codebooks. Counts are left for :func:`sync_counts`."""
    united: dict[str, dict] = {}
    for book in books:
        for key, entry in book.items():
            if key not in united:
                united[key] = deepcopy(entry)
                continue
            current = united[key]
            current.setdefault("history", []).extend(entry.get("history") or [])
            if len(str(entry.get("definition") or "")) > len(str(current.get("definition") or "")):
                current["definition"] = entry["definition"]
            current["in_vivo"] = bool(current.get("in_vivo") or entry.get("in_vivo"))
            current["flagged_overloaded"] = bool(current.get("flagged_overloaded") or entry.get("flagged_overloaded"))
            hints = list(current.get("relation_hints") or [])
            for hint in entry.get("relation_hints") or []:
                if hint not in hints:
                    hints.append(hint)
            current["relation_hints"] = hints
    return united


def candidate_pair_keys(candidates: list[Mapping[str, Any]]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for item in candidates:
        source = code_key(str(item.get("source") or ""))
        target = code_key(str(item.get("target") or ""))
        if source and target and source != target:
            pairs.add(frozenset((source, target)))
    return pairs


def apply_sweep(
    codebook: dict[str, dict],
    segments: list[dict],
    *,
    merges: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]],
    definitions: list[Mapping[str, Any]] | None = None,
    overloaded: list[str] | None = None,
    relation_hints: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Confirm candidate merges, sharpen definitions, and relabel segments.

    A merge is applied only when its unordered pair was a candidate. The
    sweep model confirms; it does not invent pairs.
    """
    allowed = candidate_pair_keys(candidates)
    parent = {key: key for key in codebook}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    accepted: list[dict] = []
    rejected: list[dict] = []
    for merge in merges:
        source = code_key(str(merge.get("source") or ""))
        target = code_key(str(merge.get("target") or ""))
        record = {"source": merge.get("source"), "target": merge.get("target"), "rationale": merge.get("rationale", "")}
        if source not in parent or target not in parent or source == target:
            rejected.append({**record, "reason": "unknown_or_same"})
            continue
        if frozenset((source, target)) not in allowed:
            rejected.append({**record, "reason": "not_a_candidate"})
            continue
        root_source, root_target = find(source), find(target)
        if root_source == root_target:
            rejected.append({**record, "reason": "already_merged"})
            continue
        parent[root_source] = root_target
        accepted.append(record)

    book: dict[str, dict] = {}
    merge_map: dict[str, str] = {}
    for key, entry in codebook.items():
        root = find(key)
        merge_map[key] = root
        if root not in book:
            survivor = deepcopy(codebook[root])
            survivor["key"] = root
            survivor["code"] = codebook[root]["code"]
            survivor["history"] = list(survivor.get("history") or [])
            book[root] = survivor
        if key != root:
            book[root].setdefault("history", []).append(
                {
                    "event": "merged_from",
                    "code": entry.get("code"),
                    "key": key,
                    "rationale": next(
                        (item.get("rationale", "") for item in accepted if code_key(str(item.get("source") or "")) == key),
                        "",
                    ),
                }
            )

    for update in definitions or []:
        key = code_key(str(update.get("code") or ""))
        if key not in parent:
            rejected.append({"source": update.get("code"), "target": None, "reason": "unknown_definition"})
            continue
        root = find(key)
        definition = str(update.get("definition") or "").strip()
        if root in book and definition:
            book[root]["definition"] = definition
            book[root].setdefault("history", []).append({"event": "definition_sharpened", "definition": definition})

    for label in overloaded or []:
        key = code_key(str(label))
        if key not in parent:
            continue
        root = find(key)
        if root in book:
            book[root]["flagged_overloaded"] = True
            book[root].setdefault("history", []).append({"event": "flagged_overloaded", "code": label})

    new_segments = []
    for segment in segments:
        updated = dict(segment)
        key = updated.get("code_key") or code_key(str(updated.get("code") or ""))
        if key in parent:
            root = find(key)
            updated["code_key"] = root
            updated["code"] = book[root]["code"]
        new_segments.append(updated)

    book = sync_counts(book, new_segments)
    hints_applied = []
    for hint in relation_hints or []:
        source = code_key(str(hint.get("source") or ""))
        target = code_key(str(hint.get("target") or ""))
        if source not in parent or target not in parent:
            continue
        source_root, target_root = find(source), find(target)
        if source_root not in book or target_root not in book:
            continue
        record = {
            "source": book[source_root]["code"],
            "target": book[target_root]["code"],
            "relation": str(hint.get("relation") or "related"),
            "note": str(hint.get("note") or ""),
        }
        for root in {source_root, target_root}:
            current = book[root].setdefault("relation_hints", [])
            if record not in current:
                current.append(record)
        hints_applied.append(record)

    display_map = {}
    for key, root in merge_map.items():
        if key != root and root in book and key in codebook:
            display_map[codebook[key]["code"]] = book[root]["code"]

    return {
        "codebook": book,
        "segments": new_segments,
        "accepted_merges": accepted,
        "rejected_merges": rejected,
        "merge_map": display_map,
        "relation_hints": hints_applied,
    }


def codebook_list(codebook: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = [dict(entry) for entry in codebook.values()]
    rows.sort(key=lambda entry: (-int(entry.get("frequency") or 0), str(entry.get("code") or "").casefold()))
    return rows
