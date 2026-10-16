"""Respondent-level negative-case retrieval.

A negative case is a respondent who shows one side of a link, an opposite
pole, or a different outcome. Passage overlap with both endpoints is one
signal. It is not the only one. The 2x2 counts describe the corpus. They do
not validate or invalidate a link. Model calls are spent only on candidates
that pass a lexical or embedding pre-filter. When the budget is exhausted and
candidates remain, the search records ``budget`` and does not invent a
fallback call.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from automated_grounded_theory.relationship_tiers import case_id_of
from automated_grounded_theory.retrieval import cosine
from automated_grounded_theory.validation import GENERIC_CONTENT_STEMS, meaningful_tokens

COSINE_THRESHOLD = 0.55
_COUNTER_MARKERS = (
    " however ",
    " although ",
    " though ",
    " except ",
    " unless ",
    " instead ",
    " rather than ",
    " not always ",
    " rarely ",
    " seldom ",
    " no issues ",
    " did not ",
    " didn't ",
    " does not ",
    " doesn't ",
    " was not ",
    " wasn't ",
    " exception ",
    " contrary ",
    " on the other hand ",
    " but ",
)
_KIND_RANK = {
    "contrasting_outcome": 0,
    "opposite_pole": 1,
    "a_without_b": 2,
    "b_without_a": 2,
    "both_endpoints": 3,
    "embedding": 4,
}
_BOTH_SOURCES = frozenset({"contrasting_outcome", "both_endpoints", "embedding"})
_DESCRIPTIVE = (
    "Descriptive respondent counts only. They do not validate or invalidate a relationship."
)


def _distinctive(tokens: set[str]) -> set[str]:
    return {token for token in tokens if token not in GENERIC_CONTENT_STEMS}


def _entry_text(entry: Mapping[str, Any] | None) -> str:
    if not isinstance(entry, Mapping):
        return ""
    return " ".join(str(entry.get(name) or "") for name in ("code", "definition", "label"))


def _codebook_entry(codebook: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None, code_id: str) -> Mapping[str, Any] | None:
    if not code_id or codebook is None:
        return None
    if isinstance(codebook, Mapping):
        row = codebook.get(code_id)
        return row if isinstance(row, Mapping) else None
    for row in codebook:
        if isinstance(row, Mapping) and str(row.get("code_id") or "") == code_id:
            return row
    return None


def endpoint_token_sets(
    relationship: Mapping[str, Any],
    codebook: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    segments: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[set[str], set[str]]:
    """Meaningful stems for each endpoint, including generic stems."""
    sets: list[set[str]] = []
    for key in ("from_id", "to_id"):
        code_id = str(relationship.get(key) or "")
        parts = [_entry_text(_codebook_entry(codebook, code_id))]
        for segment in segments or []:
            if not isinstance(segment, Mapping) or str(segment.get("code_id") or "") != code_id:
                continue
            parts.append(str(segment.get("quote") or segment.get("text") or ""))
        sets.append(meaningful_tokens(" ".join(parts)))
    return sets[0], sets[1]


def _vector(row: Mapping[str, Any] | None) -> list[float] | None:
    if not isinstance(row, Mapping):
        return None
    for key in ("embedding", "vector"):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and value:
            try:
                return [float(item) for item in value]
            except (TypeError, ValueError):
                return None
    return None


def _cosine(passage: Mapping[str, Any], claim_vector: Sequence[float] | None) -> float:
    left = _vector(passage)
    if not left or not claim_vector:
        return 0.0
    try:
        return float(cosine(list(left), list(claim_vector)))
    except (TypeError, ValueError):
        return 0.0


def _has_counter(text: str) -> bool:
    padded = f" {' '.join(str(text or '').casefold().split())} "
    return any(marker in padded for marker in _COUNTER_MARKERS)


def _tier_number(row: Mapping[str, Any]) -> int | None:
    if "evidence_tier" not in row or row.get("evidence_tier") is None:
        return None
    try:
        return int(row.get("evidence_tier"))
    except (TypeError, ValueError):
        return None


def _empty_matrix() -> dict[str, Any]:
    return {
        "both": None,
        "a_only": None,
        "b_only": None,
        "neither": None,
        "descriptive": True,
        "note": _DESCRIPTIVE,
    }


def _matrix(both: int, a_only: int, b_only: int, neither: int) -> dict[str, Any]:
    return {
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "descriptive": True,
        "note": _DESCRIPTIVE,
    }


def _passage_kind(
    text: str,
    *,
    hits_a: bool,
    hits_b: bool,
    semantic: float,
    coded_a: bool,
    coded_b: bool,
) -> str:
    """The strongest reason this passage is a negative-case candidate."""
    present_a = hits_a or coded_a
    present_b = hits_b or coded_b
    counter = _has_counter(text)
    if present_a and present_b and counter and hits_a and hits_b:
        return "contrasting_outcome"
    if counter and ((hits_a and not hits_b) or (hits_b and not hits_a)):
        return "opposite_pole"
    if present_a and not present_b:
        return "a_without_b"
    if present_b and not present_a:
        return "b_without_a"
    if hits_a and hits_b:
        return "both_endpoints"
    if semantic >= COSINE_THRESHOLD:
        return "embedding"
    return ""


def _respondent_rank(passages: Sequence[dict[str, Any]]) -> tuple:
    best = min(passages, key=lambda item: (item["rank"], item["passage_id"]))
    return (best["rank"], -int(best["score"]), str(best["respondent_id"]))


def search_relationships(
    relationships: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    *,
    slots: int,
    codebook: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    segments: Sequence[Mapping[str, Any]] | None = None,
    dimensions: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Queue plausible negative-case passages and record the search.

    Tier 0 rows are category evidence and are not searched. A missing tier is
    still searched. One passage is taken from each plausible respondent before
    a second passage from the same respondent.
    """
    del dimensions  # reserved for a later pole table; lexical poles are read from the text
    by_id = {
        str(item.get("passage_id") or ""): item
        for item in passages
        if isinstance(item, Mapping) and item.get("passage_id") and not item.get("synthetic")
    }
    coded: dict[str, set[str]] = {}
    for segment in segments or []:
        if not isinstance(segment, Mapping):
            continue
        passage_id = str(segment.get("passage_id") or "")
        code_id = str(segment.get("code_id") or "")
        if passage_id and code_id:
            coded.setdefault(passage_id, set()).add(code_id)
    queued: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    per_row: dict[str, Any] = {}
    skipped_none = 0
    skipped_low = 0
    considered = 0
    filtered = 0
    filter_reasons: dict[str, int] = {}
    opposite_found = 0
    searched: set[str] = set()
    plausible_total = 0

    def add(relationship_id: str, passage_id: str, source: str, respondent_id: str) -> None:
        if len(queued) >= slots or not relationship_id or not passage_id:
            return
        key = (relationship_id, passage_id)
        if key in seen or passage_id not in by_id:
            return
        seen.add(key)
        queued.append(
            {
                "relationship_id": relationship_id,
                "passage_id": passage_id,
                "source": source,
                "respondent_id": respondent_id,
            }
        )

    for relationship in relationships:
        if not isinstance(relationship, Mapping):
            continue
        relationship_id = str(relationship.get("relationship_id") or "")
        tier = _tier_number(relationship)
        if tier == 0:
            per_row[relationship_id] = {
                "relationship_id": relationship_id,
                "skipped": True,
                "no_call_reason": "tier_0_category_evidence",
                "candidates_considered": 0,
                "filtered_count": 0,
                "queued": 0,
                "cooccurrence_2x2": _empty_matrix(),
                "note": "Tier 0 is category evidence. It is not sent to negative-case search.",
            }
            continue
        cited = str(relationship.get("passage_id") or "")
        from_id = str(relationship.get("from_id") or "")
        to_id = str(relationship.get("to_id") or "")
        left, right = endpoint_token_sets(relationship, codebook, segments)
        topics_left, topics_right = _distinctive(left), _distinctive(right)
        claim_vector = _vector(relationship) or _vector(by_id.get(cited))
        by_respondent: dict[str, dict[str, Any]] = {}
        row_filtered = 0
        row_reasons: dict[str, int] = {}

        def touch(passage_id: str, passage: Mapping[str, Any]) -> dict[str, Any]:
            respondent_id = case_id_of(passage) or str(passage.get("doc_id") or passage_id)
            return by_respondent.setdefault(
                respondent_id,
                {"respondent_id": respondent_id, "passages": [], "a": False, "b": False, "plausible": []},
            )

        if cited and cited in by_id:
            cited_passage = by_id[cited]
            slot = touch(cited, cited_passage)
            cited_tokens = meaningful_tokens(str(cited_passage.get("text") or ""))
            cited_codes = coded.get(cited, set())
            if from_id in cited_codes or bool(topics_left and cited_tokens & topics_left):
                slot["a"] = True
            if to_id in cited_codes or bool(topics_right and cited_tokens & topics_right):
                slot["b"] = True
        for passage_id, passage in by_id.items():
            if passage_id == cited:
                continue
            slot = touch(passage_id, passage)
            respondent_id = str(slot["respondent_id"])
            slot = by_respondent.setdefault(
                respondent_id,
                {"respondent_id": respondent_id, "passages": [], "a": False, "b": False, "plausible": []},
            )
            text = str(passage.get("text") or "")
            tokens = meaningful_tokens(text)
            hits_a = bool(topics_left) and bool(tokens & topics_left)
            hits_b = bool(topics_right) and bool(tokens & topics_right)
            codes = coded.get(passage_id, set())
            coded_a = from_id in codes
            coded_b = to_id in codes
            if hits_a or coded_a:
                slot["a"] = True
            if hits_b or coded_b:
                slot["b"] = True
            semantic = _cosine(passage, claim_vector)
            kind = _passage_kind(
                text,
                hits_a=hits_a,
                hits_b=hits_b,
                semantic=semantic,
                coded_a=coded_a,
                coded_b=coded_b,
            )
            generic_only = bool(tokens & ((left | right) - (topics_left | topics_right))) and not (hits_a or hits_b)
            if not kind:
                reason = "low_relevance" if generic_only or (0 < semantic < COSINE_THRESHOLD) else "no_endpoint_signal"
                row_filtered += 1
                row_reasons[reason] = row_reasons.get(reason, 0) + 1
                if reason == "low_relevance":
                    skipped_low += 1
                else:
                    skipped_none += 1
                slot["passages"].append({"passage_id": passage_id, "kind": "", "filtered": True, "reason": reason})
                continue
            score = len(tokens & topics_left) + len(tokens & topics_right)
            if _has_counter(text):
                score += 2
            slot["plausible"].append(
                {
                    "passage_id": passage_id,
                    "kind": kind,
                    "rank": _KIND_RANK[kind],
                    "score": score,
                    "respondent_id": respondent_id,
                    "semantic": round(semantic, 4),
                }
            )
            slot["passages"].append({"passage_id": passage_id, "kind": kind, "filtered": False})
        both = a_only = b_only = neither = 0
        plausible_respondents = []
        for respondent_id, slot in by_respondent.items():
            if slot["a"] and slot["b"]:
                both += 1
            elif slot["a"]:
                a_only += 1
            elif slot["b"]:
                b_only += 1
            else:
                neither += 1
            if not slot["passages"] and not slot["plausible"]:
                continue
            considered += 1
            if slot["plausible"]:
                plausible_respondents.append(slot)
            else:
                filtered += 1
        for reason, count in row_reasons.items():
            filter_reasons[reason] = filter_reasons.get(reason, 0) + count
        opposite_found += sum(
            1
            for slot in plausible_respondents
            for item in slot["plausible"]
            if item["kind"] == "opposite_pole"
        )
        ordered: list[dict[str, Any]] = []
        ranked_respondents = sorted(
            plausible_respondents,
            key=lambda slot: _respondent_rank(slot["plausible"]),
        )
        rounds = max((len(slot["plausible"]) for slot in ranked_respondents), default=0)
        for index in range(rounds):
            for slot in ranked_respondents:
                chosen = sorted(slot["plausible"], key=lambda item: (item["rank"], -int(item["score"]), item["passage_id"]))
                if index < len(chosen):
                    ordered.append(chosen[index])
        row_queued = []
        if relationship_id:
            searched.add(relationship_id)
        for item in ordered:
            before = len(queued)
            add(relationship_id, str(item["passage_id"]), str(item["kind"]), str(item["respondent_id"]))
            if len(queued) > before:
                row_queued.append(queued[-1])
        plausible_total += len(ordered)
        if not ordered:
            reason = "no_plausible_candidates"
        elif slots <= 0:
            reason = "budget"
        elif not row_queued:
            reason = "budget"
        else:
            reason = None
        per_row[relationship_id] = {
            "relationship_id": relationship_id,
            "skipped": False,
            "candidates_considered": len(by_respondent),
            "filtered_count": sum(1 for slot in by_respondent.values() if not slot["plausible"]),
            "passages_filtered": row_filtered,
            "filter_reasons": row_reasons,
            "plausible": len(ordered),
            "queued": len(row_queued),
            "queued_passages": list(row_queued),
            "no_call_reason": reason,
            "cooccurrence_2x2": _matrix(both, a_only, b_only, neither) if by_id else _empty_matrix(),
            "sources": sorted({item["source"] for item in row_queued}),
        }
    research = [item for item in queued if item.get("source") != "synthetic_counterexample"]
    if not research and slots <= 0 and plausible_total:
        no_call = "budget"
    elif not research and plausible_total:
        no_call = "budget"
    elif not research:
        no_call = "no_plausible_candidates"
    else:
        no_call = None
    notes = []
    if skipped_none:
        notes.append(
            f"Negative-case search filtered {skipped_none} passages with no endpoint signal."
        )
    if skipped_low:
        notes.append(
            f"Negative-case search filtered {skipped_low} passages with only weak overlap."
        )
    if no_call == "budget":
        notes.append(
            "Negative-case search made no research calls because the call budget was exhausted. "
            f"{plausible_total} plausible candidates were not sent."
        )
    elif no_call == "no_plausible_candidates":
        notes.append(
            "Negative-case search made no research calls because no respondent passed the pre-filter."
        )
    elif research:
        notes.append(
            "Negative-case search queued "
            f"{len(research)} passages from respondents with one-sided evidence, an opposite pole, "
            "a different outcome, both endpoints, or an embedding match."
        )
    stats = {
        "queued": len(queued),
        "skipped_no_relevant_passage": skipped_none,
        "skipped_low_relevance": skipped_low,
        "notes": notes,
        "searched_relationship_ids": sorted(item for item in searched if item),
        "both_endpoint_queued": sum(1 for item in research if item.get("source") in _BOTH_SOURCES),
        "opposite_pole_candidates": opposite_found,
        "opposite_pole_fallback_queued": 0,
        "no_call_reason": no_call,
        "fallback": None,
        "fallback_reason": None,
        "candidates_considered": considered,
        "filtered_count": filtered,
        "filter_reasons": filter_reasons,
        "plausible_candidates": plausible_total,
        "by_relationship": per_row,
        "cooccurrence_2x2": _empty_matrix() if not per_row else next(
            (
                row["cooccurrence_2x2"]
                for row in per_row.values()
                if isinstance(row.get("cooccurrence_2x2"), Mapping) and row["cooccurrence_2x2"].get("both") is not None
            ),
            _empty_matrix(),
        ),
    }
    return queued, stats


def _claim_terms(claim: Mapping[str, Any]) -> set[str]:
    return _distinctive(
        meaningful_tokens(" ".join(str(claim.get(key) or "") for key in ("pole", "text", "quote", "name")))
    )


def _category_terms(entry: Mapping[str, Any] | None) -> set[str]:
    return _distinctive(meaningful_tokens(_entry_text(entry)))


def claim_respondent_search(
    claims: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]] | None = None,
    codebook: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Respondent candidates for a claim. Recording them does not spend a call.

    A respondent coded to the category whose text does not state the claim, a
    respondent on an opposite pole, and a respondent who mentions the topic
    and a different outcome are candidates. The 2x2 is descriptive.
    """
    by_id = {
        str(item.get("passage_id") or ""): item
        for item in passages
        if isinstance(item, Mapping) and item.get("passage_id") and not item.get("synthetic")
    }
    assigned: dict[str, set[str]] = {}
    for segment in segments or []:
        if not isinstance(segment, Mapping):
            continue
        code_id = str(segment.get("code_id") or "")
        passage_id = str(segment.get("passage_id") or "")
        if code_id and passage_id:
            assigned.setdefault(passage_id, set()).add(code_id)
    poles: dict[str, set[str]] = {}
    entries = list(codebook.values()) if isinstance(codebook, Mapping) else list(codebook or [])
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for dimension in entry.get("dimensions") or []:
            if not isinstance(dimension, Mapping):
                continue
            name = " ".join(str(dimension.get("name") or "").casefold().split())
            pole = " ".join(str(dimension.get("value") or dimension.get("pole") or "").casefold().split())
            if name and pole:
                poles.setdefault(name, set()).add(pole)
    plausible: list[dict[str, str]] = []
    considered = 0
    filtered = 0
    per_claim = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        claim_id = str(claim.get("entry_id") or claim.get("claim_id") or "")
        category_id = str(claim.get("category_id") or "")
        outcome = _claim_terms(claim)
        category = _category_terms(_codebook_entry(codebook, category_id))
        pole = " ".join(str(claim.get("pole") or "").casefold().split())
        name = " ".join(str(claim.get("name") or "").casefold().split())
        other_poles = set(poles.get(name, set())) - ({pole} if pole else set())
        cited = str(claim.get("passage_id") or "")
        both = a_only = b_only = neither = 0
        respondents: dict[str, dict[str, Any]] = {}
        for passage_id, passage in by_id.items():
            respondent_id = case_id_of(passage) or passage_id
            slot = respondents.setdefault(
                respondent_id, {"a": False, "b": False, "passages": []}
            )
            text = str(passage.get("text") or "")
            tokens = meaningful_tokens(text)
            has_category = category_id in assigned.get(passage_id, set()) or bool(category and tokens & category)
            has_outcome = bool(outcome) and bool(tokens & outcome)
            folded = f" {' '.join(text.casefold().split())} "
            opposite = any(item and item in folded for item in other_poles)
            if has_category:
                slot["a"] = True
            if has_outcome or opposite:
                slot["b"] = True
            kind = ""
            if opposite or (_has_counter(text) and (has_category or bool(tokens & (category | outcome)))):
                kind = "opposite_pole" if opposite or not (has_category and has_outcome) else "contrasting_outcome"
                if has_category and has_outcome and _has_counter(text):
                    kind = "contrasting_outcome"
                elif opposite:
                    kind = "opposite_pole"
            elif has_category and not has_outcome and passage_id != cited:
                kind = "category_without_outcome"
            elif has_outcome and not has_category and passage_id != cited:
                kind = "outcome_without_category"
            if kind:
                slot["passages"].append({"passage_id": passage_id, "kind": kind, "respondent_id": respondent_id, "claim_id": claim_id})
        for slot in respondents.values():
            considered += 1
            if slot["a"] and slot["b"]:
                both += 1
            elif slot["a"]:
                a_only += 1
            elif slot["b"]:
                b_only += 1
            else:
                neither += 1
            if not slot["passages"]:
                filtered += 1
                continue
            best = sorted(slot["passages"], key=lambda item: item["passage_id"])[0]
            plausible.append(best)
        per_claim.append(
            {
                "claim_id": claim_id,
                "candidates_considered": len(respondents),
                "cooccurrence_2x2": _matrix(both, a_only, b_only, neither) if by_id else _empty_matrix(),
            }
        )
    plausible_rows = [item for item in plausible if item.get("passage_id")]
    if not plausible_rows:
        reason = "no_plausible_candidates"
    else:
        reason = None
    return {
        "candidates_considered": considered,
        "filtered_count": filtered,
        "plausible": plausible_rows,
        "plausible_count": len(plausible_rows),
        "no_call_reason": reason,
        "by_claim": per_claim,
        "cooccurrence_2x2": per_claim[0]["cooccurrence_2x2"] if per_claim else _empty_matrix(),
        "note": _DESCRIPTIVE,
    }
