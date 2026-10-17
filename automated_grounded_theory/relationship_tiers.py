"""Tiered evidence for a relationship between two categories.

A category can be present on one anchored quote. A relationship is a
hypothesis until the same link is anchored inside more than one case.
Counts describe that pattern. They do not validate it. Causal wording
waits for an attribution and a rival check. ``verified`` is a human
promotion with a named scope. This module never sets that tier on its own.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from automated_grounded_theory.validation import GENERIC_CONTENT_STEMS, locate_quote, meaningful_tokens

PATTERN_MIN_CASES = 2
PATTERN_MIN_CASES_NOTE = (
    "A design choice for this package. It is not a quotation from the method literature."
)
TIER_LABELS = {
    0: "category_evidence",
    1: "hypothesis",
    2: "pattern_supported",
    3: "attribution",
    4: "verified",
}
_CAUSAL_WORD = frozenset({"reduce", "lead", "cause", "foster", "result", "produce", "because", "consequence", "effect"})
_CAUSAL_PHRASES = (
    "leads to",
    "lead to",
    "results in",
    "result of",
    "results from",
    "as a result",
    "because",
    "caused",
    "causes",
    "due to",
    "effect of",
    "consequence",
    "consequences of",
)
_CAUSAL_ROLES = frozenset({"consequence", "cause"})
_ATTRIBUTION = (
    "because",
    "due to",
    "leads to",
    "lead to",
    "caused",
    "causes",
    "results in",
    "result of",
    "effect of",
    "consequence",
    "attributed to",
)
_GROUNDED_STEMS = 2
_WHY = ("because", "due to", "the reason", "why ")
_RELEVANT = frozenset({"supports", "contradicts", "qualifies"})


def allowed_wording(tier: int | None, n: int | None = None) -> str | None:
    """The wording a relationship at this tier may use. None when the tier is unknown."""
    if tier is None:
        return None
    if tier <= 0:
        return "a category is present"
    if tier == 1:
        return "may be related"
    if tier == 2:
        count = "N" if n is None else str(n)
        return f"co-occurs in {count} respondents"
    if tier == 3:
        return "participants attribute X to Y"
    return "verified within the recorded scope"


def case_id_of(passage: Mapping[str, Any] | None) -> str:
    """Respondent when the passage names one, otherwise the document."""
    if not isinstance(passage, Mapping):
        return ""
    for key in ("respondent_id", "participant_id", "case_id"):
        value = str(passage.get(key) or "").strip()
        if value:
            return value
    for key in ("doc_id", "document_id"):
        value = str(passage.get(key) or "").strip()
        if value:
            return value
    return str(passage.get("passage_id") or "")


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def _verbatim(quote: str, text: str) -> bool:
    needle = _norm(quote)
    if len(needle) < 4:
        return False
    return needle in _norm(text)


def _distinctive(text: str) -> set[str]:
    return {token for token in meaningful_tokens(text) if token not in GENERIC_CONTENT_STEMS}


def _entry(codebook: Mapping[str, Any] | None, code_id: str) -> Mapping[str, Any] | None:
    if not code_id or not isinstance(codebook, Mapping):
        return None
    row = codebook.get(code_id)
    return row if isinstance(row, Mapping) else None


def _labels(entry: Mapping[str, Any] | None, code_id: str) -> set[str]:
    labels = set()
    if code_id:
        labels.add(_norm(code_id))
    if isinstance(entry, Mapping):
        for key in ("code", "code_id", "label"):
            text = _norm(str(entry.get(key) or ""))
            if text:
                labels.add(text)
    labels.discard("")
    return labels


def _endpoint_terms(
    code_id: str,
    codebook: Mapping[str, Any] | None,
    segments: Sequence[Mapping[str, Any]] | None,
) -> set[str]:
    entry = _entry(codebook, code_id)
    parts = []
    if isinstance(entry, Mapping):
        parts.append(" ".join(str(entry.get(name) or "") for name in ("code", "definition", "label")))
    for segment in segments or []:
        if not isinstance(segment, Mapping) or str(segment.get("code_id") or "") != code_id:
            continue
        parts.append(str(segment.get("quote") or segment.get("text") or ""))
    return _distinctive(" ".join(parts))


def _usable_passages(passages: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for passage_id, passage in (passages or {}).items():
        if not isinstance(passage, Mapping) or passage.get("synthetic"):
            continue
        if str(passage.get("passage_id") or passage_id) in {"synthetic-counterexample", "p-contrast"}:
            continue
        rows[str(passage_id)] = passage
    return rows


def _locate(quote: str, passages: Mapping[str, Mapping[str, Any]], cited: str = "") -> str:
    if cited and cited in passages and _verbatim(quote, str(passages[cited].get("text") or "")):
        return cited
    for passage_id, passage in passages.items():
        if _verbatim(quote, str(passage.get("text") or "")):
            return str(passage_id)
    return ""


def _is_label(quote: str, labels: set[str]) -> bool:
    return _norm(quote) in labels


def _quote_row(
    endpoint: str,
    quote: str,
    passage_id: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "quote": quote,
        "passage_id": passage_id,
        "quote_passage_id": passage_id or None,
        "quote_start": start,
        "quote_end": end,
        "anchored": True,
        "verbatim": True,
    }


def _add_quote(
    bucket: dict[str, list[dict[str, Any]]],
    *,
    endpoint: str,
    quote: str,
    passage_id: str,
    passages: Mapping[str, Mapping[str, Any]],
    terms: set[str],
    labels: set[str],
) -> None:
    text = str(quote or "").strip()
    if not text or not terms or _is_label(text, labels):
        return
    overlap = _distinctive(text) & terms
    if not overlap:
        return
    found = _locate(text, passages, passage_id)
    if not found:
        return
    passage = passages.get(found) or {}
    case_id = case_id_of(passage)
    if not case_id:
        return
    span = locate_quote(text, str(passage.get("text") or ""))
    row = _quote_row(
        endpoint,
        text,
        found,
        start=None if span is None else span[0],
        end=None if span is None else span[1],
    )
    row["grounded"] = len(overlap) >= _GROUNDED_STEMS
    row["matched_stems"] = sorted(overlap)
    existing = bucket.setdefault(case_id, [])
    key = (endpoint, _norm(text), found)
    if key not in {(item["endpoint"], _norm(item["quote"]), item["passage_id"]) for item in existing}:
        existing.append(row)


def _collect_quotes(
    relationship: Mapping[str, Any],
    passages: Mapping[str, Mapping[str, Any]],
    codebook: Mapping[str, Any] | None,
    segments: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Per-case verbatim quotes for each endpoint. A category label is not a quote."""
    found: dict[str, dict[str, list[dict[str, Any]]]] = {"from": {}, "to": {}}
    ends = {}
    for endpoint, key in (("from", "from_id"), ("to", "to_id")):
        code_id = str(relationship.get(key) or "")
        ends[endpoint] = {
            "code_id": code_id,
            "terms": _endpoint_terms(code_id, codebook, segments),
            "labels": _labels(_entry(codebook, code_id), code_id),
        }
    cited = str(relationship.get("passage_id") or "")
    shared = str(relationship.get("quote") or "").strip()
    named = {
        "from": str(relationship.get("quote_from") or "").strip(),
        "to": str(relationship.get("quote_to") or "").strip(),
    }
    for endpoint, meta in ends.items():
        quotes = []
        if named[endpoint]:
            quotes.append((named[endpoint], cited))
        elif shared:
            quotes.append((shared, cited))
        for item in relationship.get("case_quotes") or []:
            if not isinstance(item, Mapping):
                continue
            side = str(item.get("endpoint") or "both")
            if side not in {endpoint, "both"}:
                continue
            quotes.append((str(item.get("quote") or ""), str(item.get("passage_id") or cited)))
        for segment in segments or []:
            if not isinstance(segment, Mapping) or str(segment.get("code_id") or "") != meta["code_id"]:
                continue
            segment_quote = str(segment.get("quote") or segment.get("text") or "").strip()
            if segment_quote:
                quotes.append((segment_quote, str(segment.get("passage_id") or "")))
        for quote, passage_id in quotes:
            _add_quote(
                found[endpoint],
                endpoint=endpoint,
                quote=quote,
                passage_id=passage_id,
                passages=passages,
                terms=meta["terms"],
                labels=meta["labels"],
            )
    return found


def _supporting_cases(found: Mapping[str, Mapping[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    case_ids = set(found.get("from") or {}) | set(found.get("to") or {})
    rows = []
    for case_id in sorted(case_ids):
        quotes = list((found.get("from") or {}).get(case_id) or []) + list((found.get("to") or {}).get(case_id) or [])
        passage_ids = []
        for item in quotes:
            if item["passage_id"] not in passage_ids:
                passage_ids.append(item["passage_id"])
        from_passages = {
            item["passage_id"] for item in quotes if item["endpoint"] == "from" and item.get("grounded", True)
        }
        to_passages = {
            item["passage_id"] for item in quotes if item["endpoint"] == "to" and item.get("grounded", True)
        }
        both = bool(from_passages) and bool(to_passages)
        same = bool(from_passages & to_passages)
        rows.append(
            {
                "case_id": case_id,
                "passage_ids": passage_ids,
                "quotes": quotes,
                "both_endpoints": both,
                "same_passage": same,
                "within_case": both,
            }
        )
    return rows


def _cooccurrence(
    relationship: Mapping[str, Any],
    passages: Mapping[str, Mapping[str, Any]] | None,
    codebook: Mapping[str, Any] | None,
    segments: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    empty = {
        "both": None,
        "a_only": None,
        "b_only": None,
        "neither": None,
        "descriptive": True,
        "note": "Descriptive co-occurrence only. It is not a validity claim and it does not raise the evidence tier.",
    }
    usable = _usable_passages(passages)
    if passages is None or not usable:
        return empty
    from_id = str(relationship.get("from_id") or "")
    to_id = str(relationship.get("to_id") or "")
    from_terms = _endpoint_terms(from_id, codebook, None)
    to_terms = _endpoint_terms(to_id, codebook, None)
    if not from_terms or not to_terms:
        return empty
    by_case: dict[str, dict[str, Any]] = {}
    for passage_id, passage in usable.items():
        case_id = case_id_of(passage)
        if not case_id:
            continue
        slot = by_case.setdefault(case_id, {"passages": [], "a": False, "b": False})
        slot["passages"].append(passage_id)
        tokens = meaningful_tokens(str(passage.get("text") or ""))
        if tokens & from_terms:
            slot["a"] = True
        if tokens & to_terms:
            slot["b"] = True
    for segment in segments or []:
        if not isinstance(segment, Mapping):
            continue
        passage = usable.get(str(segment.get("passage_id") or ""))
        if passage is None:
            continue
        case_id = case_id_of(passage)
        slot = by_case.get(case_id)
        if slot is None:
            continue
        code_id = str(segment.get("code_id") or "")
        if code_id == from_id:
            slot["a"] = True
        elif code_id == to_id:
            slot["b"] = True
    both = a_only = b_only = neither = 0
    for slot in by_case.values():
        if slot["a"] and slot["b"]:
            both += 1
        elif slot["a"]:
            a_only += 1
        elif slot["b"]:
            b_only += 1
        else:
            neither += 1
    return {
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "descriptive": True,
        "note": "Descriptive co-occurrence only. It is not a validity claim and it does not raise the evidence tier.",
    }


def _search_report(relationship: Mapping[str, Any], *, kind: str) -> str:
    if kind == "variation":
        value = relationship.get("variation_search")
        if value is True or value == "reported" or relationship.get("variation_cases"):
            return "reported"
        return "undefined"
    judgments = []
    for case in relationship.get("counterexamples") or []:
        if not isinstance(case, Mapping) or case.get("synthetic"):
            continue
        if str(case.get("passage_id") or "") == "synthetic-counterexample":
            continue
        judgments.append(case)
    relevant = any(
        case.get("valid_test") and str(case.get("judgment") or "") in _RELEVANT for case in judgments
    )
    if relevant:
        return "reported"
    return "undefined"


def _participant_attribution(
    relationship: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    passages: Mapping[str, Mapping[str, Any]],
    from_terms: set[str],
    to_terms: set[str],
) -> bool:
    if not from_terms or not to_terms:
        return False
    quotes = []
    for case in cases:
        quotes.extend(case.get("quotes") or [])
    for key in ("quote", "quote_from", "quote_to"):
        text = str(relationship.get(key) or "").strip()
        if text:
            quotes.append({"quote": text, "passage_id": str(relationship.get("passage_id") or "")})
    for item in quotes:
        text = str(item.get("quote") or "")
        folded = f" {_norm(text)} "
        if not any(marker in folded for marker in _ATTRIBUTION) and not re.search(r"\bmakes\b", folded):
            continue
        tokens = _distinctive(text)
        if not (tokens & from_terms) or not (tokens & to_terms):
            continue
        passage_id = _locate(text, passages, str(item.get("passage_id") or ""))
        if passage_id:
            return True
    return False


def _analyst_attribution(
    relationship: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    memos: Sequence[Mapping[str, Any]] | None,
) -> bool:
    relationship_id = str(relationship.get("relationship_id") or "")
    quote_tokens: set[str] = set()
    for case in cases:
        for item in case.get("quotes") or []:
            quote_tokens |= meaningful_tokens(str(item.get("quote") or ""))
    quote_tokens |= meaningful_tokens(str(relationship.get("quote") or ""))
    if not quote_tokens:
        return False
    for memo in memos or []:
        if not isinstance(memo, Mapping):
            continue
        if str(memo.get("relationship_id") or "") != relationship_id:
            continue
        text = str(memo.get("text") or "")
        folded = f" {_norm(text)} "
        if not any(marker in folded for marker in _WHY):
            continue
        if meaningful_tokens(text) & quote_tokens:
            return True
    return False


def _undefined_fields(pattern_min_cases: int, cooccurrence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "evidence_tier": None,
        "evidence_label": None,
        "allowed_wording": None,
        "cross_case": None,
        "within_case_strength": None,
        "supporting_cases": None,
        "cooccurrence_2x2": cooccurrence
        or {
            "both": None,
            "a_only": None,
            "b_only": None,
            "neither": None,
            "descriptive": True,
            "note": "Descriptive co-occurrence only. It is not a validity claim and it does not raise the evidence tier.",
        },
        "pattern_min_cases": pattern_min_cases,
        "pattern_min_cases_note": PATTERN_MIN_CASES_NOTE,
        "negative_case_search_report": "undefined",
        "variation_search": "undefined",
        "rival_explanation_check": "undefined",
        "attribution_source": None,
        "cases_with_both_endpoints": None,
    }


def assess_relationship(
    relationship: Mapping[str, Any],
    *,
    passages: Mapping[str, Mapping[str, Any]] | None = None,
    codebook: Mapping[str, Any] | None = None,
    segments: Sequence[Mapping[str, Any]] | None = None,
    memos: Sequence[Mapping[str, Any]] | None = None,
    pattern_min_cases: int = PATTERN_MIN_CASES,
) -> dict[str, Any]:
    """Score a relationship up to tier 3. Tier 4 is left to ``promote_verified``."""
    minimum = max(1, int(pattern_min_cases or PATTERN_MIN_CASES))
    if passages is None:
        return _undefined_fields(minimum)
    usable = _usable_passages(passages)
    cooccurrence = _cooccurrence(relationship, usable, codebook, segments)
    found = _collect_quotes(relationship, usable, codebook, segments)
    cases = _supporting_cases(found)
    from_cases = {case["case_id"] for case in cases if any(item["endpoint"] == "from" for item in case["quotes"])}
    to_cases = {case["case_id"] for case in cases if any(item["endpoint"] == "to" for item in case["quotes"])}
    both_cases = [case for case in cases if case.get("both_endpoints")]
    if not from_cases and not to_cases:
        return _undefined_fields(minimum, cooccurrence)
    cross_case = bool(from_cases and to_cases and from_cases != to_cases and not both_cases)
    if any(case.get("same_passage") for case in both_cases):
        strength = "same_passage"
    elif both_cases:
        strength = "within_respondent"
    elif from_cases and to_cases and not both_cases:
        strength = "cross_case_only"
        cross_case = True
    else:
        strength = None
    rival = _search_report(relationship, kind="negative")
    variation = _search_report(relationship, kind="variation")
    from_terms = _endpoint_terms(str(relationship.get("from_id") or ""), codebook, segments)
    to_terms = _endpoint_terms(str(relationship.get("to_id") or ""), codebook, segments)
    attribution = None
    if _participant_attribution(relationship, cases, usable, from_terms, to_terms):
        attribution = "participant"
    elif _analyst_attribution(relationship, both_cases or cases, memos):
        attribution = "analyst_memo"
    if not from_cases or not to_cases:
        tier = 0
    elif len(both_cases) >= minimum:
        tier = 3 if attribution and rival == "reported" else 2
    else:
        tier = 1
    return {
        "evidence_tier": tier,
        "evidence_label": TIER_LABELS[tier],
        "allowed_wording": allowed_wording(tier, len(both_cases) if tier >= 2 else None),
        "cross_case": cross_case if tier == 1 else False,
        "within_case_strength": strength,
        "supporting_cases": cases,
        "cooccurrence_2x2": cooccurrence,
        "pattern_min_cases": minimum,
        "pattern_min_cases_note": PATTERN_MIN_CASES_NOTE,
        "negative_case_search_report": rival,
        "variation_search": variation,
        "rival_explanation_check": rival,
        "attribution_source": attribution if tier >= 2 else None,
        "cases_with_both_endpoints": len(both_cases),
    }


def has_causal_wording(text: str) -> bool:
    """True when ``text`` uses causal or consequence language."""
    if not str(text or "").strip():
        return False
    return _causal(f" {_norm(text)} ")


def denies_causal_language(text: str) -> bool:
    """True when a reviewer says the text does not use causal language."""
    folded = f" {_norm(text)} "
    return any(
        phrase in folded
        for phrase in (
            "no causal",
            "not causal",
            "without causal",
            "no cause ",
            "not a cause",
            "no causal language",
        )
    )


def wording_overreach(text: str, tier: int | None, *, role: str = "") -> bool:
    """True when ``text`` or ``role`` uses wording a higher tier would be needed to license.

    A missing tier is undefined and is not an overreach. Role ``consequence``
    or ``cause`` at tier 1 or 2 is causal wording even when the sentence is mild.
    """
    if tier is None:
        return False
    chosen = str(role or "").strip().casefold().replace(" ", "_")
    if tier < 3 and chosen in _CAUSAL_ROLES:
        return True
    if not str(text or "").strip():
        return False
    folded = f" {_norm(text)} "
    if tier < 4 and re.search(r"\bverified\b", folded):
        return True
    if tier < 3 and (_causal(folded) or re.search(r"\battribut", folded)):
        return True
    if tier < 2 and re.search(r"\bco-?occurs?\b", folded):
        return True
    if tier < 2 and re.search(r"\bsupported\b", folded) and "not supported" not in folded and "unsupported" not in folded:
        return True
    return False


def _causal(folded: str) -> bool:
    if any(phrase in folded for phrase in _CAUSAL_PHRASES):
        return True
    stems = meaningful_tokens(folded)
    return bool(stems & _CAUSAL_WORD)


def apply_evidence_tier(relationship: dict, **kwargs: Any) -> dict[str, Any]:
    """Write the assessment onto the row. Does not decide disposition or tier 4."""
    fields = assess_relationship(relationship, **kwargs)
    relationship.update(fields)
    flags = list(relationship.get("flags") or [])
    tier = fields.get("evidence_tier")
    if tier == 1 and fields.get("cross_case") and "cross_case" not in flags:
        flags.append("cross_case")
    if tier == 1 and "hypothesis_not_supported" not in flags:
        flags.append("hypothesis_not_supported")
    statement = " ".join(
        str(relationship.get(key) or "") for key in ("statement", "human_statement", "model_statement")
    )
    role = str(
        relationship.get("human_paradigm_role") or relationship.get("paradigm_role") or relationship.get("model_paradigm_role") or ""
    )
    if wording_overreach(statement, tier if isinstance(tier, int) else None, role=role) and "wording_exceeds_tier" not in flags:
        flags.append("wording_exceeds_tier")
        relationship["needs_review"] = True
    relationship["flags"] = flags
    disposition = str(relationship.get("disposition") or "")
    if disposition not in {"rejected", "merged"} and isinstance(tier, int):
        relationship["evidence_status"] = fields["evidence_label"]
    return fields


def promote_verified(relationship: dict, *, actor: str, scope: str) -> str:
    """Promote a tier-3 row when a researcher names the scope.

    Returns ``verified``, ``withheld``, ``refused``, or ``missing_scope``.
    A model actor is refused. The row is kept either way.
    """
    if str(actor or "") != "human":
        flags = list(relationship.get("flags") or [])
        if "model_cannot_verify" not in flags:
            flags.append("model_cannot_verify")
        relationship["flags"] = flags
        relationship["needs_review"] = True
        return "refused"
    named = str(scope or "").strip()
    if not named:
        return "missing_scope"
    tier = relationship.get("evidence_tier")
    try:
        tier_n = int(tier)
    except (TypeError, ValueError):
        tier_n = -1
    if tier_n < 3:
        flags = list(relationship.get("flags") or [])
        if "verification_withheld" not in flags:
            flags.append("verification_withheld")
        relationship["flags"] = flags
        relationship["needs_review"] = True
        relationship["verification_scope"] = named
        return "withheld"
    relationship["evidence_tier"] = 4
    relationship["evidence_label"] = "verified"
    relationship["evidence_status"] = "verified"
    relationship["allowed_wording"] = allowed_wording(4)
    relationship["verified_scope"] = named
    relationship["verification_scope"] = named
    relationship["verification_actor"] = "human"
    flags = [str(flag) for flag in (relationship.get("flags") or []) if str(flag) != "verification_withheld"]
    if "verified" not in flags:
        flags.append("verified")
    relationship["flags"] = flags
    if str(relationship.get("evidence_condition") or "") != "disputed":
        relationship["downstream_eligible"] = True
    return "verified"


def citation_qualifies(row: Mapping[str, Any]) -> bool:
    """A tier of 2 or higher whose supporting cases each quote both endpoints."""
    try:
        tier = int(row.get("evidence_tier"))
    except (TypeError, ValueError):
        return False
    if tier < 2:
        return False
    cases = row.get("supporting_cases")
    if not isinstance(cases, list):
        return False
    both = [case for case in cases if isinstance(case, Mapping) and case.get("both_endpoints")]
    minimum = int(row.get("pattern_min_cases") or PATTERN_MIN_CASES)
    if len(both) < minimum:
        return False
    for case in both:
        ends = {str(item.get("endpoint") or "") for item in (case.get("quotes") or []) if isinstance(item, Mapping)}
        verbatim = [
            item
            for item in (case.get("quotes") or [])
            if isinstance(item, Mapping) and item.get("verbatim") and item.get("anchored")
        ]
        if len(verbatim) < 2:
            return False
        if not {"from", "to"} <= ends:
            return False
    return True


def candidate_cross_case_passages(
    passages: Mapping[str, Mapping[str, Any]],
    codebook: Mapping[str, Any] | None,
    pairs: Sequence[Mapping[str, Any]],
    *,
    excluded_ids: set[str] | None = None,
    selected_case_ids: set[str] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Passages from other respondents whose text contains both endpoints' distinctive terms.

    The list is empty when no slot remains. Callers keep this inside the evidence budget.
    """
    if limit <= 0:
        return []
    usable = _usable_passages(passages)
    skip = set(excluded_ids or ())
    occupied = set(selected_case_ids or ())
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in pairs:
        from_terms = _endpoint_terms(str(pair.get("from_id") or ""), codebook, None)
        to_terms = _endpoint_terms(str(pair.get("to_id") or ""), codebook, None)
        if not from_terms or not to_terms:
            continue
        code_ids = [code_id for code_id in (str(pair.get("from_id") or ""), str(pair.get("to_id") or "")) if code_id]
        for passage_id, passage in usable.items():
            if passage_id in skip or passage_id in seen:
                continue
            case_id = case_id_of(passage)
            if occupied and case_id in occupied:
                continue
            tokens = meaningful_tokens(str(passage.get("text") or ""))
            if not (tokens & from_terms) or not (tokens & to_terms):
                continue
            seen.add(passage_id)
            chosen.append(
                {
                    "passage_id": passage_id,
                    "doc_id": passage.get("doc_id") or passage.get("document_id"),
                    "respondent_id": case_id,
                    "text": passage.get("text"),
                    "code_ids": list(code_ids),
                    "candidate_cross_case": True,
                }
            )
            if len(chosen) >= limit:
                return chosen
    return chosen
