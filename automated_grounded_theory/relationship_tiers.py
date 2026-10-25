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

from automated_grounded_theory.decision_review import PATTERN_QUOTE_RULE
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
_RELEVANT = frozenset({"supports", "contradicts", "qualifies"})
_ANALYST_LABELS = (
    "analyst interprets",
    "analyst interpretation",
    "i interpret",
    "my interpretation",
)
ANALYST_WORDING = "analyst interprets X as related to Y"
_CLAIM_FLAGS = frozenset(
    {
        "analyst_interpretation",
        "participant_denies_link",
        "participant_direction_reversed",
        "participant_link_hedged",
        "participant_link_not_a_relationship",
        "participant_direction_unspecified",
        "participant_link_contested",
        "participant_link_hypothetical",
        "participant_link_ambiguous",
        "participant_link_unsupported",
        "frequency_negation",
        "frequency_qualified",
        "negation_does_not_reject_link",
        "shared_quote_inspect",
    }
)
# Longer phrases are listed with the short ones. Matching keeps the longest
# phrase at a position, so "caused by" is not read as the active verb "caused".
_CONNECTORS = (
    ("leads to by", "backward"),
    ("leads to", "forward"),
    ("lead to by", "backward"),
    ("led to by", "backward"),
    ("lead to", "forward"),
    ("led to", "forward"),
    ("results in", "forward"),
    ("because of", "backward"),
    ("due to", "backward"),
    ("result of", "backward"),
    ("consequence of", "backward"),
    ("attributed to", "backward"),
    ("effect of", "backward"),
    ("caused by", "backward"),
    ("caused", "forward"),
    ("causes", "forward"),
    ("cause", "forward"),
    ("makes", "forward"),
    ("because", "backward"),
    ("related to", "undirected"),
    ("connected to", "undirected"),
    ("associated with", "undirected"),
    ("goes with", "undirected"),
    ("tied to", "undirected"),
)
_HEDGE_RE = re.compile(
    r"\b(?:might|maybe|perhaps|could|seems|seem|possibly|possible|sort of|not sure|i think|may)\b"
)
_DENIAL_RE = re.compile(
    r"\b(?:do not|does not|don't|doesn't|did not|didn't|never|cannot|can't|"
    r"isn't|aren't|wasn't|weren't|is not(?!\s+only)|are not(?!\s+only)|"
    r"was not(?!\s+only)|were not(?!\s+only)|not what|no relation|unrelated|not related|"
    r"no relationship|no connection|not connected|not a cause|not because)\b"
)
# Denied means the participant rejects the from-to link. A negation that does
# not govern that link is not a denial.
_LINK_REJECTION_RE = re.compile(
    r"\b(?:"
    r"(?:do not|does not|don't|doesn't|did not|didn't|never|cannot|can't)\s+"
    r"(?:really\s+|always\s+|actually\s+|just\s+)?"
    r"(?:lead|leads|led|cause|causes|caused|result|results|make|makes|"
    r"relate|relates|related|connect|connects|connected|associate|associates|associated)\b"
    r"|(?:is|are|was|were)\s+not\s+(?:related|connected|associated|a cause|because|caused)\b"
    r"|(?:isn't|aren't|wasn't|weren't)\s+(?:related|connected|associated|a cause|because|caused)\b"
    r"|not related|no relation|unrelated|not a cause|not because|"
    r"no relationship|no connection|not connected|"
    r"not what (?:causes|caused|leads|lead|makes|made)"
    r")"
)
# A frequency qualifier, or a softener in that same scope, is resolved before
# a categorical rejection. Contractions and a few intervening adverbs count.
_FREQUENCY_WORD = r"(?:always|often|usually|generally|rarely|sometimes)"
_NEGATION_AUX = (
    r"(?:do not|does not|don't|doesn't|did not|didn't|"
    r"is not|are not|isn't|aren't|was not|were not|wasn't|weren't)"
)
_INTERVENING_ADVERB = r"(?:really|actually|just|necessarily|typically|quite|very)"
_FREQUENCY_SCOPE_RE = re.compile(
    rf"\b(?:{_NEGATION_AUX}(?:\s+{_INTERVENING_ADVERB}){{0,3}}\s+(?:{_FREQUENCY_WORD}|necessarily)"
    rf"|not\s+{_FREQUENCY_WORD}|rarely|sometimes)\b"
)
_NOTHING_TO_DO_RE = re.compile(r"\bnothing to do with\b")
READING_BASIS = "deterministic_provisional"
# Markers other than bare ``if``. ``if`` is scoped in ``_conditional_governs_link``.
_OTHER_HYPOTHETICAL_RE = re.compile(
    r"\b(?:would|someone said|they said|i heard|suppose|supposing|what if|had said)\b"
)
# Exact conversational hedges. A span counts only when the whole span is one
# of these phrases, and only when the phrase cannot be a real condition.
# ``if I can`` and ``if possible`` can be genuine conditions, so they stay in
# the quote. ``if you help``, ``if you understand``, and ``if it rains`` stay
# too. This is not a word class.
_DISCOURSE_ASIDES = frozenset({
    "if that makes sense",
    "if you know what i mean",
    "if i'm honest",
    "if i am honest",
})
_ASIDE_BOUNDARY_RE = re.compile(r",|;|\u2014|—|--")
# Condition markers, longest first so ``provided that`` wins over ``provided``.
# This is a scope check, not a list of the only sentences that can be conditional.
_CONDITION_MARKER_RE = re.compile(
    r"\b(?:on condition that|provided that|providing that|as long as|so long as|"
    r"only when|except when|except if|in case|assuming that|provided|providing|"
    r"assuming|unless|if)\b"
)
# Positive evidence that ``if`` or ``whether`` is inside a noun phrase.
# ``the reason`` and ``part of why`` are not that evidence.
_EMBEDDED_CONDITION_RE = re.compile(r"\bquestion of\s*$")
_MATRIX_COPULA_RE = re.compile(r"\s+is\s+(?:simple|clear|obvious|plain)\b.*$")
ENDPOINT_UNREVIEWED_BY_FOCUS = "endpoint_unreviewed_by_focus"
ENDPOINT_NOT_SELECTED_BY_FOCUS = "endpoint_not_selected_by_focus"
_FOCUS_ENDPOINT_FLAGS = frozenset({ENDPOINT_UNREVIEWED_BY_FOCUS, ENDPOINT_NOT_SELECTED_BY_FOCUS})
# Epistemic and existential-negation markers. A hit in the clause text before
# the first connector is unsupported. This is not a list of the only sentences
# that can fail: any of these markers fails closed. ``doubt`` also matches
# inside ``no doubt``. No affirmative exception is recognized for that phrase.
_EPISTEMIC_NEGATION_RE = re.compile(
    r"\b(?:no evidence|not clear|unclear|doubt(?:ful)?|unknown|not known|"
    r"nothing to suggest|nothing indicates|nothing to indicate|nothing that|"
    r"i don't know|i do not know|i dont know|"
    r"no sign|no indication|no proof|no reason to|"
    r"not established|not obvious|cannot tell|can't tell|"
    r"no data|no support|not sure whether|unsure whether|"
    r"there is no|there are no|there's no|there isn't|there aren't|"
    r"no doubt|without evidence|absent evidence|lack of evidence|"
    r"not certain|uncertain|questionable whether|hard to say whether)\b"
)
_CLAUSE_SPLIT_RE = re.compile(
    r"[.!?]+|\s+(?:but|although|though|however|while|whereas)\s+|;\s*"
)
_LEGACY_ANALYST_NOTE = (
    "0.9.19 stored this analyst interpretation as a row-level exclusion. "
    "0.9.20 keeps it as a hypothesis component. "
    "Eligibility is derived again from the current evidence."
)


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
    """Respondent when the passage names one, otherwise the document.

    Passages keep ``respondent_id`` on ``metadata``. A top-level id still wins.
    Reading the metadata id collapses a respondent's question-documents into
    one case, so a tier-2 count on a multi-question corpus gets stricter.
    """
    if not isinstance(passage, Mapping):
        return ""
    for key in ("respondent_id", "participant_id", "case_id"):
        value = str(passage.get(key) or "").strip()
        if value:
            return value
    metadata = passage.get("metadata") if isinstance(passage.get("metadata"), Mapping) else {}
    for key in ("respondent_id", "participant_id", "case_id"):
        value = str(metadata.get(key) or "").strip()
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


def _quote_identity(item: Mapping[str, Any]) -> tuple[str, str]:
    return (_norm(str(item.get("quote") or "")), str(item.get("passage_id") or ""))


def _supporting_cases(found: Mapping[str, Mapping[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """One row per case. Endpoint support may share a span.

    A case is the respondent when the passage names one, otherwise the
    document. Distinct quote strings do not add a case, and one case is not
    counted twice. Each endpoint is supported when the located text grounds
    it, including an overlapping or identical span. An identical span used
    for both endpoints is an inspect flag. It is not a disqualification.
    This reverses the 0.9.25 rule that dropped a shared quotation before
    deciding whether the case supported both categories.
    """
    case_ids = set(found.get("from") or {}) | set(found.get("to") or {})
    rows = []
    for case_id in sorted(case_ids):
        from_all = list((found.get("from") or {}).get(case_id) or [])
        to_all = list((found.get("to") or {}).get(case_id) or [])
        shared = {
            item
            for item in ({_quote_identity(row) for row in from_all} & {_quote_identity(row) for row in to_all})
            if item[0]
        }
        from_grounded = [row for row in from_all if row.get("grounded", True)]
        to_grounded = [row for row in to_all if row.get("grounded", True)]
        quotes = from_all + to_all
        passage_ids = []
        for item in quotes:
            if item["passage_id"] not in passage_ids:
                passage_ids.append(item["passage_id"])
        from_passages = {item["passage_id"] for item in from_grounded}
        to_passages = {item["passage_id"] for item in to_grounded}
        both = bool(from_grounded) and bool(to_grounded)
        same = bool(from_passages & to_passages)
        rows.append(
            {
                "case_id": case_id,
                "passage_ids": passage_ids,
                "quotes": quotes,
                "both_endpoints": both,
                "same_passage": same,
                "within_case": both,
                "shared_quote_for_both_endpoints": bool(shared),
                "shared_quote_inspect": bool(shared),
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


def _empty_claims() -> dict[str, Any]:
    """The three claim components. None of them support a core until evidence says so."""
    return {
        "pattern": {
            "component": "pattern",
            "present": False,
            "evidence_tier": None,
            "allowed_wording": None,
            "evidence_use": HYPOTHESIS_EXPLORATION,
            "downstream_eligible": False,
            "supports_core": False,
            "validated_by": None,
        },
        "participant_explanation": {
            "component": "participant_explanation",
            "present": False,
            "link_text": None,
            "passage_id": None,
            "stated_direction": None,
            "polarity": None,
            "hedging": None,
            "status": "absent",
            "validated_by": None,
            "qualifications": [],
            "evidence_use": HYPOTHESIS_EXPLORATION,
            "downstream_eligible": False,
            "supports_core": False,
            "authorizes_tier_3": False,
        },
        "analyst_interpretation": {
            "component": "analyst_interpretation",
            "present": False,
            "allowed_wording": None,
            "evidence_use": HYPOTHESIS_EXPLORATION,
            "downstream_eligible": False,
            "supports_core": False,
            "validated_by": None,
        },
    }


def _side_has(side: str, terms: set[str]) -> bool:
    return bool(_distinctive(side) & terms)


def _connectors(folded: str) -> list[tuple[int, str, str]]:
    """Recognized connector phrases, longest match at each position.

    A phrase must be a whole word or a whole word sequence. ``cause`` does not
    match inside ``because``. ``caused`` does not win over ``caused by``.
    """
    found: list[tuple[int, int, str, str]] = []
    for phrase, kind in _CONNECTORS:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        for match in re.finditer(pattern, folded):
            found.append((match.start(), -len(phrase), phrase, kind))
    found.sort()
    kept: list[tuple[int, str, str]] = []
    for pos, _length, phrase, kind in found:
        if any(pos >= earlier and pos < earlier + len(earlier_phrase) for earlier, earlier_phrase, _kind in kept):
            continue
        kept.append((pos, phrase, kind))
    return kept


def _connector_direction(
    folded: str,
    pos: int,
    phrase: str,
    kind: str,
    from_terms: set[str],
    to_terms: set[str],
) -> str | None:
    left = folded[:pos]
    right = folded[pos + len(phrase) :]
    left_from = _side_has(left, from_terms)
    left_to = _side_has(left, to_terms)
    right_from = _side_has(right, from_terms)
    right_to = _side_has(right, to_terms)
    if kind == "forward":
        if left_from and right_to and not (left_to and right_from):
            return "from_to"
        if left_to and right_from and not (left_from and right_to):
            return "to_from"
        return None
    if kind == "backward":
        if left_to and right_from and not (left_from and right_to):
            return "from_to"
        if left_from and right_to and not (left_to and right_from):
            return "to_from"
        return None
    if (left_from and right_to) or (left_to and right_from) or (left_from and left_to) or (right_from and right_to):
        return "undirected"
    return None


def _clauses(folded: str) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(folded) if part and part.strip()]
    return parts or [folded]


def _direction_of(folded: str, from_terms: set[str], to_terms: set[str]) -> str | None:
    """Direction of one clear clause, or ``ambiguous`` when the quote is not that.

    Passive ``caused by`` and ``led to by`` point at the cause on the right.
    Two claim-bearing clauses, or two connectors in one clause that disagree,
    are ``ambiguous``. A clause with no recognized connector does not count.
    """
    claimed: list[str] = []
    for clause in _clauses(folded):
        directions = []
        for pos, phrase, kind in _connectors(clause):
            direction = _connector_direction(clause, pos, phrase, kind, from_terms, to_terms)
            if direction:
                directions.append(direction)
        if not directions:
            continue
        if len(set(directions)) > 1:
            return "ambiguous"
        claimed.append(directions[0])
    if not claimed:
        return None
    if len(claimed) > 1:
        return "ambiguous"
    return claimed[0]


def _claim_clause(folded: str, from_terms: set[str], to_terms: set[str]) -> str:
    """The single clause that states the link, or the whole quote when there is not one."""
    found = []
    for clause in _clauses(folded):
        if _direction_of(clause, from_terms, to_terms):
            found.append(clause)
    if len(found) == 1:
        return found[0]
    return folded


def _discourse_aside_text(text: str) -> bool:
    """True when ``text`` is one exact conversational hedge.

    The whole span must be on ``_DISCOURSE_ASIDES``. ``if that makes sense``,
    ``if you know what I mean``, and ``if I'm honest`` are hedges because the
    phrase cannot state a condition on the link. ``if I can``, ``if possible``,
    ``if you help``, ``if you understand``, and ``if it rains`` can be
    conditions and stay in the quote.
    """
    folded = _norm(text).strip(" ,;()—-.").replace("\u2019", "'").replace("`", "'")
    folded = folded.strip(" .,!?;:")
    return bool(folded) and folded in _DISCOURSE_ASIDES


def _strip_discourse_asides(folded: str) -> str:
    """Drop parenthetical and comma-bounded discourse asides. Leave every other ``if``."""
    without_parens = re.sub(
        r"\([^)]*\)",
        lambda match: " " if _discourse_aside_text(match.group(0)) else match.group(0),
        folded,
    )
    pieces: list[str] = []
    cursor = 0
    for match in _ASIDE_BOUNDARY_RE.finditer(without_parens):
        piece = without_parens[cursor:match.start()]
        boundary = match.group(0)
        if _discourse_aside_text(piece):
            pieces.append(" ")
        else:
            pieces.append(piece)
        pieces.append(boundary)
        cursor = match.end()
    tail = without_parens[cursor:]
    pieces.append(" " if _discourse_aside_text(tail) else tail)
    return _norm("".join(pieces))


def _condition_text(clause: str) -> str | None:
    """The condition span that governs ``clause``, or None.

    Discourse asides are removed first. ``if that makes sense`` does not count.
    ``if I can``, ``if possible``, ``unless``, ``provided that``, ``as long as``,
    ``only when``, ``in case``, ``assuming``, ``on condition that``, and
    ``except when`` do. The span runs from the marker through the next clause
    break. A trailing matrix copula (``if it rains is simple``) is not part of
    the condition. ``if`` or ``whether`` counts as embedded only when the words
    immediately before it are ``question of``. ``the reason`` and ``part of why``
    are not embedding, so the condition stays.
    """
    stripped = _strip_discourse_asides(_norm(clause))
    if not stripped:
        return None
    for match in _CONDITION_MARKER_RE.finditer(stripped):
        marker = match.group(0)
        prefix = stripped[:match.start()]
        if marker in {"if", "whether"} and _EMBEDDED_CONDITION_RE.search(prefix):
            continue
        tail = stripped[match.start():]
        end = len(tail)
        for index, char in enumerate(tail):
            if char in ".?!;,":
                end = index
                break
        span = _MATRIX_COPULA_RE.sub("", tail[:end]).strip()
        if span:
            return span
    return None


def _conditional_governs_link(clause: str) -> bool:
    """True when a conditional governs the from-to link in ``clause``.

    A condition marker keeps the link conditional. ``the reason ... if ...``
    does not remove that marker. Only ``question of if`` or ``question of
    whether`` is noun-phrase embedding. A discourse aside does not govern:
    the span is an exact hedge that cannot be a condition, such as ``if that
    makes sense``, ``if you know what I mean``, or ``if I'm honest``.
    ``if I can``, ``if possible``, ``unless it rains``, and ``provided that
    it rains`` stay. ``would`` and reported speech stay hypothetical markers.
    The reading is still provisional: this is a closed pattern, not a parse
    of the sentence.
    """
    folded = _norm(clause)
    if not folded:
        return False
    if _OTHER_HYPOTHETICAL_RE.search(folded):
        return True
    if _condition_text(folded) is None:
        return False
    return bool(_connectors(_strip_discourse_asides(folded)))


def _recognized_affirmative_prefix(prefix: str) -> bool:
    """No affirmative exception is recognized for an epistemic or negation marker.

    ``there is no doubt that`` and ``I have no doubt`` stay unsupported.
    The argument is unused on purpose so a later exception would be a visible
    change here rather than a silent pass.
    """
    del prefix
    return False


def _unsupported_before_connector(clause: str) -> bool:
    """True when a negation or epistemic marker sits before the first connector.

    The marker may occur anywhere in that prefix, not only at the start.
    A marker that appears only after the connector does not by itself fail
    the clause. A clause with no connector is left to the denial rules.
    """
    connectors = _connectors(clause)
    if not connectors:
        return False
    pos = min(item[0] for item in connectors)
    prefix = clause[:pos]
    if not _EPISTEMIC_NEGATION_RE.search(prefix):
        return False
    if _recognized_affirmative_prefix(prefix):
        return False
    return True


def _interpret_quote(text: str, from_terms: set[str], to_terms: set[str]) -> dict[str, Any] | None:
    """Read one candidate quote. A keyword hit is not itself a tier-3 authorization.

    The rule is deterministic and provisional. It does not call a model.
    Tier 3 requires one clause, one recognized connector, affirmative
    polarity, no hedge, and the from-to direction. Passive voice reverses
    ``caused by`` and ``led to by``. A frequency qualifier is resolved before
    a categorical rejection, so ``not always``, ``doesn't always``, ``rarely``,
    ``sometimes``, and an intervening softener such as ``necessarily`` are
    qualified and never denied. Negation that governs the connector, including
    ``nothing to do with``, denies the link. An epistemic or
    existential-negation marker before that connector is unsupported and does
    not authorize tier 3. ``would`` and ``someone said`` are hypothetical.
    A condition marker is hypothetical when it governs this from-to link.
    ``unless``, ``provided that``, ``as long as``, ``only when``, ``in case``,
    ``assuming``, ``on condition that``, and ``except when`` are condition
    markers, as is ``if``. ``the reason`` before ``if`` does not remove the
    condition. Only ``question of if`` or ``question of whether`` is embedding.
    An exact hedge that cannot be a condition, such as ``if that makes sense``,
    is not hypothetical. ``if I can`` and ``if possible`` stay in the quote.
    The condition span is kept on the reading. A provisional reading does not
    authorize tier 3. This is not a general reading of the sentence. Every
    reading stays provisional until it is checked on natural-language data.
    Human-adjudicated examples are not in this package.
    """
    folded = _norm(text)
    if not folded:
        return None
    padded = f" {folded} "
    nothing_to_do = bool(_NOTHING_TO_DO_RE.search(padded))
    mentions_link = (
        bool(_connectors(folded))
        or any(marker in padded for marker in _ATTRIBUTION)
        or nothing_to_do
    )
    # A denial or a hypothetical with no connector is still a reading. It does not authorize tier 3.
    if not mentions_link and not _DENIAL_RE.search(padded) and not _conditional_governs_link(folded):
        return None
    tokens = _distinctive(text)
    if not (tokens & from_terms) or not (tokens & to_terms):
        return None
    direction = _direction_of(folded, from_terms, to_terms)
    clause = _claim_clause(folded, from_terms, to_terms)
    clause_padded = f" {clause} "
    frequency_scope = bool(
        _FREQUENCY_SCOPE_RE.search(clause_padded) or _FREQUENCY_SCOPE_RE.search(padded)
    )
    # Frequency scope wins over the rejection pattern. ``do not always lead``
    # is qualified. ``do not lead`` and ``nothing to do with`` stay denied.
    rejects_link = (not frequency_scope) and (
        bool(_LINK_REJECTION_RE.search(clause_padded)) or nothing_to_do
    )
    other_negation = bool(_DENIAL_RE.search(clause_padded) or (direction is None and _DENIAL_RE.search(padded)))
    if frequency_scope:
        polarity = "affirmative"
        negation_scope = "frequency"
    elif rejects_link:
        polarity = "negative"
        negation_scope = "link"
    elif other_negation:
        polarity = "affirmative"
        negation_scope = "not_the_link"
    else:
        polarity = "affirmative"
        negation_scope = None
    stripped_clause = _strip_discourse_asides(clause)
    # Hedge words count only outside an exact aside. ``if possible`` is not
    # an aside: it can be a real condition, and ``possible`` still counts.
    hedging = "qualified" if frequency_scope else (
        "hedged" if _HEDGE_RE.search(f" {stripped_clause} ") else "none"
    )
    condition = _condition_text(clause) or _condition_text(folded)
    hypothetical = _conditional_governs_link(clause) or bool(condition)
    unsupported = _unsupported_before_connector(clause)
    if unsupported:
        status = "unsupported"
        if direction == "ambiguous":
            direction = None
    elif direction == "ambiguous":
        status = "ambiguous"
        direction = None
    elif hypothetical:
        status = "hypothetical"
    elif frequency_scope:
        status = "qualified"
    elif rejects_link:
        status = "denied"
    elif other_negation and not rejects_link:
        status = "unvalidated"
    elif direction is None:
        status = "not_a_link"
    elif direction == "to_from":
        status = "reversed"
    elif direction == "undirected":
        status = "direction_unspecified"
    elif hedging == "hedged":
        status = "hedged"
    elif direction == "from_to":
        status = "validated"
    else:
        status = "unvalidated"
    qualifications = []
    if status == "qualified":
        qualifications.append("frequency_qualified")
    elif status == "denied":
        qualifications.append("participant_denies_link")
    elif negation_scope == "not_the_link":
        qualifications.append("negation_does_not_reject_link")
    elif status == "reversed":
        qualifications.append("participant_direction_reversed")
    elif status == "hedged":
        qualifications.append("participant_link_hedged")
    elif status == "not_a_link":
        qualifications.append("participant_link_not_a_relationship")
    elif status == "direction_unspecified":
        qualifications.append("participant_direction_unspecified")
    elif status == "hypothetical":
        qualifications.append("participant_link_hypothetical")
        if condition:
            qualifications.append("participant_link_conditional")
    elif status == "ambiguous":
        qualifications.append("participant_link_ambiguous")
    elif status == "unsupported":
        qualifications.append("participant_link_unsupported")
    return {
        "link_text": str(text).strip(),
        "stated_direction": direction,
        "polarity": polarity,
        "hedging": hedging,
        "negation_scope": negation_scope,
        "status": status,
        "validated_by": "deterministic_rule",
        "reading_basis": READING_BASIS,
        "reading_provisional": True,
        "qualifications": qualifications,
        # The condition span stays on the reading. A later check requires the
        # proposed statement to retain it before the quote can authorize tier 3.
        "condition": condition,
        "condition_retained": None,
        # Only a validated reading can authorize tier 3. Hypothetical,
        # ambiguous, and every other provisional status cannot.
        "authorizes_tier_3": status == "validated",
    }


def _candidate_quotes(
    relationship: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    quotes = []
    for case in cases:
        for item in case.get("quotes") or []:
            if isinstance(item, Mapping):
                quotes.append(
                    {
                        "quote": str(item.get("quote") or ""),
                        "passage_id": str(item.get("passage_id") or ""),
                    }
                )
    for key in ("quote", "quote_from", "quote_to"):
        text = str(relationship.get(key) or "").strip()
        if text:
            quotes.append({"quote": text, "passage_id": str(relationship.get("passage_id") or "")})
    return quotes


def _participant_link(
    relationship: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    passages: Mapping[str, Mapping[str, Any]],
    from_terms: set[str],
    to_terms: set[str],
) -> dict[str, Any]:
    """The participant component. Keyword overlap only nominates a quote."""
    component = _empty_claims()["participant_explanation"]
    if not from_terms or not to_terms:
        return component
    readings = []
    seen: set[tuple[str, str]] = set()
    for item in _candidate_quotes(relationship, cases):
        text = item["quote"]
        key = (_norm(text), item["passage_id"])
        if not text or key in seen:
            continue
        seen.add(key)
        reading = _interpret_quote(text, from_terms, to_terms)
        if reading is None:
            continue
        passage_id = _locate(text, passages, item["passage_id"])
        if not passage_id:
            continue
        reading["passage_id"] = passage_id
        readings.append(reading)
    if not readings:
        return component
    authorizing = [item for item in readings if item["authorizes_tier_3"]]
    denied = [item for item in readings if item["status"] == "denied"]
    if authorizing and denied:
        chosen = dict(authorizing[0])
        chosen["status"] = "contested"
        chosen["authorizes_tier_3"] = False
        chosen["reading_basis"] = READING_BASIS
        chosen["reading_provisional"] = True
        chosen["qualifications"] = ["participant_denies_link", "participant_link_contested"]
    elif authorizing:
        chosen = authorizing[0]
    else:
        order = (
            "denied",
            "reversed",
            "qualified",
            "hedged",
            "hypothetical",
            "ambiguous",
            "direction_unspecified",
            "not_a_link",
            "unsupported",
            "unvalidated",
        )
        chosen = sorted(readings, key=lambda item: order.index(item["status"]) if item["status"] in order else len(order))[0]
    chosen = dict(chosen)
    condition = chosen.get("condition")
    statement = _norm(str(relationship.get("statement") or ""))
    retained = None
    if condition:
        retained = _norm(str(condition)) in statement
        if not retained:
            qualifications = [str(item) for item in (chosen.get("qualifications") or [])]
            if "condition_not_in_statement" not in qualifications:
                qualifications.append("condition_not_in_statement")
            chosen["qualifications"] = qualifications
            chosen["authorizes_tier_3"] = False
    chosen["condition_retained"] = retained
    component.update(chosen)
    component["present"] = True
    component["evidence_use"] = HYPOTHESIS_EXPLORATION
    component["downstream_eligible"] = False
    component["supports_core"] = False
    return component


def _model_memo(memo: Mapping[str, Any]) -> bool:
    """A memo this package stored from the model, not from an analyst."""
    actor = str(memo.get("actor") or "").strip().casefold()
    kind = str(memo.get("kind") or "").strip().casefold()
    return actor == "model" or kind == "model_memo"


def _analyst_interpretation(
    relationship: Mapping[str, Any],
    from_terms: set[str],
    to_terms: set[str],
    memos: Sequence[Mapping[str, Any]] | None,
) -> bool:
    """An explicitly labeled analyst reading that names both endpoints.

    A shared token plus an explanatory word is not that reading. A memo the
    model wrote cannot stand in for the analyst, and it cannot become
    participant attribution.
    """
    if not from_terms or not to_terms:
        return False
    relationship_id = str(relationship.get("relationship_id") or "")
    if not relationship_id:
        return False
    for memo in memos or []:
        if not isinstance(memo, Mapping):
            continue
        if str(memo.get("relationship_id") or "") != relationship_id:
            continue
        if _model_memo(memo):
            continue
        text = str(memo.get("text") or "")
        folded = f" {_norm(text)} "
        if not any(label in folded for label in _ANALYST_LABELS):
            continue
        tokens = _distinctive(text)
        if tokens & from_terms and tokens & to_terms:
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
        "claims": _empty_claims(),
        "participant_link": None,
        "core_support_component": None,
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
    """Score a relationship up to tier 3. Tier 4 is left to ``promote_verified``.

    Tier 3 is a participant quote the deterministic rule accepts as an
    affirmative from-to link. A keyword hit only nominates that quote. An
    analyst memo stays on its own component and does not change the row tier
    or the row wording.
    """
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
    participant = _participant_link(relationship, cases, usable, from_terms, to_terms)
    analyst_present = _analyst_interpretation(relationship, from_terms, to_terms, memos)
    authorizes = bool(participant.get("authorizes_tier_3"))
    if not from_cases or not to_cases:
        tier = 0
    elif len(both_cases) >= minimum:
        tier = 3 if authorizes and rival == "reported" else 2
    else:
        tier = 1
    wording = allowed_wording(tier, len(both_cases) if tier >= 2 else None)
    claims = _empty_claims()
    pattern_supported = tier >= 2
    claims["pattern"] = {
        "component": "pattern",
        "present": pattern_supported,
        "evidence_tier": 2 if pattern_supported else tier,
        "allowed_wording": allowed_wording(2, len(both_cases)) if pattern_supported else wording,
        "evidence_use": SUPPORTED_EVIDENCE if pattern_supported else HYPOTHESIS_EXPLORATION,
        "downstream_eligible": pattern_supported,
        "supports_core": pattern_supported,
        "validated_by": "deterministic_rule" if pattern_supported else None,
        "cases_with_both_endpoints": len(both_cases),
    }
    claims["participant_explanation"] = participant
    claims["analyst_interpretation"] = {
        "component": "analyst_interpretation",
        "present": analyst_present,
        "allowed_wording": ANALYST_WORDING if analyst_present else None,
        "evidence_use": HYPOTHESIS_EXPLORATION,
        "downstream_eligible": False,
        "supports_core": False,
        "validated_by": "deterministic_rule" if analyst_present else None,
    }
    attribution = "participant" if authorizes and tier >= 2 else None
    return {
        "evidence_tier": tier,
        "evidence_label": TIER_LABELS[tier],
        "allowed_wording": wording,
        "cross_case": cross_case if tier == 1 else False,
        "within_case_strength": strength,
        "supporting_cases": cases,
        "cooccurrence_2x2": cooccurrence,
        "pattern_min_cases": minimum,
        "pattern_min_cases_note": PATTERN_MIN_CASES_NOTE,
        "negative_case_search_report": rival,
        "variation_search": variation,
        "rival_explanation_check": rival,
        "attribution_source": attribution,
        "cases_with_both_endpoints": len(both_cases),
        "claims": claims,
        "participant_link": participant if participant.get("present") else None,
        "condition": participant.get("condition"),
        "condition_retained": participant.get("condition_retained"),
        "core_support_component": "pattern" if pattern_supported else None,
        "shared_quote_for_both_endpoints": any(
            isinstance(case, Mapping) and case.get("shared_quote_for_both_endpoints") for case in cases
        ),
        "shared_quote_inspect": any(
            isinstance(case, Mapping) and case.get("shared_quote_inspect") for case in cases
        ),
        "pattern_quote_rule": PATTERN_QUOTE_RULE,
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
            "non-causal wording",
            "non causal wording",
            "non-causal language",
            "non causal language",
            "uses non-causal",
            "uses non causal",
        )
    )


_ROLE_EVIDENCE_ROLES = frozenset({"consequence", "cause", "causal_condition"})


def _role_key(role: str) -> str:
    return str(role or "").strip().casefold().replace(" ", "_")


def statement_wording_overreach(text: str, tier: int | None) -> bool:
    """True when the sentence itself needs a higher tier. The role is ignored."""
    return wording_overreach(text, tier, role="")


def role_evidence_overreach(tier: int | None, role: str = "") -> bool:
    """True when a causal role is used below tier 3.

    An association can stay ``unspecified``. The role is a separate check from
    the sentence. A missing tier is not an overreach.
    """
    if tier is None:
        return False
    return tier < 3 and _role_key(role) in _ROLE_EVIDENCE_ROLES


def wording_overreach(text: str, tier: int | None, *, role: str = "") -> bool:
    """True when ``text`` or ``role`` uses wording a higher tier would be needed to license.

    A missing tier is undefined and is not an overreach. Role ``consequence``
    or ``cause`` at tier 1 or 2 is causal wording even when the sentence is mild.
    ``causal_condition`` is reported by :func:`role_evidence_overreach`.
    """
    if tier is None:
        return False
    chosen = _role_key(role)
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


_WEAK_STEM_REASONS = frozenset({"stem_only", "generic_stem_only"})
HYPOTHESIS_EXPLORATION = "hypothesis_exploration"
SUPPORTED_EVIDENCE = "supported_evidence"
WITHHELD_EVIDENCE = "withheld"


def weak_stem_relationship(row: Mapping[str, Any]) -> bool:
    """True when an endpoint is only a single stem or a generic stem.

    Quote containment does not clear this. ``same_passage_stem_only`` and
    ``weak_stem_anchor`` are the flags the axial annotator writes.
    """
    if not isinstance(row, Mapping):
        return False
    if str(row.get("evidence_scope") or "") == "same_passage_stem_only":
        return True
    flags = {str(item) for item in (row.get("flags") or [])}
    if "weak_stem_anchor" in flags:
        return True
    for endpoint in row.get("endpoint_anchoring") or []:
        if isinstance(endpoint, Mapping) and str(endpoint.get("reason") or "") in _WEAK_STEM_REASONS:
            return True
    return False


def selection_limits_eligibility(metadata: Mapping[str, Any] | None) -> bool:
    """True when a focused selection was withheld or left provisional.

    A complete selection does not apply the unreviewed-endpoint limit. A run
    with no focused selection does not either: missing fields are not the
    same as "every code was left unreviewed."
    """
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get("selection_review_complete") is False:
        return True
    if metadata.get("selection_withheld") is True:
        return True
    return str(metadata.get("selection_status") or "") == "provisional"


def selection_restricts_to_raised(metadata: Mapping[str, Any] | None) -> bool:
    """True when a finished focused selection limits later stages to raised codes.

    An incomplete or withheld selection does not. Its raised list is empty
    because the review was not applied, not because nothing was selected.
    A run with no focused selection does not restrict either.
    """
    if not isinstance(metadata, Mapping) or selection_limits_eligibility(metadata):
        return False
    if metadata.get("selection_review_complete") is not True:
        return False
    if metadata.get("selection_withheld") is True:
        return False
    if str(metadata.get("selection_status") or "") != "applied":
        return False
    return isinstance(metadata.get("raised_code_ids"), (list, tuple))


def stamp_focus_review(row: dict, metadata: Mapping[str, Any] | None) -> None:
    """Copy the focused selection onto a relationship, when one applies.

    A withheld or provisional selection stamps the codes it reviewed. A
    complete applied selection stamps the codes it raised. A run with no
    focused selection leaves the row unchanged. The stamp is what later
    eligibility reads, so a refresh does not drop the limit.
    """
    if not isinstance(row, dict):
        return
    if selection_limits_eligibility(metadata):
        reviewed = [str(item) for item in ((metadata or {}).get("selection_reviewed_code_ids") or [])]
        row["focus_review"] = {
            "limits_eligibility": True,
            "reviewed_code_ids": reviewed,
            "rule": "unreviewed_endpoint_v1",
        }
        return
    if not selection_restricts_to_raised(metadata):
        return
    source = metadata or {}
    reviewed = [str(item) for item in (source.get("selection_reviewed_code_ids") or [])]
    raised = [str(item) for item in (source.get("raised_code_ids") or [])]
    row["focus_review"] = {
        "limits_eligibility": True,
        "reviewed_code_ids": reviewed,
        "raised_code_ids": raised,
        "rule": "raised_endpoint_v1",
    }


def _sync_unreviewed_flag(row: dict) -> None:
    """Mark an endpoint the focused selection did not leave eligible.

    A withheld selection flags an endpoint it never reviewed. A complete
    selection flags an endpoint it reviewed and did not raise as
    ``endpoint_not_selected_by_focus``. An endpoint that was never reviewed
    stays ``endpoint_unreviewed_by_focus``. Both flags are removed when the
    stamp is absent.
    """
    flags = [str(item) for item in (row.get("flags") or []) if str(item) not in _FOCUS_ENDPOINT_FLAGS]
    review = row.get("focus_review")
    limited = isinstance(review, Mapping) and review.get("limits_eligibility") is True
    reason = str(row.get("eligibility_reason") or "")
    if limited:
        reviewed = {str(item) for item in (review.get("reviewed_code_ids") or [])}
        ends = [str(row.get("from_id") or ""), str(row.get("to_id") or "")]
        if str(review.get("rule") or "") == "raised_endpoint_v1":
            raised = {str(item) for item in (review.get("raised_code_ids") or [])}
            reviewed_known = isinstance(review.get("reviewed_code_ids"), list)
            if reviewed_known and any(end and end not in reviewed for end in ends):
                flags.append(ENDPOINT_UNREVIEWED_BY_FOCUS)
                row["eligibility_reason"] = ENDPOINT_UNREVIEWED_BY_FOCUS
            elif any(end and end not in raised for end in ends):
                flags.append(ENDPOINT_NOT_SELECTED_BY_FOCUS)
                row["eligibility_reason"] = ENDPOINT_NOT_SELECTED_BY_FOCUS
            elif reason in _FOCUS_ENDPOINT_FLAGS:
                row.pop("eligibility_reason", None)
        elif any(end and end not in reviewed for end in ends):
            flags.append(ENDPOINT_UNREVIEWED_BY_FOCUS)
            row["eligibility_reason"] = ENDPOINT_UNREVIEWED_BY_FOCUS
        elif reason == ENDPOINT_UNREVIEWED_BY_FOCUS:
            row.pop("eligibility_reason", None)
    elif reason in _FOCUS_ENDPOINT_FLAGS:
        row.pop("eligibility_reason", None)
    row["flags"] = flags


def relationship_evidence_use(row: Mapping[str, Any]) -> str:
    """Classify a relationship for later stages.

    ``supported_evidence`` is the only class that may set ``downstream_eligible``.
    A tier-1 proposal and a weak-stem anchor are ``hypothesis_exploration``.
    An analyst-interpretation component does not change this class: the pattern
    component carries eligibility, and the analyst component stays a hypothesis.
    Rejected, empty, synthetic, unanchored, disputed, and causal-wording rows
    are ``withheld``. A missing tier is not supported evidence. An endpoint
    the withheld focused selection never reviewed is ``hypothesis_exploration``
    with reason ``endpoint_unreviewed_by_focus``. An endpoint a complete
    selection reviewed and did not raise is ``hypothesis_exploration`` with
    reason ``endpoint_not_selected_by_focus``.
    """
    if not isinstance(row, Mapping) or row.get("synthetic"):
        return WITHHELD_EVIDENCE
    if str(row.get("passage_id") or "") == "synthetic-counterexample":
        return WITHHELD_EVIDENCE
    if not str(row.get("statement") or "").strip():
        return WITHHELD_EVIDENCE
    flags = {str(item) for item in (row.get("flags") or [])}
    # An endpoint the focused selection never reviewed stays exploratory.
    # That holds even when quotes would otherwise support the row.
    if ENDPOINT_UNREVIEWED_BY_FOCUS in flags:
        return HYPOTHESIS_EXPLORATION
    if ENDPOINT_NOT_SELECTED_BY_FOCUS in flags:
        return HYPOTHESIS_EXPLORATION
    disposition = str(row.get("human_disposition") or row.get("disposition") or "").strip().casefold()
    if disposition in {"rejected", "merged"}:
        return WITHHELD_EVIDENCE
    if str(row.get("evidence_condition") or "") in {"unanchored", "disputed"}:
        return WITHHELD_EVIDENCE
    if "causal_wording_withheld" in flags or "role_evidence_exceeds_tier" in flags:
        return WITHHELD_EVIDENCE
    tier = row.get("evidence_tier")
    try:
        tier_n = int(tier) if tier is not None and not isinstance(tier, bool) else None
    except (TypeError, ValueError):
        tier_n = None
    if isinstance(tier_n, int) and tier_n < 3 and "wording_exceeds_tier" in flags:
        return WITHHELD_EVIDENCE
    if "acceptance_withheld" in flags or str(row.get("evidence_status") or "") == "cross_respondent_pattern":
        return HYPOTHESIS_EXPLORATION
    if weak_stem_relationship(row) or tier_n is None or tier_n < 2:
        return HYPOTHESIS_EXPLORATION
    ends = [item for item in (row.get("endpoint_anchoring") or []) if isinstance(item, Mapping)]
    if ends and any(not item.get("anchored") for item in ends):
        return HYPOTHESIS_EXPLORATION
    return SUPPORTED_EVIDENCE


def assign_relationship_eligibility(row: dict) -> str:
    """Write row and component eligibility from one rule.

    A claim component is downstream eligible, and the pattern component can
    support a core, only when the row itself is supported evidence. A
    withheld row clears both flags. Participant and analyst components stay
    ineligible. A stamped withheld focused selection keeps an unreviewed
    endpoint exploratory.
    """
    _sync_unreviewed_flag(row)
    use = relationship_evidence_use(row)
    row["evidence_use"] = use
    row_ok = use == SUPPORTED_EVIDENCE
    row["downstream_eligible"] = row_ok
    claims = row.get("claims")
    pattern = None
    if isinstance(claims, dict):
        raw = claims.get("pattern")
        if isinstance(raw, Mapping):
            pattern = dict(raw)
            pattern_ok = (
                row_ok
                and bool(pattern.get("present"))
                and int(pattern.get("evidence_tier") or 0) >= 2
            )
            pattern["downstream_eligible"] = pattern_ok
            pattern["supports_core"] = pattern_ok
            pattern["evidence_use"] = SUPPORTED_EVIDENCE if pattern_ok else (WITHHELD_EVIDENCE if not row_ok else HYPOTHESIS_EXPLORATION)
            claims["pattern"] = pattern
        for key in ("participant_explanation", "analyst_interpretation"):
            component = claims.get(key)
            if isinstance(component, Mapping):
                copied = dict(component)
                copied["downstream_eligible"] = False
                copied["supports_core"] = False
                claims[key] = copied
        row["claims"] = claims
    if row_ok and isinstance(pattern, Mapping) and pattern.get("supports_core"):
        row["core_support_component"] = "pattern"
    else:
        row["core_support_component"] = None
    return use


def effective_statement(row: Mapping[str, Any]) -> str:
    """The statement validation reads now.

    A researcher's sentence replaces the working statement. The model's
    earlier sentence stays on ``model_statement`` and is not rechecked.
    """
    human = str(row.get("human_statement") or "").strip()
    if human:
        return human
    return str(row.get("statement") or "").strip()


def effective_paradigm_role(row: Mapping[str, Any]) -> str:
    """The role validation, eligibility, and export read now.

    A human role wins. Until a human sets one, a recorded reviewer role is
    the working role. Otherwise the row's ``paradigm_role`` stands. Actor
    fields stay on the row for provenance.
    """
    human = str(row.get("human_paradigm_role") or "").strip()
    if human:
        return human
    reviewer = str(row.get("reviewer_paradigm_role") or "").strip()
    if reviewer:
        return reviewer
    return str(row.get("paradigm_role") or "").strip()


_VALIDATION_FLAGS = frozenset(
    {
        "wording_exceeds_tier",
        "role_evidence_exceeds_tier",
        "causal_wording_withheld",
        "hypothesis_not_supported",
        "cross_case",
    }
)


def sync_validation_flags(row: dict) -> None:
    """Replace current validation flags from the effective statement, role, and tier.

    Flags this function does not own stay. ``validation_basis`` is the
    statement and role this call just used, including when the owned flags
    did not change. When the owned set changes, the previous flags and the
    basis they used are appended to ``validation_revision_history``. A
    statement edit that leaves the flags the same does not append. A second
    call with the same inputs does not append again. ``needs_review`` is set
    when the current row overreaches and is not cleared here.
    """
    tier = row.get("evidence_tier")
    if isinstance(tier, bool) or not isinstance(tier, int):
        tier = None
    statement = effective_statement(row)
    role = effective_paradigm_role(row)
    statement_over = statement_wording_overreach(statement, tier)
    role_over = role_evidence_overreach(tier, role)
    wanted: set[str] = set()
    if tier == 1:
        wanted.add("hypothesis_not_supported")
        if row.get("cross_case"):
            wanted.add("cross_case")
    if statement_over:
        wanted.add("wording_exceeds_tier")
    if role_over:
        wanted.add("role_evidence_exceeds_tier")
    if (statement_over or role_over) and isinstance(tier, int) and tier < 3:
        wanted.add("causal_wording_withheld")
    current = [str(item) for item in (row.get("flags") or [])]
    previous = [item for item in current if item in _VALIDATION_FLAGS]
    basis = row.get("validation_basis") if isinstance(row.get("validation_basis"), Mapping) else None
    if set(previous) != wanted:
        if previous or basis is not None:
            history = [dict(item) for item in (row.get("validation_revision_history") or []) if isinstance(item, Mapping)]
            history.append(
                {
                    "flags": previous,
                    "statement": None if basis is None else basis.get("statement"),
                    "role": None if basis is None else basis.get("role"),
                }
            )
            row["validation_revision_history"] = history
    row["validation_basis"] = {"statement": statement, "role": role}
    if statement_over or role_over:
        row["needs_review"] = True
    row["flags"] = [item for item in current if item not in _VALIDATION_FLAGS] + [
        item for item in ("cross_case", "hypothesis_not_supported", "wording_exceeds_tier", "role_evidence_exceeds_tier", "causal_wording_withheld") if item in wanted
    ]


def _claim_flags(claims: Mapping[str, Any]) -> list[str]:
    participant = claims.get("participant_explanation") if isinstance(claims.get("participant_explanation"), Mapping) else {}
    status = str(participant.get("status") or "")
    flags = []
    if status in {"denied", "contested"}:
        flags.append("participant_denies_link")
    qualifications = {str(item) for item in (participant.get("qualifications") or [])}
    if "frequency_negation" in qualifications:
        flags.append("frequency_negation")
    if "frequency_qualified" in qualifications:
        flags.append("frequency_qualified")
    if "negation_does_not_reject_link" in qualifications:
        flags.append("negation_does_not_reject_link")
    if status == "contested":
        flags.append("participant_link_contested")
    if status == "reversed":
        flags.append("participant_direction_reversed")
    if status == "hedged":
        flags.append("participant_link_hedged")
    if status == "not_a_link":
        flags.append("participant_link_not_a_relationship")
    if status == "direction_unspecified":
        flags.append("participant_direction_unspecified")
    if status == "hypothetical":
        flags.append("participant_link_hypothetical")
    if status == "ambiguous":
        flags.append("participant_link_ambiguous")
    if status == "unsupported":
        flags.append("participant_link_unsupported")
    analyst = claims.get("analyst_interpretation") if isinstance(claims.get("analyst_interpretation"), Mapping) else {}
    if analyst.get("present"):
        flags.append("analyst_interpretation")
    return flags


def _claim_snapshot(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if "claims" not in row:
        return None
    claims = row.get("claims") if isinstance(row.get("claims"), Mapping) else {}
    participant = claims.get("participant_explanation") if isinstance(claims.get("participant_explanation"), Mapping) else {}
    analyst = claims.get("analyst_interpretation") if isinstance(claims.get("analyst_interpretation"), Mapping) else {}
    return {
        "evidence_tier": row.get("evidence_tier"),
        "attribution_source": row.get("attribution_source"),
        "downstream_eligible": row.get("downstream_eligible"),
        "allowed_wording": row.get("allowed_wording"),
        "claim_flags": [str(item) for item in (row.get("flags") or []) if str(item) in _CLAIM_FLAGS],
        "participant_status": participant.get("status"),
        "participant_link_text": participant.get("link_text"),
        "stated_direction": participant.get("stated_direction"),
        "polarity": participant.get("polarity"),
        "hedging": participant.get("hedging"),
        "analyst_present": bool(analyst.get("present")),
        "analyst_wording": analyst.get("allowed_wording"),
    }


def _note_legacy_analyst_exclusion(relationship: dict) -> None:
    """Remember a 0.9.19 row-level exclusion before this reassessment replaces it."""
    if "claims" in relationship:
        return
    flags = {str(item) for item in (relationship.get("flags") or [])}
    source = str(relationship.get("attribution_source") or "")
    if "analyst_interpretation" not in flags and source != "analyst_interpretation":
        return
    history = [dict(item) for item in (relationship.get("claim_revision_history") or []) if isinstance(item, Mapping)]
    history.append(
        {
            "policy_version": "0.9.19",
            "attribution_source": relationship.get("attribution_source"),
            "downstream_eligible": relationship.get("downstream_eligible"),
            "evidence_tier": relationship.get("evidence_tier"),
            "allowed_wording": relationship.get("allowed_wording"),
            "claim_flags": ["analyst_interpretation"],
            "analyst_present": True,
            "note": _LEGACY_ANALYST_NOTE,
        }
    )
    relationship["claim_revision_history"] = history


def _record_claim_transition(relationship: dict, previous: dict[str, Any] | None) -> None:
    current = _claim_snapshot(relationship)
    if previous is None or current is None or previous == current:
        return
    history = [dict(item) for item in (relationship.get("claim_revision_history") or []) if isinstance(item, Mapping)]
    history.append(previous)
    relationship["claim_revision_history"] = history


def apply_evidence_tier(relationship: dict, **kwargs: Any) -> dict[str, Any]:
    """Write the assessment onto the row. Does not decide disposition or tier 4.

    Claim flags are recomputed from the current evidence. An earlier
    interpretation stays on ``claim_revision_history`` and does not keep the
    row ineligible.
    """
    _note_legacy_analyst_exclusion(relationship)
    previous = _claim_snapshot(relationship)
    fields = assess_relationship(relationship, **kwargs)
    relationship.update(fields)
    flags = [str(item) for item in (relationship.get("flags") or []) if str(item) not in _CLAIM_FLAGS]
    claim_flags = _claim_flags(fields.get("claims") or {})
    if fields.get("shared_quote_inspect") and "shared_quote_inspect" not in claim_flags:
        claim_flags.append("shared_quote_inspect")
    relationship["flags"] = flags + claim_flags
    if claim_flags:
        relationship["needs_review"] = True
    sync_validation_flags(relationship)
    tier = fields.get("evidence_tier")
    disposition = str(relationship.get("disposition") or "")
    if disposition not in {"rejected", "merged"} and isinstance(tier, int):
        relationship["evidence_status"] = fields["evidence_label"]
    assign_relationship_eligibility(relationship)
    _record_claim_transition(relationship, previous)
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
    assign_relationship_eligibility(relationship)
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
    lexical_only: list | None = None,
) -> list[dict[str, Any]]:
    """Passages from other respondents whose text contains both endpoints' distinctive terms.

    A passage that shares only one distinctive stem with either endpoint is not
    offered. When ``lexical_only`` is a list, those passages are appended there
    with ``support_cap`` 1. The returned list is empty when no slot remains.
    Callers keep this inside the evidence budget.
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
            from_overlap = tokens & from_terms
            to_overlap = tokens & to_terms
            if not from_overlap or not to_overlap:
                continue
            # One distinctive stem on either side is the seed11 "training" shape.
            # It is labeled and withheld. It is not offered as cross-case evidence
            # and it cannot support a tier above 1. A weak match does not reserve
            # the passage: a later pair may still qualify it.
            if len(from_overlap) < _GROUNDED_STEMS or len(to_overlap) < _GROUNDED_STEMS:
                if lexical_only is not None:
                    lexical_only.append(
                        {
                            "passage_id": passage_id,
                            "doc_id": passage.get("doc_id") or passage.get("document_id"),
                            "respondent_id": case_id,
                            "code_ids": list(code_ids),
                            "matched_stems": sorted(from_overlap | to_overlap),
                            "lexical_only": True,
                            "candidate_cross_case": False,
                            "support_cap": 1,
                        }
                    )
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
                    "lexical_only": False,
                    "matched_stems": sorted(from_overlap | to_overlap),
                }
            )
            if len(chosen) >= limit:
                return chosen
    return chosen


_EVIDENCE_CLASSES = ("same_passage", "same_respondent", "lexical_cross_case")


def _doc_id_of(passage: Mapping[str, Any]) -> str:
    return str(passage.get("doc_id") or passage.get("document_id") or passage.get("passage_id") or "")


def classify_pair_evidence(
    category_ids: Sequence[str],
    passages: Sequence[Mapping[str, Any]] | None,
    segments: Sequence[Mapping[str, Any]] | None,
    codebook: Mapping[str, Any] | None = None,
    *,
    pattern_min_cases: int = PATTERN_MIN_CASES,
) -> dict[str, Any]:
    """Strongest evidence class for each unordered pair, in a stable order.

    Same-passage co-occurrence comes first, then the same respondent, then a
    lexical cross-case candidate that shares at least two distinctive stems
    with each endpoint. A one-stem overlap is counted as ``lexical_only`` and
    is not an evidence class. The result does not change which pairs exist.
    """
    ids: list[str] = []
    for item in category_ids:
        text = str(item)
        if text and text not in ids:
            ids.append(text)
    by_id: dict[str, Mapping[str, Any]] = {}
    ordered_passages: list[Mapping[str, Any]] = []
    for item in passages or []:
        if not isinstance(item, Mapping) or item.get("synthetic"):
            continue
        passage_id = str(item.get("passage_id") or "")
        if not passage_id or passage_id in by_id:
            continue
        by_id[passage_id] = item
        ordered_passages.append(item)
    occurrences: dict[str, list[str]] = {code_id: [] for code_id in ids}
    for segment in segments or []:
        if not isinstance(segment, Mapping):
            continue
        code_id = str(segment.get("code_id") or "")
        passage_id = str(segment.get("passage_id") or "")
        if code_id not in occurrences or passage_id not in by_id:
            continue
        if passage_id not in occurrences[code_id]:
            occurrences[code_id].append(passage_id)
    cases: dict[str, list[str]] = {}
    docs: dict[str, list[str]] = {}
    for code_id, passage_ids in occurrences.items():
        case_ids: list[str] = []
        doc_ids: list[str] = []
        for passage_id in passage_ids:
            passage = by_id[passage_id]
            case_id = case_id_of(passage)
            doc_id = _doc_id_of(passage)
            if case_id and case_id not in case_ids:
                case_ids.append(case_id)
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        cases[code_id] = case_ids
        docs[code_id] = doc_ids
    classes: dict[str, list[tuple[str, str]]] = {name: [] for name in _EVIDENCE_CLASSES}
    lexical_only_pairs = 0
    reaching = 0
    minimum = max(1, int(pattern_min_cases or PATTERN_MIN_CASES))
    occurrence_sets = {code_id: set(passage_ids) for code_id, passage_ids in occurrences.items()}
    case_sets = {code_id: set(case_ids) for code_id, case_ids in cases.items()}
    endpoint_terms = {code_id: _endpoint_terms(code_id, codebook, segments) for code_id in ids}
    passage_tokens: dict[str, set[str]] = {}
    passage_case: dict[str, str] = {}
    token_index: dict[str, list[str]] = {}
    passage_order: list[str] = []
    for passage in ordered_passages:
        passage_id = str(passage.get("passage_id") or "")
        if not passage_id or passage_id in passage_tokens:
            continue
        tokens = meaningful_tokens(str(passage.get("text") or ""))
        passage_tokens[passage_id] = tokens
        passage_case[passage_id] = case_id_of(passage)
        passage_order.append(passage_id)
        for token in tokens:
            token_index.setdefault(token, []).append(passage_id)
    endpoint_hits: dict[str, set[str]] = {}
    for code_id, terms in endpoint_terms.items():
        hits: set[str] = set()
        for token in terms:
            hits.update(token_index.get(token) or ())
        endpoint_hits[code_id] = hits
    order_index = {passage_id: index for index, passage_id in enumerate(passage_order)}
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            left_passages = occurrence_sets.get(left) or set()
            right_passages = occurrence_sets.get(right) or set()
            right_cases = case_sets.get(right) or set()
            shared_cases = [case_id for case_id in cases.get(left) or [] if case_id in right_cases]
            if len(shared_cases) >= minimum:
                reaching += 1
            if left_passages and right_passages and left_passages & right_passages:
                classes["same_passage"].append((left, right))
                continue
            if shared_cases:
                classes["same_respondent"].append((left, right))
                continue
            from_terms = endpoint_terms.get(left) or set()
            to_terms = endpoint_terms.get(right) or set()
            if not from_terms or not to_terms:
                continue
            lexical = False
            stem_only = False
            occupied = (case_sets.get(left) or set()) | (case_sets.get(right) or set())
            from_hits = endpoint_hits.get(left) or set()
            to_hits = endpoint_hits.get(right) or set()
            shared_hits = sorted(from_hits & to_hits, key=order_index.__getitem__)
            for passage_id in shared_hits:
                case_id = passage_case.get(passage_id) or ""
                if occupied and case_id in occupied:
                    continue
                tokens = passage_tokens.get(passage_id) or set()
                from_overlap = tokens & from_terms
                to_overlap = tokens & to_terms
                if not from_overlap or not to_overlap:
                    continue
                if len(from_overlap) >= _GROUNDED_STEMS and len(to_overlap) >= _GROUNDED_STEMS:
                    lexical = True
                    break
                stem_only = True
            if lexical:
                classes["lexical_cross_case"].append((left, right))
            elif stem_only:
                lexical_only_pairs += 1
    single_document = sum(1 for code_id in ids if len(docs.get(code_id) or []) == 1)
    return {
        "by_class": {name: [{"from_id": left, "to_id": right} for left, right in rows] for name, rows in classes.items()},
        "evidence_available": {name: len(rows) for name, rows in classes.items()},
        "plausible_count": sum(len(rows) for rows in classes.values()),
        "lexical_only_count": lexical_only_pairs,
        "single_document_code_count": single_document,
        "pairs_reaching_pattern_min_cases": reaching,
        "pattern_min_cases": minimum,
        "code_count": len(ids),
    }


def _attribution_probe(quote: str) -> dict[str, Any]:
    """Score one synthetic tools/crew quote with the public tier function.

    The corpus is the same shape the regression tests use: two respondents,
    a rival passage, and endpoint labels. It is not study data.
    """
    text = f"The tools for the shift sit beside coworkers on the crew tonight {quote}"
    passages = {
        "p1": {"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": text},
        "p2": {"passage_id": "p2", "doc_id": "d2", "respondent_id": "r2", "text": text},
        "p3": {
            "passage_id": "p3",
            "doc_id": "d3",
            "respondent_id": "r3",
            "text": "The schedule changed and nobody mentioned either side.",
        },
    }
    codebook = {
        "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
        "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
    }
    segments = [
        {"segment_id": "s1", "code_id": "tools", "passage_id": "p1", "quote": "tools for the shift sit"},
        {"segment_id": "s2", "code_id": "crew", "passage_id": "p1", "quote": "coworkers on the crew tonight"},
        {"segment_id": "s3", "code_id": "tools", "passage_id": "p2", "quote": "tools for the shift sit"},
        {"segment_id": "s4", "code_id": "crew", "passage_id": "p2", "quote": "coworkers on the crew tonight"},
    ]
    row = {
        "relationship_id": "rel-probe",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": "",
        "quote_from": "tools for the shift sit",
        "quote_to": "coworkers on the crew tonight",
        "statement": "tools for the shift may be related to coworkers on the crew",
        "disposition": "proposed",
        "flags": [],
        "evidence_condition": "anchored",
        "paradigm_role": "unspecified",
        "counterexamples": [
            {
                "passage_id": "p3",
                "judgment": "contradicts",
                "valid_test": True,
                "synthetic": False,
                "quote": "The schedule changed",
            }
        ],
        "case_quotes": [
            {"passage_id": "p1", "quote": quote, "endpoint": "both"},
            {"passage_id": "p2", "quote": quote, "endpoint": "both"},
            {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
            {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
            {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        ],
    }
    apply_evidence_tier(row, passages=passages, codebook=codebook, segments=segments, memos=[])
    return row


def score_attribution_accuracy(cases: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Score the deterministic attribution rule on quotes a caller supplies.

    ``None`` or an empty list leaves ``accuracy`` null and ``correct_count``
    null. This function does not load the packaged adversarial fixtures.
    Those fixtures are regression coverage, not this metric. No model is called.
    Each case needs ``quote`` and ``expected_status``. ``authorizes_tier_3``
    is checked when the case includes it.
    """
    empty = {
        "accuracy": None,
        "denominator": 0,
        "correct_count": None,
        "model_judgments_used": False,
        "note": (
            "No external quotes were supplied. "
            "The packaged attribution fixtures are regression coverage, not this metric."
        ),
    }
    if not cases:
        return empty
    correct = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("An attribution accuracy case must be an object.")
        quote = str(case.get("quote") or "")
        expected = case.get("expected_status", case.get("status"))
        expected_status = None if expected is None else str(expected)
        assessed = _attribution_probe(quote)
        link = assessed.get("participant_link") if isinstance(assessed.get("participant_link"), Mapping) else {}
        status_ok = expected_status is not None and link.get("status") == expected_status
        auth_ok = True
        if "authorizes_tier_3" in case:
            auth_ok = link.get("authorizes_tier_3") is case.get("authorizes_tier_3")
        ok = status_ok and auth_ok
        correct += int(ok)
        rows.append(
            {
                "quote": quote,
                "expected_status": expected_status,
                "status": link.get("status"),
                "authorizes_tier_3": link.get("authorizes_tier_3"),
                "evidence_tier": assessed.get("evidence_tier"),
                "correct": ok,
            }
        )
    total = len(rows)
    return {
        "accuracy": round(correct / total, 4) if total else None,
        "denominator": total,
        "correct_count": correct,
        "model_judgments_used": False,
        "note": "Scored only the quotes passed in. The packaged fixtures were not loaded.",
        "cases": rows,
    }


def score_conditional_readings(cases: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Score exact-hedge conditionals on labeled examples the caller supplies.

    ``None`` or an empty list leaves ``accuracy`` null. This function does not
    load the packaged fixture. Passing synthetic examples is a processing
    check. It is not adjudication, and it does not establish that a selection
    is valid. No model is called. Each case needs ``quote``, ``from_terms``,
    ``to_terms``, and ``expected_status``. ``authorizes_tier_3`` is checked
    when the case includes it. A non-validated reading must not authorize tier 3.
    """
    note = (
        "No labeled examples were supplied. Synthetic fixtures are processing checks, "
        "not adjudication and not selection validity. Human-adjudicated examples are not included."
    )
    empty = {
        "accuracy": None,
        "denominator": 0,
        "correct_count": None,
        "model_judgments_used": False,
        "adjudication_data_included": False,
        "selection_validity": False,
        "note": note,
    }
    if not cases:
        return empty
    correct = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("A conditional example must be an object.")
        quote = str(case.get("quote") or "")
        from_terms = _distinctive(" ".join(str(item) for item in (case.get("from_terms") or [])))
        to_terms = _distinctive(" ".join(str(item) for item in (case.get("to_terms") or [])))
        expected = case.get("expected_status", case.get("status"))
        expected_status = None if expected is None else str(expected)
        reading = _interpret_quote(quote, from_terms, to_terms) or {}
        status = reading.get("status")
        authorizes = bool(reading.get("authorizes_tier_3"))
        status_ok = expected_status is not None and status == expected_status
        auth_ok = True
        if "authorizes_tier_3" in case:
            auth_ok = authorizes is bool(case.get("authorizes_tier_3"))
        if status != "validated" and authorizes:
            auth_ok = False
        ok = status_ok and auth_ok
        correct += int(ok)
        rows.append(
            {
                "quote": quote,
                "expected_status": expected_status,
                "status": status,
                "authorizes_tier_3": authorizes,
                "reading_provisional": True if reading else None,
                "correct": ok,
            }
        )
    total = len(rows)
    return {
        "accuracy": round(correct / total, 4) if total else None,
        "denominator": total,
        "correct_count": correct,
        "model_judgments_used": False,
        "adjudication_data_included": False,
        "selection_validity": False,
        "note": (
            "Scored only the synthetic examples passed in. "
            "That is processing coverage, not adjudication and not selection validity."
        ),
        "cases": rows,
    }
