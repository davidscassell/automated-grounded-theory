"""Deterministic checks on model output.

A quote mismatch never drops a code. ``anchor_quote`` searches the passage,
then the neighbor windows, then the document. Before it declares a quote
unanchored it also searches the other passages in the same request. A hit
there is kept, flagged ``evidence_outside_passage``, and attributed to that
passage, the same way a neighbor hit is. It records ``quote_status`` as
``exact``, ``normalized_match``, ``fuzzy_match``, ``possibly_modified``, or
``unanchored``, plus the best source span when one is found. The stored quote
is that source slice when a span exists, so exports stay tied to the text
even if the model wrapped a line or swapped a curly quote. A hit that sits
wholly outside the passage being coded is still kept. That is a textual
match, not an assignment to the passage under coding: the code is flagged
``evidence_outside_passage``. Attribution uses the actual matched spans, not
the bounding interval between them. One overlapping passage is the
attribution. Spans in several passages stay multi-attributed and the review
flag stays. When the spans cannot be placed, attribution is unresolved and
the coded passage is not used as a fallback.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

GENERIC_CODES = frozenset(
    {
        "discussion",
        "discussions",
        "experience",
        "experiences",
        "stuff",
        "thing",
        "things",
        "data",
        "information",
        "topic",
        "topics",
        "theme",
        "themes",
        "other",
        "miscellaneous",
        "general",
        "n/a",
        "na",
        "none",
        "text",
        "content",
        "response",
        "responses",
        "answer",
        "answers",
        "participant",
        "participants",
        "interview",
        "interviews",
        "coding",
        "code",
        "codes",
        "misc",
        "various",
        "issue",
        "issues",
        "aspect",
        "aspects",
        "comment",
        "comments",
        "feedback",
        "opinion",
        "opinions",
        "feeling",
        "feelings",
        "thought",
        "thoughts",
    }
)
_STOPWORDS = frozenset({"a", "an", "the", "of", "and", "to", "in", "on", "for", "with", "or"})
_FALSE_GERUNDS = frozenset(
    {
        "something",
        "anything",
        "nothing",
        "everything",
        "morning",
        "evening",
        "string",
        "during",
    }
)
_GERUND = re.compile(r"\b[\w']+ing\b", re.IGNORECASE)
_FOLD = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
)
# A near-copy (typo, missing letter, dropped function word) is a fuzzy match.
# A real wording change is only possibly the same span, and is flagged.
FUZZY_MATCH_MIN = 0.90
POSSIBLY_MODIFIED_MIN = 0.60
QUOTE_FLAG_STATUSES = frozenset({"possibly_modified", "unanchored"})
_ELLIPSIS = re.compile(r"\.\.\.|…|\[(?:\.\.\.|…|\.\s\.\s\.)\]")


def _is_letter_or_number(char: str) -> bool:
    """Unicode letters (L*) and numbers (N*), not an ASCII range."""
    category = unicodedata.category(char)
    return category.startswith("L") or category.startswith("N")


def _is_word_char(char: str) -> bool:
    return char == "'" or _is_letter_or_number(char)


def has_unicode_letter(text: str) -> bool:
    return any(unicodedata.category(char).startswith("L") for char in text)


def has_unicode_alnum(text: str) -> bool:
    return any(_is_letter_or_number(char) for char in text)


def iter_words(text: str):
    """Yield ``(start, end, token)`` for letters, numbers, and apostrophes.

    ASCII tokens match the old ``[A-Za-z0-9']+`` split. Letters outside that
    range stay inside the token instead of disappearing.
    """
    start = None
    for index, char in enumerate(text):
        if _is_word_char(char):
            if start is None:
                start = index
        elif start is not None:
            yield start, index, text[start:index]
            start = None
    if start is not None:
        yield start, len(text), text[start:]


def code_key(label: str) -> str:
    """Identity key: case-folded, whitespace collapsed."""
    return re.sub(r"\s+", " ", label.strip().casefold())


def display_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip())


# A blank label is stored under this name so the incident stays in the book.
EMPTY_LABEL = "(empty label)"
# Empty and over-long labels are retried, then kept. Generic labels are not.
LENIENT_LABEL_REASONS = frozenset({"empty_code", "code_too_long"})
LABEL_HOLD_REASONS = ("empty_code", "code_too_long", "bundled_concepts")
_CONCEPT_AND = re.compile(r"\band\b", re.IGNORECASE)


def gerund_ok(label: str) -> bool:
    """True when the label has an -ing token that is not a false friend."""
    for match in _GERUND.finditer(label):
        if match.group(0).casefold() not in _FALSE_GERUNDS:
            return True
    return False


def is_generic_code(label: str) -> bool:
    key = code_key(label)
    if not key or key in GENERIC_CODES:
        return True
    words = [word.casefold() for _start, _end, word in iter_words(key) if word.casefold() not in _STOPWORDS]
    if not words:
        return True
    return all(word in GENERIC_CODES for word in words)


def flexible_contains(needle: str, haystack: str) -> bool:
    tokens = needle.split()
    if not tokens:
        return False
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    return re.search(pattern, haystack, flags=re.IGNORECASE) is not None


def in_vivo_supported(label: str, passage: str) -> bool:
    """An in-vivo label has to be the participant's words, present in the passage."""
    return flexible_contains(label, passage)


def split_joined_label(label: str) -> list[str]:
    """Split a semicolon-joined label into separate candidate codes.

    A label with no semicolon is one piece, including a blank label. Empty
    pieces around a semicolon are dropped. A label that is only semicolons
    becomes one blank piece so it is flagged instead of discarded.
    """
    text = str(label or "")
    if ";" not in text:
        return [text]
    parts = [display_label(part) for part in text.split(";") if display_label(part)]
    return parts or [""]


def bundled_concepts(label: str) -> bool:
    """True when a label names several concepts instead of one.

    A comma list of three or more concepts counts. So does a chain of three
    or more concepts joined by ``and`` (``flexibility and friction and
    coworkers``). One pair, such as ``pay and benefits``, does not.
    """
    shown = display_label(label)
    if not shown:
        return False
    comma_parts = [part.strip() for part in shown.split(",") if part.strip()]
    if len(comma_parts) >= 3:
        return True
    and_parts = [part.strip() for part in _CONCEPT_AND.split(shown) if part.strip()]
    return len(and_parts) >= 3


def lenient_label_fields(label: str, *, in_vivo: bool, passage: str) -> dict[str, Any]:
    """Normalized fields for an empty or over-long label that is kept.

    A blank label is stored as ``(empty label)`` so the incident has a key
    and can be reviewed. The caller still flags the original failure.
    """
    shown = display_label(label) or EMPTY_LABEL
    kept_in_vivo = bool(in_vivo) and shown != EMPTY_LABEL and in_vivo_supported(shown, passage)
    return {
        "code": shown,
        "key": code_key(shown),
        "in_vivo": kept_in_vivo,
        "gerund_ok": gerund_ok(shown),
    }


@dataclass(frozen=True)
class QuoteAnchor:
    """Where a model quote sits in the source, and how sure that placement is."""

    status: str
    similarity: float | None
    source: str
    quote: str
    model_quote: str
    spans: tuple[tuple[int, int], ...] = ()
    doc_spans: tuple[tuple[int, int], ...] = ()
    flags: tuple[str, ...] = ()
    located_passage_id: str = ""
    located_doc_id: str = ""
    located_doc_name: str = ""

    @property
    def doc_start(self) -> int | None:
        if not self.doc_spans:
            return None
        return self.doc_spans[0][0]

    @property
    def doc_end(self) -> int | None:
        if not self.doc_spans:
            return None
        return self.doc_spans[-1][1]


def locate_quote(quote: str, passage: str) -> tuple[int, int] | None:
    """Return one source span when the quote matches exactly or after normalization.

    Fuzzy, possibly-modified, and unanchored quotes return ``None``. Call
    :func:`anchor_quote` when the code must be kept either way.
    """
    anchor = anchor_quote(quote, passage)
    if anchor.status not in {"exact", "normalized_match"} or not anchor.spans:
        return None
    start = anchor.spans[0][0]
    end = anchor.spans[-1][1]
    verbatim = passage[start:end]
    if len(verbatim.strip()) < 3 or not has_unicode_letter(verbatim):
        return None
    return start, end


def anchor_quote(
    quote: str,
    passage: str,
    *,
    prev_text: str = "",
    next_text: str = "",
    document_text: str = "",
    passage_start: int = 0,
    passage_end: int | None = None,
    prefer_offset: int | None = None,
    occurrence: int | None = None,
    request_passages: Sequence[Mapping[str, Any]] | None = None,
    exclude_passage_id: str = "",
) -> QuoteAnchor:
    """Anchor ``quote`` without ever rejecting it.

    Search the passage, then the previous and next neighbor windows, then the
    whole document. If that still finds nothing, search the other passages in
    the same request. A hit there is source ``request``: it is kept, flagged
    like a neighbor (``evidence_outside_passage``), and attributed to that
    passage. ``...`` and ``…`` split the quote into ordered spans.
    Whitespace, case, and punctuation differences are ``normalized_match``.
    A small edit is ``fuzzy_match`` and carries ``similarity``. A looser hit is
    ``possibly_modified``. No usable hit is ``unanchored``.
    """
    model_quote = "" if quote is None else str(quote).strip()
    end_offset = passage_start + len(passage) if passage_end is None else passage_end
    regions: list[tuple[str, str, int | None, int | None]] = [
        ("passage", passage or "", passage_start, prefer_offset),
    ]
    if prev_text:
        origin = _region_origin(document_text, prev_text, passage_start, before=True)
        regions.append(("neighbor", prev_text, origin, 0))
    if next_text:
        origin = _region_origin(document_text, next_text, end_offset, before=False)
        regions.append(("neighbor", next_text, origin, 0))
    if document_text:
        regions.append(("document", document_text, 0, passage_start))

    best: QuoteAnchor | None = None
    for source, text, origin, bias in regions:
        allow_fuzzy = best is None or best.status in QUOTE_FLAG_STATUSES
        hit = _anchor_in(
            model_quote,
            text,
            bias=bias,
            allow_fuzzy=allow_fuzzy,
            occurrence=occurrence,
        )
        if hit is None:
            continue
        anchored = _place(hit, source=source, origin=origin, haystack=text, model_quote=model_quote)
        if best is None or _better(anchored, best):
            best = anchored
        if best.status in {"exact", "normalized_match"} and source in {"passage", "neighbor"}:
            break
    if best is not None:
        return best
    requested = _anchor_in_request(
        model_quote,
        request_passages,
        exclude_passage_id=exclude_passage_id,
    )
    if requested is not None:
        return requested
    return QuoteAnchor(
        status="unanchored",
        similarity=None,
        source="",
        quote=model_quote,
        model_quote=model_quote,
    )


def _windows_overlap(start: int, end: int, left: int, right: int) -> bool:
    return start < right and end > left


def evidence_wholly_outside(
    anchor: QuoteAnchor,
    *,
    passage_start: int,
    passage_end: int,
) -> bool:
    """True when a located quote does not overlap the passage being coded.

    Textual status (exact, neighbor, document) is separate from whether that
    hit is evidence for this passage. An unanchored quote is not outside
    evidence; it has no location. A hit that overlaps the passage window is
    not wholly outside.
    """
    if anchor.status == "unanchored" or anchor.source == "passage":
        return False
    if anchor.source == "request":
        return True
    if anchor.source not in {"neighbor", "document"}:
        return False
    spans = [(int(start), int(end)) for start, end in anchor.doc_spans if end > start]
    if not spans:
        return True
    return not any(_windows_overlap(start, end, passage_start, passage_end) for start, end in spans)


@dataclass(frozen=True)
class EvidenceAttribution:
    """Where a located quote sits relative to the passage being coded.

    ``inside`` overlaps the coded passage. ``single`` is wholly outside it
    and the matched spans overlap one other passage. ``multi`` means those
    spans overlap more than one passage; each of those passages is kept.
    ``unresolved`` means the quote is outside the coded passage and the spans
    do not land on a passage. ``passage_ids`` is empty in that case. The coded
    passage is not filled in when attribution fails.
    """

    status: str
    passage_ids: tuple[str, ...] = ()

    @property
    def primary(self) -> str:
        """The one passage id, or empty when attribution is multi or unresolved."""
        if self.status in {"inside", "single"} and len(self.passage_ids) == 1:
            return self.passage_ids[0]
        return ""


def attribute_evidence(
    anchor: QuoteAnchor,
    passages: Sequence[Mapping[str, Any]] | None,
    *,
    doc_id: str = "",
    coded_passage_id: str = "",
    passage_start: int = 0,
    passage_end: int | None = None,
) -> EvidenceAttribution:
    """Attribute ``anchor`` from the matched spans, not their bounding interval.

    A discontinuous quote such as ``I resigned. ... I returned.`` overlaps the
    two end passages. The long passage between them does not contain either
    span, so it is not an attribution. Several passages stay ``multi``. No
    passage stays ``unresolved``.
    """
    end = passage_start if passage_end is None else passage_end
    outside = evidence_wholly_outside(anchor, passage_start=passage_start, passage_end=end)
    if not outside:
        return EvidenceAttribution("inside", (coded_passage_id,) if coded_passage_id else ())
    if anchor.source == "request" and anchor.located_passage_id:
        return EvidenceAttribution("single", (anchor.located_passage_id,))
    spans = [(int(start), int(end)) for start, end in anchor.doc_spans if end > start]
    found = _passages_overlapping(passages or (), spans, doc_id=doc_id)
    if not found and not spans:
        needle = _passage_containing(passages or (), (), doc_id=doc_id, quote=anchor.quote)
        found = (needle,) if needle else ()
    if len(found) == 1:
        return EvidenceAttribution("single", found)
    if len(found) > 1:
        return EvidenceAttribution("multi", found)
    return EvidenceAttribution("unresolved", ())


def attribute_evidence_passage(
    anchor: QuoteAnchor,
    passages: Sequence[Mapping[str, Any]] | None,
    *,
    doc_id: str = "",
    coded_passage_id: str = "",
    passage_start: int = 0,
    passage_end: int | None = None,
) -> str:
    """One passage id for ``anchor``, or empty when that would be a guess.

    Inside the coded passage, this is the coded passage. A single outside
    passage is that passage. Multi-passage and unresolved quotes return an
    empty string. Callers record :attr:`EvidenceAttribution.passage_ids` and
    must not substitute the coded passage for that empty string.
    """
    return attribute_evidence(
        anchor,
        passages,
        doc_id=doc_id,
        coded_passage_id=coded_passage_id,
        passage_start=passage_start,
        passage_end=passage_end,
    ).primary


def attributed_passage_ids(row: Mapping[str, Any]) -> list[str]:
    """Passages a segment or accepted code counts toward.

    Multi-attribution returns every overlapped passage. Unresolved attribution
    returns none, including when ``passage_id`` is still the passage that was
    being coded. Older rows with no attribution status use the single
    attributed id, then ``passage_id``.
    """
    status = str(row.get("attribution") or "")
    raw = row.get("attributed_passage_ids")
    ids = [str(item) for item in raw if str(item)] if isinstance(raw, (list, tuple)) else []
    if status == "unresolved":
        return []
    if status == "multi":
        return ids
    single = str(row.get("attributed_passage_id") or "")
    if single:
        return [single]
    if ids:
        return ids
    if status == "":
        passage = str(row.get("passage_id") or "")
        return [passage] if passage else []
    return []


def _passage_candidates(
    passages: Sequence[Mapping[str, Any]],
    *,
    doc_id: str,
) -> list[Mapping[str, Any]]:
    candidates = []
    for passage in passages:
        if doc_id and str(passage.get("doc_id") or "") not in {"", doc_id}:
            continue
        candidates.append(passage)
    return candidates


def _span_overlap_width(span: tuple[int, int], start: int, end: int) -> int:
    return max(0, min(span[1], end) - max(span[0], start))


def _passages_overlapping(
    passages: Sequence[Mapping[str, Any]],
    spans: Sequence[tuple[int, int]],
    *,
    doc_id: str,
) -> tuple[str, ...]:
    """Passage ids whose text overlaps a matched span. The gap between spans does not count."""
    if not spans:
        return ()
    found: list[str] = []
    for passage in _passage_candidates(passages, doc_id=doc_id):
        start = int(passage.get("start_offset") or 0)
        end = int(passage.get("end_offset") or 0)
        if end <= start:
            continue
        if any(_span_overlap_width(span, start, end) > 0 for span in spans):
            passage_id = str(passage.get("passage_id") or "")
            if passage_id and passage_id not in found:
                found.append(passage_id)
    return tuple(found)


def _passage_containing(
    passages: Sequence[Mapping[str, Any]],
    spans: Sequence[tuple[int, int]],
    *,
    doc_id: str,
    quote: str,
) -> str:
    """One containing passage, or empty when the spans are split or unplaced.

    Overlap is computed per matched span. The bounding interval from the first
    span to the last is not a span, so a long passage that only fills the gap
    is not chosen.
    """
    found = _passages_overlapping(passages, spans, doc_id=doc_id)
    if len(found) == 1:
        return found[0]
    if spans:
        return ""
    needle = (quote or "").strip()
    if needle:
        for passage in _passage_candidates(passages, doc_id=doc_id):
            if needle in str(passage.get("text") or ""):
                return str(passage.get("passage_id") or "")
    return ""


def _better(left: QuoteAnchor, right: QuoteAnchor) -> bool:
    rank = {
        "exact": 4,
        "normalized_match": 3,
        "fuzzy_match": 2,
        "possibly_modified": 1,
        "unanchored": 0,
    }
    left_rank = rank.get(left.status, 0)
    right_rank = rank.get(right.status, 0)
    if left_rank != right_rank:
        return left_rank > right_rank
    return (left.similarity or 0) > (right.similarity or 0) + 1e-9


def _region_origin(document: str, snippet: str, boundary: int, *, before: bool) -> int | None:
    if not document or not snippet:
        return None
    if before:
        index = document.rfind(snippet, 0, max(boundary, 0))
    else:
        index = document.find(snippet, max(boundary, 0))
    if index < 0:
        return None
    return index


def _place(
    hit: "_Hit",
    *,
    source: str,
    origin: int | None,
    haystack: str,
    model_quote: str,
    located_passage_id: str = "",
    located_doc_id: str = "",
    located_doc_name: str = "",
) -> QuoteAnchor:
    doc_spans: tuple[tuple[int, int], ...] = ()
    if origin is not None and hit.spans:
        doc_spans = tuple((origin + start, origin + end) for start, end in hit.spans)
    return QuoteAnchor(
        status=hit.status,
        similarity=hit.similarity,
        source=source,
        quote=_render_quote(haystack, hit.spans, model_quote),
        model_quote=model_quote,
        spans=hit.spans,
        doc_spans=doc_spans,
        flags=hit.flags,
        located_passage_id=located_passage_id,
        located_doc_id=located_doc_id,
        located_doc_name=located_doc_name,
    )


def _anchor_in_request(
    quote: str,
    passages: Sequence[Mapping[str, Any]] | None,
    *,
    exclude_passage_id: str,
) -> QuoteAnchor | None:
    """Last resort before ``unanchored``: the other passages in this request."""
    best: QuoteAnchor | None = None
    for other in passages or ():
        other_id = str(other.get("passage_id") or "")
        if exclude_passage_id and other_id == exclude_passage_id:
            continue
        other_text = str(other.get("text") or "")
        if not other_text:
            continue
        hit = _anchor_in(quote, other_text, bias=0, allow_fuzzy=True)
        if hit is None:
            continue
        origin_raw = other.get("start_offset")
        origin = int(origin_raw) if origin_raw is not None and origin_raw != "" else 0
        anchored = _place(
            hit,
            source="request",
            origin=origin,
            haystack=other_text,
            model_quote=quote,
            located_passage_id=other_id,
            located_doc_id=str(other.get("doc_id") or ""),
            located_doc_name=str(other.get("doc_name") or ""),
        )
        if best is None or _better(anchored, best):
            best = anchored
        if best.status in {"exact", "normalized_match"}:
            break
    return best


def _render_quote(haystack: str, spans: tuple[tuple[int, int], ...], model_quote: str) -> str:
    if not spans:
        return model_quote
    pieces = [haystack[start:end] for start, end in spans]
    if len(pieces) == 1:
        return pieces[0]
    return " ... ".join(pieces)


@dataclass(frozen=True)
class _Hit:
    status: str
    similarity: float | None
    spans: tuple[tuple[int, int], ...]
    flags: tuple[str, ...] = ()


def _anchor_in(
    quote: str,
    haystack: str,
    *,
    bias: int | None,
    allow_fuzzy: bool,
    occurrence: int | None = None,
) -> _Hit | None:
    if not quote or not haystack:
        return None
    parts = _quote_parts(quote)
    if not parts:
        return None
    if len(parts) >= 2:
        multi = _anchor_parts(parts, haystack)
        if multi is not None:
            return multi
    target = parts[0] if len(parts) == 1 else " ".join(parts)
    exact = _exact_span(target, haystack, bias=bias, occurrence=occurrence)
    if exact is not None and _span_ok(haystack, exact[0]):
        return _Hit("exact", 1.0, (exact[0],), exact[1])
    normalized = _normalized_span(target, haystack, bias=bias)
    if normalized is not None and _span_ok(haystack, normalized):
        return _Hit("normalized_match", 1.0, (normalized,))
    if not allow_fuzzy:
        return None
    fuzzy = _fuzzy_span(target, haystack, bias=bias if bias is not None else 0)
    if fuzzy is None:
        return None
    ratio, start, end = fuzzy
    if ratio >= FUZZY_MATCH_MIN:
        status = "fuzzy_match"
    elif ratio >= POSSIBLY_MODIFIED_MIN:
        status = "possibly_modified"
    else:
        return None
    if not _span_ok(haystack, (start, end)):
        return None
    return _Hit(status, round(ratio, 4), ((start, end),))


def _quote_parts(quote: str) -> list[str]:
    folded = str(quote).translate(_FOLD)
    parts = []
    for part in _ELLIPSIS.split(folded):
        cleaned = re.sub(r"\s+", " ", part).strip(" \t\"'")
        if cleaned and cleaned not in {"...", "…"}:
            parts.append(cleaned)
    return parts


def _anchor_parts(parts: list[str], haystack: str) -> _Hit | None:
    cursor = 0
    spans: list[tuple[int, int]] = []
    saw_normalized = False
    for part in parts:
        found = _match_part(part, haystack, cursor)
        if found is None:
            return None
        status, span = found
        if span[0] - cursor > 800 and cursor:
            return None
        if status == "normalized_match":
            saw_normalized = True
        spans.append(span)
        cursor = span[1]
    status = "normalized_match" if saw_normalized else "exact"
    return _Hit(status, 1.0, tuple(spans))


def _match_part(part: str, haystack: str, cursor: int) -> tuple[str, tuple[int, int]] | None:
    window_end = min(len(haystack), cursor + 800 + len(part) + 80)
    region = haystack[cursor:window_end]
    exact = _exact_span(part, region, bias=None)
    if exact is not None and _span_ok(region, exact[0]):
        span = exact[0]
        return "exact", (cursor + span[0], cursor + span[1])
    normalized = _normalized_span(part, region, bias=None)
    if normalized is not None and _span_ok(region, normalized):
        return "normalized_match", (cursor + normalized[0], cursor + normalized[1])
    return None


def _exact_span(
    quote: str,
    haystack: str,
    *,
    bias: int | None,
    occurrence: int | None = None,
) -> tuple[tuple[int, int], tuple[str, ...]] | None:
    cleaned = quote.strip()
    if not cleaned:
        return None
    spans = []
    start = 0
    while True:
        index = haystack.find(cleaned, start)
        if index < 0:
            break
        spans.append((index, index + len(cleaned)))
        start = index + max(1, len(cleaned))
    if not spans:
        return None
    if occurrence is not None:
        if occurrence < 0 or occurrence >= len(spans):
            return None
        return spans[occurrence], ()
    # A repeated quote is not silently the first hit. Neighborhood bias picks
    # one; with no neighborhood the span is kept and flagged.
    if len(spans) > 1 and bias is None:
        return spans[0], ("ambiguous_span",)
    return _closest(spans, bias), ()


def _normalized_span(quote: str, haystack: str, *, bias: int | None) -> tuple[int, int] | None:
    word_hit = _word_span(quote, haystack, bias=bias)
    if word_hit is not None:
        return word_hit
    needle, _needle_map = _normalize_with_map(quote)
    folded, indexes = _normalize_with_map(haystack)
    if not needle or not folded:
        return None
    spans = []
    start = 0
    while True:
        index = folded.find(needle, start)
        if index < 0:
            break
        origin = indexes[index]
        finish = indexes[index + len(needle) - 1] + 1
        finish = _extend_trailing(haystack, finish, quote)
        if _on_word_boundary(haystack, origin, finish):
            spans.append((origin, finish))
        start = index + max(1, len(needle))
    if not spans:
        return None
    return _closest(spans, bias)


def _word_span(quote: str, haystack: str, *, bias: int | None) -> tuple[int, int] | None:
    """Match case, whitespace, and punctuation, but not a prefix of a longer word."""
    needle = _norm_tokens(quote)
    hay_tokens = _token_spans(haystack)
    if not needle or len(hay_tokens) < len(needle):
        return None
    spans = []
    width = len(needle)
    words = [item[2] for item in hay_tokens]
    for index in range(len(words) - width + 1):
        if words[index : index + width] != needle:
            continue
        start = hay_tokens[index][0]
        finish = _extend_trailing(haystack, hay_tokens[index + width - 1][1], quote)
        spans.append((start, finish))
    if not spans:
        return None
    return _closest(spans, bias)


def _on_word_boundary(haystack: str, start: int, end: int) -> bool:
    if start > 0 and haystack[start - 1].isalnum():
        return False
    if end < len(haystack) and haystack[end].isalnum():
        return False
    return True


def _closest(spans: list[tuple[int, int]], bias: int | None) -> tuple[int, int]:
    if bias is None:
        return spans[0]
    return min(spans, key=lambda span: (abs(span[0] - bias), span[0]))


def _extend_trailing(haystack: str, end: int, quote: str) -> int:
    """Keep trailing punctuation the quote itself included."""
    needed = []
    for char in reversed(quote.strip()):
        folded = char.translate(_FOLD)
        if folded.isalnum() or folded == "'":
            break
        if folded.isspace():
            continue
        needed.append(folded)
    needed.reverse()
    index = end
    for want in needed:
        while index < len(haystack) and haystack[index].isspace():
            index += 1
        if index < len(haystack) and haystack[index].translate(_FOLD) == want:
            index += 1
        else:
            break
    return index


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    indexes: list[int] = []
    for index, char in enumerate(text):
        for folded in char.translate(_FOLD).casefold():
            if folded.isalnum() or folded == "'":
                chars.append(folded)
                indexes.append(index)
    return "".join(chars), indexes


def _fuzzy_span(quote: str, haystack: str, *, bias: int) -> tuple[float, int, int] | None:
    quote_tokens = _norm_tokens(quote)
    hay_tokens = _token_spans(haystack)
    if len(quote_tokens) < 2 or len("".join(quote_tokens)) < 8 or not hay_tokens:
        return None
    anchors = [token for token in quote_tokens if len(token) >= 4] or quote_tokens
    positions: list[int] = []
    for token in sorted(anchors, key=len, reverse=True):
        positions = [index for index, item in enumerate(hay_tokens) if item[2] == token]
        if positions:
            break
    if not positions:
        return None
    positions = positions[:30]
    joined = " ".join(quote_tokens)
    width = len(quote_tokens)
    best: tuple[float, int, int] | None = None
    for pos in positions:
        for size in range(max(1, width - 2), width + 3):
            first = max(0, pos - size + 1)
            last_start = min(pos, len(hay_tokens) - size)
            if last_start < first:
                continue
            for start_i in range(first, last_start + 1):
                end_i = start_i + size
                window = " ".join(hay_tokens[index][2] for index in range(start_i, end_i))
                ratio = SequenceMatcher(None, window, joined).ratio()
                start = hay_tokens[start_i][0]
                end = hay_tokens[end_i - 1][1]
                if best is None or ratio > best[0] + 1e-9 or (
                    abs(ratio - best[0]) <= 1e-9 and abs(start - bias) < abs(best[1] - bias)
                ):
                    best = (ratio, start, end)
    if best is None or best[0] < POSSIBLY_MODIFIED_MIN:
        return None
    return best


def _norm_tokens(text: str) -> list[str]:
    return [token.casefold() for _start, _end, token in iter_words(text.translate(_FOLD))]


def _token_spans(text: str) -> list[tuple[int, int, str]]:
    return [
        (start, end, token.translate(_FOLD).casefold())
        for start, end, token in iter_words(text)
    ]


def _span_ok(haystack: str, span: tuple[int, int]) -> bool:
    start, end = span
    if start < 0 or end > len(haystack) or end - start < 3:
        return False
    verbatim = haystack[start:end]
    return has_unicode_letter(verbatim)


def assess_label(label: str, *, in_vivo: bool, passage: str, require_gerund: bool) -> tuple[bool, list[str], dict[str, Any]]:
    """Return whether a label may enter the codebook, review flags, and normalized fields."""
    reasons: list[str] = []
    shown = display_label(label)
    if not shown:
        return False, ["empty_code"], {}
    word_limit = 12 if in_vivo else 8
    char_limit = 120 if in_vivo else 80
    words = shown.split()
    if len(words) > word_limit or len(shown) > char_limit:
        return False, ["code_too_long"], {}
    if is_generic_code(shown):
        return False, ["generic_code"], {}
    in_vivo_flag = bool(in_vivo)
    if in_vivo_flag and not in_vivo_supported(shown, passage):
        in_vivo_flag = False
        reasons.append("in_vivo_not_in_passage")
    has_gerund = gerund_ok(shown)
    if not has_gerund and not in_vivo_flag:
        if require_gerund:
            return False, ["gerund_required"], {}
        reasons.append("gerund_warning")
    return True, reasons, {"code": shown, "key": code_key(shown), "in_vivo": in_vivo_flag, "gerund_ok": has_gerund}


_UNCERTAINTY_EXACT = frozenset(
    {
        "i don't know",
        "i dont know",
        "i do not know",
        "don't know",
        "dont know",
        "do not know",
        "idk",
        "not sure",
        "unsure",
        "uncertain",
        "no idea",
        "dunno",
    }
)
_REFUSAL_EXACT = frozenset(
    {
        "no comment",
        "no comments",
        "prefer not to say",
        "prefer not to answer",
        "rather not say",
        "rather not answer",
        "decline to answer",
        "declined to answer",
        "declined",
        "refused",
        "refusal",
        "i refuse",
        "i'd rather not say",
        "i would rather not say",
        "no response",
    }
)
_NON_ANSWER_EXACT = frozenset(
    {
        "n/a",
        "na",
        "n.a",
        "n.a.",
        "none",
        "nil",
        "null",
        "nothing",
        "-",
        "--",
        "---",
    }
)
_TRIVIAL_EXACT = _UNCERTAINTY_EXACT | _REFUSAL_EXACT | _NON_ANSWER_EXACT | frozenset({".", "...", "?"})


_EMPTY_MEMO = re.compile(
    r"no property, dimension|no comparison was recorded|nothing to memo|no memo\b",
    re.IGNORECASE,
)


def substantive_memo(text: str) -> bool:
    """True when a memo says something about the incidents.

    The placeholder that only reports the absence of a property is not a memo.
    Short leftovers ("none", "n/a") are not either. A short analytic note can
    still be true here; :func:`memo_review_reasons` is what flags it as thin.
    """
    return memo_review_reasons(text) is not None and len(re.sub(r"\s+", " ", (text or "").strip())) >= 12


def memo_review_reasons(text: str) -> list[str] | None:
    """Flags for a free-form memo, or None when the text is not a memo.

    Blank notes and placeholders ("none", "n/a", "no property was recorded")
    are not memos. A real note is kept. Under 20 words it is flagged ``thin``.
    A missing comparison, property, or question is not a reason to drop it
    and is not by itself a flag.
    """
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return None
    folded = cleaned.casefold().strip(" .")
    if folded in {"none", "n/a", "na", "no comparison", "nothing to add"}:
        return None
    if _EMPTY_MEMO.search(cleaned):
        return None
    if len(cleaned.split()) < 20:
        return ["thin"]
    return []


_HEX_CODE_ID = re.compile(r"^[0-9a-f]{16}$")


def is_code_id_text(label: str) -> bool:
    """True when a label is a raw 16-character code id, not a human name."""
    return bool(_HEX_CODE_ID.fullmatch(str(label or "").strip().casefold()))


def nonsubstantive_kind(text: str) -> str | None:
    """Why a reply is non-substantive, or None when it is empty or substantive.

    ``uncertainty`` covers "I don't know" and close variants. ``refusal`` covers
    "no comment", "prefer not to say", and similar. ``non_answer`` covers
    ``n/a`` and ``none``. ``no_substantive_text`` is punctuation with no
    letters or digits. Empty or whitespace text is not a kind: it stays excluded.
    A longer sentence that happens to mention uncertainty is substantive.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    folded = re.sub(r"\s+", " ", stripped).casefold().strip(" .")
    if folded in _UNCERTAINTY_EXACT:
        return "uncertainty"
    if folded in _REFUSAL_EXACT:
        return "refusal"
    if folded in _NON_ANSWER_EXACT:
        return "non_answer"
    if folded in _TRIVIAL_EXACT or not has_unicode_alnum(stripped):
        return "no_substantive_text"
    return None


def is_trivial_passage(text: str, *, code_nonsubstantive: bool = True) -> bool:
    """True when a passage is not sent to the model.

    Only empty or whitespace text is excluded when ``code_nonsubstantive`` is
    true. Uncertainty, refusal, ``n/a``, and punctuation-only replies are data
    and are coded. Set ``code_nonsubstantive`` false to exclude those as well.
    A short substantive reply such as "yes" is kept either way.
    """
    return exclusion_reason(text, code_nonsubstantive=code_nonsubstantive) is not None


def exclusion_reason(text: str, *, code_nonsubstantive: bool = True) -> str | None:
    """Why a row stays out of the model, or None when it should be coded.

    Empty or whitespace text is ``empty`` and is always excluded. Uncertainty,
    refusal, ``n/a``, and punctuation-only text are coded when
    ``code_nonsubstantive`` is true; :func:`nonsubstantive_kind` marks them so
    a reader can filter. When ``code_nonsubstantive`` is false those rows are
    excluded as ``non_answer`` or ``no_substantive_text``.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "empty"
    kind = nonsubstantive_kind(text)
    if kind is None or code_nonsubstantive:
        return None
    if kind == "no_substantive_text":
        return "no_substantive_text"
    return "non_answer"


_LEAK_STOP = frozenset(
    "a an the of and or to in on for with by as is are be from that this their its "
    "do you we should any other how what when where who whom whose".split()
)


def context_term_set(context: Any) -> set[str]:
    """Content words from the research context. Labels should not be these words restated."""
    if isinstance(context, str):
        parts = [context]
    else:
        parts = [
            str(getattr(context, "research_question", "") or ""),
            str(getattr(context, "population", "") or ""),
            str(getattr(context, "setting", "") or ""),
            str(getattr(context, "notes", "") or ""),
            " ".join(str(item) for item in (getattr(context, "sensitizing_concepts", []) or [])),
        ]
    words = re.findall(r"[a-z]+", " ".join(parts).lower())
    return {word for word in words if word not in _LEAK_STOP and len(word) > 2}


def label_restates_context(label: str, terms: set[str]) -> bool:
    """True when most of the label's content words are already in the research context.

    The leading gerund is ignored. A label such as ``leaving home`` against a
    sensitizing concept of the same words is a restatement. ``packing bags``
    against that context is not.
    """
    if not terms or not label:
        return False
    words = re.findall(r"[a-z]+", label.lower())
    if words and words[0].endswith("ing") and len(words) > 1:
        words = words[1:]
    content = [word for word in words if word not in _LEAK_STOP and len(word) > 2]
    if not content:
        return False
    return sum(word in terms for word in content) / len(content) >= 0.5


def bounded_unit(value: Any, default: float = 1.0) -> float:
    """Clamp a model confidence to ``[0, 1]``. Unreadable values use ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return min(1.0, max(0.0, number))


def attribute_leak(text: str, metadata: dict | None) -> list[str]:
    """Metadata values that appear as whole tokens in ``text``.

    Definitions should name the incident, not the respondent. A hit is a
    review flag, not a rejected code.
    """
    if not text or not metadata:
        return []
    folded = text.casefold()
    hits: list[str] = []
    for key, value in metadata.items():
        raw = str(value).strip()
        if len(raw) < 4:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(raw.casefold()) + r"(?![a-z0-9])"
        if re.search(pattern, folded):
            hits.append(str(key))
    return hits


def review_passage_flags(
    passage_text: str,
    *,
    quote: str,
    long_passage_chars: int,
    low_confidence: bool,
    confidence: float,
    review_below: float = 0.5,
) -> list[str]:
    reasons: list[str] = []
    if long_passage_chars and len(passage_text) >= long_passage_chars:
        reasons.append("long_passage")
    # The number is the model's own diagnostic, not a probability.
    if low_confidence or (isinstance(confidence, (int, float)) and confidence < review_below):
        reasons.append("low_confidence")
    if passage_text and quote == passage_text and len(passage_text) > 400:
        reasons.append("quote_is_whole_passage")
    elif len(passage_text) > 800 and quote and len(quote) / len(passage_text) > 0.9:
        reasons.append("quote_is_whole_passage")
    return reasons
