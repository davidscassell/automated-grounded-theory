"""Deterministic checks on model output.

A quote mismatch never drops a code. ``anchor_quote`` searches the passage,
then the neighbor windows, then the document. It records ``quote_status`` as
``exact``, ``normalized_match``, ``fuzzy_match``, ``possibly_modified``, or
``unanchored``, plus the best source span when one is found. The stored quote
is that source slice when a span exists, so exports stay tied to the text
even if the model wrapped a line or swapped a curly quote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

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
_WORD = re.compile(r"[A-Za-z0-9']+")


def code_key(label: str) -> str:
    """Identity key: case-folded, whitespace collapsed."""
    return re.sub(r"\s+", " ", label.strip().casefold())


def display_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip())


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
    words = [word for word in re.findall(r"[a-z0-9']+", key) if word not in _STOPWORDS]
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
    if len(verbatim.strip()) < 3 or not re.search(r"[A-Za-z]", verbatim):
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
) -> QuoteAnchor:
    """Anchor ``quote`` without ever rejecting it.

    Search the passage, then the previous and next neighbor windows, then the
    whole document. ``...`` and ``…`` split the quote into ordered spans.
    Whitespace, case, and punctuation differences are ``normalized_match``.
    A small edit is ``fuzzy_match`` and carries ``similarity``. A looser hit is
    ``possibly_modified``. No usable hit is ``unanchored``.
    """
    model_quote = "" if quote is None else str(quote).strip()
    end_offset = passage_start + len(passage) if passage_end is None else passage_end
    regions: list[tuple[str, str, int | None, int | None]] = [
        ("passage", passage or "", passage_start, 0),
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
        hit = _anchor_in(model_quote, text, bias=bias, allow_fuzzy=allow_fuzzy)
        if hit is None:
            continue
        anchored = _place(hit, source=source, origin=origin, haystack=text, model_quote=model_quote)
        if best is None or _better(anchored, best):
            best = anchored
        if best.status in {"exact", "normalized_match"} and source in {"passage", "neighbor"}:
            break
    if best is not None:
        return best
    return QuoteAnchor(
        status="unanchored",
        similarity=None,
        source="",
        quote=model_quote,
        model_quote=model_quote,
    )


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


def _place(hit: "_Hit", *, source: str, origin: int | None, haystack: str, model_quote: str) -> QuoteAnchor:
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
    )


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


def _anchor_in(quote: str, haystack: str, *, bias: int | None, allow_fuzzy: bool) -> _Hit | None:
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
    exact = _exact_span(target, haystack, bias=bias)
    if exact is not None and _span_ok(haystack, exact):
        return _Hit("exact", 1.0, (exact,))
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
    if exact is not None and _span_ok(region, exact):
        return "exact", (cursor + exact[0], cursor + exact[1])
    normalized = _normalized_span(part, region, bias=None)
    if normalized is not None and _span_ok(region, normalized):
        return "normalized_match", (cursor + normalized[0], cursor + normalized[1])
    return None


def _exact_span(quote: str, haystack: str, *, bias: int | None) -> tuple[int, int] | None:
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
    return _closest(spans, bias)


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
    return [match.group(0).casefold() for match in _WORD.finditer(text.translate(_FOLD))]


def _token_spans(text: str) -> list[tuple[int, int, str]]:
    return [
        (match.start(), match.end(), match.group(0).translate(_FOLD).casefold())
        for match in _WORD.finditer(text)
    ]


def _span_ok(haystack: str, span: tuple[int, int]) -> bool:
    start, end = span
    if start < 0 or end > len(haystack) or end - start < 3:
        return False
    verbatim = haystack[start:end]
    return re.search(r"[A-Za-z]", verbatim) is not None


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


_TRIVIAL_EXACT = frozenset(
    {
        "n/a",
        "na",
        "n.a",
        "n.a.",
        "none",
        "nil",
        "null",
        "nothing",
        "no comment",
        "no comments",
        "-",
        "--",
        "---",
        ".",
        "...",
        "?",
    }
)


_EMPTY_MEMO = re.compile(
    r"no property, dimension|no comparison was recorded|nothing to memo|no memo\b",
    re.IGNORECASE,
)


def substantive_memo(text: str) -> bool:
    """True when a memo says something about the incidents.

    The placeholder that only reports the absence of a property is not a memo.
    Short leftovers ("none", "n/a") are not either.
    """
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) < 12:
        return False
    folded = cleaned.casefold().strip(" .")
    if folded in {"none", "n/a", "na", "no comparison", "nothing to add"}:
        return False
    if _EMPTY_MEMO.search(cleaned):
        return False
    return True


def is_trivial_passage(text: str) -> bool:
    """True when a passage has no incident worth a model call.

    Empty comments, punctuation, and explicit non-answers are skipped.
    A short but substantive reply such as "yes" is kept.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    folded = re.sub(r"\s+", " ", stripped).casefold()
    if folded in _TRIVIAL_EXACT:
        return True
    return re.search(r"[A-Za-z0-9]", stripped) is None


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


def review_passage_flags(passage_text: str, *, quote: str, long_passage_chars: int, low_confidence: bool, confidence: float) -> list[str]:
    reasons: list[str] = []
    if long_passage_chars and len(passage_text) >= long_passage_chars:
        reasons.append("long_passage")
    if low_confidence or (isinstance(confidence, (int, float)) and confidence < 0.5):
        reasons.append("low_confidence")
    if passage_text and quote == passage_text and len(passage_text) > 400:
        reasons.append("quote_is_whole_passage")
    elif len(passage_text) > 800 and quote and len(quote) / len(passage_text) > 0.9:
        reasons.append("quote_is_whole_passage")
    return reasons
