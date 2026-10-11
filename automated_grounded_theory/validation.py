"""Deterministic checks on model output.

Quotes are kept only when they can be located in the passage. The stored
quote is the passage slice, so later exports stay verbatim even if the model
wrapped a line or swapped a curly quote.
"""

from __future__ import annotations

import re
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
    }
)


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


def locate_quote(quote: str, passage: str) -> tuple[int, int] | None:
    """Find ``quote`` in ``passage`` and return the verbatim span.

    Search order: exact substring, whitespace-flexible token sequence, then
    the same searches after folding curly quotes and dashes (those folds are
    one character to one character, so indexes still refer to ``passage``).
    """
    if quote is None:
        return None
    cleaned = str(quote).strip()
    if not cleaned:
        return None
    span = _locate(cleaned, passage)
    if span is not None:
        return _accept(passage, span)
    folded_quote = cleaned.translate(_FOLD)
    folded_passage = passage.translate(_FOLD)
    span = _locate(folded_quote, folded_passage)
    if span is None:
        return None
    return _accept(passage, span)


def _locate(quote: str, passage: str) -> tuple[int, int] | None:
    index = passage.find(quote)
    if index >= 0:
        return index, index + len(quote)
    tokens = quote.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, passage)
    if not match:
        return None
    return match.start(), match.end()


def _accept(passage: str, span: tuple[int, int]) -> tuple[int, int] | None:
    start, end = span
    verbatim = passage[start:end]
    if len(verbatim.strip()) < 3 or not re.search(r"[A-Za-z]", verbatim):
        return None
    return start, end


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
