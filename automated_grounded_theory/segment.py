"""Segment documents and shuffle passages with a seeded RNG.

Neighbors stay the document neighbors. Only the coding order is randomized,
so a shuffled passage still carries the text around it (Charmaz's concern
that line-by-line coding can lose context, and the JMIR 2025 note that LLM
coding is weak on context).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Literal

from automated_grounded_theory.corpus import LoadedDocument, load_corpus, normalize_newlines
from automated_grounded_theory.validation import is_trivial_passage

Unit = Literal["paragraph", "sentence", "line", "speaker_turn"]
UNITS: tuple[str, ...] = ("paragraph", "sentence", "line", "speaker_turn")

# Titles and a few Latin abbreviations. The sentence splitter does not try to
# solve "U.S.A." or similar chains.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "eg",
        "ie",
        "e.g",
        "i.e",
    }
)
_SPEAKER = re.compile(r"(?m)^([A-Za-z][\w .'\-]{0,40}):[ \t]+")


@dataclass
class CodingPlan:
    """Passages in coding order, plus the seed that produced that order."""

    passages: list[dict]
    seed: int
    warnings: list[str] = field(default_factory=list)
    queue: list[int] = field(default_factory=list)
    branch_queues: dict[str, list[int]] = field(default_factory=dict)
    parallel: bool = False


def plan_coding(
    corpus,
    *,
    unit: Unit = "paragraph",
    seed: int | None = None,
    parallel_documents: bool = False,
) -> CodingPlan:
    """Segment ``corpus`` and return passages in the order a run would code them.

    Does not call a model. ``seed`` is recorded so the order can be repeated.
    When ``seed`` is omitted, one is drawn and returned on the plan.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {', '.join(UNITS)}. Got {unit!r}.")
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**31)
    documents = load_corpus(corpus)
    warnings: list[str] = []
    passages: list[dict] = []
    for document in documents:
        pieces, piece_warnings = segment_document(document, unit=unit)
        warnings.extend(piece_warnings)
        if not pieces:
            pieces = [_empty_passage(document, unit)]
        passages.extend(pieces)
    if not passages:
        warnings.append("No passages were produced from the corpus.")
        return CodingPlan(passages=[], seed=seed, warnings=warnings, parallel=parallel_documents)

    rng = random.Random(seed)
    if parallel_documents:
        branch_queues: dict[str, list[int]] = {}
        by_doc: dict[str, list[int]] = {}
        for index, passage in enumerate(passages):
            by_doc.setdefault(passage["doc_id"], []).append(index)
        for doc_id in sorted(by_doc):
            indices = list(by_doc[doc_id])
            rng.shuffle(indices)
            branch_queues[doc_id] = indices
        _assign_parallel_order(passages, branch_queues)
        queue = [0] * len(passages)
        for index, passage in enumerate(passages):
            queue[passage["coding_order"]] = index
        _mark_trivial(passages, warnings)
        return CodingPlan(
            passages=passages,
            seed=seed,
            warnings=warnings,
            queue=_live(passages, queue),
            branch_queues={doc_id: _live(passages, indices) for doc_id, indices in branch_queues.items()},
            parallel=True,
        )

    queue = list(range(len(passages)))
    rng.shuffle(queue)
    for order, index in enumerate(queue):
        passages[index]["coding_order"] = order
    _mark_trivial(passages, warnings)
    return CodingPlan(
        passages=passages,
        seed=seed,
        warnings=warnings,
        queue=_live(passages, queue),
        branch_queues={},
        parallel=False,
    )


def segment_document(document: LoadedDocument, *, unit: Unit) -> tuple[list[dict], list[str]]:
    """Split one document. Offsets refer to ``document.text``."""
    text = normalize_newlines(document.text)
    warnings: list[str] = []
    if unit == "paragraph":
        spans = _split_paragraphs(text)
    elif unit == "sentence":
        spans = _split_sentences(text)
    elif unit == "line":
        spans = _split_lines(text)
    elif unit == "speaker_turn":
        spans, speaker_warning = _split_speaker_turns(text)
        if speaker_warning:
            warnings.append(f"{document.doc_id}: {speaker_warning}")
    else:
        raise ValueError(f"Unknown unit {unit!r}.")

    spans = [(start, end) for start, end in spans if text[start:end].strip()]
    if not spans and text.strip():
        stripped = text.strip()
        start = text.find(stripped)
        spans = [(start, start + len(stripped))]

    passages: list[dict] = []
    for position, (start, end) in enumerate(spans):
        prev_text = text[spans[position - 1][0] : spans[position - 1][1]] if position else ""
        next_text = text[spans[position + 1][0] : spans[position + 1][1]] if position + 1 < len(spans) else ""
        passages.append(
            {
                "passage_id": f"{document.doc_id}-p{position:04d}",
                "doc_id": document.doc_id,
                "doc_name": document.name,
                "source": document.source,
                "position": position,
                "text": text[start:end],
                "start_offset": start,
                "end_offset": end,
                "prev_text": prev_text,
                "next_text": next_text,
                "unit": unit,
                "coding_order": position,
                "metadata": dict(document.metadata),
                "skipped": False,
            }
        )
    return passages, warnings


def _empty_passage(document: LoadedDocument, unit: str) -> dict:
    """Keep a blank document in the passage list without sending it to a model."""
    return {
        "passage_id": f"{document.doc_id}-p0000",
        "doc_id": document.doc_id,
        "doc_name": document.name,
        "source": document.source,
        "position": 0,
        "text": document.text,
        "start_offset": 0,
        "end_offset": len(document.text),
        "prev_text": "",
        "next_text": "",
        "unit": unit,
        "coding_order": 0,
        "metadata": dict(document.metadata),
        "skipped": True,
    }


def _mark_trivial(passages: list[dict], warnings: list[str]) -> None:
    skipped = 0
    for passage in passages:
        passage["skipped"] = is_trivial_passage(passage.get("text") or "")
        skipped += int(passage["skipped"])
    if skipped:
        noun = "passage" if skipped == 1 else "passages"
        warnings.append(f"Skipped {skipped} empty or trivial {noun} before the model was called.")


def _live(passages: list[dict], indices: list[int]) -> list[int]:
    return [index for index in indices if not passages[index].get("skipped")]


def _assign_parallel_order(passages: list[dict], branch_queues: dict[str, list[int]]) -> None:
    """Number coding order round by round, documents sorted by id."""
    max_rounds = max((len(indices) for indices in branch_queues.values()), default=0)
    order = 0
    for round_index in range(max_rounds):
        for doc_id in sorted(branch_queues):
            indices = branch_queues[doc_id]
            if round_index < len(indices):
                passages[indices[round_index]]["coding_order"] = order
                order += 1


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    # A paragraph is text that may contain single newlines but not a blank line.
    pattern = re.compile(r"(?:[^\n]|\n(?!\s*\n))+")
    for match in pattern.finditer(text):
        raw = match.group(0)
        if not raw.strip():
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        if start < end:
            spans.append((start, end))
    return spans


def _split_lines(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in text.split("\n"):
        if line.strip():
            leading = len(line) - len(line.lstrip())
            trailing = len(line) - len(line.rstrip())
            start = offset + leading
            end = offset + len(line) - trailing
            if start < end:
                spans.append((start, end))
        offset += len(line) + 1
    return spans


def _word_before(text: str, index: int) -> str:
    end = index
    start = index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] in ".'"):
        start -= 1
    return text[start:end]


def _split_sentences(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    n = len(text)
    start = 0
    index = 0
    while index < n and text[index].isspace():
        index += 1
        start = index
    while index < n:
        character = text[index]
        if character in ".!?":
            end_punct = index + 1
            while end_punct < n and text[end_punct] in ".!?\"'”’":
                end_punct += 1
            word = _word_before(text, index).lower().rstrip(".")
            if character == "." and word in _ABBREVIATIONS:
                index += 1
                continue
            end = end_punct
            while end > start and text[end - 1].isspace():
                end -= 1
            piece_start = start
            while piece_start < end and text[piece_start].isspace():
                piece_start += 1
            if piece_start < end:
                spans.append((piece_start, end))
            index = end_punct
            while index < n and text[index].isspace():
                index += 1
            start = index
            continue
        index += 1
    if start < n and text[start:].strip():
        end = n
        while end > start and text[end - 1].isspace():
            end -= 1
        piece_start = start
        while piece_start < end and text[piece_start].isspace():
            piece_start += 1
        if piece_start < end:
            spans.append((piece_start, end))
    return spans


def _split_speaker_turns(text: str) -> tuple[list[tuple[int, int]], str | None]:
    matches = list(_SPEAKER.finditer(text))
    if not matches:
        return _split_paragraphs(text), (
            "no speaker turns found (expected lines like 'Name: ...'); fell back to paragraphs"
        )
    spans: list[tuple[int, int]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        preamble = text[: matches[0].start()]
        leading = len(preamble) - len(preamble.lstrip())
        trailing = len(preamble) - len(preamble.rstrip())
        spans.append((leading, len(preamble) - trailing))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        chunk_end = end
        while chunk_end > match.start() and text[chunk_end - 1].isspace():
            chunk_end -= 1
        if match.start() < chunk_end:
            spans.append((match.start(), chunk_end))
    return spans, None
