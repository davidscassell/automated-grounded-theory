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
from automated_grounded_theory.validation import exclusion_reason

Unit = Literal["paragraph", "sentence", "line", "speaker_turn", "incident"]
UNITS: tuple[str, ...] = ("paragraph", "sentence", "line", "speaker_turn", "incident")
INCIDENT_FLOOR = 150
INCIDENT_TARGET = 800
INCIDENT_CAP = 1200
_QUESTION_SPEAKER = re.compile(
    r"^(interviewer|moderator|question|q)\b",
    re.IGNORECASE,
)

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
    review_mode: str = "document",
) -> CodingPlan:
    """Segment ``corpus`` and return passages in the order a run would code them.

    Does not call a model. ``seed`` is recorded so the order can be repeated.
    When ``seed`` is omitted, one is drawn and returned on the plan.

    ``review_mode='document'`` (the default) shuffles documents and keeps each
    document's passages in reading order. ``review_mode='wave'`` shuffles
    passages across documents. That weakens constant comparison, because a
    later passage in a document can be coded before an earlier one.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {', '.join(UNITS)}. Got {unit!r}.")
    if review_mode not in {"document", "wave"}:
        raise ValueError(f"review_mode must be 'document' or 'wave'. Got {review_mode!r}.")
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

    if review_mode == "document":
        by_doc: dict[str, list[int]] = {}
        doc_order: list[str] = []
        for index, passage in enumerate(passages):
            doc_id = str(passage.get("doc_id") or "")
            if doc_id not in by_doc:
                doc_order.append(doc_id)
                by_doc[doc_id] = []
            by_doc[doc_id].append(index)
        rng.shuffle(doc_order)
        queue = [index for doc_id in doc_order for index in by_doc[doc_id]]
    else:
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
    elif unit == "incident":
        return _segment_incidents(document, text)
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
                "question": "",
                "skipped": False,
                "coverage_status": "pending",
                "exclusion_reason": "",
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
        "question": "",
        "skipped": True,
        "coverage_status": "excluded",
        "exclusion_reason": exclusion_reason(document.text) or "empty",
    }


def _mark_trivial(passages: list[dict], warnings: list[str]) -> None:
    skipped = 0
    for passage in passages:
        reason = exclusion_reason(passage.get("text") or "")
        passage["skipped"] = reason is not None
        passage["exclusion_reason"] = reason or ""
        passage["coverage_status"] = "excluded" if reason else passage.get("coverage_status") or "pending"
        skipped += int(passage["skipped"])
    if skipped:
        noun = "passage" if skipped == 1 else "passages"
        warnings.append(
            f"Retained {skipped} empty or non-answer {noun} with an exclusion reason. "
            "They were not sent to the model."
        )


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


def _segment_incidents(document: LoadedDocument, text: str) -> tuple[list[dict], list[str]]:
    """Question plus full answer, then sentence-bounded incidents of about 150–800 characters.

    A short answer stays whole. A long answer splits only on sentence boundaries
    and never mid-answer. One sentence longer than the cap stays intact.
    """
    warnings: list[str] = []
    pairs = _question_answer_pairs(text)
    passages: list[dict] = []
    for question, answer_start, answer_end in pairs:
        pieces = _pack_incidents(text, answer_start, answer_end)
        if not pieces and text[answer_start:answer_end].strip():
            pieces = [(answer_start, answer_end)]
        for start, end in pieces:
            position = len(passages)
            prev_text = passages[-1]["text"] if passages else ""
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
                    "next_text": "",
                    "unit": "incident",
                    "coding_order": position,
                    "metadata": dict(document.metadata),
                    "question": question,
                    "skipped": False,
                    "coverage_status": "pending",
                    "exclusion_reason": "",
                }
            )
    for index, passage in enumerate(passages):
        if index + 1 < len(passages):
            passage["next_text"] = passages[index + 1]["text"]
    if not passages and text.strip():
        warnings.append(f"{document.doc_id}: incident segmentation found no answer text.")
    return passages, warnings


def _question_answer_pairs(text: str) -> list[tuple[str, int, int]]:
    turns = _speaker_spans(text)
    if not turns:
        stripped = text.strip()
        if not stripped:
            return []
        start = text.find(stripped)
        return [("", start, start + len(stripped))]
    pairs: list[tuple[str, int, int]] = []
    index = 0
    while index < len(turns):
        speaker, start, end, body = turns[index]
        if _QUESTION_SPEAKER.match(speaker):
            question = body
            index += 1
            answer_start = None
            answer_end = end
            while index < len(turns) and not _QUESTION_SPEAKER.match(turns[index][0]):
                if answer_start is None:
                    answer_start = turns[index][1]
                answer_end = turns[index][2]
                index += 1
            if answer_start is None:
                continue
            pairs.append((question, answer_start, answer_end))
            continue
        pairs.append(("", start, end))
        index += 1
    return pairs


def _speaker_spans(text: str) -> list[tuple[str, int, int, str]]:
    matches = list(_SPEAKER.finditer(text))
    if len(matches) < 2:
        return []
    found = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        chunk_end = end
        while chunk_end > match.end() and text[chunk_end - 1].isspace():
            chunk_end -= 1
        body_start = match.end()
        speaker = match.group(1).strip()
        if body_start < chunk_end:
            found.append((speaker, match.start(), chunk_end, text[body_start:chunk_end].strip()))
    return found


def _pack_incidents(text: str, start: int, end: int) -> list[tuple[int, int]]:
    region = text[start:end]
    sentences = _split_sentences(region)
    if not sentences:
        stripped = region.strip()
        if not stripped:
            return []
        local = region.find(stripped)
        return [(start + local, start + local + len(stripped))]
    absolute = [(start + span[0], start + span[1]) for span in sentences]
    if sum(span[1] - span[0] for span in absolute) <= INCIDENT_TARGET:
        return [(absolute[0][0], absolute[-1][1])]
    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    current_len = 0
    for span in absolute:
        length = span[1] - span[0]
        if current and current_len >= INCIDENT_FLOOR and current_len + length > INCIDENT_TARGET:
            groups.append(current)
            current = [span]
            current_len = length
            continue
        if current and current_len + length > INCIDENT_CAP and current_len >= INCIDENT_FLOOR:
            groups.append(current)
            current = [span]
            current_len = length
            continue
        current.append(span)
        current_len += length
    if current:
        groups.append(current)
    return [(group[0][0], group[-1][1]) for group in groups]


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
