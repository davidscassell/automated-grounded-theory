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
from typing import Literal, Mapping, Sequence

from automated_grounded_theory.corpus import LoadedDocument, load_corpus, normalize_newlines
from automated_grounded_theory.validation import exclusion_reason, nonsubstantive_kind

Unit = Literal["paragraph", "sentence", "line", "speaker_turn", "incident"]
UNITS: tuple[str, ...] = ("paragraph", "sentence", "line", "speaker_turn", "incident")
INCIDENT_FLOOR = 150
INCIDENT_TARGET = 800
INCIDENT_CAP = 1200
_QUESTION_SPEAKER = re.compile(
    r"^(interviewer|moderator|question|q)\b",
    re.IGNORECASE,
)
# Zero keeps every speaker turn as its own passage. A positive limit merges a
# short turn into an adjacent answer, but only when no interviewer question
# sits between them. The default is off: a short answer such as "I quit." is
# often the incident, and a length cap cannot tell that from filler like "Ovid."
SHORT_TURN_CHARS = 0
_KNOWN_SPEAKER_LABELS = frozenset(
    {
        "interviewer",
        "moderator",
        "question",
        "q",
        "participant",
        "respondent",
        "interviewee",
        "subject",
        "facilitator",
        "speaker",
    }
)
# Mid-line colons that are not speakers. A label already used at the start of a
# line is still accepted, so a person named one of these is not blocked.
_MIDLINE_SPEAKER_BLOCKLIST = frozenset(
    {
        "note",
        "notes",
        "warning",
        "example",
        "eg",
        "ie",
        "ps",
        "re",
        "fig",
        "figure",
        "table",
        "chapter",
        "section",
        "abstract",
        "summary",
        "time",
        "date",
        "ratio",
        "source",
        "result",
        "results",
        "update",
        "important",
        "caveat",
        "and",
        "but",
        "or",
        "the",
        "then",
        "so",
        "if",
        "when",
        "that",
        "this",
        "well",
        "yes",
        "no",
        "oh",
        "okay",
        "ok",
    }
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
# A label part may be a word, a number (Participant 02), or initials (J.D.).
# The first alternative is initials so "J.D.:" is not read as the word "J".
# Speaker labels in this grammar are Latin-script role names (Interviewer, Participant 02, J.D.).
# Reply text in other scripts is still substantive; nonsubstantive_kind uses Unicode letters.
_WORD_PART = r"[A-Za-z][\w'\-]{0,30}"
_NUMBER_PART = r"\d{1,4}"
_INITIALS_PART = r"[A-Za-z](?:\.[ \t]*[A-Za-z]){1,3}\.?"
_SPEAKER_LABEL = (
    rf"(?:{_INITIALS_PART}(?:[ \t]+{_WORD_PART}){{0,2}}"
    rf"|{_WORD_PART}(?:[ \t]+(?:{_WORD_PART}|{_NUMBER_PART})){{0,3}})"
)
# Line start, after optional indent. "best.   Interviewer:" is not one label:
# a period ends the previous sentence instead of joining the next speaker.
_SPEAKER_LINE = re.compile(rf"(?m)^[ \t]*({_SPEAKER_LABEL}):[ \t]+")
# Mid-line only after a sentence boundary, and only for a short or known label.
_SPEAKER_MID = re.compile(rf"(?<=[.!?])[ \t]+({_SPEAKER_LABEL}):[ \t]+")
_INITIALS_LABEL = re.compile(rf"^{_INITIALS_PART}$")
# Coverage uses this, not the segmentation parser. It is intentionally looser
# so a label the parser missed can still be flagged inside question context.
_LOOSE_SPEAKER = re.compile(
    r"(?m)(?:^|(?<=[.!?]))[ \t]*([A-Za-z][A-Za-z0-9.'\-]{0,40}(?:[ \t]+[A-Za-z0-9][A-Za-z0-9.'\-]{0,20}){0,3}):[ \t]+\S"
)
_SPEAKER = _SPEAKER_LINE


@dataclass
class CodingPlan:
    """Passages in coding order, plus the seed that produced that order."""

    passages: list[dict]
    seed: int
    warnings: list[str] = field(default_factory=list)
    queue: list[int] = field(default_factory=list)
    branch_queues: dict[str, list[int]] = field(default_factory=dict)
    parallel: bool = False
    uncovered_answers: list[dict] = field(default_factory=list)
    attached_preambles: list[dict] = field(default_factory=list)
    source_uncovered: list[dict] = field(default_factory=list)
    question_context_turns: list[dict] = field(default_factory=list)


def interleave_document_queue(passages: Sequence[Mapping], queue: Sequence[int], batch_size: int) -> list[int]:
    """Round-robin passages across documents, keeping each document in reading order.

    A prefix of the result, up to ``batch_size`` passages, takes at most one
    passage from each document. When only one document still has passages, the
    rest of the queue is that document, in order. ``queue`` already lists
    documents in seeded order and passages in reading order within a document.
    """
    width = max(1, int(batch_size))
    by_doc: dict[str, list[int]] = {}
    doc_order: list[str] = []
    for index in queue:
        doc_id = str(passages[index].get("doc_id") or "")
        if doc_id not in by_doc:
            doc_order.append(doc_id)
            by_doc[doc_id] = []
        by_doc[doc_id].append(index)
    cursors = {doc: 0 for doc in doc_order}
    remaining = {doc: len(indices) for doc, indices in by_doc.items()}
    ordered: list[int] = []
    count = len(doc_order)
    cursor = 0
    while any(remaining.values()):
        alive = [doc for doc in doc_order if remaining[doc] > 0]
        if len(alive) <= 1:
            doc = alive[0]
            ordered.extend(by_doc[doc][cursors[doc] :])
            break
        taken: set[str] = set()
        guard = 0
        while len(taken) < min(width, len(alive)) and guard < count * 2:
            doc = doc_order[cursor % count]
            cursor += 1
            guard += 1
            if remaining[doc] <= 0 or doc in taken:
                continue
            ordered.append(by_doc[doc][cursors[doc]])
            cursors[doc] += 1
            remaining[doc] -= 1
            taken.add(doc)
        if not taken:
            break
    return ordered


def next_round_count(passages: Sequence[Mapping], indices: Sequence[int], batch_size: int) -> int:
    """How many leading passages belong to the next interleaved round.

    The count stops before a document would contribute a second passage, and
    it stops at ``batch_size``. A run of one document is cut only by ``batch_size``.
    """
    if not indices:
        return 0
    width = max(1, int(batch_size))
    docs = [str(passages[index].get("doc_id") or "") for index in indices]
    if len(set(docs)) <= 1:
        return min(width, len(indices))
    seen: set[str] = set()
    count = 0
    for doc in docs:
        if doc in seen or count >= width:
            break
        seen.add(doc)
        count += 1
    return count


def plan_coding(
    corpus,
    *,
    unit: Unit = "paragraph",
    seed: int | None = None,
    parallel_documents: bool = False,
    review_mode: str = "document",
    short_turn_chars: int = SHORT_TURN_CHARS,
    code_nonsubstantive: bool = True,
) -> CodingPlan:
    """Segment ``corpus`` and return passages in the order a run would code them.

    Does not call a model. ``seed`` is recorded so the order can be repeated.
    When ``seed`` is omitted, one is drawn and returned on the plan.

    ``review_mode='document'`` (the default) shuffles documents and keeps each
    document's passages in reading order. ``review_mode='wave'`` shuffles
    passages across documents. That weakens constant comparison, because a
    later passage in a document can be coded before an earlier one.

    ``short_turn_chars`` defaults to 0, so every speaker turn stays its own
    passage. A positive value merges a shorter turn into an adjacent answer,
    and never across an interviewer question that sits between them. ``I quit.``
    is a meaningful incident that a length cap would otherwise hide. The text
    is not skipped and not dropped. This changes passage boundaries for a new
    run.

    ``code_nonsubstantive`` defaults to true. Uncertainty, refusal, ``n/a``,
    and punctuation-only replies are queued and flagged (``nonsubstantive``,
    ``nonsubstantive_kind``) so a reader can filter them. Only empty or
    whitespace text is excluded. Set the flag false to exclude those replies
    with ``exclusion_reason`` and leave them out of the model queue.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {', '.join(UNITS)}. Got {unit!r}.")
    if int(short_turn_chars) < 0:
        raise ValueError("short_turn_chars must be zero or more.")
    if review_mode not in {"document", "wave"}:
        raise ValueError(f"review_mode must be 'document' or 'wave'. Got {review_mode!r}.")
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**31)
    documents = load_corpus(corpus)
    warnings: list[str] = []
    passages: list[dict] = []
    for document in documents:
        pieces, piece_warnings = segment_document(
            document,
            unit=unit,
            short_turn_chars=int(short_turn_chars),
        )
        warnings.extend(piece_warnings)
        if not pieces:
            pieces = [_empty_passage(document, unit)]
        passages.extend(pieces)
    if not passages:
        warnings.append("No passages were produced from the corpus.")
        return CodingPlan(
            passages=[],
            seed=seed,
            warnings=warnings,
            parallel=parallel_documents,
            uncovered_answers=_collect_uncovered(passages),
            attached_preambles=_collect_preambles(passages),
            source_uncovered=[],
            question_context_turns=[],
        )

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
        _mark_trivial(passages, warnings, code_nonsubstantive=code_nonsubstantive)
        return CodingPlan(
            passages=passages,
            seed=seed,
            warnings=warnings,
            queue=_live(passages, queue),
            branch_queues={doc_id: _live(passages, indices) for doc_id, indices in branch_queues.items()},
            parallel=True,
            uncovered_answers=_collect_uncovered(passages),
            attached_preambles=_collect_preambles(passages),
            source_uncovered=_collect_marked(passages, "source_uncovered_text", "source_uncovered_chars"),
            question_context_turns=_collect_turns(passages),
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
    _mark_trivial(passages, warnings, code_nonsubstantive=code_nonsubstantive)
    return CodingPlan(
        passages=passages,
        seed=seed,
        warnings=warnings,
        queue=_live(passages, queue),
        branch_queues={},
        parallel=False,
        uncovered_answers=_collect_uncovered(passages),
        attached_preambles=_collect_preambles(passages),
        source_uncovered=_collect_marked(passages, "source_uncovered_text", "source_uncovered_chars"),
        question_context_turns=_collect_turns(passages),
    )


def segment_document(
    document: LoadedDocument,
    *,
    unit: Unit,
    short_turn_chars: int = SHORT_TURN_CHARS,
) -> tuple[list[dict], list[str]]:
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
        return _segment_incidents(document, text, short_turn_chars=short_turn_chars)
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
    if unit == "speaker_turn" and passages:
        passages = _absorb_short_passages(
            text,
            passages,
            limit=int(short_turn_chars),
            answers_only=True,
        )
        _attach_leading_preamble(text, passages, warnings, document.doc_id)
    _apply_source_coverage(text, passages, warnings, document.doc_id)
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


def _mark_trivial(passages: list[dict], warnings: list[str], *, code_nonsubstantive: bool = True) -> None:
    skipped = 0
    flagged = 0
    for passage in passages:
        text = passage.get("text") or ""
        reason = exclusion_reason(text, code_nonsubstantive=code_nonsubstantive)
        kind = nonsubstantive_kind(text) or ""
        passage["skipped"] = reason is not None
        passage["exclusion_reason"] = reason or ""
        passage["coverage_status"] = "excluded" if reason else passage.get("coverage_status") or "pending"
        passage["nonsubstantive"] = bool(kind)
        passage["nonsubstantive_kind"] = kind
        skipped += int(passage["skipped"])
        flagged += int(bool(kind) and reason is None)
    if skipped:
        noun = "passage" if skipped == 1 else "passages"
        if code_nonsubstantive:
            warnings.append(
                f"Retained {skipped} empty {noun}. They were not sent to the model."
            )
        else:
            warnings.append(
                f"Retained {skipped} empty or non-answer {noun} with an exclusion reason. "
                "They were not sent to the model."
            )
    if flagged:
        noun = "passage" if flagged == 1 else "passages"
        warnings.append(
            f"Queued {flagged} non-substantive {noun} for coding and flagged them "
            "so they can be filtered."
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


def _segment_incidents(
    document: LoadedDocument,
    text: str,
    *,
    short_turn_chars: int = SHORT_TURN_CHARS,
) -> tuple[list[dict], list[str]]:
    """Question plus full answer, then sentence-bounded incidents of about 150–800 characters.

    A short answer stays whole. A long answer splits only on sentence boundaries
    and never mid-answer. One sentence longer than the cap stays intact.

    A speaker turn whose text is at most ``short_turn_chars`` is merged into
    the adjacent answer when nothing that looks like an interviewer question
    sits between them. The default is 0, so the merge is off: ``I quit.`` is
    a meaningful incident, and a length cap cannot tell it from filler such
    as ``Ovid.``. The turn is not skipped and its characters stay in a
    passage. New runs pick up the new boundaries; a checkpoint does not
    re-segment.
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
    passages = _absorb_short_passages(
        text,
        passages,
        limit=int(short_turn_chars),
        answers_only=False,
    )
    _attach_leading_preamble(text, passages, warnings, document.doc_id)
    _apply_source_coverage(text, passages, warnings, document.doc_id)
    gap = _uncovered_answer_text(text, pairs, passages)
    if gap:
        warnings.append(
            f"{document.doc_id}: {len(gap)} characters of answer text fell outside every passage."
        )
        for passage in passages:
            metadata = dict(passage.get("metadata") or {})
            metadata["uncovered_answer_chars"] = len(gap)
            metadata["uncovered_answer_text"] = gap
            passage["metadata"] = metadata
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
    matches = _speaker_matches(text)
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
    matches = _speaker_matches(text)
    if not matches:
        return _split_paragraphs(text), (
            "no speaker turns found (expected lines like 'Name: ...'); fell back to paragraphs"
        )
    spans: list[tuple[int, int]] = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        chunk_end = end
        while chunk_end > match.start() and text[chunk_end - 1].isspace():
            chunk_end -= 1
        if match.start() < chunk_end:
            spans.append((match.start(), chunk_end))
    return spans, None


class _SpeakerHit:
    """One speaker label. ``start`` is the label, not the indent before it."""

    def __init__(self, match: re.Match[str], *, label_start: int | None = None, label: str | None = None) -> None:
        self._match = match
        self._label_start = label_start
        self._label = label

    def start(self) -> int:
        if self._label_start is not None:
            return self._label_start
        return self._match.start(1)

    def end(self) -> int:
        return self._match.end()

    def group(self, index: int) -> str:
        if index == 1 and self._label is not None:
            return self._label
        return self._match.group(index)


def _line_speaker_hit(match: re.Match[str]) -> _SpeakerHit:
    """If a line-start label ends in a known role, keep only that word.

    ``download the results   Interviewer:`` is the role ``Interviewer``. The
    words before it stay on the previous turn.
    """
    label = match.group(1)
    words = label.split()
    if len(words) >= 2 and words[-1].casefold() in _KNOWN_SPEAKER_LABELS and label.endswith(words[-1]):
        last = words[-1]
        return _SpeakerHit(match, label_start=match.start(1) + len(label) - len(last), label=last)
    return _SpeakerHit(match)


def _speaker_matches(text: str) -> list[_SpeakerHit]:
    """Speaker labels at line start (indent allowed) and short or known labels mid-line.

    A mid-line label has to follow sentence punctuation. That keeps ``best.
    Interviewer:`` and `` Participant:`` without splitting on ``Note:`` or
    ``I said:``. A line-start label whose last word is a known role matches
    only that word, so ``download the results   Interviewer:`` is ``Interviewer``.
    """
    line_hits = [
        hit
        for hit in (_line_speaker_hit(match) for match in _SPEAKER_LINE.finditer(text))
        if _label_allowed(hit.group(1))
    ]
    seen = {hit.group(1).strip().casefold() for hit in line_hits}
    covered = [(hit.start(), hit.end()) for hit in line_hits]
    found = list(line_hits)
    for match in _SPEAKER_MID.finditer(text):
        if any(start <= match.start(1) < end for start, end in covered):
            continue
        label = match.group(1).strip()
        if not _midline_label_ok(label, seen) or not _label_allowed(label):
            continue
        found.append(_SpeakerHit(match))
        covered.append((match.start(), match.end()))
    found.sort(key=lambda item: item.start())
    kept: list[_SpeakerHit] = []
    last_end = -1
    for match in found:
        if match.start() < last_end:
            continue
        kept.append(match)
        last_end = match.end()
    return kept


def _is_initials(label: str) -> bool:
    return _INITIALS_LABEL.fullmatch(label.strip()) is not None


def _initials_blocked(label: str) -> bool:
    """Latin abbreviations and similar dotted forms are not speakers."""
    stripped = label.strip().casefold().rstrip(".")
    letters = re.sub(r"[^a-z]", "", stripped)
    blocked = {item.rstrip(".") for item in _ABBREVIATIONS}
    blocked.update({"eg", "ie", "ps", "vs", "us", "usa", "eg", "ie"})
    return stripped in blocked or letters in blocked or letters in _ABBREVIATIONS


def _label_allowed(label: str) -> bool:
    """Numbered and initialed labels, without figure/table/abbreviation hits.

    A number is allowed on a label whose other words are not the midline
    blocklist (``Table 1`` is not a speaker; ``Participant 02`` is). Initials
    such as ``J.D.`` are speakers. ``e.g.`` and ``U.S.`` are not.
    """
    text = label.strip()
    if not text:
        return False
    if _is_initials(text):
        return not _initials_blocked(text)
    words = text.split()
    if any(word.isdigit() for word in words):
        letters = [word for word in words if not word.isdigit()]
        if not letters or not any(character.isalpha() for character in text):
            return False
        if any(word.casefold() in _MIDLINE_SPEAKER_BLOCKLIST for word in letters):
            return False
    return True


def _midline_label_ok(label: str, seen: set[str]) -> bool:
    key = label.strip().casefold()
    if not key:
        return False
    if key in _KNOWN_SPEAKER_LABELS or key in seen:
        return True
    if _is_initials(label) and not _initials_blocked(label):
        return True
    words = label.split()
    letter_words = [word for word in words if not word.isdigit()]
    if any(word.isdigit() for word in words):
        if not letter_words or any(word.casefold() in _MIDLINE_SPEAKER_BLOCKLIST for word in letter_words):
            return False
        if not all(word[:1].isupper() for word in letter_words):
            return False
        if any(word.casefold() in _KNOWN_SPEAKER_LABELS for word in letter_words):
            return True
        return 2 <= len(label) <= 24 and 1 <= len(letter_words) <= 2
    short = 2 <= len(label) <= 24 and 1 <= len(words) <= 2 and all(word[:1].isupper() for word in words)
    if not short or key in _MIDLINE_SPEAKER_BLOCKLIST:
        return False
    return True


def _passage_is_question(passage: Mapping) -> bool:
    chunk = str(passage.get("text") or "").lstrip()
    match = re.match(r"([A-Za-z][\w .'\-]{0,40}):", chunk)
    if not match:
        return False
    return _QUESTION_SPEAKER.match(match.group(1).strip()) is not None


def _absorb_short_passages(
    text: str,
    passages: list[dict],
    *,
    limit: int,
    answers_only: bool,
) -> list[dict]:
    """Fold a short turn into the adjacent answer. Never drop its characters.

    ``limit`` of zero leaves the passages as they are. The merged passage is
    one slice of ``text``, so characters between the two turns stay in that
    slice. A question that was only stored on the short turn is kept on the
    surviving passage's question field.
    """
    if limit <= 0 or len(passages) < 2:
        _relink_passages(passages)
        return passages

    def is_answer(passage: Mapping) -> bool:
        if not answers_only:
            return True
        return not _passage_is_question(passage)

    def is_short(passage: Mapping) -> bool:
        return len(str(passage.get("text") or "").strip()) <= limit

    remaining = [dict(passage) for passage in passages]
    index = 0
    changed = False
    while index < len(remaining):
        passage = remaining[index]
        if not is_short(passage) or not is_answer(passage):
            index += 1
            continue
        target = _adjacent_answer(text, remaining, index, is_answer)
        if target is None:
            index += 1
            continue
        _extend_passage(text, target, passage)
        del remaining[index]
        changed = True
    if changed:
        _reindex_passages(remaining)
    else:
        _relink_passages(passages)
        return passages
    return remaining


def _gap_has_question(text: str, left: Mapping, right: Mapping) -> bool:
    """True when an interviewer question sits between two passages.

    Merging across that question would attach one answer to a different
    question. The characters between the spans are the check. A question that
    is only stored on a passage's question field still appears in ``text``.
    """
    left_start = int(left.get("start_offset") or 0)
    left_end = int(left.get("end_offset") or 0)
    right_start = int(right.get("start_offset") or 0)
    right_end = int(right.get("end_offset") or 0)
    gap_start = min(left_end, right_end)
    gap_end = max(left_start, right_start)
    if gap_end <= gap_start:
        return False
    gap = text[gap_start:gap_end]
    for hit in _speaker_matches(gap):
        if _QUESTION_SPEAKER.match(hit.group(1).strip()):
            return True
    return False


def _adjacent_answer(text: str, passages: Sequence[Mapping], index: int, is_answer) -> dict | None:
    """Nearest answer that can be merged without crossing an interviewer question."""
    passage = passages[index]
    candidates: list[Mapping] = []
    for cursor in range(index - 1, -1, -1):
        if is_answer(passages[cursor]):
            candidates.append(passages[cursor])
            break
    for cursor in range(index + 1, len(passages)):
        if is_answer(passages[cursor]):
            candidates.append(passages[cursor])
            break
    for target in candidates:
        if _gap_has_question(text, passage, target):
            continue
        return target
    return None


def _extend_passage(text: str, target: dict, extra: Mapping) -> None:
    start = min(int(target.get("start_offset") or 0), int(extra.get("start_offset") or 0))
    end = max(int(target.get("end_offset") or 0), int(extra.get("end_offset") or 0))
    target["start_offset"] = start
    target["end_offset"] = end
    target["text"] = text[start:end]
    extra_question = str(extra.get("question") or "").strip()
    target_question = str(target.get("question") or "").strip()
    if extra_question and extra_question not in target_question:
        target["question"] = f"{extra_question}\n{target_question}".strip()
    metadata = dict(target.get("metadata") or {})
    merged = [dict(item) for item in (metadata.get("merged_passage_offsets") or []) if isinstance(item, Mapping)]
    merged.append(
        {
            "start": int(extra.get("start_offset") or 0),
            "end": int(extra.get("end_offset") or 0),
            "passage_id": str(extra.get("passage_id") or ""),
        }
    )
    metadata["merged_passage_offsets"] = merged
    metadata["merged_short_turn"] = True
    target["metadata"] = metadata


def _reindex_passages(passages: list[dict]) -> None:
    for index, passage in enumerate(passages):
        doc_id = str(passage.get("doc_id") or "")
        passage["position"] = index
        passage["passage_id"] = f"{doc_id}-p{index:04d}"
        passage["coding_order"] = index
    _relink_passages(passages)


def _relink_passages(passages: Sequence[dict]) -> None:
    for index, passage in enumerate(passages):
        passage["prev_text"] = passages[index - 1]["text"] if index else ""
        passage["next_text"] = passages[index + 1]["text"] if index + 1 < len(passages) else ""


def _attach_leading_preamble(text: str, passages: list[dict], warnings: list[str], doc_id: str) -> None:
    """Pull text before the first speaker label into the earliest passage.

    ``Participant 02`` ahead of ``Institution:`` is the example. The characters
    stay in that passage. A warning and ``attached_preamble_text`` record it
    the same way uncovered answer text is recorded.
    """
    matches = _speaker_matches(text)
    if not matches or not passages:
        return
    boundary = matches[0].start()
    prefix = text[:boundary]
    if boundary <= 0 or not prefix.strip():
        return
    lead = len(prefix) - len(prefix.lstrip())
    if any(int(item.get("start_offset") or 0) <= lead and int(item.get("end_offset") or 0) >= boundary for item in passages):
        return
    host = None
    for passage in passages:
        start = int(passage.get("start_offset") or 0)
        if start >= boundary and not text[boundary:start].strip():
            host = passage
            break
    if host is None:
        host = min(passages, key=lambda item: int(item.get("start_offset") or 0))
    start = int(host.get("start_offset") or 0)
    end = int(host.get("end_offset") or 0)
    if lead >= start:
        return
    host["start_offset"] = lead
    host["end_offset"] = max(end, lead)
    host["text"] = text[lead : int(host["end_offset"])]
    snippet = prefix.strip()
    warnings.append(
        f"{doc_id}: {len(snippet)} characters before the first speaker label "
        "were attached to the first passage."
    )
    for passage in passages:
        metadata = dict(passage.get("metadata") or {})
        metadata["attached_preamble_chars"] = len(snippet)
        metadata["attached_preamble_text"] = snippet
        passage["metadata"] = metadata
    _relink_passages(passages)


def _collect_preambles(passages: Sequence[Mapping]) -> list[dict]:
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for passage in passages:
        metadata = passage.get("metadata") or {}
        gap = str(metadata.get("attached_preamble_text") or "")
        doc_id = str(passage.get("doc_id") or "")
        marker = (doc_id, gap)
        if not gap or marker in seen:
            continue
        seen.add(marker)
        found.append({"doc_id": doc_id, "chars": len(gap), "text": gap})
    return found


def _uncovered_answer_text(
    text: str,
    pairs: Sequence[tuple[str, int, int]],
    passages: Sequence[Mapping],
) -> str:
    """Answer characters that sit in no passage. Whitespace between slices does not count."""
    if not text or not pairs:
        return ""
    covered = [False] * len(text)
    for passage in passages:
        start = max(0, int(passage.get("start_offset") or 0))
        end = min(len(text), int(passage.get("end_offset") or 0))
        for index in range(start, end):
            covered[index] = True
    regions: list[str] = []
    for _question, start, end in pairs:
        buf_start = None
        left = max(0, start)
        right = min(len(text), end)
        for index in range(left, right):
            if covered[index]:
                if buf_start is not None:
                    regions.append(text[buf_start:index])
                    buf_start = None
                continue
            if buf_start is None:
                buf_start = index
        if buf_start is not None:
            regions.append(text[buf_start:right])
    kept = [region.strip() for region in regions if region.strip()]
    return "\n".join(kept)


def _apply_source_coverage(text: str, passages: list[dict], warnings: list[str], doc_id: str) -> None:
    """Compare source characters with passage text and question context.

    This does not ask the speaker parser which regions are answers. Every
    non-whitespace character has to sit in a passage span or in question
    context (the question string plus the speaker label that introduces it).
    A speaker-like stretch that is only inside a question is flagged. Any
    other missing character is warned.
    """
    report = _source_text_coverage(text, passages)
    uncovered = str(report.get("uncovered") or "")
    chars = int(report.get("uncovered_chars") or 0)
    turns = [str(item) for item in (report.get("question_turns") or []) if str(item)]
    if chars:
        warnings.append(
            f"{doc_id}: {chars} non-whitespace characters are in neither a passage nor question context."
        )
    if turns:
        warnings.append(f"{doc_id}: speaker turn text is only in question context: " + " | ".join(turns))
    if not passages or (not chars and not turns):
        return
    for passage in passages:
        metadata = dict(passage.get("metadata") or {})
        if chars:
            metadata["source_uncovered_chars"] = chars
            metadata["source_uncovered_text"] = uncovered
        if turns:
            metadata["question_context_turns"] = list(turns)
        passage["metadata"] = metadata


def _source_text_coverage(text: str, passages: Sequence[Mapping]) -> dict:
    """Parser-independent coverage of ``text``.

    ``covered`` is 2 inside a passage span and 1 inside question context.
    Passage text wins when the two overlap. The return value counts
    non-whitespace characters that are still 0, and lists speaker-like text
    that is only in question context.
    """
    covered = bytearray(len(text))
    for passage in passages:
        start = max(0, int(passage.get("start_offset") or 0))
        end = min(len(text), int(passage.get("end_offset") or 0))
        for index in range(start, end):
            covered[index] = 2
    bodies: list[tuple[int, int]] = []
    for passage in passages:
        question = str(passage.get("question") or "")
        if not question:
            continue
        start_offset = int(passage.get("start_offset") or 0)
        index = text.rfind(question, 0, max(start_offset, 0))
        if index < 0:
            index = text.find(question)
        if index < 0:
            continue
        bodies.append((index, index + len(question)))
        line_start = text.rfind("\n", 0, index) + 1
        prefix = text[line_start:index]
        context_start = index
        if re.fullmatch(r"[ \t]*[^\n:]{1,80}:[ \t]*", prefix):
            context_start = line_start
        for cursor in range(context_start, index + len(question)):
            if covered[cursor] == 0:
                covered[cursor] = 1
    uncovered_chars = 0
    regions: list[str] = []
    region_start = None
    for index, character in enumerate(text):
        if covered[index] == 0:
            # Whitespace stays inside the run so the warning is the source text.
            if not character.isspace():
                uncovered_chars += 1
                if region_start is None:
                    region_start = index
            continue
        if region_start is not None:
            regions.append(text[region_start:index].strip())
            region_start = None
    if region_start is not None:
        regions.append(text[region_start:].strip())
    turns: list[str] = []
    seen: set[str] = set()
    for match in _LOOSE_SPEAKER.finditer(text):
        label_at = match.start(1)
        if not any(start <= label_at < end for start, end in bodies):
            continue
        if label_at < len(covered) and covered[label_at] == 2:
            continue
        label = match.group(1).strip()
        if not _question_context_speaker(label):
            continue
        line_end = text.find("\n", match.end())
        if line_end < 0:
            line_end = len(text)
        snippet = text[match.start(1) : line_end].strip()
        if snippet and snippet not in seen:
            seen.add(snippet)
            turns.append(snippet)
    kept = [region for region in regions if region]
    return {
        "uncovered": "\n".join(kept),
        "uncovered_chars": uncovered_chars,
        "question_turns": turns,
    }


def _question_context_speaker(label: str) -> bool:
    """True when a label inside a question looks like its own speaker turn.

    ``Note:`` and ``Table 1:`` stay ordinary prose. ``Participant 02:`` and
    ``J.D.:`` do not. The question's own interviewer label is outside the
    question body, so a role label found in the body is an extra turn.
    """
    text = label.strip()
    if not text:
        return False
    if _is_initials(text):
        return not _initials_blocked(text)
    words = text.split()
    letters = [word for word in words if not word.isdigit()]
    if letters and all(word.casefold() in _MIDLINE_SPEAKER_BLOCKLIST for word in letters):
        return False
    if any(word.casefold() in _MIDLINE_SPEAKER_BLOCKLIST for word in letters) and not any(
        word.casefold() in _KNOWN_SPEAKER_LABELS for word in letters
    ):
        if not any(word.isdigit() for word in words):
            return False
    if _QUESTION_SPEAKER.match(text):
        return True
    if any(word.casefold() in _KNOWN_SPEAKER_LABELS for word in letters):
        return True
    if any(word.isdigit() for word in words) and letters:
        return True
    return bool(letters) and all(word[:1].isupper() for word in letters)


def _collect_marked(passages: Sequence[Mapping], text_key: str, chars_key: str) -> list[dict]:
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for passage in passages:
        metadata = passage.get("metadata") or {}
        gap = str(metadata.get(text_key) or "")
        doc_id = str(passage.get("doc_id") or "")
        marker = (doc_id, gap)
        if not gap or marker in seen:
            continue
        seen.add(marker)
        found.append({"doc_id": doc_id, "chars": int(metadata.get(chars_key) or len(gap)), "text": gap})
    return found


def _collect_turns(passages: Sequence[Mapping]) -> list[dict]:
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for passage in passages:
        metadata = passage.get("metadata") or {}
        doc_id = str(passage.get("doc_id") or "")
        for item in metadata.get("question_context_turns") or []:
            text = str(item or "").strip()
            marker = (doc_id, text)
            if not text or marker in seen:
                continue
            seen.add(marker)
            found.append({"doc_id": doc_id, "text": text})
    return found


def _collect_uncovered(passages: Sequence[Mapping]) -> list[dict]:
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for passage in passages:
        metadata = passage.get("metadata") or {}
        gap = str(metadata.get("uncovered_answer_text") or "")
        doc_id = str(passage.get("doc_id") or "")
        marker = (doc_id, gap)
        if not gap or marker in seen:
            continue
        seen.add(marker)
        found.append({"doc_id": doc_id, "chars": len(gap), "text": gap})
    return found
