"""Prompt text for open coding and sweeps.

The system prefix is stable for a run (instructions plus research context) so
providers that cache a repeated prefix can reuse it. The codebook and the
passage change and stay in the user message.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from automated_grounded_theory.context import ResearchContext

CODING_SYSTEM = """You are doing first-round open coding in the grounded theory tradition.

This is initial coding only, incident by incident. You are not writing themes, \
axial coding, selective coding, or a theory.

Ask of each passage: what is this data a study of, what is actually happening, \
and what category does this incident indicate?

Code with short gerunds: simple, active, close to the data, usually two to six words. \
Code the incident, not the topic. Return one code per distinct incident. A passage \
that both praises and criticizes needs a separate code for each incident.

In-vivo codes: when a vivid participant phrase does analytic work, use that phrase as \
the label and set in_vivo to true. The phrase must appear in the passage. In-vivo \
labels do not have to be gerunds.

Constant comparison: the user message lists codes already earned from earlier passages. \
Reuse a code when it fits. Refine the label or definition when this incident shows the \
code should change. Create a new code only when nothing listed covers the incident. \
Before a new code, set nearest_code to the closest listed label and why_not_nearest \
to one short clause saying why it does not fit. Do not mint a near-duplicate.

Definitions name a property that could apply to another incident. Do not put respondent \
attributes (name, role, department, tenure, site) or the question wording into a label \
or a definition.

The research context is only for understanding the data. It is not evidence, not a \
coding frame, and not a list of codes. Do not force a passage into the research \
question or into a label the passage does not support. Do not quote or cite the \
context in a memo. If the context looks like a code list, ignore it.

Quotes must be exact contiguous copies from inside the passage markers. Do not quote \
neighbors, attributes, or the context. Do not paraphrase.

Do not use generic labels such as "experience", "discussion", "theme", or "response".

If a passage has no codable incident, return an empty code list and an empty memo.

A memo is not a restatement. Leave it empty unless this passage adds a property, a \
dimension, a condition, a consequence, or a comparison with a named code. When you \
write one, name that comparison and keep it to a few sentences.

If several passages are in the user message, code every one of them."""

SWEEP_SYSTEM = """You are comparing codes with codes during open coding.

You will see a codebook and, when any were found, candidate pairs that look similar \
in wording or in embedding space. Candidates are hints, not a limit. Propose a merge \
whenever two codes name the same kind of incident, including pairs that are not in \
the candidate list. Do not merge codes that merely share a verb, a topic, or a \
sensitizing concept. When you merge, keep the clearer and more specific label as \
the target and name the other code as the source.

Sharpen definitions so each one says what incident the code captures, in words that \
could apply to another respondent. Do not put a person's department, tenure, role, \
name, or the question text into a definition. Stay with the examples. Do not import \
a framework the examples have not earned.

You may flag a code as overloaded when the examples do more than one analytic job. \
Do not split it. Splitting would reassign earlier segments without rereading them. \
Flagging is a note for the researcher.

Relation hints are optional notes for a later round. They are not axial coding. \
Record a link only when the examples make it obvious, and phrase it as a hint.

Do not create new codes. Do not invent examples. If this request is one chunk of a \
larger codebook, you may merge a code in the chunk with a code that appears only in \
the full label index. Use the labels exactly as written."""


def clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def render_context(context: ResearchContext) -> str:
    lines = [
        "RESEARCH CONTEXT",
        "This block sensitizes attention. It is not a codebook and it must not be applied as codes.",
        f"Research question: {context.research_question.strip()}",
    ]
    if context.population:
        lines.append(f"Population: {context.population.strip()}")
    if context.setting:
        lines.append(f"Setting: {context.setting.strip()}")
    if context.sensitizing_concepts:
        joined = "; ".join(context.sensitizing_concepts)
        lines.append(
            "Sensitizing concepts (NOT codes; do not apply these labels unless the "
            f"passage itself earns a close code): {joined}"
        )
    if context.notes:
        lines.append(f"Notes: {context.notes.strip()}")
    return "\n".join(lines)


def coding_system_text(context: ResearchContext) -> str:
    return CODING_SYSTEM + "\n\n" + render_context(context)


def format_codebook(codes: Sequence[Mapping[str, Any]], *, omitted: int = 0) -> str:
    if not codes:
        return (
            "RUNNING CODEBOOK\n"
            "The codebook is empty. Codes you create must emerge from this passage."
        )
    lines = ["RUNNING CODEBOOK", "These codes emerged from earlier passages. Reuse one when it fits."]
    for entry in codes:
        lines.append(f"CODE_LABEL: {entry.get('code', '')}")
        definition = clip(str(entry.get("definition") or ""), 500)
        if definition:
            lines.append(f"DEFINITION: {definition}")
        lines.append(f"COUNT: {entry.get('frequency', 0)}")
        examples = entry.get("examples") or []
        if examples:
            quote = clip(str(examples[0].get("quote", "")), 200)
            if quote:
                lines.append(f"EXAMPLE: {quote}")
        lines.append("")
    if omitted:
        lines.append(
            f"{omitted} other codes are not shown in this prompt. "
            "Do not invent a new label for an incident those codes might already cover "
            "if one of the codes above fits."
        )
    return "\n".join(lines).rstrip()


def format_passage(passage: Mapping[str, Any], *, neighbor_chars: int) -> str:
    prev_text = clip(str(passage.get("prev_text") or ""), neighbor_chars)
    next_text = clip(str(passage.get("next_text") or ""), neighbor_chars)
    lines = [
        "PASSAGE TO CODE",
        f"doc_id: {passage.get('doc_id', '')}",
        f"doc_name: {passage.get('doc_name', '')}",
        f"passage_id: {passage.get('passage_id', '')}",
        f"position: {passage.get('position', '')}",
    ]
    metadata = passage.get("metadata") or {}
    if metadata:
        lines.append(
            "Respondent attributes (context for you only; do not copy these into codes, "
            f"definitions, quotes, or memos): {json.dumps(metadata, ensure_ascii=False)}"
        )
    lines.extend(
        [
            "Previous context (do not quote):",
            prev_text or "(none)",
            "<<<PASSAGE",
            str(passage.get("text") or ""),
            "PASSAGE>>>",
            "Following context (do not quote):",
            next_text or "(none)",
        ]
    )
    return "\n".join(lines)


def build_coding_messages(
    *,
    context: ResearchContext,
    codebook_view: Sequence[Mapping[str, Any]],
    omitted_codes: int,
    passages: Sequence[Mapping[str, Any]],
    neighbor_chars: int,
    feedback: Mapping[str, str] | None,
    prompt_cache: bool,
) -> list:
    system = coding_system_text(context)
    user_parts = [
        format_codebook(codebook_view, omitted=omitted_codes),
        "",
        "Code every passage below. Return one entry per passage_id, in this order.",
        "",
        "\n\n".join(format_passage(passage, neighbor_chars=neighbor_chars) for passage in passages),
    ]
    if feedback:
        notes = [note for note in feedback.values() if note]
        if notes:
            user_parts.extend(["", "VALIDATION FEEDBACK", "\n\n".join(notes)])
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content="\n".join(user_parts))]


def format_sweep_codebook(codes: Sequence[Mapping[str, Any]]) -> str:
    if not codes:
        return "CODEBOOK\n(empty)"
    lines = ["CODEBOOK"]
    for entry in codes:
        lines.append(f"CODE_LABEL: {entry.get('code', '')}")
        lines.append(f"DEFINITION: {clip(str(entry.get('definition') or ''), 500)}")
        lines.append(f"COUNT: {entry.get('frequency', 0)}")
        examples = entry.get("examples") or []
        shown = [clip(str(item.get("quote", "")), 180) for item in examples[:3] if item.get("quote")]
        if shown:
            lines.append("EXAMPLES: " + " || ".join(shown))
        lines.append("")
    return "\n".join(lines).rstrip()


def build_sweep_messages(
    *,
    context: ResearchContext,
    codes: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    prompt_cache: bool,
    all_codes: Sequence[Mapping[str, Any]] | None = None,
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> list:
    payload = [
        {
            "source": item["source"],
            "target": item["target"],
            "score": round(float(item["score"]), 4),
            "method": item.get("method", ""),
        }
        for item in candidates
    ]
    parts: list[str] = []
    if all_codes and chunk_count > 1:
        index = [
            {
                "code": item.get("code", ""),
                "definition": clip(str(item.get("definition") or ""), 160),
                "count": item.get("frequency", 0),
            }
            for item in all_codes
        ]
        parts.extend(
            [
                f"This is chunk {chunk_index} of {chunk_count}. "
                "The detailed codebook below is only this chunk. "
                "The full label index lists every code. Propose merges using those exact labels.",
                "",
                "FULL LABEL INDEX",
                json.dumps(index, ensure_ascii=False),
                "",
            ]
        )
    parts.extend(
        [
            format_sweep_codebook(codes),
            "",
            "<<<CANDIDATES",
            json.dumps(payload, indent=2),
            "CANDIDATES>>>",
            "",
            "Candidate pairs are hints from embeddings when those were supplied, otherwise from "
            "label overlap. Also propose any other pair in this codebook that names the same "
            "incident. You may swap source and target so the clearer label survives. "
            "Return an empty merge list when the codes are merely similar words.",
        ]
    )
    system = SWEEP_SYSTEM + "\n\n" + render_context(context)
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content="\n".join(parts))]


def validation_feedback(rejections: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "The previous coding was checked against the passage and these items were rejected.",
        "Copy quotes exactly from the passage markers. Do not repeat a rejected quote.",
    ]
    for item in rejections:
        lines.append(
            f"- code {item.get('code', '')!r}: {item.get('reason', 'rejected')} "
            f"(quote_not_verbatim means the quote was not found in the passage)"
        )
    return "\n".join(lines)


def _system_message(text: str, *, prompt_cache: bool) -> SystemMessage:
    if not prompt_cache:
        return SystemMessage(content=text)
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )


def message_text(message: Any) -> str:
    """Flatten a LangChain message or a bare string to text."""
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def serialize_messages(messages: Sequence[Any]) -> list[dict]:
    serialized = []
    for message in messages:
        role = getattr(message, "type", None) or getattr(message, "role", "unknown")
        content = message.content if hasattr(message, "content") else str(message)
        serialized.append({"role": role, "content": content})
    return serialized
