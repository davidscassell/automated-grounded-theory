"""Prompt text for open coding and sweeps.

The system prefix is stable for a run (instructions plus research context) so
providers that cache a repeated prefix can reuse it. The codebook and the
passage change and stay in the user message.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from automated_grounded_theory.context import ResearchContext

CODING_SYSTEM = """You are doing first-round open coding in the grounded theory tradition.

This is initial coding only, incident by incident. You are not writing themes, \
axial coding, selective coding, or a theory.

Ask of the passage:
- What is this data a study of?
- What category does this incident indicate?
- What is actually happening here?

Code with short gerunds: labels should be simple, active, and close to the data, \
usually two to six words, naming an action or process. Code the incident, not the topic.

In-vivo codes: when a participant's own phrase does analytic work, you may use that \
phrase as the label and set in_vivo to true. The phrase must appear in the passage. \
In-vivo labels do not have to be gerunds.

Constant comparison: the user message lists codes already earned from earlier passages. \
For each incident, reuse a code when it fits, refine the label or definition when this \
incident shows the code should change, or create a new code when nothing listed covers \
the incident. Do not mint a near-duplicate of a code you can see.

The research context sensitizes attention. It is not a coding frame and it is not a \
list of codes. Do not force the passage into the research question, into a paradigm \
of conditions, strategies, and consequences, or into any label the passage itself does \
not support. If the context looks like a code list, ignore it.

Quotes must be exact, contiguous copies of the passage between the passage markers. \
Do not quote the neighboring context. Do not paraphrase the quote. Do not invent text.

Do not use generic labels such as "experience", "discussion", "theme", or "response".

If nothing in the passage is codable, return an empty code list and say so in the memo.

The memo is a brief analytic note: what the incident suggests, a comparison with the \
codebook, a property you noticed, a question. It is not a summary of the passage."""

SWEEP_SYSTEM = """You are comparing codes with codes during open coding.

You will see a codebook and candidate pairs that look similar in wording or in \
embedding space. Confirm a merge only when two codes name the same kind of incident. \
Do not merge codes that merely share a verb, a topic, or a sensitizing concept. \
When you merge, keep the clearer and more specific label as the target and name the \
other code as the source.

Sharpen definitions so each one says what incident the code captures and, when the \
examples support it, what distinguishes it from a nearby code. Stay with the examples. \
Do not import a theoretical framework the examples have not earned.

You may flag a code as overloaded when the examples are doing more than one analytic \
job. Do not split it. Splitting would reassign earlier segments without rereading them. \
Flagging is a note for the researcher.

Relation hints are optional notes for a later round. They are not axial coding. \
Record a link only when the examples make it obvious, and phrase it as a hint.

Do not create new codes. Do not invent examples. Do not merge a pair that is not in \
the candidate list."""


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
        "Previous context (do not quote):",
        prev_text or "(none)",
        "<<<PASSAGE",
        str(passage.get("text") or ""),
        "PASSAGE>>>",
        "Following context (do not quote):",
        next_text or "(none)",
    ]
    return "\n".join(lines)


def build_coding_messages(
    *,
    context: ResearchContext,
    codebook_view: Sequence[Mapping[str, Any]],
    omitted_codes: int,
    passage: Mapping[str, Any],
    neighbor_chars: int,
    feedback: str | None,
    prompt_cache: bool,
) -> list:
    system = coding_system_text(context)
    user_parts = [
        format_codebook(codebook_view, omitted=omitted_codes),
        "",
        format_passage(passage, neighbor_chars=neighbor_chars),
    ]
    if feedback:
        user_parts.extend(["", "VALIDATION FEEDBACK", feedback])
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
) -> list:
    import json

    payload = [
        {
            "source": item["source"],
            "target": item["target"],
            "score": round(float(item["score"]), 4),
            "method": item.get("method", ""),
        }
        for item in candidates
    ]
    user = "\n".join(
        [
            render_context(context),
            "",
            format_sweep_codebook(codes),
            "",
            "<<<CANDIDATES",
            json.dumps(payload, indent=2),
            "CANDIDATES>>>",
            "",
            "Confirm a merge only for a pair listed in the candidate JSON. "
            "You may swap source and target so the clearer label survives. "
            "Return an empty merge list when the candidates are merely similar words.",
        ]
    )
    return [_system_message(SWEEP_SYSTEM, prompt_cache=prompt_cache), HumanMessage(content=user)]


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
