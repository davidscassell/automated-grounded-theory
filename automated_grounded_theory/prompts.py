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

In-vivo codes: set in_vivo true only when every one of these is true. If one fails, \
in_vivo is false.
- The label is a contiguous copy of the participant's words from this passage.
- Those words are a metaphor, a repetition, or a phrase the participant coined.
- A gerund you compose would lose that analytic point.
Ordinary wording stays a gerund with in_vivo false, even when the quote contains it. \
Do not set in_vivo true merely because the label and the passage share words.

Examples:
- "Pay. Pay. Pay. We're 10-15% below market" → code "Pay. Pay. Pay.", in_vivo true. \
The repetition is the incident.
- "now it feels like a number" → code "feels like a number", in_vivo true. \
The metaphor is the incident.
- "I like my team" has no vivid phrase → code "staying for the team", in_vivo false.

Constant comparison: the user message lists the nearest existing codes, retrieved by \
similarity, plus other frequent codes. Prefer reuse. Prefer refine when this incident \
shows the label or definition should change. Create a new code only when none of the \
nearest codes fits. Before action new, set nearest_code to one of those labels and \
why_not_nearest to the property this incident has that the nearest code does not. \
Do not mint a near-duplicate.

Definitions name a property that could apply to another incident. Do not put respondent \
attributes (name, role, department, tenure, site) or the question wording into a label \
or a definition.

The research context only sensitizes what you notice. Codes must come from the \
participant's language in the passage. Do not turn the research question, the \
population, the setting, or a sensitizing concept into a code unless those exact \
words are the participant's. A label that could have been written from the context \
alone, without this passage, is a leak. Do not write it. Do not quote or cite the \
context in a memo.

Quotes must be exact contiguous copies from inside the passage markers. Do not quote \
neighbors, attributes, or the context. Do not paraphrase.

Do not use generic labels such as "experience", "discussion", "theme", or "response".

If a passage has no codable incident, return an empty code list and an empty memo.

A memo is not a restatement. Leave it empty unless you can write about 50 to 120 \
words that include all three of these: a comparison with a named code (or a statement \
that the book had no close code), a property or dimension this incident varies, and \
one emerging question the next passage could answer. Do not write a sentence that \
only says no property was recorded.

If several passages are in the user message, code every one of them."""

SWEEP_SYSTEM = """You are comparing codes with codes during open coding.

The codebook chunk is the evidence. It is sorted alphabetically by label, then by \
definition, so near-duplicate wording sits together. Read every code. Compare each \
one with the others in the chunk. Propose a merge when two codes name the same \
incident, and put a rationale on that merge that says what incident they share. \
Do this whether or not a similarity hint lists the pair.

Hints, when the user message includes any, are optional and cheap. They come from \
token overlap, and from embeddings only when the caller supplied an embeddings \
model. A hint is not a merge. The absence of a hint is not a reason to keep two \
codes apart. Embeddings are not required.

Each hint has a pair_id. Decide every hinted pair_id once. Put a hint you accept \
in merges. Put a hint you reject in rejections, with that pair_id and a rationale. \
Do not omit a hinted pair. Do not reject a pair that was not hinted. Also propose \
every other duplicate pair you find in this chunk, each with a rationale. Do not \
merge codes that merely share a verb, a topic, or a sensitizing concept. When you \
merge, keep the clearer and more specific label as the target and name the other \
code as the source.

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
        "This block sensitizes attention. It is not a codebook. Codes must come from the participant's words.",
        f"Research question: {context.research_question.strip()}",
    ]
    if context.population:
        lines.append(f"Population: {context.population.strip()}")
    if context.setting:
        lines.append(f"Setting: {context.setting.strip()}")
    if context.sensitizing_concepts:
        joined = "; ".join(context.sensitizing_concepts)
        lines.append(
            "Sensitizing concepts (NOT codes; do not copy these words into a label unless the "
            f"participant used them): {joined}"
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
    lines = [
        "RUNNING CODEBOOK",
        "These codes emerged from earlier passages.",
        "Prefer reuse or refine. A new code requires nearest_code and why_not_nearest.",
    ]
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
    nearest: Sequence[Sequence[str]] | None = None,
) -> list:
    system = coding_system_text(context)
    blocks = []
    for index, passage in enumerate(passages):
        block = format_passage(passage, neighbor_chars=neighbor_chars)
        labels = list(nearest[index]) if nearest and index < len(nearest) else []
        if labels:
            lines = [
                "NEAREST EXISTING CODES",
                "Retrieved by similarity to this passage. Reuse or refine one of these when it fits.",
                "Before action new, set nearest_code to one of these labels and why_not_nearest to the property it does not cover.",
            ]
            lines.extend(f"- {label}" for label in labels)
            block = block + "\n" + "\n".join(lines)
        blocks.append(block)
    user_parts = [
        format_codebook(codebook_view, omitted=omitted_codes),
        "",
        "Code every passage below. Return one entry per passage_id, in this order.",
        "",
        "\n\n".join(blocks),
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
    payload = []
    for index, item in enumerate(candidates, start=1):
        pair_id = str(item.get("pair_id") or f"p{index:04d}")
        item["pair_id"] = pair_id
        payload.append(
            {
                "pair_id": pair_id,
                "source": item["source"],
                "target": item["target"],
                "score": round(float(item["score"]), 4),
                "method": item.get("method", ""),
            }
        )
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
            "Read every CODE_LABEL above and propose duplicate or near-duplicate pairs yourself. "
            "Each merge needs a rationale naming the shared incident. "
            "The candidate list is optional hints (token overlap, plus embeddings only when "
            "those were supplied). Decide every pair_id once: merges if you accept the hint, "
            "rejections if you do not, copying pair_id, source, and target. "
            "Do not reject a pair that is not in that list. "
            "Do propose extra merges for duplicates the hints missed. "
            "You may swap source and target so the clearer label survives.",
        ]
    )
    system = SWEEP_SYSTEM + "\n\n" + render_context(context)
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content="\n".join(parts))]


THEME_SYSTEM = """You are doing focused coding, the round after open coding (Charmaz).

The open codes already exist. Group them into themes. A theme gathers codes \
about the same kind of incident. Name the theme, define the incident it gathers, \
and list a few properties (dimensions that vary inside it) when the codes show any.

Assign every CODE_LABEL to exactly one theme. Copy the label text exactly. Do not \
invent a code, do not rename a code, and do not drop one to keep the list short. \
A code that could sit in two themes still goes to the one it fits best.

Themes are categories of the codes in front of you. The research context is only \
for understanding those codes. It is not a list of themes to force, and it is not \
evidence. Do not mint a theme the codes have not earned. Do not put a respondent's \
department, tenure, role, or name into a theme name or definition.

Tensions are places where themes pull against each other. Negative cases are codes \
that fit a theme awkwardly. An optional quote must be copied from an EXAMPLE line. \
Leave the quote empty rather than paraphrase. passage_id, when you set it, must be \
the one shown next to that example."""


def theme_system_text(context: ResearchContext, n_themes: int | None, max_themes: int = 12) -> str:
    if n_themes:
        count = (
            f"Propose exactly {n_themes} themes. "
            "Do not split one incident across near-duplicate theme names."
        )
    else:
        ceiling = max(1, int(max_themes))
        floor = min(6, ceiling)
        count = (
            f"Propose at most {ceiling} themes, and about {floor} to {ceiling} when the codes "
            "support that many distinct incidents. Fewer is right when the codebook is small. "
            "Do not create one theme per code. Fold near-duplicate themes (two names for pay, "
            "two names for staffing, two names for tools) into one. "
            "Do not drop a code to keep the list short."
        )
    return THEME_SYSTEM + "\n\n" + count + "\n\n" + render_context(context)


def format_theme_codebook(codes: Sequence[Mapping[str, Any]]) -> str:
    if not codes:
        return "OPEN CODES\n(none)"
    lines = [
        "OPEN CODES",
        "Assign every CODE_LABEL below to exactly one theme. Copy the label text exactly.",
    ]
    for entry in codes:
        lines.append(f"CODE_LABEL: {entry.get('code', '')}")
        definition = clip(str(entry.get("definition") or ""), 500)
        if definition:
            lines.append(f"DEFINITION: {definition}")
        lines.append(f"COUNT: {entry.get('frequency', 0)}")
        examples = entry.get("examples") or []
        for example in examples[:2]:
            quote = clip(str(example.get("quote") or ""), 240)
            if not quote:
                continue
            lines.append(f"EXAMPLE: {quote}")
            if example.get("passage_id"):
                lines.append(f"PASSAGE_ID: {example.get('passage_id')}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_theme_messages(
    *,
    context: ResearchContext,
    codes: Sequence[Mapping[str, Any]],
    n_themes: int | None,
    feedback: str,
    prompt_cache: bool,
    chunk_index: int = 1,
    chunk_count: int = 1,
    max_themes: int = 12,
) -> list:
    parts = []
    if chunk_count > 1:
        parts.extend(
            [
                f"This is chunk {chunk_index} of {chunk_count}. "
                "Assign only the CODE_LABEL lines in this message. Other chunks are reconciled afterwards.",
                "",
            ]
        )
    parts.extend(
        [
            format_theme_codebook(codes),
            "",
            "Return one theme per group. Every CODE_LABEL appears under exactly one theme.",
        ]
    )
    if feedback:
        parts.extend(["", feedback])
    system = theme_system_text(context, n_themes, max_themes)
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content="\n".join(parts))]


JUDGE_SYSTEM = """You compare two open codes from a grounded-theory codebook.

Say whether they name the same kind of incident. Shared wording, a shared verb, \
or a shared topic is not enough. same is true only when a segment coded with one \
would still be coded with the other."""


def build_judge_messages(left: Mapping[str, Any], right: Mapping[str, Any]) -> list:
    """One pair. The caller keeps the pair identity; the model only answers same/not."""
    body = "\n".join(
        [
            f"CODE_A: {left.get('code', '')}",
            f"DEFINITION_A: {left.get('definition', '')}",
            f"CODE_B: {right.get('code', '')}",
            f"DEFINITION_B: {right.get('definition', '')}",
        ]
    )
    return [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=body)]


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
