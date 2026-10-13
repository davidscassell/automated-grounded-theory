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
from automated_grounded_theory.schemas import (
    CodeInput,
    CodingRequest,
    CandidateDiscoveryRequest,
    HintGroupInput,
    HintInput,
    JudgeRequest,
    LeftoverRequest,
    MergeCheckRequest,
    MergePairInput,
    PassageInput,
    RewriteCodeInput,
    RewriteRequest,
    LabelRef,
    SweepRequest,
    ThemeInput,
    ThemeRequest,
    ThemeReviewRequest,
)
from automated_grounded_theory.retrieval import select_prompt_examples
from automated_grounded_theory.validation import code_key

STANCE_POLICY = (
    "Stance is context, not a merge veto, and it is not by itself a reason to alias."
)

CODING_SYSTEM = """You are doing first-round open coding in the grounded theory tradition.

This is initial coding only, incident by incident. You are not writing themes, \
axial coding, selective coding, or a theory.

Ask of each passage: what is this data a study of, what is actually happening, \
and what category does this incident indicate?

Code with short gerunds: simple, active, close to the data, usually two to six words. \
Prefer one action and one object. Before you finish a code, ask whether it hides a \
second idea (who, why, or a consequence). If it does, return another code. Return \
one code per distinct incident. Spans may overlap. A passage may carry zero codes \
or several. A passage that both praises and criticizes needs a separate code for \
each incident. One new label names a single concept. Do not join two codes with a \
semicolon. Do not put a comma list of concepts, or several concepts joined by \
"and", into one new label. Return each concept as its own code.

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
- "I like my team" has no vivid phrase → code "expressing liking for the team", in_vivo false.

Constant comparison: the user message lists the nearest existing codes, retrieved by \
similarity, plus other frequent codes. Prefer reuse. Prefer refine when this incident \
shows the label or definition should change. Create a new code only when none of the \
nearest codes fits. Before action create, set nearest_code to one of those labels, \
nearest_code_ids to their code_id values, and why_not_nearest to the property this \
incident has that the nearest code does not. \
Do not mint a near-duplicate. nearest_code may be empty when no code exists yet \
or no listed code is close. Set valence to positive, negative, neutral, or mixed. \
Valence is context. It does not stop two codes from informing one category, and it \
is not by itself a reason to merge them. Set uncertainty_flags when the span is \
ambiguous, the context is thin, two codes compete, or the inference is unsupported. \
Confidence is a diagnostic, not a probability. Set confidence \
from 0 to 1 on the action, on in_vivo, and on the memo. Anchors: 0.2 thin or \
contradictory evidence, 0.5 plausible, 0.8 the quotes support it, 0.95 the quotes \
show it directly. Write a rationale of one to three sentences for the action, for \
in_vivo, and for the memo.

Definitions name a property that could apply to another incident. Do not put respondent \
attributes (name, role, department, tenure, site) or the question wording into a label \
or a definition.

The research context only sensitizes what you notice. Codes must come from the \
participant's language in the passage. Do not turn the research question, the \
population, the setting, or a sensitizing concept into a code unless those exact \
words are the participant's. A label that could have been written from the context \
alone, without this passage, is a leak. Do not write it. Do not quote or cite the \
context in a memo.

Quotes should be copied from the participant's words. Prefer the passage. If the \
incident continues in the previous or next context, copy from there instead of \
paraphrasing. Use ... only for words you skip. Do not invent wording.

Do not use generic labels such as "experience", "discussion", "theme", or "response".

If a passage has no codable incident, return an empty code list and an empty memo.

A memo is not a restatement. Leave it empty unless you can write about 50 to 120 \
words that include all three of these: a comparison with a named code (or a statement \
that the book had no close code), a property or dimension this incident varies, and \
one emerging question the next passage could answer. Do not write a sentence that \
only says no property was recorded.

If several passages are in the user message, return one passages item for every \
passage_id. A missing passage_id is a failed slot, not an empty coding."""

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
merge codes that merely share a verb, a topic, a platform name, or a sensitizing \
concept. alias_duplicate only when the definitions name the same concept and the \
quotes show that same concept. Set same_action and same_object true only then. \
Before alias_duplicate, set contrast_checked true and material_contrast false: \
you compared both definitions and their quotes and found no different object or \
aspect. If the objects or aspects differ, set comparison_op to keep_distinct. \
If the quotes disagree, set evidence_same false and comparison_op to defer. \
If they belong in one category but are not one code, set comparison_op to \
link_to_category. Set same_action, same_object, and same_stance on every merge \
and rejection. Keep the clearer label as the target and name the other code as \
the source. Record source_valence and target_valence. """ + STANCE_POLICY + """ Do not treat a chain of \
pairs as one merge. On each accept, set rewritten_label and rewritten_definition \
from the member quotes, and set quote_fits for each example quote (fits false \
means the evidence is mixed, so do not alias). Set confidence from 0 to 1. \
Anchors: 0.2 thin or contradictory, 0.5 \
plausible, 0.8 the quotes support it, 0.95 the quotes show it directly. Write a \
rationale of one to three sentences on every merge, rejection, and definition.

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
larger codebook, definitions and examples are only on the codes in this chunk and on \
cross_chunk, a bounded neighborhood of similar codes from other chunks. A separate \
candidate-discovery step chooses that neighborhood. It is not a full label index \
repeated on every chunk. Merge a code in the chunk with a cross_chunk code when \
they name the same incident. Use the labels exactly as written."""


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
        examples = select_prompt_examples(entry, limit=1)
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


def stable_code_id(entry: Mapping[str, Any]) -> str:
    """The pipeline id for a codebook row. Never invent a different id for the same label."""
    existing = str(entry.get("code_id") or "").strip()
    if existing:
        return existing
    from automated_grounded_theory.codebook import _code_id

    key = code_key(str(entry.get("key") or entry.get("code") or entry.get("label") or ""))
    return _code_id(key) if key else ""


def code_input(entry: Mapping[str, Any]) -> CodeInput:
    examples = []
    for item in select_prompt_examples(entry, limit=2):
        quote = clip(str(item.get("quote") or ""), 240)
        if not quote:
            continue
        examples.append({"quote": quote, "passage_id": str(item.get("passage_id") or "")})
    return CodeInput(
        code_id=stable_code_id(entry),
        label=str(entry.get("code") or entry.get("label") or ""),
        definition=clip(str(entry.get("definition") or ""), 500),
        frequency=int(entry.get("frequency") or 0),
        examples=examples,
        valence=str(entry.get("valence") or ""),
    )


def label_ref(entry: Mapping[str, Any]) -> LabelRef:
    """Id plus a short label. No definition and no examples."""
    return LabelRef(
        code_id=stable_code_id(entry),
        label=clip(str(entry.get("code") or entry.get("label") or ""), 80),
    )


def dump_request(payload: Any) -> str:
    """Compact JSON for a model request. No indent and no extra spaces."""
    if hasattr(payload, "model_dump_json"):
        return payload.model_dump_json()
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def extract_payload(text: str) -> dict:
    """The typed JSON object in a user message, or an empty dict."""
    decoder = json.JSONDecoder()
    start = 0
    while True:
        index = text.find("{", start)
        if index < 0:
            return {}
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            start = index + 1
            continue
        if isinstance(obj, dict) and any(
            key in obj for key in ("passages", "codebook", "codes", "hints", "themes", "left", "pairs")
        ):
            return obj
        start = index + 1


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
    del neighbor_chars
    by_label = {code_key(str(entry.get("code") or "")): stable_code_id(entry) for entry in codebook_view}
    passage_rows = []
    for index, passage in enumerate(passages):
        labels = list(nearest[index]) if nearest and index < len(nearest) else []
        passage_rows.append(
            PassageInput(
                passage_id=str(passage.get("passage_id") or ""),
                doc_id=str(passage.get("doc_id") or ""),
                doc_name=str(passage.get("doc_name") or ""),
                position=str(passage.get("position") or ""),
                text=str(passage.get("text") or ""),
                prev_text=clip(str(passage.get("prev_text") or ""), 400),
                next_text=clip(str(passage.get("next_text") or ""), 400),
                metadata=dict(passage.get("metadata") or {}),
                nearest_code_ids=[by_label.get(code_key(label), "") for label in labels if code_key(label)],
                question=str(passage.get("question") or ""),
            )
        )
    notes = [note for note in (feedback or {}).values() if note]
    request = CodingRequest(
        note="" if codebook_view else "The codebook is empty. Codes you create must emerge from this passage.",
        omitted_codes=omitted_codes,
        codebook=[code_input(entry) for entry in codebook_view],
        passages=passage_rows,
        feedback=notes,
    )
    return [
        _system_message(coding_system_text(context), prompt_cache=prompt_cache),
        HumanMessage(content=dump_request(request)),
    ]


def format_sweep_codebook(codes: Sequence[Mapping[str, Any]]) -> str:
    if not codes:
        return "CODEBOOK\n(empty)"
    lines = ["CODEBOOK"]
    for entry in codes:
        lines.append(f"CODE_LABEL: {entry.get('code', '')}")
        lines.append(f"DEFINITION: {clip(str(entry.get('definition') or ''), 500)}")
        lines.append(f"COUNT: {entry.get('frequency', 0)}")
        shown = [
            clip(str(item.get("quote", "")), 180)
            for item in select_prompt_examples(entry, limit=3)
            if item.get("quote")
        ]
        if shown:
            lines.append("EXAMPLES: " + " || ".join(shown))
        lines.append("")
    return "\n".join(lines).rstrip()


def build_candidate_discovery_messages(
    *,
    context: ResearchContext,
    codes: Sequence[Mapping[str, Any]],
    prompt_cache: bool,
) -> list:
    """One label-only request for the whole book. No definitions and no examples."""
    request = CandidateDiscoveryRequest(
        note=(
            "Candidate discovery for the whole codebook. These are ids and short labels only. "
            "Name pairs that might be the same incident. Do not merge in this step."
        ),
        labels=[label_ref(entry) for entry in codes],
    )
    system = (
        "You are listing possible duplicate codes from ids and short labels. "
        "You do not have definitions or examples here. Return pairs that might name the same incident. "
        "This step only discovers candidates. It does not merge them.\n\n" + render_context(context)
    )
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(request))]


def build_sweep_messages(
    *,
    context: ResearchContext,
    codes: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    prompt_cache: bool,
    all_codes: Sequence[Mapping[str, Any]] | None = None,
    cross_chunk: Sequence[Mapping[str, Any]] | None = None,
    chunk_index: int = 1,
    chunk_count: int = 1,
    consolidation: bool = False,
    audit: str = "full",
    full_label_index: bool = False,
    neighborhood: Sequence[Mapping[str, Any]] | None = None,
    neighborhood_limit: int = 16,
) -> list:
    by_label = {code_key(str(entry.get("code") or "")): stable_code_id(entry) for entry in list(codes) + list(all_codes or [])}
    hints = []
    for index, item in enumerate(candidates, start=1):
        pair_id = str(item.get("pair_id") or f"p{index:04d}")
        item["pair_id"] = pair_id
        source = str(item.get("source") or "")
        target = str(item.get("target") or "")
        hints.append(
            HintInput(
                pair_id=pair_id,
                source_id=str(item.get("source_id") or by_label.get(code_key(source), "")),
                target_id=str(item.get("target_id") or by_label.get(code_key(target), "")),
                source=source,
                target=target,
                score=round(float(item.get("score") or 0), 4),
                method=str(item.get("method") or ""),
            )
        )
    note = (
        "Merge only when the definitions name the same concept and the quotes show it: "
        "same action, same object, contrast checked, and no material contrast. "
        + STANCE_POLICY
        + " Record same_stance, but do not require it. "
        "Rewrite the survivor label and definition from the member quotes in this same response."
    )
    if consolidation:
        note += " This is the final consolidation pass over the full codebook, in sorted order."
    if not hints:
        note += " No similarity hints. Read every code and propose duplicates yourself."
    mode = "incremental" if str(audit or "") == "incremental" else "full"
    if mode == "incremental":
        note += (
            " Incremental review: these codes are new or their label, definition, or examples changed. "
            "Compare them with the other codes in this request."
        )
    elif all_codes and full_label_index:
        note += " Full codebook audit."
    elif chunk_count > 1:
        note += " Full codebook audit." if mode == "full" else ""
    if full_label_index and all_codes:
        note += (
            " Definitions and examples are on this chunk and on cross_chunk. "
            "The label index is ids and short labels for the whole codebook."
        )
        index_rows = list(all_codes)
    else:
        if chunk_count > 1:
            note += (
                " Definitions and examples are on this chunk and on cross_chunk, "
                "a bounded neighborhood of similar codes. Other candidates were chosen "
                "in a separate discovery step and are not repeated here."
            )
        cap = max(0, int(neighborhood_limit))
        index_rows = list(neighborhood or [])[:cap]
    request = SweepRequest(
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        note=note,
        audit=mode if all_codes or mode == "incremental" or chunk_count > 1 else "",
        codes=[code_input(entry) for entry in codes],
        cross_chunk=[code_input(entry) for entry in (cross_chunk or [])],
        label_index=[label_ref(entry) for entry in index_rows],
        hints=hints,
    )
    system = SWEEP_SYSTEM + "\n\n" + render_context(context)
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(request))]


def aggregate_sweep_chars(
    codes: Sequence[Mapping[str, Any]],
    *,
    context: ResearchContext,
    chunk_size: int = 20,
    cross_chunks: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    audit: str = "full",
    user_only: bool = False,
    full_label_index: bool = False,
    discover: bool = True,
) -> int:
    """Character length of the sweep messages for ``codes`` at ``chunk_size``.

    ``cross_chunks`` lines up with chunk order. A missing entry is empty.
    ``user_only`` drops the stable system prefix, which does not grow with the book.
    ``full_label_index`` repeats every label on every chunk (the 0.6.5 payload).
    Otherwise each chunk carries a bounded neighborhood, and one compact
    label-only discovery request is counted when the book spans several chunks.
    """
    rows = list(codes)
    size = max(1, int(chunk_size))
    chunks = [rows[start : start + size] for start in range(0, len(rows), size)] or [[]]
    total = 0
    if discover and not full_label_index and len(chunks) > 1:
        discovery = build_candidate_discovery_messages(context=context, codes=rows, prompt_cache=False)
        if user_only:
            total += len(message_text(discovery[-1]))
        else:
            total += sum(len(message_text(message)) for message in discovery)
    for index, chunk in enumerate(chunks, start=1):
        cross: Sequence[Mapping[str, Any]] = []
        if cross_chunks and index - 1 < len(cross_chunks):
            cross = cross_chunks[index - 1]
        chunk_ids = {stable_code_id(entry) for entry in chunk}
        show_index = any(stable_code_id(entry) not in chunk_ids for entry in rows)
        messages = build_sweep_messages(
            context=context,
            codes=chunk,
            candidates=[],
            prompt_cache=False,
            all_codes=rows if (full_label_index and show_index) or show_index else None,
            cross_chunk=cross,
            chunk_index=index,
            chunk_count=len(chunks),
            audit=audit,
            full_label_index=full_label_index and show_index,
        )
        if user_only:
            total += len(message_text(messages[-1]))
        else:
            total += sum(len(message_text(message)) for message in messages)
    return total


IN_BATCH_SYSTEM = """You are comparing codes that were just proposed in one round.

These codes are not in the frozen codebook yet. The passages were coded in \
parallel, so the proposals have not been compared with each other. Read every \
code. Alias two of them only when the definitions name the same concept and \
the quotes show that same concept. Set comparison_op to alias_duplicate only \
then, with same_action and same_object true, contrast_checked true, and \
material_contrast false. If the objects or aspects differ, set comparison_op \
to keep_distinct. If the quotes disagree, set evidence_same false and \
comparison_op to defer. If they belong in one category but are not one code, \
set comparison_op to link_to_category. """ + STANCE_POLICY + """ Do not invent codes. Do not merge a chain \
into one code. Set confidence from 0 to 1. Write a rationale of one to three \
sentences on every decision."""


def build_in_batch_messages(
    *,
    context: ResearchContext,
    codes: Sequence[Mapping[str, Any]],
    prompt_cache: bool,
) -> list:
    from automated_grounded_theory.schemas import InBatchCode, InBatchRequest

    request = InBatchRequest(
        note=(
            "These codes were proposed against the same frozen snapshot. "
            "Compare them with each other before they are committed. "
            "Leave a distinct concept as its own code."
        ),
        codes=[
            InBatchCode(
                label=str(entry.get("code") or ""),
                definition=str(entry.get("definition") or ""),
                quotes=[str(quote) for quote in (entry.get("quotes") or []) if str(quote).strip()][:3],
                valence=str(entry.get("valence") or ""),
            )
            for entry in codes
        ],
    )
    system = IN_BATCH_SYSTEM + "\n\n" + render_context(context)
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(request))]


THEME_SYSTEM = """You are doing focused coding, the round after open coding (Charmaz).

The open codes already exist. Group them into themes. A theme gathers codes \
about the same kind of incident. Name the theme, define the incident it gathers, \
and list a few properties (dimensions that vary inside it) when the codes show any.

Assign every CODE_LABEL to at least one theme. A code may belong to more than one \
theme when it genuinely informs both. Copy the label text exactly. Do not invent a \
code, do not rename a code, and do not drop one to keep the list short. \
max_themes is a presentation budget, not a reason to erase a membership.

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
        for example in select_prompt_examples(entry, limit=2):
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
    request = ThemeRequest(
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        note="Assign every code_id to exactly one theme. Copy labels exactly.",
        codes=[code_input(entry) for entry in codes],
        feedback=feedback,
    )
    system = theme_system_text(context, n_themes, max_themes)
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(request))]


JUDGE_SYSTEM = """You compare two open codes from a grounded-theory codebook.

Say whether they name the same kind of incident. Shared wording, a shared verb, \
or a shared topic is not enough. same is true only when a segment coded with one \
would still be coded with the other."""


def build_judge_messages(left: Mapping[str, Any], right: Mapping[str, Any]) -> list:
    """One pair. The caller keeps the pair identity; the model only answers same/not."""
    request = JudgeRequest(left=code_input(left), right=code_input(right))
    return [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=dump_request(request))]


def build_theme_review_messages(
    *,
    context: ResearchContext,
    themes: Sequence[Mapping[str, Any]],
    groups: Sequence[Sequence[int]],
    max_theme_share: float,
    prompt_cache: bool,
) -> list:
    """Alphabetical themes plus optional hint groups, as typed JSON."""
    rows = [
        ThemeInput(
            theme_id=f"t{index:04d}",
            name=str(theme.get("name") or ""),
            definition=str(theme.get("definition") or ""),
            codes=[str(label) for label in theme.get("codes") or []],
        )
        for index, theme in enumerate(themes)
    ]
    hints = []
    for number, indexes in enumerate(groups, start=1):
        if len(indexes) < 2:
            continue
        hints.append(
            HintGroupInput(
                hint_id=f"h{number:04d}",
                theme_ids=[rows[index].theme_id for index in indexes],
                names=[rows[index].name for index in indexes],
            )
        )
    ceiling = round(100 * max_theme_share)
    request = ThemeReviewRequest(
        note=(
            "Read every theme. Put duplicates in one group with a rationale and a confidence. "
            f"Do not build a group with more than {ceiling}% of the open codes. "
            + ("No similarity hints." if not hints else "Hints are optional.")
        ),
        themes=rows,
        hints=hints,
    )
    from automated_grounded_theory.prompts import render_context as _render

    system = (
        "You are comparing themes with themes after focused coding. "
        "Members are theme names, not open codes. "
        "Propose a group when two themes gather the same kind of incident, and say why in rationale. "
        "Set confidence on every group. "
        "Hints are optional. Embeddings are not required. "
        f"One theme must not hold more than {ceiling}% of the codes.\n\n" + _render(context)
    )
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(request))]


def build_leftover_messages(
    *,
    context: ResearchContext,
    themes: Sequence[Mapping[str, Any]],
    codes: Sequence[Mapping[str, Any]],
    prompt_cache: bool,
) -> list:
    request = LeftoverRequest(
        note="Assign every leftover code_id to one existing theme, or propose a theme that holds only those codes.",
        themes=[
            ThemeInput(
                theme_id=f"t{index:04d}",
                name=str(theme.get("name") or ""),
                definition=str(theme.get("definition") or ""),
                codes=[str(label) for label in theme.get("codes") or []],
            )
            for index, theme in enumerate(themes)
        ],
        codes=[code_input(entry) for entry in codes],
    )
    system = (
        "You are finishing focused coding. Assign only the leftover open codes. "
        "Do not invent a code_id.\n\n" + render_context(context)
    )
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(request))]


def build_merge_check_messages(
    *,
    context: ResearchContext,
    codebook: Mapping[str, Mapping[str, Any]],
    merges: Sequence[Mapping[str, Any]],
    prompt_cache: bool,
) -> list:
    """Second structured look at proposed merges. Stance is context, not a veto."""
    by_key = {code_key(str(entry.get("key") or entry.get("code") or "")): entry for entry in codebook.values()}
    pairs = []
    for merge in merges:
        source_key = code_key(str(merge.get("source") or ""))
        target_key = code_key(str(merge.get("target") or ""))
        source = by_key.get(source_key, {})
        target = by_key.get(target_key, {})
        pairs.append(
            MergePairInput(
                pair_id=str(merge.get("pair_id") or ""),
                source_id=str(source.get("code_id") or merge.get("source_id") or ""),
                target_id=str(target.get("code_id") or merge.get("target_id") or ""),
                source=str(source.get("code") or merge.get("source") or ""),
                target=str(target.get("code") or merge.get("target") or ""),
                source_definition=str(source.get("definition") or ""),
                target_definition=str(target.get("definition") or ""),
                source_valence=str(source.get("valence") or ""),
                target_valence=str(target.get("valence") or ""),
                source_quotes=[
                    str(item.get("quote") or "")
                    for item in select_prompt_examples(source, limit=3)
                    if item.get("quote")
                ],
                target_quotes=[
                    str(item.get("quote") or "")
                    for item in select_prompt_examples(target, limit=3)
                    if item.get("quote")
                ],
            )
        )
    request = MergeCheckRequest(
        note=(
            "Accept a pair only when both codes name the same incident. "
            + STANCE_POLICY
            + " Set confidence from 0 to 1."
        ),
        pairs=pairs,
    )
    system = (
        "You are verifying proposed code merges. "
        "Accept only when the quotes show the same incident. "
        + STANCE_POLICY
        + " Set confidence on every check.\n\n"
        + render_context(context)
    )
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(request))]


def build_rewrite_messages(
    *,
    context: ResearchContext,
    codes: Sequence[Mapping[str, Any]],
    prompt_cache: bool,
) -> list:
    """Ask for a survivor definition written from the quotes now under that code."""
    request = RewriteRequest(
        note=(
            "Each code survived a merge and kept its own label. "
            "Rewrite the definition from the quotes. Do not reuse a retired code's definition. "
            "Do not change the label. Set confidence from 0 to 1."
        ),
        codes=[
            RewriteCodeInput(
                code_id=str(entry.get("code_id") or ""),
                label=str(entry.get("code") or entry.get("label") or ""),
                definition=str(entry.get("definition") or ""),
                retired_labels=[str(label) for label in (entry.get("retired_labels") or [])],
                quotes=[str(quote) for quote in (entry.get("quotes") or [])],
            )
            for entry in codes
        ],
    )
    system = (
        "You are rewriting a code definition after a merge. "
        "The label stays. The definition must fit the quotes listed for that code, "
        "not the definition of a code that was retired.\n\n" + render_context(context)
    )
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(request))]


_LABEL_FEEDBACK = {
    "empty_code": (
        "The label was empty. Return one short code for this incident. Do not leave the code blank."
    ),
    "code_too_long": (
        "The label is too long. Return one short gerund, usually two to six words "
        "(at most 8 words and 80 characters, or 12 words and 120 characters when in_vivo is true)."
    ),
    "bundled_concepts": (
        "This new label bundles several concepts. Return a single concept. "
        "Put each other concept in its own code. Do not join codes with a semicolon, "
        "a comma list, or repeated 'and'."
    ),
}


# A retry that pastes every quote can dwarf the passage. The fitter may
# shorten this further; the cap stops the unbounded case at the source.
_FEEDBACK_CHAR_CAP = 2000


def validation_feedback(rejections: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "The previous coding was checked against the passage, its neighbors, and the document.",
        "Copy the participant's words. A closer quote is better.",
        "quote_not_verbatim means the quote was not found as written. The code is kept either way.",
    ]
    saw: set[str] = set()
    for item in rejections:
        item_reasons = [str(item.get("reason") or "rejected")]
        for extra in item.get("reasons") or []:
            name = str(extra)
            if name not in item_reasons:
                item_reasons.append(name)
        status = item.get("quote_status") or item_reasons[0]
        lines.append(
            f"- code {item.get('code', '')!r}: {item_reasons[0]} "
            f"(quote_status={status}; quote_not_verbatim means the quote was not found in the passage)"
        )
        for reason in item_reasons:
            saw.add(reason)
            detail = _LABEL_FEEDBACK.get(reason)
            if detail:
                lines.append(f"  {detail}")
    if "context_restatement" in saw:
        lines.append(
            "A rejected label mostly restates the research context. "
            "Reword it from the participant's words in the passage."
        )
    text = "\n".join(lines)
    if len(text) <= _FEEDBACK_CHAR_CAP:
        return text
    clipped = text[:_FEEDBACK_CHAR_CAP].rstrip()
    return clipped + "\nFurther validation notes were omitted so the retry stays within the token budget."


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


def build_leak_messages(*, context: ResearchContext, labels: Sequence[Mapping[str, Any]], prompt_cache: bool) -> list:
    """Cheap check: does a label restate the research context?"""
    payload = {
        "note": (
            "leaks is true when the label could have been written from the research context alone. "
            "If leaks is true, set reworded_label from the participant's words. "
            "Set confidence with anchors 0.2, 0.5, 0.8, 0.95 and a one to three sentence rationale."
        ),
        "labels": [
            {"code_id": str(item.get("code_id") or ""), "label": str(item.get("code") or item.get("label") or "")}
            for item in labels
        ],
    }
    system = (
        "You check code labels for context leakage. "
        "A leak restates the research question, population, setting, or a sensitizing concept. "
        "A grounded label uses the participant's words. "
        "Set confidence and a rationale on every check.\n\n" + render_context(context)
    )
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(payload))]


def build_quote_fit_messages(*, context: ResearchContext, codes: Sequence[Mapping[str, Any]], prompt_cache: bool) -> list:
    """Cheap check: which quotes do not fit a rewritten code."""
    payload = {
        "note": (
            "For each quote, fits is false when it is a different incident from the label and definition. "
            "Set split_to to the better label when you know one. "
            "Set confidence with anchors 0.2, 0.5, 0.8, 0.95 and a one to three sentence rationale."
        ),
        "codes": [
            {
                "code_id": str(item.get("code_id") or ""),
                "label": str(item.get("label") or ""),
                "quotes": list(item.get("quotes") or []),
            }
            for item in codes
        ],
    }
    system = (
        "You check whether each quote still fits its code after a merge. "
        "Split a quote out only when it names a different action, object, or stance. "
        "Set confidence and a rationale on every check.\n\n" + render_context(context)
    )
    return [_system_message(system, prompt_cache=prompt_cache), HumanMessage(content=dump_request(payload))]


MAPPING_SYSTEM = """You compare one coder's labels with another coder's labels.

The human codes are the reference. For each LLM code, copy the human code that \
names the same incident, or leave human_code empty when no human code does. Do \
not invent a human code. Do not map on a shared topic alone: the action, the \
object, and the stance have to match. Set confidence from 0 to 1. Anchors: 0.2 \
thin or contradictory, 0.5 plausible, 0.8 the definitions support it, 0.95 the \
labels show the same incident directly. Write a rationale of one to three \
sentences on every row. Set theme only when the human code list names one."""


def build_code_match_messages(
    *,
    llm_codes: Sequence[Mapping[str, Any]],
    human_codes: Sequence[Mapping[str, Any]],
    prompt_cache: bool = False,
) -> list:
    """Stable system prefix. The two code lists stay in the user JSON."""
    payload = {
        "note": (
            "Map each llm_code onto one human_code, or leave human_code empty when the LLM code is novel. "
            "Copy labels exactly."
        ),
        "human_codes": [
            {
                "code": str(item.get("code") or item.get("label") or ""),
                "definition": str(item.get("definition") or ""),
                "theme": str(item.get("theme") or ""),
            }
            for item in human_codes
        ],
        "llm_codes": [
            {
                "code": str(item.get("code") or item.get("label") or item or ""),
                "definition": str(item.get("definition") or "") if isinstance(item, Mapping) else "",
            }
            for item in llm_codes
        ],
    }
    return [
        _system_message(MAPPING_SYSTEM, prompt_cache=prompt_cache),
        HumanMessage(content=dump_request(payload)),
    ]


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
