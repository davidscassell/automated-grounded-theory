"""Structured outputs for passage coding and codebook sweeps.

These schemas are what the chat model must fill. They are not a codebook:
nothing here is a preset code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


CONFIDENCE_HELP = (
    "The model's own diagnostic from 0 to 1, not a probability and not a calibrated frequency. "
    "Anchors: 0.2 thin or contradictory evidence, "
    "0.5 plausible, 0.8 the quotes support it, 0.95 the quotes show it directly."
)
RATIONALE_HELP = "One to three sentences explaining this decision. Do not leave this empty."


class CodeProposal(_LLMModel):
    """One incident coded against the running codebook."""

    action: Literal["reuse", "refine", "new"] = Field(
        description=(
            "reuse an existing code, refine its label or definition, "
            "or mint a new code only when nothing listed fits"
        )
    )
    code: str = Field(
        description=(
            "Existing label for reuse or refine. For new, a short gerund "
            "(or the participant's own words when in_vivo is true)."
        )
    )
    nearest_code: str = Field(
        default="",
        description=(
            "Required before action new when any nearest code was listed: copy one of those labels. "
            "Empty when reusing, refining, or when the book is empty."
        ),
    )
    why_not_nearest: str = Field(
        default="",
        description=(
            "Required with nearest_code: the property this incident has that the nearest code does not cover. "
            "Empty unless action is new."
        ),
    )
    refined_label: str | None = Field(
        default=None,
        description="Replacement label when action is refine and the label should change. Otherwise null.",
    )
    definition: str = Field(
        default="",
        description="What incident this code captures, grounded in the quote. Empty only when reusing unchanged.",
    )
    quote: str = Field(
        description=(
            "Copy the participant's words from the target passage only. "
            "Do not quote the previous context, the following context, or another passage. "
            "Use ... only to skip words inside that passage. Do not paraphrase."
        )
    )
    in_vivo: bool = Field(
        default=False,
        description=(
            "True when the label is a vivid participant phrase copied from the passage "
            "(a metaphor, a repetition, or a complaint in their words). Consider that "
            "phrase before leaving this false. False when the label is a gerund you composed."
        ),
    )
    rationale: str = Field(default="", description=RATIONALE_HELP)
    code_id: str = Field(
        default="",
        description="code_id of an existing code for reuse or refine. Empty when creating.",
    )
    nearest_code_ids: list[str] = Field(
        default_factory=list,
        description="code_id values of the nearest codes you considered before creating. Empty when reusing.",
    )
    uncertainty_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Optional flags: ambiguous_span, insufficient_context, competing_code, unsupported_inference. "
            "Confidence stays a diagnostic and does not decide the code."
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=CONFIDENCE_HELP + " This score is for the reuse, refine, or create action.",
    )
    in_vivo_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=CONFIDENCE_HELP + " This score is for the in_vivo flag.",
    )
    in_vivo_rationale: str = Field(
        default="",
        description="One to three sentences on why the in_vivo flag is true or false.",
    )
    valence: Literal["positive", "negative", "neutral", "mixed"] = Field(
        default="neutral",
        description=(
            "Stance of this incident. positive and negative are opposites and must not share a code. "
            "mixed is both in one incident. neutral is neither."
        ),
    )


class PassageCoding(_LLMModel):
    """Open coding of one passage. One code per distinct incident."""

    passage_id: str = Field(default="", description="The passage_id from the prompt. Copy it exactly.")
    codes: list[CodeProposal] = Field(
        default_factory=list,
        description=(
            "One code per distinct incident in this passage only. "
            "Do not write a code that spans several passages. Empty when nothing is codable."
        ),
    )
    memo: str = Field(
        default="",
        description=(
            "Free-form analytic memo, or empty. Write one when something analytically interesting happens. "
            "A comparison, a property, and a question are useful and not required. About 20 to 400 words."
        ),
    )
    memo_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=CONFIDENCE_HELP + " Use 0.95 when the memo is empty.",
    )
    low_confidence: bool = Field(
        default=False,
        description="True when this passage should be read again by a human coder.",
    )


class MergeDecision(_LLMModel):
    action: Literal["accept"] = Field(default="accept", description="Accept this merge.")
    pair_id: str = Field(
        default="",
        description="pair_id copied from a hint. Empty when you found the duplicate yourself.",
    )
    source_id: str = Field(default="", description="code_id of the code to retire.")
    target_id: str = Field(default="", description="code_id of the code that remains.")
    canonical_id: str = Field(
        default="",
        description="code_id that should survive. Empty to let the pipeline choose the stable id.",
    )
    source: str = Field(description="Label to fold into the target. This label is retired.")
    target: str = Field(description="Label that remains. Prefer the clearer, more specific gerund.")
    rationale: str = Field(default="", description=RATIONALE_HELP)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    same_action: bool | None = Field(
        default=None,
        description="True when both codes name the same action (for example criticizing, or praising).",
    )
    same_object: bool | None = Field(
        default=None,
        description="True when both codes are about the same object (for example rollout training, not promotion).",
    )
    same_stance: bool | None = Field(
        default=None,
        description=(
            "True when both codes take the same stance. Record it when the stances differ. "
            "Stance is context, not a merge veto."
        ),
    )
    source_valence: str = Field(default="", description="positive, negative, neutral, or mixed for the source code.")
    target_valence: str = Field(default="", description="positive, negative, neutral, or mixed for the target code.")
    rewritten_label: str = Field(
        default="",
        description="Label for the survivor, written from the member quotes. Empty to keep the survivor label.",
    )
    rewritten_definition: str = Field(
        default="",
        description="Definition for the survivor, written from the member quotes. Empty to keep the survivor definition.",
    )
    quote_fits: list["QuoteFit"] = Field(
        default_factory=list,
        description="One row per member quote. fits false means the evidence is mixed and the pair must not be aliased.",
    )
    comparison_op: str = Field(
        default="",
        description=(
            "alias_duplicate only when definition and evidence name one concept. "
            "Otherwise keep_distinct, link_to_category, record_dimension, or defer."
        ),
    )
    contrast_checked: bool = Field(
        default=False,
        description="True only after comparing both definitions and their quotes for a different object or aspect.",
    )
    material_contrast: bool = Field(
        default=False,
        description="True when that contrast is a real difference. Do not alias when this is true.",
    )
    evidence_same: bool | None = Field(
        default=None,
        description="True when the quotes support one concept. False when the evidence is mixed.",
    )


class MergeRejection(_LLMModel):
    """A candidate pair that should stay as two codes."""

    action: Literal["reject"] = Field(default="reject", description="Reject this merge.")
    pair_id: str = Field(default="", description="pair_id from the candidate list. Do not invent one.")
    source_id: str = Field(default="", description="code_id of one hinted code.")
    target_id: str = Field(default="", description="code_id of the other hinted code.")
    source: str = Field(description="One label from a candidate pair. Copy it exactly.")
    target: str = Field(description="The other label from that candidate pair. Copy it exactly.")
    rationale: str = Field(default="", description=RATIONALE_HELP)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    same_action: bool | None = None
    same_object: bool | None = None
    same_stance: bool | None = None
    source_valence: str = ""
    target_valence: str = ""


class DefinitionUpdate(_LLMModel):
    code_id: str = Field(default="", description="code_id of the code whose definition changes.")
    code: str = Field(default="", description="Label of that code. Copy it exactly.")
    definition: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    rationale: str = Field(default="", description=RATIONALE_HELP)


class MergeCheck(_LLMModel):
    """Second look at one proposed merge. This confirms or rejects the proposal."""

    pair_id: str = Field(default="", description="pair_id from the proposal. Empty if the proposal had none.")
    source_id: str = Field(default="", description="code_id of one code. Copy it from the proposal.")
    target_id: str = Field(default="", description="code_id of the other code. Copy it from the proposal.")
    source: str = Field(default="", description="Label of one code. Copy it exactly.")
    target: str = Field(default="", description="Label of the other code. Copy it exactly.")
    action: Literal["accept", "reject"] = Field(
        description=(
            "accept only when both codes name the same incident. "
            "Stance is context, not a merge veto, and it is not by itself a reason to alias."
        )
    )
    rationale: str = Field(default="", description=RATIONALE_HELP)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    same_action: bool | None = None
    same_object: bool | None = None
    same_stance: bool | None = None


class MergeCheckBatch(_LLMModel):
    """One verification for every proposed merge in the request."""

    checks: list[MergeCheck] = Field(default_factory=list)


class QuoteFit(_LLMModel):
    """Whether one member quote still belongs under the rewritten code."""

    code_id: str = Field(default="", description="code_id of the code this quote is currently under.")
    quote: str = Field(default="", description="The quote, copied exactly.")
    fits: bool = Field(default=True, description="False when this quote is a different incident from the rewritten code.")
    split_to: str = Field(
        default="",
        description="Label to move the quote back to when fits is false. Empty to restore its previous code.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    rationale: str = Field(default="", description=RATIONALE_HELP)


class QuoteFitBatch(_LLMModel):
    checks: list[QuoteFit] = Field(default_factory=list)


class LeakCheck(_LLMModel):
    """Whether a label mostly restates the research context."""

    code_id: str = ""
    label: str = ""
    leaks: bool = True
    reworded_label: str = Field(
        default="",
        description="A participant-grounded label when leaks is true. Empty when the label is already grounded.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    rationale: str = Field(default="", description=RATIONALE_HELP)


class LeakCheckBatch(_LLMModel):
    checks: list[LeakCheck] = Field(default_factory=list)


class DefinitionRewrite(_LLMModel):
    """A survivor label and definition rewritten from the quotes now gathered under that code."""

    code_id: str = Field(default="", description="code_id of the surviving code. Copy it from the input.")
    code: str = Field(default="", description="Current label. Copy it, or replace it with rewritten_label.")
    rewritten_label: str = Field(default="", description="Label written from the member quotes. Empty to keep code.")
    definition: str = Field(
        default="",
        description="What incident the surviving code captures, written from the quotes.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    rationale: str = Field(default="", description=RATIONALE_HELP)


class DefinitionRewriteBatch(_LLMModel):
    rewrites: list[DefinitionRewrite] = Field(default_factory=list)


class RelationHint(_LLMModel):
    """A note for a later round. Not axial coding."""

    source: str
    target: str
    relation: str = "related"
    note: str = ""


class BatchCoding(_LLMModel):
    """Open coding for every passage in one request."""

    passages: list[PassageCoding] = Field(default_factory=list)


class ThemeProposal(_LLMModel):
    """One reporting theme. It groups open codes; it is not a new code."""

    name: str = Field(description="Short theme name. Not an open-code label.")
    definition: str = Field(description="What kind of incident this theme gathers.")
    properties: list[str] = Field(
        default_factory=list,
        description="A few dimensions that vary inside the theme. Empty if none stand out.",
    )
    theme_id: str = Field(default="", description="Leave empty. The pipeline assigns a stable theme id.")
    code_ids: list[str] = Field(
        default_factory=list,
        description="code_id of each open code in this theme. Copy ids from the input.",
    )
    codes: list[str] = Field(
        default_factory=list,
        description=(
            "Open-code labels in this theme. Copy each label exactly. "
            "A code may belong to more than one theme. The first theme that lists it is the primary theme."
        ),
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    rationale: str = Field(default="", description=RATIONALE_HELP)


class ThemeTension(_LLMModel):
    """A pull between themes, or a negative case, grounded in the data."""

    themes: list[str] = Field(
        default_factory=list,
        description="Theme names this note is about. Use the names you just proposed.",
    )
    note: str = Field(default="", description="How these themes pull against each other.")
    quote: str = Field(
        default="",
        description="Optional exact copy of an EXAMPLE quote from the prompt. Empty if you have none.",
    )
    passage_id: str = Field(
        default="",
        description="passage_id of that example when the input shows one. Otherwise empty.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    rationale: str = Field(default="", description=RATIONALE_HELP)


class ThemeNegative(_LLMModel):
    """A code or passage that sits awkwardly inside a theme."""

    theme: str = Field(default="", description="Theme name this case strains.")
    note: str = Field(default="")
    code: str = Field(default="", description="Open-code label, or empty.")
    quote: str = Field(default="", description="Exact EXAMPLE quote, or empty.")
    passage_id: str = Field(default="")


class ThemeGroup(_LLMModel):
    """One theme after a merge pass. Members are the theme names being folded in."""

    name: str
    definition: str = ""
    member_ids: list[str] = Field(
        default_factory=list,
        description="theme_id of each theme in this group. Copy ids from the input.",
    )
    canonical_id: str = Field(
        default="",
        description="theme_id that should survive when members are duplicates. Empty to keep the pipeline's survivor.",
    )
    members: list[str] = Field(default_factory=list)
    rationale: str = Field(
        default="",
        description="Why these themes are the same incident, when members lists more than one theme. Empty for a theme that stands alone.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=CONFIDENCE_HELP + " For a theme that stands alone, use 0.95.",
    )


class ThemeMergePlan(_LLMModel):
    """Rename and combine themes. Every input theme name appears in exactly one group."""

    themes: list[ThemeGroup] = Field(default_factory=list)


class ThemeCoding(_LLMModel):
    """A reporting grouping of one codebook. Does not invent open codes and does not reread passages."""

    themes: list[ThemeProposal] = Field(default_factory=list)
    tensions: list[ThemeTension] = Field(default_factory=list)
    negative_cases: list[ThemeNegative] = Field(default_factory=list)


class InBatchCode(_LLMModel):
    """One new code proposed in this round, not yet in the frozen codebook."""

    label: str
    definition: str = ""
    quotes: list[str] = Field(default_factory=list)
    valence: str = ""


class InBatchRequest(_LLMModel):
    """The new codes from one round, for a single comparison call."""

    note: str = ""
    codes: list[InBatchCode] = Field(default_factory=list)


class InBatchComparison(_LLMModel):
    """Decisions among the new codes of one round. Does not invent codes."""

    merges: list[MergeDecision] = Field(
        default_factory=list,
        description=(
            "Pairs among these newly proposed codes. "
            "alias_duplicate only when definition and evidence name one concept. "
            "Otherwise keep_distinct, link_to_category, or defer."
        ),
    )


class SameCodeJudgment(_LLMModel):
    """Whether two open codes name the same kind of incident."""

    same: bool = Field(description="True when both codes capture the same incident, not merely the same topic.")
    rationale: str = Field(default="", description="One short clause. Empty if obvious.")


class SweepCoding(_LLMModel):
    """Code-to-code comparison. May propose merges. Does not invent codes."""

    merges: list[MergeDecision] = Field(
        default_factory=list,
        description=(
            "Duplicates you found by comparing the codes, including hinted pairs you accept. "
            "Every hint must be here or in rejections. Each merge needs a rationale."
        ),
    )
    rejections: list[MergeRejection] = Field(
        default_factory=list,
        description="Every hinted pair you do not merge. One entry per rejected hint, with the reason. Do not reject a pair that was not hinted.",
    )
    definitions: list[DefinitionUpdate] = Field(default_factory=list)
    overloaded_codes: list[str] = Field(
        default_factory=list,
        description="Codes that seem to cover more than one kind of incident. Do not split them here.",
    )
    relation_hints: list[RelationHint] = Field(default_factory=list)
    integrative_memo: str = Field(
        default="",
        description="A few sentences on properties or comparisons. Empty if the sweep changed nothing analytic.",
    )
    reflexive_memo: str = ""


class CodeInput(_LLMModel):
    """One codebook row sent to the model. Ids are assigned by the pipeline.

    Examples live only in ``examples``. A separate ``example`` string would
    repeat the first quote.
    """

    code_id: str
    label: str
    definition: str = ""
    frequency: int = 0
    examples: list[dict] = Field(default_factory=list)
    valence: str = ""


class LabelRef(_LLMModel):
    """Compact codebook row: id and a short label, no definition or examples."""

    code_id: str
    label: str


class PassageInput(_LLMModel):
    """One passage sent to the model."""

    passage_id: str
    doc_id: str = ""
    doc_name: str = ""
    position: str = ""
    text: str = ""
    prev_text: str = ""
    next_text: str = ""
    metadata: dict = Field(default_factory=dict)
    nearest_code_ids: list[str] = Field(default_factory=list)
    question: str = ""
    nonsubstantive: bool = Field(
        default=False,
        description=(
            "True for uncertainty, refusal, n/a, or punctuation only. Code it from the question."
        ),
    )
    nonsubstantive_kind: str = Field(
        default="",
        description=(
            "uncertainty, refusal, non_answer, or no_substantive_text. "
            "Empty when the passage is substantive."
        ),
    )


class CodingRequest(_LLMModel):
    """Typed input for passage coding."""

    note: str = ""
    omitted_codes: int = 0
    codebook: list[CodeInput] = Field(default_factory=list)
    passages: list[PassageInput] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)


class HintInput(_LLMModel):
    """One optional similarity hint. The model may ignore it."""

    pair_id: str
    source_id: str = ""
    target_id: str = ""
    source: str
    target: str
    score: float = 0.0
    method: str = ""


class DiscoveryPair(_LLMModel):
    """Two labels a discovery pass thinks might be the same incident."""

    source: str = ""
    target: str = ""


class CandidateDiscovery(_LLMModel):
    """Pairs from one compact label-only pass. Not a merge decision."""

    pairs: list[DiscoveryPair] = Field(default_factory=list)


class CandidateDiscoveryRequest(_LLMModel):
    """Ids and short labels for the whole codebook. No definitions or examples."""

    note: str = ""
    labels: list[LabelRef] = Field(default_factory=list)


class SweepRequest(_LLMModel):
    """Typed input for a sweep chunk.

    ``codes`` and ``cross_chunk`` carry definitions and examples. ``cross_chunk``
    is a bounded neighborhood. ``label_index``, when present, is a bounded set
    of extra candidate labels, not a copy of the whole book. ``audit`` is
    ``full`` or ``incremental``.
    """

    chunk_index: int = 1
    chunk_count: int = 1
    note: str = ""
    audit: str = ""
    codes: list[CodeInput] = Field(default_factory=list)
    cross_chunk: list[CodeInput] = Field(default_factory=list)
    label_index: list[LabelRef] = Field(default_factory=list)
    hints: list[HintInput] = Field(default_factory=list)


class ThemeRequest(_LLMModel):
    """Typed input for the theme report. Does not reread passages."""

    chunk_index: int = 1
    chunk_count: int = 1
    note: str = ""
    codes: list[CodeInput] = Field(default_factory=list)
    feedback: str = ""


class ThemeInput(_LLMModel):
    """One theme sent back for duplicate review or leftover assignment."""

    theme_id: str
    name: str
    definition: str = ""
    code_ids: list[str] = Field(default_factory=list)
    codes: list[str] = Field(default_factory=list)


class HintGroupInput(_LLMModel):
    """Themes that look similar. The model may reject the group."""

    hint_id: str
    theme_ids: list[str] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)


class ThemeReviewRequest(_LLMModel):
    """Typed input for duplicate-theme decisions."""

    note: str = ""
    themes: list[ThemeInput] = Field(default_factory=list)
    hints: list[HintGroupInput] = Field(default_factory=list)


class LeftoverRequest(_LLMModel):
    """Typed input for codes the first assignment missed."""

    note: str = ""
    themes: list[ThemeInput] = Field(default_factory=list)
    codes: list[CodeInput] = Field(default_factory=list)


class JudgeRequest(_LLMModel):
    """Typed input for a same-incident judgment."""

    left: CodeInput
    right: CodeInput


class MergePairInput(_LLMModel):
    """One proposed merge sent for a second look."""

    pair_id: str = ""
    source_id: str = ""
    target_id: str = ""
    source: str
    target: str
    source_definition: str = ""
    target_definition: str = ""
    source_valence: str = ""
    target_valence: str = ""
    source_quotes: list[str] = Field(default_factory=list)
    target_quotes: list[str] = Field(default_factory=list)


class MergeCheckRequest(_LLMModel):
    """Typed input for the merge verification step."""

    note: str = ""
    pairs: list[MergePairInput] = Field(default_factory=list)


class RewriteCodeInput(_LLMModel):
    """A surviving code whose definition should be rewritten from its quotes."""

    code_id: str
    label: str
    definition: str = ""
    retired_labels: list[str] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)


class RewriteRequest(_LLMModel):
    """Typed input for post-merge definition rewrites."""

    note: str = ""
    codes: list[RewriteCodeInput] = Field(default_factory=list)


class CodeMatch(_LLMModel):
    """One LLM code judged against the human codebook."""

    llm_code: str = Field(description="An LLM code label, copied exactly.")
    human_code: str = Field(
        default="",
        description="The human code that names the same incident. Empty when the LLM code is novel.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)
    rationale: str = Field(default="", description=RATIONALE_HELP)
    theme: str = Field(default="", description="Human theme name when the human code sits under one. Empty otherwise.")


class CodeMatchBatch(_LLMModel):
    """A mapping from LLM codes to human codes."""

    matches: list[CodeMatch] = Field(default_factory=list)


class FocusedCodeDecision(_LLMModel):
    """One code the focused reread keeps, adds, or proposes to revise or drop."""

    action: Literal["keep", "add", "revise", "drop"] = Field(
        default="keep",
        description=(
            "keep an initial code, add a code the passage supports, or propose revise or drop. "
            "revise and drop are proposals for the researcher. They are not applied without review."
        ),
    )
    code: str = Field(default="", description="Label. Empty only when you cannot name the incident.")
    code_id: str = Field(
        default="",
        description="code_id when the code is already in the raised set. Empty when the label is new.",
    )
    definition: str = Field(default="", description="What incident this code captures. Empty when keeping unchanged.")
    quote: str = Field(
        default="",
        description="Participant words from the target passage. Do not paraphrase. Do not quote another passage.",
    )
    rationale: str = Field(default="", description=RATIONALE_HELP)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=CONFIDENCE_HELP)


class FocusedPropertyNote(_LLMModel):
    """An analytic property of a category. The citation is the evidence, not a description of the passage."""

    category_id: str = Field(default="", description="code_id of the category this property develops.")
    name: str = Field(
        default="",
        description="Analytic property: what varies inside the category. Not a descriptive summary of the passage.",
    )
    text: str = Field(default="", description="What varies, in a sentence grounded in the quote. Not a paraphrase of the whole passage.")
    document_id: str = Field(default="", description="Document id of the passage that contains the quote.")
    passage_id: str = Field(default="", description="passage_id that contains the quote.")
    quote: str = Field(default="", description="Words copied from that passage. Empty citations are kept and flagged.")


class FocusedDimensionNote(_LLMModel):
    """One pole of an analytic dimension. Cite the passage that sits at that pole."""

    category_id: str = Field(default="", description="code_id of the category.")
    name: str = Field(
        default="",
        description="Analytic dimension: the range the category varies along. Not a description of the passage and not a relationship between categories.",
    )
    pole: str = Field(default="", description="The end of the range this passage sits at, including an opposite pole when the passage supports one.")
    text: str = Field(default="", description="How this pole shows up in the quote. Not a descriptive summary.")
    document_id: str = Field(default="", description="Document id of the passage that contains the quote.")
    passage_id: str = Field(default="", description="passage_id that contains the quote.")
    quote: str = Field(default="", description="Words copied from that passage.")


class FocusedPassageRecode(_LLMModel):
    """Focused reread of one passage against the raised codes."""

    passage_id: str = Field(default="", description="passage_id from the prompt. Copy it exactly.")
    codes: list[FocusedCodeDecision] = Field(
        default_factory=list,
        description="Codes this passage still supports. Omit an initial code only when the passage does not support it.",
    )
    properties: list[FocusedPropertyNote] = Field(
        default_factory=list,
        description="Analytic properties of the raised categories, each with document_id, passage_id, and a quote. Do not substitute a descriptive summary.",
    )
    dimensions: list[FocusedDimensionNote] = Field(
        default_factory=list,
        description="Analytic dimensions and the pole this passage occupies, each with document_id, passage_id, and a quote.",
    )
    memo: str = Field(
        default="",
        description="Analytic note, including a competing interpretation when another raised code or an opposite pole also fits. Not a saturation judgment and not a descriptive summary.",
    )


class FocusedRecodeBatch(_LLMModel):
    """One focused-coding request. One entry per passage in the request."""

    passages: list[FocusedPassageRecode] = Field(default_factory=list)


MergeDecision.model_rebuild()
