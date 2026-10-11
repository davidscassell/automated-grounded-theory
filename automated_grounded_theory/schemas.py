"""Structured outputs for passage coding and codebook sweeps.

These schemas are what the chat model must fill. They are not a codebook:
nothing here is a preset code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


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
        description="Exact contiguous substring copied from the passage. Not from the neighboring context."
    )
    in_vivo: bool = Field(
        default=False,
        description=(
            "True when the label is a vivid participant phrase copied from the passage "
            "(a metaphor, a repetition, or a complaint in their words). Consider that "
            "phrase before leaving this false. False when the label is a gerund you composed."
        ),
    )
    rationale: str = Field(
        default="",
        description="At most 12 words on why this code fits. Not a summary of the passage.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="0 to 1. Lower this when the incident is ambiguous or the passage is thin.",
    )


class PassageCoding(_LLMModel):
    """Open coding of one passage. One code per distinct incident."""

    passage_id: str = Field(default="", description="The passage_id from the prompt. Copy it exactly.")
    codes: list[CodeProposal] = Field(
        default_factory=list,
        description="One code per distinct incident. Empty when nothing is codable.",
    )
    memo: str = Field(
        default="",
        description=(
            "Empty, or about 50 to 120 words with a comparison to a named code, "
            "a property or dimension, and one emerging question. Do not restate the passage."
        ),
    )
    low_confidence: bool = Field(
        default=False,
        description="True when this passage should be read again by a human coder.",
    )


class MergeDecision(_LLMModel):
    pair_id: str = Field(default="", description="pair_id from the candidate list. Empty only for an extra merge.")
    source: str = Field(description="Code to fold into the target. This label is retired.")
    target: str = Field(description="Code that remains. Prefer the clearer, more specific gerund.")
    rationale: str = ""


class MergeRejection(_LLMModel):
    """A candidate pair that should stay as two codes."""

    pair_id: str = Field(default="", description="pair_id from the candidate list. Do not invent one.")
    source: str = Field(description="One label from a candidate pair. Copy it exactly.")
    target: str = Field(description="The other label from that candidate pair. Copy it exactly.")
    rationale: str = Field(default="", description="Why these are different incidents.")


class DefinitionUpdate(_LLMModel):
    code: str
    definition: str


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
    """One focused-coding theme. It groups open codes; it is not a new code."""

    name: str = Field(description="Short theme name. Not an open-code label.")
    definition: str = Field(description="What kind of incident this theme gathers.")
    properties: list[str] = Field(
        default_factory=list,
        description="A few dimensions that vary inside the theme. Empty if none stand out.",
    )
    codes: list[str] = Field(
        default_factory=list,
        description="Open-code labels in this theme. Copy each CODE_LABEL exactly. Every code belongs to one theme.",
    )


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
        description="passage_id of that example when the prompt shows one. Otherwise empty.",
    )


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
    members: list[str] = Field(default_factory=list)


class ThemeMergePlan(_LLMModel):
    """Rename and combine themes. Every input theme name appears in exactly one group."""

    themes: list[ThemeGroup] = Field(default_factory=list)


class ThemeCoding(_LLMModel):
    """Focused coding of one codebook. Does not invent open codes."""

    themes: list[ThemeProposal] = Field(default_factory=list)
    tensions: list[ThemeTension] = Field(default_factory=list)
    negative_cases: list[ThemeNegative] = Field(default_factory=list)


class SameCodeJudgment(_LLMModel):
    """Whether two open codes name the same kind of incident."""

    same: bool = Field(description="True when both codes capture the same incident, not merely the same topic.")
    rationale: str = Field(default="", description="One short clause. Empty if obvious.")


class SweepCoding(_LLMModel):
    """Code-to-code comparison. May propose merges. Does not invent codes."""

    merges: list[MergeDecision] = Field(
        default_factory=list,
        description=(
            "Candidate pairs you accept, plus any other pair that names the same incident. "
            "Every candidate must be here or in rejections."
        ),
    )
    rejections: list[MergeRejection] = Field(
        default_factory=list,
        description="Every candidate pair you do not merge. One entry per rejected candidate, with the reason.",
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
