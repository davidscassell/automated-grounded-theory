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
            "or mint a new code because nothing in the codebook fits"
        )
    )
    code: str = Field(
        description=(
            "For reuse or refine, the existing code label. For new, a short "
            "gerund phrase close to this incident (or the participant's own "
            "words when in_vivo is true)."
        )
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
        description="True only when the label is the participant's own phrase and that phrase is in the passage.",
    )
    rationale: str = Field(
        default="",
        description="Why this code fits this incident, including how it compares with the codebook.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="0 to 1. Lower this when the incident is ambiguous or the passage is thin.",
    )


class PassageCoding(_LLMModel):
    """Open coding of a single passage."""

    what_is_happening: str = Field(
        default="",
        description="One sentence: what is actually happening in this passage.",
    )
    codes: list[CodeProposal] = Field(default_factory=list)
    memo: str = Field(
        default="",
        description="Brief analytic memo: comparisons, properties, questions. Not a summary of the passage.",
    )
    low_confidence: bool = Field(
        default=False,
        description="True when this passage should be read again by a human coder.",
    )


class MergeDecision(_LLMModel):
    source: str = Field(description="Code to fold into the target. This label is retired.")
    target: str = Field(description="Code that remains. Prefer the clearer, more specific gerund.")
    rationale: str = ""


class DefinitionUpdate(_LLMModel):
    code: str
    definition: str


class RelationHint(_LLMModel):
    """A note for a later round. Not axial coding."""

    source: str
    target: str
    relation: str = "related"
    note: str = ""


class SweepCoding(_LLMModel):
    """Code-to-code comparison. Confirms merges; does not invent codes."""

    merges: list[MergeDecision] = Field(default_factory=list)
    definitions: list[DefinitionUpdate] = Field(default_factory=list)
    overloaded_codes: list[str] = Field(
        default_factory=list,
        description="Codes that seem to cover more than one kind of incident. Do not split them here.",
    )
    relation_hints: list[RelationHint] = Field(default_factory=list)
    integrative_memo: str = ""
    reflexive_memo: str = ""
