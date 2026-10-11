"""Research context that sensitizes attention and cannot carry preset codes.

Glaser (1992) distinguishes emergence from forcing. A research question,
population, setting, or sensitizing concept may steer the coder toward what
is worth noticing. A list of codes may not.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Keys that would hand the coder a frame to force data into.
FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "codes",
        "code",
        "preset_codes",
        "predetermined_codes",
        "a_priori_codes",
        "apriori_codes",
        "a_priori_codebook",
        "codebook",
        "tags",
        "tag_list",
        "categories",
        "axial_codes",
        "selective_codes",
        "coding_paradigm",
        "paradigm",
    }
)


class PresetCodesError(ValueError):
    """Raised when context tries to supply codes instead of sensitizing attention."""


class ResearchContext(BaseModel):
    """Sensitizing context for one open-coding run.

    ``sensitizing_concepts`` are attention guides (Blumer, via Charmaz). They
    are not codes and are never inserted into the codebook.
    """

    model_config = ConfigDict(extra="forbid")

    research_question: str
    population: str | None = None
    setting: str | None = None
    sensitizing_concepts: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_preset_codes(cls, data: Any) -> Any:
        if isinstance(data, dict):
            found = sorted(FORBIDDEN_CONTEXT_KEYS & {str(key).lower() for key in data})
            if found:
                raise PresetCodesError(
                    "Context cannot include preset codes "
                    f"({', '.join(found)}). Open coding has to let codes emerge "
                    "from the passages (Glaser, 1992, emergence versus forcing). "
                    "Pass a research question and, if you need them, sensitizing "
                    "concepts only."
                )
        return data

    @model_validator(mode="after")
    def _question_present(self) -> ResearchContext:
        if not self.research_question or not self.research_question.strip():
            raise ValueError("research_question must be a non-empty string.")
        self.sensitizing_concepts = [item.strip() for item in self.sensitizing_concepts if item and item.strip()]
        return self


def _preset_codes_in(data: dict[str, Any]) -> list[str]:
    return sorted(FORBIDDEN_CONTEXT_KEYS & {str(key).lower() for key in data})


def parse_context(context: ResearchContext | dict[str, Any] | str) -> ResearchContext:
    """Accept a :class:`ResearchContext`, a mapping, or a question string."""
    if isinstance(context, ResearchContext):
        return context
    if isinstance(context, str):
        if not context.strip():
            raise ValueError("research_question must be a non-empty string.")
        return ResearchContext(research_question=context.strip())
    if isinstance(context, dict):
        data = dict(context)
        if "research_question" not in data and "question" in data:
            data["research_question"] = data.pop("question")
        found = _preset_codes_in(data)
        if found:
            raise PresetCodesError(
                "Context cannot include preset codes "
                f"({', '.join(found)}). Open coding has to let codes emerge "
                "from the passages (Glaser, 1992, emergence versus forcing). "
                "Pass a research question and, if you need them, sensitizing concepts only."
            )
        return ResearchContext.model_validate(data)
    raise TypeError("context must be a ResearchContext, a mapping, or a research-question string.")
