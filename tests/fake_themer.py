"""A chat model that groups open-code labels. No network calls."""

from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from automated_grounded_theory.prompts import extract_payload, message_text
from automated_grounded_theory.schemas import (
    ThemeCoding,
    ThemeGroup,
    ThemeMergePlan,
    ThemeNegative,
    ThemeProposal,
    ThemeTension,
)

_DEFS = {
    "Packing to leave": "Getting possessions together in order to go.",
    "Crying at departure": "Distress at the moment of leaving.",
    "Starting work": "A new job pulling the person away.",
    "Other incidents": "Incidents that do not sit with the other themes.",
}
_PROPS = {
    "Packing to leave": ["preparation", "what is taken"],
    "Crying at departure": ["who is present"],
    "Starting work": ["timing"],
    "Other incidents": ["leftover"],
}


def _bucket(label: str) -> str:
    lowered = label.casefold()
    if "pack" in lowered:
        return "Packing to leave"
    if "cry" in lowered:
        return "Crying at departure"
    if "job" in lowered or "hired" in lowered or "work" in lowered:
        return "Starting work"
    return "Other incidents"


class FakeThemer(BaseChatModel):
    """Assigns each CODE_LABEL to a theme from keywords in the label."""

    call_count: int = 0
    # never | until_feedback | always. until_feedback omits one code until the retry lists it.
    drop_last: str = "never"
    invent_code: bool = False
    seen_labels: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-themer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            text = "\n".join(message_text(message) for message in messages)
            payload = extract_payload(text)
            if getattr(schema, "__name__", "") == "ThemeMergePlan":
                names: list[str] = []
                seen: set[str] = set()
                for theme in payload.get("themes") or []:
                    found = str(theme.get("name") or "")
                    if found and found not in seen:
                        seen.add(found)
                        names.append(found)
                if not names:
                    for found in re.findall(r"^THEME_NAME: (.+)$", text, re.M):
                        if found not in seen:
                            seen.add(found)
                            names.append(found)
                plan = ThemeMergePlan(
                    themes=[ThemeGroup(name=name, members=[name]) for name in names]
                )
                if not include_raw:
                    return plan
                raw = AIMessage(
                    content="{}",
                    usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                )
                return {"raw": raw, "parsed": plan, "parsing_error": None}
            labels = [
                str(item.get("label") or item.get("code") or "")
                for item in (payload.get("codes") or [])
                if item.get("label") or item.get("code")
            ]
            if not labels:
                labels = re.findall(r"^CODE_LABEL: (.+)$", text, re.M)
            if self.drop_last == "always" or (self.drop_last == "until_feedback" and "VALIDATION FAILED" not in text):
                labels = labels[:-1]
            self.seen_labels.append(list(labels))
            groups: dict[str, list[str]] = {}
            for label in labels:
                groups.setdefault(_bucket(label), []).append(label)
            if self.invent_code and groups:
                next(iter(groups.values())).append("not a real code")
            themes = [
                ThemeProposal(
                    name=name,
                    definition=_DEFS[name],
                    properties=list(_PROPS[name]),
                    codes=codes,
                )
                for name, codes in groups.items()
                if codes
            ]
            example = ""
            for item in payload.get("codes") or []:
                for sample in item.get("examples") or []:
                    quote = str(sample.get("quote") or "").strip()
                    if quote:
                        example = quote
                        break
                if example:
                    break
            if not example:
                match = re.search(r"^EXAMPLE: (.+)$", text, re.M)
                if match:
                    example = match.group(1).strip()
            names = [theme.name for theme in themes]
            tensions = []
            if len(names) >= 2:
                tensions.append(
                    ThemeTension(
                        themes=names[:2],
                        note="These incidents pull in different directions.",
                        quote="NOT IN THE DATA",
                        passage_id="missing",
                    )
                )
            if example and names:
                tensions.append(
                    ThemeTension(
                        themes=names[:1],
                        note="A passage shows the incident in the participant's words.",
                        quote=example,
                    )
                )
            negatives = []
            if names:
                negatives.append(
                    ThemeNegative(
                        theme=names[0],
                        note="A thin reply sits awkwardly in this theme.",
                        code=themes[0].codes[0],
                        quote="NOT IN THE DATA",
                    )
                )
            coding = ThemeCoding(themes=themes, tensions=tensions, negative_cases=negatives)
            if not include_raw:
                return coding
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            )
            return {"raw": raw, "parsed": coding, "parsing_error": None}

        return RunnableLambda(_call)
