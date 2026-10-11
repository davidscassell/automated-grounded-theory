"""A chat model that codes from the passage text. No network calls."""

from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.prompts import message_text
from automated_grounded_theory.schemas import (
    CodeProposal,
    DefinitionUpdate,
    MergeDecision,
    PassageCoding,
    SweepCoding,
)


def _passage(text: str) -> str:
    match = re.search(r"<<<PASSAGE\n(.*)\nPASSAGE>>>", text, re.S)
    return match.group(1) if match else ""


def _labels(text: str) -> set[str]:
    return {label.casefold() for label in re.findall(r"^CODE_LABEL: (.+)$", text, re.M)}


def _sentence(passage: str, keyword: str) -> str:
    lowered = passage.casefold()
    index = lowered.find(keyword)
    if index < 0:
        return passage.strip()
    start = max(passage.rfind(".", 0, index), passage.rfind("?", 0, index), passage.rfind("!", 0, index))
    start = 0 if start < 0 else start + 1
    end = len(passage)
    for punct in ".!?":
        found = passage.find(punct, index)
        if found != -1:
            end = min(end, found + 1)
    return passage[start:end].strip()


def _rule(passage: str) -> tuple[str, str] | None:
    lowered = passage.casefold()
    if "suitcase" in lowered:
        return "packing bags", _sentence(passage, "suitcase")
    if "boxes" in lowered:
        return "packing bags up", _sentence(passage, "boxes")
    if "crying" in lowered or "cried" in lowered:
        return "crying at the station", _sentence(passage, "crying" if "crying" in lowered else "cried")
    if "hired" in lowered or "new job" in lowered:
        return "starting a new job", _sentence(passage, "hired" if "hired" in lowered else "job")
    return None


class FakeCoder(BaseChatModel):
    """Returns structured open codes derived from keywords in the passage."""

    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-grounded-theory"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            self.call_count += 1
            text = "\n".join(message_text(message) for message in messages)
            if getattr(schema, "__name__", "") == "SweepCoding":
                return self.sweep(text)
            return self.code(text)

        return RunnableLambda(_call)

    def code(self, text: str) -> PassageCoding:
        passage = _passage(text)
        rule = _rule(passage)
        if rule is None:
            return PassageCoding(what_is_happening="Nothing codable stood out.", codes=[], memo="No incident to code.")
        label, quote = rule
        action = "reuse" if label.casefold() in _labels(text) else "new"
        return PassageCoding(
            what_is_happening=quote,
            codes=[
                CodeProposal(
                    action=action,
                    code=label,
                    definition=f"Incidents of {label}.",
                    quote=quote,
                    in_vivo=False,
                    rationale="The incident matches this gerund.",
                    confidence=0.9,
                )
            ],
            memo="Compared with the codebook and kept the code close to the quote.",
        )

    def sweep(self, text: str) -> SweepCoding:
        match = re.search(r"<<<CANDIDATES\n(.*)\nCANDIDATES>>>", text, re.S)
        pairs = json.loads(match.group(1)) if match else []
        merges = []
        for pair in pairs:
            source, target = pair["source"], pair["target"]
            if "packing" in source.casefold() and "packing" in target.casefold():
                if "up" in source.casefold() and "up" not in target.casefold():
                    source, target = target, source
                merges.append(MergeDecision(source=source, target=target, rationale="Same packing incident."))
        return SweepCoding(
            merges=merges,
            definitions=[
                DefinitionUpdate(
                    code="packing bags up",
                    definition="Bagging and boxing possessions while leaving a place.",
                )
            ],
            integrative_memo="Constant comparison collapsed the packing codes.",
            reflexive_memo="These codes came from a model. The researcher judges saturation.",
        )


class FlakyCoder(FakeCoder):
    """First passage coding quotes text that is not in the passage, then corrects."""

    def code(self, text: str) -> PassageCoding:
        if "quote_not_verbatim" in text:
            return super().code(text)
        passage = _passage(text)
        rule = _rule(passage)
        label = rule[0] if rule else "packing bags"
        return PassageCoding(
            what_is_happening="A first pass missed the wording.",
            codes=[
                CodeProposal(
                    action="new",
                    code=label,
                    definition=f"Incidents of {label}.",
                    quote="NOT IN THE PASSAGE",
                    rationale="This quote is wrong on purpose.",
                    confidence=0.4,
                )
            ],
            memo="Retry me.",
        )
