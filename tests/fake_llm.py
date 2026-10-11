"""A chat model that codes from the passage text. No network calls."""

from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.prompts import extract_payload, message_text
from automated_grounded_theory.schemas import (
    BatchCoding,
    CodeProposal,
    DefinitionRewrite,
    DefinitionRewriteBatch,
    DefinitionUpdate,
    MergeCheck,
    MergeCheckBatch,
    MergeDecision,
    PassageCoding,
    SweepCoding,
)


def _payload(text: str) -> dict:
    return extract_payload(text)


def _passage(text: str) -> str:
    passages = _payload(text).get("passages") or []
    if passages:
        return str(passages[0].get("text") or "")
    match = re.search(r"<<<PASSAGE\n(.*?)\nPASSAGE>>>", text, re.S)
    return match.group(1) if match else ""


def _labels(text: str) -> list[str]:
    payload = _payload(text)
    labels = []
    for item in list(payload.get("codebook") or []) + list(payload.get("codes") or []):
        label = str(item.get("label") or item.get("code") or "")
        if label:
            labels.append(label)
    if labels:
        return labels
    return re.findall(r"^CODE_LABEL: (.+)$", text, re.M)


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
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            text = "\n".join(message_text(message) for message in messages)
            name = getattr(schema, "__name__", "")
            if name == "SweepCoding":
                parsed = self.sweep(text)
            elif name == "MergeCheckBatch":
                parsed = self.verify(text)
            elif name == "DefinitionRewriteBatch":
                parsed = self.rewrite(text)
            elif name == "BatchCoding":
                parsed = self.batch(text)
            else:
                parsed = self.code(text)
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)

    def batch(self, text: str) -> BatchCoding:
        payload = _payload(text)
        passages = payload.get("passages") or []
        feedback = "\n".join(str(item) for item in (payload.get("feedback") or []))
        items = []
        for row in passages:
            one = {
                "codebook": payload.get("codebook") or [],
                "passages": [row],
                "feedback": payload.get("feedback") or [],
            }
            coded = self.code(json.dumps(one) + "\n" + feedback)
            coded.passage_id = str(row.get("passage_id") or "")
            items.append(coded)
        if items:
            return BatchCoding(passages=items)
        blocks = re.findall(r"<<<PASSAGE\n(.*?)\nPASSAGE>>>", text, re.S)
        ids = re.findall(r"^passage_id: (.+)$", text, re.M)
        header = text.split("PASSAGE TO CODE", 1)[0]
        if "VALIDATION FEEDBACK" in text:
            feedback = text[text.index("VALIDATION FEEDBACK") :]
        for index, block in enumerate(blocks):
            passage_id = ids[index] if index < len(ids) else ""
            one = (
                header
                + "\nPASSAGE TO CODE\n"
                + f"passage_id: {passage_id}\n"
                + "<<<PASSAGE\n"
                + block
                + "\nPASSAGE>>>\n"
                + feedback
            )
            coded = self.code(one)
            coded.passage_id = passage_id
            items.append(coded)
        if not items:
            items.append(self.code(text))
        return BatchCoding(passages=items)

    def code(self, text: str) -> PassageCoding:
        passage = _passage(text)
        rule = _rule(passage)
        if rule is None:
            return PassageCoding(codes=[], memo="")
        label, quote = rule
        known = {item.casefold() for item in _labels(text)}
        action = "reuse" if label.casefold() in known else "new"
        others = [item for item in _labels(text) if item.casefold() != label.casefold()]
        nearest = others[0] if action == "new" and others else ""
        why = "It names a different incident." if nearest else ""
        return PassageCoding(
            codes=[
                CodeProposal(
                    action=action,
                    code=label,
                    nearest_code=nearest,
                    why_not_nearest=why,
                    definition=f"Incidents of {label}.",
                    quote=quote,
                    in_vivo=False,
                    rationale="Matches this incident.",
                    confidence=0.9,
                )
            ],
            memo="Compared with the codebook and kept the code close to the quote.",
        )

    def verify(self, text: str) -> MergeCheckBatch:
        checks = []
        for pair in _payload(text).get("pairs") or []:
            source = str(pair.get("source") or "")
            target = str(pair.get("target") or "")
            accept = "packing" in source.casefold() and "packing" in target.casefold()
            checks.append(
                MergeCheck(
                    pair_id=str(pair.get("pair_id") or ""),
                    source_id=str(pair.get("source_id") or ""),
                    target_id=str(pair.get("target_id") or ""),
                    source=source,
                    target=target,
                    action="accept" if accept else "reject",
                    rationale="Same packing incident." if accept else "Different incidents.",
                    confidence=0.95 if accept else 0.9,
                )
            )
        return MergeCheckBatch(checks=checks)

    def rewrite(self, text: str) -> DefinitionRewriteBatch:
        rewrites = []
        for code in _payload(text).get("codes") or []:
            label = str(code.get("label") or code.get("code") or "")
            if "packing" not in label.casefold():
                continue
            rewrites.append(
                DefinitionRewrite(
                    code_id=str(code.get("code_id") or ""),
                    code=label,
                    definition="Bagging and boxing possessions while leaving a place.",
                    confidence=0.9,
                )
            )
        return DefinitionRewriteBatch(rewrites=rewrites)

    def sweep(self, text: str) -> SweepCoding:
        payload = _payload(text)
        pairs = payload.get("hints") or []
        if not pairs:
            match = re.search(r"<<<CANDIDATES\n(.*)\nCANDIDATES>>>", text, re.S)
            pairs = json.loads(match.group(1)) if match else []
        merges = []
        for pair in pairs:
            source, target = pair["source"], pair["target"]
            if "packing" in source.casefold() and "packing" in target.casefold():
                if "up" in source.casefold() and "up" not in target.casefold():
                    source, target = target, source
                merges.append(
                    MergeDecision(
                        source=source,
                        target=target,
                        rationale="These packing labels name the same incident of bagging possessions before leaving.",
                        rewritten_definition="Bagging and boxing possessions while leaving a place.",
                        confidence=0.9,
                        same_action=True,
                        same_object=True,
                        same_stance=True,
                        contrast_checked=True,
                        material_contrast=False,
                        evidence_same=True,
                        comparison_op="alias_duplicate",
                    )
                )
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
