"""A chat model that returns fixed focused-coding JSON. No network calls."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.prompts import extract_payload, message_text
from automated_grounded_theory.schemas import (
    FocusedAlternativeReading,
    FocusedClaimJudgment,
    FocusedClaimTest,
    FocusedCodeDecision,
    FocusedCodeVerdict,
    FocusedCompeteBatch,
    FocusedCompetePassage,
    FocusedDimensionNote,
    FocusedPassageRecode,
    FocusedPropertyNote,
    FocusedRecodeBatch,
)


class FakeFocusedCoder(BaseChatModel):
    """Schema-shaped focused recodes. ``mode`` selects the fixture."""

    call_count: int = 0
    mode: str = "echo"
    outside_quote: str = ""
    reported_usage: dict | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-focused"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            text = "\n".join(message_text(message) for message in messages)
            name = getattr(schema, "__name__", "")
            payload = extract_payload(text)
            if name == "FocusedCompeteBatch":
                parsed = self.compete_batch(payload)
            elif name == "FocusedClaimTest":
                parsed = self.claim_test_batch(payload)
            elif name == "FocusedRecodeBatch":
                parsed = self.batch(payload)
            else:
                parsed = FocusedRecodeBatch(passages=[])
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata=dict(
                    self.reported_usage
                    or {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42}
                ),
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)

    def batch(self, payload: dict) -> FocusedRecodeBatch:
        if self.mode == "empty":
            return FocusedRecodeBatch(passages=[])
        raised = list(payload.get("raised_codes") or [])
        code = raised[0] if raised else {"code_id": "", "code": "noticing a gap", "definition": ""}
        feedback = str(payload.get("feedback") or "")
        rows = []
        for passage in payload.get("passages") or []:
            text = str(passage.get("text") or "")
            passage_id = str(passage.get("passage_id") or "")
            doc_id = str(passage.get("doc_id") or "")
            quote = text.strip()
            if self.mode == "retry_label" and "LABEL RETRY" not in feedback:
                decision = FocusedCodeDecision(action="add", code="", quote=quote, code_id="")
                rows.append(FocusedPassageRecode(passage_id=passage_id, codes=[decision]))
                continue
            if self.mode == "drop":
                rows.append(FocusedPassageRecode(passage_id=passage_id, codes=[]))
                continue
            if self.mode == "alt_span":
                quote = "packing my suitcase" if "packing my suitcase" in text else text[: max(1, len(text) // 2)]
            if self.mode == "new_label":
                decision = FocusedCodeDecision(
                    action="add",
                    code="marking the hour",
                    code_id="",
                    quote=quote,
                    definition="Noting the time of the incident.",
                    rationale="The passage marks when the incident happened.",
                )
                rows.append(
                    FocusedPassageRecode(
                        passage_id=passage_id,
                        codes=[decision],
                        memo="A competing reading is that the passage is marking the hour rather than only packing.",
                        properties=[
                            FocusedPropertyNote(
                                category_id="",
                                name="timing of the incident",
                                text="The analytic property is when the incident sits in the day, not a summary of the scene.",
                                document_id=doc_id,
                                passage_id=passage_id,
                                quote=quote,
                            )
                        ],
                        dimensions=[
                            FocusedDimensionNote(
                                category_id="",
                                name="time of day",
                                pole="morning",
                                text="This pole is morning rather than night.",
                                document_id=doc_id,
                                passage_id=passage_id,
                                quote=quote,
                            )
                        ],
                    )
                )
                continue
            if self.mode == "outside":
                quote = self.outside_quote or quote
            decision = FocusedCodeDecision(
                action="keep",
                code=str(code.get("code") or ""),
                code_id=str(code.get("code_id") or ""),
                quote=quote,
                definition=str(code.get("definition") or ""),
                rationale="The passage still supports this code.",
            )
            properties = []
            dimensions = []
            if self.mode == "unsupported":
                properties.append(
                    FocusedPropertyNote(
                        category_id=str(code.get("code_id") or ""),
                        name="unsupported hunch",
                        text="A property with no citation.",
                    )
                )
                dimensions.append(
                    FocusedDimensionNote(
                        category_id=str(code.get("code_id") or ""),
                        name="pace",
                        pole="rushed",
                        text="The quote is not in the corpus.",
                        document_id=doc_id,
                        passage_id=passage_id,
                        quote="this quote is not in any passage",
                    )
                )
            elif self.mode == "outside":
                properties.append(
                    FocusedPropertyNote(
                        category_id=str(code.get("code_id") or ""),
                        name="located elsewhere",
                        text="The quote sits in another passage.",
                        document_id=doc_id,
                        passage_id=passage_id,
                        quote=quote,
                    )
                )
            else:
                properties.append(
                    FocusedPropertyNote(
                        category_id=str(code.get("code_id") or ""),
                        name="timing",
                        text="The incident is tied to this passage.",
                        document_id=doc_id,
                        passage_id=passage_id,
                        quote=quote,
                    )
                )
                dimensions.append(
                    FocusedDimensionNote(
                        category_id=str(code.get("code_id") or ""),
                        name="timing",
                        pole="during",
                        text="The incident happens in the quoted moment.",
                        document_id=doc_id,
                        passage_id=passage_id,
                        quote=quote,
                    )
                )
            rows.append(
                FocusedPassageRecode(
                    passage_id=passage_id,
                    codes=[decision],
                    properties=properties,
                    dimensions=dimensions,
                )
            )
        return FocusedRecodeBatch(passages=rows)

    def claim_test_batch(self, payload: dict) -> FocusedClaimTest:
        """Judge the claim in the prompt. The default is supports, which is not a code change."""
        if self.mode == "empty":
            return FocusedClaimTest(passages=[])
        rows = []
        for passage in payload.get("passages") or []:
            claim = passage.get("claim") or {}
            rows.append(
                FocusedClaimJudgment(
                    passage_id=str(passage.get("passage_id") or ""),
                    entry_id=str(claim.get("entry_id") or ""),
                    judgment="supports",
                    rationale="The passage is consistent with the supplied claim.",
                    quote=str(passage.get("text") or ""),
                )
            )
        return FocusedClaimTest(passages=rows)

    def compete_batch(self, payload: dict) -> FocusedCompeteBatch:
        """Explicit keep/revise/drop verdicts. The alternative is a different code and fits less well."""
        if self.mode == "empty":
            return FocusedCompeteBatch(passages=[])
        rows = []
        for passage in payload.get("passages") or []:
            text = str(passage.get("text") or "")
            passage_id = str(passage.get("passage_id") or "")
            doc_id = str(passage.get("doc_id") or "")
            quote = text.strip()
            initial = list(passage.get("initial_codes") or [])
            if not initial:
                initial = list(payload.get("codes") or [])[:1]
            if not initial:
                initial = [{"code_id": "", "code": "noticing a gap", "definition": ""}]
            verdict_name = "drop" if self.mode == "drop" else "keep"
            rationale = (
                "The passage does not support this code."
                if verdict_name == "drop"
                else "The disconfirming reading does not displace this code."
            )
            verdicts = [
                FocusedCodeVerdict(
                    code_id=str(code.get("code_id") or ""),
                    code=str(code.get("code") or ""),
                    verdict=verdict_name,
                    rationale=rationale,
                    quote=quote,
                )
                for code in initial
            ]
            first = initial[0]
            kept_ids = {str(code.get("code_id") or "") for code in initial}
            other = next(
                (
                    code
                    for code in (payload.get("codes") or [])
                    if str(code.get("code_id") or "") not in kept_ids and str(code.get("code") or "")
                ),
                None,
            )
            if other is None:
                alt_code, alt_id = "a competing reading", ""
            else:
                alt_code, alt_id = str(other.get("code") or ""), str(other.get("code_id") or "")
            rows.append(
                FocusedCompetePassage(
                    passage_id=passage_id,
                    verdicts=verdicts,
                    strongest_alternative=FocusedAlternativeReading(
                        code=alt_code,
                        code_id=alt_id,
                        rationale="This alternative fits the passage less well than the kept code.",
                        alternative_fit="weaker",
                        quote=quote,
                    ),
                    properties=[
                        FocusedPropertyNote(
                            category_id=str(first.get("code_id") or ""),
                            name="timing",
                            text="The incident is tied to this passage.",
                            document_id=doc_id,
                            passage_id=passage_id,
                            quote=quote,
                        )
                    ],
                    dimensions=[
                        FocusedDimensionNote(
                            category_id=str(first.get("code_id") or ""),
                            name="timing",
                            pole="during",
                            text="The incident happens in the quoted moment.",
                            document_id=doc_id,
                            passage_id=passage_id,
                            quote=quote,
                        )
                    ],
                )
            )
        return FocusedCompeteBatch(passages=rows)
