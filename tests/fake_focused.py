"""A chat model that returns fixed focused-coding JSON. No network calls."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.prompts import extract_payload, message_text
from automated_grounded_theory.schemas import (
    FocusedCodeDecision,
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
            if name == "FocusedRecodeBatch":
                parsed = self.batch(extract_payload(text))
            else:
                parsed = FocusedRecodeBatch(passages=[])
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
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
