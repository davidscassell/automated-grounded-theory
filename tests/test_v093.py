"""0.9.3 review fixes. No network and no paid model calls."""

from __future__ import annotations

import asyncio
import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import __version__, axial_code, focused_code, selective_code, theoretical_sample
from automated_grounded_theory.axial import AxialRuntime, _counterexample_node, _select_counterexamples
from automated_grounded_theory.records import POLICY_VERSION
from tests.fake_focused import FakeFocusedCoder
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v090 import _kwargs


def test_version_is_093():
    assert __version__ == "0.9.34"
    assert POLICY_VERSION == "0.9.34"


def _book():
    return {
        "prep": {"code": "preparing suitcase departure", "definition": "packing before a trip"},
        "back": {"code": "returning office schedule", "definition": "coming back to the desk"},
    }


def _relationship(relationship_id="rel-1", **extra):
    row = {
        "relationship_id": relationship_id,
        "from_id": "prep",
        "to_id": "back",
        "passage_id": "cited",
        "doc_id": "d1",
        "respondent_id": "r-cited",
        "statement": "Preparing the suitcase before departure may be related to the office return.",
        "quote": "packing the suitcase before departure",
        "evidence_tier": 1,
    }
    row.update(extra)
    return row


def _split_passages():
    return [
        {
            "passage_id": "cited",
            "doc_id": "d1",
            "respondent_id": "r-cited",
            "text": "Packing the suitcase before departure.",
        },
        {
            "passage_id": "split-a",
            "doc_id": "d2",
            "respondent_id": "r-split",
            "text": "Packing the suitcase before the departure.",
        },
        {
            "passage_id": "split-b",
            "doc_id": "d2",
            "respondent_id": "r-split",
            "text": "Returning to the office schedule the next week.",
        },
    ]


def test_split_questions_are_one_respondent_with_the_complementary_passage():
    queued, stats = _select_counterexamples(
        [_relationship()],
        _split_passages(),
        slots=4,
        synthetic=False,
        codebook=_book(),
    )
    assert len(queued) == 1
    item = queued[0]
    assert item["respondent_id"] == "r-split"
    assert item["source"] == "both_endpoints"
    assert item["source"] not in {"a_without_b", "b_without_a"}
    assert set(item["complementary_passage_ids"]) == {"split-a", "split-b"} - {item["passage_id"]}
    matrix = stats["by_relationship"]["rel-1"]["cooccurrence_2x2"]
    assert matrix["both"] >= 1
    runtime = AxialRuntime(coder=FakeAxialCoder(), request_timeout=0)
    state = {
        "relationships": [_relationship()],
        "passages": _split_passages(),
        "codebook": _book(),
        "segments": [],
        "warnings": [],
        "prompt_log": [],
        "metadata": {
            "procedure": {
                "counterexample_calls": 1,
                "negative_case_search": True,
                "axial_synthetic_counterexample": False,
                "max_axial_review_tokens": 20000,
            },
            "performance": {"batch_size": 1, "axial_token_budget": 6000},
            "stage_outcomes": {"proposals": {"status": "complete"}},
        },
        "axial_calls_used": 0,
    }
    result = asyncio.run(_counterexample_node(runtime)(state))
    rendered = json.dumps(result["prompt_log"])
    assert "Packing the suitcase" in rendered
    assert "Returning to the office" in rendered
    assert "split across questions" in rendered


def _one_sided(passage_id, respondent_id, relationship_id="rel-1"):
    return {
        "passage_id": passage_id,
        "doc_id": respondent_id,
        "respondent_id": respondent_id,
        "text": "The suitcase was packed before the departure and nothing else was said.",
    }


def test_round_robin_slots_and_remaining_candidates_are_budget_limited():
    book = _book()
    relationships = [_relationship("rel-1"), _relationship("rel-2")]
    passages = [
        {"passage_id": "cited", "doc_id": "d0", "respondent_id": "r-cited", "text": "Packing the suitcase before departure."}
    ]
    for index in range(3):
        passages.append(_one_sided(f"a-{index}", f"r-a-{index}"))
    queued, stats = _select_counterexamples(relationships, passages, slots=2, synthetic=False, codebook=book)
    assert stats["plausible_candidates"] == 6
    assert stats["queued"] == 2
    assert stats["attempted"] == 0
    assert stats["remaining_candidates"] == 4
    assert stats["no_call_reason"] == "budget"
    by_relationship = [item["relationship_id"] for item in queued]
    assert by_relationship == ["rel-1", "rel-2"]
    assert stats["by_relationship"]["rel-2"]["no_call_reason"] == "budget"
    runtime = AxialRuntime(coder=FakeAxialCoder(), request_timeout=0)
    state = {
        "relationships": relationships,
        "passages": passages,
        "codebook": book,
        "segments": [],
        "warnings": [],
        "prompt_log": [],
        "metadata": {
            "procedure": {
                "counterexample_calls": 1,
                "negative_case_search": True,
                "axial_synthetic_counterexample": False,
                "max_axial_review_tokens": 20000,
            },
            "performance": {"batch_size": 1, "axial_token_budget": 6000},
            "stage_outcomes": {"proposals": {"status": "complete"}},
        },
        "axial_calls_used": 0,
    }
    result = asyncio.run(_counterexample_node(runtime)(state))
    stage = result["metadata"]["stage_outcomes"]["counterexamples"]
    relevance = result["metadata"]["axial_counterexample_relevance"]
    assert stage["status"] == "budget_limited"
    assert stage["reason"] == "budget"
    assert relevance["plausible_candidates"] == 6
    assert relevance["queued"] == 1
    assert relevance["attempted"] == 1
    assert relevance["remaining_candidates"] == 5
    assert result["metadata"]["axial_counterexample_relevance"]["by_relationship"]["rel-2"]["no_call_reason"] == "budget"


class _MalformedSampling(BaseChatModel):
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-malformed-sampling"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="not json"))])

    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            self.call_count += 1
            raw = AIMessage(
                content="not json",
                usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
            )
            return {"raw": raw, "parsed": None, "parsing_error": "malformed"}

        return RunnableLambda(_call)


def test_sampling_request_budget_is_not_sent_and_malformed_output_is_incomplete():
    coder = _MalformedSampling()
    tight = theoretical_sample(
        _corpus(),
        coder,
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        sampling_token_budget=1,
        **_kwargs(),
    )
    assert coder.call_count == 0
    assert tight.metadata["status"] == "incomplete"
    assert tight.metadata["stop"] == "request_budget"
    assert tight.metadata["model_sampling_status"] == "incomplete"
    omitted = [row for row in tight.prompt_log if row.get("status") == "omitted"]
    assert omitted
    assert omitted[0]["received"] is False
    assert int(omitted[0]["estimated_input_tokens"]) > 1
    broken = _MalformedSampling()
    failed = theoretical_sample(
        _corpus(),
        broken,
        decision_mode="llm",
        pursuit="brief_only",
        interrupt=False,
        **_kwargs(),
    )
    assert broken.call_count >= 2
    assert failed.metadata["status"] == "incomplete"
    assert failed.metadata["model_sampling_status"] == "incomplete"
    attempts = failed.metadata.get("sampling_attempts") or []
    assert len(attempts) >= 1
    assert any(int((row.get("usage") or {}).get("total_tokens") or 0) > 0 for row in attempts)
    assert any(row.get("status") in {"parse_failed", "schema_unparsed"} for row in failed.prompt_log)


def test_reread_prompt_includes_the_sampling_question_and_stores_the_direction():
    from tests.test_v090 import FakeSamplingCoder

    coder = FakeSamplingCoder()
    result = theoretical_sample(
        _corpus(),
        coder,
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        **_kwargs(),
    )
    rereads = [row for row in result.prompt_log if row.get("kind") == "sampling_reread"]
    assert rereads
    rendered = json.dumps(rereads)
    assert "sampling_query" in rendered
    assert "comparison" in rendered
    assert result.reread_notes
    note = result.reread_notes[0]
    assert note["direction_id"]
    assert note["target_id"]
    assert note["sampling_query"]
    assert note["direction_id"] in rendered


def _wide_corpus():
    """Fourteen categories with a shared passage, so each proposal is a real request."""
    labels = [f"category{index}unique" for index in range(14)]
    return {
        "codebook": {
            f"c{index}": {"code_id": f"c{index}", "code": label, "definition": label}
            for index, label in enumerate(labels)
        },
        "passages": [{"passage_id": "p1", "doc_id": "d1", "text": " ".join(labels)}],
        "segments": [
            {"segment_id": f"s{index}", "code_id": f"c{index}", "passage_id": "p1", "quote": label}
            for index, label in enumerate(labels)
        ],
        "properties": [],
        "dimensions": [],
        "memos": [],
    }


def test_axial_step_allowance_covers_the_old_fixed_limit_and_a_low_limit_is_incomplete():
    finished = axial_code(
        _wide_corpus(),
        FakeAxialCoder(),
        seed=1,
        max_axial_review_calls=100,
        axial_share_propose=1.0,
        axial_share_counterexample=0.0,
        batch_size=1,
        interrupt=False,
        axial_request_timeout=0,
        max_axial_review_tokens=2_000_000,
        negative_case_search=False,
    )
    assert finished.metadata.get("stop") != "recursion_limit"
    assert int(finished.metadata["performance"]["recursion_limit"]) > 80
    assert finished.metadata.get("provider_budget", {}).get("calls_used", 0) > 80
    assert len(finished.prompt_log) > 80
    held = axial_code(
        _wide_corpus(),
        FakeAxialCoder(),
        seed=1,
        max_axial_review_calls=100,
        axial_share_propose=1.0,
        axial_share_counterexample=0.0,
        batch_size=1,
        interrupt=False,
        axial_request_timeout=0,
        max_axial_review_tokens=2_000_000,
        negative_case_search=False,
        recursion_limit=12,
    )
    assert held.metadata["status"] == "incomplete"
    assert held.metadata["stop"] == "recursion_limit"
    assert held.prompt_log
    assert held.metadata.get("provider_budget", {}).get("calls_used", 0) >= 1
    assert held.relationships


def test_sampling_selective_and_focused_step_limits_stay_inside_the_result():
    sampled = theoretical_sample(_corpus(), interrupt=False, recursion_limit=2, **_kwargs())
    assert sampled.metadata["status"] == "incomplete"
    assert sampled.metadata["stop"] == "recursion_limit"
    selected = selective_code(
        _corpus(),
        FakeAxialCoder(),
        interrupt=False,
        recursion_limit=2,
        seed=1,
        axial_request_timeout=0,
        include_memos=False,
        include_storyline=False,
    )
    assert selected.metadata["status"] == "incomplete"
    assert selected.metadata["stop"] == "recursion_limit"
    focused = focused_code(
        {
            "codebook": [{"code_id": "c1", "code": "leaving home", "definition": "leaving"}],
            "passages": [{"passage_id": "p1", "doc_id": "d1", "text": "I was leaving home."}],
            "segments": [{"segment_id": "s1", "code_id": "c1", "passage_id": "p1", "quote": "leaving home"}],
            "memos": [],
        },
        model=FakeFocusedCoder(),
        interrupt=False,
        recursion_limit=2,
        seed=1,
        focused_request_timeout=0,
    )
    assert focused.metadata["status"] == "incomplete"
    assert focused.metadata["stop"] == "recursion_limit"
