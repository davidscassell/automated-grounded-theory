"""0.9.4 review fixes. No network and no paid model calls."""

from __future__ import annotations

import asyncio
import json

from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, axial_code, theoretical_sample
from automated_grounded_theory.async_support import ProviderAllowance
from automated_grounded_theory.axial import (
    _annotate_endpoints,
    _apply_counterexample,
    _cited_story_passages,
    _stored_quote_passage,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import apply_evidence_tier
from automated_grounded_theory.sampling import SamplingRuntime, _model_sampling_pairs, _reread
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v090 import FakeSamplingCoder, _kwargs


def test_version_is_094():
    assert __version__ == "0.9.7"
    assert POLICY_VERSION == "0.9.7"


def _assigned_passage(paused):
    donor = ""
    passage = ""
    for group in paused.passage_proposals:
        ids = [str(row.get("passage_id") or "") for row in (group.get("proposals") or [])]
        if ids:
            donor = str(group.get("direction_id") or "")
            passage = ids[0]
            break
    other = next(str(row["direction_id"]) for row in paused.directions if str(row["direction_id"]) != donor)
    return donor, other, passage


def _resume(donor, other, passage, *, both=False):
    return {
        "directions": [
            {
                "direction_id": donor,
                "action": "accept",
                "sampled_passage_ids": [passage] if both else [],
            },
            {
                "direction_id": other,
                "action": "edit",
                "sampled_passage_ids": [passage],
            },
        ]
    }


def test_human_assignment_is_reread_under_that_direction():
    saver = InMemorySaver()
    coder = FakeSamplingCoder()
    common = _kwargs(pursuit="reread")
    paused = theoretical_sample(
        _corpus(),
        coder,
        checkpointer=saver,
        thread_id="direction-pair",
        **common,
    )
    donor, other, passage = _assigned_passage(paused)
    resumed = theoretical_sample(
        _corpus(),
        coder,
        checkpointer=saver,
        thread_id="direction-pair",
        resuming=True,
        resume=_resume(donor, other, passage),
        **common,
    )
    assert resumed.reread_notes
    notes = [row for row in resumed.reread_notes if row["passage_id"] == passage]
    assert len(notes) == 1
    assert notes[0]["direction_id"] == other
    assert notes[0]["direction_id"] != donor
    rendered = json.dumps([row for row in resumed.prompt_log if row.get("kind") == "sampling_reread"])
    assert other in rendered
    assert f"direction {other}" in rendered
    pairs = resumed.metadata["researcher_sampling"]["pairs"]
    assert {"direction_id": other, "passage_id": passage} in pairs
    assert {"direction_id": donor, "passage_id": passage} not in pairs


def test_one_passage_on_two_directions_is_two_rereads():
    saver = InMemorySaver()
    coder = FakeSamplingCoder()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=4)
    paused = theoretical_sample(_corpus(), coder, checkpointer=saver, thread_id="two-directions", **common)
    donor, other, passage = _assigned_passage(paused)
    resumed = theoretical_sample(
        _corpus(),
        coder,
        checkpointer=saver,
        thread_id="two-directions",
        resuming=True,
        resume=_resume(donor, other, passage, both=True),
        **common,
    )
    notes = [row for row in resumed.reread_notes if row["passage_id"] == passage]
    assert {row["direction_id"] for row in notes} == {donor, other}
    assert len(notes) == 2
    assert resumed.metadata["provider_budget"]["calls_used"] == 2


def test_a_second_direction_is_counted_when_the_call_allowance_allows_one():
    saver = InMemorySaver()
    coder = FakeSamplingCoder()
    common = _kwargs(pursuit="reread", max_sampling_review_calls=1)
    paused = theoretical_sample(_corpus(), coder, checkpointer=saver, thread_id="one-call", **common)
    donor, other, passage = _assigned_passage(paused)
    resumed = theoretical_sample(
        _corpus(),
        coder,
        checkpointer=saver,
        thread_id="one-call",
        resuming=True,
        resume=_resume(donor, other, passage, both=True),
        **common,
    )
    notes = [row for row in resumed.reread_notes if row["passage_id"] == passage]
    unsent = list(resumed.metadata.get("reread_unsent_pairs") or [])
    assert len(notes) == 1
    assert len(unsent) == 1
    assert unsent[0]["passage_id"] == passage
    assert {notes[0]["direction_id"], unsent[0]["direction_id"]} == {donor, other}
    assert resumed.metadata["reread_status"] == "budget_limited"
    assert resumed.metadata["provider_budget"]["calls_used"] == 1


def test_flat_model_lists_do_not_collapse_two_proposal_directions():
    pairs, unpaired = _model_sampling_pairs(
        {},
        ["d1", "d2"],
        ["p"],
        [
            {"direction_id": "d1", "proposals": [{"passage_id": "p"}]},
            {"direction_id": "d2", "proposals": [{"passage_id": "p"}]},
        ],
    )
    assert unpaired == []
    assert pairs == [
        {"direction_id": "d1", "passage_id": "p"},
        {"direction_id": "d2", "passage_id": "p"},
    ]
    explicit, flagged = _model_sampling_pairs(
        {"selections": [{"direction_id": "d2", "passage_id": "p"}]},
        ["d1", "d2"],
        ["p"],
        [
            {"direction_id": "d1", "proposals": [{"passage_id": "p"}]},
            {"direction_id": "d2", "proposals": [{"passage_id": "p"}]},
        ],
    )
    assert flagged == []
    assert explicit == [{"direction_id": "d2", "passage_id": "p"}]
    lost, missing = _model_sampling_pairs(
        {},
        ["d1", "d2"],
        ["p"],
        [{"direction_id": "d1", "proposals": [{"passage_id": "other"}]}],
    )
    assert lost == []
    assert missing == ["p"]

    async def _both():
        runtime = SamplingRuntime(coder=FakeSamplingCoder(), pursuit="reread", decision_mode="llm")
        runtime.limiter.provider_allowance = ProviderAllowance(max_tokens=100_000, stages={"coding": 4})
        state = {
            "passages": [{"passage_id": "p", "text": "A single suitcase sat by the door."}],
            "directions": [
                {"direction_id": "d1", "kind": "thin_category", "target_id": "t1", "sampling_query": "question one"},
                {"direction_id": "d2", "kind": "no_negative_case", "target_id": "t2", "sampling_query": "question two"},
            ],
            "passage_proposals": [
                {"direction_id": "d1", "proposals": [{"passage_id": "p"}]},
                {"direction_id": "d2", "proposals": [{"passage_id": "p"}]},
            ],
            "metadata": {
                "decision_mode": "llm",
                "model_direction_ids": ["d1", "d2"],
                "model_passage_ids": ["p"],
                "procedure": {"pursuit": "reread"},
            },
            "reread_notes": [],
            "prompt_log": [],
        }
        result = {}
        for _ in range(4):
            result = await _reread(runtime, state)
            state = {**state, **result}
            if result["metadata"].get("reread_status") in {"complete", "budget_limited", "incomplete", "skipped"}:
                break
        return result

    result = asyncio.run(_both())
    assert {row["direction_id"] for row in result["reread_notes"]} == {"d1", "d2"}
    assert len(result["reread_notes"]) == 2


def test_complementary_quote_keeps_its_own_passage():
    primary = {"passage_id": "a", "doc_id": "d1", "text": "The weather was fine that morning."}
    complementary = {
        "passage_id": "b",
        "doc_id": "d2",
        "respondent_id": "r2",
        "text": "Packing the suitcase before returning to the office.",
    }
    relationship = {
        "relationship_id": "rel-1",
        "from_id": "pack",
        "to_id": "back",
        "passage_id": "a",
        "quote": "Packing the suitcase before returning to the office",
        "counterexamples": [],
        "flags": [],
    }
    quote = "Packing the suitcase before returning to the office"
    _apply_counterexample(
        relationship,
        {"judgment": "qualifies", "quote": quote, "rationale": "The office return is a condition."},
        primary,
        complementary_passage_ids=["b"],
        complementary_passages=[complementary],
    )
    record = relationship["counterexamples"][0]
    assert record["passage_id"] == "a"
    assert record["quote_passage_id"] == "b"
    assert record["valid_test"] is True
    assert record["complementary_passage_ids"] == ["b"]
    assert record["quote_start"] == 0
    assert complementary["text"][record["quote_start"] : record["quote_end"]] == quote
    assert primary["text"].casefold().find(quote.casefold()) < 0

    book = {
        "pack": {"code": "packing suitcase", "definition": "packing before a trip"},
        "back": {"code": "returning office", "definition": "coming back to the desk"},
    }
    passages = {
        "a": primary,
        "b": complementary,
    }
    relationship["quote_from"] = "The weather was fine"
    relationship["quote_to"] = quote
    _annotate_endpoints(relationship, book, [], passages)
    ends = {row["endpoint"]: row for row in relationship["endpoint_anchoring"]}
    assert ends["from"]["passage_id"] == "a"
    assert ends["from"]["quote_passage_id"] == "a"
    assert ends["to"]["passage_id"] == "a"
    assert ends["to"]["quote_passage_id"] == "b"
    assert ends["to"]["quote_start"] == 0
    assert relationship["evidence_scope"] == "cross_respondent"
    assert "cross_respondent_pattern" in relationship["flags"]
    fields = apply_evidence_tier(relationship, passages=passages, codebook=book, pattern_min_cases=2)
    quoted = [
        item["passage_id"]
        for case in (fields["supporting_cases"] or [])
        for item in case["quotes"]
    ]
    assert quoted
    assert set(quoted) == {"b"}
    assert all(item["quote_passage_id"] == "b" for case in fields["supporting_cases"] for item in case["quotes"])
    bundle = _cited_story_passages(
        {"passages": [primary, complementary], "segments": []},
        [relationship],
        [],
    )
    assert "b" in [row["passage_id"] for row in bundle]
    older = {"passage_id": "a"}
    assert _stored_quote_passage(older) == "a"
    older_bundle = _cited_story_passages(
        {"passages": [primary, complementary], "segments": []},
        [{"passage_id": "a", "counterexamples": [{"passage_id": "a", "quote": quote}]}],
        [],
    )
    assert [row["passage_id"] for row in older_bundle] == ["a"]


def _mixed_corpus():
    labels = [f"category{index}unique" for index in range(20)]
    evidenced = set(range(10))
    return {
        "codebook": {
            f"c{index}": {"code_id": f"c{index}", "code": label, "definition": label}
            for index, label in enumerate(labels)
        },
        "passages": [{"passage_id": "p1", "doc_id": "d1", "text": " ".join(labels)}],
        "segments": [
            {"segment_id": f"s{index}", "code_id": f"c{index}", "passage_id": "p1", "quote": label}
            for index, label in enumerate(labels)
            if index in evidenced
        ],
        "properties": [],
        "dimensions": [],
        "memos": [],
    }


def _mixed_kwargs(**extra):
    fields = dict(
        seed=1,
        max_axial_review_calls=200,
        axial_share_propose=1.0,
        axial_share_counterexample=0.0,
        batch_size=2,
        interrupt=False,
        axial_request_timeout=0,
        max_axial_review_tokens=2_000_000,
        negative_case_search=False,
    )
    fields.update(extra)
    return fields


def test_mixed_evidence_finishes_supported_pairs_under_the_derived_limit():
    finished = axial_code(_mixed_corpus(), FakeAxialCoder(), **_mixed_kwargs())
    assert finished.metadata.get("stop") != "recursion_limit"
    accounting = finished.metadata["pair_accounting"]
    assert accounting["completed_count"] == 45
    assert finished.metadata["provider_budget"]["calls_used"] == 23
    assert finished.metadata["missing_category_evidence"]
    assert int(finished.metadata["performance"]["recursion_limit"]) > 119
    held = axial_code(_mixed_corpus(), FakeAxialCoder(), **_mixed_kwargs(recursion_limit=12))
    assert held.metadata["status"] == "incomplete"
    assert held.metadata["stop"] == "recursion_limit"
    assert held.prompt_log
    assert held.metadata.get("provider_budget", {}).get("calls_used", 0) >= 1
