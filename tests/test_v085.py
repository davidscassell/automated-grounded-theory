"""0.8.5 regressions for the v26 live-eval findings. No network calls."""

from __future__ import annotations

import json

from langgraph.checkpoint.memory import InMemorySaver

from automated_grounded_theory import __version__, axial_code, focused_code, selective_code
from automated_grounded_theory.async_support import ProviderAllowance
from automated_grounded_theory.axial import (
    AxialRuntime,
    _IdleModel,
    _apply_counterexample,
    _coverage,
    _pack_storyline,
    _saturation,
    _selective_split,
    _story_core,
)
from automated_grounded_theory.category_state import plan_relation_pairs
from automated_grounded_theory.focused import _claim_pair_relevant, _rationale_indicates_irrelevance, _review_node
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.saturation import thematic_saturation_report

from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _entry, _source
from tests.test_v074 import _passage, _review_state, _segment
from tests.test_v080 import QUOTE, FakeAxialCoder, _corpus


def test_version_is_085():
    assert __version__ == "0.9.36"
    assert POLICY_VERSION == "0.9.36"


def test_endpoint_anchoring_is_not_a_single_quote():
    book = {
        "tools": {"code_id": "tools", "code": "tools for the work", "definition": "implements used on shift"},
        "coworkers": {"code_id": "coworkers", "code": "valuing shift coworkers", "definition": "people on the crew"},
    }
    one_sided = {
        "relationship_id": "rel-0003",
        "from_id": "tools",
        "to_id": "coworkers",
        "quote": "coworkers",
        "passage_id": "p9",
        "statement": "tools for the work leads to valuing shift coworkers",
        "flags": [],
        "negative_case_search": False,
    }
    both = {
        "relationship_id": "rel-0004",
        "from_id": "tools",
        "to_id": "coworkers",
        "quote": "tools and coworkers on the shift",
        "passage_id": "p9",
        "flags": [],
        "negative_case_search": True,
    }
    held = {"codebook": book, "relationships": [one_sided, both], "storyline": [], "metadata": {}, "empty_slots": []}
    coverage = _coverage(held)
    assert coverage["citation_anchoring_rate"] is None
    assert coverage["endpoint_anchoring_rate"] == 0.5
    assert coverage["endpoint_anchored"] == 2
    assert coverage["endpoint_total"] == 4
    from automated_grounded_theory.axial import _endpoint_anchoring

    rows = _endpoint_anchoring(one_sided, book)
    assert rows[0]["anchored"] is False
    assert rows[0]["reason"] == "quote_misses_category"
    assert rows[1]["anchored"] is False
    assert rows[1]["reason"] == "stem_only"
    only = _coverage({"codebook": book, "relationships": [both], "storyline": [], "metadata": {}, "empty_slots": []})
    assert only["citation_anchoring_rate"] is None
    assert only["citation_anchoring_by_tier"]["pattern_supported"]["citation_rate"] is None
    assert only["citation_anchoring_by_tier"]["pattern_supported"]["n"] == 0
    assert only["endpoint_anchoring_rate"] == 1.0


def test_negative_case_rate_uses_every_relationship():
    book = {"a": {"code": "alpha side"}, "b": {"code": "beta side"}}
    judged = {
        "relationship_id": "rel-1",
        "from_id": "a",
        "to_id": "b",
        "quote": "alpha",
        "flags": [],
        "counterexamples": [{"judgment": "contradicts", "valid_test": True, "synthetic": False}],
    }
    untouched = {"relationship_id": "rel-2", "from_id": "a", "to_id": "b", "quote": "alpha", "flags": [], "counterexamples": []}
    irrelevant = {
        "relationship_id": "rel-3",
        "from_id": "a",
        "to_id": "b",
        "quote": "alpha",
        "flags": [],
        "counterexamples": [{"judgment": "irrelevant", "valid_test": True, "synthetic": False}],
    }
    coverage = _coverage(
        {"codebook": book, "relationships": [judged, untouched, irrelevant], "storyline": [], "metadata": {}, "empty_slots": []}
    )
    assert coverage["relationships_with_negative_case_search"] == round(1 / 3, 4)
    uninformative = _coverage(
        {"codebook": book, "relationships": [irrelevant], "storyline": [], "metadata": {}, "empty_slots": []}
    )
    assert uninformative["relationships_with_negative_case_search"] is None
    assert uninformative["negative_case_relevance"] == "undefined_no_relevant_judgment"


def test_axial_counterexample_rejects_irrelevant_rationales():
    relationship = {"relationship_id": "rel-0006", "flags": [], "counterexamples": [], "negative_cases": []}
    passage = {"passage_id": "p2", "text": "The schedule changed and coworkers stayed."}
    _apply_counterexample(
        relationship,
        {"judgment": "contradicts", "quote": "coworkers stayed", "rationale": "The passages are not linked."},
        passage,
    )
    _apply_counterexample(
        relationship,
        {
            "judgment": "supports",
            "quote": "coworkers stayed",
            "rationale": "The passage does not mention the tools.",
        },
        passage,
    )
    assert relationship["negative_cases"] == []
    assert [row["reason"] for row in relationship["counterexamples"]] == [
        "rationale_indicates_irrelevance",
        "rationale_indicates_irrelevance",
    ]
    assert all(row["valid_test"] is False for row in relationship["counterexamples"])
    assert "rationale_indicates_irrelevance" in relationship["flags"]
    assert _rationale_indicates_irrelevance("not unrelated to the claim") is False


def test_storyline_packs_cited_evidence_and_reports_omissions():
    book = {
        f"c{index}": {"code_id": f"c{index}", "code": f"category {index}", "definition": ("definition " * 40)}
        for index in range(20)
    }
    passages = [{"passage_id": f"p{index}", "text": f"opening filler {index} " + ("word " * 30)} for index in range(10)]
    passages.append({"passage_id": "cited", "text": "coworkers stayed after the layoff"})
    relationship = {
        "relationship_id": "rel-0003",
        "from_id": "c0",
        "to_id": "c1",
        "quote": "coworkers stayed",
        "passage_id": "cited",
        "disposition": "proposed",
        "statement": "category 0 leads to category 1",
    }
    rejected = {
        "relationship_id": "rel-0009",
        "from_id": "c2",
        "to_id": "c3",
        "quote": "coworkers stayed",
        "passage_id": "p0",
        "disposition": "rejected",
        "model_disposition": "rejected",
        "statement": "a rejected link",
    }
    state = {
        "codebook": book,
        "passages": passages,
        "segments": [{"code_id": "c0", "passage_id": "cited"}, {"code_id": "c1", "passage_id": "cited"}],
        "relationships": [relationship, rejected],
        "metadata": {"performance": {"axial_token_budget": 6000}, "procedure": {"max_axial_review_tokens": 24000}},
    }
    runtime = AxialRuntime(coder=_IdleModel(), request_timeout=0)
    runtime.limiter.provider_allowance = ProviderAllowance(max_tokens=24000, stages={"selective": 4})
    payload, _messages, estimate, omitted = _pack_storyline(
        state, core_id="", provenance="unset", metadata=state["metadata"], runtime=runtime
    )
    sent = [row["passage_id"] for row in payload["passages"]]
    assert sent[0] == "cited"
    assert "p1" not in sent
    assert "rel-0009" in omitted["relationship_ids"]
    assert all(row["relationship_id"] != "rel-0009" for row in payload["relationships"])
    assert "c4" in omitted["category_ids"]
    assert estimate <= 6000
    assert {row["code_id"] for row in payload["categories"]} <= {"c0", "c1"}


def test_memos_reserve_a_storyline_call(tmp_path):
    assert _selective_split(4, memos=True, storyline=True) == (3, 1)
    assert _selective_split(1, memos=True, storyline=True) == (0, 1)
    coded = axial_code(_corpus(), FakeAxialCoder(), seed=3, interrupt=False)
    both = selective_code(
        coded,
        FakeAxialCoder(),
        include_memos=True,
        include_storyline=True,
        max_selective_review_calls=4,
        interrupt=False,
        output_dir=tmp_path,
    )
    assert both.metadata["procedure"]["memo_calls"] == 3
    assert both.metadata["procedure"]["storyline_calls"] == 1
    story_row = next(row for row in both.prompt_log if row.get("kind") == "storyline")
    story_body = json.loads(story_row["messages"][-1]["content"])
    story_passages = [row["passage_id"] for row in story_body["passages"]]
    assert "p1" in story_passages
    assert "p-none" not in story_passages
    assert both.memos
    assert "memo_quote_unsupported" not in both.memos[0]["flags"]
    assert "memo_evidence_strength_unstated" in both.memos[0]["flags"]
    assert both.memos[0]["needs_review"] is True
    story = json.loads((tmp_path / "storyline.json").read_text(encoding="utf-8"))
    assert "core_id" in story and "core_provenance" in story
    assert story["core_provenance"] == "unset"
    reserved = selective_code(
        coded,
        FakeAxialCoder(),
        include_memos=True,
        include_storyline=True,
        max_selective_review_calls=1,
        interrupt=False,
    )
    assert reserved.memos == []
    assert any(row.get("kind") == "storyline" and row.get("status") != "omitted" for row in reserved.prompt_log)


def test_rejected_relationship_is_flagged_on_the_storyline(tmp_path):
    coded = axial_code(_corpus(), FakeAxialCoder(), seed=3, interrupt=False)
    coded.relationships[0]["disposition"] = "rejected"
    coded.relationships[0]["model_disposition"] = "rejected"
    finished = selective_code(
        coded,
        FakeAxialCoder(),
        include_storyline=True,
        interrupt=False,
        output_dir=tmp_path,
    )
    story = next(row for row in finished.prompt_log if row.get("kind") == "storyline")
    assert story["stage"] == "selective"
    assert "proposing grounded-theory relationships" not in story["messages"][0]["content"]
    body = json.loads(story["messages"][-1]["content"])
    assert body["relationships"] == []
    cited = [row["passage_id"] for row in body["passages"]]
    assert "p-none" not in cited
    assert cited == []
    flagged = [row for row in finished.storyline if "rel-0001" in row["relationship_ids"]]
    assert flagged
    assert all("storyline_rejected_relationship" in row["flags"] for row in flagged)
    exported = json.loads((tmp_path / "storyline.json").read_text(encoding="utf-8"))
    assert exported["core_provenance"] == "unset"


def test_combined_run_keeps_the_coding_budget():
    common = dict(seed=3, max_axial_review_calls=4, interrupt=False)
    axial_only = axial_code(_corpus(), FakeAxialCoder(), **common)
    combined = axial_code(
        _corpus(),
        FakeAxialCoder(),
        also_selective=True,
        include_storyline=True,
        **common,
    )
    budget = combined.metadata["provider_budget"]
    assert "coding" in budget["stages"]
    assert budget["stages"]["coding"]["calls_used"] == axial_only.metadata["provider_budget"]["stages"]["coding"]["calls_used"]
    assert budget["calls_used"] > axial_only.metadata["provider_budget"]["calls_used"]
    llm = axial_code(
        _corpus(),
        FakeAxialCoder(),
        also_selective=True,
        include_storyline=True,
        decision_mode="llm",
        interrupt=False,
        seed=3,
        max_axial_review_calls=4,
    )
    assert "coding" in llm.metadata["provider_budget"]["stages"]
    assert llm.metadata["provider_budget"]["calls_used"] >= llm.metadata["provider_budget"]["stages"]["coding"]["calls_used"]
    review = next(row for row in llm.prompt_log if row.get("checkpoint") == "axial_saturation_review")
    indicators = json.loads(review["messages"][-1]["content"])["indicators"]
    assert "code_frequency_stabilization" not in indicators
    assert "maximal change" in indicators["code_frequency_shift_label"]
    assert llm.saturation["judgment"] is None
    assert llm.saturation["model_saturation"]["model_judgment"] is True
    assert llm.saturation["code_frequency_shift"] == llm.saturation["code_frequency_stabilization"]


def test_shift_label_and_partial_rates_stay_undefined():
    report = thematic_saturation_report(
        [
            {"document_id": "d1", "new_codes": 1, "code_frequencies": {"A": 99, "B": 1}},
            {"document_id": "d2", "new_codes": 0, "code_frequencies": {"A": 1, "B": 99}},
        ]
    )
    assert report["code_frequency_shift"] == report["code_frequency_stabilization"]
    assert "1 is maximal change" in report["code_frequency_shift_label"]
    assert "not a stabilization score" in report["note"]
    partial = _saturation(
        {
            "passages": [{"passage_id": "p1", "doc_id": "d1", "text": "a"}],
            "segments": [],
            "properties": [],
            "dimensions": [],
            "relationships": [],
            "unexamined_pairs": [{"from_id": "a", "to_id": "b"}],
            "metadata": {"pair_accounting": {"remaining_count": 4}},
        }
    )
    assert partial["new_relationship_rate"] is None
    assert partial["new_property_rate"] is None
    assert partial["new_relationship_rate_coverage"] == "undefined_partial_coverage"
    assert partial["judgment"] is None
    complete = _saturation(
        {
            "passages": [{"passage_id": "p1", "doc_id": "d1", "text": "a"}],
            "segments": [],
            "properties": [],
            "dimensions": [],
            "relationships": [],
            "unexamined_pairs": [],
            "metadata": {},
        }
    )
    assert complete["new_relationship_rate"] == 0
    assert complete["new_relationship_rate_coverage"] == "complete"


def test_empty_core_and_automated_resume_name_their_actor():
    assert _story_core({"metadata": {"decision_mode": "human", "researcher_core_id": ""}, "core_candidates": []}) == ("", "unset")
    assert _story_core({"metadata": {"decision_mode": "llm", "model_core_id": ""}, "core_candidates": []}) == ("", "unset")
    assert _story_core(
        {
            "metadata": {"decision_mode": "llm", "model_core_id": "c-prep"},
            "core_candidates": [{"code_id": "c-prep"}],
        }
    ) == ("c-prep", "model")
    saver = InMemorySaver()
    common = dict(model=FakeAxialCoder(), seed=3, checkpointer=saver, thread_id="auto-sat", interrupt=True)
    axial_code(_corpus(), **common)
    reviewed = axial_code(
        _corpus(),
        resuming=True,
        resume={"accept": True},
        **common,
    )
    assert reviewed.interrupt_payload["kind"] == "thematic_saturation"
    finished = axial_code(
        _corpus(),
        resuming=True,
        resume={"decision": "not_claimed", "rationale": "The curve was not claimed.", "automated": True},
        **common,
    )
    assert finished.saturation["researcher_saturation"]["actor"] == "automated"
    assert finished.saturation["researcher_saturation"]["decision"] == "not_claimed"
    assert finished.saturation["judgment"] is None


def test_planning_rotates_obvious_pairs_and_reports_coverage():
    ids = [f"c{index}" for index in range(8)]
    first = plan_relation_pairs(ids, round_size=3, obvious_share=1, seed=0)
    second = plan_relation_pairs(ids, round_size=3, obvious_share=1, seed=1)
    assert [(row["from_id"], row["to_id"]) for row in first["round"]] != [
        (row["from_id"], row["to_id"]) for row in second["round"]
    ]
    assert first["pair_count"] == 28
    assert first["examined_count"] == 3
    assert first["pair_coverage"] == round(3 / 28, 4)
    coded = axial_code(_corpus(), FakeAxialCoder(), seed=3, interrupt=False)
    assert coded.metadata["pair_coverage"]["pair_count"] == 1
    assert coded.metadata["pair_coverage"]["pair_coverage"] == 1.0


def test_claim_tests_require_the_subject_and_reallocation_counts_agree():
    claim = {"name": "coworkers", "pole": "valuing", "text": "shift coworkers", "quote": "coworkers"}
    assert _claim_pair_relevant(
        {"text": "the dime was on the table after the meeting"},
        claim,
        same_category=True,
        opposite=False,
    ) is False
    assert _claim_pair_relevant(
        {"text": "valuing coworkers during the shift"},
        claim,
        same_category=True,
        opposite=False,
    ) is True
    assert _claim_pair_relevant({"text": "unrelated words"}, claim, same_category=False, opposite=True) is True
    moved = _review_node(_review_state(transfers=[{"from": "claim_compete", "to": "outside", "calls": 2}]))
    note = next(item for item in moved["warnings"] if item.startswith("Focused reread purpose claim_compete"))
    assert "so 2 calls were reallocated to outside" in note
    assert "so 6 calls were reallocated" not in note
    assert moved["purpose_coverage"]["claim_compete"]["calls_moved"] == 2
    assert moved["purpose_coverage"]["reallocation"]["calls_moved"] == 2
    result = focused_code(
        _source(
            passages=[_passage("p1", "alpha bravo during the incident")],
            codes=[_entry("c1", "leaving home", 1, ["p1"])],
            segments=[_segment("s1", "c1", "p1")],
        ),
        model=FakeFocusedCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=1,
        interrupt=False,
    )
    assert result.metadata["stage_outcomes"]["focused"]["status"]
    assert result.metadata["provider_budget"]["stages"]["focused"]["max_calls"] == 1
