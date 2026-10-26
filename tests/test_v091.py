"""0.9.1 relationship evidence tiers. No network calls."""

from __future__ import annotations

import json

from automated_grounded_theory import __version__, axial_code, selective_code
from automated_grounded_theory.axial import (
    AxialError,
    AxialProposalBatch,
    AxialRelationshipDraft,
    StorylineBatch,
    StorylineStatementDraft,
    _apply_human_relations,
    _apply_model_relations,
    _commit_relations,
    _coverage,
    _memo_flags,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import (
    PATTERN_MIN_CASES_NOTE,
    apply_evidence_tier,
    assess_relationship,
    candidate_cross_case_passages,
    promote_verified,
    wording_overreach,
)
from tests.test_v080 import QUOTE, FakeAxialCoder, _corpus


def _book() -> dict:
    return {
        "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
        "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
    }


def _passages() -> dict:
    shared = "The tools for the shift sit beside coworkers on the crew tonight"
    return {
        "p1": {
            "passage_id": "p1",
            "doc_id": "d1",
            "respondent_id": "r1",
            "text": shared,
        },
        "p2": {
            "passage_id": "p2",
            "doc_id": "d2",
            "respondent_id": "r2",
            "text": shared,
        },
        "p3": {
            "passage_id": "p3",
            "doc_id": "d3",
            "respondent_id": "r3",
            "text": "The schedule changed and nobody mentioned either side.",
        },
    }


def _row(**extra) -> dict:
    row = {
        "relationship_id": "rel-0001",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": "",
        "quote_from": "tools for the shift sit",
        "quote_to": "coworkers on the crew tonight",
        "statement": "tools for the shift may be related to coworkers on the crew",
        "disposition": "proposed",
        "flags": [],
        "evidence_condition": "anchored",
        "counterexamples": [],
        "case_quotes": [],
    }
    row.update(extra)
    return row


def _both_cases(row: dict) -> dict:
    row["case_quotes"] = [
        {"passage_id": "p1", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p1", "quote": "coworkers on the crew tonight", "endpoint": "to"},
        {"passage_id": "p2", "quote": "tools for the shift sit", "endpoint": "from"},
        {"passage_id": "p2", "quote": "coworkers on the crew tonight", "endpoint": "to"},
    ]
    return row


def test_version_is_091():
    assert __version__ == "0.9.38"
    assert POLICY_VERSION == "0.9.38"


def test_tier_zero_is_category_evidence_and_a_label_is_not_a_quote():
    passages = _passages()
    one_side = _row(quote_to="", quote_from="tools for the shift sit", statement="tools for the shift are present")
    fields = apply_evidence_tier(one_side, passages=passages, codebook=_book(), pattern_min_cases=2)
    assert fields["evidence_tier"] == 0
    assert fields["evidence_label"] == "category_evidence"
    assert fields["allowed_wording"] == "a category is present"
    assert one_side["evidence_status"] == "category_evidence"
    label = _row(
        quote_from="tools for the shift",
        quote_to="coworkers on the crew",
        case_quotes=[{"passage_id": "p1", "quote": "tools for the shift", "endpoint": "from"}],
    )
    # The category label alone, even when it sits in the passage, is not endpoint evidence.
    label["quote_from"] = "tools for the shift"
    bare = {
        "relationship_id": "rel-label",
        "from_id": "tools",
        "to_id": "crew",
        "passage_id": "p1",
        "quote": "tools for the shift",
        "quote_from": "",
        "quote_to": "",
        "statement": "a label",
        "flags": [],
        "case_quotes": [],
    }
    bare_fields = assess_relationship(bare, passages=passages, codebook=_book())
    assert bare_fields["evidence_tier"] is None
    assert bare_fields["supporting_cases"] is None
    missing = assess_relationship(_row(), passages=None, codebook=_book())
    assert missing["evidence_tier"] is None
    assert missing["cooccurrence_2x2"]["both"] is None
    assert missing["negative_case_search_report"] == "undefined"
    assert missing["variation_search"] == "undefined"


def test_cross_respondent_stitch_is_a_hypothesis_and_not_supported():
    book = _book()
    passages = {
        "p-a": {
            "passage_id": "p-a",
            "doc_id": "d-a",
            "respondent_id": "r-a",
            "text": "tools for the shift were late",
        },
        "p-b": {
            "passage_id": "p-b",
            "doc_id": "d-b",
            "respondent_id": "r-b",
            "text": "coworkers on the crew stayed",
        },
    }
    row = _row(
        passage_id="p-a",
        quote_from="tools for the shift were late",
        quote_to="coworkers on the crew stayed",
        statement="tools for the shift leads to coworkers on the crew",
        evidence_scope="cross_respondent",
        quotes_share_passage=False,
        endpoint_anchoring=[
            {"endpoint": "from", "anchored": True},
            {"endpoint": "to", "anchored": True},
        ],
    )
    decided = _apply_model_relations(
        [row],
        {
            "relationships": [
                {
                    "relationship_id": "rel-0001",
                    "disposition": "accepted",
                    "paradigm_role": "consequence",
                    "rationale": "The quotes are in the same passage.",
                }
            ]
        },
        {},
        passages=passages,
        codebook=book,
    )[0]
    assert decided["evidence_tier"] == 1
    assert decided["evidence_label"] == "hypothesis"
    assert decided["evidence_status"] == "hypothesis"
    assert decided["disposition"] == "provisional"
    assert decided["allowed_wording"] == "may be related"
    assert decided["cross_case"] is True
    assert "cross_case" in decided["flags"]
    assert "hypothesis_not_supported" in decided["flags"]
    assert "acceptance_withheld" in decided["flags"]
    assert "wording_exceeds_tier" in decided["flags"]
    assert "reviewer_same_passage_mismatch" in decided["flags"]
    assert decided["downstream_eligible"] is False
    assert decided["model_disposition"] == "provisional"
    coverage = _coverage({"codebook": book, "relationships": [decided], "storyline": [], "metadata": {}, "empty_slots": []})
    assert coverage["citation_anchoring_rate"] is None
    assert coverage["citation_anchoring_cross_passage"] is None
    assert coverage["citation_anchoring_by_tier"]["hypothesis"]["n"] == 1
    assert coverage["citation_anchoring_by_tier"]["hypothesis"]["citation_rate"] is None
    assert coverage["citation_anchoring_by_tier"]["pattern_supported"]["citation_rate"] is None
    assert wording_overreach("these may be related", 1) is False
    assert wording_overreach("this is supported", 1) is True
    assert wording_overreach("tools lead to coworkers", 1) is True


def test_two_cases_accept_as_pattern_supported_and_not_a_cause():
    book = _book()
    passages = _passages()
    row = _both_cases(_row(statement="tools for the shift lead to coworkers on the crew"))
    fields = apply_evidence_tier(row, passages=passages, codebook=book, pattern_min_cases=2)
    assert fields["evidence_tier"] == 2
    assert fields["evidence_label"] == "pattern_supported"
    assert fields["allowed_wording"] == "co-occurs in 2 respondents"
    assert fields["within_case_strength"] == "same_passage"
    assert fields["cases_with_both_endpoints"] == 2
    assert fields["pattern_min_cases"] == 2
    assert fields["pattern_min_cases_note"] == PATTERN_MIN_CASES_NOTE
    assert fields["negative_case_search_report"] == "undefined"
    assert fields["variation_search"] == "undefined"
    assert fields["rival_explanation_check"] == "undefined"
    assert fields["cooccurrence_2x2"]["descriptive"] is True
    assert fields["cooccurrence_2x2"]["both"] == 2
    cases = fields["supporting_cases"]
    assert {case["case_id"] for case in cases if case["both_endpoints"]} == {"r1", "r2"}
    for case in cases:
        if not case["both_endpoints"]:
            continue
        ends = {item["endpoint"] for item in case["quotes"]}
        assert ends == {"from", "to"}
        assert case["passage_ids"]
        assert all(item["verbatim"] and item["anchored"] for item in case["quotes"])
    decided = _apply_human_relations(
        [row],
        {"relationships": [{"relationship_id": "rel-0001", "action": "accept", "paradigm_role": "context"}]},
        passages,
        codebook=book,
    )[0]
    assert decided["disposition"] == "accepted"
    assert decided["evidence_status"] == "pattern_supported"
    assert decided["evidence_tier"] == 2
    assert decided["human_disposition"] == "accepted"
    assert "wording_exceeds_tier" in decided["flags"]
    assert decided["evidence_status"] != "verified"
    coverage = _coverage({"codebook": book, "relationships": [decided], "storyline": [], "metadata": {}, "empty_slots": []})
    assert coverage["citation_anchoring_rate"] == 1.0
    assert coverage["citation_anchoring_by_tier"]["pattern_supported"]["n"] == 1
    assert coverage["citation_anchoring_by_tier"]["pattern_supported"]["citation_rate"] == 1.0
    assert coverage["citation_anchoring_same_passage"] == 1.0
    assert coverage["citation_anchoring_cross_passage"] is None
    assert coverage["citation_anchoring_by_tier"]["verified"]["citation_rate"] is None


def test_survey_segments_are_one_case_when_they_share_a_respondent():
    book = {
        "prep": {"code_id": "prep", "code": "preparing the handover", "definition": "preparing the handover"},
        "back": {"code_id": "back", "code": "returning after leave", "definition": "returning after leave"},
    }
    passages = {
        "q1": {
            "passage_id": "q1",
            "doc_id": "d1",
            "respondent_id": "r1",
            "text": "I was preparing the handover notes before the trip.",
        },
        "q2": {
            "passage_id": "q2",
            "doc_id": "d1",
            "respondent_id": "r1",
            "text": "I was returning after leave on Monday.",
        },
        "q3": {
            "passage_id": "q3",
            "doc_id": "d2",
            "respondent_id": "r2",
            "text": "The meeting was virtual.",
        },
    }
    segments = [
        {"segment_id": "s1", "code_id": "prep", "passage_id": "q1", "quote": "preparing the handover notes"},
        {"segment_id": "s2", "code_id": "back", "passage_id": "q2", "quote": "returning after leave on Monday"},
    ]
    row = {
        "relationship_id": "rel-survey",
        "from_id": "prep",
        "to_id": "back",
        "passage_id": "q1",
        "quote": "",
        "quote_from": "preparing the handover notes",
        "quote_to": "returning after leave on Monday",
        "statement": "preparing the handover may be related to returning after leave",
        "flags": [],
        "case_quotes": [],
        "counterexamples": [],
    }
    fields = apply_evidence_tier(row, passages=passages, codebook=book, segments=segments, pattern_min_cases=2)
    assert fields["evidence_tier"] == 1
    assert fields["within_case_strength"] == "within_respondent"
    assert fields["cases_with_both_endpoints"] == 1
    both = [case for case in fields["supporting_cases"] if case["both_endpoints"]]
    assert both[0]["case_id"] == "r1"
    assert both[0]["same_passage"] is False
    assert set(both[0]["passage_ids"]) == {"q1", "q2"}
    counts = fields["cooccurrence_2x2"]
    assert counts["both"] == 1
    assert counts["a_only"] == 0
    assert counts["b_only"] == 0
    assert counts["neither"] == 1
    assert "validity" in counts["note"]


def test_attribution_needs_a_rival_check_and_only_then_allows_a_cause():
    book = _book()
    passages = _passages()
    quote = "tools for the shift lead to coworkers on the crew because the night ran long"
    passages["p1"]["text"] = passages["p1"]["text"] + " " + quote
    passages["p2"]["text"] = passages["p2"]["text"] + " " + quote
    row = _both_cases(_row(statement="tools for the shift lead to coworkers on the crew"))
    row["case_quotes"].append({"passage_id": "p1", "quote": quote, "endpoint": "both"})
    row["case_quotes"].append({"passage_id": "p2", "quote": quote, "endpoint": "both"})
    early = assess_relationship(row, passages=passages, codebook=book)
    assert early["evidence_tier"] == 2
    assert early["rival_explanation_check"] == "undefined"
    assert early["attribution_source"] == "participant"
    row["counterexamples"] = [
        {
            "passage_id": "p3",
            "judgment": "contradicts",
            "valid_test": True,
            "synthetic": False,
            "quote": "The schedule changed",
        }
    ]
    later = apply_evidence_tier(dict(row), passages=passages, codebook=book)
    assert later["evidence_tier"] == 3
    assert later["evidence_label"] == "attribution"
    assert later["allowed_wording"] == "participants attribute X to Y"
    assert later["rival_explanation_check"] == "reported"
    assert wording_overreach(row["statement"], 3) is False
    assert wording_overreach("this link is verified", 3) is True
    memo_row = dict(row)
    memo_row["counterexamples"] = []
    memo_row["case_quotes"] = [item for item in row["case_quotes"] if item["endpoint"] != "both"]
    memo_only = assess_relationship(
        memo_row,
        passages=passages,
        codebook=book,
        memos=[{"relationship_id": "rel-0001", "text": "The tools for the shift quote shows why coworkers on the crew changed."}],
    )
    assert memo_only["attribution_source"] is None
    assert memo_only["evidence_tier"] == 2
    assert memo_only["allowed_wording"] == "co-occurs in 2 respondents"
    flags = _memo_flags("tools for the shift lead to coworkers on the crew.", {**row, "evidence_tier": 2, "quote": quote})
    assert "memo_wording_exceeds_tier" in flags
    quiet = _memo_flags("The evidence on this side is partial.", {"quote": QUOTE, "endpoint_anchoring": [{"anchored": True}, {"anchored": False}]})
    assert "memo_wording_exceeds_tier" not in quiet
    assert "memo_degenerate" not in quiet
    assert "memo_missing_quote" not in quiet


def test_human_can_verify_and_the_model_cannot():
    book = _book()
    passages = _passages()
    quote = "tools for the shift lead to coworkers on the crew because the night ran long"
    passages["p1"]["text"] = passages["p1"]["text"] + " " + quote
    passages["p2"]["text"] = passages["p2"]["text"] + " " + quote
    base = _both_cases(
        _row(
            statement="participants attribute the crew to the tools for the shift",
            counterexamples=[
                {"passage_id": "p3", "judgment": "qualifies", "valid_test": True, "synthetic": False, "quote": "schedule changed"}
            ],
        )
    )
    base["case_quotes"].append({"passage_id": "p1", "quote": quote, "endpoint": "both"})
    base["case_quotes"].append({"passage_id": "p2", "quote": quote, "endpoint": "both"})
    try:
        _apply_human_relations(
            [dict(base)],
            {"relationships": [{"relationship_id": "rel-0001", "action": "verify"}]},
            passages,
            codebook=book,
        )
    except AxialError as exc:
        assert "scope" in str(exc)
    else:
        raise AssertionError("verify without a scope should fail")
    verified = _apply_human_relations(
        [dict(base)],
        {"relationships": [{"relationship_id": "rel-0001", "action": "verify", "scope": "these two respondents"}]},
        passages,
        codebook=book,
    )[0]
    assert verified["evidence_tier"] == 4
    assert verified["evidence_status"] == "verified"
    assert verified["human_disposition"] == "verified"
    assert verified["disposition"] == "accepted"
    assert verified["verified_scope"] == "these two respondents"
    assert verified["verification_actor"] == "human"
    assert "verified" in verified["flags"]
    low = _row(statement="tools for the shift may be related to coworkers on the crew")
    withheld = _apply_human_relations(
        [low],
        {"relationships": [{"relationship_id": "rel-0001", "action": "verify", "scope": "a later round"}]},
        passages,
        codebook=book,
    )[0]
    assert withheld["evidence_tier"] == 1
    assert "verification_withheld" in withheld["flags"]
    assert withheld["evidence_status"] != "verified"
    refused = _apply_model_relations(
        [dict(base)],
        {
            "relationships": [
                {
                    "relationship_id": "rel-0001",
                    "disposition": "verified",
                    "paradigm_role": "consequence",
                    "rationale": "The pattern is enough to verify.",
                }
            ]
        },
        {},
        passages=passages,
        codebook=book,
    )[0]
    assert refused["evidence_tier"] == 3
    assert refused["evidence_status"] == "attribution"
    assert "model_cannot_verify" in refused["flags"]
    assert refused.get("verified_scope") in {None, ""}
    outcome = promote_verified(dict(base), actor="llm", scope="held out interviews")
    assert outcome == "refused"
    held = dict(base)
    apply_evidence_tier(held, passages=passages, codebook=book)
    held["counterexamples"] = []
    held["verification_requested"] = True
    held["verification_scope"] = "the next three interviews"
    held["verification_actor"] = "human"
    # The rival check is not on the row yet, so commit must not verify it.
    early = _commit_relations(
        {
            "relationships": [dict(held)],
            "passages": list(passages.values()),
            "codebook": book,
            "segments": [],
            "memos": [],
            "metadata": {"procedure": {"pattern_min_cases": 2, "min_relation_documents": 2, "coding_lens": "paradigm"}},
            "warnings": [],
        }
    )
    assert early["relationships"][0]["evidence_tier"] == 2
    assert "verification_withheld" in early["relationships"][0]["flags"]
    held["counterexamples"] = [
        {"passage_id": "p3", "judgment": "contradicts", "valid_test": True, "synthetic": False, "quote": "schedule changed"}
    ]
    promoted = _commit_relations(
        {
            "relationships": [held],
            "passages": list(passages.values()),
            "codebook": book,
            "segments": [],
            "memos": [],
            "metadata": {"procedure": {"pattern_min_cases": 2, "min_relation_documents": 2, "coding_lens": "paradigm"}},
            "warnings": [],
        }
    )
    assert promoted["relationships"][0]["evidence_tier"] == 4
    assert promoted["relationships"][0]["evidence_status"] == "verified"
    assert promoted["relationships"][0]["verified_scope"] == "the next three interviews"


def test_storyline_flags_wording_above_the_tier():
    coded = axial_code(_corpus(), FakeAxialCoder(), seed=3, interrupt=False, max_axial_review_calls=2)
    coded.relationships[0]["disposition"] = "accepted"
    coded.relationships[0]["evidence_status"] = "pattern_supported"
    coded.relationships[0]["evidence_tier"] = 2
    coded.relationships[0]["evidence_label"] = "pattern_supported"
    coded.relationships[0]["allowed_wording"] = "co-occurs in 2 respondents"

    class CausalStory(FakeAxialCoder):
        def storyline(self, body):
            del body
            return StorylineBatch(
                statements=[
                    StorylineStatementDraft(
                        text="Preparing to leave leads to returning to work.",
                        category_ids=["c-prep", "c-return"],
                        relationship_ids=["rel-0001"],
                        passage_id="p1",
                        quote=QUOTE,
                    )
                ]
            )

    finished = selective_code(coded, CausalStory(), include_storyline=True, interrupt=False)
    flags = finished.storyline[0]["flags"]
    assert "storyline_wording_exceeds_tier" in flags
    assert "storyline_unaccepted_relationship" in flags
    coded.relationships[0]["evidence_tier"] = 3
    coded.relationships[0]["evidence_status"] = "attribution"
    coded.relationships[0]["evidence_label"] = "attribution"
    allowed = selective_code(coded, CausalStory(), include_storyline=True, interrupt=False)
    assert "storyline_wording_exceeds_tier" not in allowed.storyline[0]["flags"]
    assert "storyline_unaccepted_relationship" not in allowed.storyline[0]["flags"]


def test_cross_case_candidates_stay_inside_the_evidence_budget():
    passages = _passages()
    passages["p4"] = {
        "passage_id": "p4",
        "doc_id": "d4",
        "respondent_id": "r4",
        "text": "tools for the shift and coworkers on the crew appear together here",
    }
    none = candidate_cross_case_passages(
        passages,
        _book(),
        [{"from_id": "tools", "to_id": "crew"}],
        excluded_ids={"p1"},
        selected_case_ids={"r1"},
        limit=0,
    )
    assert none == []
    found = candidate_cross_case_passages(
        passages,
        _book(),
        [{"from_id": "tools", "to_id": "crew"}],
        excluded_ids={"p1", "p2"},
        selected_case_ids={"r1", "r2"},
        limit=1,
    )
    assert [row["passage_id"] for row in found] == ["p4"]
    assert found[0]["candidate_cross_case"] is True
    corpus = _corpus()
    # 0.9.10 offers a lexical cross-case passage only when each endpoint shares
    # at least two distinctive stems. "returning to work" has one, so the
    # return category names the desk as well.
    corpus["codebook"]["c-return"] = {
        "code_id": "c-return",
        "code": "returning to the desk",
        "definition": "returning to the desk",
    }
    corpus["passages"].append(
        {
            "passage_id": "p-cross",
            "doc_id": "d-cross",
            "respondent_id": "r-cross",
            "text": "Someone was preparing to leave and then returning to the desk.",
        }
    )
    corpus["segments"] = [
        {"segment_id": "s1", "code_id": "c-prep", "passage_id": "p1"},
        {"segment_id": "s2", "code_id": "c-return", "passage_id": "p1"},
    ]

    class Recording(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or "c-prep"),
                        to_id=str(pair.get("to_id") or "c-return"),
                        paradigm_role="context",
                        statement="Packing the suitcase before departure shapes the return to work.",
                        passage_id="p1",
                        quote=QUOTE,
                    )
                ]
            )

    result = axial_code(corpus, Recording(), seed=3, max_axial_review_calls=2, axial_evidence_limit=4, interrupt=False)
    proposal = next(row for row in result.prompt_log if row.get("kind") == "axial_proposal" and row.get("status") != "deferred")
    body = json.loads(proposal["messages"][-1]["content"])
    assert body["candidate_cross_case"]
    assert "p-cross" in {row["passage_id"] for row in body["candidate_cross_case"]}
    assert "candidate_cross_case" in proposal["messages"][0]["content"]
    sent = {row["passage_id"] for row in body["passages"]}
    assert "p-cross" in sent
    assert result.metadata["procedure"]["pattern_min_cases"] == 2
    assert "design choice" in result.metadata["procedure"]["pattern_min_cases_note"]
    tight = axial_code(
        corpus,
        Recording(),
        seed=3,
        max_axial_review_calls=2,
        axial_evidence_limit=1,
        interrupt=False,
    )
    tight_body = json.loads(
        next(row for row in tight.prompt_log if row.get("kind") == "axial_proposal")["messages"][-1]["content"]
    )
    assert tight_body["candidate_cross_case"] == []
    assert tight_body["candidate_same_passage_cooccurrence"][0]["passage_id"] == "p1"
