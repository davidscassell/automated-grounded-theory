"""0.7.3 focused-coding fixes: claim share, compete verdicts, usage, property stability."""

from __future__ import annotations

from automated_grounded_theory import __version__, focused_code, property_stability
from automated_grounded_theory.category_state import merge_properties, property_key
from automated_grounded_theory.focused import FOCUSED_COMPETE_SYSTEM, plan_recode_queue
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.schemas import (
    FocusedAlternativeReading,
    FocusedCodeVerdict,
    FocusedCompeteBatch,
    FocusedCompetePassage,
    FocusedDimensionNote,
)

from tests.fake_focused import FakeFocusedCoder
from tests.test_v070 import _book, _entry, _passages, _source
from tests.test_v072 import _passages_for_shares, _share_segments


def test_version_is_073():
    assert __version__ == "0.9.30"
    assert POLICY_VERSION == "0.9.30"


def test_claim_compete_has_a_reserved_share_and_its_own_coverage():
    passages = _passages_for_shares()[:3]
    segments = _share_segments(_passages_for_shares())[:1]
    codes = [_entry("c1", "leaving home", 1, ["d1"])]
    plan = plan_recode_queue(
        passages,
        segments,
        ["c1"],
        codes,
        seed=1,
        neighbor_limit=0,
        outside_limit=0,
        negative_case_limit=0,
        max_calls=4,
    )
    assert plan["call_targets"] == {"develop": 1, "compete": 1, "outside": 1, "claim_compete": 1}
    assert all(item["purpose"] != "claim_compete" for item in plan["queue"])

    class _PoleCoder(FakeFocusedCoder):
        def batch(self, payload):
            from automated_grounded_theory.schemas import (
                FocusedCodeDecision,
                FocusedPassageRecode,
                FocusedRecodeBatch,
            )

            rows = []
            for passage in payload.get("passages") or []:
                quote = str(passage.get("text") or "")
                code = (payload.get("raised_codes") or [{"code_id": "c1", "code": "leaving home"}])[0]
                rows.append(
                    FocusedPassageRecode(
                        passage_id=str(passage.get("passage_id") or ""),
                        codes=[
                            FocusedCodeDecision(
                                action="keep",
                                code=str(code.get("code") or ""),
                                code_id=str(code.get("code_id") or ""),
                                quote=quote,
                            )
                        ],
                        dimensions=[
                            FocusedDimensionNote(
                                category_id=str(code.get("code_id") or ""),
                                name="leaving",
                                pole="staying",
                                text="The other pole is staying.",
                                document_id=str(passage.get("doc_id") or ""),
                                passage_id=str(passage.get("passage_id") or ""),
                                quote=quote,
                            )
                        ],
                    )
                )
            return FocusedRecodeBatch(passages=rows)

    result = focused_code(
        _source(passages=passages, codes=codes, segments=segments),
        model=_PoleCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        max_focused_review_calls=4,
        interrupt=False,
    )
    claim = result.purpose_coverage["claim_compete"]
    assert claim["calls"] >= 1
    assert "p2" in claim["passage_ids"]
    assert claim["requested_share"] == 0.25
    assert claim["applied_share"] > 0
    assert "p2" in claim["reread"]

    starved = focused_code(
        _source(passages=passages, codes=codes, segments=segments),
        model=_PoleCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=1,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        max_focused_review_calls=1,
        interrupt=False,
    )
    assert starved.purpose_coverage["claim_compete"]["calls"] == 0
    assert starved.purpose_coverage["claim_compete"]["reserved_calls"] == 0


def test_compete_verdicts_name_an_alternative_and_do_not_apply_drop():
    assert "equal footing" in FOCUSED_COMPETE_SYSTEM
    assert "strongest_alternative" in FOCUSED_COMPETE_SYSTEM
    assert "revise" in FOCUSED_COMPETE_SYSTEM and "drop" in FOCUSED_COMPETE_SYSTEM
    assert FocusedCodeVerdict.model_fields["verdict"].is_required()
    assert FocusedCodeVerdict.model_fields["rationale"].is_required()
    assert FocusedCompetePassage.model_fields["strongest_alternative"].is_required()
    assert FocusedCompetePassage.model_fields["verdicts"].is_required()

    passages = _passages()[:1]
    text = passages[0]["text"]

    class _VerdictCoder(FakeFocusedCoder):
        def compete_batch(self, payload):
            assert "raised_codes" not in payload
            assert "other_codes" not in payload
            assert payload["codes"][0]["code_id"] == "c-rare"
            passage = payload["passages"][0]
            initial = passage["initial_codes"]
            assert {item["code_id"] for item in initial} >= {"c-freq", "c-rare"}
            return FocusedCompeteBatch(
                passages=[
                    FocusedCompetePassage(
                        passage_id=passage["passage_id"],
                        verdicts=[
                            FocusedCodeVerdict(
                                code_id="c-freq",
                                code="packing bags",
                                verdict="drop",
                                rationale="A disconfirming reading is that this is hiring, not packing.",
                                quote=passage["text"],
                            ),
                            FocusedCodeVerdict(
                                code_id="c-freq",
                                code="packing bags",
                                verdict="revise",
                                rationale="If it stays, the definition should name the morning.",
                                revised_definition="Morning packing before a departure.",
                                quote=passage["text"],
                            ),
                        ],
                        strongest_alternative=FocusedAlternativeReading(
                            code="writing letters",
                            code_id="c-rare",
                            rationale="Writing letters is the stronger reading and was not chosen.",
                            alternative_fit="stronger",
                            quote=passage["text"],
                        ),
                        memo="writing letters fits this passage better than packing bags.",
                    )
                ]
            )

    dropped = focused_code(
        _source(
            passages=passages,
            codes=[_book()[0], _book()[1]],
            segments=[
                {
                    "segment_id": "s1",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": text,
                    "review_reasons": [],
                },
                {
                    "segment_id": "s2",
                    "code_id": "c-rare",
                    "code": "writing letters",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": text,
                    "review_reasons": [],
                },
            ],
        ),
        model=_VerdictCoder(),
        seed=1,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0,
        focused_share_compete=1,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        max_focused_review_calls=1,
        interrupt=False,
    )
    segment = next(item for item in dropped.segments if item["segment_id"] == "s1")
    assert segment["quote"] == text
    assert "focused_drop_proposal" in segment["review_reasons"]
    assert "focused_revise_proposal" in segment["review_reasons"]
    assert segment["focused_drop_proposal"]["applied"] is False
    assert segment["focused_revise_proposal"]["applied"] is False
    assert "focused_disagreement" not in segment["review_reasons"]
    entry = next(item for item in dropped.codebook if item["code_id"] == "c-freq")
    assert entry["definition"] == "Incidents of packing bags."

    class _KeepCoder(FakeFocusedCoder):
        def compete_batch(self, payload):
            passage = payload["passages"][0]
            return FocusedCompeteBatch(
                passages=[
                    FocusedCompetePassage(
                        passage_id=passage["passage_id"],
                        verdicts=[
                            FocusedCodeVerdict(
                                code_id="c-freq",
                                code="packing bags",
                                verdict="keep",
                                rationale="Kept after considering a disconfirming reading.",
                                quote=passage["text"],
                            )
                        ],
                        strongest_alternative=FocusedAlternativeReading(
                            code="marking the hour",
                            code_id="c-hour",
                            rationale="Marking the hour fits this passage better than the kept code.",
                            alternative_fit="stronger",
                            quote=passage["text"],
                        ),
                        memo="The memo names writing letters as the alternative.",
                    )
                ]
            )

    kept = focused_code(
        _source(
            passages=passages,
            codes=[_book()[0], _book()[1], _entry("c-hour", "marking the hour", 1, ["d9"])],
            segments=[
                {
                    "segment_id": "s1",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": text,
                    "review_reasons": [],
                },
                {
                    "segment_id": "s2",
                    "code_id": "c-rare",
                    "code": "writing letters",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": text,
                    "review_reasons": [],
                },
            ],
        ),
        model=_KeepCoder(),
        seed=1,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=0,
        focused_share_compete=1,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        max_focused_review_calls=1,
        interrupt=False,
    )
    kept_segment = next(item for item in kept.segments if item["segment_id"] == "s1")
    assert "alternative_named_not_chosen" in kept_segment["review_reasons"]
    assert "alternative_is_kept_code" not in kept_segment["review_reasons"]
    named = kept_segment["alternative_named_not_chosen"]["alternatives"]
    assert any(item["code_id"] == "c-hour" for item in named)
    assert any(item["source"] == "strongest_alternative" for item in named)
    assert all(item["alternative_fit"] == "stronger" for item in named)
    assert kept_segment["alternative_named_not_chosen"]["applied"] is False


def test_provider_usage_is_separate_from_the_estimate():
    passages = _passages()[:1]
    coder = FakeFocusedCoder(
        reported_usage={
            "input_tokens": 40,
            "output_tokens": 8,
            "total_tokens": 48,
            "input_token_details": {"cache_read": 10},
        }
    )
    result = focused_code(
        _source(
            passages=passages,
            codes=[_book()[0]],
            segments=[
                {
                    "segment_id": "s1",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": passages[0]["text"],
                    "review_reasons": [],
                }
            ],
        ),
        model=coder,
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=1,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        max_focused_review_calls=1,
        interrupt=False,
    )
    usage = result.metadata["provider_token_usage"]
    assert usage["input_tokens"] == 40
    assert usage["cached_tokens"] == 10
    assert usage["output_tokens"] == 8
    assert usage["total_tokens"] == 48
    assert result.metadata["focused_tokens_used"] == 48
    assert result.metadata["focused_tokens_estimated"] > result.metadata["focused_tokens_used"]
    assert "heuristic" in result.metadata["token_accounting_note"]
    assert "not provider usage" in result.metadata["token_accounting_note"]


def test_property_keys_merge_and_stability_is_category_and_name():
    assert property_key("Timing of the departure") == property_key("departure timings")
    assert property_key("Timing of the departure") != property_key("pace")
    merged = merge_properties(
        [
            {"kind": "property", "category_id": "c1", "name": "Timing of the departure", "passage_id": "p1", "quote": "a"},
            {"kind": "property", "category_id": "c1", "name": "departure timings", "passage_id": "p2", "quote": "b"},
            {"kind": "property", "category_id": "c1", "name": "pace", "passage_id": "p3", "quote": "c"},
        ]
    )
    assert len(merged) == 2
    timing = next(item for item in merged if item["property_key"] == property_key("departure timing"))
    assert len(timing["citations"]) == 2
    assert timing["merged_names"] == ["Timing of the departure", "departure timings"]

    same = property_stability(
        [
            [{"category_id": "c1", "name": "Timing of the departure", "property_key": property_key("Timing of the departure")}],
            [{"category_id": "c1", "name": "departure timings", "property_key": property_key("departure timings")}],
        ]
    )
    assert same["category_level"] == 1
    assert same["name_level"] == 1
    shifted = property_stability(
        [
            [{"category_id": "c1", "name": "timing", "property_key": "timing"}],
            [{"category_id": "c1", "name": "pace", "property_key": "pace"}],
        ]
    )
    assert shifted["category_level"] == 1
    assert shifted["name_level"] == 0
    other_category = property_stability(
        [
            [{"category_id": "c1", "name": "timing", "property_key": "timing"}],
            [{"category_id": "c2", "name": "timing", "property_key": "timing"}],
        ]
    )
    assert other_category["category_level"] == 0
    assert other_category["name_level"] == 0

    class _NamingCoder(FakeFocusedCoder):
        def batch(self, payload):
            from automated_grounded_theory.schemas import (
                FocusedCodeDecision,
                FocusedPassageRecode,
                FocusedPropertyNote,
                FocusedRecodeBatch,
            )

            passage = payload["passages"][0]
            name = "Departure timing" if passage["passage_id"] == "p1" else "timing of the departure"
            return FocusedRecodeBatch(
                passages=[
                    FocusedPassageRecode(
                        passage_id=passage["passage_id"],
                        codes=[
                            FocusedCodeDecision(
                                action="keep",
                                code="packing bags",
                                code_id="c-freq",
                                quote=passage["text"],
                            )
                        ],
                        properties=[
                            FocusedPropertyNote(
                                category_id="c-freq",
                                name=name,
                                text="When the packing sits.",
                                document_id=passage["doc_id"],
                                passage_id=passage["passage_id"],
                                quote=passage["text"],
                            )
                        ],
                    )
                ]
            )

    passages = _passages()[:2]
    result = focused_code(
        _source(
            passages=passages,
            codes=[_book()[0]],
            segments=[
                {
                    "segment_id": "s1",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": passages[0]["text"],
                    "review_reasons": [],
                },
                {
                    "segment_id": "s2",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p2",
                    "doc_id": "d2",
                    "quote": passages[1]["text"],
                    "review_reasons": [],
                },
            ],
        ),
        model=_NamingCoder(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=1,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        max_focused_review_calls=2,
        interrupt=False,
    )
    report = result.metadata["property_stability"]
    assert report["category_level"] is None
    assert report["name_level"] is None
    assert report["raw_properties"] == 2
    assert report["merged_properties"] == 1
    assert report["property_keys"] == [property_key("departure timing")]
    assert "category_id, property_key" in report["note"]
    assert len(result.properties) == 1
    assert len(result.properties[0]["citations"]) == 2
    compared = property_stability([result.properties, result.properties])
    assert compared["category_level"] == 1
    assert compared["name_level"] == 1
