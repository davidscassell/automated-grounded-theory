"""0.6.3: split joined labels, retry bad ones once, and keep them flagged."""

import json

from automated_grounded_theory import open_code
from automated_grounded_theory.codebook import interpret_coding
from automated_grounded_theory.prompts import validation_feedback
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.schemas import CodeProposal, PassageCoding
from automated_grounded_theory.validation import (
    EMPTY_LABEL,
    bundled_concepts,
    split_joined_label,
)
from automated_grounded_theory.version import __version__
from tests.corpus import CONTEXT
from tests.fake_llm import FakeCoder

TEXT = "The form has too many steps, but my coworkers make the day easier."
QUOTE = "The form has too many steps"
CATCHALL = "valuing flexibility, low friction, tangible benefits, and coworkers"
SECOND = "criticizing procedural friction"
AND_JOINED = "seeking flexibility and avoiding friction and praising coworkers"
TOO_LONG = (
    "describing the many procedural steps involved in filing a routine request for time away from the team"
)


def test_version_is_0_6_3():
    assert __version__ == "0.9.36"
    assert POLICY_VERSION == "0.9.36"


def test_split_and_bundle_detectors():
    assert split_joined_label(f"  {CATCHALL} ; {SECOND} ; ") == [CATCHALL, SECOND]
    assert split_joined_label(";;;") == [""]
    assert split_joined_label("packing bags") == ["packing bags"]
    assert bundled_concepts(CATCHALL)
    assert bundled_concepts(AND_JOINED)
    assert bundled_concepts("pay, benefits, and time")
    assert not bundled_concepts("pay and benefits")
    assert not bundled_concepts("asking for pay, then leaving")
    assert not bundled_concepts("packing bags")
    assert not bundled_concepts("")


def test_feedback_explains_each_label_problem():
    text = validation_feedback(
        [
            {"code": "", "reason": "empty_code"},
            {"code": TOO_LONG, "reason": "code_too_long"},
            {"code": CATCHALL, "reason": "bundled_concepts", "reasons": ["bundled_concepts"]},
        ]
    )
    assert "Do not leave the code blank" in text
    assert "The label is too long" in text
    assert "Return a single concept" in text


def _interpret(codes, book=None):
    return interpret_coding(
        {"codes": codes, "memo": ""},
        {"text": TEXT, "passage_id": "p1"},
        book or {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
    )


def _proposal(label: str) -> dict:
    return {
        "action": "new",
        "code": label,
        "definition": "Incidents named by this label.",
        "quote": QUOTE,
        "confidence": 0.9,
        "rationale": "The passage shows this incident.",
    }


def test_semicolon_split_reuses_the_existing_code():
    book = {"catchall": {"code": CATCHALL, "key": CATCHALL.casefold()}}
    interpreted = _interpret([_proposal(f"{CATCHALL}; {SECOND}")], book)
    accepted = interpreted["accepted"]
    assert [(item["action"], item["code"]) for item in accepted] == [
        ("reuse", CATCHALL),
        ("new", SECOND),
    ]
    assert interpreted["label_retries"] == []
    assert not any(item["reason"] == "code_too_long" for item in interpreted["rejections"])
    assert "bundled_concepts" not in accepted[0]["review_reasons"]


def test_semicolon_split_keeps_a_new_second_concept():
    interpreted = _interpret([_proposal(f"{CATCHALL}; {SECOND}")])
    codes = [item["code"] for item in interpreted["accepted"]]
    assert codes == [CATCHALL, SECOND]
    assert not any(item["reason"] == "code_too_long" for item in interpreted["rejections"])
    bundled = interpreted["label_retries"]
    assert [item["reason"] for item in bundled] == ["bundled_concepts"]
    assert "bundled_concepts" in interpreted["accepted"][0]["review_reasons"]
    assert "code_too_long" not in interpreted["accepted"][1]["review_reasons"]


def test_and_joined_new_code_is_a_bundle_and_is_kept():
    interpreted = _interpret([_proposal(AND_JOINED)])
    assert interpreted["accepted"][0]["code"] == AND_JOINED
    assert interpreted["label_retries"][0]["reason"] == "bundled_concepts"
    assert "bundled_concepts" in interpreted["accepted"][0]["review_reasons"]
    assert interpreted["rejections"] == []


def test_empty_and_too_long_labels_are_kept_for_review():
    empty = _interpret([_proposal("   ")])
    assert empty["accepted"][0]["code"] == EMPTY_LABEL
    assert empty["label_retries"][0]["reason"] == "empty_code"
    assert "empty_code" in empty["accepted"][0]["review_reasons"]
    assert empty["accepted"][0]["needs_review"]
    assert not any(item["reason"] == "empty_code" for item in empty["rejections"])

    long = _interpret([_proposal(TOO_LONG)])
    assert long["accepted"][0]["code"] == TOO_LONG
    assert long["label_retries"][0]["reason"] == "code_too_long"
    assert "code_too_long" in long["accepted"][0]["review_reasons"]
    assert long["rejections"] == []


def test_generic_labels_are_still_rejected():
    interpreted = _interpret([_proposal("experience")])
    assert interpreted["accepted"] == []
    assert interpreted["rejections"][0]["reason"] == "generic_code"
    assert interpreted["label_retries"] == []


def test_too_long_label_that_matches_an_existing_code_is_reused():
    label = CATCHALL + " today"
    book = {"row": {"code": label, "key": label.casefold(), "gerund_ok": True, "in_vivo": False}}
    interpreted = _interpret([_proposal(label)], book)
    assert interpreted["accepted"][0]["action"] == "reuse"
    assert interpreted["accepted"][0]["code"] == label
    assert interpreted["label_retries"] == []
    assert "code_too_long" not in interpreted["accepted"][0]["review_reasons"]


class ScriptCoder(FakeCoder):
    """First response uses ``first``. A retry that mentions ``marker`` uses ``later``."""

    first: str = SECOND
    later: str = SECOND
    marker: str = ""
    drop_on_retry: bool = False

    def code(self, text: str) -> PassageCoding:
        if self.marker and self.marker in text:
            if self.drop_on_retry:
                return PassageCoding(codes=[], memo="")
            return _coding(self.later)
        return _coding(self.first)


def _coding(label: str) -> PassageCoding:
    return PassageCoding(
        codes=[
            CodeProposal(
                action="new",
                code=label,
                definition="Incidents named by this label.",
                quote=QUOTE,
                rationale="The passage shows this incident.",
                confidence=0.9,
            )
        ],
        memo="",
    )


def _run(model, tmp_path):
    return open_code(
        [{"id": "sam", "name": "Sam", "text": TEXT}],
        CONTEXT,
        model=model,
        seed=1,
        sweep_every=100,
        batch_size=1,
        concurrency=1,
        pipeline=False,
        validation_retries=1,
        output_dir=tmp_path,
    )


def _coding_prompts(result):
    return [item for item in result.prompt_log if item["kind"] == "code_passage"]


def _labels(result):
    return {entry["code"] for entry in result.codebook}


def test_semicolon_label_is_split_and_the_bundle_is_retried(tmp_path):
    result = _run(ScriptCoder(first=f"{CATCHALL}; {SECOND}", later=f"{CATCHALL}; {SECOND}", marker="bundled_concepts"), tmp_path)
    assert _labels(result) == {CATCHALL, SECOND}
    assert len(result.segments) == 2
    assert {entry["frequency"] for entry in result.codebook} == {1}
    prompts = _coding_prompts(result)
    assert len(prompts) == 2
    feedback = json.dumps(prompts[1]["messages"])
    assert "bundled_concepts" in feedback
    assert "Return a single concept" in feedback
    catchall = next(entry for entry in result.codebook if entry["code"] == CATCHALL)
    assert catchall["needs_review"] is True
    assert "bundled_concepts" in catchall["review_reasons"]
    other = next(segment for segment in result.segments if segment["code"] == SECOND)
    assert "code_too_long" not in (other.get("review_reasons") or [])
    assert any(item.get("stage") == "retry" for item in result.validation_log)
    assert any(item.get("reason") == "bundled_concepts" for item in result.review)


def test_empty_label_retries_once_and_uses_the_replacement(tmp_path):
    result = _run(ScriptCoder(first="   ", later=SECOND, marker="empty_code"), tmp_path)
    assert _labels(result) == {SECOND}
    prompts = _coding_prompts(result)
    assert len(prompts) == 2
    assert "Do not leave the code blank" in json.dumps(prompts[1]["messages"])
    assert EMPTY_LABEL not in _labels(result)
    assert any(item.get("stage") == "retry" for item in result.validation_log)


def test_empty_label_is_kept_with_a_review_flag_when_the_retry_is_still_empty(tmp_path):
    result = _run(ScriptCoder(first="   ", later="   ", marker="empty_code"), tmp_path)
    assert _labels(result) == {EMPTY_LABEL}
    segment = result.segments[0]
    assert segment["code"] == EMPTY_LABEL
    assert segment["needs_review"] is True
    assert "empty_code" in segment["review_reasons"]
    entry = result.codebook[0]
    assert entry["needs_review"] is True
    assert "empty_code" in entry["review_reasons"]
    assert any(item.get("reason") == "empty_code" for item in result.review)
    assert len(_coding_prompts(result)) == 2
    assert any(item.get("stage") == "kept" for item in result.validation_log)


def test_empty_label_is_kept_when_the_retry_returns_no_codes(tmp_path):
    result = _run(ScriptCoder(first="   ", later=SECOND, marker="empty_code", drop_on_retry=True), tmp_path)
    assert _labels(result) == {EMPTY_LABEL}
    assert "empty_code" in result.segments[0]["review_reasons"]
    assert len(_coding_prompts(result)) == 2


def test_too_long_label_retries_once_with_feedback(tmp_path):
    result = _run(ScriptCoder(first=TOO_LONG, later=SECOND, marker="code_too_long"), tmp_path)
    assert _labels(result) == {SECOND}
    feedback = json.dumps(_coding_prompts(result)[1]["messages"])
    assert "code_too_long" in feedback
    assert "The label is too long" in feedback
    assert TOO_LONG not in _labels(result)


def test_too_long_label_is_kept_with_a_review_flag_after_retry(tmp_path):
    result = _run(ScriptCoder(first=TOO_LONG, later=TOO_LONG, marker="code_too_long"), tmp_path)
    assert _labels(result) == {TOO_LONG}
    assert "code_too_long" in result.segments[0]["review_reasons"]
    assert result.codebook[0]["needs_review"] is True
    assert "code_too_long" in result.codebook[0]["review_reasons"]
    assert any(item.get("reason") == "code_too_long" for item in result.review)
    assert len(_coding_prompts(result)) == 2


def test_semicolon_retry_keeps_the_other_concept(tmp_path):
    result = _run(
        ScriptCoder(first=f"{CATCHALL}; {SECOND}", later="valuing flexibility", marker="bundled_concepts"),
        tmp_path,
    )
    assert _labels(result) == {"valuing flexibility", SECOND}
    assert CATCHALL not in _labels(result)
    assert len(result.segments) == 2


def test_comma_bundle_retries_once_and_asks_for_one_concept(tmp_path):
    result = _run(ScriptCoder(first=CATCHALL, later="valuing flexibility", marker="bundled_concepts"), tmp_path)
    assert _labels(result) == {"valuing flexibility"}
    feedback = json.dumps(_coding_prompts(result)[1]["messages"])
    assert "bundled_concepts" in feedback
    assert "Return a single concept" in feedback
    assert CATCHALL not in _labels(result)


def test_and_joined_bundle_retries_once_for_a_single_concept(tmp_path):
    result = _run(ScriptCoder(first=AND_JOINED, later="seeking flexibility", marker="bundled_concepts"), tmp_path)
    assert _labels(result) == {"seeking flexibility"}
    feedback = json.dumps(_coding_prompts(result)[1]["messages"])
    assert "Return a single concept" in feedback
    assert AND_JOINED not in _labels(result)


def test_bundled_label_is_kept_with_a_review_flag_when_the_retry_stays_bundled(tmp_path):
    result = _run(ScriptCoder(first=CATCHALL, later=CATCHALL, marker="bundled_concepts"), tmp_path)
    assert _labels(result) == {CATCHALL}
    assert "bundled_concepts" in result.segments[0]["review_reasons"]
    assert result.codebook[0]["needs_review"] is True
    assert "bundled_concepts" in result.codebook[0]["review_reasons"]
    assert any(item.get("reason") == "bundled_concepts" for item in result.review)
    assert len(_coding_prompts(result)) == 2
