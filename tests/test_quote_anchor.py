"""Quote mismatch keeps the code and records how the quote was anchored."""

import json

from automated_grounded_theory.api import open_code
from automated_grounded_theory.codebook import interpret_coding
from automated_grounded_theory.evaluate import evaluate_run
from automated_grounded_theory.schemas import PassageCoding
from automated_grounded_theory.validation import anchor_quote, locate_quote
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder


def _coding(quote: str, label: str = "packing bags", confidence: float = 0.9) -> dict:
    return {
        "codes": [
            {
                "action": "new",
                "code": label,
                "definition": f"Incidents of {label}.",
                "quote": quote,
                "confidence": confidence,
                "rationale": "The incident is in the participant's words.",
            }
        ],
        "memo": "",
    }


def _passage() -> dict:
    return {
        "passage_id": "p1",
        "doc_id": "d1",
        "text": "I packed my suitcase.",
        "start_offset": 0,
        "end_offset": len("I packed my suitcase."),
        "prev_text": "",
        "next_text": "",
    }


def test_locate_quote_still_accepts_whitespace_and_rejects_missing_text():
    passage = "I packed my suitcase."
    assert locate_quote("I packed my suitcase.", passage) == (0, len(passage))
    assert locate_quote("I packed\nmy suitcase.", "I packed\nmy suitcase.") == (0, len("I packed\nmy suitcase."))
    assert locate_quote("I packed my suitcase.", "I packed  my suitcase.") == (0, len("I packed  my suitcase."))
    assert locate_quote("not in the text", passage) is None


def test_anchor_statuses_and_ellipsis_spans():
    passage = "I packed my suitcase."
    exact = anchor_quote("I packed my suitcase.", passage)
    assert exact.status == "exact"
    assert exact.similarity == 1.0
    assert exact.spans == ((0, len(passage)),)
    assert (exact.doc_start, exact.doc_end) == (0, len(passage))

    normalized = anchor_quote("i packed my suitcase", "I packed  my suitcase.")
    assert normalized.status == "normalized_match"
    assert normalized.quote == "I packed  my suitcase"
    with_period = anchor_quote("I packed my suitcase.", "I packed  my suitcase.")
    assert with_period.status == "normalized_match"
    assert with_period.quote == "I packed  my suitcase."

    spaced = anchor_quote("I search it", "I sear ch it")
    assert spaced.status == "normalized_match"
    assert spaced.quote == "I sear ch it"

    fuzzy = anchor_quote("I packed my suitcas.", passage)
    assert fuzzy.status == "fuzzy_match"
    assert fuzzy.similarity is not None and fuzzy.similarity >= 0.9
    assert fuzzy.spans
    assert passage[fuzzy.spans[0][0] : fuzzy.spans[0][1]]

    modified = anchor_quote("I packed my backpack.", passage)
    assert modified.status == "possibly_modified"
    assert modified.similarity is not None
    assert modified.doc_start is not None and modified.doc_end is not None
    assert passage[modified.doc_start : modified.doc_end]

    missing = anchor_quote("NOT IN THE PASSAGE", passage)
    assert missing.status == "unanchored"
    assert missing.spans == ()
    assert missing.doc_start is None

    ellipsis = anchor_quote("I do search it…", "I do search it every morning before rounds.")
    assert ellipsis.status == "exact"
    assert ellipsis.spans == ((0, len("I do search it")),)

    split = anchor_quote("I packed ... suitcase", "I packed my suitcase before dawn.")
    assert split.status == "exact"
    assert len(split.spans) == 2
    assert split.quote == "I packed ... suitcase"


def test_neighbor_then_document_search():
    passage = "Then I left."
    prev = "I packed my suitcase."
    document = prev + "\n\n" + passage
    start = len(prev) + 2
    neighbor = anchor_quote(
        "I packed my suitcase.",
        passage,
        prev_text=prev,
        document_text=document,
        passage_start=start,
        passage_end=start + len(passage),
    )
    assert neighbor.status == "exact"
    assert neighbor.source == "neighbor"
    assert document[neighbor.doc_start : neighbor.doc_end] == "I packed my suitcase."

    only_doc = anchor_quote(
        "I packed my suitcase.",
        "Alpha only.",
        document_text="Before. I packed my suitcase. Alpha only.",
        passage_start=len("Before. I packed my suitcase. "),
    )
    assert only_doc.status == "exact"
    assert only_doc.source == "document"
    assert only_doc.quote == "I packed my suitcase."


def test_quote_mismatch_keeps_the_code_and_flags_review():
    passage = _passage()
    interpreted = interpret_coding(
        _coding("NOT IN THE PASSAGE AT ALL"),
        passage,
        {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
    )
    assert [item["code"] for item in interpreted["accepted"]] == ["packing bags"]
    assert not any(item["reason"] == "quote_not_verbatim" for item in interpreted["rejections"])
    kept = interpreted["accepted"][0]
    assert kept["quote_status"] == "unanchored"
    assert kept["needs_review"]
    assert "quote_unanchored" in kept["review_reasons"]
    assert interpreted["quote_retries"][0]["reason"] == "quote_not_verbatim"
    flagged = [item for item in interpreted["review"] if item["reason"] == "quote_unanchored"]
    assert flagged
    assert flagged[0]["rationale"].strip()
    assert flagged[0]["quote_status"] == "unanchored"

    both = interpret_coding(
        {
            "codes": [
                _coding("I packed my suitcase.")["codes"][0],
                _coding("I packed my backpack.", label="hauling a backpack")["codes"][0],
            ],
            "memo": "",
        },
        passage,
        {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
    )
    assert [item["code"] for item in both["accepted"]] == ["packing bags", "hauling a backpack"]
    assert both["accepted"][0]["quote_status"] == "exact"
    assert both["accepted"][1]["quote_status"] == "possibly_modified"
    assert both["accepted"][1]["doc_start"] is not None


class MisquoteCoder(FakeCoder):
    """Always quotes text that is not in the source."""

    def code(self, text: str) -> PassageCoding:
        coding = super().code(text)
        if coding.codes:
            coding.codes[0].quote = "NOT IN THE PASSAGE AT ALL"
        return coding


class DropOnRetry(FakeCoder):
    """First response misquotes. The retry returns no codes."""

    def code(self, text: str) -> PassageCoding:
        if "quote_not_verbatim" in text:
            return PassageCoding(codes=[], memo="")
        coding = super().code(text)
        if coding.codes:
            coding.codes[0].quote = "NOT IN THE PASSAGE AT ALL"
        return coding


def test_open_code_keeps_unanchored_quotes(tmp_path):
    result = open_code(
        [CORPUS[0]],
        CONTEXT,
        model=MisquoteCoder(),
        seed=1,
        sweep_every=25,
        validation_retries=0,
        output_dir=tmp_path,
    )
    assert result.segments
    assert result.segments[0]["quote_status"] == "unanchored"
    assert "quote_unanchored" in result.segments[0]["review_reasons"]
    assert any(item.get("reason") == "quote_unanchored" for item in result.review)
    coded = (tmp_path / "coded_segments.csv").read_text(encoding="utf-8")
    header = coded.splitlines()[0]
    assert "quote_status" in header
    assert "unanchored" in coded
    review = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert any(item.get("reason") == "quote_unanchored" for item in review)
    taguette = (tmp_path / "taguette_codebook.csv").read_text(encoding="utf-8").splitlines()[0]
    assert taguette == "tag,description,number of highlights,number of documents"
    metrics = evaluate_run(result)
    assert metrics["quote_unanchored_rate"] > 0
    assert metrics["quote_flag_rate"] == metrics["quote_unanchored_rate"]
    assert metrics["quote_status_counts"]["unanchored"] >= 1


def test_retry_cannot_drop_a_code_that_only_mismatched(tmp_path):
    result = open_code(
        [CORPUS[0]],
        CONTEXT,
        model=DropOnRetry(),
        seed=1,
        sweep_every=25,
        validation_retries=1,
        output_dir=tmp_path,
    )
    assert result.segments
    assert result.segments[0]["code"]
    assert result.segments[0]["quote_status"] == "unanchored"
    blob = json.dumps(result.validation_log)
    assert "quote_not_verbatim" in blob
    assert any(item.get("stage") == "kept" for item in result.validation_log)
