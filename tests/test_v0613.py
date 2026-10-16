"""0.6.13 regression tests for the live-run fixes."""

import asyncio
import logging

from automated_grounded_theory.codebook import commit_passage, empty_counters, interpret_coding
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.evaluate import assignment_passage_ids
from automated_grounded_theory.graph import (
    RunConfig,
    _attach_failed_passages,
    _invoke_coding,
    _segmentation_metadata,
)
from automated_grounded_theory.prompts import CODING_SYSTEM, extract_payload
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.results import result_from_state
from automated_grounded_theory.retrieval import LexicalCodeRetriever
from automated_grounded_theory.schemas import BatchCoding, CodeProposal
from automated_grounded_theory.segment import (
    _collect_uncovered,
    _speaker_matches,
    _uncovered_answer_text,
    plan_coding,
)
from automated_grounded_theory.validation import anchor_quote
from automated_grounded_theory.version import __version__
from tests.fake_llm import FakeCoder


def test_version_is_0_6_13():
    assert __version__ == "0.9.6"
    assert POLICY_VERSION == "0.9.6"


RAND_LIKE = """ Interviewer: What did you read before you left?
 Participant: I read a great deal of poetry. The novels stayed with me for years and I kept returning to them.
best.   Interviewer: Which author stayed with you?
 Participant: Ovid wrote the exile poems I reread while I packed two bags and left the city.
"""


def _covers(source: str, needle: str, passages: list[dict]) -> bool:
    start = source.index(needle)
    end = start + len(needle)
    return any(int(passage["start_offset"]) <= start and int(passage["end_offset"]) >= end for passage in passages)


def test_indented_and_midline_speakers_leave_no_answer_uncovered():
    plan = plan_coding([{"id": "rand", "text": RAND_LIKE}], unit="incident", seed=1)
    assert plan.uncovered_answers == []
    assert not any("fell outside every passage" in warning for warning in plan.warnings)
    poetry = next(passage for passage in plan.passages if "poetry" in passage["text"])
    ovid = next(passage for passage in plan.passages if "Ovid wrote" in passage["text"])
    assert poetry is not ovid
    assert "best." in poetry["text"]
    assert "Which author stayed with you?" in ovid["question"]
    assert "Which author stayed with you?" not in poetry["question"]
    for needle in (
        "I read a great deal of poetry",
        "best.",
        "Ovid wrote the exile poems",
    ):
        assert _covers(RAND_LIKE, needle, plan.passages)
    gap = _uncovered_answer_text(RAND_LIKE, [("ignored", 0, len(RAND_LIKE))], plan.passages)
    # The whole document is not all answer text. The participant needles above are the check.
    assert "Ovid wrote the exile poems" not in gap
    assert "I read a great deal of poetry" not in gap


def test_uncovered_answer_text_is_warned_and_stored():
    text = "Interviewer: What happened?\nParticipant: I packed my suitcase before dawn.\n"
    start = text.index("Participant:")
    end = len(text.rstrip())
    gap = _uncovered_answer_text(text, [("What happened?", start, end)], [])
    assert "I packed my suitcase before dawn." in gap
    passages = [
        {
            "doc_id": "rand",
            "metadata": {"uncovered_answer_text": gap, "uncovered_answer_chars": len(gap)},
        }
    ]
    uncovered = _collect_uncovered(passages)
    assert uncovered == [{"doc_id": "rand", "chars": len(gap), "text": gap}]
    metadata = _segmentation_metadata({"unit": "incident", "seed": 1}, type("Plan", (), {"seed": 1, "passages": [], "uncovered_answers": uncovered})())
    assert metadata["uncovered_answers"] == uncovered


def test_note_after_a_period_is_not_a_new_speaker():
    text = "Interviewer: What happened?\nParticipant: I went home. Note: the bag was heavy and I kept it.\n"
    plan = plan_coding([{"id": "n", "text": text}], unit="incident", seed=1)
    assert len(plan.passages) == 1
    assert "Note: the bag was heavy" in plan.passages[0]["text"]


FILLER = """Interviewer: What did you read?
Participant: I read poetry for years and the novels stayed with me through the winter.
Interviewer: Which author?
Participant: Ovid.
Interviewer: Why that one?
Participant: The exile poems matched packing a life into two bags and leaving home behind.
"""


def test_short_filler_turns_merge_into_the_adjacent_answer_and_can_be_disabled():
    # The default is off. "Ovid." stays its own passage, and so does "I quit."
    separate = plan_coding([{"id": "f", "text": FILLER}], unit="incident", seed=1)
    ovid = next(passage for passage in separate.passages if passage["text"].strip() == "Participant: Ovid.")
    assert "poetry" not in ovid["text"]
    assert _covers(FILLER, "Ovid.", separate.passages)
    assert separate.passages
    # Even with the merge turned on, an interviewer question between the turns blocks it.
    blocked = plan_coding([{"id": "f", "text": FILLER}], unit="incident", seed=1, short_turn_chars=60)
    ovid_blocked = next(passage for passage in blocked.passages if passage["text"].strip() == "Participant: Ovid.")
    assert "poetry" not in ovid_blocked["text"]
    assert "Which author?" not in ovid_blocked["text"]
    assert _covers(FILLER, "Ovid.", blocked.passages)


class _DropSecond(FakeCoder):
    """Omit every passage but the first when the request has several."""

    def batch(self, text: str) -> BatchCoding:
        coding = super().batch(text)
        passages = extract_payload(text).get("passages") or []
        if len(passages) > 1 and len(coding.passages) > 1:
            coding.passages = [coding.passages[0]]
        return coding


class _AlwaysDrop(FakeCoder):
    """Never return passage p2, even when it is requested alone."""

    def batch(self, text: str) -> BatchCoding:
        coding = super().batch(text)
        coding.passages = [item for item in coding.passages if item.passage_id != "p2"]
        return coding


def _runtime(coder: FakeCoder) -> RunConfig:
    return RunConfig(
        coder=coder,
        sweeper=FakeCoder(),
        retriever=LexicalCodeRetriever(),
        embeddings=None,
        context=ResearchContext(research_question="What is happening?"),
        prompt_cache=False,
        top_k_codes=4,
        frequent_codes=0,
        token_budget=6000,
        validation_retries=0,
        pipeline=False,
        missing_slot_retries=2,
    )


def _two_passages() -> list[dict]:
    return [
        {"passage_id": "p1", "text": "I packed my suitcase before dawn.", "doc_id": "d"},
        {"passage_id": "p2", "text": "I was crying at the station until the train left.", "doc_id": "d"},
    ]


def test_missing_batch_slot_is_retried_alone(caplog):
    caplog.set_level(logging.WARNING)
    coder = _DropSecond()
    runtime = _runtime(coder)
    aligned = asyncio.run(_invoke_coding(runtime, _two_passages(), {}, None))
    assert [item["passage_id"] for item in aligned] == ["p1", "p2"]
    assert all(item.get("coverage_status") != "failed" for item in aligned)
    assert coder.call_count == 2
    assert any("Requesting it alone" in note for note in runtime.notes)
    assert "Requesting it alone" in caplog.text
    assert runtime.failed_passage_ids == []


def test_still_failed_passages_are_logged_and_listed(caplog):
    caplog.set_level(logging.WARNING)
    coder = _AlwaysDrop()
    runtime = _runtime(coder)
    aligned = asyncio.run(_invoke_coding(runtime, _two_passages(), {}, None))
    assert aligned[0]["coverage_status"] != "failed"
    assert aligned[1]["passage_id"] == "p2"
    assert aligned[1]["coverage_status"] == "failed"
    # The batch call plus two solo calls. Retries bypass the response cache,
    # so the second solo request reaches the model instead of replaying it.
    assert coder.call_count == 3
    assert runtime.limiter.cache_hits == 0
    assert "attempt 1/2" in caplog.text
    assert "attempt 2/2" in caplog.text
    assert runtime.failed_passage_ids == ["p2"]
    assert any("stayed failed" in note for note in runtime.notes)
    assert any(note.startswith("Coding replies left passages failed: p2") for note in runtime.notes)
    assert "p2" in caplog.text
    metadata = _attach_failed_passages(runtime, {"metadata": {"seed": 1}}, None)
    passages = [
        {"passage_id": "p1", "coverage_status": "coded", "coding_order": 0, "doc_id": "d", "text": "a"},
        {"passage_id": "p2", "coverage_status": "failed", "coding_order": 1, "doc_id": "d", "text": "b"},
    ]
    result = result_from_state(
        {
            "metadata": metadata,
            "passages": passages,
            "warnings": list(runtime.notes),
            "documents": [],
            "segments": [],
            "codebook": {},
        },
        interrupted=False,
        thread_id="",
        resumable=False,
    )
    assert result.metadata["summary"]["failed_passages"] == ["p2"]
    assert result.metadata["failed_passages"] == ["p2"]
    assert any("p2" in warning and "failed" in warning for warning in result.warnings)


def _passages() -> list[dict]:
    first = "Then I left the city."
    second = "I packed my suitcase before dawn."
    return [
        {
            "passage_id": "p1",
            "doc_id": "d",
            "doc_name": "d",
            "text": first,
            "start_offset": 0,
            "end_offset": len(first),
            "prev_text": "",
            "next_text": "",
            "metadata": {},
        },
        {
            "passage_id": "p2",
            "doc_id": "d",
            "doc_name": "d",
            "text": second,
            "start_offset": 100,
            "end_offset": 100 + len(second),
            "prev_text": "",
            "next_text": "",
            "metadata": {},
        },
    ]


def _coding(quote: str, label: str = "packing bags") -> dict:
    return {
        "codes": [
            {
                "action": "new",
                "code": label,
                "definition": f"Incidents of {label}.",
                "quote": quote,
                "confidence": 0.9,
                "rationale": "The incident is in the participant's words.",
            }
        ],
        "memo": "",
    }


def test_request_passage_quote_is_kept_and_attributed():
    passages = _passages()
    quote = passages[1]["text"]
    anchor = anchor_quote(
        quote,
        passages[0]["text"],
        request_passages=passages,
        exclude_passage_id="p1",
    )
    assert anchor.status == "exact"
    assert anchor.source == "request"
    assert anchor.located_passage_id == "p2"
    interpreted = interpret_coding(
        _coding(quote),
        passages[0],
        {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
        request_passages=passages,
        passages=passages,
    )
    kept = interpreted["accepted"][0]
    assert kept["code"] == "packing bags"
    assert kept["quote_status"] == "exact"
    assert "evidence_outside_passage" in kept["review_reasons"]
    assert kept["attributed_passage_id"] == "p2"
    assert not any(item["reason"] == "quote_not_verbatim" for item in interpreted["rejections"])
    missing = interpret_coding(
        _coding("NOT IN ANY PASSAGE AT ALL"),
        passages[0],
        {},
        max_codes=6,
        require_gerund=False,
        long_passage_chars=2000,
        request_passages=passages,
        passages=passages,
    )
    assert missing["accepted"][0]["quote_status"] == "unanchored"
    assert missing["accepted"][0]["code"] == "packing bags"


def test_cross_passage_duplicate_merges_instead_of_double_counting():
    passages = _passages()
    quote = passages[1]["text"]
    kwargs = dict(max_codes=6, require_gerund=False, long_passage_chars=2000)
    inside = interpret_coding(_coding(quote), passages[1], {}, passages=passages, **kwargs)
    committed = commit_passage({}, [], [], {}, empty_counters(), passages[1], inside)
    outside = interpret_coding(
        _coding(quote),
        passages[0],
        committed["codebook"],
        passages=passages,
        request_passages=passages,
        **kwargs,
    )
    assert outside["accepted"][0]["attributed_passage_id"] == "p2"
    again = commit_passage(
        committed["codebook"],
        committed["segments"],
        committed["memos"],
        committed["highlight_ids"],
        committed["counters"],
        passages[0],
        outside,
    )
    assert len(again["segments"]) == 1
    provenance = again["segments"][0]["provenance"]["coded_passage_ids"]
    assert provenance == ["p2", "p1"]
    key = again["segments"][0]["code_key"]
    assert again["codebook"][key]["frequency"] == 1
    assert assignment_passage_ids(again["segments"][0]) == ["p2"]
    other = interpret_coding(
        _coding(quote, label="leaving the city"),
        passages[0],
        again["codebook"],
        passages=passages,
        request_passages=passages,
        **kwargs,
    )
    third = commit_passage(
        again["codebook"],
        again["segments"],
        again["memos"],
        again["highlight_ids"],
        again["counters"],
        passages[0],
        other,
    )
    assert len(third["segments"]) == 2


EXACT_FALSE_LABEL = (
    "Participant: It's a pain because you can't \n"
    "download the results   Interviewer: Do you happen to recall which trials registry it was?\n"
    "Participant: I usually search clinicaltrials.gov and the World Health Organization registry.\n"
)


def test_line_start_known_role_does_not_swallow_the_words_before_it():
    assert "download the results   Interviewer:" in EXACT_FALSE_LABEL
    labels = [(hit.group(1), hit.start()) for hit in _speaker_matches(EXACT_FALSE_LABEL)]
    assert "download the results   Interviewer" not in [label for label, _start in labels]
    interviewer = next(start for label, start in labels if label == "Interviewer")
    assert EXACT_FALSE_LABEL[interviewer:interviewer + len("Interviewer:")] == "Interviewer:"
    download = EXACT_FALSE_LABEL.index("download the results")
    assert interviewer > download
    plan = plan_coding([{"id": "2", "text": EXACT_FALSE_LABEL}], unit="incident", seed=1, short_turn_chars=0)
    answer = next(passage for passage in plan.passages if "download the results" in passage["text"])
    assert "which trials registry" not in answer["text"]
    assert answer["start_offset"] <= download < answer["end_offset"]
    recall = next(passage for passage in plan.passages if "clinicaltrials.gov" in passage["text"])
    assert "Do you happen to recall which trials registry it was?" in recall["question"]
    assert plan.uncovered_answers == []


PREAMBLE = (
    "Participant 02\n"
    "Institution: Academic health sciences library\n"
    "Interviewer: Where do you work now?\n"
    "Participant: I work in the library stacks on the north side of the building every morning.\n"
)


def test_preamble_before_the_first_speaker_is_attached_and_warned():
    plan = plan_coding([{"id": "2", "text": PREAMBLE}], unit="incident", seed=1, short_turn_chars=0)
    assert plan.passages[0]["text"].startswith("Participant 02")
    assert plan.passages[0]["start_offset"] == 0
    assert "Participant 02" in plan.passages[0]["text"]
    assert plan.uncovered_answers == []
    assert plan.attached_preambles == [{"doc_id": "2", "chars": len("Participant 02"), "text": "Participant 02"}]
    assert any("before the first speaker label" in warning for warning in plan.warnings)
    assert plan.passages[0]["metadata"]["attached_preamble_text"] == "Participant 02"
    covered = [False] * len(PREAMBLE)
    for passage in plan.passages:
        for index in range(passage["start_offset"], passage["end_offset"]):
            covered[index] = True
    missing = "".join(char for index, char in enumerate(PREAMBLE) if not char.isspace() and not covered[index])
    assert missing == "Interviewer:Wheredoyouworknow?"
    metadata = _segmentation_metadata(
        {"unit": "incident", "seed": 1},
        type("Plan", (), {"seed": 1, "passages": plan.passages, "uncovered_answers": [], "attached_preambles": plan.attached_preambles})(),
    )
    assert metadata["attached_preambles"] == plan.attached_preambles


def test_document_prompt_codes_each_passage_and_quotes_only_the_target():
    folded = CODING_SYSTEM.casefold()
    assert "code each passage separately" in folded
    assert "spans several passages" in folded
    assert "target passage" in folded
    assert "do not quote the previous context" in folded
    description = CodeProposal.model_fields["quote"].description.casefold()
    assert "target passage" in description
    assert "previous or next context, copy from there" not in description
