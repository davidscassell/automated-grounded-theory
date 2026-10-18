"""0.6.14 regression tests for the review findings."""

import asyncio
import csv
import tempfile

from automated_grounded_theory.async_support import CallLimiter
from automated_grounded_theory.codebook import commit_passage, empty_counters, interpret_coding
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.evaluate import assignment_passage_ids
from automated_grounded_theory.export import export_result
from automated_grounded_theory.graph import _invoke_coding
from automated_grounded_theory.prompts import (
    build_sweep_messages,
    code_input,
    format_sweep_codebook,
    message_text,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.results import OpenCodeResult, result_from_state
from automated_grounded_theory.retrieval import (
    PROMPT_EXAMPLE_LIMIT,
    LexicalCodeRetriever,
    aembed_texts,
    assignment_ids,
    cached_aembed_texts,
    clear_embedding_cache,
    code_needs_review,
    configure_embedding_cache,
    embedding_cache_stats,
    record_review,
    review_fingerprint,
    select_prompt_examples,
)
from automated_grounded_theory.saturation import saturation_report
from automated_grounded_theory.segment import (
    SHORT_TURN_CHARS,
    _apply_source_coverage,
    _source_text_coverage,
    _speaker_matches,
    plan_coding,
)
from automated_grounded_theory.themes import _one_theme
from automated_grounded_theory.version import __version__
from tests.fake_llm import FakeCoder
from tests.test_v0613 import _AlwaysDrop, _DropSecond, _runtime, _two_passages


def test_version_is_0_6_14():
    assert __version__ == "0.9.18"
    assert POLICY_VERSION == "0.9.18"
    assert SHORT_TURN_CHARS == 0
    assert PROMPT_EXAMPLE_LIMIT == 2


MIXED = """Interviewer: What happened after the meeting?
Participant 02: I packed two bags and left the city before dawn.
Interviewer: Who stayed behind?
J.D.: I stayed because the rent was already paid through the month.
Interviewer: And the person with spaced initials?
J. D.: I work in the stacks on the north side every morning before opening.
"""


def _covers(source: str, needle: str, passages: list[dict]) -> bool:
    start = source.index(needle)
    end = start + len(needle)
    return any(int(passage["start_offset"]) <= start and int(passage["end_offset"]) >= end for passage in passages)


def test_numbered_and_initialed_speakers_are_not_folded_into_the_question():
    """The reviewer's mixed transcript: Participant 02 and J.D. were not speakers."""
    labels = [hit.group(1) for hit in _speaker_matches(MIXED)]
    assert "Participant 02" in labels
    assert "J.D." in labels
    assert "J. D." in labels
    plan = plan_coding([{"id": "m", "text": MIXED}], unit="incident", seed=1, short_turn_chars=0)
    packed = next(passage for passage in plan.passages if "I packed two bags" in passage["text"])
    stayed = next(passage for passage in plan.passages if "rent was already paid" in passage["text"])
    stacks = next(passage for passage in plan.passages if "north side" in passage["text"])
    assert packed is not stayed
    assert "Participant 02" not in packed["question"]
    assert "I packed two bags" not in packed["question"]
    assert packed["question"] == "What happened after the meeting?"
    assert stayed["question"] == "Who stayed behind?"
    assert "J.D." not in stayed["question"]
    assert stacks["question"] == "And the person with spaced initials?"
    assert plan.uncovered_answers == []
    assert plan.source_uncovered == []
    assert plan.question_context_turns == []
    assert not any("fell outside" in warning or "question context" in warning or "neither a passage" in warning for warning in plan.warnings)
    for needle in (
        "I packed two bags and left the city before dawn.",
        "I stayed because the rent was already paid through the month.",
        "I work in the stacks on the north side every morning before opening.",
    ):
        assert _covers(MIXED, needle, plan.passages)


def test_source_coverage_flags_a_speaker_turn_that_is_only_question_context():
    """Coverage compares characters. It does not ask the speaker parser what an answer is."""
    text = (
        "Interviewer: What happened?\n"
        "Participant 02: I packed two bags and left before dawn.\n"
        "Participant: I came back the next week with one suitcase.\n"
    )
    start = text.index("Participant: I came")
    passages = [
        {
            "doc_id": "m",
            "start_offset": start,
            "end_offset": len(text.rstrip()),
            "question": "What happened?\nParticipant 02: I packed two bags and left before dawn.",
        }
    ]
    report = _source_text_coverage(text, passages)
    assert report["uncovered_chars"] == 0
    assert any("Participant 02" in turn and "I packed two bags" in turn for turn in report["question_turns"])
    warnings: list[str] = []
    _apply_source_coverage(text, passages, warnings, "m")
    assert any("speaker turn text is only in question context" in warning and "Participant 02" in warning for warning in warnings)
    assert passages[0]["metadata"]["question_context_turns"]

    orphan = text + "ORPHAN leftover words that no passage holds.\n"
    warnings.clear()
    orphan_passages = [
        {
            "doc_id": "m",
            "start_offset": start,
            "end_offset": len(text.rstrip()),
            "question": "What happened?",
        }
    ]
    missed = _source_text_coverage(orphan, orphan_passages)
    assert missed["uncovered_chars"] > 0
    assert "ORPHAN leftover words" in missed["uncovered"]
    _apply_source_coverage(orphan, orphan_passages, warnings, "m")
    assert any("non-whitespace characters are in neither a passage nor question context" in warning for warning in warnings)
    assert "ORPHAN leftover words" in orphan_passages[0]["metadata"]["source_uncovered_text"]


def test_numbered_labels_do_not_false_positive_on_captions_or_abbreviations():
    caption = (
        "Interviewer: What did the table show?\n"
        "Table 1: the rows are complete and the total is shown here.\n"
        "Figure 2: the line rises after the first month of the study.\n"
        "Chapter 3: the archive moved across town.\n"
        "Participant: I read every row before I left the archive that morning.\n"
    )
    labels = [hit.group(1) for hit in _speaker_matches(caption)]
    assert labels == ["Interviewer", "Participant"]
    plan = plan_coding([{"id": "c", "text": caption}], unit="incident", seed=1)
    answer = next(passage for passage in plan.passages if "every row" in passage["text"])
    assert "Table 1" in answer["question"]
    assert "Figure 2" in answer["question"]
    assert plan.question_context_turns == []
    assert plan.source_uncovered == []

    abbreviations = (
        "e.g.: this is only an example of the pattern.\n"
        "U.S.: the office stayed open through the winter.\n"
        "Participant: I left the building before the doors were locked.\n"
        "Interviewer: Where did you go after that?\n"
    )
    abbrev_labels = [hit.group(1) for hit in _speaker_matches(abbreviations)]
    assert "e.g." not in abbrev_labels
    assert "U.S." not in abbrev_labels
    assert "Participant" in abbrev_labels

    prose = "Participant: I went home. Note: the bag was heavy and I kept it with me.\nInterviewer: Why keep it?\n"
    assert "Note" not in [hit.group(1) for hit in _speaker_matches(prose)]
    said = "Participant: I said: hello to the group before I left the room for good.\nInterviewer: Where did you go?\n"
    assert not any(hit.group(1).casefold() == "said" or hit.group(1).casefold().startswith("i said") for hit in _speaker_matches(said))

    known = (
        "Participant: It's a pain because you can't \n"
        "download the results   Interviewer: Which registry was it?\n"
        "Participant: I usually search clinicaltrials.gov before dawn.\n"
    )
    known_labels = [hit.group(1) for hit in _speaker_matches(known)]
    assert "Interviewer" in known_labels
    assert not any("download" in label for label in known_labels)
    numbered = "Participant: I finished the form.   Participant 02: I packed two bags and left before dawn.\n"
    assert "Participant 02" in [hit.group(1) for hit in _speaker_matches(numbered)]


ADJACENT = """Participant: I packed the books into two bags and carried them down the stairs before dawn.
Participant: Yes.
Participant: The exile poems matched the feeling of leaving a city I had lived in for years.
"""

QUIT = """Interviewer: What did you decide?
Participant: I quit.
Interviewer: Why?
Participant: The shift ended after midnight and the bus had already left the station.
"""


def test_short_turn_merge_stays_off_and_never_crosses_a_question():
    assert SHORT_TURN_CHARS == 0
    separate = plan_coding([{"id": "a", "text": ADJACENT}], unit="incident", seed=1)
    assert any(passage["text"].strip() == "Participant: Yes." for passage in separate.passages)
    opted = plan_coding([{"id": "a", "text": ADJACENT}], unit="incident", seed=1, short_turn_chars=60)
    merged = next(passage for passage in opted.passages if "carried them down the stairs" in passage["text"])
    assert "Participant: Yes." in merged["text"]
    assert "exile poems" not in merged["text"]
    # "I quit." is the incident. A question between turns blocks the merge even when the cap is on.
    for limit in (0, 60):
        plan = plan_coding([{"id": "q", "text": QUIT}], unit="incident", seed=1, short_turn_chars=limit)
        quit = next(passage for passage in plan.passages if passage["text"].strip() == "Participant: I quit.")
        assert "bus had already left" not in quit["text"]
        assert quit["question"] == "What did you decide?"
        assert "Why?" not in quit["text"]


def _proposal(quote: str, label: str = "packing bags") -> dict:
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


def _people():
    alice = {
        "passage_id": "p1",
        "doc_id": "dA",
        "doc_name": "Alice",
        "text": "I stayed in the library all morning and shelved the new journals.",
        "start_offset": 0,
        "end_offset": 62,
        "metadata": {"respondent_id": "alice"},
    }
    bob_text = "I packed two bags and left before dawn."
    bob = {
        "passage_id": "p2",
        "doc_id": "dB",
        "doc_name": "Bob",
        "text": bob_text,
        "start_offset": 100,
        "end_offset": 100 + len(bob_text),
        "metadata": {"respondent_id": "bob"},
    }
    return alice, bob, bob_text


def test_cross_document_evidence_keeps_the_located_document():
    """passage_id p2 is Bob's. The old segment stored Alice's doc_id and doc_name."""
    alice, bob, quote = _people()
    passages = [alice, bob]
    kwargs = dict(max_codes=6, require_gerund=False, long_passage_chars=2000, passages=passages, request_passages=passages)
    outside = interpret_coding(_proposal(quote), alice, {}, **kwargs)
    kept = outside["accepted"][0]
    assert kept["evidence_doc_id"] == "dB"
    assert kept["evidence_doc_name"] == "Bob"
    assert kept["evidence_metadata"]["respondent_id"] == "bob"
    assert kept["doc_start"] == 100
    assert kept["attributed_passage_id"] == "p2"
    committed = commit_passage({}, [], [], {}, empty_counters(), alice, outside)
    segment = committed["segments"][0]
    assert segment["passage_id"] == "p2"
    assert segment["doc_id"] == "dB"
    assert segment["doc_name"] == "Bob"
    assert segment["doc_start"] == 100
    assert segment["metadata"]["respondent_id"] == "bob"
    assert segment["provenance"]["coded_passage_id"] == "p1"
    assert segment["provenance"]["coded_doc_id"] == "dA"
    assert segment["provenance"]["coded_doc_name"] == "Alice"
    assert segment["provenance"]["coded_documents"] == [
        {"passage_id": "p1", "doc_id": "dA", "doc_name": "Alice"}
    ]
    key = segment["code_key"]
    assert committed["codebook"][key]["frequency"] == 1
    assert committed["codebook"][key]["document_ids"] == ["dB"]

    theme = _one_theme(
        {
            "name": "Leaving",
            "definition": "Leaving a place.",
            "codes": ["packing bags"],
            "confidence": 0.9,
            "rationale": "The quotes show the same incident.",
        },
        segments=committed["segments"],
        passages_by_id={item["passage_id"]: item for item in passages},
        definitions={key: "Incidents of packing bags."},
        totals={"n_passages": 2, "n_documents": 2, "n_respondents": 2},
        respondent_key="respondent_id",
        quotes_per_theme=3,
        min_quote_chars=10,
    )
    assert theme["quotes"][0]["respondent"] == "bob"
    assert theme["quotes"][0]["doc_id"] == "dB"
    assert theme["code_counts"][0]["documents"] == 1

    result = OpenCodeResult(
        codebook=list(committed["codebook"].values()),
        segments=committed["segments"],
        memos=[],
        passages=passages,
        saturation=[],
        saturation_report=saturation_report([]),
        validation_log=[],
        sweep_log=[],
        prompt_log=[],
        metadata={},
    )
    with tempfile.TemporaryDirectory() as directory:
        export_result(result, directory)
        with open(f"{directory}/highlights.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    assert rows[0]["document"] == "Bob"
    assert "bob" in rows[0]["metadata"]
    assert "alice" not in rows[0]["metadata"]


def _commit_order(first_passage, first_interpreted, second_passage, second_interpreted):
    first = commit_passage({}, [], [], {}, empty_counters(), first_passage, first_interpreted)
    return commit_passage(
        first["codebook"],
        first["segments"],
        first["memos"],
        first["highlight_ids"],
        first["counters"],
        second_passage,
        second_interpreted,
    )


def test_duplicate_evidence_merges_in_either_order_and_keeps_distinct_offsets():
    alice, bob, quote = _people()
    # Same document, so this is the order bug and not the cross-document bug.
    same = [
        {**alice, "doc_id": "d", "doc_name": "d", "metadata": {}},
        {**bob, "doc_id": "d", "doc_name": "d", "metadata": {}},
    ]
    kwargs = dict(max_codes=6, require_gerund=False, long_passage_chars=2000, passages=same)
    inside = interpret_coding(_proposal(quote), same[1], {}, **kwargs)
    outside = interpret_coding(_proposal(quote), same[0], {}, request_passages=same, **kwargs)
    correct_first = _commit_order(same[1], inside, same[0], outside)
    cross_first = _commit_order(same[0], outside, same[1], inside)
    for committed, order in ((correct_first, ["p2", "p1"]), (cross_first, ["p1", "p2"])):
        assert len(committed["segments"]) == 1
        segment = committed["segments"][0]
        key = segment["code_key"]
        assert committed["codebook"][key]["frequency"] == 1
        assert segment["provenance"]["coded_passage_ids"] == order
        assert assignment_passage_ids(segment) == ["p2"]
        assert segment["doc_start"] == 100

    repeated = "I left the city."
    early = {
        "passage_id": "p1",
        "doc_id": "d",
        "doc_name": "d",
        "text": repeated,
        "start_offset": 0,
        "metadata": {},
    }
    late = {
        "passage_id": "p2",
        "doc_id": "d",
        "doc_name": "d",
        "text": repeated,
        "start_offset": 80,
        "metadata": {},
    }
    pair = [early, late]
    first = interpret_coding(_proposal(repeated, "leaving the city"), early, {}, passages=pair, max_codes=6, require_gerund=False, long_passage_chars=2000)
    second = interpret_coding(_proposal(repeated, "leaving the city"), late, {}, passages=pair, max_codes=6, require_gerund=False, long_passage_chars=2000)
    both = _commit_order(early, first, late, second)
    assert len(both["segments"]) == 2
    assert both["codebook"][both["segments"][0]["code_key"]]["frequency"] == 2
    assert {(item["doc_start"], item["doc_end"]) for item in both["segments"]} == {(0, len(repeated)), (80, 80 + len(repeated))}


class _BatchQueries:
    model = "review8-batch"
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def aembed_queries(self, texts):
        self.calls.append(list(texts))
        return [[float(index + 1), 0.5] for index, _text in enumerate(texts)]


class _SingleQuery:
    model = "review8-single"
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def aembed_query(self, text):
        self.calls.append(text)
        return [float(len(text)), 0.25]


class _Documents:
    model = "review8-docs"
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def aembed_documents(self, texts):
        self.calls.append(list(texts))
        return [[0.1, float(len(text))] for text in texts]


def test_vector_cap_returns_every_query_in_the_request():
    """vector_limit=1 used to raise 'embedding cache missed a query it had claimed'."""
    limiter = CallLimiter(2, max_retries=0, backoff=0, max_backoff=0, request_timeout=0)
    texts = ["alpha query", "beta query"]

    async def one(embeddings, role: str):
        clear_embedding_cache()
        configure_embedding_cache(vector_limit=1)
        direct = await aembed_texts(embeddings, texts, batch_size=8, limiter=limiter, role=role)
        clear_embedding_cache()
        configure_embedding_cache(vector_limit=1)
        wrapped = await cached_aembed_texts(embeddings, texts, batch_size=8, limiter=limiter, role=role)
        assert embedding_cache_stats()["vectors"] == 1
        return direct, wrapped

    try:
        batch = _BatchQueries()
        single = _SingleQuery()
        documents = _Documents()
        batch_direct, batch_wrapped = asyncio.run(one(batch, "query"))
        single_direct, single_wrapped = asyncio.run(one(single, "query"))
        document_direct, document_wrapped = asyncio.run(one(documents, "document"))
    finally:
        configure_embedding_cache(vector_limit=None, vector_bytes=None)
        clear_embedding_cache()

    assert batch_direct == [[1.0, 0.5], [2.0, 0.5]]
    assert batch_wrapped == batch_direct
    assert single_direct == [[11.0, 0.25], [10.0, 0.25]]
    assert single_wrapped == single_direct
    assert document_direct == [[0.1, 11.0], [0.1, 10.0]]
    assert document_wrapped == document_direct


def _long_assignments(count: int, *, contrast_at: int | None = None) -> list[dict]:
    rows = []
    body = "packed the suitcase and left the city before dawn "
    for index in range(count):
        if contrast_at is not None and index == contrast_at:
            quote = "The landlord raised the rent and I signed a new lease downtown instead."
        else:
            quote = f"incident {index} " + body * 8
        rows.append(
            {
                "segment_id": f"s{index:03d}",
                "passage_id": f"p{index}",
                "doc_id": "d" if index < count - 10 else "other",
                "doc_start": index * 20,
                "doc_end": index * 20 + 10,
                "quote": quote,
            }
        )
    return rows


def test_new_evidence_is_sent_in_bounded_batches_and_unsent_stays_pending():
    rows = _long_assignments(100)
    entry = {"code": "leaving", "definition": "Leaving.", "revision": 1, "assignments": rows, "frequency": 100}
    chosen = select_prompt_examples(entry, limit=2)
    assert len(chosen) == 2
    assert sum(len(item["quote"]) for item in chosen) < 5000
    fresh = select_prompt_examples(entry, limit=PROMPT_EXAMPLE_LIMIT)
    assert len(fresh) == 2
    fingerprint = review_fingerprint(entry)
    record_review(entry, fingerprint)
    assert entry["reviewed_assignment_ids"] == [item["segment_id"] for item in fresh]
    assert len(entry["reviewed_assignment_ids"]) == 2
    assert entry["available_assignment_ids"] == assignment_ids(entry)
    assert len(entry["available_assignment_ids"]) == 100
    seen = entry["reviews_seen"][-1]
    assert seen["sent_assignment_ids"] == entry["reviewed_assignment_ids"]
    assert seen["assignment_ids"] == seen["sent_assignment_ids"]
    assert len(seen["available_assignment_ids"]) == 100
    assert code_needs_review(entry)
    pending = select_prompt_examples(entry, limit=2)
    assert len(pending) == 2
    assert set(item["segment_id"] for item in pending).isdisjoint(entry["reviewed_assignment_ids"])

    sent = list(entry["reviewed_assignment_ids"])
    for _ in range(60):
        if not code_needs_review(entry):
            break
        batch = select_prompt_examples(entry, limit=2)
        assert 1 <= len(batch) <= 2
        sent.extend(item["segment_id"] for item in batch)
        record_review(entry, fingerprint, assignment_ids(entry), [item["segment_id"] for item in batch])
    assert not code_needs_review(entry)
    assert len(set(sent)) == 100

    # Ten assignments, and the bookkeeping that used to mark all ten reviewed.
    ten = {"code": "leaving", "definition": "Leaving.", "revision": 1, "assignments": rows[:10], "frequency": 10}
    shown = select_prompt_examples(ten, limit=2)
    record_review(ten, review_fingerprint(ten), assignment_ids(ten), [item["segment_id"] for item in shown])
    assert len(ten["reviewed_assignment_ids"]) == 2
    assert len(ten["available_assignment_ids"]) == 10
    assert code_needs_review(ten)

    contrasting = {
        "code": "leaving",
        "definition": "Leaving.",
        "revision": 1,
        "assignments": _long_assignments(9, contrast_at=8),
        "frequency": 9,
    }
    # Same document for the contrast check: drop the other-doc split by rewriting ids.
    for item in contrasting["assignments"]:
        item["doc_id"] = "d"
    picked = select_prompt_examples(contrasting, limit=2)
    assert any("new lease" in item["quote"] for item in picked)

    context = ResearchContext(research_question="What is happening when someone leaves?")
    messages = build_sweep_messages(context=context, codes=[entry], candidates=[], prompt_cache=False)
    prompt = "\n".join(message_text(message) for message in messages)
    assert len(prompt) < 8000
    assert prompt.count("incident ") <= 4
    rendered = format_sweep_codebook([entry])
    assert rendered.count("incident ") <= 4
    coded = code_input(entry)
    assert len(coded.examples) <= PROMPT_EXAMPLE_LIMIT


def test_missing_slot_retry_cost_is_on_the_run_metrics():
    runtime = _runtime(_DropSecond())
    aligned = asyncio.run(_invoke_coding(runtime, _two_passages(), {}, None))
    assert [item["passage_id"] for item in aligned] == ["p1", "p2"]
    retries = [entry for entry in runtime.prompt_records if entry.get("missing_slot_retry")]
    assert len(retries) == 1
    result = result_from_state(
        {"metadata": {}, "passages": [], "prompt_log": list(runtime.prompt_records), "documents": [], "segments": [], "codebook": {}},
        interrupted=False,
        thread_id="",
        resumable=False,
    )
    metrics = result.metadata["metrics"]
    assert metrics["missing_slot_retry_calls"] == 1
    assert metrics["missing_slot_retry_input_tokens"] == 12
    assert metrics["missing_slot_retry_output_tokens"] == 6
    assert metrics["missing_slot_retry_tokens"] == 18
    assert metrics["input_tokens"] > metrics["missing_slot_retry_input_tokens"]

    failed = _runtime(_AlwaysDrop())
    asyncio.run(_invoke_coding(failed, _two_passages(), {}, None))
    failed_result = result_from_state(
        {"metadata": {}, "passages": [], "prompt_log": list(failed.prompt_records), "documents": [], "segments": [], "codebook": {}},
        interrupted=False,
        thread_id="",
        resumable=False,
    )
    assert failed_result.metadata["metrics"]["missing_slot_retry_calls"] == 2
    assert failed_result.metadata["metrics"]["missing_slot_retry_tokens"] == 36
