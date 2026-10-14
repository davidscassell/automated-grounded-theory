"""0.6.4: spelling-only auto-merge, sequential commit, pair cache, stance, agreement."""

import json
import warnings

from automated_grounded_theory import open_code, strict_sequential_settings
from automated_grounded_theory.api import resolve_coding_schedule
from automated_grounded_theory.codebook import (
    apply_sweep,
    classify_comparison,
    frozen_pair_keys,
    lexical_comparison_proposals,
    remember_pair_decisions,
    trivial_merges,
    trivial_pair,
)
from automated_grounded_theory.context import parse_context
from automated_grounded_theory.evaluate import aligned_assignment_agreement, assignment_ari
from automated_grounded_theory.prompts import (
    CODING_SYSTEM,
    IN_BATCH_SYSTEM,
    STANCE_POLICY,
    SWEEP_SYSTEM,
    build_merge_check_messages,
    build_sweep_messages,
    message_text,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.retrieval import jaccard
from automated_grounded_theory.version import __version__
from tests.corpus import CONTEXT
from tests.fake_llm import FakeCoder

TRUST = "trusting managers to share information"
DISTRUST = "not trusting managers to share information"
TRUST_DEF = "Employees trust managers to share information."
DISTRUST_DEF = "Employees do not trust managers to share information."


def _entry(label, definition, valence="neutral", revision="1", examples=None):
    return {
        "code": label,
        "key": label,
        "definition": definition,
        "valence": valence,
        "revision": revision,
        "examples": list(examples or []),
        "history": [],
    }


def _book():
    return {
        TRUST: _entry(TRUST, TRUST_DEF, "neutral", examples=[{"quote": "I trust them.", "passage_id": "p1"}]),
        DISTRUST: _entry(
            DISTRUST,
            DISTRUST_DEF,
            "neutral",
            examples=[{"quote": "I do not trust them.", "passage_id": "p2"}],
        ),
    }


def test_version_is_0_6_5():
    assert __version__ == "0.7.7"
    assert POLICY_VERSION == "0.7.7"


def test_negated_near_duplicate_is_proposed_not_merged():
    assert jaccard(TRUST, DISTRUST) == 0.8333333333333334 or abs(jaccard(TRUST, DISTRUST) - 5 / 6) < 1e-9
    left = _entry(TRUST, TRUST_DEF, "neutral")
    right = _entry(DISTRUST, DISTRUST_DEF, "neutral")
    assert not trivial_pair(left, right)
    assert trivial_merges([left, right]) == []
    book = _book()
    decision, reason = classify_comparison(
        {
            "source": DISTRUST,
            "target": TRUST,
            "trivial": True,
            "confidence": 0.95,
            "same_action": True,
            "same_object": True,
            "same_stance": True,
            "contrast_checked": True,
            "material_contrast": False,
            "evidence_same": True,
            "comparison_op": "alias_duplicate",
        },
        book,
    )
    assert decision == "keep_distinct"
    assert reason == "polarity_differs"
    proposals = lexical_comparison_proposals([left, right])
    assert len(proposals) == 1
    assert proposals[0]["method"] == "lexical"
    assert proposals[0]["proposed"] is True
    assert {proposals[0]["source"], proposals[0]["target"]} == {TRUST, DISTRUST}
    applied = apply_sweep(
        book,
        [
            {"code_key": TRUST, "code": TRUST, "quote": "I trust them.", "passage_id": "p1", "doc_id": "d"},
            {"code_key": DISTRUST, "code": DISTRUST, "quote": "I do not trust them.", "passage_id": "p2", "doc_id": "d"},
        ],
        merges=[
            {
                "source": DISTRUST,
                "target": TRUST,
                "trivial": True,
                "confidence": 0.95,
                "rationale": "The labels overlap.",
                "same_action": True,
                "same_object": True,
                "same_stance": True,
                "contrast_checked": True,
                "material_contrast": False,
                "evidence_same": True,
            }
        ],
        candidates=[],
    )
    assert applied["accepted_merges"] == []
    assert applied["rejected_merges"][0]["reason"] == "polarity_differs"
    assert set(applied["codebook"]) == {TRUST, DISTRUST}

    variant = _entry("Trusting managers.", TRUST_DEF, "neutral")
    base = _entry("trusting managers", TRUST_DEF, "neutral")
    assert trivial_pair(variant, base)
    plural = _entry("sharing information", "Employees share information.", "neutral")
    gerund = _entry("share information", "Employees share information.", "neutral")
    assert trivial_pair(plural, gerund)
    contradicted = _entry("Trusting managers.", DISTRUST_DEF, "neutral")
    assert not trivial_pair(variant, contradicted)


def test_pair_cache_reopens_on_any_definition_or_evidence_change():
    book = _book()
    cache = remember_pair_decisions(
        [],
        book,
        [
            {
                "source": TRUST,
                "target": DISTRUST,
                "decision": "reject",
                "confidence": 0.9,
                "rationale": "One trusts managers and the other does not.",
            }
        ],
    )
    assert frozen_pair_keys(cache, book)
    negated = json.loads(json.dumps(book))
    negated[TRUST]["definition"] = DISTRUST_DEF
    assert frozen_pair_keys(cache, negated) == set()
    revised = json.loads(json.dumps(book))
    revised[TRUST]["revision"] = "2"
    assert frozen_pair_keys(cache, revised) == set()
    reassigned = json.loads(json.dumps(book))
    reassigned[TRUST]["examples"] = [{"quote": "Managers keep us posted.", "passage_id": "p9"}]
    assert frozen_pair_keys(cache, reassigned) == set()


def test_sweep_prompts_treat_stance_as_context():
    context = parse_context(CONTEXT)
    messages = build_sweep_messages(
        context=context,
        codes=[{"code": TRUST, "definition": TRUST_DEF}],
        candidates=[],
        prompt_cache=False,
    )
    text = "\n".join(message_text(message) for message in messages)
    check = build_merge_check_messages(
        context=context,
        codebook={TRUST: _entry(TRUST, TRUST_DEF)},
        merges=[{"source": TRUST, "target": DISTRUST, "pair_id": "p0001"}],
        prompt_cache=False,
    )
    check_text = "\n".join(message_text(message) for message in check)
    for prompt in (text, check_text, SWEEP_SYSTEM, IN_BATCH_SYSTEM):
        assert STANCE_POLICY in prompt
        assert "same stance all hold" not in prompt.casefold()
        assert "same incident and the same stance" not in prompt.casefold()
        assert "never accept a positive" not in prompt.casefold()
    assert "not by itself a reason to merge" in CODING_SYSTEM
    assert "expressing liking for the team" in CODING_SYSTEM


def test_concurrency_does_not_widen_the_codebook_window():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schedule = resolve_coding_schedule(batch_size=1, concurrency=4)
    assert schedule["passages_per_request"] == 1
    assert schedule["max_concurrent_requests"] == 4
    assert schedule["passages_per_codebook_update"] == 1
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    explicit = resolve_coding_schedule(batch_size=1, concurrency=4, passages_per_codebook_update=4)
    assert explicit["passages_per_codebook_update"] == 4
    preset = resolve_coding_schedule(procedure="strict_sequential")
    assert preset["review_mode"] == "document"
    assert preset["passages_per_request"] == 1
    assert preset["max_concurrent_requests"] == 1
    assert preset["passages_per_codebook_update"] == 1
    assert preset["pipeline"] is False
    assert preset["parallel_documents"] is False


def _labels(prompt) -> list[str]:
    for message in prompt["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in content
            )
        if not isinstance(content, str) or "codebook" not in content:
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if "codebook" not in payload:
            continue
        return [str(item.get("label") or "") for item in payload["codebook"]]
    return []


def test_strict_sequential_codes_a_document_in_four_rounds():
    text = "\n\n".join(f"I packed suitcase number {index} before leaving." for index in range(1, 5))
    result = open_code(
        [{"id": "one", "name": "One", "text": text}],
        CONTEXT,
        model=FakeCoder(),
        seed=1,
        unit="paragraph",
        sweep_every=100,
        **strict_sequential_settings(),
    )
    rounds = result.metadata["metrics"]["rounds"]
    assert [row["size"] for row in rounds] == [1, 1, 1, 1]
    assert len({row["snapshot_id"] for row in rounds}) == 4
    prompts = [item for item in result.prompt_log if item["kind"] == "code_passage"]
    assert len(prompts) == 4
    seen = [_labels(item) for item in prompts]
    assert seen[0] == []
    for later in seen[1:]:
        assert "packing bags" in later
    assert result.metadata["passages_per_codebook_update"] == 1
    assert result.metadata["pipeline"] is False
    assert result.metadata["review_mode"] == "document"
    assert any(memo["kind"] == "provenance" for memo in result.memos)


def test_singleton_codes_score_ari_one_and_low_set_agreement():
    left = {
        "segments": [
            {"passage_id": "p1", "code": "alpha"},
            {"passage_id": "p2", "code": "beta"},
            {"passage_id": "p3", "code": "gamma"},
        ]
    }
    right = {
        "segments": [
            {"passage_id": "p1", "code": "one"},
            {"passage_id": "p2", "code": "two"},
            {"passage_id": "p3", "code": "three"},
        ]
    }
    assert assignment_ari(left, right) == 1.0
    score = aligned_assignment_agreement(left, right)
    assert score["mean_jaccard"] == 0.0
    assert score["krippendorff_alpha"] < 0.5
    assert aligned_assignment_agreement(left, left)["mean_jaccard"] == 1.0
    assert aligned_assignment_agreement(left, left)["krippendorff_alpha"] == 1.0
