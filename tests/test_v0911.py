"""0.9.11 regressions. Fake models only. No live evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from automated_grounded_theory import __version__, axial_code
from automated_grounded_theory.axial import (
    DEFAULT_AXIAL_TOKEN_BUDGET,
    DEFAULT_MAX_AXIAL_REVIEW_TOKENS,
    AxialProposalBatch,
    _fit_proposal_request,
    _messages,
)
from automated_grounded_theory.batching import request_input_tokens
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import (
    PATTERN_MIN_CASES,
    _EVIDENCE_CLASSES,
    _GROUNDED_STEMS,
    _doc_id_of,
    _endpoint_terms,
    candidate_cross_case_passages,
    case_id_of,
    classify_pair_evidence,
)
from automated_grounded_theory.saturation import STOPPING_QUIET_ROUNDS, stopping_criteria_report
from automated_grounded_theory.validation import meaningful_tokens
from tests.test_v080 import FakeAxialCoder, _corpus


def test_version_is_0911():
    assert __version__ == "0.9.15"
    assert POLICY_VERSION == "0.9.15"
    assert PATTERN_MIN_CASES == 2
    assert DEFAULT_AXIAL_TOKEN_BUDGET == 6000
    assert DEFAULT_MAX_AXIAL_REVIEW_TOKENS == 24000
    assert STOPPING_QUIET_ROUNDS == 2


def _code(code_id: str, label: str) -> dict:
    return {"code_id": code_id, "code": label, "definition": label}


def test_rejected_lexical_matches_do_not_consume_the_proposal_budget(tmp_path):
    """One hundred weak matches stay in the export and off the prompt.

    The useful evidence is one passage that contains both categories. The
    default request budget is 6000 tokens. The proposal call is sent.
    """
    codebook = {
        "c-a": _code("c-a", "alpha protocol briefing"),
        "c-b": _code("c-b", "beta protocol handbook"),
    }
    passages = [
        {
            "passage_id": "p-both",
            "doc_id": "d-both",
            "respondent_id": "R-both",
            "text": "The alpha protocol briefing sat beside the beta protocol handbook.",
        }
    ]
    for index in range(100):
        passages.append(
            {
                "passage_id": f"p-weak-{index:03d}",
                "doc_id": f"d-weak-{index:03d}",
                "respondent_id": f"R-weak-{index:03d}",
                "text": "The protocol was noted.",
            }
        )
    corpus = {
        "codebook": codebook,
        "passages": passages,
        "segments": [
            {"segment_id": "s-a", "code_id": "c-a", "passage_id": "p-both"},
            {"segment_id": "s-b", "code_id": "c-b", "passage_id": "p-both"},
        ],
    }

    class _None(FakeAxialCoder):
        def proposals(self, body):
            del body
            return AxialProposalBatch(relationships=[], nothing_to_propose=True)

    result = axial_code(
        corpus,
        _None(),
        seed=1,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
        output_dir=tmp_path,
    )
    sent = [
        row
        for row in result.prompt_log
        if row.get("kind") == "axial_proposal" and row.get("status") == "validated"
    ]
    assert len(sent) == 1
    assert sent[0].get("reason") != "evidence_does_not_fit"
    assert not any(row.get("reason") == "evidence_does_not_fit" for row in result.prompt_log)
    assert int(sent[0]["estimated_input_tokens"]) <= DEFAULT_AXIAL_TOKEN_BUDGET
    body = json.loads(sent[0]["messages"][-1]["content"])
    assert "p-weak-099" not in sent[0]["messages"][-1]["content"]
    assert "lexical_only_withheld" not in body
    summary = body["lexical_only_withheld_summary"]
    assert summary["count"] == 100
    assert summary["support_cap"] == 1
    assert summary["lexical_only"] is True
    assert {row["passage_id"] for row in body["passages"]} == {"p-both"}
    exported_ids = [row["passage_id"] for row in result.metadata["lexical_only_withheld"]]
    assert exported_ids == [f"p-weak-{index:03d}" for index in range(100)]
    saved = json.loads((Path(tmp_path) / "axial_run.json").read_text(encoding="utf-8"))
    assert [row["passage_id"] for row in saved["metadata"]["lexical_only_withheld"]] == exported_ids


def test_optional_diagnostics_are_dropped_before_passages_are_shortened():
    """A diagnostic list that blows the budget is removed before any passage is."""
    passages = [
        {
            "passage_id": "p-both",
            "text": "alpha protocol briefing meets beta protocol handbook",
            "code_ids": ["c-a", "c-b"],
        },
        {
            "passage_id": "p-extra",
            "text": "alpha protocol briefing still meets beta protocol handbook",
            "code_ids": ["c-a", "c-b"],
        },
    ]
    chunk = [{"from_id": "c-a", "to_id": "c-b"}]
    fitted = {
        "pairs": chunk,
        "passages": [dict(row) for row in passages],
        "categories": [],
        "omitted_diagnostics": {
            "dropped_before_evidence": ["lexical_only_withheld", "lexical_only_withheld_summary"]
        },
    }
    budget = request_input_tokens(_messages("axial_proposal", fitted), AxialProposalBatch)
    withheld = [
        {
            "passage_id": f"p-weak-{index:03d}",
            "code_ids": ["c-a", "c-b"],
            "matched_stems": ["protocol"],
            "lexical_only": True,
            "support_cap": 1,
        }
        for index in range(100)
    ]
    payload = {
        "pairs": chunk,
        "passages": [dict(row) for row in passages],
        "categories": [],
        "lexical_only_withheld": withheld,
        "lexical_only_withheld_summary": {"count": 100, "support_cap": 1, "lexical_only": True},
    }
    with_list = request_input_tokens(_messages("axial_proposal", payload), AxialProposalBatch)
    assert with_list > budget
    estimate = _fit_proposal_request(payload, chunk, AxialProposalBatch, budget)
    assert estimate <= budget
    assert [row["passage_id"] for row in payload["passages"]] == ["p-both", "p-extra"]
    assert "lexical_only_withheld" not in payload
    assert "lexical_only_withheld_summary" not in payload
    assert payload["omitted_diagnostics"]["dropped_before_evidence"] == [
        "lexical_only_withheld",
        "lexical_only_withheld_summary",
    ]
    assert int((payload.get("omitted_evidence") or {}).get("omitted_count") or 0) == 0


def _quiet(rows, *, quiet_rounds=2):
    report = stopping_criteria_report({"n_documents": 3}, rows, quiet_rounds=quiet_rounds)
    return next(row for row in report["criteria"] if row["criterion"] == "sampling_rounds_without_new_directions")


def _budget_round(reason: str = "request_budget") -> dict:
    return {
        "analytic_new_direction_keys": [],
        "reread_unsent_pairs": [
            {
                "reason": reason,
                "direction_id": "dir-1-0001",
                "direction_key": "gap:c1",
                "passage_id": "p-long",
            }
        ],
        "reread_status": "budget_limited",
        "reread_stop": reason,
    }


def test_request_budget_rereads_do_not_count_as_quiet_rounds():
    """Three rounds, the same long passage, each reread blocked by request budget.

    The quiet criterion looks at those rounds. It is not met.
    """
    rows = [_budget_round("request_budget") for _ in range(3)]
    quiet = _quiet(rows, quiet_rounds=2)
    assert quiet["status"] == "not_evaluable"
    assert quiet["observed"] is None
    assert quiet["reason"] == "selected_pairs_unsent_or_unrecorded"
    whole = _quiet(rows, quiet_rounds=3)
    assert whole["status"] == "not_evaluable"
    assert whole["observed"] is None
    for reason in (
        "token_allowance",
        "schema_unparsed",
        "parse_failed",
        "timeout",
        "incomplete",
        "missing_passage",
        "unknown_reason",
    ):
        blocked = _quiet([_budget_round(reason), _budget_round(reason)])
        assert blocked["status"] == "not_evaluable"
        assert blocked["observed"] is None
    failed = _quiet(
        [
            {
                "analytic_new_direction_keys": [],
                "reread_unsent_pairs": [],
                "reread_failed_pairs": [
                    {"reason": "schema_unparsed", "direction_key": "gap:c1", "passage_id": "p-long"}
                ],
            },
            {
                "analytic_new_direction_keys": [],
                "reread_unsent_pairs": [],
                "reread_failed_pairs": [
                    {"reason": "schema_unparsed", "direction_key": "gap:c1", "passage_id": "p-long"}
                ],
            },
        ]
    )
    assert failed["status"] == "not_evaluable"
    assert failed["observed"] is None
    unresolved = _quiet(
        [
            {
                "analytic_new_direction_keys": [],
                "reread_unsent_pairs": [],
                "reread_status": "timeout",
                "reread_stop": "timeout",
            },
            {
                "analytic_new_direction_keys": [],
                "reread_unsent_pairs": [],
                "reread_status": "timeout",
                "reread_stop": "timeout",
            },
        ]
    )
    assert unresolved["status"] == "not_evaluable"
    assert unresolved["observed"] is None
    completed = _quiet(
        [
            {
                "analytic_new_direction_keys": [],
                "reread_unsent_pairs": [{"reason": "completed_pair", "direction_key": "gap:c1"}],
            },
            {
                "analytic_new_direction_keys": [],
                "reread_unsent_pairs": [{"reason": "completed_pair", "direction_key": "gap:c1"}],
            },
        ]
    )
    assert completed["status"] == "met"
    closed = _quiet(
        [
            {
                "analytic_new_direction_keys": [],
                "closed_direction_keys": ["gap:c1"],
                "reread_unsent_pairs": [{"reason": "request_budget", "direction_key": "gap:c1"}],
            },
            {
                "analytic_new_direction_keys": [],
                "closed_direction_keys": ["gap:c1"],
                "reread_unsent_pairs": [{"reason": "request_budget", "direction_key": "gap:c1"}],
            },
        ]
    )
    assert closed["status"] == "met"
    empty = _quiet(
        [
            {"analytic_new_direction_keys": [], "reread_unsent_pairs": []},
            {"analytic_new_direction_keys": [], "reread_unsent_pairs": []},
        ]
    )
    assert empty["status"] == "met"
    missing = _quiet([{"analytic_new_direction_keys": []}, {"analytic_new_direction_keys": []}])
    assert missing["status"] == "not_evaluable"
    assert missing["observed"] is None


def _classify_pair_evidence_0910(category_ids, passages, segments, codebook, *, pattern_min_cases=PATTERN_MIN_CASES):
    """The 0.9.10 nested-loop classifier, kept here as the equivalence oracle."""
    ids: list[str] = []
    for item in category_ids:
        text = str(item)
        if text and text not in ids:
            ids.append(text)
    by_id: dict[str, dict] = {}
    ordered_passages: list[dict] = []
    for item in passages or []:
        if not isinstance(item, dict) or item.get("synthetic"):
            continue
        passage_id = str(item.get("passage_id") or "")
        if not passage_id or passage_id in by_id:
            continue
        by_id[passage_id] = item
        ordered_passages.append(item)
    occurrences: dict[str, list[str]] = {code_id: [] for code_id in ids}
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        code_id = str(segment.get("code_id") or "")
        passage_id = str(segment.get("passage_id") or "")
        if code_id not in occurrences or passage_id not in by_id:
            continue
        if passage_id not in occurrences[code_id]:
            occurrences[code_id].append(passage_id)
    cases: dict[str, list[str]] = {}
    docs: dict[str, list[str]] = {}
    for code_id, passage_ids in occurrences.items():
        case_ids: list[str] = []
        doc_ids: list[str] = []
        for passage_id in passage_ids:
            passage = by_id[passage_id]
            case_id = case_id_of(passage)
            doc_id = _doc_id_of(passage)
            if case_id and case_id not in case_ids:
                case_ids.append(case_id)
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        cases[code_id] = case_ids
        docs[code_id] = doc_ids
    classes: dict[str, list[tuple[str, str]]] = {name: [] for name in _EVIDENCE_CLASSES}
    lexical_only_pairs = 0
    reaching = 0
    minimum = max(1, int(pattern_min_cases or PATTERN_MIN_CASES))
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            left_passages = set(occurrences.get(left) or [])
            right_passages = set(occurrences.get(right) or [])
            shared_cases = [case_id for case_id in cases.get(left) or [] if case_id in set(cases.get(right) or [])]
            if len(shared_cases) >= minimum:
                reaching += 1
            if left_passages and right_passages and left_passages & right_passages:
                classes["same_passage"].append((left, right))
                continue
            if shared_cases:
                classes["same_respondent"].append((left, right))
                continue
            from_terms = _endpoint_terms(left, codebook, segments)
            to_terms = _endpoint_terms(right, codebook, segments)
            if not from_terms or not to_terms:
                continue
            lexical = False
            stem_only = False
            occupied = set(cases.get(left) or []) | set(cases.get(right) or [])
            for passage in ordered_passages:
                case_id = case_id_of(passage)
                if occupied and case_id in occupied:
                    continue
                tokens = meaningful_tokens(str(passage.get("text") or ""))
                from_overlap = tokens & from_terms
                to_overlap = tokens & to_terms
                if not from_overlap or not to_overlap:
                    continue
                if len(from_overlap) >= _GROUNDED_STEMS and len(to_overlap) >= _GROUNDED_STEMS:
                    lexical = True
                    break
                stem_only = True
            if lexical:
                classes["lexical_cross_case"].append((left, right))
            elif stem_only:
                lexical_only_pairs += 1
    single_document = sum(1 for code_id in ids if len(docs.get(code_id) or []) == 1)
    return {
        "by_class": {name: [{"from_id": left, "to_id": right} for left, right in rows] for name, rows in classes.items()},
        "evidence_available": {name: len(rows) for name, rows in classes.items()},
        "plausible_count": sum(len(rows) for rows in classes.values()),
        "lexical_only_count": lexical_only_pairs,
        "single_document_code_count": single_document,
        "pairs_reaching_pattern_min_cases": reaching,
        "pattern_min_cases": minimum,
        "code_count": len(ids),
    }


def _evidence_fixture():
    codebook = {
        "c-a": _code("c-a", "shared passage wording"),
        "c-b": _code("c-b", "shared passage wording extra"),
        "c-c": _code("c-c", "morning checklist"),
        "c-d": _code("c-d", "evening checklist"),
        "c-e": _code("c-e", "alpha briefing protocol"),
        "c-f": _code("c-f", "beta handbook manual"),
        "c-g": _code("c-g", "gamma tower signal"),
        "c-h": _code("c-h", "delta beacon lantern"),
        "c-i": _code("c-i", "orphan concept"),
    }
    passages = [
        {"passage_id": "p-one", "doc_id": "d-lex", "respondent_id": "r-lex", "text": "alpha beta"},
        {
            "passage_id": "p-lex",
            "doc_id": "d-lex2",
            "respondent_id": "r-lex",
            "text": "alpha briefing meets beta handbook",
        },
        {"passage_id": "p-weak", "doc_id": "d-weak", "respondent_id": "r-weak", "text": "gamma delta"},
        {"passage_id": "p-same", "doc_id": "d1", "respondent_id": "r1", "text": "shared passage wording extra"},
        {"passage_id": "p-same-2", "doc_id": "d2", "respondent_id": "r2", "text": "shared passage wording extra again"},
        {"passage_id": "p-c", "doc_id": "d-c", "respondent_id": "r-shared", "text": "morning checklist only"},
        {"passage_id": "p-d", "doc_id": "d-d", "respondent_id": "r-shared", "text": "evening checklist only"},
        {"passage_id": "p-solo", "doc_id": "d-solo", "respondent_id": "r-solo", "text": "orphan concept sits alone"},
    ]
    segments = [
        {"code_id": "c-a", "passage_id": "p-same"},
        {"code_id": "c-b", "passage_id": "p-same"},
        {"code_id": "c-a", "passage_id": "p-same-2"},
        {"code_id": "c-b", "passage_id": "p-same-2"},
        {"code_id": "c-c", "passage_id": "p-c"},
        {"code_id": "c-d", "passage_id": "p-d"},
        {"code_id": "c-i", "passage_id": "p-solo"},
    ]
    return list(codebook), passages, segments, codebook


def test_classify_pair_evidence_matches_the_0910_result():
    ids, passages, segments, codebook = _evidence_fixture()
    previous = _classify_pair_evidence_0910(ids, passages, segments, codebook)
    current = classify_pair_evidence(ids, passages, segments, codebook)
    assert previous["by_class"]["same_passage"]
    assert previous["by_class"]["same_respondent"]
    assert previous["by_class"]["lexical_cross_case"]
    assert previous["lexical_only_count"] > 0
    assert previous["pairs_reaching_pattern_min_cases"] >= 1
    assert previous["single_document_code_count"] >= 1
    assert current == previous


def test_classify_pair_evidence_tokenizes_each_passage_once(monkeypatch):
    """60 categories and 400 passages stay within one tokenization per text.

    The 0.9.10 nested loop measured about 40 seconds on this size. The bound
    here is generous. The operation count is the stable check.
    """
    n_codes = 60
    n_passages = 400
    ids = [f"c-{index:03d}" for index in range(n_codes)]
    codebook = {code_id: _code(code_id, f"alpha{index:03d} bravo{index:03d}") for index, code_id in enumerate(ids)}
    passages = [
        {
            "passage_id": f"p-{index:03d}",
            "doc_id": f"d-{index:03d}",
            "respondent_id": f"r-{index:03d}",
            "text": f"note{index:03d} alpha000 bravo000",
        }
        for index in range(n_passages)
    ]
    calls = {"n": 0}
    real = meaningful_tokens

    def _counted(text: str):
        calls["n"] += 1
        return real(text)

    monkeypatch.setattr("automated_grounded_theory.relationship_tiers.meaningful_tokens", _counted)
    started = time.perf_counter()
    result = classify_pair_evidence(ids, passages, [], codebook)
    elapsed = time.perf_counter() - started
    assert calls["n"] == n_passages + n_codes
    assert elapsed < 8.0
    assert result["code_count"] == n_codes
    assert result["pattern_min_cases"] == 2


def test_classification_is_skipped_when_no_proposal_is_scheduled(monkeypatch):
    calls = {"n": 0}
    real = classify_pair_evidence

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("automated_grounded_theory.axial.classify_pair_evidence", _wrapped)
    skipped = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=1,
        max_axial_review_calls=0,
        interrupt=False,
    )
    assert calls["n"] == 0
    assert skipped.metadata["pair_evidence"] == {
        "classification_skipped": True,
        "reason": "no_proposal_requests",
        "lexical_only_count": None,
        "by_class": None,
    }
    assert "pattern_reach_warning" not in skipped.metadata
    assert not any("No candidate pair can reach pattern_min_cases" in str(item) for item in skipped.warnings)
    scheduled = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=1,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    assert calls["n"] == 1
    assert scheduled.metadata["pair_evidence"]["classification_skipped"] is False
    assert "pattern_reach_warning" in scheduled.metadata


def test_cross_case_offer_does_not_depend_on_pair_order():
    """A weak overlap for A-B must not hide a grounded overlap for B-C."""
    codebook = {
        "c-a": _code("c-a", "alpha"),
        "c-b": _code("c-b", "bravo charlie"),
        "c-c": _code("c-c", "delta echo"),
    }
    passages = {
        "p-shared": {
            "passage_id": "p-shared",
            "doc_id": "d-shared",
            "respondent_id": "R-shared",
            "text": "alpha bravo charlie delta echo",
        }
    }
    weak_first = [{"from_id": "c-a", "to_id": "c-b"}, {"from_id": "c-b", "to_id": "c-c"}]
    strong_first = [{"from_id": "c-b", "to_id": "c-c"}, {"from_id": "c-a", "to_id": "c-b"}]
    weak_only: list[dict] = []
    strong_only: list[dict] = []
    weak_offered = candidate_cross_case_passages(
        passages, codebook, weak_first, limit=2, lexical_only=weak_only
    )
    strong_offered = candidate_cross_case_passages(
        passages, codebook, strong_first, limit=2, lexical_only=strong_only
    )
    assert [row["passage_id"] for row in weak_offered] == ["p-shared"]
    assert [row["passage_id"] for row in strong_offered] == ["p-shared"]
    assert weak_offered[0]["code_ids"] == ["c-b", "c-c"]
    assert strong_offered[0]["code_ids"] == ["c-b", "c-c"]
    assert weak_offered[0]["candidate_cross_case"] is True
    assert [(row["passage_id"], row["code_ids"]) for row in weak_offered] == [
        (row["passage_id"], row["code_ids"]) for row in strong_offered
    ]
    assert [row["code_ids"] for row in weak_only] == [["c-a", "c-b"]]
    assert strong_only == []
