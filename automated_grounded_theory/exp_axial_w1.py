"""Opt-in experiment ``exp-axial-w1``: a wider axial round of unexamined pairs.

The default axial planner is unchanged. This module runs only when the caller
asks for the experiment. Focused rereading is not part of it. The inputs it
reads are the caller's frozen corpus and a local examined-pair file. That file
is never written by this package and must not be committed.

No path here calls a paid or live model. The caller supplies the model. Tests
use a fake.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from automated_grounded_theory.axial import (
    AxialError,
    _KIND_PROMPTS,
    axial_code,
)
from automated_grounded_theory.backends import chat_model_settings, describe_model
from automated_grounded_theory.batching import estimate_tokens
from automated_grounded_theory.evaluation_plan import INPUT_USD_PER_MILLION, OUTPUT_USD_PER_MILLION
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES, classify_pair_evidence

EXPERIMENT_ID = "exp-axial-w1"
DRAW_SAME_RESPONDENT = 16
DRAW_LEXICAL = 8
DEFAULT_SEED = 4401
CODING_CALLS = 36
CODING_TOKENS = 216_000
REQUEST_INPUT_CAP = 6000
SHARED_RETRY_CALLS = 6
SATURATION_REVIEW_CALLS = 1
HARD_CAP_USD = 0.60
KILL_GUARD_USD = 0.55
REDUCE_STEP = 4
CLASS_SAME = "same_respondent"
CLASS_LEXICAL = "lexical_cross_case"
# Quotes and rationales in review payloads are already cut at 800 characters.
FIELD_CHAR_CAP = 800
LEXICAL_ELIGIBILITY_RULE = "lexical_cross_case_v1"
LEXICAL_ELIGIBILITY_STATEMENT = (
    "A lexical pair is eligible for this draw when it is absent from the examined-pair "
    "history and classify_pair_evidence places it in lexical_cross_case. That class requires "
    "a passage whose respondent is not an endpoint case and whose text contains at least two "
    "distinctive stems of each endpoint. pattern_min_cases does not admit or exclude the class. "
    "It still governs whether a finished relationship is downstream-eligible."
)
FROZEN_TOKEN_PRICES = {
    "input_usd_per_million": INPUT_USD_PER_MILLION,
    "output_usd_per_million": OUTPUT_USD_PER_MILLION,
    "model": "grok-4.20-0309-non-reasoning",
    "source": "evaluation_plan.LIVE_PLAN prices",
}


def _canonical(payload: Any) -> str:
    """Same JSON encoding as a core match fingerprint."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _pair_key(row: Mapping[str, Any] | Sequence[str]) -> tuple[str, str]:
    if isinstance(row, Mapping):
        left = str(row.get("from_id") or "")
        right = str(row.get("to_id") or "")
    else:
        left, right = str(row[0]), str(row[1])
    return tuple(sorted((left, right)))


def _package_repo() -> Path:
    """The repository that contains this package, not the process working directory."""
    return Path(__file__).resolve().parents[1]


def _all_prompts() -> dict[str, str]:
    from automated_grounded_theory.axial import _AXIAL_REVIEW_SYSTEM, _RELATION_REVIEW_INSTRUCTION
    from automated_grounded_theory.decision_review import SYSTEM as DECISION_SYSTEM

    prompts = {str(name): str(text) for name, text in _KIND_PROMPTS.items()}
    prompts["axial_review_system"] = _AXIAL_REVIEW_SYSTEM
    prompts["relation_review_instruction"] = _RELATION_REVIEW_INSTRUCTION
    prompts["decision_review_system"] = DECISION_SYSTEM
    return prompts


def policy_fingerprint(model: Any = None) -> str:
    """Eligibility, model, and prompts. A change here rejects reuse."""
    return _sha({
        "experiment": EXPERIMENT_ID,
        "policy_version": POLICY_VERSION,
        "pattern_min_cases": PATTERN_MIN_CASES,
        "one_document_one_respondent": False,
        "same_respondent_rule": "single_document_cooccurrence_1",
        "lexical_rule": LEXICAL_ELIGIBILITY_RULE,
        "evidence_order": ["same_passage", "same_respondent", "lexical_cross_case"],
        "model": describe_model(model) if model is not None else "",
        "prompts": _all_prompts(),
    })


def proposal_request_fingerprint(
    from_id: str,
    to_id: str,
    *,
    evidence: Sequence[Any] | None = None,
    model: Any = None,
    passage_ids: Sequence[str] | None = None,
) -> str:
    """Pair, evidence payload, model, and passage set.

    The evidence rows are the passages that would be sent. A different passage
    set or a different model is a different request.
    """
    left, right = _pair_key({"from_id": from_id, "to_id": to_id})
    rows = []
    for item in evidence or []:
        if isinstance(item, Mapping):
            rows.append({
                "passage_id": item.get("passage_id"),
                "doc_id": item.get("doc_id"),
                "text": item.get("text"),
            })
    rows.sort(key=lambda row: str(row.get("passage_id") or ""))
    ids = sorted({str(item) for item in (passage_ids or []) if str(item)} | {str(row.get("passage_id") or "") for row in rows if row.get("passage_id")})
    return _sha({
        "experiment": EXPERIMENT_ID,
        "from_id": left,
        "to_id": right,
        "evidence": rows,
        "model": describe_model(model) if model is not None else "",
        "passage_ids": ids,
    })


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_package_repo()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _as_book(codebook: Any) -> dict[str, dict]:
    from automated_grounded_theory.axial import _book

    return _book(codebook)


def startup_fingerprint(source: Mapping[str, Any], model: Any, *, commit: str | None = None) -> dict[str, Any]:
    """Commit, export manifest, model config, prompts, and eligibility rules.

    The export manifest is the axial evidence hash, which already accepts a
    list codebook. Prompts include the review prompts. The commit is read
    from this package's repository.
    """
    from automated_grounded_theory.axial import _evidence_source_hash

    book = _as_book(source.get("codebook"))
    normalized = dict(source)
    normalized["codebook"] = book
    raised = [str(item) for item in (source.get("raised_code_ids") or (source.get("metadata") or {}).get("raised_code_ids") or [])]
    manifest = _sha({
        "evidence": _evidence_source_hash(normalized),
        "raised_code_ids": sorted(raised),
        "code_ids": sorted(book),
    })
    eligibility = _sha({
        "pattern_min_cases": PATTERN_MIN_CASES,
        "one_document_one_respondent": False,
        "policy_version": POLICY_VERSION,
        "experiment": EXPERIMENT_ID,
        "lexical_rule": LEXICAL_ELIGIBILITY_RULE,
    })
    return {
        "commit": _git_commit() if commit is None else str(commit),
        "export_manifest": manifest,
        "model_config": _sha({
            "model": describe_model(model),
            "settings": chat_model_settings(model),
            "prices": {
                "input_usd_per_million": FROZEN_TOKEN_PRICES["input_usd_per_million"],
                "output_usd_per_million": FROZEN_TOKEN_PRICES["output_usd_per_million"],
            },
        }),
        "prompts": _sha(_all_prompts()),
        "eligibility_rules": eligibility,
    }


def fingerprint_differences(frozen: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    names = ("commit", "export_manifest", "model_config", "prompts", "eligibility_rules")
    return [name for name in names if frozen.get(name) != current.get(name)]


def split_preserving_ratio(total: int) -> tuple[int, int]:
    """Keep the 16:8 ratio as nearly as an integer split allows.

    Same-respondent pairs take two thirds, rounded half up. A positive total
    keeps both classes when it can.
    """
    total = max(0, int(total))
    if total == 0:
        return 0, 0
    same = (total * 2 + 1) // 3
    lexical = total - same
    if total >= 2 and same == 0:
        same, lexical = 1, total - 1
    if total >= 2 and lexical == 0:
        lexical, same = 1, total - 1
    return same, lexical


def _prices(token_prices: Mapping[str, Any] | None = None) -> tuple[float, float]:
    """Frozen model rates. A caller price does not replace them."""
    del token_prices
    return float(FROZEN_TOKEN_PRICES["input_usd_per_million"]), float(FROZEN_TOKEN_PRICES["output_usd_per_million"])


def _stage_max_output() -> dict[str, int]:
    """Worst-case output of one call, in the character heuristic.

    Each text field is the 800-character cap already used when a review quote
    is truncated. The figure is that stage's max output for the preflight.
    """
    blob = "m" * FIELD_CHAR_CAP
    payloads = {
        "axial_proposal": {
            "relationships": [{
                "from_id": "c" * 32,
                "to_id": "c" * 32,
                "paradigm_role": "intervening_condition",
                "statement": blob,
                "passage_id": "p" * 32,
                "quote": blob,
                "quote_from": blob,
                "quote_to": blob,
                "case_quotes": [{"passage_id": "p" * 32, "quote": blob, "endpoint": "both"}],
            }],
            "nothing_to_propose": False,
        },
        "statement_copy_retry": None,
        "axial_counterexample": {
            "counterexamples": [{
                "relationship_id": "r" * 32,
                "passage_id": "p" * 32,
                "judgment": "contradicts",
                "quote": blob,
                "rationale": blob,
            }],
        },
        "axial_relation_review": {
            "relationships": [{
                "relationship_id": "r" * 32,
                "disposition": "accepted",
                "paradigm_role": "intervening_condition",
                "rationale": blob,
            }],
        },
        "axial_saturation_review": {
            "assessment": "not_saturated",
            "rationale": blob,
        },
        "axial_core_review": {
            "decision": "select",
            "abstention_reason": "insufficient_evidence",
            "core_id": "c" * 32,
            "rationale": blob,
        },
    }
    payloads["statement_copy_retry"] = payloads["axial_proposal"]
    return {name: estimate_tokens(json.dumps(body, ensure_ascii=False)) for name, body in payloads.items()}


STAGE_MAX_OUTPUT = _stage_max_output()


def workload_calls(same: int, lexical: int) -> dict[str, Any]:
    """Worst-case calls. Retries are one shared pool of six.

    Relation review is one call per proposed row, because a row that fills the
    request is not packed with another. Counterexamples are one per row. A
    counterexample that does not fit the coding ceiling is required_but_blocked.
    """
    pairs = max(0, int(same)) + max(0, int(lexical))
    retries = SHARED_RETRY_CALLS if pairs else 0
    relation = pairs
    counter_required = pairs
    call_room = max(0, CODING_CALLS - pairs - retries)
    token_room = max(0, (CODING_TOKENS // REQUEST_INPUT_CAP) - pairs - retries) if REQUEST_INPUT_CAP else 0
    counter_fit = min(counter_required, call_room, token_room)
    blocked = max(0, counter_required - counter_fit)
    # Relation review is not part of the coding ceiling. Its allowance is the
    # row count, so a fat row still has its own call.
    relation_fit = relation
    relation_shortfall = 0
    return {
        "axial_proposal": pairs,
        "statement_copy_retry": retries,
        "axial_counterexample": counter_fit,
        "axial_counterexample_required": counter_required,
        "axial_counterexample_required_but_blocked": blocked,
        "axial_relation_review": relation_fit,
        "axial_relation_review_required": relation,
        "axial_relation_review_shortfall": relation_shortfall,
        "axial_saturation_review": SATURATION_REVIEW_CALLS if pairs else 0,
    }


def preflight_workload(
    same: int,
    lexical: int,
    *,
    token_prices: Mapping[str, Any] | None = None,
    per_call_input: int = REQUEST_INPUT_CAP,
    core_candidates: int = 0,
) -> dict[str, Any]:
    """Worst case against the coding ceiling and the frozen dollar cap.

    Input is the per-request cap. Output is that stage's max output. Dollars
    use the frozen model rates, $1.25 per million input and $2.50 per million
    output. ``token_prices`` does not replace those rates. Core selection is
    not part of this experiment, so ``core_candidates`` is ignored and the
    preflight has no core line.
    """
    del token_prices, core_candidates
    calls = dict(workload_calls(same, lexical))
    sent = {
        "axial_proposal": int(calls["axial_proposal"]),
        "statement_copy_retry": int(calls["statement_copy_retry"]),
        "axial_counterexample": int(calls["axial_counterexample"]),
        "axial_relation_review": int(calls["axial_relation_review"]),
        "axial_saturation_review": int(calls["axial_saturation_review"]),
    }
    per_call = max(0, int(per_call_input))
    input_tokens = {name: count * per_call for name, count in sent.items()}
    output_tokens = {name: count * int(STAGE_MAX_OUTPUT[name]) for name, count in sent.items()}
    coding_names = ("axial_proposal", "statement_copy_retry", "axial_counterexample")
    coding_calls = sum(sent[name] for name in coding_names)
    coding_input = sum(input_tokens[name] for name in coding_names)
    total_input = sum(input_tokens.values())
    total_output = sum(output_tokens.values())
    in_rate, out_rate = _prices()
    input_dollars = round(total_input / 1_000_000 * in_rate, 6)
    output_dollars = round(total_output / 1_000_000 * out_rate, 6)
    dollars = round(input_dollars + output_dollars, 6)
    coding_ok = coding_calls <= CODING_CALLS and coding_input <= CODING_TOKENS
    blocked = int(calls["axial_counterexample_required_but_blocked"])
    relation_shortfall = int(calls["axial_relation_review_shortfall"])
    dollar_ok = dollars <= HARD_CAP_USD
    fits = bool(coding_ok and dollar_ok and blocked == 0 and relation_shortfall == 0)
    reasons = []
    if not coding_ok:
        reasons.append("coding_ceiling")
    if blocked:
        reasons.append("counterexample_required_but_blocked")
    if relation_shortfall:
        reasons.append("relation_review_shortfall")
    if not dollar_ok:
        reasons.append("dollar_hard_cap")
    return {
        "experiment": EXPERIMENT_ID,
        "same_respondent": int(same),
        "lexical_cross_case": int(lexical),
        "pairs": int(same) + int(lexical),
        "calls": calls,
        "calls_sent": sent,
        "coding_calls": coding_calls,
        "coding_input_tokens": coding_input,
        "input_tokens_by_stage": input_tokens,
        "output_tokens_by_stage": output_tokens,
        "stage_max_output": dict(STAGE_MAX_OUTPUT),
        "total_calls": sum(sent.values()),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "per_call_input_tokens": per_call,
        "prices": dict(FROZEN_TOKEN_PRICES),
        "input_dollars": input_dollars,
        "output_dollars": output_dollars,
        "dollars": dollars,
        "hard_cap_usd": HARD_CAP_USD,
        "kill_guard_usd": KILL_GUARD_USD,
        "coding_call_ceiling": CODING_CALLS,
        "coding_token_ceiling": CODING_TOKENS,
        "fits_coding_ceiling": coding_ok,
        "fits_dollar_cap": dollar_ok,
        "counterexample_required_but_blocked": blocked,
        "relation_review_shortfall": relation_shortfall,
        "fits": fits,
        "reasons": reasons,
        "note": (
            "Worst case prices every sent call at the 6000-token input cap and at that "
            "stage's max output. Rates are the frozen model config: $1.25 per million "
            "input tokens and $2.50 per million output tokens. A counterexample that "
            "does not fit the coding ceiling is required_but_blocked. Relation review "
            "is one call per row and is not limited by the coding ceiling. Core "
            "selection is not requested. The call total can exceed the coding-only "
            "ceiling because relation review and saturation are separate."
        ),
    }


def reduce_scope(
    token_prices: Mapping[str, Any] | None = None,
    *,
    per_call_input: int = REQUEST_INPUT_CAP,
    core_candidates: int = 0,
) -> dict[str, Any]:
    """Shrink the 16+8 draw by four pairs until the worst case fits.

    The class ratio stays as close to 2:1 as the integer split allows. The
    original 16+8 preflight is kept either way.
    """
    same, lexical = DRAW_SAME_RESPONDENT, DRAW_LEXICAL
    original = preflight_workload(
        same, lexical, token_prices=token_prices, per_call_input=per_call_input, core_candidates=core_candidates,
    )
    steps = [original]
    current = original
    while current["fits"] is False and (same + lexical) >= REDUCE_STEP:
        total = same + lexical - REDUCE_STEP
        same, lexical = split_preserving_ratio(total)
        current = preflight_workload(
            same, lexical, token_prices=token_prices, per_call_input=per_call_input, core_candidates=core_candidates,
        )
        steps.append(current)
        if total == 0:
            break
    return {
        "original": original,
        "reduced": current,
        "steps": steps,
        "reduced_scope": {
            "same_respondent": same,
            "lexical_cross_case": lexical,
            "pairs": same + lexical,
        },
        "scope_changed": (same, lexical) != (DRAW_SAME_RESPONDENT, DRAW_LEXICAL),
    }


def load_examined_history(path: str | Path) -> dict[str, Any]:
    """Read the local exclusion file. This function does not write it."""
    place = Path(path)
    if not place.is_file():
        raise AxialError(f"Examined-pair history was not found at {place}.")
    try:
        payload = json.loads(place.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AxialError(f"Examined-pair history is not JSON: {place}.") from exc
    if not isinstance(payload, dict):
        raise AxialError("Examined-pair history must be a JSON object.")
    examined = []
    for row in payload.get("examined_pairs") or []:
        if isinstance(row, Mapping):
            key = _pair_key(row)
            if key[0] and key[1]:
                examined.append({"from_id": key[0], "to_id": key[1]})
    spend = payload.get("historical_expenditure")
    if not isinstance(spend, Mapping):
        spend = {"calls": None, "input_tokens": None, "dollars": None}
    reuse = [dict(row) for row in (payload.get("reuse") or []) if isinstance(row, Mapping)]
    return {
        "path": str(place),
        "examined_pairs": examined,
        "historical_expenditure": {
            "calls": spend.get("calls"),
            "input_tokens": spend.get("input_tokens"),
            "dollars": spend.get("dollars"),
            "note": (
                "Historical expenditure is prior spend. It is not added to this run."
            ),
        },
        "reuse": reuse,
    }


def _endpoint_cases(source: Mapping[str, Any]) -> dict[str, set[str]]:
    from automated_grounded_theory.relationship_tiers import case_id_of

    passages = {
        str(row.get("passage_id") or ""): row
        for row in (source.get("passages") or [])
        if isinstance(row, Mapping)
    }
    found: dict[str, set[str]] = {}
    for segment in source.get("segments") or []:
        if not isinstance(segment, Mapping):
            continue
        code_id = str(segment.get("code_id") or "")
        passage = passages.get(str(segment.get("passage_id") or ""))
        if not code_id or not isinstance(passage, Mapping):
            continue
        case_id = case_id_of(passage)
        if case_id:
            found.setdefault(code_id, set()).add(case_id)
    return found


def _with_original_cases(rows: Sequence[Mapping[str, Any]], cases: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        item = dict(row)
        left = cases.get(str(item.get("from_id") or ""), set())
        right = cases.get(str(item.get("to_id") or ""), set())
        item["original_case_ids"] = sorted(left | right)
        annotated.append(item)
    return annotated


def _eligible_pools(
    evidence: Mapping[str, Any],
    examined: set[tuple[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    by_class = evidence.get("by_class") if isinstance(evidence.get("by_class"), Mapping) else {}
    same: list[dict[str, Any]] = []
    for row in by_class.get(CLASS_SAME) or []:
        if not isinstance(row, Mapping):
            continue
        key = _pair_key(row)
        if not key[0] or not key[1] or key in examined:
            continue
        if int(row.get("document_cooccurrence") or 0) != 1:
            continue
        same.append({"from_id": key[0], "to_id": key[1], "band": CLASS_SAME, "evidence_class": CLASS_SAME})
    lexical: list[dict[str, Any]] = []
    for row in by_class.get(CLASS_LEXICAL) or []:
        if not isinstance(row, Mapping):
            continue
        key = _pair_key(row)
        if not key[0] or not key[1] or key in examined:
            continue
        lexical.append({"from_id": key[0], "to_id": key[1], "band": CLASS_LEXICAL, "evidence_class": CLASS_LEXICAL})
    same.sort(key=lambda row: (row["from_id"], row["to_id"]))
    lexical.sort(key=lambda row: (row["from_id"], row["to_id"]))
    return {CLASS_SAME: same, CLASS_LEXICAL: lexical}


def lexical_eligibility_report(
    evidence: Mapping[str, Any],
    examined_pairs: Sequence[Mapping[str, Any]],
    lexical_rows: Sequence[Mapping[str, Any]],
    *,
    required: int = DRAW_LEXICAL,
) -> dict[str, Any]:
    """Whether each lexical pair is unexamined and in the existing class.

    The rule is ``lexical_cross_case_v1``. A pair that fails it is listed.
    The caller does not replace it with another class.
    """
    examined = {_pair_key(row) for row in examined_pairs if isinstance(row, Mapping)}
    by_class = evidence.get("by_class") if isinstance(evidence.get("by_class"), Mapping) else {}
    class_keys = {_pair_key(row) for row in (by_class.get(CLASS_LEXICAL) or []) if isinstance(row, Mapping)}
    meeting = []
    failed = []
    for row in lexical_rows:
        if not isinstance(row, Mapping):
            continue
        key = _pair_key(row)
        unexamined = key not in examined
        in_class = key in class_keys
        record = {"from_id": key[0], "to_id": key[1], "previously_unexamined": unexamined, "in_class": in_class}
        if unexamined and in_class:
            meeting.append(record)
        else:
            failed.append(record)
    return {
        "rule": LEXICAL_ELIGIBILITY_RULE,
        "statement": LEXICAL_ELIGIBILITY_STATEMENT,
        "pattern_min_cases_is_the_rule": False,
        "required": int(required),
        "meeting_rule": len(meeting),
        "eligible": len(meeting) >= int(required) and not failed,
        "failed": failed,
    }


def select_unexamined_draw(
    evidence: Mapping[str, Any],
    examined_pairs: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    same_count: int = DRAW_SAME_RESPONDENT,
    lexical_count: int = DRAW_LEXICAL,
) -> dict[str, Any]:
    """Seeded draw of unexamined single-document and lexical pairs.

    A short class stops the draw. Pairs are not borrowed from the other class.
    """
    examined = {_pair_key(row) for row in examined_pairs if isinstance(row, Mapping)}
    pools = _eligible_pools(evidence, examined)
    shortfall = {}
    if len(pools[CLASS_SAME]) < int(same_count):
        shortfall[CLASS_SAME] = {
            "needed": int(same_count),
            "available": len(pools[CLASS_SAME]),
        }
    if len(pools[CLASS_LEXICAL]) < int(lexical_count):
        shortfall[CLASS_LEXICAL] = {
            "needed": int(lexical_count),
            "available": len(pools[CLASS_LEXICAL]),
        }
    lexical_report = lexical_eligibility_report(
        evidence, examined_pairs, pools[CLASS_LEXICAL], required=int(lexical_count),
    )
    if shortfall or lexical_report["eligible"] is False:
        return {
            "status": "stopped",
            "reason": "class_shortfall" if shortfall else "lexical_ineligible",
            "seed": int(seed),
            "shortfall": shortfall,
            "lexical_eligibility": lexical_report,
            "available": {name: len(rows) for name, rows in pools.items()},
            "round": [],
            "note": (
                "A class had fewer unexamined eligible pairs than the draw, or a lexical "
                "pair missed lexical_cross_case_v1. The run stops. Pairs are not taken from the other class."
            ),
        }
    rng = random.Random(int(seed))
    same = rng.sample(pools[CLASS_SAME], int(same_count))
    lexical = rng.sample(pools[CLASS_LEXICAL], int(lexical_count))
    drawn_report = lexical_eligibility_report(evidence, examined_pairs, lexical, required=int(lexical_count))
    if drawn_report["eligible"] is False:
        return {
            "status": "stopped",
            "reason": "lexical_ineligible",
            "seed": int(seed),
            "shortfall": {},
            "lexical_eligibility": drawn_report,
            "available": {name: len(rows) for name, rows in pools.items()},
            "round": [],
            "note": "A drawn lexical pair is not unexamined and eligible. It is not replaced.",
        }
    return {
        "status": "selected",
        "reason": "",
        "seed": int(seed),
        "shortfall": {},
        "lexical_eligibility": drawn_report,
        "available": {name: len(rows) for name, rows in pools.items()},
        "sampled": {CLASS_SAME: same, CLASS_LEXICAL: lexical},
        "round": list(same) + list(lexical),
    }


def _take_prefix(draw: Mapping[str, Any], same: int, lexical: int) -> list[dict[str, Any]]:
    sampled = draw.get("sampled") if isinstance(draw.get("sampled"), Mapping) else {}
    same_rows = list(sampled.get(CLASS_SAME) or [])[: int(same)]
    lexical_rows = list(sampled.get(CLASS_LEXICAL) or [])[: int(lexical)]
    return same_rows + lexical_rows


def _content_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Stable relationship identity. Per-run relationship ids are not part of it."""
    return {
        "from_id": str(row.get("from_id") or ""),
        "to_id": str(row.get("to_id") or ""),
        "statement": str(row.get("statement") or ""),
        "quote": str(row.get("quote") or ""),
        "evidence_tier": row.get("evidence_tier"),
        "downstream_eligible": row.get("downstream_eligible") is True,
        "paradigm_role": str(row.get("paradigm_role") or ""),
        "document_count": row.get("document_count"),
    }


def core_snapshot(relationships: Sequence[Any], codebook: Mapping[str, Any] | None) -> dict[str, Any]:
    """Eligible content identities and the per-category counts core selection uses."""
    rows = [row for row in relationships if isinstance(row, Mapping) and not row.get("synthetic")]
    eligible = [_content_identity(row) for row in rows if row.get("downstream_eligible") is True]
    eligible.sort(key=_canonical)
    book = _as_book(codebook)
    counts = {}
    for code_id in book:
        counts[str(code_id)] = sum(
            1
            for row in rows
            if row.get("downstream_eligible") is True
            and str(code_id) in {str(row.get("from_id") or ""), str(row.get("to_id") or "")}
        )
    return {"eligible_relationships": eligible, "category_counts": counts}


def core_differs(current: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> bool | None:
    """Content comparison kept for inspection. This experiment does not call it.

    Core selection is not requested. A difference here does not reserve a core
    call or start selective coding. ``eligible_ids`` is ignored.
    """
    if not isinstance(baseline, Mapping):
        return None
    baseline_rows = baseline.get("eligible_relationships")
    current_rows = current.get("eligible_relationships")
    if not isinstance(baseline_rows, list):
        baseline_rows = []
    if not isinstance(current_rows, list):
        current_rows = []
    if [_canonical(row) for row in baseline_rows] != [_canonical(row) for row in current_rows]:
        return True
    base_counts = baseline.get("category_counts") if isinstance(baseline.get("category_counts"), Mapping) else {}
    current_counts = current.get("category_counts") if isinstance(current.get("category_counts"), Mapping) else {}
    keys = set(str(key) for key in base_counts) | set(str(key) for key in current_counts)
    for key in keys:
        if int(base_counts.get(key) or 0) != int(current_counts.get(key) or 0):
            return True
    return False


def _class_metrics() -> dict[str, Any]:
    return {
        "new_pairs_examined": None,
        "grounded_assertions_per_pair": None,
        "additional_supporting_cases": None,
        "reviewer_accepted": None,
        "reviewer_rejected": None,
        "eligibility_failures_by_reason": None,
        "counterexample_calls": None,
        "counterexample_returned": None,
        "counterexample_relevant": None,
        "counterexample_relevance": None,
        "input_tokens": None,
        "dollars": None,
        "elapsed_seconds": None,
    }


def _disposition(row: Mapping[str, Any]) -> str:
    return str(
        row.get("reviewer_disposition") or row.get("human_disposition") or row.get("disposition") or ""
    ).strip().casefold()


def _evidence_failure_reasons(row: Mapping[str, Any]) -> list[str]:
    """Eligibility failures from the evidence rules. Acceptance is not one of them."""
    found: list[str] = []
    blocks = row.get("eligibility_blocks")
    if isinstance(blocks, list):
        for block in blocks:
            text = str(block or "").strip()
            if text and text != "accepted" and text not in found:
                found.append(text)
    reason = str(row.get("eligibility_reason") or "").strip()
    if reason and reason != "accepted" and reason not in found:
        found.append(reason)
    if row.get("downstream_eligible") is not True and not found:
        found.append("ineligible")
    return [item for item in found if item != "accepted"]


def build_metrics(
    *,
    round_rows: Sequence[Mapping[str, Any]],
    relationships: Sequence[Any],
    prompt_log: Sequence[Any],
    elapsed_seconds: float | None,
    token_prices: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Per-class metrics. A figure that was not computed stays null."""
    by_pair: dict[tuple[str, str], str] = {}
    original_cases: dict[tuple[str, str], set[str]] = {}
    for row in round_rows:
        if isinstance(row, Mapping):
            key = _pair_key(row)
            by_pair[key] = str(row.get("evidence_class") or row.get("band") or "")
            if isinstance(row.get("original_case_ids"), list):
                original_cases[key] = {str(item) for item in row["original_case_ids"] if str(item)}
    classes = (CLASS_SAME, CLASS_LEXICAL)
    examined = {name: 0 for name in classes}
    seen_pairs: set[tuple[str, str]] = set()
    grounded = {name: 0 for name in classes}
    accepted = {name: 0 for name in classes}
    rejected = {name: 0 for name in classes}
    additional = {name: 0 for name in classes}
    additional_known = {name: False for name in classes}
    failures: dict[str, dict[str, int]] = {name: {} for name in classes}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in relationships:
        if not isinstance(row, Mapping) or row.get("synthetic"):
            continue
        key = _pair_key(row)
        if by_pair.get(key) in examined:
            grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        evidence_class = by_pair.get(key)
        if evidence_class not in examined or key in seen_pairs:
            continue
        seen_pairs.add(key)
        examined[evidence_class] += 1
        dispositions = {_disposition(row) for row in rows}
        if "accepted" in dispositions or "verified" in dispositions:
            accepted[evidence_class] += 1
        if "rejected" in dispositions or "reject" in dispositions:
            rejected[evidence_class] += 1
            failures[evidence_class]["reviewer_rejected"] = failures[evidence_class].get("reviewer_rejected", 0) + 1
        if any(row.get("downstream_eligible") is True for row in rows):
            grounded[evidence_class] += 1
        else:
            pair_reasons: list[str] = []
            for row in rows:
                for reason in _evidence_failure_reasons(row):
                    if reason not in pair_reasons:
                        pair_reasons.append(reason)
            for reason in pair_reasons:
                failures[evidence_class][reason] = failures[evidence_class].get(reason, 0) + 1
        row = rows[0]
        original = original_cases.get(key)
        cases = {
            str(item.get("case_id"))
            for item in (row.get("supporting_cases") or [])
            if isinstance(item, Mapping) and item.get("case_id")
        }
        if original is None:
            continue
        additional_known[evidence_class] = True
        additional[evidence_class] += len(cases - original)
    in_rate, _out_rate = _prices(token_prices)
    tokens = {name: 0 for name in classes}
    tokens_known = {name: False for name in classes}
    for entry in prompt_log or []:
        if not isinstance(entry, Mapping):
            continue
        status = str(entry.get("status") or "")
        if status in {"omitted", "deferred", "refused", "not_sent", "reused"}:
            continue
        usage = entry.get("usage") if isinstance(entry.get("usage"), Mapping) else {}
        if usage.get("input_tokens") is None:
            continue
        pair_rows = entry.get("pairs") if isinstance(entry.get("pairs"), list) else []
        evidence_class = ""
        for pair in pair_rows:
            if isinstance(pair, Mapping):
                evidence_class = by_pair.get(_pair_key(pair), "")
                if evidence_class:
                    break
        if evidence_class not in tokens:
            continue
        tokens[evidence_class] += int(usage.get("input_tokens") or 0)
        tokens_known[evidence_class] = True
    relevance = {name: None for name in classes}
    reports = []
    by_class = {}
    for name in classes:
        metrics = _class_metrics()
        count = examined[name]
        if count:
            metrics["new_pairs_examined"] = count
            metrics["grounded_assertions_per_pair"] = round(grounded[name] / count, 4)
            metrics["reviewer_accepted"] = accepted[name]
            metrics["reviewer_rejected"] = rejected[name]
            metrics["eligibility_failures_by_reason"] = dict(failures[name])
        if additional_known[name]:
            metrics["additional_supporting_cases"] = additional[name]
        metrics["counterexample_relevance"] = relevance[name]
        if tokens_known[name]:
            metrics["input_tokens"] = tokens[name]
            if in_rate is not None:
                metrics["dollars"] = round(tokens[name] / 1_000_000 * in_rate, 6)
        if elapsed_seconds is not None and count:
            share = count / max(1, sum(examined.values()))
            metrics["elapsed_seconds"] = round(float(elapsed_seconds) * share, 6)
        by_class[name] = metrics
        reports.append(metrics)
    total_examined = sum(examined.values())
    overall = _class_metrics()
    if total_examined:
        overall["new_pairs_examined"] = total_examined
        overall["grounded_assertions_per_pair"] = round(sum(grounded.values()) / total_examined, 4)
        overall["reviewer_accepted"] = sum(accepted.values())
        overall["reviewer_rejected"] = sum(rejected.values())
        merged: dict[str, int] = {}
        for name in classes:
            for reason, count in failures[name].items():
                merged[reason] = merged.get(reason, 0) + count
        overall["eligibility_failures_by_reason"] = merged
    if any(additional_known.values()):
        overall["additional_supporting_cases"] = sum(additional.values())
    if any(tokens_known.values()):
        overall["input_tokens"] = sum(tokens.values())
        if in_rate is not None:
            overall["dollars"] = round(sum(tokens.values()) / 1_000_000 * in_rate, 6)
    if elapsed_seconds is not None:
        overall["elapsed_seconds"] = round(float(elapsed_seconds), 6)
    overall["counterexample_relevance"] = None
    return {"by_class": by_class, "overall": overall}


def counterexample_relevance_by_class(
    judgments: Sequence[Mapping[str, Any]],
    round_rows: Sequence[Mapping[str, Any]],
    *,
    calls_by_class: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Calls, returned assessments, and relevant assessments stay separate.

    ``counterexample_relevance`` is relevant assessments over returned
    assessments. It is null only when no assessment was returned. A call
    with no returned assessment does not make the rate zero.
    """
    by_pair = {}
    for row in round_rows:
        if isinstance(row, Mapping):
            by_pair[_pair_key(row)] = str(row.get("evidence_class") or row.get("band") or "")
    counts = {CLASS_SAME: {"calls": 0, "returned": 0, "relevant": 0}, CLASS_LEXICAL: {"calls": 0, "returned": 0, "relevant": 0}}
    if isinstance(calls_by_class, Mapping):
        for name, count in calls_by_class.items():
            if name in counts:
                counts[name]["calls"] = int(count or 0)
    for row in judgments:
        if not isinstance(row, Mapping):
            continue
        evidence_class = by_pair.get(_pair_key(row))
        if evidence_class not in counts:
            continue
        judgment = str(row.get("judgment") or "")
        if not judgment:
            continue
        counts[evidence_class]["returned"] += 1
        if judgment in {"supports", "contradicts", "qualifies"}:
            counts[evidence_class]["relevant"] += 1
    report = {}
    for name, item in counts.items():
        returned = int(item["returned"])
        report[name] = {
            "calls": int(item["calls"]),
            "returned_assessments": returned,
            "relevant_assessments": int(item["relevant"]),
            "counterexample_relevance": None if returned == 0 else round(item["relevant"] / returned, 4),
        }
    returned_total = sum(item["returned"] for item in counts.values())
    relevant_total = sum(item["relevant"] for item in counts.values())
    report["overall"] = {
        "calls": sum(item["calls"] for item in counts.values()),
        "returned_assessments": returned_total,
        "relevant_assessments": relevant_total,
        "counterexample_relevance": None if returned_total == 0 else round(relevant_total / returned_total, 4),
    }
    return report


def apply_experiment_procedure(procedure: dict[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Install the experiment allowances. Package defaults are not edited here."""
    allowances = spec.get("allowances") if isinstance(spec.get("allowances"), Mapping) else {}
    procedure = dict(procedure)
    procedure["exp_axial_w1"] = dict(spec)
    procedure["max_axial_review_calls"] = CODING_CALLS
    procedure["max_axial_review_tokens"] = CODING_TOKENS
    procedure["propose_calls"] = int(allowances.get("axial_proposal") or 0)
    procedure["statement_copy_retry_calls"] = int(allowances.get("statement_copy_retry") or 0)
    procedure["counterexample_calls"] = int(allowances.get("axial_counterexample") or 0)
    procedure["allowance_transfers"] = []
    procedure["reallocated_propose_calls"] = False
    procedure["counterexample_token_reserve"] = procedure["counterexample_calls"] * REQUEST_INPUT_CAP
    procedure["allowance_split_note"] = (
        "exp-axial-w1 gives proposals, retries, and counterexamples separate allowances. "
        "Calls are not moved between them. The coding ceiling for this experiment is "
        f"{CODING_CALLS} calls and {CODING_TOKENS} tokens. Package defaults stay 8 calls and 24000 tokens."
    )
    return procedure


LEDGER_KEY = "exp_attempt_ledger"
DISPATCHED_ATTEMPT = {"settled", "timeout", "unknown", "failed"}
TERMINAL_PAIR_STATUS = {
    "attempted",
    "successfully_examined",
    "empty_response",
    "failed",
    "budget_blocked",
}


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _worst_case_call_cost(stage: str, input_estimate: int, output_allowance: int | None = None) -> float:
    in_rate, out_rate = _prices()
    output = int(STAGE_MAX_OUTPUT.get(stage) or STAGE_MAX_OUTPUT["axial_proposal"]) if output_allowance is None else int(output_allowance)
    return max(0, int(input_estimate)) / 1_000_000 * in_rate + max(0, output) / 1_000_000 * out_rate


def _output_allowance(stage: str) -> int:
    return int(STAGE_MAX_OUTPUT.get(stage) or STAGE_MAX_OUTPUT["axial_proposal"])


def _experiment_spec(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    procedure = metadata.get("procedure") if isinstance(metadata.get("procedure"), Mapping) else {}
    spec = procedure.get("exp_axial_w1")
    if isinstance(spec, Mapping) and spec.get("enabled"):
        return spec
    return None


def _ledger_rows(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = metadata.get(LEDGER_KEY)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _ledger_figures(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, bool]:
    """Known expenditure, possible exposure, unresolved reservations, and unknown flag.

    A reservation is possible exposure while it is open. It is not expenditure.
    A timeout is possible spend. Unknown provider usage stays out of expenditure.
    """
    known = 0.0
    possible = 0.0
    unresolved = 0.0
    unknown = False
    for row in rows:
        status = str(row.get("status") or "")
        reservation = row.get("reservation") if isinstance(row.get("reservation"), Mapping) else {}
        reserved = _number(reservation.get("dollars"))
        settlement = row.get("settlement") if isinstance(row.get("settlement"), Mapping) else {}
        if status == "reserved":
            unresolved += reserved
            possible += reserved
            continue
        if status in {"budget_blocked", "not_sent", "released"}:
            continue
        if status == "timeout" or row.get("timed_out") is True:
            unknown = True
            possible += _number(settlement.get("possible_dollars")) or reserved
            continue
        usage = row.get("provider_usage")
        if usage is None or status == "unknown":
            unknown = True
            possible += _number(settlement.get("possible_dollars")) or reserved
            continue
        spent = _number(settlement.get("dollars"))
        known += spent
        possible += spent
    return known, possible, unresolved, unknown


def restore_spend_from_ledger(metadata: dict[str, Any]) -> dict[str, Any]:
    """Rebuild spend from the ledger. A restart does not zero a recorded attempt.

    Caller-supplied counters captured before the first ledger row stay as a
    baseline. They are prior exposure, not a second copy of the ledger.
    """
    if "exp_baseline_locked" not in metadata:
        metadata["exp_baseline_spent"] = _number(metadata.get("exp_dollars_spent"))
        metadata["exp_baseline_possible"] = _number(metadata.get("exp_possible_dollars"))
        metadata["exp_baseline_locked"] = True
    known, possible, unresolved, unknown = _ledger_figures(_ledger_rows(metadata))
    baseline_spent = _number(metadata.get("exp_baseline_spent"))
    baseline_possible = max(_number(metadata.get("exp_baseline_possible")), baseline_spent)
    metadata["exp_dollars_spent"] = round(baseline_spent + known, 6)
    metadata["exp_possible_dollars"] = round(baseline_possible + possible, 6)
    metadata["exp_unresolved_reservation_dollars"] = round(unresolved, 6)
    if unknown:
        metadata["exp_spend_unknown"] = True
    return metadata


def _exposure(metadata: Mapping[str, Any]) -> float:
    if isinstance(metadata, dict):
        restore_spend_from_ledger(metadata)
    return max(_number(metadata.get("exp_possible_dollars")), _number(metadata.get("exp_dollars_spent")))


def dispatch_spend_gate(
    metadata: Mapping[str, Any],
    stage: str,
    input_estimate: int,
    output_allowance: int | None = None,
) -> str | None:
    """Stop before a call when the kill guard or the hard cap would be crossed.

    Exposure is settled usage plus open reservations plus timeout possible
    spend. The next reservation prices input and the stage's max output.
    """
    if _experiment_spec(metadata) is None:
        return None
    nxt = _worst_case_call_cost(stage, input_estimate, output_allowance)
    exposure = _exposure(metadata)
    if exposure + nxt > HARD_CAP_USD:
        return "hard_cap"
    if exposure + nxt > KILL_GUARD_USD:
        return "kill_guard"
    return None


def _next_attempt_id(metadata: dict[str, Any]) -> str:
    seq = int(metadata.get("exp_attempt_seq") or 0) + 1
    metadata["exp_attempt_seq"] = seq
    return f"exp-{seq}"


def _append_attempt(metadata: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    rows = metadata.get(LEDGER_KEY)
    if not isinstance(rows, list):
        rows = []
        metadata[LEDGER_KEY] = rows
    rows.append(row)
    restore_spend_from_ledger(metadata)
    return row


def reserve_attempt(
    metadata: dict[str, Any],
    stage: str,
    input_estimate: int,
    *,
    payload_fingerprint: str,
    output_allowance: int | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Hold input plus maximum output cost before a billable attempt is sent.

    A refused attempt is ``budget_blocked`` and takes no reservation. An open
    reservation is not expenditure. Provider usage stays null until settlement.
    """
    if _experiment_spec(metadata) is None:
        return {"status": "not_applicable", "attempt_id": "", "stage": stage}
    pricing = stage if stage in STAGE_MAX_OUTPUT else "axial_proposal"
    allowance = _output_allowance(pricing) if output_allowance is None else int(output_allowance)
    estimate = max(0, int(input_estimate))
    blocked = dispatch_spend_gate(metadata, pricing, estimate, allowance)
    identifier = attempt_id or _next_attempt_id(metadata)
    if blocked:
        return _append_attempt(metadata, {
            "attempt_id": identifier,
            "stage": stage,
            "payload_fingerprint": str(payload_fingerprint),
            "status": "budget_blocked",
            "reason": blocked,
            "estimated_input": estimate,
            "output_allowance": allowance,
            "provider_usage": None,
            "reservation": None,
            "settlement": None,
        })
    dollars = round(_worst_case_call_cost(pricing, estimate, allowance), 6)
    return _append_attempt(metadata, {
        "attempt_id": identifier,
        "stage": stage,
        "payload_fingerprint": str(payload_fingerprint),
        "status": "reserved",
        "reason": "",
        "estimated_input": estimate,
        "output_allowance": allowance,
        "provider_usage": None,
        "reservation": {
            "input_tokens": estimate,
            "output_tokens": allowance,
            "dollars": dollars,
        },
        "settlement": None,
    })


def _find_attempt(metadata: Mapping[str, Any], attempt_id: str) -> dict[str, Any] | None:
    for row in _ledger_rows(metadata):
        if str(row.get("attempt_id") or "") == str(attempt_id):
            return row
    return None


def settle_attempt(
    metadata: dict[str, Any],
    attempt_id: str,
    *,
    usage: Mapping[str, Any] | None = None,
    known: bool = False,
    timed_out: bool = False,
    failed: bool = False,
    released: bool = False,
) -> dict[str, Any] | None:
    """Close one reservation. An estimate is not written as provider usage."""
    if _experiment_spec(metadata) is None:
        return None
    row = _find_attempt(metadata, attempt_id)
    if row is None or str(row.get("status") or "") != "reserved":
        return row
    reservation = row.get("reservation") if isinstance(row.get("reservation"), Mapping) else {}
    reserved = _number(reservation.get("dollars"))
    if released:
        row["status"] = "released"
        row["reason"] = str(row.get("reason") or "not_sent")
        row["provider_usage"] = None
        row["reservation"] = None
        row["settlement"] = None
        restore_spend_from_ledger(metadata)
        return row
    if timed_out:
        row["status"] = "timeout"
        row["timed_out"] = True
        row["reason"] = "timeout"
        row["provider_usage"] = None
        row["settlement"] = {"dollars": None, "possible_dollars": round(reserved, 6), "kind": "timeout"}
        metadata["exp_spend_unknown"] = True
        restore_spend_from_ledger(metadata)
        return row
    if not known:
        row["status"] = "failed" if failed else "unknown"
        row["reason"] = str(row.get("reason") or row["status"])
        row["provider_usage"] = None
        row["settlement"] = {"dollars": None, "possible_dollars": round(reserved, 6), "kind": row["status"]}
        metadata["exp_spend_unknown"] = True
        restore_spend_from_ledger(metadata)
        return row
    in_rate, out_rate = _prices()
    input_n = int((usage or {}).get("input_tokens") or 0)
    output_n = int((usage or {}).get("output_tokens") or 0)
    spent = input_n / 1_000_000 * in_rate + output_n / 1_000_000 * out_rate
    row["status"] = "settled"
    row["provider_usage"] = {"input_tokens": input_n, "output_tokens": output_n}
    row["settlement"] = {"dollars": round(spent, 6), "possible_dollars": round(spent, 6), "kind": "usage"}
    restore_spend_from_ledger(metadata)
    return row


def account_call_spend(
    metadata: dict[str, Any],
    stage: str,
    usage: Mapping[str, Any] | None,
    *,
    known: bool,
    timed_out: bool = False,
    input_estimate: int = REQUEST_INPUT_CAP,
) -> None:
    """Settle the open reservation for ``stage``, or record one attempt if none was held.

    A timeout is possible spend. Missing usage stays null and is not expenditure.
    """
    if _experiment_spec(metadata) is None:
        return
    open_row = None
    for row in reversed(_ledger_rows(metadata)):
        if str(row.get("stage") or "") == stage and str(row.get("status") or "") == "reserved":
            open_row = row
            break
    if open_row is None:
        open_row = reserve_attempt(
            metadata,
            stage,
            int(input_estimate),
            payload_fingerprint=_sha({"stage": stage, "unreserved_settlement": int(metadata.get("exp_attempt_seq") or 0)}),
        )
        if str(open_row.get("status") or "") != "reserved":
            return
    settle_attempt(
        metadata,
        str(open_row.get("attempt_id") or ""),
        usage=usage,
        known=known,
        timed_out=timed_out,
        failed=not known and not timed_out,
    )


def payload_fingerprint(payload: Any) -> str:
    return _sha(payload)


def proposal_gate(metadata: Mapping[str, Any], payload: Mapping[str, Any], estimate: int) -> dict[str, Any] | None:
    """Reuse a matching proposal, or refuse when the kill guard would trip.

    None means the caller should send the request.
    """
    procedure = metadata.get("procedure") if isinstance(metadata.get("procedure"), Mapping) else {}
    spec = procedure.get("exp_axial_w1")
    if not isinstance(spec, Mapping) or not spec.get("enabled"):
        return None
    pairs = [row for row in (payload.get("pairs") or []) if isinstance(row, Mapping)]
    if len(pairs) != 1:
        return None
    pair = pairs[0]
    passages = [row for row in (payload.get("passages") or []) if isinstance(row, Mapping)]
    model_name = str(spec.get("model") or "")
    request_fp = proposal_request_fingerprint(
        str(pair.get("from_id") or ""),
        str(pair.get("to_id") or ""),
        evidence=passages,
        model=model_name,
        passage_ids=[str(row.get("passage_id") or "") for row in passages],
    )
    policy_fp = str(spec.get("policy_fingerprint") or policy_fingerprint(model_name))
    ledger = metadata.get("exp_axial_w1_reuse_ledger")
    if not isinstance(ledger, list):
        ledger = []
        if isinstance(metadata, dict):
            metadata["exp_axial_w1_reuse_ledger"] = ledger
    for record in spec.get("reuse") or []:
        if not isinstance(record, Mapping):
            continue
        if _pair_key(record) != _pair_key(pair):
            continue
        request_ok = str(record.get("request_fingerprint") or "") == request_fp
        policy_ok = str(record.get("policy_fingerprint") or "") == policy_fp
        if request_ok and policy_ok:
            ledger.append({
                "from_id": _pair_key(pair)[0],
                "to_id": _pair_key(pair)[1],
                "decision": "accepted",
                "request_fingerprint": request_fp,
                "policy_fingerprint": policy_fp,
            })
            stored = record.get("relationships")
            if not isinstance(stored, list):
                one = record.get("relationship")
                stored = [one] if isinstance(one, Mapping) else []
            drafts = []
            for item in stored:
                if not isinstance(item, Mapping):
                    continue
                draft = dict(item)
                draft.setdefault("from_id", pair.get("from_id"))
                draft.setdefault("to_id", pair.get("to_id"))
                drafts.append(draft)
            return {
                "refused": False,
                "unparsed": False,
                "repair_pending": False,
                "reason": "",
                "parsed": {"relationships": drafts, "nothing_to_propose": not drafts},
                "usage": {},
                "sent": [],
                "audits": [{
                    "kind": "axial_proposal",
                    "status": "reused",
                    "reason": "request_and_policy_fingerprints_match",
                    "messages": [],
                    "parsed": {"relationships": drafts},
                    "usage": {},
                    "estimated_input_tokens": 0,
                    "pairs": [_pair_record(pair)],
                }],
            }
        ledger.append({
            "from_id": _pair_key(pair)[0],
            "to_id": _pair_key(pair)[1],
            "decision": "rejected",
            "request_match": request_ok,
            "policy_match": policy_ok,
            "request_fingerprint": request_fp,
            "policy_fingerprint": policy_fp,
            "passage_ids": [str(row.get("passage_id") or "") for row in passages],
        })
        break
    stop = dispatch_spend_gate(metadata, "axial_proposal", estimate)
    if stop:
        if isinstance(metadata, dict):
            reserve_attempt(
                metadata,
                "axial_proposal",
                int(estimate),
                payload_fingerprint=payload_fingerprint({"pairs": [_pair_record(pair)], "estimate": int(estimate)}),
            )
        return {
            "refused": True,
            "unparsed": False,
            "repair_pending": False,
            "reason": stop,
            "parsed": {},
            "usage": {},
            "sent": [],
            "audits": [{
                "kind": "axial_proposal",
                "status": "not_sent",
                "reason": stop,
                "messages": [],
                "parsed": {},
                "usage": {},
                "estimated_input_tokens": int(estimate),
            }],
        }
    return None


def _pair_record(pair: Mapping[str, Any]) -> dict[str, str]:
    key = _pair_key(pair)
    return {"from_id": key[0], "to_id": key[1]}


def note_provider_spend(metadata: dict[str, Any], usage: Mapping[str, Any] | None, *, stage: str = "axial_proposal") -> None:
    """Settlement is recorded at dispatch. This hook must not charge the attempt again."""
    del metadata, usage, stage


def _allowances_for(same: int, lexical: int) -> dict[str, int]:
    calls = workload_calls(same, lexical)
    return {
        "axial_proposal": int(calls["axial_proposal"]),
        "statement_copy_retry": int(calls["statement_copy_retry"]),
        "axial_counterexample": int(calls["axial_counterexample_required"]),
        "axial_relation_review": int(calls["axial_relation_review_required"]),
        "axial_saturation_review": int(calls["axial_saturation_review"]),
    }


def _core_record() -> dict[str, Any]:
    return {
        "status": "not_requested",
        "reason": "evaluated separately",
        "calls": 0,
        "dollars": 0,
    }


def _close_without_calls(report: dict[str, Any], *, status: str, reason: str) -> dict[str, Any]:
    report["status"] = status
    report["reason"] = reason
    report["model_calls"] = 0
    report["round"] = []
    report["core"] = _core_record()
    report["saturation"] = {"status": "not_applicable", "reason": reason, "calls": 0}
    report["pair_outcomes"] = []
    report["new_spend"] = {
        "calls": 0,
        "input_tokens": 0,
        "dollars": 0,
        "elapsed_seconds": 0,
        "note": "No call was sent. A reservation was not opened.",
    }
    return report


def _empty_scope_status(universe_n: int, draw: Mapping[str, Any]) -> str:
    available = draw.get("available") if isinstance(draw.get("available"), Mapping) else {}
    same_n = int(available.get(CLASS_SAME) or 0)
    lexical_n = int(available.get(CLASS_LEXICAL) or 0)
    if int(universe_n) <= 0 or (same_n == 0 and lexical_n == 0):
        return "no_work"
    return "preflight_blocked"


def _pair_status_rank(status: str) -> int:
    return {
        "budget_blocked": 1,
        "attempted": 2,
        "failed": 3,
        "empty_response": 4,
        "successfully_examined": 5,
    }.get(status, 0)


def pair_outcomes(
    round_rows: Sequence[Mapping[str, Any]],
    relationships: Sequence[Any],
    prompt_log: Sequence[Any],
    *,
    reuse_ledger: Sequence[Any] | None = None,
    pair_terminal_status: Mapping[str, Any] | None = None,
    stop_reason: str = "",
) -> list[dict[str, Any]]:
    """One terminal status for every selected pair.

    ``successfully_examined`` is a stored or returned relationship, including
    a reused proposal. A selected pair that produced no relationship used to
    disappear from the examined count. It is now ``empty_response``, ``failed``,
    ``budget_blocked``, or ``attempted``.
    """
    order: list[tuple[str, str]] = []
    classes: dict[tuple[str, str], str] = {}
    for row in round_rows:
        if not isinstance(row, Mapping):
            continue
        key = _pair_key(row)
        if not key[0] or not key[1] or key in classes:
            continue
        classes[key] = str(row.get("evidence_class") or row.get("band") or "")
        order.append(key)
    status: dict[tuple[str, str], str] = {}

    def put(key: tuple[str, str], label: str) -> None:
        if key not in classes or label not in TERMINAL_PAIR_STATUS:
            return
        current = status.get(key)
        if current is None or _pair_status_rank(label) >= _pair_status_rank(current):
            status[key] = label

    for row in reuse_ledger or []:
        if isinstance(row, Mapping) and str(row.get("decision") or "") == "accepted":
            put(_pair_key(row), "successfully_examined")
    budget_reasons = {"kill_guard", "hard_cap", "call_allowance", "token_allowance", "dollar_allowance", "budget"}
    for entry in prompt_log or []:
        if not isinstance(entry, Mapping):
            continue
        kind = str(entry.get("kind") or entry.get("checkpoint") or "")
        if kind not in {"axial_proposal", "statement_copy_retry"} and entry.get("statement_copy_retry") is not True:
            continue
        pairs = entry.get("pairs") if isinstance(entry.get("pairs"), list) else []
        entry_status = str(entry.get("status") or "")
        reason = str(entry.get("reason") or "")
        response = entry.get("response") if isinstance(entry.get("response"), Mapping) else {}
        parsed = entry.get("parsed") if isinstance(entry.get("parsed"), Mapping) else {}
        for pair in pairs:
            if not isinstance(pair, Mapping):
                continue
            key = _pair_key(pair)
            if entry_status in {"not_sent", "omitted", "refused", "deferred"} and reason in budget_reasons:
                put(key, "budget_blocked")
            elif entry_status == "reused":
                put(key, "successfully_examined")
            elif entry_status in {"schema_unparsed", "error", "parse_failed"}:
                put(key, "failed")
            elif response.get("nothing_to_propose") is True or parsed.get("nothing_to_propose") is True:
                put(key, "empty_response")
            elif entry_status not in {"omitted", "deferred", "refused", "not_sent"}:
                put(key, "attempted")
    stored = pair_terminal_status if isinstance(pair_terminal_status, Mapping) else {}
    for label, value in stored.items():
        parts = str(label).split("|")
        if len(parts) != 2:
            continue
        if str(value) == "nothing_to_propose":
            put((parts[0], parts[1]) if parts[0] <= parts[1] else (parts[1], parts[0]), "empty_response")
    examined_keys = set()
    for row in relationships:
        if not isinstance(row, Mapping) or row.get("synthetic"):
            continue
        key = _pair_key(row)
        if key in classes:
            examined_keys.add(key)
            put(key, "successfully_examined")
    if stop_reason in budget_reasons or any(value == "budget_blocked" for value in status.values()):
        for key in order:
            if key not in status:
                put(key, "budget_blocked")
    for key in order:
        if key not in status:
            put(key, "failed")
    rows = []
    for key in order:
        rows.append({
            "from_id": key[0],
            "to_id": key[1],
            "evidence_class": classes[key],
            "status": status.get(key) or "failed",
        })
    return rows


def _counterexample_calls_by_class(
    prompt_log: Sequence[Any],
    relationships: Sequence[Any],
    round_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    by_pair = {}
    for row in round_rows:
        if isinstance(row, Mapping):
            by_pair[_pair_key(row)] = str(row.get("evidence_class") or row.get("band") or "")
    by_relationship = {}
    for row in relationships:
        if isinstance(row, Mapping) and row.get("relationship_id"):
            by_relationship[str(row.get("relationship_id"))] = by_pair.get(_pair_key(row), "")
    counts = {CLASS_SAME: 0, CLASS_LEXICAL: 0}
    for entry in prompt_log or []:
        if not isinstance(entry, Mapping) or str(entry.get("kind") or "") != "axial_counterexample":
            continue
        if str(entry.get("status") or "") in {"not_sent", "omitted", "refused", "reused", "deferred"}:
            continue
        text = json.dumps(entry.get("messages") or [], ensure_ascii=False, default=str)
        evidence_class = ""
        for relationship_id, label in by_relationship.items():
            if relationship_id and relationship_id in text:
                evidence_class = label
                break
        if evidence_class in counts:
            counts[evidence_class] += 1
    return counts


def saturation_record(
    metadata: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    *,
    interrupted: bool = False,
    interrupt_kind: str = "",
) -> dict[str, Any]:
    """The planned saturation review ran, or the report says why it did not."""
    rows = [row for row in ledger if str(row.get("stage") or "") == "axial_saturation_review"]
    blocked = [row for row in rows if str(row.get("status") or "") == "budget_blocked"]
    dispatched = [row for row in rows if str(row.get("status") or "") in DISPATCHED_ATTEMPT]
    human = metadata.get("researcher_saturation") if isinstance(metadata.get("researcher_saturation"), Mapping) else None
    model = metadata.get("model_saturation") if isinstance(metadata.get("model_saturation"), Mapping) else None
    if dispatched or model:
        return {
            "status": "executed",
            "reason": str((model or {}).get("assessment") or ""),
            "calls": len(dispatched),
            "actor": "llm",
        }
    if human:
        return {
            "status": "executed",
            "reason": str(human.get("decision") or ""),
            "calls": 0,
            "actor": str(human.get("actor") or "human"),
        }
    if blocked:
        return {"status": "blocked", "reason": str(blocked[-1].get("reason") or "blocked"), "calls": 0}
    stop = str(metadata.get("decision_stop") or metadata.get("axial_stop") or "")
    if stop in {"kill_guard", "hard_cap", "call_allowance", "token_allowance"}:
        return {"status": "blocked", "reason": stop, "calls": 0}
    if interrupted and interrupt_kind == "thematic_saturation":
        return {"status": "not_applicable", "reason": "awaiting_saturation_resume", "calls": 0}
    if interrupted:
        return {"status": "not_applicable", "reason": "run_interrupted_before_saturation", "calls": 0}
    return {"status": "not_applicable", "reason": "saturation_review_did_not_run", "calls": 0}


def spend_from_ledger(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Expenditure is settled known usage. Reservations and estimates are not."""
    if isinstance(metadata, dict):
        restore_spend_from_ledger(metadata)
    rows = _ledger_rows(metadata)
    dispatched = [row for row in rows if str(row.get("status") or "") in DISPATCHED_ATTEMPT]
    unknown = bool(metadata.get("exp_spend_unknown"))
    return {
        "calls": len(dispatched),
        "input_tokens": None if any(row.get("provider_usage") is None for row in dispatched) else sum(
            int((row.get("provider_usage") or {}).get("input_tokens") or 0) for row in dispatched
        ),
        "dollars": None if unknown else metadata.get("exp_dollars_spent"),
        "possible_dollars": metadata.get("exp_possible_dollars"),
        "unresolved_reservation_dollars": metadata.get("exp_unresolved_reservation_dollars"),
        "note": (
            "calls counts dispatched attempts. A reservation is not expenditure. "
            "An estimate is not usage. A timeout is possible spend and leaves dollars null. "
            "A resumed result is not a new attempt."
        ),
    }


def exp_axial_w1(
    source: Mapping[str, Any],
    model: Any,
    *,
    history_path: str | Path,
    frozen_fingerprints: Mapping[str, Any],
    baseline_core: Mapping[str, Any] | None = None,
    token_prices: Mapping[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    commit: str | None = None,
    **axial_kwargs: Any,
) -> dict[str, Any]:
    """Run the experiment offline against the caller's model.

    The default axial entry point is not this function. Focused coding is not
    called. A fingerprint mismatch raises before a model call.
    """
    if int(axial_kwargs.get("pattern_min_cases", PATTERN_MIN_CASES) or PATTERN_MIN_CASES) != PATTERN_MIN_CASES:
        raise AxialError("exp-axial-w1 does not change pattern_min_cases.")
    if bool(axial_kwargs.get("one_document_one_respondent", False)):
        raise AxialError("exp-axial-w1 keeps one_document_one_respondent off.")
    current = startup_fingerprint(source, model, commit=commit)
    mismatched = fingerprint_differences(frozen_fingerprints, current)
    if mismatched:
        raise AxialError(
            "exp-axial-w1 refused to start. Frozen fingerprints differ: " + ", ".join(mismatched)
        )
    history = load_examined_history(history_path)
    from automated_grounded_theory.relationship_tiers import selection_restricts_to_raised

    book = _as_book(source.get("codebook"))
    meta = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
    if source.get("raised_code_ids") is not None and "raised_code_ids" not in meta:
        meta = dict(meta)
        meta["raised_code_ids"] = list(source.get("raised_code_ids") or [])
    category_ids = list(book)
    restricted = selection_restricts_to_raised(meta)
    if restricted:
        raised = {str(item) for item in (meta.get("raised_code_ids") or [])}
        category_ids = [item for item in category_ids if item in raised]
    evidence = classify_pair_evidence(
        category_ids,
        source.get("passages") or [],
        source.get("segments") or [],
        book,
        pattern_min_cases=PATTERN_MIN_CASES,
    )
    universe_n = len(category_ids)
    draw = select_unexamined_draw(evidence, history["examined_pairs"], seed=int(seed))
    reduction = reduce_scope(token_prices, core_candidates=universe_n)
    report = {
        "experiment": EXPERIMENT_ID,
        "seed": int(seed),
        "history_path": history["path"],
        "historical_expenditure": history["historical_expenditure"],
        "preflight": reduction,
        "universe": {
            "codebook_codes": len(book),
            "raised_codes": universe_n if restricted else None,
            "restricted_to_raised": restricted,
            "pair_universe": universe_n * (universe_n - 1) // 2,
            "same_respondent_unexamined": (draw.get("available") or {}).get(CLASS_SAME),
            "lexical_unexamined": (draw.get("available") or {}).get(CLASS_LEXICAL),
            "note": "pair_universe is raised_codes * (raised_codes - 1) / 2 when the selection limits the pool, otherwise the codebook size.",
        },
        "draw": {
            key: draw[key]
            for key in ("status", "reason", "shortfall", "available", "seed", "lexical_eligibility")
            if key in draw
        },
        "draw_method": "direct_seeded_draw",
        "new_spend": {"calls": 0, "input_tokens": 0, "dollars": 0, "elapsed_seconds": 0},
        "core": _core_record(),
    }
    del baseline_core
    if draw["status"] != "selected":
        reason = str(draw.get("reason") or "class_shortfall")
        return _close_without_calls(report, status=_empty_scope_status(universe_n, draw), reason=reason)
    scope = reduction["reduced_scope"]
    if reduction["reduced"]["fits"] is not True:
        return _close_without_calls(report, status="preflight_blocked", reason="workload_does_not_fit")
    direct = select_unexamined_draw(
        evidence,
        history["examined_pairs"],
        seed=int(seed),
        same_count=int(scope["same_respondent"]),
        lexical_count=int(scope["lexical_cross_case"]),
    )
    if direct["status"] != "selected":
        report["draw"] = {
            key: direct[key]
            for key in ("status", "reason", "shortfall", "available", "seed", "lexical_eligibility")
            if key in direct
        }
        return _close_without_calls(
            report,
            status=_empty_scope_status(universe_n, direct),
            reason=str(direct.get("reason") or "class_shortfall"),
        )
    chosen = _with_original_cases(list(direct.get("round") or []), _endpoint_cases({**source, "codebook": book}))
    report["executed_draw"] = {
        "method": "direct_seeded_draw",
        "seed": int(seed),
        "lexical_eligibility": direct.get("lexical_eligibility"),
        "note": (
            "The executed round is a seeded sample of the fitting scope. "
            "It is not a prefix of the 16+8 sample."
        ),
    }
    allowances = _allowances_for(scope["same_respondent"], scope["lexical_cross_case"])
    spec = {
        "id": EXPERIMENT_ID,
        "enabled": True,
        "seed": int(seed),
        "round": chosen,
        "allowances": allowances,
        "reuse": history["reuse"],
        "model": describe_model(model),
        "policy_fingerprint": policy_fingerprint(model),
        "token_prices": dict(FROZEN_TOKEN_PRICES),
        "hard_cap_usd": HARD_CAP_USD,
        "kill_guard_usd": KILL_GUARD_USD,
        "provider_max_attempts": 1,
        "provider_retries": "disabled",
        "provider_retry_note": (
            "Provider retries are not reserved as separate attempts, so max_retries is 0. "
            "Quote-copy retries and schema repairs are dispatched through the accounted pool."
        ),
    }
    axial_kwargs.pop("provider_max_attempts", None)
    started = time.perf_counter()
    result = axial_code(
        source,
        model,
        seed=int(seed),
        max_axial_review_calls=CODING_CALLS,
        max_axial_review_tokens=CODING_TOKENS,
        axial_token_budget=REQUEST_INPUT_CAP,
        pattern_min_cases=PATTERN_MIN_CASES,
        one_document_one_respondent=False,
        batch_size=1,
        examined_pairs=history["examined_pairs"],
        max_decision_calls=int(allowances["axial_relation_review"]),
        provider_max_attempts=1,
        experiment=spec,
        **{key: value for key, value in axial_kwargs.items() if key not in {"max_decision_calls", "provider_max_attempts"}},
    )
    elapsed = time.perf_counter() - started
    relationships = list(result.relationships or [])
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if isinstance(metadata, dict):
        restore_spend_from_ledger(metadata)
    ledger = _ledger_rows(metadata)
    outcomes = pair_outcomes(
        chosen,
        relationships,
        list(result.prompt_log or []),
        reuse_ledger=list(metadata.get("exp_axial_w1_reuse_ledger") or []),
        pair_terminal_status=metadata.get("pair_terminal_status") if isinstance(metadata.get("pair_terminal_status"), Mapping) else {},
        stop_reason=str(metadata.get("axial_stop") or metadata.get("decision_stop") or ""),
    )
    metrics = build_metrics(
        round_rows=chosen,
        relationships=relationships,
        prompt_log=list(result.prompt_log or []),
        elapsed_seconds=elapsed,
        token_prices=FROZEN_TOKEN_PRICES,
    )
    judgments = []
    relationship_class = {}
    for row in chosen:
        relationship_class[_pair_key(row)] = str(row.get("evidence_class") or row.get("band") or "")
    for row in relationships:
        if not isinstance(row, Mapping) or row.get("synthetic"):
            continue
        evidence_class = relationship_class.get(_pair_key(row), "")
        for verdict in row.get("counterexamples") or []:
            if isinstance(verdict, Mapping) and not verdict.get("synthetic"):
                judgments.append({**dict(verdict), "from_id": row.get("from_id"), "to_id": row.get("to_id"), "evidence_class": evidence_class})
    call_counts = _counterexample_calls_by_class(list(result.prompt_log or []), relationships, chosen)
    relevance = counterexample_relevance_by_class(judgments, chosen, calls_by_class=call_counts)
    for name in (CLASS_SAME, CLASS_LEXICAL):
        detail = relevance.get(name) if isinstance(relevance.get(name), Mapping) else {}
        metrics["by_class"][name]["counterexample_calls"] = detail.get("calls")
        metrics["by_class"][name]["counterexample_returned"] = detail.get("returned_assessments")
        metrics["by_class"][name]["counterexample_relevant"] = detail.get("relevant_assessments")
        metrics["by_class"][name]["counterexample_relevance"] = detail.get("counterexample_relevance")
    overall_detail = relevance.get("overall") if isinstance(relevance.get("overall"), Mapping) else {}
    metrics["overall"]["counterexample_calls"] = overall_detail.get("calls")
    metrics["overall"]["counterexample_returned"] = overall_detail.get("returned_assessments")
    metrics["overall"]["counterexample_relevant"] = overall_detail.get("relevant_assessments")
    metrics["overall"]["counterexample_relevance"] = overall_detail.get("counterexample_relevance")
    examined = sum(1 for row in outcomes if row.get("status") == "successfully_examined")
    metrics["overall"]["new_pairs_examined"] = examined
    metrics["overall"]["selected_pairs"] = len(outcomes)
    by_status: dict[str, int] = {}
    for row in outcomes:
        by_status[str(row.get("status") or "")] = by_status.get(str(row.get("status") or ""), 0) + 1
    spend = spend_from_ledger(metadata)
    spend["elapsed_seconds"] = round(elapsed, 6)
    blocked = [
        row for row in ledger
        if str(row.get("status") or "") == "budget_blocked" and str(row.get("stage") or "") != "axial_proposal"
    ]
    report.update({
        "status": "interrupted" if result.interrupted else "completed",
        "round": chosen,
        "allowances": allowances,
        "reuse_ledger": list(metadata.get("exp_axial_w1_reuse_ledger") or []),
        "attempt_ledger": ledger,
        "pair_outcomes": outcomes,
        "pair_coverage": {
            "selected": len(outcomes),
            "successfully_examined": examined,
            "by_status": by_status,
            "note": (
                "new_pairs_examined counts successfully_examined pairs. "
                "A selected pair with no relationship previously disappeared from that count. "
                "It now has a terminal status: attempted, empty_response, failed, or budget_blocked."
            ),
        },
        "metrics": metrics,
        "counterexamples": overall_detail,
        "core": _core_record(),
        "saturation": saturation_record(
            metadata,
            ledger,
            interrupted=bool(result.interrupted) and not metadata.get("researcher_saturation"),
            interrupt_kind=str((result.interrupt_payload or {}).get("kind") or "") if isinstance(result.interrupt_payload, Mapping) else "",
        ),
        "guard_exhaustion": None if not blocked else {
            "stage": blocked[-1].get("stage"),
            "reason": blocked[-1].get("reason"),
            "attempt_id": blocked[-1].get("attempt_id"),
            "during_execution": True,
        },
        "result": result,
        "new_spend": spend,
        "provider_retries": "disabled",
        "provider_max_attempts": 1,
    })
    return report
