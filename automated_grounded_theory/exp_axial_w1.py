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
    selective_code,
)
from automated_grounded_theory.backends import chat_model_settings, describe_model
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES, classify_pair_evidence

EXPERIMENT_ID = "exp-axial-w1"
DRAW_SAME_RESPONDENT = 16
DRAW_LEXICAL = 8
DEFAULT_SEED = 4401
CODING_CALLS = 36
CODING_TOKENS = 216_000
REQUEST_INPUT_CAP = 6000
COUNTEREXAMPLE_CALLS = 4
RELATION_REVIEW_CALLS = 4
SATURATION_REVIEW_CALLS = 1
HARD_CAP_USD = 0.60
KILL_GUARD_USD = 0.55
REDUCE_STEP = 4
CLASS_SAME = "same_respondent"
CLASS_LEXICAL = "lexical_cross_case"


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pair_key(row: Mapping[str, Any] | Sequence[str]) -> tuple[str, str]:
    if isinstance(row, Mapping):
        left = str(row.get("from_id") or "")
        right = str(row.get("to_id") or "")
    else:
        left, right = str(row[0]), str(row[1])
    return tuple(sorted((left, right)))


def policy_fingerprint() -> str:
    """Eligibility and experiment policy. A change here rejects reuse."""
    return _sha({
        "experiment": EXPERIMENT_ID,
        "policy_version": POLICY_VERSION,
        "pattern_min_cases": PATTERN_MIN_CASES,
        "one_document_one_respondent": False,
        "same_respondent_rule": "single_document_cooccurrence_1",
        "lexical_rule": "lexical_cross_case",
        "evidence_order": ["same_passage", "same_respondent", "lexical_cross_case"],
    })


def proposal_request_fingerprint(from_id: str, to_id: str) -> str:
    """The proposal request identity: the pair and the proposal prompt.

    Passage text is not part of the hash. A different prompt, or a different
    pair, is a different request.
    """
    left, right = _pair_key({"from_id": from_id, "to_id": to_id})
    return _sha({
        "experiment": EXPERIMENT_ID,
        "from_id": left,
        "to_id": right,
        "prompt_sha256": hashlib.sha256(_KIND_PROMPTS["axial_proposal"].encode("utf-8")).hexdigest(),
    })


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def startup_fingerprint(source: Mapping[str, Any], model: Any, *, commit: str | None = None) -> dict[str, Any]:
    """Commit, export manifest, model config, prompts, and eligibility rules.

    The manifest hashes ids only. Passage text is not in the fingerprint.
    """
    codebook = source.get("codebook") if isinstance(source.get("codebook"), Mapping) else {}
    passages = [row for row in (source.get("passages") or []) if isinstance(row, Mapping)]
    segments = [row for row in (source.get("segments") or []) if isinstance(row, Mapping)]
    manifest = _sha({
        "code_ids": sorted(str(item) for item in codebook),
        "passage_ids": sorted(str(row.get("passage_id") or "") for row in passages),
        "segments": sorted(
            f"{row.get('code_id')}|{row.get('passage_id')}" for row in segments
        ),
    })
    prompts = _sha({
        name: _KIND_PROMPTS[name]
        for name in ("axial_proposal", "axial_counterexample")
    })
    eligibility = _sha({
        "pattern_min_cases": PATTERN_MIN_CASES,
        "one_document_one_respondent": False,
        "policy_version": POLICY_VERSION,
        "experiment": EXPERIMENT_ID,
    })
    return {
        "commit": _git_commit() if commit is None else str(commit),
        "export_manifest": manifest,
        "model_config": _sha({
            "model": describe_model(model),
            "settings": chat_model_settings(model),
        }),
        "prompts": prompts,
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


def _prices(token_prices: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    if not isinstance(token_prices, Mapping):
        return None, None
    def _rate(name: str) -> float | None:
        value = token_prices.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
    return _rate("input_usd_per_million"), _rate("output_usd_per_million")


def workload_calls(same: int, lexical: int) -> dict[str, int]:
    """Worst-case calls for one draw. Each stage keeps its own allowance.

    Proposals and retries are one each per pair. They are not transferred.
    Counterexamples stay at the experiment allowance. Relation review is
    capped at its own allowance. Saturation review is one call.
    """
    pairs = max(0, int(same)) + max(0, int(lexical))
    return {
        "axial_proposal": pairs,
        "statement_copy_retry": pairs,
        "axial_counterexample": COUNTEREXAMPLE_CALLS if pairs else 0,
        "axial_relation_review": min(RELATION_REVIEW_CALLS, pairs),
        "axial_saturation_review": SATURATION_REVIEW_CALLS if pairs else 0,
    }


def preflight_workload(
    same: int,
    lexical: int,
    *,
    token_prices: Mapping[str, Any] | None = None,
    per_call_input: int = REQUEST_INPUT_CAP,
) -> dict[str, Any]:
    """Worst case of one draw against the coding ceiling and the dollar cap.

    Each call is priced at the per-request input cap. Output dollars stay null
    when no output price is configured. An unset input price leaves dollars
    null and does not count as fitting the cap.
    """
    calls = workload_calls(same, lexical)
    coding_calls = calls["axial_proposal"] + calls["statement_copy_retry"] + calls["axial_counterexample"]
    decision_calls = calls["axial_relation_review"] + calls["axial_saturation_review"]
    total_calls = coding_calls + decision_calls
    per_call = max(0, int(per_call_input))
    coding_tokens = coding_calls * per_call
    total_tokens = total_calls * per_call
    in_rate, out_rate = _prices(token_prices)
    if in_rate is None:
        dollars = None
    else:
        dollars = round(total_tokens / 1_000_000 * in_rate, 6)
    output_dollars = None if out_rate is None else round(total_calls * per_call / 1_000_000 * out_rate, 6)
    coding_ok = coding_calls <= CODING_CALLS and coding_tokens <= CODING_TOKENS
    dollar_ok = None if dollars is None else dollars <= HARD_CAP_USD
    fits = bool(coding_ok and dollar_ok is True)
    reasons = []
    if not coding_ok:
        reasons.append("coding_ceiling")
    if dollar_ok is None:
        reasons.append("dollar_prices_unset")
    elif dollar_ok is False:
        reasons.append("dollar_hard_cap")
    return {
        "experiment": EXPERIMENT_ID,
        "same_respondent": int(same),
        "lexical_cross_case": int(lexical),
        "pairs": int(same) + int(lexical),
        "calls": calls,
        "coding_calls": coding_calls,
        "coding_input_tokens": coding_tokens,
        "total_calls": total_calls,
        "total_input_tokens": total_tokens,
        "per_call_input_tokens": per_call,
        "dollars": dollars,
        "output_dollars": output_dollars,
        "hard_cap_usd": HARD_CAP_USD,
        "kill_guard_usd": KILL_GUARD_USD,
        "coding_call_ceiling": CODING_CALLS,
        "coding_token_ceiling": CODING_TOKENS,
        "fits_coding_ceiling": coding_ok,
        "fits_dollar_cap": dollar_ok,
        "fits": fits,
        "reasons": reasons,
        "note": (
            "Worst case prices every planned call at the per-request input cap. "
            "Output dollars are null unless an output price is set. "
            "A null dollar figure does not mean the workload fits the hard cap."
        ),
    }


def reduce_scope(
    token_prices: Mapping[str, Any] | None = None,
    *,
    per_call_input: int = REQUEST_INPUT_CAP,
) -> dict[str, Any]:
    """Shrink the 16+8 draw by four pairs until the worst case fits.

    The class ratio stays as close to 2:1 as the integer split allows. The
    original 16+8 preflight is kept either way.
    """
    same, lexical = DRAW_SAME_RESPONDENT, DRAW_LEXICAL
    original = preflight_workload(same, lexical, token_prices=token_prices, per_call_input=per_call_input)
    steps = [original]
    current = original
    while current["fits"] is False and (same + lexical) >= REDUCE_STEP:
        total = same + lexical - REDUCE_STEP
        same, lexical = split_preserving_ratio(total)
        current = preflight_workload(same, lexical, token_prices=token_prices, per_call_input=per_call_input)
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
    if shortfall:
        return {
            "status": "stopped",
            "reason": "class_shortfall",
            "seed": int(seed),
            "shortfall": shortfall,
            "available": {name: len(rows) for name, rows in pools.items()},
            "round": [],
            "note": (
                "A class had fewer unexamined eligible pairs than the draw. "
                "The run stops. Pairs are not taken from the other class."
            ),
        }
    rng = random.Random(int(seed))
    same = rng.sample(pools[CLASS_SAME], int(same_count))
    lexical = rng.sample(pools[CLASS_LEXICAL], int(lexical_count))
    return {
        "status": "selected",
        "reason": "",
        "seed": int(seed),
        "shortfall": {},
        "available": {name: len(rows) for name, rows in pools.items()},
        "sampled": {CLASS_SAME: same, CLASS_LEXICAL: lexical},
        "round": list(same) + list(lexical),
    }


def _take_prefix(draw: Mapping[str, Any], same: int, lexical: int) -> list[dict[str, Any]]:
    sampled = draw.get("sampled") if isinstance(draw.get("sampled"), Mapping) else {}
    same_rows = list(sampled.get(CLASS_SAME) or [])[: int(same)]
    lexical_rows = list(sampled.get(CLASS_LEXICAL) or [])[: int(lexical)]
    return same_rows + lexical_rows


def core_snapshot(relationships: Sequence[Any], codebook: Mapping[str, Any] | None) -> dict[str, Any]:
    """The eligible set and the per-category counts core selection uses."""
    rows = [row for row in relationships if isinstance(row, Mapping) and not row.get("synthetic")]
    eligible = sorted(
        str(row.get("relationship_id") or "")
        for row in rows
        if row.get("downstream_eligible") is True and row.get("relationship_id")
    )
    codes = [str(code_id) for code_id in (codebook or {})]
    counts = {}
    for code_id in codes:
        counts[code_id] = sum(
            1
            for row in rows
            if row.get("downstream_eligible") is True
            and code_id in {str(row.get("from_id") or ""), str(row.get("to_id") or "")}
        )
    return {"eligible_ids": eligible, "category_counts": counts}


def core_differs(current: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> bool | None:
    """Null when no baseline was supplied. True when either core input changed."""
    if not isinstance(baseline, Mapping):
        return None
    baseline_ids = [str(item) for item in (baseline.get("eligible_ids") or [])]
    current_ids = [str(item) for item in (current.get("eligible_ids") or [])]
    if baseline_ids != current_ids:
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
        "eligibility_failures_by_reason": None,
        "counterexample_relevance": None,
        "input_tokens": None,
        "dollars": None,
        "elapsed_seconds": None,
    }


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
    grounded = {name: 0 for name in classes}
    additional = {name: 0 for name in classes}
    additional_known = {name: False for name in classes}
    failures: dict[str, dict[str, int]] = {name: {} for name in classes}
    for row in relationships:
        if not isinstance(row, Mapping) or row.get("synthetic"):
            continue
        key = _pair_key(row)
        evidence_class = by_pair.get(key)
        if evidence_class not in examined:
            continue
        examined[evidence_class] += 1
        if row.get("downstream_eligible") is True:
            grounded[evidence_class] += 1
        else:
            reason = str(row.get("reviewer_disposition") or row.get("disposition") or "ineligible")
            if reason in {"rejected", "reject"}:
                reason = "reviewer_rejected"
            failures[evidence_class][reason] = failures[evidence_class].get(reason, 0) + 1
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
) -> dict[str, Any]:
    """Relevant judgments over judgments sent, by the pair's class. Null if none were sent."""
    by_pair = {}
    for row in round_rows:
        if isinstance(row, Mapping):
            by_pair[_pair_key(row)] = str(row.get("evidence_class") or row.get("band") or "")
    counts = {CLASS_SAME: [0, 0], CLASS_LEXICAL: [0, 0]}
    for row in judgments:
        if not isinstance(row, Mapping):
            continue
        evidence_class = by_pair.get(_pair_key(row))
        if evidence_class not in counts:
            continue
        judgment = str(row.get("judgment") or "")
        if not judgment:
            continue
        counts[evidence_class][1] += 1
        if judgment in {"supports", "contradicts", "qualifies"}:
            counts[evidence_class][0] += 1
    report = {}
    for name, (relevant, sent) in counts.items():
        report[name] = None if sent == 0 else round(relevant / sent, 4)
    sent_total = sum(item[1] for item in counts.values())
    relevant_total = sum(item[0] for item in counts.values())
    report["overall"] = None if sent_total == 0 else round(relevant_total / sent_total, 4)
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
    request_fp = proposal_request_fingerprint(str(pair.get("from_id") or ""), str(pair.get("to_id") or ""))
    policy_fp = policy_fingerprint()
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
            draft = dict(record.get("relationship") or {})
            draft.setdefault("from_id", pair.get("from_id"))
            draft.setdefault("to_id", pair.get("to_id"))
            return {
                "refused": False,
                "unparsed": False,
                "repair_pending": False,
                "reason": "",
                "parsed": {"relationships": [draft], "nothing_to_propose": False},
                "usage": {},
                "sent": [],
                "audits": [{
                    "kind": "axial_proposal",
                    "status": "reused",
                    "reason": "request_and_policy_fingerprints_match",
                    "messages": [],
                    "parsed": {"relationships": [draft]},
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
        })
        break
    in_rate, _out_rate = _prices(spec.get("token_prices") if isinstance(spec.get("token_prices"), Mapping) else None)
    if in_rate is None:
        return None
    spent = metadata.get("exp_dollars_spent")
    spent_n = float(spent) if isinstance(spent, (int, float)) and not isinstance(spent, bool) else 0.0
    next_cost = max(0, int(estimate)) / 1_000_000 * in_rate
    if spent_n + next_cost > KILL_GUARD_USD:
        return {
            "refused": True,
            "unparsed": False,
            "repair_pending": False,
            "reason": "kill_guard",
            "parsed": {},
            "usage": {},
            "sent": [],
            "audits": [{
                "kind": "axial_proposal",
                "status": "not_sent",
                "reason": "kill_guard",
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


def note_provider_spend(metadata: dict[str, Any], usage: Mapping[str, Any] | None) -> None:
    procedure = metadata.get("procedure") if isinstance(metadata.get("procedure"), Mapping) else {}
    spec = procedure.get("exp_axial_w1")
    if not isinstance(spec, Mapping) or not spec.get("enabled"):
        return
    in_rate, _out_rate = _prices(spec.get("token_prices") if isinstance(spec.get("token_prices"), Mapping) else None)
    if in_rate is None or not isinstance(usage, Mapping) or usage.get("input_tokens") is None:
        return
    spent = metadata.get("exp_dollars_spent")
    spent_n = float(spent) if isinstance(spent, (int, float)) and not isinstance(spent, bool) else 0.0
    metadata["exp_dollars_spent"] = round(spent_n + int(usage["input_tokens"]) / 1_000_000 * in_rate, 6)


def _allowances_for(same: int, lexical: int) -> dict[str, int]:
    calls = workload_calls(same, lexical)
    return {
        "axial_proposal": calls["axial_proposal"],
        "statement_copy_retry": calls["statement_copy_retry"],
        "axial_counterexample": calls["axial_counterexample"],
        "axial_relation_review": RELATION_REVIEW_CALLS,
        "axial_saturation_review": RELATION_REVIEW_CALLS,
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
    evidence = classify_pair_evidence(
        list((source.get("codebook") or {}).keys()),
        source.get("passages") or [],
        source.get("segments") or [],
        source.get("codebook") if isinstance(source.get("codebook"), Mapping) else {},
        pattern_min_cases=PATTERN_MIN_CASES,
    )
    draw = select_unexamined_draw(evidence, history["examined_pairs"], seed=int(seed))
    reduction = reduce_scope(token_prices)
    report = {
        "experiment": EXPERIMENT_ID,
        "seed": int(seed),
        "history_path": history["path"],
        "historical_expenditure": history["historical_expenditure"],
        "preflight": reduction,
        "draw": {key: draw[key] for key in ("status", "reason", "shortfall", "available", "seed")},
        "new_spend": {"calls": None, "input_tokens": None, "dollars": None, "elapsed_seconds": None},
    }
    if draw["status"] != "selected":
        report["status"] = "stopped"
        report["model_calls"] = 0
        return report
    scope = reduction["reduced_scope"]
    if reduction["reduced"]["fits"] is not True:
        report["status"] = "stopped"
        report["reason"] = "workload_does_not_fit"
        report["model_calls"] = 0
        return report
    chosen = _with_original_cases(
        _take_prefix(draw, scope["same_respondent"], scope["lexical_cross_case"]),
        _endpoint_cases(source),
    )
    allowances = _allowances_for(scope["same_respondent"], scope["lexical_cross_case"])
    spec = {
        "id": EXPERIMENT_ID,
        "enabled": True,
        "seed": int(seed),
        "round": chosen,
        "allowances": allowances,
        "reuse": history["reuse"],
        "token_prices": None if token_prices is None else dict(token_prices),
        "hard_cap_usd": HARD_CAP_USD,
        "kill_guard_usd": KILL_GUARD_USD,
        "baseline_core": None if baseline_core is None else dict(baseline_core),
    }
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
        experiment=spec,
        **axial_kwargs,
    )
    elapsed = time.perf_counter() - started
    relationships = list(result.relationships or [])
    metrics = build_metrics(
        round_rows=chosen,
        relationships=relationships,
        prompt_log=list(result.prompt_log or []),
        elapsed_seconds=elapsed,
        token_prices=token_prices,
    )
    judgments = []
    for row in result.metadata.get("counterexample_requests") or []:
        if isinstance(row, Mapping):
            judgments.append(row)
    relevance = counterexample_relevance_by_class(judgments, chosen)
    for name in (CLASS_SAME, CLASS_LEXICAL):
        metrics["by_class"][name]["counterexample_relevance"] = relevance.get(name)
    metrics["overall"]["counterexample_relevance"] = relevance.get("overall")
    snapshot = core_snapshot(relationships, source.get("codebook") if isinstance(source.get("codebook"), Mapping) else {})
    differs = core_differs(snapshot, baseline_core)
    in_rate, _output_rate = _prices(token_prices)
    if differs is False:
        core = {
            "status": "deferred",
            "dollars": 0,
            "calls": 0,
            "reason": "eligible_set_and_category_counts_match_baseline",
        }
    elif differs is None:
        core = {"status": "not_run", "dollars": None, "calls": 0, "reason": "baseline_missing"}
    else:
        selective = selective_code(
            result,
            model,
            seed=int(seed),
            decision_mode=str(axial_kwargs.get("decision_mode") or "llm"),
            interrupt=False,
            include_memos=False,
            include_storyline=False,
            one_document_one_respondent=False,
        )
        core_rows = [
            row for row in (selective.prompt_log or [])
            if isinstance(row, Mapping)
            and row.get("checkpoint") == "axial_core_review"
            and str(row.get("status") or "") not in {"omitted", "deferred", "refused", "not_sent", "reused"}
        ]
        core_tokens = 0
        core_tokens_known = False
        for row in core_rows:
            usage = row.get("usage") if isinstance(row.get("usage"), Mapping) else {}
            if usage.get("input_tokens") is None:
                continue
            core_tokens += int(usage["input_tokens"])
            core_tokens_known = True
        if in_rate is None or not core_tokens_known:
            core_dollars = None
        else:
            core_dollars = round(core_tokens / 1_000_000 * in_rate, 6)
        core = {
            "status": "ran",
            "dollars": core_dollars,
            "calls": len(core_rows),
            "input_tokens": core_tokens if core_tokens_known else None,
            "reason": "eligible_set_or_category_counts_differ",
        }
    new_tokens = metrics["overall"]["input_tokens"]
    report.update({
        "status": "completed",
        "round": chosen,
        "allowances": allowances,
        "reuse_ledger": list(result.metadata.get("exp_axial_w1_reuse_ledger") or []),
        "metrics": metrics,
        "core": core,
        "core_snapshot": snapshot,
        "result": result,
        "new_spend": {
            "calls": metrics["overall"]["new_pairs_examined"],
            "input_tokens": new_tokens,
            "dollars": metrics["overall"]["dollars"],
            "elapsed_seconds": metrics["overall"]["elapsed_seconds"],
            "note": "New spend does not include historical_expenditure.",
        },
    })
    return report
