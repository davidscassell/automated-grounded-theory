"""Offline comparison plan for initial coding versus focused rereading.

The live path is an estimate only. This module does not call a provider.
A person has to approve a live run before any API call is made.

Prices are the xAI standard rates for ``grok-4.20-0309-non-reasoning`` under
200k context, as published at
https://docs.x.ai/developers/models/grok-4.20-0309-non-reasoning
and https://docs.x.ai/developers/pricing (read 2026-10-14): $1.25 per million
input tokens and $2.50 per million output tokens.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from automated_grounded_theory.category_state import MEMO_COUNT_NOTE
from automated_grounded_theory.focused import select_focused_codes

MODEL_ID = "grok-4.20-0309-non-reasoning"
INPUT_USD_PER_MILLION = 1.25
OUTPUT_USD_PER_MILLION = 2.50

# A small researcher-reviewed comparison. Allowances in the package are ceilings.
# These token counts are the planning estimate, not a measurement.
LIVE_PLAN = {
    "model": MODEL_ID,
    "open_coding_calls": 40,
    "open_input_tokens_each": 4000,
    "open_output_tokens_each": 800,
    "focused_calls": 24,
    "focused_input_tokens_each": 6000,
    "focused_output_tokens_each": 1000,
    "focused_runs": 1,
    "ablation_rereads": 3,
    "disagreement_seeds": 2,
    "retry_contingency": 0.15,
}


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * INPUT_USD_PER_MILLION + (
        output_tokens / 1_000_000
    ) * OUTPUT_USD_PER_MILLION


def estimate_live_cost(plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Dollar estimate for the proposed live comparison. Does not call the API."""
    spec = dict(LIVE_PLAN if plan is None else plan)
    open_in = int(spec["open_coding_calls"]) * int(spec["open_input_tokens_each"])
    open_out = int(spec["open_coding_calls"]) * int(spec["open_output_tokens_each"])
    open_cost = _cost(open_in, open_out)
    one_focused_in = int(spec["focused_calls"]) * int(spec["focused_input_tokens_each"])
    one_focused_out = int(spec["focused_calls"]) * int(spec["focused_output_tokens_each"])
    one_focused = _cost(one_focused_in, one_focused_out)
    focused_runs = int(spec["focused_runs"]) + int(spec["ablation_rereads"]) + int(spec["disagreement_seeds"])
    focused_cost = one_focused * focused_runs
    subtotal = open_cost + focused_cost
    contingency = subtotal * float(spec["retry_contingency"])
    total = subtotal + contingency
    return {
        "model": spec["model"],
        "prices": {
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "source": "https://docs.x.ai/developers/pricing",
            "read_on": "2026-10-14",
        },
        "open_coding_usd": round(open_cost, 4),
        "one_focused_reread_usd": round(one_focused, 4),
        "focused_runs": focused_runs,
        "focused_usd": round(focused_cost, 4),
        "contingency_usd": round(contingency, 4),
        "total_usd": round(total, 4),
        "target_usd": 2.0,
        "within_target": total <= 2.0,
        "live_calls": False,
        "note": (
            "This is a ceiling-style plan, not a measured bill. "
            "The harness does not place the calls. A person approves a live run first. "
            "Allowances in the package are ceilings, not measured costs."
        ),
    }


def ablation_selection(
    codebook,
    *,
    passages,
    segments=(),
    memos=(),
    sweep_log=(),
    pins=(),
    min_score=None,
) -> dict[str, Any]:
    """Selection with every analytic feature, then with each feature removed.

    Memo count can reward verbosity. The ablated row shows the proposal without it.
    """
    full = select_focused_codes(
        codebook,
        passages=passages,
        segments=segments,
        memos=memos,
        sweep_log=sweep_log,
        pins=pins,
        min_score=min_score,
    )
    dropped = {}
    for feature in ("memo", "in_vivo", "comparison"):
        kept = tuple(name for name in ("memo", "in_vivo", "comparison") if name != feature)
        dropped[feature] = select_focused_codes(
            codebook,
            passages=passages,
            segments=segments,
            memos=memos,
            sweep_log=sweep_log,
            pins=pins,
            min_score=min_score,
            analytic_features=kept,
        )
    return {
        "full": full,
        "ablated": dropped,
        "memo_note": MEMO_COUNT_NOTE,
        "changed_when_memo_ablated": full["raised_code_ids"] != dropped["memo"]["raised_code_ids"],
    }


def comparison_report(
    *,
    initial_code_ids: Sequence[str],
    focused_code_ids: Sequence[str],
    properties: Sequence[Mapping[str, Any]],
    counterexample_ids: Sequence[str],
    disagreements: Sequence[Mapping[str, Any]],
    prompt_log: Sequence[Mapping[str, Any]] = (),
    researcher_correction_minutes: float | None = None,
) -> dict[str, Any]:
    """Metrics for initial coding alone versus a focused reread.

    ``claim_support`` stays undefined until a researcher reviews. Citation
    anchoring is reported separately. Researcher correction time is undefined
    until someone records it. System disagreements are part of the published
    example, not dropped.
    """
    input_tokens = 0
    output_tokens = 0
    for row in prompt_log:
        usage = row.get("usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
    anchored = [
        row
        for row in properties
        if str(row.get("quote_status") or "") in {"exact", "normalized_match", "fuzzy_match"}
    ]
    supported = [row for row in properties if row.get("claim_support") == "supported"]
    undefined = sum(1 for row in properties if (row.get("claim_support") or "undefined") == "undefined")
    token_cost = _cost(input_tokens, output_tokens)
    per_claim = None if not supported else round(token_cost / len(supported), 6)
    revised = [code_id for code_id in focused_code_ids if code_id not in set(initial_code_ids)]
    return {
        "initial_codes": list(initial_code_ids),
        "focused_codes": list(focused_code_ids),
        "category_revisions": revised,
        "category_revision_count": len(revised),
        "properties": len(properties),
        "citation_anchored_properties": len(anchored),
        "supported_properties": len(supported),
        "claim_support_undefined": undefined,
        "counterexamples": list(counterexample_ids),
        "system_disagreements": [dict(item) for item in disagreements],
        "researcher_correction_minutes": researcher_correction_minutes,
        "researcher_correction_status": "undefined" if researcher_correction_minutes is None else "recorded",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_cost_usd": round(token_cost, 6),
        "cost_per_supported_claim_usd": per_claim,
        "cost_per_supported_claim_status": "undefined" if per_claim is None else "estimated",
        "live_calls": False,
    }
