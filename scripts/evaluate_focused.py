#!/usr/bin/env python3
"""Compare initial coding with focused rereading, offline, and estimate a live run.

This script does not call a model provider. ``--estimate`` prints the proposed
live cost for grok-4.20-0309-non-reasoning. ``--live`` exits without calling
the API. A person has to approve a live run before one is started.

    python scripts/evaluate_focused.py --estimate
    python scripts/evaluate_focused.py --offline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from automated_grounded_theory.evaluation_plan import ablation_selection, comparison_report, estimate_live_cost


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Focused-coding evaluation harness. No live API calls.")
    parser.add_argument("--estimate", action="store_true", help="Print the proposed live cost and exit.")
    parser.add_argument("--offline", action="store_true", help="Run the fake-model comparison and print the report.")
    parser.add_argument("--live", action="store_true", help="Refused. Live calls need a separate approval.")
    parser.add_argument("--ablate", default="", help="Comma-separated analytic features to drop: memo,in_vivo,comparison.")
    args = parser.parse_args(argv)
    if args.live:
        print(
            "Live API calls are not run by this harness. "
            "Approve a live run separately. See docs/focused-evaluation.md.",
            file=sys.stderr,
        )
        return 2
    if args.estimate or not args.offline:
        print(json.dumps(estimate_live_cost(), indent=2))
    if args.offline:
        from tests.fake_focused import FakeFocusedCoder

        from automated_grounded_theory.focused import focused_code
        from tests.test_v070 import _book, _memos, _passages, _segments_for_raised, _source, _sweep

        features = [item.strip() for item in args.ablate.split(",") if item.strip()]
        kept = tuple(name for name in ("memo", "in_vivo", "comparison") if name not in features)
        ablation = ablation_selection(
            _book(),
            passages=_passages(),
            memos=_memos(),
            sweep_log=_sweep(),
            pins=["c-rare"],
            min_score=1.0,
        )
        coder = FakeFocusedCoder(mode="drop")
        source = _source(segments=_segments_for_raised()[:1], memos=_memos(), sweep=_sweep())
        result = focused_code(
            source,
            model=coder,
            seed=1,
            pins=["c-rare"],
            focused_min_score=1.0,
            focused_neighbor_limit=0,
            focused_outside_limit=0,
            focused_analytic_features=kept if features else None,
            interrupt=False,
            max_focused_review_calls=4,
        )
        disagreements = [
            {
                "segment_id": segment.get("segment_id"),
                "code_id": segment.get("code_id"),
                "review_reasons": list(segment.get("review_reasons") or []),
            }
            for segment in result.segments
            if "focused_disagreement" in (segment.get("review_reasons") or [])
        ]
        report = comparison_report(
            initial_code_ids=[row["code_id"] for row in _book()],
            focused_code_ids=result.raised_code_ids,
            properties=result.properties,
            counterexample_ids=result.metadata.get("purpose_coverage", {}).get("compete", {}).get("reread", []),
            disagreements=disagreements,
            prompt_log=result.prompt_log,
        )
        report["ablation"] = {
            "dropped": features,
            "full_raised": ablation["full"]["raised_code_ids"],
            "memo_ablated_raised": ablation["ablated"]["memo"]["raised_code_ids"],
            "memo_note": ablation["memo_note"],
            "changed_when_memo_ablated": ablation["changed_when_memo_ablated"],
        }
        report["system_disagreements_published"] = disagreements
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
