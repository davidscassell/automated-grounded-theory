# Focused reread evaluation

This comparison asks what a focused reread adds to initial coding on a small corpus a researcher can review. It is not a saturation decision. `scripts/evaluate_focused.py` runs the offline half and prints a live-cost estimate. It does not call an API. A person approves a live run before any call is made.

## What is compared

Initial coding alone, then the same book after focused rereading.

| Measure | Where it comes from |
| --- | --- |
| Category revisions | Codes raised or revised in the focused run that were not the initial set |
| Citation-anchored properties | Quotes whose status is `exact`, `normalized_match`, or `fuzzy_match` |
| Supported properties | `claim_support == supported`. Unreviewed rows stay `undefined` and are reported as undefined |
| Counterexamples | Passages reread under `compete`, including within-category negative cases |
| System disagreements | Assignments flagged `focused_disagreement`, kept in the published example |
| Researcher correction time | Minutes the researcher records. Undefined until then |
| Token cost | Input and output tokens from the prompt log, priced at the rates below |
| Cost per supported claim | Token cost divided by claims the researcher marked `supported`. Undefined when that count is zero |

Citation anchoring is not claim support. A quote can locate a span and still be contradicted.

## Ablation

`focused_analytic_features` drops `memo`, `in_vivo`, or `comparison` from the selection score. Memo count can reward a verbose memo. The offline report prints whether the raised set changes when memo count is removed. `--ablate memo` passes the remaining features into the focused run.

## Live plan

Model: `grok-4.20-0309-non-reasoning`.

Prices read 2026-10-14 from the xAI pricing page (https://docs.x.ai/developers/pricing) and the model page (https://docs.x.ai/developers/models/grok-4.20-0309-non-reasoning). Standard context under 200k tokens: $1.25 per million input tokens, $2.50 per million output tokens. Cached input and the long-context rates are not in this plan. Package allowances are ceilings, not these token counts. The counts below are the planning estimate.

| Piece | Calls | Input tokens each | Output tokens each | Runs |
| --- | --- | --- | --- | --- |
| Open coding, about 40 incidents | 40 | 4,000 | 800 | 1 |
| Focused reread, balanced budget | 24 | 6,000 | 1,000 | 1 |
| Ablation rereads (memo, in-vivo, comparison) | 24 | 6,000 | 1,000 | 3 |
| Extra focused seeds for disagreements | 24 | 6,000 | 1,000 | 2 |

That is one open-coding pass and six focused passes. Subtotal about $1.72. A 15% retry contingency brings the plan to about $1.98, under a $2 target. `estimate_live_cost()` prints the figure. `--live` exits without calling the provider.

Researcher review of the claims is human time. It is not in the API estimate. Correction minutes stay undefined until the researcher writes them down.

Do not start this run from the script. Approve the corpus, the model, and the spend first.
