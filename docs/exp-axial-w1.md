# Experiment exp-axial-w1

Status: opt-in. Package `policy_version` stays `0.9.44`. The default axial planner, the default call ceilings (8 calls and 24,000 tokens), focused rereading, `pattern_min_cases` (2), and `one_document_one_respondent` (off) are unchanged. No LLM judge is added. This experiment does not call a model by itself. The caller passes the model. A test uses a fake.

`exp_axial_w1` is the entry point. `axial_code` does not select this draw unless that entry point passes `experiment`.

## Draw

The draw uses the same category universe as axial pair planning. When a finished focused selection limits later stages to raised codes, both classes are drawn from those codes only. The codebook may be the list `load_focused_export` returns. A list is not treated as empty. `pair_universe` is `n * (n - 1) / 2` for that n. The sample is `random.Random(seed)` over each class. The default seed is 4401. The executed round is a direct seeded draw of the scope that fits the caps. It is not a prefix of the 16+8 sample. If either class has fewer pairs that meet its rule, the run stops and names the shortfall. It does not take pairs from the other class.

## History and reuse

The examined-pair file is a local path the caller passes. The package does not write that file and it is not part of the repository. Prior outcomes in that file are an exclusion list. A stored proposal is applied only when its request fingerprint and its policy fingerprint both match. The request fingerprint covers the evidence payload, the model, and the passage set. The policy fingerprint covers the model and every prompt, including review. A match keeps every relationship on the stored proposal. A mismatch is recorded on the reuse ledger and the pair is sent. Prior spend is `historical_expenditure`. It is not added to this run. `new_spend.calls` counts calls sent, not pairs.

## Allowances

Proposals, quote-copy retries, counterexamples, relation review, and saturation review each have their own allowance. Calls are not moved between them. Retries are a shared pool of 6, not one retry per proposal. For this experiment the coding ceiling is 36 calls and 216,000 input tokens. It bounds proposals, retries, and counterexamples. Relation review is not under that ceiling. Its allowance is one call per row. The package defaults stay 8 and 24,000. Worst case is one proposal per pair, 6 retries, one counterexample per row, one relation-review call per row, and 1 saturation-review call. A counterexample that does not fit the coding ceiling is `required_but_blocked`. The run does not finish with a row left unreviewed. The 8+4 workload is 30 coding calls and 180,000 coding input tokens, inside that ceiling. The 43-call and 258,000-input-token total for 8+4 includes relation review and saturation, so it is larger than the coding-only ceiling on purpose. Core, priced when the raised set is known, is additional.

## Lexical eligibility

The rule is `lexical_cross_case_v1`. A lexical pair is eligible when it is absent from the examined-pair history and `classify_pair_evidence` places it in `lexical_cross_case`: a passage whose respondent is not an endpoint case contains at least two distinctive stems of each endpoint. `pattern_min_cases` does not admit or exclude that class. It still decides whether a finished relationship is downstream-eligible. If fewer than 8 pairs meet the rule, the run stops and reports the shortfall. It does not take pairs from the same-respondent class to fill the gap.

## Preflight

Prices are the frozen model config: $1.25 per million input tokens and $2.50 per million output tokens. Each call's input is the 6,000-token request cap. Each call's output is that stage's max output, the character heuristic of one response whose quotes, statements, and rationales are at the 800-character cap. Output dollars are part of the total.

The 16+8 worst case does not fit. Coding calls that can be sent are 36 (24 proposals, 6 retries, and 6 of the 24 required counterexamples). The other 18 counterexamples are `required_but_blocked`. Relation review is 24 calls. With saturation, that is 61 sent calls, 366,000 input tokens, 55,567 output tokens, and $0.596417. That is under the $0.60 hard cap and over the counterexample fit, so the draw still shrinks.

The hard cap is $0.60. The kill guard is $0.55. Both apply at the common dispatch, before a proposal, retry, counterexample, relation review, saturation review, or core call. Known input and output tokens are priced. A timeout counts as that stage's worst-case possible spend. Usage the provider did not report stays unknown, and the measured dollar figure stays null. The draw shrinks by 4 pairs until the worst case fits. The scope that fits without core is 8 same-respondent pairs and 4 lexical pairs: 12 proposals, 6 retries, 12 counterexamples, 12 relation reviews, and 1 saturation review. That is 43 calls, 258,000 input tokens, 37,681 output tokens, and $0.416702. No counterexample is `required_but_blocked`. Core calls, one per candidate after the first, are added when the raised-set size is known.

## Core

Core selection stays deferred when the eligible relationships match the baseline by content, not by per-run id: endpoints, statement, quote, tier, eligibility, role, and document count, plus the per-category eligible counts. That deferral is recorded at $0. A missing baseline leaves the dollar figure null and does not run core. A difference runs core selection, and that call is under the same cap and kill guard.

## Metrics

Each figure is null when it was not computed. The report breaks these out for same-respondent pairs, lexical pairs, and the round:

- new pairs actually examined
- grounded assertions per pair
- additional supporting cases beyond the cases already on the two endpoints
- eligibility failures by reason
- counterexample relevance
- input tokens, dollars, and elapsed seconds

## Startup

The run refuses to start when the commit, the export manifest, the model config, the prompts, or the eligibility rules differ from the frozen fingerprints. The manifest uses the axial evidence hash, so a list codebook is read as codes rather than as an empty mapping. Prompts include proposal, counterexample, memo, storyline, and review. The commit is `git rev-parse HEAD` in this package's repository, not the process working directory.
