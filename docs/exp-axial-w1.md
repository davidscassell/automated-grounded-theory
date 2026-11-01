# Experiment exp-axial-w1

Status: opt-in. Package `policy_version` stays `0.9.44`. The default axial planner, the default call ceilings (8 calls and 24,000 tokens), focused rereading, `pattern_min_cases` (2), and `one_document_one_respondent` (off) are unchanged. No LLM judge is added. This experiment does not call a model by itself. The caller passes the model. A test uses a fake.

`exp_axial_w1` is the entry point. `axial_code` does not select this draw unless that entry point passes `experiment`.

## Draw

The draw is 16 previously unexamined same-respondent pairs whose document co-occurrence is 1, plus 8 previously unexamined lexical cross-case pairs. The sample is `random.Random(seed)` over each class in that order. The default seed is 4401 and the seed is stored. If either class has fewer pairs that meet its rule, the run stops and names the shortfall. It does not take pairs from the other class.

## History and reuse

The examined-pair file is a local path the caller passes. The package does not write that file and it is not part of the repository. Prior outcomes in that file are an exclusion list. A stored proposal is applied only when its request fingerprint and its policy fingerprint both match the pair and the current prompt and eligibility policy. A mismatch is recorded on the reuse ledger and the pair is sent. Request fingerprints cover the pair and the axial proposal prompt. Policy fingerprints cover `pattern_min_cases`, `one_document_one_respondent`, `policy_version`, and this experiment id. Prior spend is `historical_expenditure`. It is not added to this run's tokens or dollars.

## Allowances

Proposals, quote-copy retries, counterexamples, relation review, and saturation review each have their own allowance. Calls are not moved between them. Retries are a shared pool of 6, not one retry per proposal. For this experiment the coding ceiling is 36 calls and 216,000 input tokens. The package defaults stay 8 and 24,000. Worst case is one proposal per pair, 6 retries, one counterexample per row, one relation-review call per row, and 1 saturation-review call. A counterexample that does not fit the coding ceiling is `required_but_blocked`. Relation review is sized to the row count. A shortfall reduces the draw. The run does not finish with a row left unreviewed.

## Lexical eligibility

The rule is `lexical_cross_case_v1`. A lexical pair is eligible when it is absent from the examined-pair history and `classify_pair_evidence` places it in `lexical_cross_case`: a passage whose respondent is not an endpoint case contains at least two distinctive stems of each endpoint. `pattern_min_cases` does not admit or exclude that class. It still decides whether a finished relationship is downstream-eligible. If fewer than 8 pairs meet the rule, the run stops and reports the shortfall. It does not take pairs from the same-respondent class to fill the gap.

## Preflight

Prices are the frozen model config: $1.25 per million input tokens and $2.50 per million output tokens. Each call's input is the 6,000-token request cap. Each call's output is that stage's max output, the character heuristic of one response whose quotes, statements, and rationales are at the 800-character cap. Output dollars are part of the total.

The 16+8 worst case does not fit. Coding calls that can be sent are 36 (24 proposals, 6 retries, and 6 of the 24 required counterexamples). The other 18 counterexamples are `required_but_blocked`. Relation review is 24 calls. With saturation, that is 61 sent calls, 366,000 input tokens, 55,567 output tokens, and $0.596417. That is under the $0.60 hard cap and over the counterexample fit, so the draw still shrinks.

The hard cap is $0.60. The runtime kill guard is $0.55. The draw shrinks by 4 pairs and the 2:1 class ratio is kept as nearly as the integer split allows. The reduced scope is recorded before the run. The scope that fits is 8 same-respondent pairs and 4 lexical pairs: 12 proposals, 6 retries, 12 counterexamples, 12 relation reviews, and 1 saturation review. That is 43 calls, 258,000 input tokens, 37,681 output tokens, and $0.416702. No counterexample is `required_but_blocked` and the relation-review shortfall is 0.

## Core

Core selection stays deferred when the downstream-eligible relationship ids and the per-category eligible counts match the baseline the caller supplies. That deferral is recorded at $0. A missing baseline leaves the dollar figure null and does not run core. A difference runs core selection and is not described as deferred.

## Metrics

Each figure is null when it was not computed. The report breaks these out for same-respondent pairs, lexical pairs, and the round:

- new pairs actually examined
- grounded assertions per pair
- additional supporting cases beyond the cases already on the two endpoints
- eligibility failures by reason
- counterexample relevance
- input tokens, dollars, and elapsed seconds

## Startup

The run refuses to start when the commit, the export manifest, the model config, the prompts, or the eligibility rules differ from the frozen fingerprints. The manifest hashes code, passage, and segment ids. It does not hash passage text.
