# Experiment exp-axial-w1

Status: opt-in. Package `policy_version` stays `0.9.44`. The default axial planner, the default call ceilings (8 calls and 24,000 tokens), focused rereading, `pattern_min_cases` (2), and `one_document_one_respondent` (off) are unchanged. No LLM judge is added. This experiment does not call a model by itself. The caller passes the model. A test uses a fake.

`exp_axial_w1` is the entry point. `axial_code` does not select this draw unless that entry point passes `experiment`.

## Draw

The draw is 16 previously unexamined same-respondent pairs whose document co-occurrence is 1, plus 8 previously unexamined lexical cross-case pairs. Both classes use `classify_pair_evidence`. A pair that is not in that class, or that is already in the examined-pair file, is not eligible. The sample is `random.Random(seed)` over each class in that order. The default seed is 4401 and the seed is stored. If either class has fewer eligible pairs than the draw, the run stops and names the shortfall. It does not take pairs from the other class.

## History and reuse

The examined-pair file is a local path the caller passes. The package does not write that file and it is not part of the repository. Prior outcomes in that file are an exclusion list. A stored proposal is applied only when its request fingerprint and its policy fingerprint both match the pair and the current prompt and eligibility policy. A mismatch is recorded on the reuse ledger and the pair is sent. Request fingerprints cover the pair and the axial proposal prompt. Policy fingerprints cover `pattern_min_cases`, `one_document_one_respondent`, `policy_version`, and this experiment id. Prior spend is `historical_expenditure`. It is not added to this run's tokens or dollars.

## Allowances

Proposals, quote-copy retries, counterexamples, relation review, and saturation review each have their own allowance. Calls are not moved between them. For this experiment the coding ceiling is 36 calls and 216,000 tokens. The package defaults stay 8 and 24,000. The worst-case plan is one proposal and one retry per pair, 4 counterexample calls, up to 4 relation-review calls, and 1 saturation-review call. Each of those calls is priced at the 6,000-token request cap.

## Preflight

The 16+8 draw is priced before any model call. With an input price configured, that worst case is 57 calls and 342,000 input tokens: 24 proposals, 24 retries, 4 counterexamples, 4 relation reviews, and 1 saturation review. Coding calls are 52, above the 36-call ceiling. Output dollars stay null unless an output price is set. A missing input price leaves dollars null and does not count as fitting the cap.

The hard cap is $0.60. The runtime kill guard is $0.55. If the worst case does not fit the coding ceiling or the hard cap, the draw shrinks by 4 pairs and the 2:1 class ratio is kept as nearly as the integer split allows. The reduced scope is recorded before the run. At $3 per million input tokens the scope that fits is 8 same-respondent pairs and 4 lexical pairs: 33 calls, 198,000 input tokens, and $0.594.

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
