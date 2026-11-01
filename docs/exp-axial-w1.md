# Experiment exp-axial-w1

Status: opt-in. Package `policy_version` stays `0.9.44`. The default axial planner, the default call ceilings (8 calls and 24,000 tokens), focused rereading, `pattern_min_cases` (2), and `one_document_one_respondent` (off) are unchanged. No LLM judge is added. This experiment does not call a model by itself. The caller passes the model. A test uses a fake.

`exp_axial_w1` is the entry point. `axial_code` does not select this draw unless that entry point passes `experiment`.

## Draw

The draw uses the same category universe as axial pair planning. When a finished focused selection limits later stages to raised codes, both classes are drawn from those codes only. The codebook may be the list `load_focused_export` returns. A list is not treated as empty. `pair_universe` is `n * (n - 1) / 2` for that n. The sample is `random.Random(seed)` over each class. The default seed is 4401. The executed round is a direct seeded draw of the scope that fits the caps. It is not a prefix of the 16+8 sample. If either class has fewer pairs that meet its rule, the run stops and names the shortfall. It does not take pairs from the other class.

## History and reuse

The examined-pair file is a local path the caller passes. The package does not write that file and it is not part of the repository. Prior outcomes in that file are an exclusion list. A stored proposal is applied only when its request fingerprint and its policy fingerprint both match. The request fingerprint covers the evidence payload, the model, and the passage set. The policy fingerprint covers the model and every prompt, including review. A match keeps every relationship on the stored proposal. A mismatch is recorded on the reuse ledger and the pair is sent. Prior spend is `historical_expenditure`. It is not added to this run. `new_spend.calls` counts calls sent, not pairs.

## Allowances

Proposals, quote-copy retries, counterexamples, relation review, and saturation review each have their own allowance. Calls are not moved between them. Retries are a shared pool of 6, not one retry per proposal. Provider retries are disabled (`provider_max_attempts` 1, so `max_retries` is 0) because a provider retry is not a separate reserved attempt. Quote-copy retries and schema repairs go through the accounted pool, one reservation each, before the call is sent. For this experiment the coding ceiling is 36 calls and 216,000 input tokens. It bounds proposals, retries, and counterexamples. Relation review is not under that ceiling. Its allowance is one call per row. The package defaults stay 8 and 24,000. Worst case is one proposal per pair, 6 retries, one counterexample per row, one relation-review call per row, and 1 saturation-review call. A counterexample that does not fit the coding ceiling is `required_but_blocked`. The run does not finish with a row left unreviewed. The 8+4 workload is 30 coding calls and 180,000 coding input tokens, inside that ceiling. The 43-call and 258,000-input-token total for 8+4 includes relation review and saturation, so it is larger than the coding-only ceiling on purpose. Core selection is not part of that total.

## Lexical eligibility

The rule is `lexical_cross_case_v1`. A lexical pair is eligible when it is absent from the examined-pair history and `classify_pair_evidence` places it in `lexical_cross_case`: a passage whose respondent is not an endpoint case contains at least two distinctive stems of each endpoint. `pattern_min_cases` does not admit or exclude that class. It still decides whether a finished relationship is downstream-eligible. If fewer than 8 pairs meet the rule, the run stops and reports the shortfall. It does not take pairs from the same-respondent class to fill the gap.

## Preflight

Prices are the frozen model config: $1.25 per million input tokens and $2.50 per million output tokens. Each call's input is the 6,000-token request cap. Each call's output is that stage's max output, the character heuristic of one response whose quotes, statements, and rationales are at the 800-character cap. Output dollars are part of the total.

The 16+8 worst case does not fit. Coding calls that can be sent are 36 (24 proposals, 6 retries, and 6 of the 24 required counterexamples). The other 18 counterexamples are `required_but_blocked`. Relation review is 24 calls. With saturation, that is 61 sent calls, 366,000 input tokens, 55,567 output tokens, and $0.596417. That is under the $0.60 hard cap and over the counterexample fit, so the draw still shrinks.

The hard cap is $0.60. The kill guard is $0.55. Both apply before every billable attempt, including a schema repair. The reservation prices the estimated input and that stage's maximum output. A reservation is not expenditure. An estimate is not provider usage. Known input and output tokens are settled after the call. A timeout counts as that reservation's possible spend and leaves measured dollars null. Usage the provider did not report stays null. The attempt ledger records the attempt id, stage, payload fingerprint, status, estimated input, output allowance, provider usage, reservation, and settlement. The draw shrinks by 4 pairs until the worst case fits. The scope that fits is 8 same-respondent pairs and 4 lexical pairs: 12 proposals, 6 retries, 12 counterexamples, 12 relation reviews, and 1 saturation review. That is 43 calls, 258,000 input tokens, 37,681 output tokens, and $0.416702. No counterexample is `required_but_blocked`. There is no core line.

A run with nothing to send returns `preflight_blocked` or `no_work`, with the reason and zero calls. It does not return `completed`. A restart restores spending and unresolved reservations from the ledger and does not reset the stage allowances. A resumed result is not charged as a new attempt.

## Core

Core selection is disabled for this experiment. The report records `status` `not_requested` and `reason` `evaluated separately`. There is no core call, no core reservation, and no core preflight line. A difference in relationship content does not start core selection.

## Pair coverage

Every selected pair has one terminal status: `attempted`, `successfully_examined`, `empty_response`, `failed`, or `budget_blocked`. `new_pairs_examined` counts `successfully_examined` only. A pair that returned no relationship used to be omitted from that count, which is why a 12-pair round could be reported as 11 examined. The missing pair now keeps its own status.

## Saturation

The planned saturation review is executed, or the report says it was `not_applicable` or `blocked` and gives the reason. A missing saturation record is not a completed review.

## Metrics

Each figure is null when it was not computed. The report breaks these out for same-respondent pairs, lexical pairs, and the round:

- new pairs actually examined (`successfully_examined`)
- grounded assertions per pair, from `downstream_eligible`, not from reviewer acceptance
- reviewer accepted and reviewer rejected, kept apart from eligibility failures
- additional supporting cases beyond the cases already on the two endpoints
- eligibility failures by reason, with each reason kept; `accepted` is not a failure reason
- counterexample calls, returned assessments, and relevant assessments; relevance is null only when no assessment was returned
- input tokens, dollars, and elapsed seconds

## Candidate manifest

`python scripts/exp_axial_w1_manifest.py --out /tmp/exp-axial-w1-candidate.json` writes a versioned candidate manifest pinned to `git rev-parse HEAD` in this package repository. It uses the same JSON fingerprint scheme as a startup fingerprint. `--source` can add an export manifest from a local file. Do not commit that file, the study export, or Muse ids. The offline gate fixture stays `tests/fixtures/exp_axial_w1_freeze.json`. Merging does not refresh it.

## Startup

The run refuses to start when the commit, the export manifest, the model config, the prompts, or the eligibility rules differ from the frozen fingerprints. The manifest uses the axial evidence hash, so a list codebook is read as codes rather than as an empty mapping. Prompts include proposal, counterexample, memo, storyline, and review. The commit is `git rev-parse HEAD` in this package's repository, not the process working directory.
