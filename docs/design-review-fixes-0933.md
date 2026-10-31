# 0.9.33 review fixes

Status: record of package 0.9.33. Package 0.9.39 is current. The 0.9.32 note stays the record of per-step token allowances, respondent case counts, and the post-selection reread estimate. This note does not change `pattern_min_cases` (2), the axial token number (24000 per step), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added. No LLM judge is added.

## Decisions

Core selection does not run a tournament. If every eligible candidate fits one per-request budget, that one request is the comparison, including when the eligible set has a single candidate. If the set does not fit, the stop is `core_candidates_do_not_fit`. No candidate is chosen from a smaller request. A withheld plan cannot resume into a larger per-request budget, because that budget is part of the plan hash. Raising the call allowance can continue a plan that already fits.

`one_document_one_respondent` defaults to false. Setting it says each document is one respondent. A passage that already names a respondent still uses that id.

The coding-pool warning still compares the proposal, counterexample, memo, and storyline estimate with 24000. That pool is not the decision allowance. Relation review, saturation review, and core selection each keep their own 24000, and they are not in the estimate.

## Reread requests

Levels 0 through 3 shorten the codebook: full definitions, then other-code definitions dropped, then every definition dropped, then raised codes only. The first level that fits `token_budget` is what is sent. A passage that does not fit at level 3 is listed on `reread_requests_over_budget`. The queue continues. Claim review still opens when it was reserved. A synthetic counterexample is a real request: it is sized and sent. Coverage helpers still omit it from corpus counts. An empty batch is not how that request is recorded. The estimate's token total and `focused_review_requests_projected` include only requests that fit the cap and the remaining token allowance. `focused_review_requests_withheld` counts passages that cannot fit the cap. The worst-case preflight reports the same projected fields, stays labeled as a worst case, and does not block selection.

## Core selection

`core_candidates_eligible` is the size of the eligible set. `core_candidates_compared` is the number of candidates in the request that was sent, or 0 when the request was not sent. The instruction says the request contains every eligible candidate and that a one-candidate request is not a reason to choose. Both counts are on the run metadata and in `core_candidates.json`.

## Conditions

`presented_statement` is the sentence a reader sees. When the recorded condition is missing from the model's `statement`, the span is appended in parentheses and `statement_omits_condition` is true. The flag `condition_not_in_statement` is on the row. `condition_retained` stays false, so the quote still cannot authorize tier 3. A statement that already contains the span is not changed. No condition leaves `statement_omits_condition` null.

## Labels

`budget_ledger_rule` is `per_step_review_ledger_v1`. Axial decision steps use `allowance_scope` `per_step_decision_review` and `shared_with` `[]`. Focused selection and claim review still share `focused_decision_review`. The warning text no longer says the planning estimate exceeds `max_axial_review_tokens` as if that number were one decision ceiling.

## Case identity

`case_identity_source` on each supporting case is `respondent`, `one_document_one_respondent`, `document_fallback`, or `passage`. `case_identity_summary` on the relationship and on the run counts them. `case_counts_are_respondents` is true only when every counted case is a respondent or a declared document, false when a fallback was used, and null when there are no cases. Tier-2 wording follows that boolean.

## Resume

The fingerprint stays `request_budget_v6`. The new keys are `core_comparison_rule`, `reread_request_rule`, `condition_statement_rule`, and `case_identity_source_rule`. `budget_ledger_rule` changed value. A 0.9.32 checkpoint is missing those values or carries the old ledger rule, and it cannot resume.

These checks are processing coverage. They are not selection validity.
