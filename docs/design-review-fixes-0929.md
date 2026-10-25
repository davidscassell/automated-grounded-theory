# 0.9.29 review fixes

Status: current for package 0.9.29. The 0.9.28 note stays the record of ineligible rows, the first discourse-aside pass, duplicate-assertion merge, the shared ledger, and the chunk-allowance stop. This note revises the merge, the aside rule, and cross-chunk consolidation. It does not change `pattern_min_cases` (2), the axial token ceiling (24000), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`.

Completed review counts are processing coverage. They are not evidence that a selection was valid.

## Duplicate assertions keep the other row's evidence

The lowest relationship id is still the kept row. Supporting quotes are still unioned. Counterexamples, variation cases, negative cases, qualification flags and notes, and the review, claim-revision, and validation-revision histories are unioned before the caller reassesses the merged quotes. `assertion_provenance` stores that evidence for each source row. `conflicting_evidence` is that provenance when the pre-merge tiers, dispositions, qualification, or those lists differ, and null when they do not. Which id is lowest does not change the evidence retained. The rule is `merge_identical_statement_v2`.

## Ambiguous conditions stay conditions

An unambiguous discourse aside is one phrase from a closed set, such as `if that makes sense`, `if I can`, or `if possible`. The match is the whole span, not a whitelist of verbs. `if you help`, `if you understand`, `if it rains`, and `if the night is long` are not in that set. A condition the set does not name is preserved, including when it could also be read as a hedge. A preserved condition is hypothetical and does not authorize tier 3. A clear validated link, with the aside removed, can still reach tier 3. Every deterministic reading stays `reading_basis` `deterministic_provisional` and `reading_provisional` true. A provisional label does not by itself authorize the higher tier; authorization is a separate flag, and it is false while the condition is preserved.

The labelled fixture is `tests/fixtures/conditional_adjudication_0929.json`. The labels are `conversational_hedge`, `genuine_condition`, and `ambiguous_condition`. The sentences are synthetic. They are not study data, and they are not the end-to-end benchmark below. The rule is `ambiguous_condition_preserved_v1`.

## Budgeted cross-chunk consolidation

Near misses are still the reviewed codes that were not raised. Raised codes plus near misses are the reviewed set, and that set is not sent as one request unless the request fits the per-request token budget. When it does not, raised codes are kept beside as many near misses as fit, the remaining near misses are packed the same way, and the winners are synthesized under the same ceiling. No request that is sent is larger than `consolidation_request_budget`, which is the reviewer's per-request token budget.

Status is `attempted`, `skipped`, or `completed`. `consolidation_done` is true only for `completed`. A skip or a partial attempt keeps the chunk reviews and does not replace the raised ids. `nothing_to_consolidate` and `request_budget` are skip reasons when nothing is sent. One decision call is reserved when consolidation is on (`consolidation_calls_reserved`). Auto-sizing adds that call so the selection chunks still have their own allowance. The pass stays off unless the caller sets it. The rule is `budgeted_cross_chunk_v1`, and the reserve rule is `reserved_request_budget_v1`.

## Failed context reviews are unknown

`measure_chunk_context_sensitivity` writes `code_raised` null when the parse is missing or the audit status is `error`. The error status is kept. `selection_changed` is null when fewer than two reviews have a boolean `code_raised`, and otherwise true only when those booleans differ. `reviews_attempted` is the number of companion groups. `reviews_successful` is the number of boolean results. The caller still supplies the reviewer. The rule is `failures_unknown_v1`.

## Eligibility on matched statements

`compare_pipeline_runs` compares `downstream_eligible` where the same statement key is in both runs and both values are boolean. A key on only one side is an addition or a removal. A matched key that is not boolean on either side is `unknown_eligibility`. Those three lists do not erase `matched_changes`. The field stays null when a run has no relationship list, or both lists are empty. The rule is `matched_statements_v1`.

## Resume fingerprint

The fingerprint is `request_budget_v6`. It requires the 0.9.28 rule keys at their 0.9.29 values, plus `consolidation_budget_rule`, `context_sensitivity_rule`, and `eligibility_comparison_rule`. A missing key fails. A 0.9.28 checkpoint raises `DecisionChunkingCheckpointError` before `allow_changes` is considered.

## Adjudicated end-to-end benchmark (not implemented)

This is a design proposal. It is not built in 0.9.29, and it does not call a paid model.

The question is whether the pipeline's distinctions agree with a frozen human adjudication. A count of completed reviews, including a count such as 355/355, answers whether the items were processed. It does not answer whether the selection was valid. Adding more synthetic fixtures does not answer it either. The conditional fixture above is a labelled regression set for one rule. It is not this benchmark.

The benchmark would freeze three human-labelled sets and score them offline:

1. Useful distinctions. A human contrast set marks distinctions worth keeping and distinctions that should not be collapsed. The score is whether those distinctions are still present after selection, duplicate merge, and consolidation.
2. Cross-chunk categories. A human marks categories that are insignificant inside one chunk and significant across chunks. The score is whether budgeted consolidation recovers them. Coverage of the reviewed codes is reported beside that score and is not a substitute for it.
3. Evidence-tier agreement. Each relationship has a human tier. Agreement is reported, and disagreement is the result. A provisional deterministic reading does not count as agreement with a tier-3 adjudication.

The harness would replay frozen reviewer fixtures and the human labels. It would not send a live model request.
