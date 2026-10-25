# 0.9.29 review fixes

Status: current for package 0.9.29. The 0.9.28 note stays the record of the accepted-downstream split, the per-step decision ceilings, the token preflight, and validated-chunk recovery. This note does not change `pattern_min_cases` (2), the axial token ceiling (24000), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added.

A decision review that reports every item reviewed, including a count such as 355 of 355, establishes processing coverage. A passing synthetic fixture does the same. Neither establishes that a selection is valid.

## Duplicate assertions keep every row's evidence

The same directed pair and the same normalized statement are still one relationship. The lowest relationship id is kept. Before that row is reassessed, provenance records each input row's supporting quotes, counterexamples, variation cases, negative cases, qualification, qualifications, review history, and any conflicting evidence already stored. Those lists are then unioned onto the kept row. The union is sorted by content, so input order and which row carries the lower id do not change the evidence that remains or the tier reassessment returns.

`conflicting_evidence` is that provenance when the rows disagree on tier, disposition, counterexamples, variation, negative cases, or qualification. It is null when they do not. Different supporting passages alone are not a conflict. An empty statement is not merged. The opposite direction and a different statement stay distinct. The rule is `merge_all_evidence_v1`.

## Exact hedges, not a word class

`if` is hypothetical only when it governs the from-to link. A discourse aside counts only when the whole span is one of these phrases and it does not state ability or permission: `if that makes sense`, `if you know what I mean`, `if I'm honest`, and `if I am honest`. `if I can`, `if you can`, `if possible`, `if you help`, `if you understand`, and `if it rains` are conditions. They stay in the quote. A hypothetical or ambiguous reading cannot authorize tier 3. `reading_provisional` stays true on a validated reading and does not by itself block tier 3.

`rule_authorized_attribution` is true when the deterministic rule assigns participant attribution. `human_validated_attribution` stays false unless that flag is already true on the row. The deterministic path does not set it. The two fields are separate. A rule-authorized tier 3 reading is not a human validation.

`score_conditional_readings` scores examples the caller passes in. `None` or an empty list leaves accuracy null and does not load the fixture. The synthetic fixture is `tests/fixtures/conditional_hedges_0929.json`. It is a processing check. It is not adjudication, and it is not selection validity. The held-out false-promotion set is `tests/fixtures/false_promotion_holdout_0929.json`. The adjudicator is the implementing agent on 2026-10-25, applying a written criterion. That person is not a study participant and not a second coder. There is no inter-rater check. The sentences are synthetic and were not taken from the aside list. The rate is false promotions divided by the cases whose label says tier 3 must not be authorized. The rule is `ability_permission_preserved_v1`.

## Budgeted consolidation

`selection_consolidation` is off unless the caller sets it. A near miss is a reviewed code that shared a chunk with a raised code. Rank is the number of raised chunks it shared, then code id. The cap is 8. A code that was reviewed and never sat beside a raise is not a near miss.

Each consolidation request is packed under the per-request ceiling. A set that does not fit in one request is split, and the group results are synthesized under the same ceiling. Request size and the call budget are separate. Before the first request, the pass counts the initial batches and an explicit synthesis budget. That total is compared with the calls still available after `claim_review_calls_reserved` is held back. When the total does not fit, the status is `skipped` with reason `completion_not_guaranteed` and nothing is sent. Claim review keeps those calls.

The status is `attempted` when some requests were sent and the pass stopped, `skipped` with a reason when none were sent, `failed` when a sent request returned an error or an invalid decision, or `completed`. `consolidation_requests_attempted` counts requests sent. `consolidation_responses_completed` counts accepted parses. `consolidation_closed` is true for a completed pass and for a skip that sent nothing, except `completion_not_guaranteed`. A failed or incomplete attempt leaves the pass open. The next selection visit that reaches the end of the chunks calls the pass again. That visit is the resume path. The pass does not retry inside one visit. A skip does not set the consolidation rule name and does not discard chunk reviews already stored. `consolidation_done` is not a result field.

The selection preflight reserves `consolidation_calls_reserved` and `consolidation_tokens_reserved` when consolidation is on. Those calls are not available to the selection chunks. The token figure is an input estimate, not a dollar bound. When consolidation is off, both reserves are 0 and the selection allowance is unchanged. The send-time allowance is `separate_consolidation_allowance_v1`.

## Chunk context and eligibility

`measure_chunk_context_sensitivity` treats a missing parse or a raised error as unknown. `code_raised` is null on that outcome, not false. `selection_changed` uses validated reviews only and is null when fewer than two succeeded. `attempted_reviews` and `successful_reviews` are both reported. The caller supplies the reviewer.

`compare_pipeline_runs` compares `downstream_eligible` on statements present in both runs. A boolean flip is `matched`. A statement only in the later run is an addition. A statement only in the earlier run is a removal. A missing boolean is `unknown_eligibility`. Duplicate statement keys with more than one distinct eligibility value are `conflicting_duplicates`, with the values sorted so row order does not change the report. None of those hide a flip on a matched statement. A run with no relationship list is an unavailable comparison. The summary aggregates the available comparisons and reports `comparisons_unavailable` against `comparisons_attempted`. The summary is null only when every comparison is unavailable. The rule is `order_invariant_matched_v1`.

## Resume

The fingerprint is `request_budget_v6`. `conditional_scope_rule` is `ability_permission_preserved_v1`. `consolidation_budget_rule` is `separate_consolidation_allowance_v1`. `eligibility_comparison_rule` is `order_invariant_matched_v1`. `duplicate_assertion_rule` is `merge_all_evidence_v1`. A checkpoint saved with the earlier 0.9.29 rule strings, or a 0.9.28 `request_budget_v5` checkpoint, raises `DecisionChunkingCheckpointError` before `allow_changes` is considered. The plan hash still does not include the call or token ceilings.

## What a frozen replay can show

Offline replay of frozen reviewer outputs checks how later stages behave on those outputs. It does not establish the quality of a live model judgment. The conditional fixture and the false-promotion holdout replay the deterministic rule. They are not an end-to-end benchmark of model selection, and they do not stand in for a live review.
