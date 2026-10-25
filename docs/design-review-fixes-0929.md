# 0.9.29 review fixes

Status: record of package 0.9.29. Package 0.9.30 is the current review note. The 0.9.28 note stays the record of the accepted-downstream split, the per-step decision ceilings, the token preflight, and validated-chunk recovery. This note does not change `pattern_min_cases` (2), the axial token ceiling (24000), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added.

A decision review that reports every item reviewed, including a count such as 355 of 355, establishes processing coverage. A passing synthetic fixture does the same. Neither establishes that a selection is valid.

## Duplicate assertions keep every row's evidence

The same directed pair and the same normalized statement are still one relationship. The lowest relationship id is kept. Before that row is reassessed, provenance records each input row's supporting quotes, counterexamples, variation cases, negative cases, qualification, qualifications, review history, and any conflicting evidence already stored. Those lists are then unioned onto the kept row. The union is sorted by content, so input order and which row carries the lower id do not change the evidence that remains or the tier reassessment returns.

`conflicting_evidence` is that provenance when the rows disagree on tier, disposition, counterexamples, variation, negative cases, or qualification. It is null when they do not. Different supporting passages alone are not a conflict. An empty statement is not merged. The opposite direction and a different statement stay distinct. The rule is `merge_all_evidence_v1`.

## Exact hedges, not a word class

`if` is hypothetical only when it governs the from-to link. A discourse aside counts only when the whole span is one of these phrases: `if that makes sense`, `if I can`, `if possible`, `if you know what I mean`, `if I'm honest`, and `if I am honest`. `if you help`, `if you understand`, and `if it rains` are conditions. They stay in the quote. A hypothetical or ambiguous reading cannot authorize tier 3. `reading_provisional` stays true on a validated reading and does not by itself block tier 3.

`score_conditional_readings` scores examples the caller passes in. `None` or an empty list leaves accuracy null and does not load the fixture. The synthetic fixture is `tests/fixtures/conditional_hedges_0929.json`. It is a processing check. It is not adjudication, and it is not selection validity. Human-adjudicated examples are not included. Checking these readings against adjudicated examples would need that data, and this package does not have it. The rule is `closed_hedge_v1`.

## Budgeted consolidation

`selection_consolidation` is off unless the caller sets it. A near miss is a reviewed code that shared a chunk with a raised code. Rank is the number of raised chunks it shared, then code id. The cap is 8. A code that was reviewed and never sat beside a raise is not a near miss.

Each consolidation request is packed under the per-request ceiling. A set that does not fit in one request is split, and the group results are synthesized under the same ceiling. A request that still cannot be sent is not sent. The status is `attempted` when some requests were sent and the pass stopped, `skipped` with a reason when none were sent, or `completed`. A skip does not set the consolidation rule name and does not discard chunk reviews already stored. `consolidation_done` is not a result field.

The selection preflight reserves `consolidation_calls_reserved` and `consolidation_tokens_reserved` when consolidation is on. Those calls are not available to the selection chunks. The token figure is an input estimate, not a dollar bound. When consolidation is off, both reserves are 0 and the selection allowance is unchanged.

## Chunk context and eligibility

`measure_chunk_context_sensitivity` treats a missing parse or a raised error as unknown. `code_raised` is null on that outcome, not false. `selection_changed` uses validated reviews only and is null when fewer than two succeeded. `attempted_reviews` and `successful_reviews` are both reported. The caller supplies the reviewer.

`compare_pipeline_runs` compares `downstream_eligible` on statements present in both runs. A boolean flip is `matched`. A statement only in the later run is an addition. A statement only in the earlier run is a removal. A missing boolean is `unknown_eligibility`. None of those hide a flip on a matched statement. The comparison is null only when both sides have no relationship rows.

## Resume

The fingerprint is `request_budget_v6`. `conditional_scope_rule` is `closed_hedge_v1`. `duplicate_assertion_rule` is `merge_all_evidence_v1`. A 0.9.28 `request_budget_v5` checkpoint raises `DecisionChunkingCheckpointError` before `allow_changes` is considered. The plan hash still does not include the call or token ceilings.
