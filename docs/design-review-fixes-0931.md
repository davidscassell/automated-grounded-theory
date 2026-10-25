# 0.9.31 review fixes

Status: current for package 0.9.31. The 0.9.30 note stays the record of unambiguous hedges, a failed consolidation left open, and the separate consolidation reserve. This note does not change `pattern_min_cases` (2), the axial token ceiling (24000), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added.

A decision review that reports every item reviewed, including a count such as 355 of 355, establishes processing coverage. A passing synthetic fixture does the same. Neither establishes that a selection is valid. Human-adjudicated examples are not in this package. No adjudication file was available for the condition checks below.

## Condition scope

A condition marker governs the from-to link. The markers include `if`, `unless`, `provided` and `providing` with or without `that`, `as long as`, `so long as`, `only when`, `in case`, `assuming` with or without `that`, `on condition that`, `except when`, and `except if`. The span is stored on the reading as `condition`. `the reason`, `part of why`, and `a reason` before `if` do not remove it. The only noun-phrase embedding this rule accepts is `question of if` or `question of whether`. Anything else keeps the condition.

The proposed statement has to contain that span. `condition_retained` is false when it does not, the qualification is `condition_not_in_statement`, and the quote cannot authorize tier 3. A statement that keeps the condition is still conditional: the status stays hypothetical, the qualification is `participant_link_conditional`, and the tier stays below 3. The relationship is allowed. It is qualified. Discourse asides that cannot be a condition (`if that makes sense`, `if you know what I mean`, `if I'm honest`, `if I am honest`) are unchanged. `if I can` and `if possible` stay in the quote.

The rule is `condition_scope_v1`. A 0.9.30 checkpoint stored `unambiguous_hedge_v1` and cannot resume. `request_budget_v6` is unchanged. The synthetic checks cover the four reported sentences, paraphrases of the same condition, and exception markers. They are processing coverage. They are not selection validity.

## Every consolidation candidate is accounted for

Each packing level, including a later synthesis packing, keeps the packer's omissions. An id that was never sent, or that was omitted after an earlier group, is `unreviewed-oversized` or `unreviewed-not-sent` and `unresolved`. Completion is allowed only when every candidate has an outcome and none are unresolved. The outcomes are `reviewed-kept`, `reviewed-merged`, `reviewed-dropped`, `unreviewed-oversized`, `unreviewed-not-sent`, and `unresolved-synthesis`. The focused export lists `selection_consolidation_candidates` and `selection_consolidation_unresolved_ids`. While any candidate is unresolved, the provisional raised ids stay as they were. The pass does not apply a partial kept set and does not set `selection_consolidation`.

## Attempt ledger

`consolidation_attempts` is append-only. Each entry has the calls, tokens, status, reason, and error for that attempt. `consolidation_calls_charged` and `consolidation_tokens_charged` are the sums. `consolidation_attempt_calls` and `consolidation_attempt_tokens` are the latest attempt. A later attempt does not call the reviewer once the cumulative calls or tokens have already met the reserved allowance. Spend is still removed from the shared decision counters, so claim review is not charged for the pass.

## Reuse and synthesis that cannot fit

A validated group is stored with its code ids, raised ids, and prompt hash. A retry with the same payload reuses that group and does not send it. When the next packing repeats a grouping that was already reviewed, the pass does not report `request_budget` and does not send those groups again. It tries one compacted request that still names every remaining id. If that request fits the per-request budget and the remaining reserve, it is sent once. If it does not fit, the status is `unresolved-synthesis`, the reason is `synthesis_cannot_fit`, and there is no error. The same budget does not try that summary again. A larger per-request budget can. An exception, a missing parse, or an invalid decision stays `failed`. Groups that already validated are reused, so the retry sends only the group that failed. The pass does not ask the model to drop a valid category in order to shrink the payload.

A failure and either unresolved status leave the stage open. Chunk reviews already stored are kept. The selection is not applied. A later visit can resume. A skip before any request is sent, including nothing to consolidate, can still close and is not done.
