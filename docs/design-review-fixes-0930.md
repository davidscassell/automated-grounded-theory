# 0.9.30 review fixes

Status: record of package 0.9.30. Package 0.9.31 is current. The 0.9.29 note stays the record of merged duplicate evidence, the first closed hedge list, and the first consolidation budget. This note does not change `pattern_min_cases` (2), the axial token ceiling (24000), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added.

A decision review that reports every item reviewed, including a count such as 355 of 355, establishes processing coverage. A passing synthetic fixture does the same. Neither establishes that a selection is valid. Human-adjudicated examples are not in this package.

## Unambiguous hedges

`if` is hypothetical only when it governs the from-to link. A discourse aside counts only when the whole span is a phrase that cannot be a condition: `if that makes sense`, `if you know what I mean`, `if I'm honest`, and `if I am honest`. `if I can` and `if possible` can be genuine conditions, including `I'd share it if I can`. They stay in the quote. `if you help`, `if you understand`, and `if it rains` stay too. A hypothetical or ambiguous reading cannot authorize tier 3. `reading_provisional` stays true on a validated reading and does not by itself block tier 3.

`score_conditional_readings` scores examples the caller passes in. `None` or an empty list leaves accuracy null and does not load a fixture. The synthetic fixture is `tests/fixtures/conditional_hedges_0929.json`. It is a processing check. It is not adjudication, and it is not selection validity. The rule is `unambiguous_hedge_v1`. A 0.9.29 checkpoint stored `closed_hedge_v1` and cannot resume.

## Failed consolidation stays open

The status is `failed` with an error, `skipped` with a reason when the pass was not attempted, or `completed`. A review error, an invalid decision, or an exception is a failure. A failure does not set `consolidation_closed`, does not set `selection_consolidation`, and does not use `selection_consolidation_skipped`. Chunk reviews already stored are kept. The selection is not applied, and a later visit can retry the pass. A skip, including nothing to consolidate or a request that does not fit before anything is sent, can still close. It is not done.

## Separate consolidation reserve

The preflight records `consolidation_calls_reserved` and `consolidation_tokens_reserved` when consolidation is on. The estimate packs every id under the per-request ceiling. When that takes more than one request, it also reserves a synthesis of the full id list, not one representative from each request. Those calls and tokens are not subtracted from the selection allowance, the claim-review allowance, or a later step. Spend on the pass is removed from the shared counters when the pass returns. `consolidation_reserve_separate` is true only when the caller turns the pass on. The figure is an input estimate, not a dollar bound. When consolidation is off, both reserves are 0 and the allowances are unchanged. Turning the pass on does not raise the default call or token ceilings.

## Order-independent eligibility

A comparison matches on the statement plus the directed pair. Duplicate rows for that key are combined without regard to order. If every boolean agrees, that value is the eligibility. If the rows disagree, or a boolean is missing beside another row, the statement is ambiguous and is not a flip. The summary lists every matched flip from a pairwise comparison that has relationship rows. A comparison with no rows stays null and does not hide flips from the other comparisons. An unknown row does not hide a flip in the same comparison.
