# Review fixes 0.9.26

`policy_version` is `0.9.26`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `axial_token_budget` stays 6000. Focused `decision_token_budget` stays 6000. `batch_size` stays 1. Tier 4 stays a human `verify`. A metric that was not computed stays null. No live or paid model call was run for this note. The axial planner still builds the full pair space in memory. That limit stays open.

This note reverses one 0.9.25 decision. The other 0.9.25 decisions stay: an incomplete selection or core choice is withheld, core candidates are the promoted focused categories, row eligibility gates components, `unique_pairs` is the examined count, and `cost_status` follows price configuration.

## Shared quotations

0.9.25 (`distinct_endpoint_quotes_v1`) dropped a quotation used for both endpoints before deciding whether a case supported both categories. The same source text then changed tier with the quote boundaries: one whole sentence for both endpoints stayed below tier 2, and two excerpts of that sentence met tier 2. No evidence had changed.

Decision: each endpoint is supported when the located text grounds it. Overlapping spans count. Identical spans count. Cases are counted by case identity, the respondent when the passage names one and otherwise the document. Distinct quote strings do not add a case, and one case cannot count twice. An identical span used for both endpoints sets `shared_quote_for_both_endpoints` and the inspect flag `shared_quote_inspect`. That flag does not disqualify the case and does not withhold eligibility. The rule is `endpoint_support_in_span_v1`. `pattern_min_cases` stays 2, so one case still cannot meet the floor by itself.

## Frequency before denial

Decision: the scope of a frequency qualifier is resolved before categorical rejection. `not always`, `doesn't always`, `don't usually`, `isn't always`, `not usually`, `not often`, `rarely`, and `sometimes` are `qualified`, polarity stays affirmative, and the qualification is `frequency_qualified`. An intervening adverb in that scope is the same reading, including `do not really always lead` and `does not necessarily lead`. Those readings are never `denied`. `do not lead`, `do not really lead`, `is not related`, `not what causes`, and `has nothing to do with` stay denied, with negative polarity. The rule is `frequency_before_rejection_v1`. It replaces `affirmative_rejection_v1` for this scope check. The rest of a denial is unchanged.

Every deterministic reading carries `reading_basis` `deterministic_provisional` and `reading_provisional` true. That marks the reading as provisional until it is checked on natural-language data. The synthetic fixture `tests/fixtures/link_denial_0925.json` is still not study data.

## Oversized selection items

Decision: a focused-selection code that alone exceeds the per-request budget is compacted and retried before it is omitted. Compaction is deterministic: one shorter quote, then no quotes, then a truncated definition. The code id stays. If a level fits, the code is reviewed in that form. It is not dropped.

If no level fits, the code is omitted with `single_code_exceeds_request_budget` and detail `omitted_oversized`. The other codes continue. Completed chunk reviews are kept and reused. The model cannot exclude the item. When every planned chunk has been validated and an omission remains, and interrupts are on, `selection_pause` asks for one of three resumes: `{"review_oversized": {"id": "...", "decision_token_budget": N}}` reviews that item at the changed request budget, `{"exclude_item": {"id": "...", "reason": "..."}}` records a human exclusion, and `{"leave_selection": true}` exports without excluding. The selection is complete only when every code was reviewed or explicitly excluded by a human. Exclusions are reported on `human_exclusions`.

Without a human (`interrupt` false), compaction still runs. If the item still does not fit, the selection stays provisional and the stop is `oversized_unrecovered`. Raising `max_decision_calls` does not review an item that was never packed. A review that validated nothing does not pause. Axial relationship and core omissions stay on the 0.9.25 omit-and-withhold path. The recovery rule is `compact_then_human_v1`.

## Fingerprint and over-withholding

The chunking fingerprint is `request_budget_v3`. Resume also requires `pattern_quote_rule`, `link_denial_rule`, and `oversized_recovery` to match the current values. A missing or 0.9.25 value raises `DecisionChunkingCheckpointError` before `allow_changes` is considered. The plan hash adds the compaction map. It still does not cover `max_tokens` or `max_calls`. A recovery chunk is stored on the open progress and does not rewrite that hash, so the completed chunks stay attached.

`export_gate_relationships` adds `withheld_reason_counts` and `inspect_flag_counts`. When the export has rows, each named reason is an integer, including zero. `shared_quote_inspect` is an inspect flag, so its withheld count can stay zero while the inspect count records it. Both maps are null when there is nothing to count. Other undefined rates stay null.

## Still open

The axial planner still builds the full pair space in memory. That limit stays open.
