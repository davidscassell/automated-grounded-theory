# 0.9.18 review fixes

`policy_version` is `0.9.18`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. A metric that was not computed stays null. No live evaluation was run for this note. The 0.9.17 notes stay as history.

## Timed-out rereads

Decision: a timeout that spent a provider call during a reread keeps the operation that was in flight. An initial reread is logged as `sampling_reread`. A reread repair, while `reread_repair_pending` is set, is logged as `schema_repair` with `schema` `SamplingRereadNote`. The row has no messages and no usage. Expenditure treats that schema field as a `SamplingRereadNote` repair, because an empty message list cannot be scanned for the schema name. A selection repair is not given that field.

A timed-out request may still be charged. Its usage stays unknown. It counts in `reread_expenditure_calls` and not in `reread_expenditure_calls_with_usage`. Coverage is the ratio of those two. `reread_expenditure_complete` is false. `expenditure_tokens_per_relevant_reread`, `expenditure_cost_per_relevant_reread_usd`, and `reread_expenditure_cost_usd` stay null. The known token sum excludes the attempt. `tokens_per_relevant_reread` still counts only attributable tokens from successful relevant responses.

A timeout with no later success is coverage 0 and a known sum of 0. Cost stays null when prices are set. That is not the complete cost of 0 reported when no reread was sent.

Selection and saturation timeouts stay kind `provider`. They are not reread spend. Axial timeout rows stay kind `provider`. Focused coding does not write this prompt row. The limiter's allowance kind stays `provider`. `provider_budget.calls_used` is still copied from the limiter. It is not recomputed from the prompt log.
