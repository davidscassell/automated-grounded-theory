# 0.9.12 review fixes

`policy_version` is `0.9.12`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. A metric that was not computed stays null. Human input still goes through `interrupt()` and `Command(resume=...)`. A sampling timeout resume is still `ainvoke(None)`. No live evaluation was run for this note.

## Recovered timeouts

A reread timeout sets `reread_status` and `reread_stop` to `timeout`. Resume sends the pair again. When that pair, and every other selected pair, then finishes, the completion path used to clear `stop` and `sampling_stop` and leave `reread_stop` as `timeout`. The ledger copied that stale value. The quiet-round check treats `reread_stop` `timeout` with no exempted pair as unresolved, so the default two-round window stayed `not_evaluable` after a successful recovery.

On that successful completion, `reread_stop` is cleared when it is `timeout`. `timeout_recovered` stays true. The prior timeout is appended to `reread_recovered_timeouts` with the pairs that had failed, and to `sampling_attempts` with status `recovered` and reason `timeout`. Those records are history. They are not the active failure. That split is a decision.

An unresolved timeout is unchanged. `reread_status` or `reread_stop` of `timeout` before a successful completion still leaves `sampling_rounds_without_new_directions` as `not_evaluable` with `observed` null.

## Question keys on pair records

A closure is a stable question key, such as `thin_category:c-prep`. A generated unsent row used to carry only `direction_id`, such as `dir-1-0001`. The exemption compared that id with `closed_direction_keys`, so a closed question never matched the passage that stayed unsent. Hand-built tests that already put `direction_key` on the row hid the mismatch.

`question_key` and `direction_key` are now the stable question on every row created for:

- `reread_unsent_pairs`
- `reread_failed_pairs`
- `reread_unattempted_pairs`
- `sampling_selection_unsent_pairs`
- `scheduled_reread_pairs`
- `sampled_pairs` on the ledger row

`direction_id` stays the round-specific id. The quiet-round exemption reads `question_key` first, then `direction_key`, then `direction_id`. A round id is not treated as a question key. That order is a decision. A closed question therefore exempts the unsent passage from the same question, including a `call_allowance` row.

`completed_pairs` already stored `question_key`. `model_sampling_pairs` and `selected_sampling_pairs` stay the model's direction id and passage id. `reread_active_pair` stays the in-flight direction id and passage id. When a timeout copies that active pair onto `reread_failed_pairs`, the copy receives the question key from the scheduled pair. Those boundaries are a decision.
