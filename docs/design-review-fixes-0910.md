# 0.9.10 review fixes

`policy_version` is `0.9.10`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. A metric that was not computed stays null. Human input still goes through `interrupt()` and `Command(resume=...)`. No live evaluation was run for this note.

## Pair planning

`plan_relation_pairs` takes pairs that already have evidence before the salience and random fill: same-passage co-occurrence, then the same respondent, then a lexical cross-case candidate. The fill is still seeded. The stage reports pairs examined, pairs that were plausible, and how many of the round each evidence class prioritized.

`max_axial_review_calls` still defaults to 8. A default series therefore examines a small fraction of the unordered pairs. That default is not raised here.

A corpus in which no candidate pair shares `pattern_min_cases` respondents records `pattern_reach_warning`, including how many codes occur in one document. The minimum is not changed and pairs are still planned.

`case_id_of` reads `metadata.respondent_id` (and `participant_id` or `case_id` there) after a top-level id and before the document id. Passages keep the respondent on `metadata`. Question-documents from one respondent collapse to one case, so a tier-2 count on a multi-question corpus gets stricter. That is a decision, recorded in the pull request. `pattern_min_cases` stays 2.

A passage that shares only one distinctive stem with either endpoint is not offered as `candidate_cross_case`. It is labeled `lexical_only` with `support_cap` 1. A candidate still needs at least two distinctive stems on each endpoint. That is a decision. The planner does not treat the one-stem list as an evidence class.

## Proposals

A tier-1 hypothesis may rest on one passage or one respondent. The prompt says so, and it says not to default to an empty list. A statement built by joining quotes from passages that share neither a passage nor a respondent is still rejected (`stitched_unrelated_passages`). The schema allows `nothing_to_propose` and nullable draft fields, so an explicit empty outcome is not a row of empty strings. A row with no statement and no quotes is `blank_proposal`. A row with quotes and no statement is `empty_statement`. Those counts are separate.

## Wording, absence, and stem scope

When `wording_exceeds_tier` is set on a relationship below tier 3, `downstream_eligible` is false and the row is flagged `causal_wording_withheld`. The statement is not rewritten. That is a decision. A reviewer rationale that says the text uses non-causal wording is read as a denial of causal language, including when the statement says "leads to", and `reviewer_causal_claim_mismatch` is set.

An absence-style candidate (`a_without_b`, `b_without_a`, and the category/outcome forms) that the model marks `contradicts` or `qualifies` is stored as `qualifies` with reason `absence_is_not_contradiction`. An accepted row keeps `relationship_qualified` and that reason. An empty `candidate_type` is not reclassified.

One distinctive stem is still not an anchor. When that weak anchor sits in the cited passage, `evidence_scope` is `same_passage_stem_only` rather than `same_passage`. The row stays `proposed`. The same-passage quote "the people in my shift. management not so much" is retained at tier 1 as a proposal. That is a decision. `weak_stem_anchor` remains, and the row is not downstream eligible because the stem is not anchored.

## Sampling

Sampling baselines carried token estimates the same way it baselines calls. `sampling_token_baseline` records the upstream estimate and this stage's `tokens_estimated` starts at 0, so the first sampling round is not stopped by tokens spent upstream. `sampling_token_budget` stays 6000 and `max_sampling_review_tokens` still bounds what this stage spends.

The saturation request is packed to `sampling_token_budget` the same way the selection request is packed. Oldest ledger rows go first, then the stopping report, then indicator fields. `n_documents` and the instruction stay. Omitted fields are recorded on `saturation_pack`. A request that still does not fit is not sent.

A human resume that names some directions does not reject the ones it leaves out. An unlisted direction stays open. An explicit `reject` still closes that direction.

`sampling_rounds_without_new_directions` is `not_evaluable`, and `observed` is null, while a selected pair remains unsent for `call_allowance`, and also when the ledger does not record `reread_unsent_pairs`. The criterion can be met when every selected pair was sent, or when each remaining unsent pair is in `closed_direction_keys`. That status string is a decision.

## Other checks

Memo accounting is `requested = produced + omitted + failed + not_attempted_rejected`. Rejected rows are labeled `not_attempted_rejected` and are not sent. An empty memo counts as failed, so the identity holds. `memo_empty` is the stage reason only when nothing was produced.

The axial token plan estimates the pairs in this round. `pool_estimate` is the raw unordered pool and is labeled as not the cap comparison. A raw focused estimate above the ceiling is not described as exceeding the cap when the guarded figure is at or below it. Both figures are named.

`axial_synthetic_counterexample` with no relationship records `synthetic_counterexample.status` `not_run` and reason `no_relationship_to_test`. A synthetic check that already has a status is not overwritten.

`export_gate_relationships` lists rows that are neither accepted nor withheld under `proposed`. The scorer still scores only accepted and withheld rows. A rate with no adjudicated rows stays null.

## Timeouts

The focused step watchdog is `request_timeout` times the sequential calls in that step, plus two seconds. For `recode_worker` that is two calls, so a 120 second limit allows about 242 seconds from the start of the step. After a call finishes, the remainder can be about 201 seconds, which is longer than the stated per-call limit. Each call is now also bounded: the limiter checks `step_deadline` before the call and before a retry, and the worker waits at most `request_timeout + 1` seconds. `timeout_elapsed_seconds` is the wait after the last successful call, with `timeout_limit_seconds` and `timeout_overrun_note` when that wait is longer than the limit. A provider read that ignores cancellation can still occupy the rest of the step watchdog. That residual is what the note reports. Bounding the call and reporting the overrun is a decision. Human timeout resume is still `ainvoke(None)`.
