# 0.9.28 review fixes

Status: historical record of package 0.9.28. The current review note is [design-review-fixes-0929.md](design-review-fixes-0929.md). 0.9.29 keeps counterexamples and qualifications on a merged assertion, preserves ambiguous conditions, and budgets cross-chunk consolidation. The 0.9.27 note stays the record of the focused export, unreviewed endpoints, the first withholding partition, link-governing `if`, the recorded reviewer decision, and the chunk-allowance stop. This note does not change `pattern_min_cases` (2), the axial token ceiling (24000), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`.

## Accepted is not the same as downstream-eligible

`export_gate_relationships` still buckets by disposition. `accepted` means `accepted` or `verified`. `accepted_downstream` is the accepted rows whose `downstream_eligible` is true. A tier-2 row can be ineligible, including `endpoint_unreviewed_by_focus`, and it stays in `accepted` with that tier. It is not an accepted downstream result. Each row exports `evidence_tier`, `reviewer_disposition`, `endpoint_review_status`, and `downstream_eligible`. A missing eligibility is null. `reviewer_disposition` is copied from the row and is not invented from `disposition`.

An explained row is disposition-withheld, `downstream_eligible` false, unreviewed at an endpoint, or still proposed with a known case or tier shortfall. A proposed row with zero supporting cases, or a tier below 2, is explained even when `downstream_eligible` was not stamped false and no reviewer decision was recorded. The cause is `insufficient_cases` unless an earlier cause applies. An unreviewed endpoint is `endpoint_unreviewed_by_focus` in any disposition bucket, including `proposed`. A proposed row that still has enough cases and unknown eligibility is not explained. Cause counts sum to those explained rows, so they match the ineligible rows. Every explained row has at least one withholding reason. A missing reviewer decision is not recorded as `llm_reviewer_rejected`.

## Decision-call allowances

Focused selection defaults to 8 decision calls. Axial and selective review default to 4. Those numbers are unchanged.

Relationship review, saturation review, and selective core selection used to share one `stages.review` pool. A completed 0.9.27 run stopped core selection on `call_allowance` after 4 of 8 decision calls. The other 4 were not retries, a reservation, or a double charge. Decision review sets `max_retries` to 0 and does not schema-repair, so a repair is not charged. The missing calls were earlier steps on that same pool: relationship-review chunks and the saturation review, carried into selective coding by `decision_calls_used`. The core report compared that cumulative total with `max_decision_calls`.

Each of those steps now has its own ceiling of `max_decision_calls`. Earlier calls are `prior_step_calls`. `prior_step_calls_reduce_allowance` is false, so they do not shrink the later step. `starting_usage` is that step's own calls already made. `calls_made` is what this step sent. `remaining_allowance` is the ceiling minus those calls. `stop_reason` is why the step stopped, or null when it finished. `allowance_scope` stays `shared_decision_review`, which names the family. It is not one pool. `decision_calls_used` is the sum of the three steps. The synthetic `provider_budget.stages.review` row has `max_calls` null and is not a ceiling. A combined axial-then-selective run can therefore spend one allowance on each step. The defaults are not raised. A caller that still builds a single `review` stage keeps that older bucket.

`decision_budget_steps` records, for each budgeted step, `allowance_scope`, `shared_with`, `allowance`, `starting_usage`, `calls_made`, `remaining_allowance`, `stop_reason`, `prior_step_calls`, and `prior_step_calls_reduce_allowance`. Focused selection and claim review share `focused_decision_review` instead. Remaining allowance is null when the allowance is unknown.

## Overlapping reasons and one primary cause

`withholding_reasons` keeps every applicable diagnostic. `withholding_reason_counts` tallies that overlap and does not sum to the explained rows.

`primary_withholding_cause` is a reporting convention. The first applicable name wins, so `withholding_cause_counts` is mutually exclusive and sums to the explained rows:

1. `endpoint_unreviewed_by_focus`
2. `endpoint_not_selected_by_focus`
3. `participant_denies_link`
4. `partial_endpoint_evidence`
5. `wording_withheld`
6. `insufficient_cases`
7. `omitted_oversized`
8. `llm_reviewer_rejected`
9. `reviewer_policy_rejection`
10. `other_withheld`

A reviewer rejection of evidence below tier 2 is `reviewer_policy_rejection`. It supplies no independent assessment of correctness, and it is not also `llm_reviewer_rejected`. Tier 2 or higher that the reviewer rejects is `llm_reviewer_rejected`. A substantive limit earlier in the list remains the primary cause, and the reviewer reason stays on the overlapping list. Inspect flags such as `shared_quote_inspect` are not causes. The gate export and the run files (`axial_run.json`, `selective_run.json`, `category_relationships.json`) read one `relationship_assessment` object.

## Discourse asides

`if` is hypothetical only when it governs the from-to link. A discourse aside is a category: a parenthetical or comma-bounded clause whose subject is a discourse pronoun (`I`, `we`, `you`, `that`, `this`, `it`) and whose other words stay in the discourse class, and which does not contain a link connector. `if that makes sense` and `if I can` are asides. `if the night is long` is not. Stripping an aside does not remove a lexical conditional elsewhere in the sentence. `would` and reported speech stay hypothetical. Readings stay `deterministic_provisional`. The synthetic fixture is `tests/fixtures/hypothetical_if_0927.json` plus the paired 0.9.28 cases. They are not study data. The rule is `discourse_aside_v1`.

A live run with no denied links and no tier 3 does not show that the denial or tier-3 detectors work. That needs known denial, qualified, and affirmative controls, plus adjudicated examples.

## Duplicate assertions

The same directed pair and the same normalized statement are one relationship. The lowest relationship id is kept. `assertion_provenance` and `merged_relationship_ids` keep the proposal ids, the pre-merge tier, the disposition, the reviewer disposition, and the passage. Supporting `case_quotes` are unioned. The merged evidence is passed through the normal tier logic. The highest input tier is not kept. `conflicting_evidence` is that provenance when the pre-merge tiers or dispositions differ, and null otherwise. An empty statement is not merged. The opposite direction and a different statement stay distinct. The rule is `merge_identical_statement_v1`.

## Repeatability

Two experiments stay distinct. `repeat_frozen_review` calls the reviewer `N` times on one payload. `repeat_pipeline` calls a pipeline callable `N` times, so upstream differences propagate. Neither helper places a paid call. The caller supplies the fake or the callable.

`compare_pipeline_runs` reports pairwise and summary Jaccard of `raised_code_ids`, Jaccard of directed pairs, Jaccard of directed pair plus statement hash, and eligibility changes. A comparison is null when it is undefined: an empty union, a missing id list, or a relationship whose `downstream_eligible` is not a boolean.

Exports record `model_configuration`: model, temperature and the other sampling settings the caller configured, seed, and prompt hashes. Temperature stays null unless the caller set it. Temperature 0 is configuration control, not a reproducibility guarantee.

## Resume

The fingerprint is `request_budget_v5`. It also requires `duplicate_assertion_rule` (`merge_identical_statement_v1`), `budget_ledger_rule` (`shared_review_ledger_v1`), `export_assessment_rule` (`shared_assessment_v1`), and `chunk_review_recovery` (`reuse_validated_v1`), plus the updated withholding and conditional rules. A missing key fails. A 0.9.27 `request_budget_v4` checkpoint raises `DecisionChunkingCheckpointError` before `allow_changes` is considered. The plan hash still does not include the call or token ceilings. A budget-only change of those ceilings can continue, including a run that already finished on an allowance stop. A finished complete stage still cannot. A changed partition cannot.

## Raised focused categories

A complete applied focused selection limits axial proposals and downstream eligibility to the raised codes. `axial_pair_pool` is `raised_focused_categories`. Codes that were reviewed and not raised are `axial_pair_pool_excluded_code_ids`. A relationship whose endpoint was reviewed and not raised is `hypothesis_exploration` with reason `endpoint_not_selected_by_focus`, and `downstream_eligible` is false. An endpoint that was never reviewed is still `endpoint_unreviewed_by_focus`, and that cause wins. An incomplete or withheld selection does not use the raised list. Its raised ids are empty because the review was not applied, and a reviewed endpoint on that selection stays eligible under the unreviewed rule. A run with no focused selection is not restricted.

## Token allowance before the first request

`auto_size_decision_calls` raises only the call allowance, and only when the caller sets it. It does not raise `max_decision_tokens`. When it is on and the configured allowance is at least one, that configured count is reserved for claim review. Selection is sized to its own chunks, so it cannot spend the claim review's calls. `claim_review_calls_reserved` is that reserve. `calls_available` is the selection portion. The reviewer's call ceiling is the selection chunks plus the reserve.

Before any focused-selection request, the run estimates the chunks that still have to be sent. `calls_required` is that count. `tokens_required` is the sum of their input estimates and `tokens_available` is the remaining token allowance. Each estimate includes the fixed system prompt, the chunk payload, and the response schema. It does not include output tokens. It does not include a retry or repair margin, because decision review does not retry or schema-repair. The export records those three facts as `token_estimate_includes_output_tokens`, `token_estimate_includes_retry_or_repair_margin`, and `token_estimate_includes_fixed_prompt_overhead`. If either remaining allowance is short, the stop is `chunk_allowance_insufficient` and nothing further is sent. `auto_size_covers_calls_only` is true when auto-sizing is on. `auto_size_raised_tokens` stays false. `preflight_can_finish` is the only statement that the plan can finish. A raised call allowance does not imply that. A zero call allowance stays `call_allowance`. The per-request budget is unchanged.

A positive `max_focused_review_calls` and a positive `max_focused_review_tokens` are estimated in that same preflight. The estimate plans the reread as if every codebook code were raised and sums the input tokens of up to `max_focused_review_calls` batches. If that required total exceeds the available tokens, the stop is still `chunk_allowance_insufficient` and nothing is sent. The export records `focused_review_tokens_required`, `focused_review_tokens_available`, and `focused_review_token_shortfall`. Zero on either focused ceiling still runs selection. The separate runtime warning for a focused-review token ceiling, including a configured 48000, is this check.

The estimate counts input only. On a completed focused review it ran about 16% above provider-reported input tokens. That gap is not a dollar bound. The estimate is not rescaled, and there is no dollar cap. `token_estimate_is_dollar_bound` is false.

## Finished stages can continue on a larger budget

A run that already exported, and whose selection, claim review, relationship review, saturation review, or core selection stopped on an allowance, can resume when the only change is a larger budget. The graph has no open interrupt. The continuation re-enters the stage that stopped, clears the allowance stop, and reuses validated chunks. `selection_withheld` and `core_selection_withheld` stay clear until that visit closes again, so the route does not export after one more chunk. A finished complete stage still raises the not-waiting error. A changed partition, seed, or source still fails. Focused budget-only fields are `max_decision_calls`, `max_decision_tokens`, `max_focused_review_calls`, and `max_focused_review_tokens`. Axial budget-only fields are `max_decision_calls`, `max_axial_review_tokens`, and `max_selective_review_calls`.

## Validated chunks can be resumed

Each validated selection chunk is stored with `input_fingerprint` (code ids, definitions, and quotes), `prompt_hash`, and `policy_fingerprint`. The policy fingerprint covers the review-procedure rules, the chunk order and its seed, and the system prompt. It does not cover the call or token ceilings. A resume whose manifest differs only in `max_decision_calls` or `max_decision_tokens` is a budget resume and does not need `allow_changes`. Matching chunks are reused and are not sent again. A changed definition, quote, prompt, or policy invalidates only the chunks whose fingerprint no longer matches. Appending one recovered oversized code does not re-send the chunks already validated. A changed partition still raises `DecisionChunkingCheckpointError`. The rule is `reuse_validated_v1`.

## Chunk order and chunk context

`selection_chunk_order` defaults to `id`, which is the order used before this setting existed. `seeded_shuffle` reorders the codes with `selection_chunk_order_seed` before they are packed, and the seed is stored. Shuffling removes the code-id preference. It does not make a partial selection complete: the export says the reviewed codes are the prefix of the shuffled order, `partial_selection_is_code_id_slice` is false, and the selection is still withheld. `chunk_order_selects_codes` stays false. Applied ids from a complete review stay in code-id order.

`measure_chunk_context_sensitivity` sends one code with each companion group the caller supplies and reports `selection_changed` when the code is raised in some groups and not in others. The caller supplies the reviewer. The helper does not choose a model and does not make a paid call.

A chunk only sees its own codes. A category that becomes significant only across the corpus can be missed. `selection_consolidation` is an optional pass, after the chunks, over the provisionally raised codes and the near misses (reviewed, not raised). It is off by default. Turning it on is a decision, not an analytic default. A pass that does not fit the remaining allowance, or that does not return ids from that set, is skipped and the chunk reviews are kept.

## Partial selection is a code-id slice

With the default order, focused selection reviews chunks in `code_id` order. When that selection is partial, `partial_selection_is_code_id_slice` is true and `partial_selection_bias` says the reviewed codes are the prefix of that order, not a sample. A complete selection leaves the note null. `chunk_order_selects_codes` stays false: an incomplete review is withheld and is not applied.
