# 0.9.28 review fixes

Status: current for package 0.9.28. The 0.9.27 note stays the record of the focused export, unreviewed endpoints, the first withholding partition, link-governing `if`, the recorded reviewer decision, and the chunk-allowance stop. This note does not change `pattern_min_cases` (2), the axial token ceiling (24000), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`.

## Accepted is not the same as downstream-eligible

`export_gate_relationships` still buckets by disposition. `accepted` means `accepted` or `verified`. `accepted_downstream` is the accepted rows whose `downstream_eligible` is true. A tier-2 row can be ineligible, including `endpoint_unreviewed_by_focus`, and it stays in `accepted` with that tier. It is not an accepted downstream result. Each row exports `evidence_tier`, `reviewer_disposition`, `endpoint_review_status`, and `downstream_eligible`. A missing eligibility is null. `reviewer_disposition` is copied from the row and is not invented from `disposition`.

An explained row is disposition-withheld, `downstream_eligible` false, unreviewed at an endpoint, or still proposed with a known case or tier shortfall. A proposed row with zero supporting cases, or a tier below 2, is explained even when `downstream_eligible` was not stamped false and no reviewer decision was recorded. The cause is `insufficient_cases` unless an earlier cause applies. An unreviewed endpoint is `endpoint_unreviewed_by_focus` in any disposition bucket, including `proposed`. A proposed row that still has enough cases and unknown eligibility is not explained. Cause counts sum to those explained rows, so they match the ineligible rows. Every explained row has at least one withholding reason. A missing reviewer decision is not recorded as `llm_reviewer_rejected`.

## Shared decision-call allowance

Focused selection defaults to 8 decision calls. Axial and selective review default to 4. Those numbers are unchanged.

On a combined axial-then-selective run the review stage is one pool. Relationship review, the saturation review, and selective core selection all spend `stages.review`. The decision reviewer does not retry (`max_retries` is 0) and does not schema-repair, so a repair is not charged to that stage. `decision_calls_used` is aligned with `stage_used["review"]`. It is not added on top of it. A core report that shows four calls and `call_allowance` against an allowance of 8 can be the remaining half of a pool that already had starting usage 4. Stopping is then correct. The misleading part was the report.

`decision_budget_steps` records, for each budgeted step, `allowance_scope`, `shared_with`, `allowance`, `starting_usage`, `calls_made`, `remaining_allowance`, and `stop_reason`. `shared_decision_review` covers `axial_relation_review`, `axial_saturation_review`, and `axial_core_review`. Focused selection and claim review share `focused_decision_review` instead. Remaining allowance is null when the allowance is unknown.

## Overlapping reasons and one primary cause

`withholding_reasons` keeps every applicable diagnostic. `withholding_reason_counts` tallies that overlap and does not sum to the explained rows.

`primary_withholding_cause` is a reporting convention. The first applicable name wins, so `withholding_cause_counts` is mutually exclusive and sums to the explained rows:

1. `endpoint_unreviewed_by_focus`
2. `participant_denies_link`
3. `partial_endpoint_evidence`
4. `wording_withheld`
5. `insufficient_cases`
6. `omitted_oversized`
7. `llm_reviewer_rejected`
8. `reviewer_policy_rejection`
9. `other_withheld`

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

The fingerprint is `request_budget_v5`. It also requires `duplicate_assertion_rule` (`merge_identical_statement_v1`), `budget_ledger_rule` (`shared_review_ledger_v1`), and `export_assessment_rule` (`shared_assessment_v1`), plus the updated withholding and conditional rules. A missing key fails. A 0.9.27 `request_budget_v4` checkpoint raises `DecisionChunkingCheckpointError` before `allow_changes` is considered. The plan hash still does not include the call or token ceilings.

## Token allowance before the first request

`auto_size_decision_calls` raises only the call allowance, and only when the caller sets it. It does not raise `max_decision_tokens`. Before the first focused-selection request, the run sums the packed chunk estimates as `tokens_required` and compares that with the decision-token allowance, alongside `chunks_required` and the call allowance. If either allowance cannot cover the plan, the stop is `chunk_allowance_insufficient`, nothing is sent, and the report names `tokens_required`, `decision_token_allowance`, and `token_shortfall`. A zero call allowance stays `call_allowance`. The per-request budget is unchanged.

## Partial selection is a code-id slice

Focused selection reviews chunks in `code_id` order. When the selection is partial, `partial_selection_is_code_id_slice` is true and `partial_selection_bias` says the reviewed codes are the prefix of that order, not a sample. A complete selection leaves the note null. `chunk_order_selects_codes` stays false: an incomplete review is withheld and is not applied.
