# Review fixes (0.9.7)

These notes record the v30-0906 checks. They do not add a graph node. No new source is cited. `pattern_min_cases` stays 2.

## Wording and the current core

`wording_overreach` treats consequence language, and the roles `consequence` and `cause`, as tier-3 wording. A reviewer rationale that says no causal language was used is checked against that detector. The flag is `reviewer_causal_claim_mismatch`.

`core_stability.n` is the number of named cores in `cores`. The current run is one of those cores when it names one. `runs` is the number of records. An unnamed record does not increase `n`. Disagreeing cores still score 0.

## Anchors and contradictions

An endpoint needs two distinctive stems, or an embedding at the evidence threshold, before it is anchored. One distinctive stem is `stem_only`, flagged `weak_stem_anchor`, and it does not count toward `cases_with_both_endpoints` or `same_passage`. A generic stem such as `work` stays `generic_stem_only`.

A `contradicts` judgment on an absence candidate (`a_without_b`, `b_without_a`, `category_without_outcome`, `outcome_without_category`) is stored as `qualifies`. The original judgment stays on `model_judgment`. The flag is `absence_is_not_contradiction`. The row is not marked contradicted or disputed. `opposite_pole` and `contrasting_outcome` can still contradict.

## Counts

The relationship's `negative_case_search_record` is copied after attempted and remaining are updated. `remaining_candidates` is the sum of those row remainings. `prompt_accounting` splits the prompt log into provider calls and unsent rows. Research counterexample judgments exclude the synthetic check. An omitted prompt is not a judgment.

## Sampling

Sampling stage ceilings start at zero. Upstream coding and review calls are on `sampling_call_baseline`. The carried `calls_used` is unchanged.

`sampling_token_budget` is not raised. The selection request drops lowest-ranked directions until the request fits. The omitted direction ids and both estimates are on `selection_pack`.

The model may return more pairs than the coding allowance can pay for. The highest-ranked pairs are kept. The rest are `reread_unsent_pairs` with reason `call_allowance`, and `sampling_stop` is `call_allowance`. `model_flat_list_mismatch` still records a disagreeing list.

Reread notes are stored on the ledger with `direction_key`. The next round does not repeat a key closed by an irrelevant note or a reject. A note that is not irrelevant adds `reread_follow_up`. `analytic_new_direction_keys` is what the quiet-round criterion reads when that field is present. A null `window_new_property_rate`, and the other window rates, are filled from complete points. The criterion observed value is that published rate. `judgment` stays null. The package does not claim saturation.

`empty_paradigm_slot` keys include the role. Duplicate keys in one round are dropped.

## Focused claims, memos, and the ceiling

A focused claim round with plausible respondent candidates left unsent is `budget_limited`. `complete_no_negative_search` remains for a round that had nothing plausible left unsent and no relevant judgment.

A memo that repeats a prompt instruction is `memo_echoes_prompt`. A rejected relationship is listed on `memo_skipped_rejected_ids` and is not sent.

`storyline_unaccepted_relationship` on an accepted relationship is set when the sentence is causal and the tier is below 3. A non-causal citation of an accepted tier-2 row is not that flag.

`focused_tokens_estimated` stays the raw heuristic. `focused_tokens_estimated_compared` is the guarded figure the ceiling uses. `ceiling_comparison` is `guarded`.

`max_axial_review_tokens` stays 24000 unless the caller sets it. `axial_token_plan` estimates planned calls times the per-request budget, records the shortfall, and does not change the ceiling.

An irrelevant synthetic judgment, or a synthetic check that did not run, is `status` `not_interpretable` and `detected` null. `contradicts` and `qualifies` stay detected.

## Left as observed

Default axial coding found 1 relationship across 13 runs in v30-0906. v29 had 11. Tier 2 is unreachable at `pattern_min_cases` 2 on that corpus. Tiers 3 and 4 were not reached. Those settings are unchanged.
