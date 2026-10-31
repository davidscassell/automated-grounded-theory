# 0.9.36 review fixes

Status: record of package 0.9.36. Package 0.9.41 is current. The 0.9.35 note stays the record of the final-decision rule, the eligible-set prompt, observed conditional patterns, and capped candidate summaries. This note does not change `pattern_min_cases` (2), the axial token number (24000 per step), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added. No LLM judge is added. No live evaluation was run.

A decision review that reports every item reviewed, including a count such as 355 of 355, establishes processing coverage. A passing synthetic fixture does the same. Neither establishes that a selection is valid. Human-adjudicated examples are not in this package.

## Decisions

Continuing the core tournament after an opening abstention is a decision. In a live run, one abstention in the first opening match examined 12 of 55 candidates, ended core selection, and reported the selection complete. Examining 12 of 55 candidates means selection stopped early. It does not mean that no core exists.

The model names one of two abstention reasons. `no_suitable_candidate` advances nobody from that match, and the other matches continue. `insufficient_evidence` marks the match unresolved. The same match is retried once with a richer evidence summary when the step's call allowance and token allowance can hold that retry. Otherwise the match stays unresolved. A call failure, a parse failure, or an exhausted budget marks the match incomplete. Those stops are not abstentions.

Winners advance to a final. A sole remaining candidate still receives a final assessment against the core-category criteria. Being the only candidate left does not make it the core.

The outcome is one of three. `core_selected` means the final assessment supports a candidate and the bracket finished. `no_core_supported` means the completed assessments found no suitable core, for example when every opening match abstained with `no_suitable_candidate`. `selection_unresolved` means missing evidence, an unfinished match, or the budget prevents a conclusion, for example when every match lacked enough evidence. A mixed case with any unresolved or incomplete match that could change the result is unresolved. `core_selection_complete` is false whenever candidates remain uncovered. `core_candidates_compared` counts the seats in matches that were sent, including an abstention. The rules are `core_match_abstention_rule` `continue_after_opening_abstention_v1`, `core_outcome_rule` `selected_unsupported_or_unresolved_v1`, and `core_final_decision_rule` `final_assessment_only_v1`.

The prompt names three different lists: the original eligible set, the candidates in this match, and the finalists. It says how many candidates are in this match is not a reason to abstain. It does not say the request is a partial scope. `contains_every_eligible_candidate` stays on the comparison-scope record. A deterministic code-id bracket is still reproducible scheduling. `core_bracket_is_substantive_stability` stays false. The rule is `core_prompt_scope_rule` `eligible_match_finalists_no_abstain_cue_v1`.

Evidence summaries carry a definition, a representative statement, and counterevidence when the source has them. Scoring keeps the configured cap, default 80 tokens, and the original source text. The request refits that summary to the match. The cap uses the match count and a quarter of the smaller of the per-request budget and the per-step allowance. It does not use the size of the whole eligible set. The insufficient-evidence retry uses half of that allowance, at least the configured cap and at most four times that cap. `core_evidence_extra_tokens_per_candidate` reports the fitted size. `core_evidence_improves_selection` stays null. Whether the summaries improve selections is unmeasured. The rule is `core_evidence_summary_rule` `match_budget_candidate_evidence_v1`.

Relation review keeps `max_decision_calls` at 4. The limit is not raised. That is a decision. When the planned requests exceed that limit, the preflight records `relation_review_call_warning` before it spends. Rows that were not sent are `relation_review_status` `unreviewed` with `relation_unreviewed_reason` `call_allowance`, unless a row was omitted for a different recorded reason. The rule is `relation_unreviewed_rule` `report_call_limit_v1`.

`withheld_over_budget` on the focused status, on `FocusedResult`, and at the top of `focused_run.json` is the same list as `coverage.withheld_over_budget`. An empty withhold is an empty list. The rule is `withheld_export_rule` `coverage_matches_top_level_v1`.

`compare_pipeline_runs` records `raised_code_counts`, the minimum, the maximum, the range, `raised_code_counts_vary`, and the ids that are not in every run. Pairwise comparisons record the count difference and the ids only on one side. A Jaccard near 1 does not hide a change from 57 ids to 55. The fields are null when a run has no `raised_code_ids`. The rule is `raised_variance_rule` `record_run_to_run_counts_v1`.

## What this note does not change

`pattern_min_cases` stays 2. `max_axial_review_tokens` stays 24000 per step. `axial_token_budget` and `decision_token_budget` stay 6000. `batch_size` stays 1. `selection_consolidation` stays off. `auto_size_decision_calls` stays off. Tier 4 stays a human `verify`. No LLM judge is added. A metric that was not computed stays null. The conditional-pattern eligibility from 0.9.35 is unchanged. The fixtures are synthetic. This package creates no adjudications.

The resume fingerprint stays `request_budget_v6`. The new procedure keys are part of that fingerprint, so a 0.9.35 checkpoint cannot resume.
