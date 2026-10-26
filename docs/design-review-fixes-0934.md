# 0.9.34 review fixes

Status: record of package 0.9.34. Package 0.9.37 is current. The 0.9.33 note stays the record of packed reread requests, the one-request core comparison, recorded conditions on the presented statement, per-step allowance labels, and case-identity source. This note does not change `pattern_min_cases` (2), the axial token number (24000 per step), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added. No LLM judge is added.

A decision review that reports every item reviewed, including a count such as 355 of 355, establishes processing coverage. A passing synthetic fixture does the same. Neither establishes that a selection is valid. Human-adjudicated examples are not in this package.

## Decisions

Core selection runs a deterministic tournament. That replaces the 0.9.33 stop `core_candidates_do_not_fit`, which withheld the whole eligible set when it did not fit one request. Candidates are ordered by code id. If the whole set fits the per-request budget, one request compares them, including a set of one. Otherwise the round is packed into matches of at least two. A leftover single is a bye and is not sent as its own request. Winners and byes advance until one request compares the finalists. The export records `core_bracket`, `core_candidates_compared_by_round`, and `core_candidates_eligible`. `core_candidates_compared` is the sum of seats compared across rounds, so a candidate who plays more than once is counted again. The bracket stays inside the core step's own call and token allowance. If the next pair does not fit the per-request budget, the stop is `core_pair_exceeds_request_budget`, nothing is sent, and no core is chosen. If a match has already been sent and the next match does not fit the remaining allowance, the stop is `core_bracket_incomplete` and no core is chosen. A token or call refusal before any match is sent stays `token_allowance` or `call_allowance`. The rule is `core_comparison_rule` `deterministic_tournament_v1`.

An evidence assessment that never carries review fields is not treated as an unreviewed axial row. Axial rows are created with `reviewer_decided` false, and those rows are not eligible until a disposition is recorded. That reading of "no review decision" is a decision.

`one_document_one_respondent` stays off unless the caller sets it.

## What the code does

A condition span starts at the marker and continues until `.`, `?`, `!`, `;`, or a comma whose preceding words are more than a connective and one pronoun. `if you, violated the data sharing agreement` keeps the clause after that first comma. `if it rains, the crew stops` still ends at the comma. The pronouns are `i`, `you`, `he`, `she`, `it`, `we`, `they`, `me`, `him`, `her`, `us`, `them`, and `one`. `if you` and `unless you` are `condition_incomplete`. `if you help` and `if it rains` are complete. An incomplete span is stored on `condition` so it can be read, and it is not appended to `presented_statement`. `statement_omits_condition` and `condition_retained` stay null. The row is not qualified and cannot authorize tier 3. The rule is `condition_completeness_rule` `full_clause_or_incomplete_v1`.

`condition_not_in_statement`, `condition_incomplete`, and `no_review_decision` are claim flags. A later evidence pass drops them and writes each one at most once.

`incomplete_condition`, `hypothetical_link`, and `no_review_decision` are withholding causes. A row can carry more than one. Any one of them sets `downstream_eligible` false and clears pattern support for a core. Endpoint, denial, wording, and case-count causes still come first when they also apply. The rule is `eligibility_decision_rule` `reviewed_complete_condition_v1`.

`focused_not_reread` includes passages withheld as `reread_request_exceeds_token_budget`. Coverage also lists them as `withheld_over_budget`. `queue_complete` is false while any remain, including when a later visit would otherwise have marked the queue complete. The rule is `reread_coverage_rule` `withheld_not_complete_v1`.

The pre-selection reread figure `focused_review_tokens_upper_bound` is `token_budget` times `max_focused_review_calls`. The label is `per-request cap times reserved reread calls`. `focused_review_upper_bound_is_a_bound` is true. The bound is recorded even when the worst-case preflight does not apply. It does not replace the estimate taken after selection, and it does not block a selection. The rule is `reread_bound_rule` `cap_times_reserved_calls_v1`.

`_axial_token_plan` still records `estimate`, `ceiling`, `shortfall`, and `pool_estimate_label`. `warning` is empty. The old sentence that named 48000 against 24000 is not emitted. Coding calls remain capped when they are sent. Relation review, saturation review, and core selection each keep an allowance of 24000, and they are not part of that planning estimate.

The resume fingerprint stays `request_budget_v6`. The new procedure keys are `condition_completeness_rule`, `eligibility_decision_rule`, `reread_coverage_rule`, and `reread_bound_rule`. `core_comparison_rule` changed value. A 0.9.33 checkpoint cannot resume.
