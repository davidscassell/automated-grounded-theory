# 0.9.27 review fixes

Status: historical record of package 0.9.27. The current review note is [design-review-fixes-0928.md](design-review-fixes-0928.md). The 0.9.26 note stays the record of shared spans, frequency qualification, and oversized recovery. This note does not change `pattern_min_cases` (2), the axial token ceiling (24000), `batch_size` (1), or the 6000 request budgets. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`.

## Raised ids on the focused export

`focused_run.json` and `focused_selection.json` both carry `raised_code_ids`, `not_raised_code_ids`, `provisional_raised_code_ids`, `selection_review_complete`, `selection_status`, `selection_withheld`, `selection_source`, `selection_reviewed_code_ids`, and `selection_uncovered_code_ids`. The same fields are copied onto the run metadata. The codebook and passages are written so a later stage can score the categories.

`axial_code` and `selective_code` accept that directory (or either of those two files). `load_focused_export` is the same read. A complete selection limits core candidates to the raised ids. An incomplete or withheld selection is `focused_selection_incomplete` and scores none. Provisional ids are stored and are not promoted into the candidate list.

## Unreviewed endpoints

When `selection_review_complete` is false, `selection_withheld` is true, or `selection_status` is `provisional`, a relationship is exploratory if either endpoint is absent from `selection_reviewed_code_ids`. The evidence use is `hypothesis_exploration`, `downstream_eligible` is false, and the reason is `endpoint_unreviewed_by_focus`. Quotes that would otherwise support tier 2 do not override that. A pair whose endpoints were both reviewed keeps the existing gates. A run that has no focused selection does not treat every code as unreviewed.

The rule is `unreviewed_endpoint_v1`.

## One withholding cause

`withheld_reason_counts` tallies withholding flags and rule names. It does not tally inspect flags, and it is not a partition. `shared_quote_inspect` can sit on an eligible row. `inspect_flag_counts` is that inspect tally, including on eligible rows.

`withholding_cause` is one primary cause on each withheld row. `withholding_cause_counts` counts those causes. The causes are `endpoint_unreviewed_by_focus`, `llm_reviewer_rejected`, `participant_denies_link`, `partial_endpoint_evidence`, `wording_withheld`, `insufficient_cases`, `omitted_oversized`, and `other_withheld` for a withheld row that matches none of the others. The counts sum to the number of withheld rows, including zeros for causes that did not occur. All three maps are null when the export has no rows. The rule is `primary_cause_v1`.

## Funnel and tiers

`attrition.grounded_endpoints` counts rows whose `evidence_tier` is 2 or higher. That is the same set `evidence_tiers` counts at those tiers, so a tier-2 row is not invisible to the grounded stage. `endpoint_anchoring_complete` counts rows whose endpoint-anchoring list has two anchored ends. That list is a different check. It is not the tier-2 count.

## Conditional `if`

`if` marks a quote hypothetical only when it governs the from-to link. `part of why ... if ...` is a noun phrase and is not hypothetical. `if I can` is an aside and is not. `if tools lead to coworkers`, and `tools lead to coworkers if the night is long`, stay hypothetical. `would` and reported speech stay hypothetical markers. Every deterministic reading keeps `reading_basis` `deterministic_provisional` and `reading_provisional` true. The synthetic fixture is `tests/fixtures/hypothetical_if_0927.json`. It is not study data. The rule is `link_governing_if_v1`.

## Reviewer variance

The LLM reviewer is the deciding step for a row it sees. The row stores `reviewer_decided` true, `reviewer_disposition` (the reviewer's own choice), `model_disposition` (that choice after the evidence gate), and `model_rationale`. A row the reviewer did not decide keeps `reviewer_decided` false and a null rationale. Reviewer payloads dump object keys in sorted order so the request text does not depend on dict insertion order. Two runs on near-identical input can still differ, because the model is that deciding step. This package does not add a second paid judge. Temperature 0 does not remove that variance.

## Chunk allowance

Before the first focused-selection request, the run counts the chunks the plan needs and compares that with `max_decision_calls`. The report stores `chunks_required`, `decision_call_allowance`, and `chunk_shortfall`. If the allowance is positive and smaller than the chunk count, the run does not send. `decision_stop` is `chunk_allowance_insufficient`. The selection stays provisional. With interrupts on, resume can continue after the allowance is raised. An allowance of zero stays the existing `call_allowance` stop: the reviewer is off.

`auto_size_decision_calls` defaults to false. It is a decision, stored on the procedure, not an analytic default. When true, and the configured allowance is at least one, the call allowance is raised to `chunks_required`. `max_decision_tokens` and `decision_token_budget` still refuse a chunk that does not fit. There is no separate dollar ceiling on this reviewer. The default of 8 calls is unchanged.

The resume fingerprint is `request_budget_v4`. It also requires `pattern_quote_rule`, `link_denial_rule`, `oversized_recovery`, `focus_endpoint_rule`, `withholding_cause_rule`, `conditional_scope_rule`, and `chunk_allowance_rule`. A missing value, or a 0.9.26 `request_budget_v3` checkpoint, raises `DecisionChunkingCheckpointError` before `allow_changes` is considered. The plan hash still does not include the call or token ceilings.

## Left open

The axial planner still builds the full unordered pair space in memory. That limit is unchanged.
