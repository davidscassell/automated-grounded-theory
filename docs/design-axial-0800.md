# Axial coding, saturation, and later steps (0.8.0, fixes in 0.8.1, 0.8.2, 0.8.3, 0.8.4, and 0.8.5)

This note is the design for package 0.8.0, with the 0.8.1, 0.8.2, 0.8.3, 0.8.4, and 0.8.5 corrections below. It sits on the focused-coding graph from 0.7.9. The default run is still initial coding plus a theme report. Focused coding is unchanged. Axial coding, saturation indicators, and selective coding are separate graphs. [design-later-stages.md](design-later-stages.md) is the longer literature note. This file is the slice that is implemented.

`policy_version` is `0.8.5`.

## 0.8.5 corrections

**Both endpoints.** `citation_anchoring_rate` is 1 only when every relationship quote shares a content token with both categories. Otherwise it is null. `endpoint_anchoring_rate` is the share of endpoints that quote names. A passage assignment is not that evidence. The row is flagged `endpoint_evidence_partial` and kept.

**Storyline.** The request carries non-rejected relationships, their endpoint categories, the core when one is set, and the passages those rows cite. It is packed to `axial_token_budget`. Omitted relationship, category, and passage ids are on the request and on `storyline_omitted`. A statement that cites only rejected relationships is flagged `storyline_rejected_relationship`. `storyline.json` includes `core_id` and `core_provenance`. An empty or unknown core is `unset`.

**Selective calls.** `memo_calls` and `storyline_calls` split `max_selective_review_calls`. When both products are requested, one call is reserved for the storyline. Those counts are procedure, not a performance setting. Memo and storyline prompts are not the relationship-proposal prompt, and their prompt-log stage is `selective`. A memo that names a category absent from its quote is flagged `memo_quote_unsupported` and `needs_review`.

**Budget and tests.** `aselective_code` restores the axial `provider_budget`, including the coding stage, and adds selective calls to that counter. `relationships_with_negative_case_search` is judgments over all relationships. An axial supports, contradicts, or qualifies verdict whose rationale says the passage is irrelevant (`not linked`, `does not mention`, and the other irrelevance cues) is `axial_test_invalid` and is not a negative case.

**Saturation and planning.** `code_frequency_shift` is total variation. 0 is the same mix and 1 is maximal change. The reviewer prompt uses that label and does not describe 1 as stabilization. `code_frequency_stabilization` still stores the same number. A model assessment has `model_judgment` and does not set `judgment`. A zero `new_property_rate` or `new_relationship_rate` is `undefined_partial_coverage` while pairs remain. Obvious pairs rotate by `seed`. `pair_coverage` is examined pairs over the unordered pair count. An automated saturation resume (`automated: true` or `actor: "automated"`) is stored as actor `automated`.

## 0.8.4 corrections

**One proposal request, then a checkpoint.** `propose_relations` sends one provider request and returns. While pairs remain and the proposal allowance has calls left, the graph runs that node again. A timeout therefore keeps the responses already written. The attempt that timed out is written onto `provider_budget` and the prompt log before resume. Resume restores that allowance, so a ceiling of three calls cannot become five attempts.

**Memos.** `memo_accounting` counts requested, produced, omitted, failed, and empty memos. A chunk that fails to parse stays unresolved. A memo with empty or missing text is stored and flagged `memo_empty`, and the stage is `incomplete`. An empty pending list is not treated as completion.

**Usage.** A received response is counted once, including a response that fails to parse. A repair that cannot fit the request budget is `not_sent`. It does not copy the original response's usage, and the warning says the request was not sent. Decision review counts a response whose status is not `validated`.

**Evidence for both categories.** A pair is deferred when either endpoint has no passage in the request, even if the other category has evidence. The deferred row records `unsupported_code_ids`.

## 0.8.3 corrections

**Checkpoints do not reset the budget.** Each provider attempt writes `provider_budget` onto the graph state: shared tokens, and calls used per stage. Resume restores that snapshot into the new runtime before the next node runs. The same snapshot is how focused and selective runs continue, including a resume whose checkpointer was loaded in a new process. A pause and an uninterrupted run of the same procedure keep the same ceiling.

**Separate call ceilings.** `max_axial_review_calls` limits coding (proposals, counterexamples, and their repairs). `max_decision_calls` limits decision review. `max_selective_review_calls` limits memos and the storyline. `max_axial_review_tokens` remains the one shared token ceiling. A repair uses the originating stage, so LLM review does not fund an extra coding call.

**Stage status.** `stage_outcomes` records proposals, counterexamples, decision review, selective core, memos, and storyline. `complete` means every requested stage finished. A ceiling stop is `budget_limited`. A preserved parse failure, including a storyline whose repair also failed, is `incomplete`. A selective failure after successful proposals stays on the later stage. The run JSON and `theory_coverage.json` include the per-stage status.

**Usage and empty evidence.** Each provider response updates `provider_token_usage` once, whether or not it parses. A failed repair keeps `repair_raw_response` and `repair_usage`. A pair whose categories have no segment assignments is deferred with `missing_category_evidence` and recorded in coverage. Packing does not call `max` on an empty passage list.

## 0.8.2 corrections

**Irrelevance.** A contradicts or qualifies verdict stays a negative case when the rationale is a paraphrase. "is about" and "about the" are not irrelevance cues. Missing subject-stem overlap is `claim_subject_uncertain` and still disputes. Only explicit language (`irrelevant`, `unrelated`, `does not address`, `does not mention`, `nothing to do with`) flags `rationale_indicates_irrelevance`, and a negation such as "not unrelated" cancels that cue. The flagged row is `disputed_pending_review`. It is not dropped and it is not `claim_test_invalid`.

**One token allowance.** `max_axial_review_tokens` is one estimated-input ceiling for coding, counterexamples, memos, the storyline, and decision review. A selective run continues from `axial_tokens_estimated`. `max_selective_review_calls` stays its own call ceiling. Focused coding still uses `max_focused_review_tokens` and does not share it with `max_decision_tokens`. Every provider request, including a schema repair and a retry, acquires the call ceiling and the token ceiling before it is sent. A repair that does not fit is its own audit row. The failed output stays, flagged, for review.

**Packing.** A batch that does not fit is shrunk to the longest prefix that fits the request budget and the shared allowance. Items left out stay queued. Proposal trimming does not remove the last passage for either category of a pair. If both sides still do not fit, the pair is deferred with a recorded reason, or the passages are bounded excerpts with `source_offset` and `source_length`. Batch size does not silently drop one side of the evidence.

**Coverage.** Graph completion is not analytical coverage. After the requests run, pairs are `planned`, `attempted`, `completed`, `omitted`, or `remaining`. `unexamined_pairs` lists pairs that were not sent. A token or call ceiling yields `budget_limited`. A preserved parse failure yields `incomplete`. `complete` means every planned pair was examined. Those counts are on `theory_coverage.json`, `axial_run.json`, and `category_relationships.json`.

**Saturation figure.** Total variation between consecutive documents describes the difference between those documents. A low figure does not establish theoretical saturation and is not a human confirmation. `judgment` stays null.

## 0.8.1 corrections

**Frequency.** `code_frequency_stabilization` is total variation between consecutive documents' normalized per-document assignment counts (`code_frequency_comparison` is `total_variation`). It is undefined with fewer than two documents or when a compared document has no assignments. 0.8.0 stored a presence bit for every code seen so far. Those reports are not comparable. New-code rate is still discovery, separate from this shift.

**Combined run.** `also_selective=True` copies the axial prompt log, warnings, saturation, coverage, and procedure onto the selective state (`axial_procedure`, `stage` `selective`). Selective counters start at zero. `export` of the selective stage writes `selective_run.json`, core, memos, and storyline. It does not overwrite `category_relationships.json`, `thematic_saturation.json`, `theory_coverage.json`, or `axial_run.json`.

**Calls, not pairs.** `propose_calls` is `min(allocated_calls, ceil(pair_count / batch_size))`. Unused allocated calls are added to `counterexample_calls`. Pair capacity for that round is `counterexample_calls * batch_size`. Neither allowance is negative. `max_decision_calls=0` does not build a reviewer and does not interrupt; `decision_stop` is `call_allowance`.

**Core and evidence.** The storyline payload carries the effective `core_id` and `core_provenance` (`human` or `model`). An unknown model id is not treated as the core. Evidence for a proposal request is round-robin across each pair's two categories, then filled to `axial_evidence_limit`. Omitted passages are `omitted_evidence`.

**Tokens and clients.** `max_axial_review_tokens` (default 24000) is the estimated-input ceiling. `axial_token_budget` (default 6000) is the per-request ceiling. Both are public. Provider usage is stored on each prompt-log row and as `provider_token_usage`. `ProviderClientFactory` owns HTTP client configuration. A foreign client bound to another loop raises `CrossLoopClientError`.

## Decisions

**Separate graphs.** `get_graph("axial")` and `get_graph("selective")` are not nodes on `open_code`, `themes`, or `focused`. Open coding still stores `relate_categories` and does not treat those rows as a theory. Focused coding still leaves `saturation_report["judgment"]` null.

**Paradigm as a tool on the axial call.** `axial_code` defaults to `coding_lens="paradigm"` and `paradigm_version="1990"`. The roles are `causal_condition`, `context`, `intervening_condition`, `action_interaction` (action/interaction strategies), and `consequence`. A role with no grounded relationship is flagged `paradigm_slot_empty` and left empty. The diagram is not filled in. `coding_lens="charmaz"` on this call still proposes relationships, with role `unspecified`, and does not require the five slots. Focused coding still accepts only `charmaz`.

**Proposals, not edits.** A model relationship is `disposition: proposed`. A human resume may `accept`, `revise`, `reject`, or `merge`. Rejected and merged rows stay. A merge keeps both rows. The kept label is the target's statement, or the `label` on the resume when the researcher supplies a clearer one. The source row records `merged_into`. Nothing is deleted.

**Human fields and model fields.** `decision_mode="human"` calls `interrupt()` inside `relation_checkpoint`, `saturation_checkpoint`, and `core_checkpoint`. Resume with `Command(resume=...)`. `{}` is not a resume value. `decision_mode="llm"` does not interrupt. The reviewer follows the focused decision-review pattern (one call, no retry, the same audit fields) with an axial instruction: a saturation assessment is a model judgment and does not set `judgment`. It writes `model_disposition`, `model_paradigm_role`, `model_saturation`, and `model_core_id`, plus an audit row on the prompt log. Human `claim_support`, `researcher_saturation`, and `researcher_core_id` stay unset. `judgment` stays null. A model saturation assessment is `model_judgment: true`. It is not a human confirmation. `paradigm_version="2008"` is stored and uses the same five roles. The conditional matrix is not built.

**Evidence.** Each relationship cites a passage and a quote. The quote is anchored in that passage or the row is `evidence_condition: unanchored` and kept. Counterexample search uses the same bar as claim compete: two content tokens in common with the relationship's statement and quote, an embedding at or above 0.55 when both vectors are stored, or a hedge/exception that also shares a token. One token is not enough. Skipped pairs are logged. Verdicts are `supports`, `contradicts`, `qualifies`, and `irrelevant`. A quote is required only for the first three. An irrelevant judgment with an empty quote is valid. A valid `contradicts` or `qualifies` sets `evidence_condition` to `disputed`, stores a negative case, and does not drop the relationship. Fewer than `min_relation_documents` (default 2) distinct documents of anchored evidence flags `tentative_few_documents`. The row stays.

**Saturation is a report.** Indicators, per document (batch size stays 1): new-code, new-property, and new-relationship rates; code-frequency shift against the previous document; property and relationship stability across runs when more than one run is supplied; and the curve of those windows. Fewer than two documents leaves frequency stabilization undefined. One run leaves cross-run stability undefined. The software does not set `judgment`. A researcher may record `not_claimed`, `sufficient_for_this_question`, or `not_saturated`, with a rationale. There is no boolean true.

**Optional later steps.** `selective_code` scores core candidates and does not pick a winner. Centrality is counted twice: proposed relationships, and relationships a researcher has accepted. `include_memos` writes one theoretical memo per relationship. `include_storyline` writes statements that cite a passage. A statement without a citation is kept and flagged `storyline_uncited`. `theory_coverage` reports the citation anchoring rate, the share of relationships that received a negative-case search, disputed relationships, and unsupported statements. Each rate is null when there is nothing to score. `axial_code(..., also_selective=True)` runs the selective graph only after the axial graph finishes without an interrupt or a timeout.

**Budgets.** `max_axial_review_calls` (default 8) is split by `axial_share_propose` and `axial_share_counterexample` (default 0.5 and 0.5). Unspent propose calls move to the counterexample share, and the transfer is recorded. `max_selective_review_calls` is a separate ceiling for memos and the storyline. `max_decision_calls` bounds the LLM reviewer only. These are ceilings. `batch_size` (default 1), `axial_request_timeout`, and retries are performance settings. They are not the analytic procedure. A timeout uses the 0.7.9 monotonic deadline. The call returns an incomplete result unless `raise_on_timeout=True`.

**Synthetic check.** `axial_synthetic_counterexample=True` injects one passage labeled `[synthetic counterexample]`. It is forced onto the counterexample queue, recorded in metadata when a contradict or qualify is returned, and omitted from research exports.

## Graphs

Axial: `load_focused` → `plan_relations` → `propose_relations` → `relation_checkpoint` → `counterexample_worker` → `commit_relations` → `saturation_indicators` → `saturation_checkpoint` → `export_axial`. `propose_relations` also returns to itself after each request while pairs remain.

Selective: `load_category_development` → `score_candidates` → `core_checkpoint` → `draft_memos` → `draft_storyline` → `export_selective`.

`draft_memos` and `draft_storyline` no-op unless that step was requested. The nodes stay on the graph so the diagram does not change with the flags.

## Exports

`category_relationships.json`, `thematic_saturation.json`, and `theory_coverage.json` on an axial run. `core_candidates.json`, `theoretical_memos.json`, and `storyline.json` when those steps run. `axial_run.json` is the audit copy. Research files omit the synthetic passage.

## Deferred

Theoretical sampling and the sampling brief, Glaser's coding families, the conditional/consequential matrix, a Taguette round-trip for relationships, and a CLI. Relation stability compares `(from_id, to_id, paradigm_role)` and does not yet score conditions beyond that role. A storyline adversarial pass beyond the relationship counterexample search is not a separate node.
