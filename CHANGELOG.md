# Changelog

## 0.9.23

- A `system_proposal` review no longer removes its proposal from independent discovery matching. The same proposal can be recovered for structural recall and linked for the review. Structural recall and precision do not change when proposal reviews are added or removed. The gate false-negative population is still per adjudication, so a review can change that rate.
- Discovery assignment gives every gold item an unmatched option. The objective is lexicographic: full-agreement recoveries, then structural matches, then canonical content order that does not use ids. One full recovery beats any number of distorted matches. Matching runs on candidate-connected components, one solve each.
- Claim reviews are stored apart from independent gold. Each review is bound to a run id, a relationship id, and `claim_statement_hash` of the statement, direction, and qualification. A changed statement, a changed direction, or a missing or different run is stale and unjudged, not correct. `claim_correct_rate` counts fresh reviews on independent matches and on id-bound proposals, once per relationship, and is null when nothing was judged. `claim_stale_count` is separate.
- The schema is `discovery-eval-3`. `discovery-eval-1` and `discovery-eval-2` files still verify under their old hashes. An unbound claim judgment loads as stale or unjudged and is never scored correct. Freezing rewrites the file to `discovery-eval-3` and a new hash.
- An epistemic or existential-negation marker before a connector, including "there is no evidence that", does not authorize tier 3. `there is no doubt` fails closed. The synthetic fixture is `tests/fixtures/attribution_adversarial_2.json`. Passing it is regression coverage, not general semantic accuracy. `score_attribution_accuracy` scores only quotes the caller supplies and is null when none are supplied.
- The axial planner still builds the full pair space in memory. That limit is open. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays human-only. The review note is [docs/design-review-fixes-0923.md](docs/design-review-fixes-0923.md). `policy_version` is `0.9.23`.

## 0.9.22

- Axial `examined_pairs` records completed examinations only, including a valid empty `nothing_to_propose` response. Planned, attempted, and completed counts are separate. `unique_pairs_explored` counts completed pairs. A failed, timed-out, or unsent pair stays retryable. A 0.9.20 or 0.9.21 `examined_pairs` list was written at planning time and cannot be split; passing it still excludes every pair in it.
- Tier 3 quote validation requires one clear clause and one unambiguous connector. Passive `caused by` and `led to by` reverse direction. Negation, a second claim clause, and hypothetical wording stay unvalidated. This is not general semantic reliability. The synthetic fixture is `tests/fixtures/attribution_adversarial_1.json`.
- Discovery recall uses maximum-weight one-to-one matching. Full agreement wins, then any structural match, then canonical content order. Adjudication ids are not a tie-break. A `system_proposal` row binds only by `system_relationship_id`. A missing or duplicate export id is flagged unmatched and is not reassigned.
- Structural recall and `claim_correct_rate` are separate. The schema is `discovery-eval-2`. A `discovery-eval-1` file still verifies under its old hash and loads as `not_judged`. Freezing it writes a new hash. Structural recall measures the specified structural fields, not recovery of the human-supported interpretation. `claim_correct_rate` is null when nothing was judged and is not folded into recall.
- `discovery_recovered_per_reread_token` and `discovery_recovered_per_reread_dollar` name reread expenditure only. The planner still builds the full pair space in memory. That limit is open. The saved output still omits a large unexamined list.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays human-only. The review note is [docs/design-review-fixes-0922.md](docs/design-review-fixes-0922.md). `policy_version` is `0.9.22`.

## 0.9.21

- Discovery evaluation is separate from gate evaluation. `load_discovery_file`, `freeze_discovery_set`, and `score_discovery` read a frozen human file of independently identified relationships and score recovered, missed, distorted, and unmatched proposals. A relationship the run never proposed can lower recall. Gate scoring still excludes an adjudication id that is absent from the export.
- The schema version is `discovery-eval-1`. JSON or JSONL is the frozen artifact. A header-only CSV template is the adjudicator worksheet. The SHA-256 `frozen_hash` is checked on load. A file with no hash is not scored. Passing no discovery file leaves every rate null. Disagreements are stored beside the gold row and do not change the rates. The design is [docs/design-discovery-evaluation.md](docs/design-discovery-evaluation.md).
- Recall, precision, and distortion use explicit denominators and stay null when the denominator is empty. Matching is code ids plus passage ids. A wrong direction is distorted, not recovered. Tier labels and model judgments are not inputs. A withheld but supported proposal is a gate false negative, with the 0.9.14 known-eligibility split. Recovered rows per token and per dollar stay null unless reread expenditure is complete.
- `examples/discovery_adjudication_example.json` is synthetic example data only. It is not a study result. No real adjudication is included.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays human-only. The review note is [docs/design-review-fixes-0921.md](docs/design-review-fixes-0921.md). `policy_version` is `0.9.21`.

## 0.9.20

- Tier 3 wording `participants attribute X to Y` requires a deterministic reading of the quote: affirmative polarity, no hedge, and the stated from-to direction, plus a relevant rival check. A keyword hit only nominates the quote. A topical explanation, a denial, a reversed direction, an undirected association, and a hedge stay below tier 3. The quote text, direction, polarity, hedging, and `validated_by` stay on the participant claim. Validation does not call a model.
- A relationship keeps three claim components: the observed pattern, the participant explanation, and the analyst interpretation. The pattern supports downstream use and a core when co-occurrence is independently supported. The participant explanation does not support a core. The analyst component stays a hypothesis and no longer excludes the row. Claim flags are recomputed on every reassessment. The previous reading stays on `claim_revision_history`. A 0.9.19 row that stored the analyst flag as a row-level exclusion stays unsupported until reassessment, which records that exclusion and then uses the pattern.
- Selective resume hashes relationships, memos, properties, and dimensions, along with the axial evidence fingerprint. A changed relationship, disposition, memo, property, or dimension raises `AxialCheckpointMismatch`. An unchanged source still resumes. A 0.9.19 selective checkpoint cannot resume. Axial resume is unchanged.
- Axial planning reserves one same-respondent slot and one lexical cross-case slot when those classes still have unexamined pairs. The call allowance is not raised. Examined pairs are passed into the plan and stored on the result. A seed rotates which pairs a class contributes when that class has more pairs than slots. The unexamined pair list is stored only for a pair space of 256 or fewer. A larger space stores the count and the selection procedure.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays human-only. The design is [docs/design-review-fixes-0920.md](docs/design-review-fixes-0920.md). `policy_version` is `0.9.20`.

## 0.9.19

- Tier 3 wording `participants attribute X to Y` requires a participant quote that states the link and is anchored to the passages, plus a relevant rival check. An analyst memo no longer raises that tier. A memo that is explicitly an analyst interpretation, and that names both endpoints, stays tier 2 with wording `analyst interprets X as related to Y`. It is flagged `analyst_interpretation`, is not downstream eligible, and does not support a core. A model memo cannot authorize either claim. A shared token plus an explanatory word is not an interpretation.
- Axial and selective resume hash code definitions, segment assignments, and case identity, following the focused-coding fingerprint and adding the respondent fields the tier count uses. A changed definition, a removed assignment, or a changed case id raises `AxialCheckpointMismatch`. An unchanged source still resumes. A 0.9.18 checkpoint, whose hash covered only passage text and code ids, cannot resume: the error says the evidence fingerprint is missing and a new run is required. The default seed still uses the passage-and-code-id hash. Sampling's `source_hash` is unchanged.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. Tier 4 stays human-only. The design is [docs/design-review-fixes-0919.md](docs/design-review-fixes-0919.md). `policy_version` is `0.9.19`.

## 0.9.18

- A timed-out reread keeps its operation. An initial reread is logged as `sampling_reread`. A reread repair is logged as `schema_repair` with schema `SamplingRereadNote`. Both count as sent calls with unknown usage. `reread_expenditure_complete` stays false, coverage counts the attempt, and the full expenditure rates and cost stay null. The known token sum excludes that attempt. Selection and saturation timeouts stay kind `provider`. Axial and focused timeout rows are unchanged. The provider budget call count is still the limiter's `calls_used`.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. The design is [docs/design-review-fixes-0918.md](docs/design-review-fixes-0918.md). `policy_version` is `0.9.18`.

## 0.9.17

- Reread requests that do not fit the per-request budget or the remaining token allowance are withheld in one visit. They are not provider calls, so they do not spend the recursion limit one passage at a time. A human assignment of 32 oversized passages, with the default four-call allowance, ends with every passage unsent and stop `request_budget`. The visit limit stays the 0.9.16 count. A model selection is still cut to the call allowance before that loop.
- A proposal batch is recovered from JSON text, from tool-call arguments, and from the repair response. Valid rows are kept. A batch with nothing valid still uses one repair. Grounding checks still run after salvage.
- `reread_expenditure_tokens` is the known sum. `reread_expenditure_calls`, `reread_expenditure_calls_with_usage`, and `reread_expenditure_usage_coverage` separate no calls from calls with missing usage. `reread_expenditure_complete` is false when a sent call did not report usage. Full expenditure rates and cost stay null in that case. A missing provider report stored as zeros is not a known spend of zero. `tokens_per_relevant_reread` is unchanged. Prices are read from source `metadata.token_prices`.
- `validation_basis` is refreshed on every validation. History is still written only when the owned flags change, and it archives the previous basis. A human accept validates the statement before it applies a role, so a sentence edit that does not change flags is the statement a later role change archives.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. The design is [docs/design-review-fixes-0917.md](docs/design-review-fixes-0917.md). `policy_version` is `0.9.17`.

## 0.9.16

- An editable install is consistent when `direct_url.json` or an editable finder names the imported checkout. A distribution version that differs from the imported module still raises `InstallationMismatch`, and axial and selective coding check that before a model call.
- The sampling recursion limit is the graph's visits: preparation, one selection repair, three visits per coding call, one withhold visit, and finalization. A 32-call reread is not stopped by a fixed overhead.
- Validation flags are recomputed from the effective statement and role. A human correction removes flags whose conditions no longer apply. The previous flags stay on `validation_revision_history`.
- Until a human sets a role, the reviewer's role is the working role for validation, eligibility, export, and `validated_decision`. The proposal, model, reviewer, and human roles stay on the row.
- The restart test resolves its own repository root.
- A proposal batch that mixes valid and invalid relationships keeps the valid rows and logs each dropped row with its index, reason, error, and draft. It does not spend a repair. A batch with nothing valid still uses one repair.
- `tokens_per_relevant_reread` is unchanged. `expenditure_tokens_per_relevant_reread` divides all reread tokens, including irrelevant responses, failed parses, and repairs, by `relevant_reread_count`. The rate is null when that count is zero, and the count is reported. `human_confirmed_useful_rereads` is null and is not computed.
- `idle_timeout` is recorded and is not enforced. Returning after a timeout does not prove the provider request stopped. Enforcing the idle limit is not feasible with the current structured calls.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. The design is [docs/design-review-fixes-0916.md](docs/design-review-fixes-0916.md). `policy_version` is `0.9.16`.

## 0.9.15

- A returned axial relationship requires a nonempty statement. `nothing_to_propose` with an empty list remains the explicit empty batch. One repair is sent against the original evidence, and a statement that is only a quote is discarded. `proposal_diagnostics.rejection_reasons` counts discarded proposals by reason.
- `downstream_eligible` is supported evidence only. A weak stem, a tier below 2, or a missing tier is `hypothesis_exploration`. Core graph support ignores a stale eligible flag on a weak stem. Memos and the storyline snapshot still receive that hypothesis row.
- `unspecified` is a valid role when the evidence is only an association. A causal role below tier 3 is `role_evidence_exceeds_tier`, apart from `wording_exceeds_tier` on the sentence. The causal gate stays on.
- Axial export records an attrition funnel and install provenance: commit, module path, module and distribution versions, model, and generation parameters. Temperature is unset unless the caller configured it. A distribution that does not match the imported checkout raises `InstallationMismatch`.
- Sampling reports split non-relevant rereads into irrelevant, unjudged, unanchored, and failed, with relevance by direction type and by document or respondent novelty, and tokens per relevant reread. Cost stays null unless prices were configured. `SamplingResult.completed_pairs` repeats `metadata.reread_completed_pairs`.
- `timeout_policy` records the elapsed provider limit apart from `idle_timeout`. The inactivity limit is stored and is not enforced. A node timeout does not stop a blocking provider call. Process isolation is the boundary that can, and the stage graphs leave it off.
- A human resume restarts the interrupted node with `Command(resume=...)`. An execution error retries that node when the same thread is invoked with `resume=None`. Human and model relation reviews write the same `validated_decision` shape with different provenance.
- `metadata.persistence` names the checkpointer and `durability`. The default saver is in-memory and is lost at process exit. `checkpoint_path` keeps a SQLite file. `durability` stays `async`.
- Sampling keeps one `pair_records` entry per direction and passage. Completed, failed, and unsent lists, plus `reread_status` and `reread_stop`, are derived from those records.
- The LangGraph floor is `>=1.2,<2`. Node timeout and retry policy need that release.
- A repair that is still over the request budget is `not_sent` and does not reuse the original usage. A repair the call allowance refuses stays `schema_unparsed` and `incomplete`. The original response remains.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. The design is [docs/design-review-fixes-0915.md](docs/design-review-fixes-0915.md). `policy_version` is `0.9.15`.

## 0.9.14

- `downstream_withheld_supported_recall` and `downstream_precision` stay null unless every row in that rate's population has a boolean `downstream_eligible`. Unknown eligibility is not a lower bound stored as a point estimate. `downstream_withheld_supported_recall_known` and `downstream_precision_known` score the known rows and report the denominator and coverage. Per-rule results use the same split. A zero denominator stays null.
- The sampling saturation check uses the configured `min_documents`. Below it, the model decision is `not_interpretable`, `minimum_documents` is that configured value, and the decision call is not spent. A human checkpoint still pauses and reports `window_ready`. The default remains 3.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. The design is [docs/design-review-fixes-0914.md](docs/design-review-fixes-0914.md). `policy_version` is `0.9.14`.

## 0.9.13

- A reread pair that times out and then succeeds leaves `reread_failed_pairs` when that pair succeeds. The timeout is kept on `reread_recovered_timeouts` and in the sampling attempt log. A later pair withheld for `call_allowance` is the current `reread_stop`. A pair is not both completed and failed. An unresolved timeout still blocks.
- Gate export copies `disposition` and `downstream_eligible` separately, plus withholding flags such as `causal_wording_withheld`. `downstream_precision` and `downstream_withheld_supported_recall` score that block. Disposition metrics keep their names. A rate with a zero denominator stays null. An export without `downstream_eligible` scores the new rates as null.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. The design is [docs/design-review-fixes-0913.md](docs/design-review-fixes-0913.md). `policy_version` is `0.9.13`.
- The README and [docs/README.md](docs/README.md) describe the current graphs. Axial, selective, and sampling stay experimental. Older `design-review-fixes-*` notes stay as history.

## 0.9.12

- A reread that times out and then completes clears the active `reread_stop`. The timeout is kept on `reread_recovered_timeouts` and in the sampling attempt log. An unresolved timeout still leaves quiet rounds `not_evaluable`.
- Unsent pairs, failed pairs, and scheduled reread pairs carry the stable `question_key`. Quiet-round exemption reads that key before the round-specific direction id, so a closed question matches the passage that was not sent.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. The design is [docs/design-review-fixes-0912.md](docs/design-review-fixes-0912.md). `policy_version` is `0.9.12`.

## 0.9.11

- A proposal request no longer includes the full `lexical_only_withheld` list. The model receives a count. The complete list stays on the run metadata and in `axial_run.json`. Optional diagnostics are dropped before passage evidence is shortened.
- Quiet rounds are `not_evaluable`, and `observed` is null, unless every selected pair was completed or explicitly exempted. A `request_budget` or `token_allowance` refusal, a parse failure, and an unresolved timeout are not silence. A completed pair and a closed direction stay exempt.
- Pair evidence is classified from passage tokens, endpoint terms, and case memberships computed once, with an inverted index for lexical overlap. Classification is skipped when no proposal request is scheduled. The class assigned to each pair is unchanged.
- A weak lexical overlap does not reserve a passage. A later pair in the same batch can still offer that passage when the overlap is strong enough. A passage is reserved only after it qualifies.
- `pattern_min_cases` stays 2. The axial token ceiling stays 24000. The design is [docs/design-review-fixes-0911.md](docs/design-review-fixes-0911.md). `policy_version` is `0.9.11`.

## 0.9.10

- Axial pair planning takes same-passage, same-respondent, and lexical cross-case pairs before the seeded fill, and reports how many pairs were examined against how many were plausible. `max_axial_review_calls` still defaults to 8, so a default series examines a small fraction of the pairs. A corpus where no pair can reach `pattern_min_cases` records a warning. The minimum stays 2 and the axial token ceiling stays 24000.
- A tier-1 hypothesis may rest on one passage or one respondent. A statement stitched from unrelated passages is still rejected. `nothing_to_propose` is an explicit empty outcome. Blank rows and statement-less rows are counted apart.
- `case_id_of` reads `metadata.respondent_id`. A one-stem lexical overlap is `lexical_only` with support cap 1 and is not offered as cross-case evidence. Causal wording below tier 3 is not `downstream_eligible`. An absence qualification keeps its reason. A same-passage stem-only quote is `same_passage_stem_only` and stays proposed.
- Sampling baselines carried token estimates. The saturation request is packed to `sampling_token_budget`, and omitted fields are recorded. An unlisted human direction stays open. Quiet rounds are `not_evaluable` while selected pairs remain unsent.
- Memo accounting balances requested, produced, omitted, failed, and rejected rows. The axial token plan estimates the planned round. A focused raw estimate above the cap is labeled apart from the guarded figure. A synthetic check with no relationship is `not_run`. `export_gate_relationships` lists proposed rows. A focused timeout reports `timeout_elapsed_seconds`.
- The design is [docs/design-review-fixes-0910.md](docs/design-review-fixes-0910.md). `policy_version` is `0.9.10`.

## 0.9.9

- A sampling question keeps a stable key. A relevant reread is an observation of that question and does not wrap it in another `reread_follow_up:` direction. A new question is added only when `follow_up_kind` and `follow_up_target_id` name a different code, relationship, or core already on the run. The note text is not matched.
- `completed_pairs` records each question and passage that was reread. That pair is not offered or reread again. Older nested `reread_follow_up:` keys are read as the underlying question.
- `export_gate_relationships` lists accepted and withheld relationships with the rules already stored on each row. `load_adjudications` and `score_gate_adjudications` score a separate adjudication file. A rate with no adjudicated rows stays null. The gates are unchanged.
- The design notes are [docs/design-review-fixes-099.md](docs/design-review-fixes-099.md) and [docs/design-gate-evaluation.md](docs/design-gate-evaluation.md). `policy_version` is `0.9.9`.

## 0.9.8

- A reread judgment is the structured field. Free text is not read for the word irrelevant. An irrelevant passage is not offered again for that direction, and the direction stays open. Closing a direction takes a human reject or a model close recorded with a reason.
- `model_sampling_pairs` keeps the validated selection. The coding-call cap applies only when `pursuit` is `reread`, and those pairs are `scheduled_reread_pairs`. `brief_only` is not `budget_limited` because no reread was requested. Pairs beyond the allowance stay unsent with reason `call_allowance` when a reread was requested.
- A selection repair reuses the packed directions and passage proposals. Proposals dropped to fit `sampling_token_budget` are not restored.
- The design is [docs/design-review-fixes-098.md](docs/design-review-fixes-098.md). `policy_version` is `0.9.8`.

## 0.9.7

- Causal and consequence wording, including the roles `consequence` and `cause`, fails the tier gate below tier 3. A reviewer who says no causal language was used is checked against that detector.
- `core_stability` includes the current run's named core. `n` is the number of named cores.
- One distinctive stem is a weak anchor. It does not count as `same_passage` or `cases_with_both_endpoints`. An absence candidate is not stored as a contradiction.
- Negative-case rows and the aggregate share one attempted and remaining count. Prompt rows are split into provider calls and unsent rows. Research judgments exclude the synthetic check.
- Sampling call ceilings count this stage. Upstream calls stay on `sampling_call_baseline`. A selection request is packed to `sampling_token_budget`. Pairs beyond the coding allowance are unsent with reason `call_allowance`.
- Reread notes are written onto the ledger. The next round drops a direction an irrelevant note or a reject closed. The window rate on the ledger is the rate the stopping criteria read. `judgment` stays null.
- A focused claim round with plausible candidates left unsent is `budget_limited`. A memo that echoes the prompt is flagged, and a rejected row is not sent for a memo. A non-causal storyline sentence does not flag an accepted relationship. The raw focused token estimate and the guarded ceiling figure are both reported. The axial token ceiling is unchanged, and a planning estimate warns when it is short. An irrelevant or missing synthetic check is `not_interpretable`.
- The design is [docs/design-review-fixes-097.md](docs/design-review-fixes-097.md). `policy_version` is `0.9.7`.

## 0.9.6

- A malformed sampling reread is checkpointed, with its usage, before the repair is sent. A repair timeout keeps that response. Resume sends the repair and does not repeat the initial request. The consumed allowance stays consumed.
- `reread_active_pair` is recorded before each provider call. A timeout uses it as the failed pair. Pairs already withheld, including a request that did not fit the per-request budget, stay on `reread_unsent_pairs` and are not labeled as the failure.
- Each nonempty flat id list is compared on its own. An omitted list is derived from the explicit selections. One matching list plus one omitted list is not `model_flat_list_mismatch`.
- The design is [docs/design-review-fixes-096.md](docs/design-review-fixes-096.md). `policy_version` is `0.9.6`.

## 0.9.5

- Each sampling reread is one graph step. A later timeout keeps the notes, usage, and prompt rows already stored. The partial result lists completed, failed, and unattempted pairs. Resume skips completed rereads and keeps the consumed allowance. A selection that does not parse is checkpointed before its repair, so a repair timeout keeps that first response.
- Explicit `selections` are the rereads. They are checked against the directions and the corpus. The flat id lists are derived from the pairs that pass. Empty flat lists were not sent. A non-empty flat list that names a different set does not replace the pairs, and the stage is incomplete. A pair that names an unknown direction or passage is rejected. Dropped reread work is not reported as complete.
- The design is [docs/design-review-fixes-095.md](docs/design-review-fixes-095.md). `policy_version` is `0.9.5`.

## 0.9.4

- A sampling reread is a `(direction_id, passage_id)` pair. A researcher's passage assignment is not moved to a direction that only retrieved the passage. A passage selected for two directions is reread once per direction, and each reread spends its own coding call. An unsent pair names the direction.
- A counterexample keeps the selected passage on `passage_id` and records `quote_passage_id`, `quote_start`, and `quote_end` for the passage that contains the quotation. Endpoint anchoring, quote locations, supporting cases, and the storyline passage list use that passage. A record without `quote_passage_id` is still read as the selected passage. Offsets are null when the quotation is not anchored.
- Pairs with no category passage are reconciled once before the proposal loop. The derived step allowance is the planned proposal requests, plus one defer or split return per planned pair, plus the existing overhead. A caller can still set `recursion_limit`. Hitting it returns the partial result.
- The design is [docs/design-review-fixes-094.md](docs/design-review-fixes-094.md). `policy_version` is `0.9.4`.

## 0.9.3

- Negative-case classification is per respondent. Passages from one respondent are combined before a type is assigned. A split interview is not queued as both `a_without_b` and `b_without_a`. The request includes a complementary passage when that passage changes the reading. One representative candidate is queued per respondent.
- Plausible, queued, attempted, and remaining counts stay separate. The stage is `budget_limited` whenever a plausible candidate was not queued or not attempted, including when another relationship was judged. The first slot is offered to each relationship before any relationship receives another.
- Sampling measures each initial request. `sampling_token_budget` and `max_sampling_review_tokens` are enforced separately. A request that does not fit is not sent, and the stage is `incomplete` with a reason. Failed sampling output is preserved. A parse failure after repair is `incomplete`, with each attempt and its usage. A reread includes the selected direction, the sampling query, and the target evidence, and the note is stored on that direction.
- Axial proposal steps no longer use a fixed graph limit of 80. The step allowance is the planned proposal requests plus graph overhead, or `recursion_limit` when the caller sets it. Focused coding already derived its limit. Sampling and selective coding do too. If a limit is still hit, the run returns the partial result and marks the stage `incomplete` with reason `recursion_limit`.
- The design is [docs/design-review-fixes-093.md](docs/design-review-fixes-093.md). `policy_version` is `0.9.3`.

## 0.9.2

- Negative-case search is respondent-level. For a relationship at tier 1 or above, and for a claim, candidates are respondents with one endpoint and not the other, an opposite pole or a different outcome, both endpoints, or an embedding at or above 0.55. A cheap pre-filter holds the rest back. Tier 0 is not searched. The search record stores candidates considered, how many were filtered, why no call was made, and a descriptive 2x2. Those counts do not validate a link.
- There is no opposite-pole fallback call. No plausible candidate is `no_plausible_candidates` and the stage is `complete_no_negative_search`. An exhausted budget with candidates still waiting is `budget_limited`. An all-irrelevant round leaves the negative-case rate and the relevance rate null.
- A valid `contradicts` flags `relationship_contradicted` and stores `contradiction_note`. The row stays. A valid `qualifies` stores a variation note and reports the variation search. It remains a relevant rival judgment. Storyline and memo text that ignore a contradiction or a variation are flagged. `negative_case_search` is a procedure switch, default on. It is not a performance setting.
- A core is graph-supported only as an endpoint of a tier-2 or higher relationship. Candidates show the counts. A model core without that support is kept as `model_core_id`, flagged `core_unsupported_by_graph`, and not promoted. The storyline does not center it. Integration reports direct links, path links, orphans, and contradicted relationships. `core_category_coverage` and `core_respondent_coverage` are null when no relationships exist.
- A storyline statement cites relationship ids and tiers, or it is `description`. Wording stays inside the tier. A contradicted relationship keeps its qualifier. `core_stability` is pairwise agreement and is null below `core_stability_min_n` (default 2, a design choice). Disagreeing cores score 0. Core selection, memos, and the storyline store the graph snapshot they were built from.
- The design is [docs/design-negative-cases-092.md](docs/design-negative-cases-092.md). `policy_version` is `0.9.2`.

## 0.9.1

- A relationship carries `evidence_tier`, `evidence_label`, and `allowed_wording`. Tier 0 is category evidence: one verbatim quote for an endpoint, not a relationship. Tier 1 (`hypothesis`) anchors each endpoint somewhere, possibly in different respondents, and is flagged `cross_case` when those quotes do not share a case. Its wording is "may be related". It is never accepted as a supported relationship.
- Tier 2 (`pattern_supported`) needs `pattern_min_cases` distinct cases (default 2). Each case has a verbatim quote for both endpoints. The same passage is the strongest within-case evidence. The same respondent across question segments is weaker and still one case. Wording is "co-occurs in N respondents", not "leads to". Accepting tier 2 records `pattern_supported`. It does not record `verified`.
- Tier 3 (`attribution`) is tier 2 plus a participant quote that links the two, or an analyst memo that says why, and a rival check that actually returned a relevant judgment. Wording is "participants attribute X to Y". Only tier 3 or 4 may use causal wording. A stronger sentence is kept and flagged `wording_exceeds_tier`, `storyline_wording_exceeds_tier`, or `memo_wording_exceeds_tier`.
- Tier 4 (`verified`) is a researcher action, `verify`, with a non-empty `scope`. The row has to already be tier 3. Otherwise `verification_withheld` and the computed tier stays. `decision_mode='llm'` cannot set `verified` (`model_cannot_verify`). `judgment` stays null.
- Supporting cases record the case id, passage ids, and per-endpoint quotes. A category label is not a quote. Respondent-level co-occurrence is a 2x2 count (`both`, `a_only`, `b_only`, `neither`) and is descriptive only. A missing corpus leaves those counts null. Variation search and the rival check stay `undefined` when they were not run.
- Proposal requests include `candidate_cross_case`: other respondents' passages that contain both endpoints' distinctive terms, inside `axial_evidence_limit`. The same-passage list is still preferred and is not displaced to make room.
- `citation_anchoring_rate` is the share of tier 2 or higher rows whose cases quote both endpoints. It is null when none qualify. `citation_anchoring_by_tier` reports `n` and `citation_rate` per tier. Tiers below 2, and empty tiers, leave `citation_rate` null. `citation_anchoring_same_passage` is that share among tier 2 or higher rows whose strongest within-case evidence is the same passage, and is null when there are none. `citation_anchoring_cross_passage` stays null.
- `pattern_min_cases` is recorded on the procedure and on the row, with a note that the floor is a design choice for this package and not a quotation from the method literature. The design is [docs/design-relationship-tiers.md](docs/design-relationship-tiers.md).
- `policy_version` is `0.9.1`.

## 0.9.0

- Theoretical sampling is its own graph, `get_graph("sampling")`. It ranks directions from the categories, relationships, saturation indicators, coverage, and open memos already on the run. Each direction names the target, why, a sampling query, and selection criteria. Retrieval proposes passages in the existing corpus that are not already evidence for that target. When none qualify, the direction is `no_suitable_unsampled_passages` and new data collection is recorded as a researcher action. The package does not collect that data.
- A human checkpoint accepts, edits, or rejects directions and can record the passages actually sampled, including an explicit `outside_corpus` note. `decision_mode='llm'` writes `model_direction_ids` and `model_passage_ids`. Invented ids are flagged and are not used as evidence. Those model fields are not copied into the researcher record.
- `pursuit` defaults to `brief_only`. `theoretical_sampling.json` includes the ledger and a `next_round` plan for a later focused or open coding run. `pursuit='reread'` spends the sampling call ceiling on notes for the selected passages. A quote that does not sit in the passage is dropped and flagged `quote_unanchored`. The note does not change the saturation curve.
- The ledger appends one round at a time. Constant comparison across rounds is the delta of the window-wide rates. A rate that was undefined stays undefined. `provider_budget` survives a human pause and a timeout. A timeout keeps the rounds already written.
- Saturation still has no automatic boolean. `saturation_decision` stores a researcher decision (`not_claimed`, `sufficient_for_this_question`, or `not_saturated`) with a rationale, an actor, and a UTC timestamp. The checkpoint shows the indicators, the ledger, and a stopping-criteria report. That report marks new codes, new properties, new relationships, code-frequency shift, quiet sampling rounds, and core-category coverage as met, unmet, or undefined, with the threshold on the row. It does not declare the theory saturated. `judgment` stays null.
- `decision_mode='llm'` writes `model_saturation_judgment` only. Below 3 documents, or below 2 sampling rounds, the assessment is withheld as `not_interpretable` and is not filled in as `not_saturated`. It is never written into `saturation_decision` or `researcher_saturation`.
- Axial and selective coding were not rebuilt. A sampling query may name a limited Glaser family (the Six Cs, strategy, or means-goal) as a sensitizing label. Axial `coding_lens` is still `paradigm` or `charmaz`. The conditional matrix, the full 1978 family catalog, and collecting new data remain unimplemented.
- v29 live-eval fixes on the same graphs, without a new graph shape. A storyline sentence that cites an unaccepted, provisional, or cross-respondent row is flagged `storyline_unaccepted_relationship` and marked as a pattern. The sentence is kept.
- The focused synthetic passage is sent as `p-contrast`. That id is mapped back to `synthetic-counterexample` before the reread is counted, so the call is addressed. The synthetic judgment stays on the audit and is excluded from `claim_test_counts` and claim-test relevance.
- Relationship proposals are shown passages that already contain both categories' segments. Endpoint quotes have to be verbatim spans of a passage id that was sent. A cross-respondent row stays a pattern. `relationship_counts` separates accepted, provisional, cross-respondent, proposed, and rejected.
- The saturation reviewer receives pair coverage and the number of relationships found. When 10 or more pairs were planned, coverage below 0.05 or fewer than 2 relationships is not interpretable. `sufficient_for_this_question` is withheld. The model's words stay on `model_said`. The stopping-criteria report has a `pair_coverage` row. Below 10 planned pairs that row stays undefined.
- A quote that is not a verbatim substring of a located passage is not anchored. `quotes_not_in_cited_passage` still fires. A very short quote that is in the passage can use a passage embedding as a second signal. If that signal is missing, the endpoint stays unanchored and the disagreement is recorded.
- A memo that only says the evidence is two-sided is `memo_degenerate`. A memo that does not quote the evidence is `memo_missing_quote`. Causal wording on a cross-respondent or unaccepted row is `memo_causal_on_pattern`.
- Counterexample retrieval records why no research calls were made (`no_candidates_meeting_both_endpoint_overlap` or `budget`). When both-endpoint overlap is empty, it queues the best same-category opposite-pole passages that fit the budget.
- Core candidates with relationship evidence are listed first. If the model still names a category with none, `model_core_id` is kept, `core_unsupported_by_graph` stays set, and `relationship_preferred_core_id` names the endpoint that was preferred. `core_stability` stays undefined and reports `n`.
- `proposal_diagnostics.rejection_reasons` counts blank rows, unknown passages, empty statements, and unrequested pairs separately. A row with no statement and no quotes is a blank row even when the passage id is unknown.
- `model_determinism` records the requested temperature and seed. It does not claim that a paused run and an uninterrupted run returned the same text.
- `policy_version` is `0.9.0`.

## 0.8.7

- A relationship records where each quote sits. `citation_anchoring_rate` is 1 only when both endpoints are anchored and both quotes are in the cited passage. Otherwise it is null. `citation_anchoring_same_passage` is the both-anchored share of same-passage rows. `citation_anchoring_cross_passage` stays null, because quotes from different passages do not evidence the link. A causal or consequence link is not accepted on cross-respondent quotes; the row stays as `cross_respondent_pattern`. The reviewer sees a per-quote passage check. A rationale or memo that says "same passage" when the quotes do not share one is flagged.
- The focused checkpoint manifest is built after temperature 0 is applied, so resuming with the same model object does not see a changed manifest. Temperature 0 does not guarantee identical completions. A seed is passed when the instance already exposes one.
- A generic stem such as work, job, or team does not anchor a category by itself. The quote has to share a distinctive stem, or an embedding at or above 0.55. Disagreements between those signals are reported.
- A blank proposal row or an empty statement is rejected, counted in `proposal_diagnostics`, and not stored as a relationship.
- A core that is not an endpoint of a relationship in the run is `core_unsupported_by_graph`. `core_stability` stays undefined and reports `n`.
- A counterexample or claim stage that made no calls, or whose judgments were all irrelevant, is `complete_no_negative_search` rather than `complete`. Counterexample retrieval prefers another respondent's passage that mentions both endpoints' distinctive terms.
- When a storyline is requested, one quarter of the token ceiling is reserved for it. That reserve is procedure, not a performance setting. If the storyline still does not fit, the stage is `budget_limited` with reason `token_allowance`.
- The synthetic passage sent to the model has no synthetic flag, no audit label, and neutral text. The audit keeps the label and the limitation.
- `provider_budget.tokens_estimated` remains the raw heuristic. `tokens_estimated_compared` is the guarded figure that is compared with the ceiling.
- The saturation reviewer receives last-document rates and window-wide rates, with the window definition. A last-document rate is not the window rate. `judgment` stays null.
- `metadata.seed` is the same value as `pair_coverage.seed`.
- `policy_version` is `0.8.7`.

## 0.8.6

- A relationship is accepted only when each endpoint has its own evidence: `quote_from` and `quote_to`, or one quote that matches both. A match is a content stem shared with the category label, its definition, or a segment quote, or an embedding at or above 0.55. Stopwords, including `from`, do not count. `matched_tokens` records the stems. A one-sided row stays in the run as `provisional` with `acceptance_withheld`, and the reviewer and the storyline see that status.
- Counterexample passages are retrieved by meaningful overlap with both endpoints, or by an embedding at the evidence threshold. One-sided overlap is skipped. `relationships_with_negative_case_search` and the claim `relevance_rate` are null when no relevant judgment was made, including a round that is entirely irrelevant.
- A storyline statement that cites a passage the request did not send is flagged `storyline_unsent_passage`. A statement with no relationship is flagged `storyline_no_relationship`. Packing drops an uncited passage before a relationship's own passage.
- `memo_quote_unsupported` fires only when the memo shares no meaningful token with the quote. A memo that says a partial row is supported is `memo_overclaims_support`. A memo that contradicts the relationship's own rationale is `memo_contradicts_relationship`. Memos are asked to state whether the evidence is two-sided or partial. `memo_accounting.produced` counts memos written in this run. `preexisting` is separate.
- An omitted planning seed is derived from the corpus hash and recorded as `seed_source`. An explicit seed, including 0, is kept. `pair_coverage_statement` reports examined pairs over the unordered pair count. The same seed examines the same pairs.
- A focused queue that finished is stage status `complete`. `complete_for_queue` remains the run status when passages were never queued, and the stage reason is `queue_complete`.
- Temperature is set to 0 when the chat model allows it and it was unset. `model_temperature`, `model_seed`, and `model_determinism` are on the run. `unset` means the instance did not take a temperature.
- After a call reports provider input tokens, an over-prediction at or below 3 times is compared as raw times 1.10 divided by that ratio, and never above the raw heuristic. An under-prediction, the first call, and a ratio above 3 stay on the raw heuristic. `token_estimate_ratio` is the raw heuristic divided by provider input tokens. The 1.28 adjusted reading is still not the stopper.
- A saturation window below 3 documents is `not_interpretable`. `n_documents` is on the report and in the reviewer prompt. `new_code_rate` is not treated as a saturation claim on that window. `judgment` stays null.
- The synthetic counterexample text sent to the model does not announce itself. The label and the limitation stay in the audit. The injection only checks that a contradicting passage can be recorded. Research exports omit it.
- `policy_version` is `0.8.6`.

## 0.8.5

- A relationship quote anchors an endpoint only when it shares a content token with that category. `citation_anchoring_rate` is 1 only when both endpoints are anchored, and otherwise null. `endpoint_anchoring_rate` is the share of endpoints the quote names.
- The storyline sends the categories and relationships that fit `axial_token_budget`, and the passages those rows cite. Omitted ids are reported. Rejected and merged relationships are left out, and a statement that cites only those rows is flagged. `storyline.json` includes `core_id` and `core_provenance`. An empty core is `unset`.
- When memos and the storyline are both requested, one selective call is reserved for the storyline. A memo that names a category the quote does not name is kept and flagged `memo_quote_unsupported`.
- A combined axial-then-selective run keeps the coding stage on `provider_budget` and adds the selective calls to the same counter.
- `relationships_with_negative_case_search` counts relationships that received a counterexample judgment, over all relationships. An axial supports, contradicts, or qualifies verdict whose rationale says the passage is irrelevant is not a negative case.
- `code_frequency_shift` is the total variation between the last two documents. 0 is the same mix and 1 is maximal change. The saturation prompt uses that name. A model assessment stays a model judgment, and `judgment` stays null. A zero property or relationship rate under partial pair coverage is `undefined_partial_coverage`.
- Obvious pairs rotate with the seed. `pair_coverage` reports how many pairs were examined. Selective calls use their own prompt and the stage label `selective`.
- Focused claim tests require a subject stem when the claim has one. Reallocation warnings use the calls that were moved. Focused runs record `stage_outcomes` and a focused stage on `provider_budget`. An automated saturation resume is stored with actor `automated`.
- `policy_version` is `0.8.5`.

## 0.8.4

- A proposal request is one graph step. The response and the allowance it consumed are checkpointed before the next request. A timeout keeps those responses and counts the timed-out attempt, so a resume cannot spend the ceiling again.
- Memo completion counts requested, produced, omitted, failed, and empty memos. A failed chunk stays unresolved. Empty or missing memo text is `incomplete`, including a schema-valid empty memo.
- Provider usage is counted once per received response. A repair that was not sent does not carry the original response's usage, and its warning says the repair was not sent. A decision-review response counts whether or not it parses.
- A proposal is deferred when either category has no passage in the request. The deferred row names the unsupported category ids.
- The fresh-process checkpoint test derives the checkout path from the test file.
- `policy_version` is `0.8.4`.

## 0.8.3

- Pausing for human review does not reset the provider allowance. Consumed calls and tokens are checkpoint state (`provider_budget`) and are restored into the runtime before a resumed node runs, including a new process. Axial, focused, and selective graphs all restore that state. Decision-review counters travel with it.
- Coding and review have separate call ceilings on one shared token allowance. A schema repair or retry spends the stage that sent the original request. `decision_mode` does not give coding extra calls.
- A run is `complete` only when every requested stage did its work. Stages are proposals, counterexamples, decision review, selective core, memos, and storyline. A ceiling stop is `budget_limited`. A preserved parse failure is `incomplete`. Per-stage status is on the run JSON.
- Every provider response is counted once, including a repair that also fails to parse. The failed repair keeps its own response and usage.
- A proposal with no segment evidence is deferred as `missing_category_evidence`. An empty evidence list does not raise inside packing.
- `policy_version` is `0.8.3`.

## 0.8.2

- A paraphrased contradiction stays a negative case. "is about" and "about the" are not irrelevance cues. Missing subject-stem overlap is `claim_subject_uncertain` and still disputes. Only explicit irrelevance language (`irrelevant`, `unrelated`, `does not address`, `does not mention`, `nothing to do with`) flags `rationale_indicates_irrelevance`, and a negation such as "not unrelated" does not. That negative case is kept as `disputed_pending_review` instead of being dropped.
- `max_axial_review_tokens` is one shared estimated-token allowance for axial coding, counterexamples, memos, the storyline, and decision review. A later selective run continues from the tokens already estimated. `max_selective_review_calls` remains a separate call ceiling. Focused coding keeps `max_focused_review_tokens` separate from `max_decision_tokens`.
- Schema repair and retries acquire that same call and token allowance and are their own audit rows. When no allowance remains, the failed output is kept and flagged for review.
- An oversized batch is shrunk until it fits. A proposal no longer drops the last passage for a category. If both sides of a pair cannot fit, the pair is deferred and the reason is recorded, or the passages are excerpts with a source offset and length.
- A finished graph is not analytical coverage. Pairs are reconciled into planned, attempted, completed, omitted, and remaining. `unexamined_pairs` lists pairs that were not sent. A run that stops on a token or call ceiling is `budget_limited`. A preserved parse failure is `incomplete`.
- Total variation between consecutive documents describes the difference between those documents. A low figure does not establish theoretical saturation and is not a human confirmation.
- `policy_version` is `0.8.2`.

## 0.8.1

- Code-frequency stabilization is the total variation between consecutive documents' normalized assignment counts. A document that assigns a code 99 times and another once is not the same distribution as the reverse. The figure is undefined with fewer than two documents, or when a compared document has no assignments. A 0.8.0 report stored presence bits and is not comparable.
- `axial_code(..., also_selective=True)` keeps the axial prompt log, warnings, saturation assessment, and procedure. Selective export writes `selective_run.json` and does not rewrite `category_relationships.json`, `thematic_saturation.json`, `theory_coverage.json`, or `axial_run.json`.
- Propose and counterexample allowances are model calls. Required proposal calls are `ceil(pair_count / batch_size)`. Only unused calls move to the counterexample share, and that share's pair capacity is calls times batch size. The split is never negative. `max_decision_calls=0` disables reviewer calls instead of clamping to one.
- The storyline prompt uses the core selected for the decision mode. An LLM choice is `model_core_id` with provenance `model`. An id that is not a candidate is not passed as the core.
- Proposal evidence is shared across the pairs in the request and both categories before any category fills the rest of `axial_evidence_limit`. A passage keeps every category id it illustrates. Passages that do not fit are `omitted_evidence`.
- An httpx client built with `ProviderClientFactory` keeps its auth, cookies, params, proxies, TLS, and `trust_env` when another loop needs a client. Two factories stay two clients. A client that was not built by the factory and is bound to another loop raises `CrossLoopClientError` instead of being replaced by a stripped client.
- Axial and selective calls use `max_axial_review_tokens` (default 24000) and `axial_token_budget` (default 6000). Every proposal, counterexample, memo, storyline, and decision-review call records provider input, output, and cached tokens. A request that does not fit is omitted and the reason is on the prompt log.
- A stall before the first model call, including a slot wait, a budget wait, or an embedding call, is under the same monotonic deadline as the request timeout. The deadline thread is not armed only after `ainvoke` starts.
- A `contradicts` or `qualifies` verdict is `claim_test_invalid` with reason `rationale_indicates_irrelevance` when the quote and rationale do not share the claim's subject and the rationale says the passage is about something else. That row does not dispute the claim. Missing overlap without that cue stays a dispute and is flagged `claim_subject_uncertain`. A claim with no subject stems stays uncertain.
- v25 claim-round relevance was about 0.03 (30 of 31 judgments irrelevant after the filter had already skipped 11). The bar is unchanged: two content tokens, a stored embedding at or above 0.55, or counterevidence. The rate is reported. Pairs below the bar are still not sent.
- `reread_reallocation.reallocated` is true when claim calls are actually moved, matching the reallocation warning. Reallocation does not queue a prompt that would pass the token allowance; calls left behind are `token_reallocation` with reason `token_allowance`.
- v25 revise and drop proposals stayed rare and unstable (one drop and two different revisions of R023_q2). `cross_seed_conflicts` already records that. No procedure change. Property-name stability on that pass was about 0.04. The score stays as measured. LLM decision mode was not in the live run. No change.
- `policy_version` is `0.8.1`.

## 0.8.0

- Axial coding is a separate graph: `load_focused`, `plan_relations`, `propose_relations`, `relation_checkpoint`, `counterexample_worker`, `commit_relations`, `saturation_indicators`, `saturation_checkpoint`, `export_axial`. It is not a node on open coding, themes, or focused coding.
- A relationship is a proposal between two categories. The default lens is the 1990 paradigm: causal conditions, context, intervening conditions, action/interaction strategies, and consequences. An empty slot is flagged `paradigm_slot_empty` and left empty. Each row cites a passage. The quote is anchored or the row stays `unanchored`.
- `category_relationships.json` stores the graph, provenance, `evidence_condition`, negative cases, and flags. A human resume may accept, revise, reject, or merge. A merge keeps both rows. The kept label is the target statement, or the clearer `label` on the resume. `{}` is not a resume value. `claim_support` stays undefined until a researcher sets it.
- Counterexample search uses the claim-compete bar: two content tokens, a stored embedding at or above 0.55, or a hedge or exception that also shares a token. One token is skipped and logged. Verdicts are supports, contradicts, qualifies, and irrelevant. A quote is required only for the first three. A valid contradicts or qualifies sets `evidence_condition` to `disputed` and keeps the row. `axial_synthetic_counterexample` injects one labeled passage and omits it from research exports.
- `decision_mode="llm"` does not interrupt. The reviewer writes `model_disposition`, `model_paradigm_role`, `model_saturation`, and `model_core_id`, and an audit row (`kind` `decision_review`, `actor` `llm`). Human fields stay unset. A model saturation assessment has `model_judgment` true. It is not a human confirmation. `judgment` stays null.
- Thematic saturation reports new-code, new-property, and new-relationship rates per document (batch size 1), code-frequency shift, and property and relationship stability when two or more runs are supplied. Fewer than two documents leaves frequency stabilization undefined. One run leaves cross-run stability undefined. A researcher may record `not_claimed`, `sufficient_for_this_question`, or `not_saturated` with a rationale. There is no boolean true.
- Selective coding is its own graph and can run alone: core candidates with no automatic winner, optional theoretical memos, and an optional storyline. A storyline statement without an anchored quote is kept and flagged `storyline_uncited`. `theory_coverage.json` reports the citation anchoring rate, the share of relationships with a negative-case search, disputed relationships, and unsupported statements. Each rate is null when there is nothing to score.
- `max_axial_review_calls` is split across propose and counterexample shares. Unspent propose calls move to the counterexample share. `max_selective_review_calls` is a separate ceiling. `batch_size`, `axial_request_timeout`, and retries are performance settings. A timeout uses the 0.7.9 monotonic deadline and returns an incomplete result unless `raise_on_timeout=True`.
- `policy_version` is `0.8.0`.

## 0.7.9

- A provider that holds a connection open is cut off by a dedicated thread waiting on a monotonic clock. The timer is not `future.result` and not `asyncio.sleep`, so it does not depend on the call's future or on either event loop. The thread waits 0.2s past the request timeout so a live loop can cancel the provider first and raise that request's own timeout. A blocked loop never runs that cancel, and the thread then abandons the model loop and unblocks the waiter. The error still names the request timeout. `process_isolation` still terminates a picklable child. Slot waits, budget waits, and shared embedding futures are bounded. httpx clients the SDK builds during the call get a socket timeout slightly past that deadline.
- A timeout returns the incomplete result (`status` `incomplete`, `stop` `timeout`, `timeout_step`, partial export when an output directory is set). `raise_on_timeout=True` raises `TimeoutError` as before.
- `focused_calls_used` counts calls written to the prompt log. A job that was only prepared is not counted. `focused_not_reread` lists queued passages that were not committed, including the passage in flight when the timeout fired.
- A claim call requires overlap with the claim's evidence (pole, text, or quote), not only the code: two content tokens, or a stored embedding at or above 0.55. Opposite-pole wording and hedges or exceptions are preferred. Pairs with no overlap or only weak overlap are skipped and logged. Unspent claim calls are reallocated to compete or outside. `claim_test_relevance` reports queued, skipped, and `relevance_rate`.
- An irrelevant judgment with an empty quote is valid. A quote is required only for supports, contradicts, or qualifies.
- `focused_synthetic_counterexample` injects one passage labeled `[synthetic counterexample]` so a live eval can check detection. It is not research data and is omitted from research exports.
- v24 compete rounds were mostly keep-only (4 of 6 passes). The two revise passes both revised R023_q2 and agreed (stability 1.0). `compete_keep_only` stays flagged. No procedure change.
- Property stability on that eval was low (name 0.10, category 0.25) because runs named different properties. The score stays as measured. No procedure change.
- The token heuristic is unchanged. 1.28 is the v23 condition (heuristic / provider total tokens about 1.27–1.29). v24 measured heuristic / provider input tokens at about 1.38, and the adjusted figure was reported within 0.3% of those input tokens. Neither number is a general calibration.
- Offline fixtures cover `decision_stop_detail`, `omitted_context`, `degenerate_absence_pole`, and `possible_direction_mismatch`. No procedure change.
- `policy_version` is `0.7.9`.

## 0.7.8

- A claim-test response is accepted only for the requested `(passage_id, claim_id)` pair, and only when the quote anchors in that passage. An unrequested pair or an unanchored quote is kept in the prompt log as `claim_test_invalid` with reason `unrequested_pair` or `unanchored_quote`. It is not a negative case and it does not dispute the claim.
- A provider that blocks the model-loop thread times out, and that loop is abandoned so the next call is not stuck behind it. The blocked thread is not killed. Hard termination is `process_isolation` (a picklable provider in a subprocess). Pooled httpx clients are constructed on the loop that uses them. httpx 0.28 does not expose the pool's loop, so ownership is explicit.
- A shortfall pole versus a passage is `possible_direction_mismatch`, a diagnostic that asks for inspection. It is not automatic `claim_test_invalid`. A paraphrase such as `visits too rarely` is not a mismatch. A shortfall phrase in a clause about something else does not suppress the diagnostic.
- LLM claim review packs against `min(decision_token_budget, remaining allowance)`. One claim that exceeds the request budget is `request_budget`. One claim that exceeds what remains is `token_allowance`. A batch that is only too big together is shrunk.
- `claims_payload` includes passages from the claim, its citations, negative cases, and claim tests, deduped, within `evidence_char_budget`. Omitted passages are `omitted_context`.
- `focused_tokens_estimated` is a character heuristic, not a guaranteed ceiling. `focused_tokens_adjusted` divides it by 1.28, an empirical adjustment from focused-coding live runs where that heuristic over provider total tokens was about 1.27–1.29. It is not a general calibration and it is not the stopper. `token_estimate_ratio` compares the heuristic to provider input tokens. Input, output, and cached usage stay separate.
- Revise and drop proposals stay unapplied. A run that does not interrupt can leave a disputed claim pending.
- `policy_version` is `0.7.8`.

## 0.7.7

- A provider that accepts a connection and never sends a body is cut off at the request timeout, on the shared model loop. The deadline thread is not the package loop's executor. Pooled async clients get connect, read, write, and pool timeouts on that loop, and a transport still bound to another loop is replaced so the read is not waited where the socket is not polled. `asyncio.wait` cancels the provider task when the clock expires. A timeout is not retried unless the wall budget can hold another full request timeout. The error keeps the attempt's own duration. A retry that still fails also names the attempt count and that shared deadline. Python 3.11 aliases `concurrent.futures.TimeoutError` to `TimeoutError`; a finished call timeout is no longer rewritten as the sync-bridge budget. The graph records status `incomplete`, stop `timeout`, `timeout_step`, and writes a partial export when an output directory is set. A watched step does not retry its own node timeout.
- A claim test requires shared content or an embedding match even when the passage carries the claim's code. An opposite pole or a negative-case hit can still qualify. An empty quote is not sent. Skips are logged. Unspent claim calls are reallocated. `claim_test_relevance` reports queued, skipped, judgments, and `relevance_rate` (null when nothing was judged).
- `supports` is `claim_test_invalid` when the pole states a shortfall (`needs to`, `should`, `more`, and the same class) and the passage does not. The judgment is kept and flagged. It does not set `claim_support`.
- `proposal_stability` keys are `(code_id, passage_id, property_key(definition))`. An empty definition matches a revise of the same code and passage, so drop versus revise is comparable. `cross_seed_conflicts` lists kind or definition disagreements. A segment list is not scored as if every segment were a proposal. A run folder or `coded_segments.json` / `focused_run.json` is read. Anything else raises.
- Property stability was low because runs named different properties (about 10–20 per pass), not because `property_key` ignored a shared wording. The score stays as measured. `category_counts`, `name_counts`, and `properties_per_run` are reported. A low name-level score with many names says the runs diverged. It is not claim support.
- `focused_tokens_estimated` stays the three-characters-per-token ceiling that stops the reread. `focused_tokens_calibrated` divides that ceiling by 1.28, near the live estimate-to-provider ratio, and is not the stopper.
- LLM decision mode was already covered offline for both checkpoints, with model provenance kept off human `claim_support`. No procedure change.
- `policy_version` is `0.7.7`.

## 0.7.6

- Timed model calls share one process-wide loop. A pooled httpx or xAI client stays bound to that loop, so the second call does not raise `Event is bound to a different event loop`. Callers do not keep the client alive. The wait is `future.result(timeout)` on a side thread. A graph step that calls the model has a watchdog (two request timeouts plus two seconds for a recode step, so a label retry still fits; an LLM checkpoint gets its own call allowance). A timeout still raises `TimeoutError`, and the checkpoint records status `incomplete` with stop `timeout`. A timeout of 0 or less still waits on the caller's loop. The sync bridge remains the whole allowance.
- A claim is tested only against a passage that claim retrieved: the same category, an opposite substantive pole, a content-token or embedding match, or a negative-case hit. `contradicts` is only when the passage asserts the opposite. Absence of mention is `irrelevant`. A dimension pole that is an absence (`no mention of ...`) is flagged `degenerate_absence_pole`, kept, and not used to generate a test. Only `contradicts` or `qualifies` from a relevant test becomes a disputed negative case. Any other such judgment is stored as `claim_test_invalid`.
- `claim_test_counts` accumulates every judgment in the run. A later commit no longer replaces the dict with the last call.
- `proposal_stability` is undefined when there are no proposals to compare. An empty pair is not scored as 1.
- A compete round that returns only `keep` sets `compete_keep_only` and a warning. Exclusion and consolidation do not drop a non-conflicting revise or drop. Those proposals are stored before the alternative check. The prompt says `keep` is not the default. A live run that still returns only `keep` is model behavior, and the flag records it.
- `property_key` also drops function words (`during`, `when`, `this`, and the same class). A low name-level score means the remaining words differ. The score is reported. It is not claim support.
- `focused_tokens_estimated` stays a ceiling of about three characters per token. It is not calibrated down to provider usage. `token_estimate_ratio` is that ceiling divided by provider `total_tokens` when the provider reported tokens, and null otherwise. Both numbers are reported.
- `policy_version` is `0.7.6`.

## 0.7.5

- Claim compete uses a dedicated claim-test prompt and schema. The request includes that property or dimension claim (category, name, text, pole, and cited quote) and asks whether the passage supports, contradicts, qualifies, or is irrelevant to it. contradicts and qualifies are negative cases: flagged, not applied, and `evidence_condition` becomes `disputed`. supports and irrelevant are not code changes. `claim_support` stays undefined until a researcher sets it. A call reallocated to outside or compete does not use this schema.
- A timed model call runs on a private thread and event loop. The provider client gets connect, read, write, and pool timeouts. A synchronous socket read inside `ainvoke` no longer blocks the package loop, so `focused_request_timeout` still fires. A timeout of 0 or less still waits with no deadline. The sync bridge stays a budget for the whole allowance, not one request.
- `strongest_alternative` cannot name a kept code or a code already assigned on the passage. The prompt lists `allowed_alternatives` and `excluded_alternative_code_ids`. A `code_id` and a label that name different codes, or a label that normalizes to an excluded code, are invalid. `alternative_named_not_chosen` is still only when `alternative_fit` is `stronger` for a different code and a raised code was kept.
- Revise and drop proposals are consolidated per code for the run. Both a revise and a drop, or two different revised definitions, is `proposal_conflict`. The proposals stay unapplied and the conflict is excluded. `actor` stays `model` in human and LLM decision mode. `proposal_stability` compares those proposals across runs. One run leaves both scores empty.
- Span splitting caches the structured-output schema token count, so a 100,000-character paragraph does not rebuild that schema on every slice.
- `policy_version` is `0.7.5`.

## 0.7.4

- Claim compete no longer depends on a pole string appearing in the passage. It searches the property's category on other passages, opposite dimension poles, lexical similarity and any stored embedding of the property text, then low-confidence or flagged assignments. If passages are still unreread, those fill the remaining slots. If nothing is left to reread, the unused calls are sent as outside or compete calls.
- A reserved purpose that was not given to another purpose is reported as unused. A purpose that was given calls, and those calls were sent, is reallocated. The warning denominator is `max_focused_review_calls`. `purpose_coverage.reallocation.reallocated` is true only when a transfer happened.
- `focused_request_timeout` is enforced on the model wait, on `run_sync` futures, and on the package event-loop bridge. The wait cancels the task and raises `TimeoutError` without waiting for a provider that swallows cancellation.
- `strongest_alternative` that names the kept code is flagged `alternative_is_kept_code` and recorded as invalid. Compete passages now require `alternative_fit` (`weaker`, `comparable`, or `stronger`). `alternative_named_not_chosen` is raised only when that fit is `stronger` for a different code than the one kept.
- `policy_version` is `0.7.4`.

## 0.7.3

- `focused_share_claim_compete` reserves calls for a compete round that runs after properties are written. The four shares sum to 1. The default is one quarter each. A three-purpose split that already sums to 1 keeps this share at 0. `purpose_coverage` reports claim compete the same way as the other purposes. Setting the share to 0 is what skips the round.
- Compete passages use a separate output. Every initial code needs a `keep`, `revise`, or `drop` verdict with a rationale, and every passage needs `strongest_alternative`. Revise and drop stay flagged proposals and are not applied. The compete prompt lists codes on equal footing, alternatives first, and does not mark which were raised. A kept raised code is flagged `alternative_named_not_chosen` when the memo or the alternative names a different code.
- `focused_tokens_used` is provider-reported input, cached, and output when the response includes them. `focused_tokens_estimated` is the pre-call ceiling that stops the reread. The estimate is not provider usage.
- Properties that share a category and a normalized `property_key` merge, keeping every citation. `property_stability` reports category-level and name-level overlap across runs. One run leaves those scores empty and records the keys.
- `policy_version` is `0.7.3`.

## 0.7.2

- Each focused reread queue is sized to its reserved share of `max_focused_review_calls`. `focused_neighbor_limit`, `focused_outside_limit`, and `focused_negative_case_limit` are optional caps (`None` by default, `0` adds none). They are not a default of 1. `purpose_coverage.applied_share` is the share of calls actually used. A purpose that receives fewer calls than it reserved is marked reallocated, with a warning.
- Status `complete` means every corpus passage was reread. Status `complete_for_queue` means the queue finished and is not corpus coverage. Passages never queued are `focused_not_queued`. Passages queued and not sent stay `focused_not_reread`.
- Compete can run before property claims exist. Seeds are a different valence or consequence, a low-confidence or flagged assignment, another code on the passage, or an opposite pole. One later round searches claims written by the first reread. The prompt asks for a disconfirming reading. `revise` and `drop` are flagged proposals and are not applied without review.
- The recode prompt includes the full codebook and says to prefer an existing code. A new code with lexical similarity at or above `focused_duplicate_similarity` (default 0.5) is flagged `possible_duplicate`.
- A property whose category is not assigned on the cited passage is flagged `property_category_mismatch`. `category_match` is reported beside citation anchoring. Anchoring is not category fit and not claim support.
- A different span is nested on the original assignment. It is not a second row in assignment or downstream views. A quote that drops words of the original is flagged `narrowed_span`.
- `focused_request_timeout` (default 120 seconds) is a wall-clock budget for one model call, including streaming and retries. `focused_request_retries` caps retries inside that budget.
- `seed` orders the queue. When the candidates fit the allowance, it does not change membership. `focused_sampling_seed` chooses membership only when a candidate list is longer than its slot.
- `policy_version` is `0.7.2`.

## 0.7.1

- Focused selection no longer raises every code when scores tie at a median cutoff. The default rule is `top_k` (the upper half, ties broken by `code_id`). `focused_selection_rule=quantile` uses the same ceiling. The score rule does not raise every code unless `focused_raise_all` is set. An explicit `focused_min_score` stays a hard threshold. The result warns when the raised fraction reaches `focused_raise_warn_fraction` (default 0.75).
- An empty `compete` queue is filled from assigned passages: another code on the passage, an opposite pole, or a raised code not yet applied. An empty `outside` queue is filled from passages not assigned to raised codes, then from the lowest-confidence raised assignments. A purpose that is still empty has its share reallocated explicitly, and the result warning and `purpose_coverage` say so.
- Loading an open-coding folder restores segment, memo, highlight, and op id counters. New memos use `m00001`. A focused reread does not mint `s00001` again.
- Re-applying a code on the same span records a confirmation on the existing assignment, with focused provenance. A different span is a separate `focused_span_proposal` and does not rewrite the stored quote.
- The focused prompt asks for analytic properties and dimensions, cited quotes, and competing interpretations. It asks the model not to substitute a descriptive summary.
- The focused output folder writes `coded_segments.json`, `assignments.json`, and `prompt_log.json` (parsed response and raw model text), along with `focused_claims.csv` and `downstream_claims.json`.
- `policy_version` is `0.7.1`.

## 0.7.0

- Focused coding is a separate graph, `get_graph("focused")`: `load_open_coding`, `select_candidates`, `researcher_checkpoint`, `prepare_recode`, `recode_worker`, `commit_focused`, `review_recode`, `export_focused`. It reads a finished open-coding run. It is not inserted into `open_code` or `themes`. `build_themes` still does not reread passages.
- `select_candidates` ranks codes by assignment frequency, document and respondent spread, and analytic weight from substantive memos, the in-vivo flag, and stored model comparisons. A researcher pin is proposed even when the score is low. A rare contrast, the only code on one pole of a recorded dimension, stays eligible. The result is a proposal. Nothing is deleted from the codebook.
- The researcher checkpoint calls `langgraph.types.interrupt()` inside `researcher_checkpoint`. Resume with `Command(resume=...)` through `focused_code(..., resuming=True, resume=...)`. The default checkpointer is `InMemorySaver`. `read_focused_interrupt` and `agt-focused interrupt` show the payload. `agt-focused resume` continues a sqlite checkpoint. Resume checks the stored manifest of corpus, source, context, model settings, and analytic procedure. Corpus identity and `seed` are not continuable. Other changes need `allow_changes` or `continuation`.
- `batch_size` (default 1), `concurrency`, and `token_budget` are performance settings in their own manifest block. They do not change the selection score.
- The reread keeps quotes that are not in the target passage. A quote found in another passage is attributed there and flagged `evidence_outside_passage`. A quote found nowhere is flagged and is not assigned to a different passage. Nothing is dropped. An initial assignment the reread does not repeat is kept and flagged `focused_disagreement`.
- Properties and dimensions go to `category_development.json`, each with a document id, passage id, and quote. Unsupported citations are flagged and kept. They are not theme properties and not relations.
- `max_focused_review_calls` (default 8) and `max_focused_review_tokens` (default 24000) bound the reread. Calls are checked first. An exhausted allowance sets `status` to `incomplete` and names the stop (`call_allowance`, `token_allowance`, or `no_progress`). Passages not reread are `focused_not_reread`. `judgment` is not set.
- `policy_version` is `0.7.0`.
- The focused reread splits its call and token ceiling across three reserved purposes, `develop`, `compete`, and `outside`, interleaved by `focused_share_develop`, `focused_share_compete`, and `focused_share_outside` (default one third each). `purpose_coverage` reports each purpose. Researcher incidents from the resume value or `focused_incidents.csv` can enter any purpose. Negative-case search includes passages inside a claim's supporting category.
- Each property and dimension keeps `disposition`, `evidence_condition`, `downstream_eligible`, `claim_support` (undefined until a researcher reviews), `category_revision_id`, and `evidence_revision_id`. Citation anchoring is reported separately from claim support. A category edit marks dependents `needs_review` and keeps history. `code_id_sidecar.json` distinguishes a Taguette rename from a removal plus an addition. `category_development.json` is versioned and `relations` may be null.
- `claim_checkpoint` is a second `interrupt()`. Resume claims with `{"leave_claims": true}`. An empty object is not a resume value.
- `focused_analytic_features` drops memo count, the in-vivo flag, or stored comparisons. Memo count can reward verbosity.
- `scripts/evaluate_focused.py` compares initial coding with focused rereading offline and estimates a live run. It does not call an API.
- `nonsubstantive_kind` treats Unicode letters and numbers as substantive text. `我辞职了`, `Я уволился`, `تركت العمل`, and `I quit.` are not `no_substantive_text`.

## 0.6.16

- A coding action whose confidence is below `review_below` (default 0.5) is applied and flagged `needs_review` with reason `low_confidence`. The same decision stays on `review.json`. The confidence score is the model's own diagnostic. It is not a probability and not a calibrated frequency. In-vivo flags, definition rewrites, theme assignments, and merges below their own gates are still not applied.
- Theme prompts agree with the code. A code may belong to more than one theme. The first theme in the proposal is primary (`primary_codes`). `max_theme_share` is a soft presentation target: the review prompt says exceeding it is noted, and the code only warns. The theme-report protocol id is `theme-report-1`.
- A merge keeps the label the model judged clearer (the target). A trivial merge, a missing judgment, or conflicting directions inside one component use a lexicographic tie-break. Aliases, history, and segment provenance stay on the survivor. A definition aimed at a retired code does not replace the survivor's wording.
- The README, theme prompts, and docs describe `build_themes` as a descriptive grouping for reporting. It does not reread the passages. The package is automated initial (open) coding plus that theme report. It does not do theoretical sampling, axial coding, selective coding, or a saturation decision. Concurrency does not speed up a default run, whose window is one passage. A faster run sets `passages_per_request` or `passages_per_codebook_update` above 1, or uses wave mode, which is a constant-comparison tradeoff. The speed table is marked as a simulated fake-coder benchmark. Accuracy against human coding has not been measured live for 0.6.x. The last live numbers are from 0.5.4. `pipeline=True` codes the next step against the pre-merge snapshot, one step of lag.
- Uncertainty, refusal, `n/a`, and punctuation-only replies are coded when `code_nonsubstantive=True` (the default) and flagged `nonsubstantive` with `nonsubstantive_kind`, so they can be filtered. The coding prompt asks the model to use the question, for example "expressing uncertainty about pay" or "declining to answer about staffing". Only empty or whitespace text stays excluded. `code_nonsubstantive=False` restores the old exclusion.
- Analytic memos are free-form. A comparison, a property or dimension, and a question are useful and not required. About 20 to 400 words is a comfortable range. A memo is never dropped for failing that shape. Under 20 words it is flagged `thin` and kept. A placeholder ("none", "n/a", a sentence that only says nothing was recorded) is still omitted. The prompt asks for a memo when something analytically interesting happens.
- `policy_version` is `0.6.16`.

## 0.6.15

- After the final consolidation pass, codes that still have unsent assignments are reviewed again. Each follow-up pass sends another batch inside the prompt example limit (2) and only includes codes `code_needs_review` still marks. The loop stops when every current assignment has been sent, when `max_final_review_calls` follow-up sweep calls have been used (default 8), when the next prompt's estimated input would pass `max_final_review_tokens` (default 24000), or when a pass sends no new assignment id. A prompt that does not fit is not sent. Zero for either allowance keeps the consolidation batch and then stops. The call allowance is checked first.
- If that loop stops with evidence still unsent, the run is not fully consolidated. Warnings name the pending codes and ids. Metadata records `review_complete` false, `review_stop` (`call_allowance`, `token_allowance`, or `no_progress`), `pending_counts`, `pending_assignment_ids`, `assignments_available`, `assignments_sent`, `assignments_pending`, `review_calls_used`, and `review_tokens_used`. `consolidated` is true only when a consolidation pass ran and `review_complete` is true. A consolidation row in the sweep log means that pass ran. It does not by itself mean every assignment was sent.
- An embedding that succeeds is published to its shared future as soon as the vector arrives, before the cache stores it and before a later vector can evict it. A follower waiting on that text receives the vector even when `vector_limit` is 1, a sibling is stored afterwards, and another sibling raises `ConnectionError`. The batch-query, single-query, and document paths all do this.
- `policy_version` is `0.6.15`.

## 0.6.14

- A speaker label may include a number (`Participant 02:`) or initials (`J.D.:`, `J. D.:`). `Table 1:`, `Figure 2:`, `Chapter 3:`, `e.g.:`, and `U.S.:` are still not speakers. A known role at the end of a line-start label still shrinks to that role, and a mid-line label still has to follow a sentence boundary. In an incident transcript, `Participant 02`'s answer is its own passage. It is not folded into the interviewer question.
- Source-text coverage compares every non-whitespace character of the document with passage spans and question context. It does not ask the speaker parser which regions are answers. Text that sits only in question context and looks like a speaker turn is flagged. Any other uncovered text is warned and stored (`source_uncovered`, `question_context_turns`).
- A quote found in another document in the same request keeps that document's id, name, and respondent metadata on the segment. The passage and document that were being coded stay on provenance (`coded_passage_id`, `coded_doc_id`, `coded_doc_name`, `coded_documents`). Frequencies, respondent analysis, and the highlight export use the located document.
- Every incoming assignment is checked for the same code, source document, and matched spans. The check is not limited to cross-passage quotes, and it does not treat equal text as the same evidence. The same words at a different offset stay a second quote. Provenance from both codings is merged onto one assignment. Coding the passage that holds the quote first, or the outside passage first, both leave one segment and frequency 1.
- An embedding request keeps the vectors it needs until it returns them. Eviction only drops what a later request can reuse. Two distinct queries in one request no longer raise `embedding cache missed a query it had claimed` when `vector_limit` is 1. The batch-query, single-query, and document paths all do this.
- New evidence is sent in batches within the prompt example limit, with new and contrasting quotes first. Each review records the assignment ids that were actually sent (`reviewed_assignment_ids`, `reviews_seen.sent_assignment_ids`) separately from the ids available then (`available_assignment_ids`). Evidence that was available but not sent stays pending and triggers another bounded review. A first review that shows 2 of 10 assignments does not mark the other 8 as reviewed. Sweeps use the same bound.
- The token cost of solo retries for a passage missing from a batch reply is on the run metrics: `missing_slot_retry_calls`, `missing_slot_retry_input_tokens`, `missing_slot_retry_output_tokens`, and `missing_slot_retry_tokens`. The batch call itself is not included.
- `short_turn_chars` defaults to 0. A length cap cannot tell filler such as `Ovid.` from a meaningful incident such as `I quit.`, so the merge stays off until that heuristic has been evaluated. When a caller turns it on, a short turn is still never merged across an interviewer question. The text is kept either way. A 0.6.13 checkpoint stored 60. Resume does not re-segment; pass 60, or allow a continuation, to resume it.
- `policy_version` is `0.6.14`.

## 0.6.13

- Speaker labels are recognized after leading whitespace and, for a short or known label, after a sentence boundary in the middle of a line. A RAND-style transcript (` Participant:` and `best.   Interviewer:`) no longer folds the participant's answer into the interviewer question, which left that answer out of every passage. Answer text that still falls outside every passage is warned and stored on the run metadata as `uncovered_answers`.
- A line-start label whose last word is a known role matches only that word. `download the results   Interviewer:` is `Interviewer:`, and `download the results` stays on the previous participant turn.
- Text before the first speaker label, such as `Participant 02`, is attached to the first passage. It is not dropped. The warning and `attached_preambles` (`doc_id`, `chars`, `text`) record it the same way `uncovered_answers` records a gap.
- A passage missing from a batch coding reply is requested again on its own, up to two times. Each retry is logged and bypasses the response cache, so a repeated solo prompt calls the model again. Passages that stay failed are listed on the run summary (`summary.failed_passages`) and in the warnings.
- Before a quote is declared unanchored, the other passages in the same request are searched. A hit is kept, flagged `evidence_outside_passage`, and attributed to the passage that contains it, the same way a neighbor quote is.
- When that cross-passage quote and code are already on the attributed passage, the new segment is merged into the existing one. The passage that was being coded is recorded on `provenance.coded_passage_ids`, so coverage is not counted twice. The coding prompt tells the model to quote only the target passage, not the previous context, the following context, or another passage.
- A speaker turn of at most `short_turn_chars` (default 60, for example `Participant: Ovid.`) is merged into the adjacent answer passage during segmentation. The text is not skipped and not dropped. Set `short_turn_chars` to 0 to keep every turn as its own passage. This changes passage boundaries for a new run. Resume does not re-segment.
- The document-mode coding prompt tells the model to code each passage separately and not to write one code that spans several passages.
- `policy_version` is `0.6.13`.

## 0.6.12

- Resuming a checkpoint still allows a continuation for changes that affect only the work left to do: research-context wording, the model and its temperature, and performance settings. Those write a continuation event. `review_mode` and `parallel_documents` are performance settings. They do not change how passages are segmented: the same corpus, unit, and seed produce the same passage ids, text, and offsets. Resume does not re-segment, so changing either one records a continuation and leaves the stored passages, coding order, queue, and branch queues as checkpointed. `review_mode` is still read when the next batch is taken from that stored queue. `parallel_documents` does not rebuild branch queues. Corpus identity, `unit`, and `seed` are rejected even with `allow_changes=True`. The error tells you to start a new run. There is no migration procedure yet. Adding a document and resuming no longer rewrites the manifest hash while the stored passages stay the original four.
- The manifest fingerprints the chat and embedding instances that will actually run: model id, provider, temperature, `top_p`, max tokens, embedding model and dimensions, and other non-secret settings. API keys are not included. Metadata records those values. `ConfigCoder(temperature=0)` and `ConfigCoder(temperature=1)` no longer compare equal, and the metadata temperature is the instance value rather than the unused `temperature=` argument.
- Codebook entries keep every assignment, not the first eight examples. Each review records the assignment ids it saw (`reviewed_assignment_ids`, and the list `reviews_seen`). The next prompt always includes evidence added since that review, then representative and contrasting quotes from the full history, bounded in size. Document diversity is applied before the remaining slots are filled in passage order. A ninth contradictory quote changes the fingerprint and appears in the next review prompt.
- A discontinuous quote is attributed from the matched spans, not from the bounding interval between them. `I resigned. ... I returned.` with a long passage in the gap is attributed to both end passages and kept flagged. It is not assigned to the gap. When the spans cannot be placed, attribution is unresolved and is not counted on the passage that was being coded. Coverage and agreement count a multi-attributed quote on each overlapped passage, and an unresolved quote on none.
- Muse unit alignment collapses whitespace and maps the match back to the original offsets, so an extra space is not a silent negative. A repeated excerpt uses `occurrence` or `order` (0-based) or an offset field when the annotation has one; otherwise it is ambiguous and is not aligned. The result reports `alignment_aligned`, `alignment_total`, and `alignment_coverage`. A unit touched by an unaligned or ambiguous annotation is left out of the score. Coverage below 0.8 makes the metric unavailable.
- Canonical embedding vectors take an optional LRU cap, `configure_embedding_cache(vector_limit=..., vector_bytes=...)`. `vector_bytes` counts 8 bytes per dimension. A vector a retained snapshot still references is not evicted. The default remains uncapped. `commit_passage` still shallow-detaches every codebook entry, because a later in-place edit must not rewrite the previous checkpoint, and it still copies the segment list. An append-only commit updates counts from the new segments plus the assignment list already on the entry. `sync_counts` still scans the segment list when a merge or split moves assignments.
- The README speed table adds the simulated row `wave with unique-text embeddings` (`--unique-passages`: wave mode, distinct passage text, embedding calls sleep 0.05s).
- `policy_version` is `0.6.12`.

## 0.6.11

- A quote that matches text outside the passage being coded is still kept. Textual status (exact, neighbor, document) is separate from assignment. Evidence that does not overlap the target passage is flagged `needs_review` with reason `evidence_outside_passage` and attributed to the passage that contains it. The segment records both `coded_passage_id` and `attributed_passage_id`. Coverage, agreement, adjusted Rand, and assignment counts use the attributed passage. The flag does not retry the quote and does not drop the code.
- Resuming a checkpoint compares a manifest of corpus identity, context, model config, procedure settings, and `policy_version`. A mismatch raises `CheckpointMismatch`. `allow_changes=True` or `continuation=True` records a continuation event (old and new values), warns, and continues. The result metadata is the settings that actually ran.
- Codebook embedding snapshots store content hashes of canonical immutable vectors, not a fresh vector list per code. Retention is an LRU cap (`configure_embedding_cache(snapshot_limit=...)`, default 8) and a run scope that drops that run's snapshots when the run ends. Growing a codebook from 1 to 50 codes keeps about 50 vectors, not the triangular 1,275 lists.
- Cancelling the owner of a query batch during sibling settlement no longer cancels a provider call another caller is waiting on. The shield covers that wait, not only the first `asyncio.wait`. A caller that joined an in-flight text and then failed on a different text detaches every waiter it registered, including one it never reached. Both apply to the single-query and batch-query paths. Document embeddings use the same per-batch ownership: a failed chunk does not drop a sibling that already succeeded, so a later request for that sibling is a cache hit.
- Nominal Krippendorff alpha is computed when the coincidence table supports it, including `[[1, 0]]` (0, not 1). With no paired ratings it returns no alpha and an explicit undefined flag. The aligned passage metric omits cells where both coders left the code off.
- The Muse benchmark unit grid takes human labels from annotation spans that overlap each passage and maps model labels through the reviewed mapping. If that alignment cannot be established, the metric is unavailable instead of scoring model positives against an empty human side.
- The review fingerprint hashes every example and assignment, plus `evidence_revision` when the codebook stored one. Adding a fourth example changes it. Frequency still does not. Prompts take a separate bounded, diverse example set.
- `commit_passage` no longer deep-copies the codebook or copies every segment. Entries are shallow-detached (nested history events stay shared) and counts are recomputed only for codes that commit touches. On a fat book of 400 codes and 3,200 segments, one additional assignment went from about 43 ms to under 1 ms. Growing 800 new codes went from about 11.4 s to about 0.9 s. Entries still have to be detached on each commit so a later in-place edit does not rewrite the previous checkpoint.
- `scripts/benchmark_review_modes.py` was remeasured on 0.6.11. The numbers are simulated: the fake coder sleeps 0.05 s per call. They are not a provider benchmark.
- `policy_version` is `0.6.11`.

## 0.6.10

- A failed query no longer drops in-flight cache entries for queries that are still running. `asyncio.gather` raised on the first error and left the other tasks going; the handler then removed their entries, so the next request started a second provider call. The one-by-one path and the batch-query path now wait until those siblings have finished before releasing ownership. An entry a task still owns is not cleared. A sibling another caller is already waiting on keeps running, and keeps its entry, when this caller is cancelled. After the failure returns, that group has nothing left running. An immediate request for a sibling that was still in flight does not embed it again.
- `scripts/benchmark_review_modes.py --unique-passages` gives the embedding row a distinct query text per passage. The default corpus reuses four stems, so that row embeds four queries. With the flag, the same 24 passages make 20 query calls: passages ranked before any code exists are not embedded, and the rest do not collapse in the cache. `--delay 0.05`:

  | embedding row | seconds | calls | passages/s | queries | peak |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | four reused stems | 0.887 | 11 | 27.07 | 4 | 3 |
  | `--unique-passages` | 1.019 | 11 | 23.55 | 20 | 4 |

- `policy_version` is `0.6.10`.

## 0.6.9

- A passage that does not fit in `token_budget` is split on whitespace without restarting the packer for each candidate slice. The splitter finalizes the slices, then those slices are ranked together. One paragraph of about 11,000 characters with `unit="paragraph"` and `token_budget=4000` completes, and so does a 100,000-character paragraph. The run used to raise `coding planner could not rank passage slices` after 64 restarts.
- Query embeddings take one slot on the shared embedding semaphore, with one timeout and one retry budget each. A query that succeeds is cached immediately, so a later retry does not embed it again. Providers that implement `aembed_queries` or `embed_queries` are called with the whole chunk. Four queries of 30 ms under a 75 ms timeout all finish, and each text is embedded once.
- `scripts/benchmark_review_modes.py` adds an embedding-latency row. It sleeps on each embedding call and sends passage text through the query API. `--embedding-delay 0` skips that row.
- `policy_version` is `0.6.9`.

## 0.6.8

- Code retrieval for a coding round is one async call per passage and codebook snapshot. Planning and coding share that ranking. Embedding calls go through the embedding semaphore, with the same retries and timeout as other embedding work, and they do not block the event loop. A passage is embedded as a query in both the sync and async retrievers, so it is not embedded a second time as a document.
- Validation feedback is capped, and a retry shortens it before dropping codes that already fit. A coding call, a validation retry, and a schema repair are measured again immediately before they are sent. A request that is still over `token_budget` after that trimming is sent. The run logs a warning and records an `oversized_request` flag on the run metadata and the review list, with the passage id and the estimated size.
- A sweep hint is sent only when both codes' definitions and examples are in that request. Partners that do not fit in the bounded neighborhood (`cross_chunk_limit`, default 8) are queued onto further bounded comparison requests. None of those pairs are dropped, and none are decided without the partner's evidence.
- `policy_version` is `0.6.8`.

## 0.6.7

- A rate-limit reservation larger than the bucket raises `OversizedReservation` instead of waiting until a provider timeout. The message names the bucket (`tokens_per_minute` or `requests_per_minute`) and `allow_oversized`. `allow_oversized=True` waits until the bucket is full, then drains it.
- In-flight embedding calls are dropped when the owner is cancelled, and other waiters shield the shared future so one cancellation does not cancel them. A later request for the same text embeds again. The cache key includes a fingerprint of output-affecting configuration (dimensions, endpoint, deployment, base URL, API version, and model kwargs). A vector whose width does not match the configured dimensions is not stored.
- `token_budget` (default 6000, was 4000) is the request that will be sent. Neighbors are retrieved and unioned across the passages in the call, then the messages, JSON schema, and a short repair allowance are counted, plus reserved output. A batch that does not fit is repacked into fewer passages, and retrieved codes are trimmed. A passage is split on whitespace only when the text itself still does not fit. A schema repair is measured again and shortened to the same budget. The default rose because the schema and the repair allowance are now inside the count.
- `aligned_assignment_agreement` aligns exact labels and spelling variants (case, whitespace, punctuation, plural, gerund). A negation does not align, so `trusting managers to share information` and `not trusting managers to share information` score 0. `mapping` is an explicit reviewed equivalence. `align_threshold` is an opt-in Jaccard fallback that still refuses a polarity flip.
- A sweep chunk no longer repeats the whole book's label index. Each chunk carries its own codes plus a bounded neighborhood (`cross_chunk_limit`, default 8). One compact label-only discovery call covers the rest of the book. User-message characters at chunk size 20:

  | codes | 0.6.4 | 0.6.5 repeated index | 0.6.7 | 0.6.7 + 8 cross-chunk codes |
  | ---: | ---: | ---: | ---: | ---: |
  | 20 | 9,347 | 5,467 | 5,467 | 5,467 |
  | 40 | 54,028 | 14,846 | 13,355 | 17,297 |
  | 80 | 178,806 | 36,922 | 26,577 | 34,437 |
  | 160 | 642,142 | 103,334 | 53,261 | 68,957 |
  | 320 | 2,423,837 | 324,781 | 106,812 | 138,180 |
  | 640 | 9,404,813 | 1,120,941 | 213,900 | 276,612 |
  | 1280 | 37,091,765 | 4,145,141 | 429,196 | 554,596 |

  From 160 to 640 codes the 0.6.7 payload grows 4.02x (four times the codes). From 20 to 1280 it grows 78.5x (64 times the codes). The repeated index grows 10.85x from 160 to 640. `scripts/benchmark_sweep_payload.py` prints the table. `policy_version` is `0.6.7`.

## 0.6.6

- Every model call shares one adaptive gate. `abatch` no longer opens a private semaphore, so two concurrent batches cannot exceed the in-flight ceiling. `max_in_flight` is the peak number of calls that held a slot. Coding, sweeps, and evaluation all use that gate.
- Coding time is no longer a batch duration copied onto each passage and then summed. `stage_seconds.coding` and `metrics.coding_wall_seconds` are wall time, with overlapping batches counted once. `metrics.coding_seconds` and `metrics.coding_request_latency_seconds` are per-request provider latency. `metrics.coding_queue_wait_seconds` is time spent waiting for a slot, a token bucket, or a retry. `coding_p50` and `coding_p95` use the per-request samples.
- A schema repair reports the failed response's token usage as well as the successful one. `usage.attempt_usage` lists each attempt when a call was repaired. The run's token totals include both.
- `tokens_per_minute` reserves estimated input tokens plus reserved output, then reconciles to the provider's actual usage. It is not a request counter. `requests_per_minute` consumes one unit per attempt. Both are recorded on the run metadata. Zero leaves that bucket open.

## 0.6.5

- Sweep requests no longer send every code's definition and examples with every chunk. A chunk carries full evidence for its own codes and for similar codes from other chunks (`cross_chunk_limit`, default 8). The rest of the book is ids and short labels. Later sweeps review only codes that are new or whose label, definition, or examples changed. Every `full_audit_every` sweeps (default 4), and the final consolidation pass, read the whole book. At chunk size 20 the old user payload is 9,347 / 54,028 / 178,806 / 642,142 characters for 20, 40, 80, and 160 codes (the review's 9.2k / 54k / 179k / 644k table). The compact request on the same codes is 5,467 / 14,846 / 36,922 / 103,334 characters, and 5,467 / 18,788 / 44,782 / 119,030 when every chunk also carries 8 full cross-chunk codes. `scripts/benchmark_sweep_payload.py` prints that table.
- Embedding vectors are cached by model and the sha256 of the exact text. Unchanged codebook rows are not embedded again. Concurrent workers and sweeps share one vector list for a snapshot.
- `token_budget` bounds the assembled coding request plus reserved output. A tokenizer is used when the model exposes `encode`; otherwise the estimate is characters / 3 over the full messages. A passage that does not fit is split, or sent alone with a warning. Request JSON is compact, and the first example quote is not repeated in a separate `example` field.

## 0.6.4

- Token overlap only proposes a comparison. Auto-merge is limited to spelling and inflection variants (case, punctuation, whitespace, plural and gerund forms) whose definitions are compatible. A negation or a polarity difference is never auto-merged. `trusting managers to share information` and `not trusting managers to share information` (Jaccard about 0.83, contradictory definitions) are proposed, not merged.
- The coding schedule is three settings: `passages_per_request`, `max_concurrent_requests`, and `passages_per_codebook_update`. Concurrency no longer widens the frozen codebook. `strict_sequential_settings()` and `procedure="strict_sequential"` read one passage at a time in document order, commit after every passage, and turn pipeline overlap and parallel documents off. `batch_size` and `concurrency` remain as aliases and warn when the old coupling would have changed the procedure.
- A cached pair decision is reopened when either code's definition text or revision changes, or when its examples or assignments change. Token overlap is not a threshold. Negating a definition (`Employees trust...` to `Employees do not trust...`) reopens the pair.
- Sweep prompts treat stance as context, not a merge veto, matching the coding prompt and the docs.
- The open-coding example for "I like my team" is `expressing liking for the team`. Automatic reading-order notes are `provenance`, not reflexivity memos. A reflexivity memo is still the one a sweep model writes.
- `adjusted_rand_index` stays. It compares passage partitions, so different singleton codes on every passage score 1.0. `aligned_assignment_agreement` scores the set of codes on each passage after label alignment (mean Jaccard, and Krippendorff's alpha on the aligned passage-by-code matrix).

## 0.6.3

- Semicolon-joined labels are split into separate candidate codes. Each piece is validated on its own, and a piece that matches an existing code is reused.
- An empty label, a label over the length limit, or a new label that bundles several concepts is retried once. The feedback names the problem. A bundle is a comma list of three or more concepts, or three or more concepts joined by "and". The retry asks for a single concept.
- If that retry is still empty, too long, or bundled, or if it returns no codes, the code is kept and flagged for review (`empty_code`, `code_too_long`, `bundled_concepts`). A blank label is stored as `(empty label)`. Generic labels are still rejected.
