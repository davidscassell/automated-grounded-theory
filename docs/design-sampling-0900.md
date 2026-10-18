# Theoretical sampling and a saturation decision (0.9.0)

`policy_version` was `0.9.0` for the first sampling graph. Package `policy_version` is `0.9.18`. The sections below record what each version added. The current sampling behavior is the whole file. The index is [README.md](README.md). This note is the design for the sampling graph. It does not replace [design-axial-0800.md](design-axial-0800.md). Axial coding and selective coding were already experimental graphs in 0.8.x. 0.9.0 does not rebuild them.

The README sentence this release retires said: "The package does not do theoretical sampling, axial coding, selective coding, or a saturation decision." That sentence was stale. The audit below is what was already true on 0.8.7 (`158bae3`), and what 0.9.0 adds.

## What already existed

| Item | On 0.8.7 | What 0.9.0 changes |
| --- | --- | --- |
| Axial coding | `get_graph("axial")`. Straussian roles `causal_condition`, `context`, `intervening_condition`, `action_interaction`, `consequence`, with `paradigm_version` `1990` or `2008`. `coding_lens` `charmaz` leaves those slots unused. Empty slots stay empty. Two-sided evidence, same-passage support, and cross-respondent patterns are separate. | Not rebuilt. |
| Selective coding | `get_graph("selective")`. Core candidates are scored and not auto-picked. Memos and a storyline are optional. `core_unsupported_by_graph` flags a core that is not a relationship endpoint. `core_stability` stays undefined and reports `n`. | Not rebuilt. The sampling directions and the stopping report reuse that core check. |
| Saturation | `thematic_saturation_report`. `judgment` is always null. Below 3 documents the window is `not_interpretable`. Window-wide rates are separate from the last document. The axial checkpoint can store `researcher_saturation` or `model_saturation`. There is no timestamp and no stopping-criteria report. | An explicit `saturation_decision` checkpoint. The researcher record has an actor and a UTC timestamp. The model record is `model_saturation_judgment` and is never copied into a researcher field. |
| Theoretical sampling | Not a stage. A seed shuffle is not sampling. [design-later-stages.md](design-later-stages.md) described a pursuit queue that had not been built. | `get_graph("sampling")`, `theoretical_sampling.json`, and a round ledger. |

## Sampling graph

`atheoretical_sample` / `theoretical_sample`. The graph is linear: `load_sampling_state` → `rank_directions` → `retrieve_passages` → `sampling_checkpoint` → `record_ledger` → `pursuit_reread` → `update_ledger` → `saturation_decision` → `export_sampling`.

`rank_directions` does not call a model. It reads the codebook, segments, properties, dimensions, relationships, empty paradigm slots, core flags, memos, and the saturation report. Directions, highest first:

- a candidate core that was not chosen, or a core that is not a relationship endpoint
- a category evidenced in fewer documents than `category_documents` (default 2)
- a category on a partial relationship
- a category with no contradict or qualify counterexample
- a category with no property, or a property with no dimension, or a dimension with one pole
- a relationship that is provisional, tentative, or a cross-respondent pattern
- an empty paradigm slot already recorded by axial coding
- a memo that asks a question and names a category

Each direction stores `target_id`, `why`, `audit`, `sampling_query`, and `selection_criteria`. The why cites those records. The query says what kind of passage to seek next and the criteria for choosing it inside this corpus.

`retrieve_passages` ranks passages that are not already evidence for that target. Overlap has to be a distinctive stem, or a stored embedding at or above 0.55. Generic stems such as work, job, and team do not count. A document with no segments is preferred. Proposals are proposals: they are not treated as data the researcher collected. When no passage qualifies, `corpus_status` is `no_suitable_unsampled_passages` and `new_data_collection` is `researcher_action`. The note says this package did not collect anything.

### Human checkpoint

`sampling_checkpoint` calls `interrupt()` when `decision_mode` is `human`. Resume with `{"accept": true}`, `{"leave_sampling": true}`, or a `directions` list of `accept`, `edit`, or `reject`. An empty object is not a resume value. `sampled_passage_ids` must be in the corpus unless `outside_corpus` is true. That flag only records data the researcher collected. The package still does not fetch it and does not invent its text.

`saturation_decision` then interrupts with the indicators, the ledger, and the stopping-criteria report. Resume with `not_claimed`, `sufficient_for_this_question`, or `not_saturated`, plus a rationale, or `{"leave_saturation": true}`. There is no boolean true. The stored record has `actor` and `timestamp`. `judgment` stays null.

### LLM checkpoint

`decision_mode='llm'` does not interrupt. The model must choose direction ids and passage ids from the request. Unknown ids are stored on `model_unknown_direction_ids` and `model_unknown_passage_ids` and are not used as evidence. `model_direction_ids`, `model_passage_ids`, and `model_sampling_rationale` are model fields. `researcher_sampling` and `saturation_decision` stay empty.

`model_saturation_judgment` is written only by this step. It requires at least 3 documents and at least 2 sampling rounds, and the saturation window must not be `not_interpretable`. Below that, `status` is `not_interpretable`, `assessment` is null, and the missing assessment is not filled in as `not_saturated`. No decision call is spent on a window that cannot be read. When the window is readable, the model sees the window-wide rates and the ledger. The judgment has `model_judgment` and `does_not_set_judgment`. It is not copied into `saturation_decision`, `researcher_saturation`, or `judgment`.

### Budget, timeout, and the ledger

`max_sampling_review_calls` is the coding ceiling for a reread. `max_decision_calls` is the review ceiling for a model selection and a model saturation judgment. They share `max_sampling_review_tokens`. `batch_size`, `sampling_request_timeout`, and `evidence_limit` are performance settings. `pursuit`, `coding_family`, `quiet_rounds`, `min_documents`, and `category_documents` are procedure.

An upstream `provider_budget` is restored at the start. The checkpoint stores the snapshot, and a resume restores it onto the new runtime instead of starting the counters at zero. A timeout keeps the directions and any round already appended, writes a partial `theoretical_sampling.json` when an output directory is set, and leaves status `incomplete` with stop `timeout`.

Each call appends one ledger round: direction keys, which of those keys are new, accepted ids, sampled ids, model ids, proposal counts, the indicator snapshot, and the constant-comparison delta against the previous round. The first round's deltas are undefined. A later round subtracts the previous snapshot only when both rates exist. A reread note does not rewrite the saturation curve, and the ledger says so.

`pursuit` defaults to `brief_only`. The export's `next_round` names passage ids and `afocused_code` as the later entry point. It does not start that run. `pursuit='reread'` asks the model for a note on each selected passage, inside the coding ceiling. The quote is kept only when it anchors in that passage. An unanchored quote is removed and flagged `quote_unanchored`. Passages that do not fit the remaining calls are `reread_unsent_passage_ids`, and the run is `budget_limited`.

Calling `theoretical_sample` again on a finished result appends the next round. It does not rewrite the open-coding order.

## Stopping criteria

`stopping_criteria_report` lists six rows. Each is `met`, `unmet`, or `undefined`, with the threshold that was used.

| Criterion | Met when | Undefined when |
| --- | --- | --- |
| `new_codes` | `window_new_code_rate` is 0 and there are at least 3 documents | the rate is missing, or there are fewer than 3 documents |
| `new_properties` | the same, for the property rate | the same |
| `new_relationships` | the same, for the relationship rate | the same |
| `code_frequency_shift` | total variation between the last two documents is 0 | the shift is null, or there are fewer than 2 documents |
| `sampling_rounds_without_new_directions` | the last `quiet_rounds` rounds (default 2) introduced no new direction key | fewer rounds than that |
| `core_category_coverage` | the chosen core is an endpoint of a non-rejected relationship that has a statement | no core was chosen |

A zero rate is not treated as a zero when the report never computed it. `declares_saturation` is false. `judgment` on this report is null. Meeting a row is not a claim that the theory is saturated.

## Axial and selective gaps

Checked against Strauss and Corbin's paradigm and against the Glaserian alternatives in the later-stages note.

Kept as they are, because they already match the 0.8 design and have tests:

- five paradigm roles, and the 1990/2008 label switch
- a Charmaz lens that does not fill paradigm slots
- empty slots left empty
- same-passage support kept apart from cross-respondent patterns
- core candidates scored, not chosen
- `core_unsupported_by_graph`, with cross-run core stability undefined

Filled, because each one is a small deterministic check with a test:

- constant comparison across sampling rounds, as indicator deltas on the ledger
- the core-endpoint check reused as a sampling direction and as a stopping criterion
- `coding_family='glaser'` on a sampling query. The labels are the Six Cs (causes, contexts, contingencies, consequences, covariances, conditions), plus strategy and means-goal. They sensitize the query. They are not codes and they are not axial roles. Axial `coding_lens` still rejects `glaser`.

Left unimplemented, and still limitations:

- collecting new fieldwork data
- an automatic boolean saturation flag, or any writer of `judgment` other than the explicit null
- Strauss and Corbin's conditional matrix
- a full process analysis
- Glaser's 1978 coding-family catalog beyond the eight labels above
- a Glaserian graph that picks a core before families
- one mandatory core category
- launching `aopen_code` or `afocused_code` from the sampling graph. The next-round plan names the passages. The default pursuit does not spend a second coding bill.

## Decisions

- The sampling graph is separate. Selective coding does not gain a pursuit node in this release, so the axial and selective drawings stay the ones 0.8.7 tested.
- `model_saturation_judgment` is a new field. The 0.8 axial field `model_saturation` is unchanged and is not reused for this decision.
- A model assessment below the window is withheld. It is not stored as `not_saturated`.
- `category_documents` (default 2) is the thin-category spread. `min_documents` (default 3) is the saturation window. They are different procedure settings.
- Code-frequency shift is met only when the total variation is 0. There is no softer cutoff. Anything else is unmet or undefined, and none of those words means saturated.
- Quiet sampling rounds default to 2. One round cannot satisfy "rounds without new directions."
- `pursuit='reread'` writes notes. It does not mint open codes or focused assignments. Doing that inside this graph would start a second coding procedure under a sampling flag.

## v29 findings recorded on the 0.8.7 graphs

These are fixes on the axial, selective, and focused paths. They do not add a node and they do not change the sampling graph. Each one flags or reports the problem. None of them drops the row to make the count look cleaner, and none of them turns an undefined value into zero.

What was already true, and what changed:

| Finding | What 0.8.7 did | What 0.9.0 records |
| --- | --- | --- |
| Storyline causes | The prompt said a cross-respondent row is not a cause. A sentence that said so anyway had no statement flag. | `storyline_unaccepted_relationship` when the cited row is unaccepted, provisional, or `cross_respondent_pattern`. Causal wording sets `phrasing` to `pattern`. The sentence stays. |
| Focused synthetic stall | The model sees `p-contrast`. The sent id is `synthetic-counterexample`. The addressed check compared the raw id, so the call was `no_progress` and later passages were not reread. The synthetic `irrelevant` or `contradicts` judgment also incremented `claim_test_counts`. | The returned id is canonicalized before that check. The audit record stays. Research counts and claim-test relevance skip it. |
| Thin relationships | Proposals listed per-category passages, not the passages that contain both. Blank rows with a bad passage id were logged only as `unknown_passage`. | `candidate_same_passage_cooccurrence` is sent and preferred. Quotes must be verbatim spans of a sent passage id. Cross-respondent rows stay patterns. `relationship_counts` separates accepted, provisional, cross-respondent, proposed, and rejected. |
| Saturation on a thin slice | The reviewer saw window rates and not pair coverage. Three runs said `sufficient_for_this_question` on coverage 0.025 from one relationship in 6 of 946 pairs. | The payload includes `pair_coverage` and `relationships_found`. At 10 or more planned pairs, coverage below 0.05 or fewer than 2 relationships is `not_interpretable`. A sufficiency assessment is withheld. `model_said` keeps the words. The stopping report has a `pair_coverage` criterion. Below 10 planned pairs that criterion is undefined. |
| Category label as a quote | Token overlap with the category label anchored the endpoint. `quotes_not_in_cited_passage` was the only catch. | A quote that is not a verbatim substring of a located passage is not anchored. The located-passage flag remains. |
| Memos | "The evidence is two-sided." was `memo_quote_unsupported`. A causal memo on a cross-respondent row was not flagged. `memo_same_passage_mismatch` only matches the phrase "same passage". | `memo_degenerate` for a strength-only sentence. `memo_missing_quote` when the memo does not use the quote. `memo_causal_on_pattern` for causal wording on a cross-respondent or unaccepted row. A memo that says the evidence is partial is not those flags. |
| Counterexample calls | Only both-endpoint overlap was queued. A round with no such passage made no calls and did not say why. | `no_call_reason` is `no_candidates_meeting_both_endpoint_overlap` or `budget`. The fallback queues same-category opposite-pole passages inside the remaining slots. |
| Unsupported core | `core_unsupported_by_graph` already fired. The reviewer was not told to prefer an endpoint. `core_stability` was already undefined, with `n`. | Candidates with relationship evidence come first, with an instruction to prefer them. The model id is kept. `relationship_preferred_core_id` names the endpoint. Stability stays undefined. |
| Blank diagnostics | `blank_rows` counted only after the unknown-passage check, so an empty row with an empty passage id was `unknown_passage` and `blank_rows` stayed 0. | A row with no statement and no quotes is `blank_proposal` first. `rejection_reasons` counts blank, unknown passage, empty statement, and unrequested pair separately. |
| Short quotes | "Its a job." and "Glad to be here" missed the lexical check. Generic stems do not anchor alone, and there was no second signal when the relationship had no embedding. | A short verbatim quote can use the passage embedding against the category. The disagreement is named. If no embedding is stored, `second_signal` is `unavailable` and the endpoint stays unanchored. |
| Determinism | Temperature 0 was recorded, with a note that it does not guarantee identical completions. Paused and uninterrupted runs still diverged. | The note says a paused-versus-uninterrupted comparison is not a reproducibility check. `model_determinism` records the setting. It does not claim the provider returned the same text. |

## Relationship evidence tiers (0.9.1)

0.9.1 does not add a sampling node. It changes how an axial relationship is accepted. The both-quotes-in-one-passage rule accepted nothing in live runs. A cross-respondent stitch is now a hypothesis, not citation support and not a dropped row. The case floor, the wording each tier may use, and the decision to keep negative-case search are in [design-relationship-tiers.md](design-relationship-tiers.md). `pattern_min_cases` (default 2) is a package design choice. It is not a quotation from the method literature.

0.9.2 does not add a sampling node either. Negative-case search now queues one-sided and opposite-pole respondents, and a core is graph-supported only at tier 2 or higher. The rule is [design-negative-cases-092.md](design-negative-cases-092.md).

## Request budget and reread context (0.9.3)

`sampling_token_budget` is the per-request ceiling. `max_sampling_review_tokens` is the total allowance. They are performance settings. Each initial selection, reread, and saturation request is measured before it is sent. A request that does not fit is not sent. The run is `incomplete`, with reason `request_budget` or `token_allowance`. A repair that would overflow the per-request budget is not sent. A response that still does not parse is kept, with each attempt and its usage, and the stage is `incomplete` rather than an exception. The reread payload includes the selected direction, the sampling query, and the target evidence. The note is stored on that direction. The step allowance is derived from the planned calls. A step limit that is still hit returns the partial result. The note is [design-review-fixes-093.md](design-review-fixes-093.md).

## Direction and passage pairs (0.9.4)

A reread is one direction and one passage. The researcher's `sampled_passage_ids` on a direction are not reassigned because retrieval proposed that passage for another direction. A passage selected for two directions is two prompts, two notes, and two coding calls. The unsent pair names the direction. `decision_mode="llm"` can return `selections`. When it does not, a passage proposed for two selected directions becomes two pairs, and a passage that no selected proposal contains is flagged instead of being given to the first proposal. The note is [design-review-fixes-094.md](design-review-fixes-094.md).

## Checkpointed rereads and explicit selections (0.9.5)

Each `(direction_id, passage_id)` reread returns before the next request, so a later timeout keeps the note and the prompt row. The partial result separates completed, failed, and unattempted pairs. Resume continues at `pursuit_reread`, skips a pair that already has a note, and keeps the calls already consumed. A selection that does not parse is stored before the repair call. When `selections` is present, those pairs are the rereads. They are checked against the directions and the corpus. The flat lists are derived from the pairs that pass. A disagreement, or a pair that names an unknown id, leaves the stage incomplete. The note is [design-review-fixes-095.md](design-review-fixes-095.md).

## Reread repairs and the active pair (0.9.6)

A reread repair is its own visit. The malformed response and its usage are stored before the repair is sent. Resume sends the repair and does not repeat the initial request. Before each provider call the visit records `reread_active_pair`. A timeout uses that pair as `reread_failed_pairs`. Pairs already in `reread_unsent_pairs` stay there. An empty flat list is omitted and derived from the selections. Each nonempty list is compared on its own. The note is [design-review-fixes-096.md](design-review-fixes-096.md).

## Sampling stage allowance, packed selection, and reread notes (0.9.7)

`max_sampling_review_calls` counts calls made in this sampling stage. Upstream `stage_used` is stored on `sampling_call_baseline` and is not charged again. `calls_used` still carries the earlier total. `sampling_token_budget` stays the per-request ceiling. A selection that does not fit drops the lowest-ranked directions until it does. One direction that still does not fit is omitted with `request_budget`. Pairs beyond the remaining coding calls are stored on `reread_unsent_pairs` with reason `call_allowance`. That is the stop. A flat-list mismatch stays on `model_flat_list_mismatch` and does not replace it.

Reread notes are copied onto the ledger row with the direction key. An irrelevant note, or a rejected direction, is removed from `analytic_new_direction_keys`, and the next round does not issue that key again. A note that is not irrelevant adds a follow-up direction. A null window rate is filled from complete document points. The ledger indicator and the stopping criterion use that value. `judgment` stays null. Empty paradigm slots for different roles do not share one direction key. The note is [design-review-fixes-097.md](design-review-fixes-097.md).

## Passage rejection and packed repair (0.9.8)

The structured `judgment` is the only reading of a passage. The note text is not searched. `judgment` `irrelevant` records that passage on `rejected_passages`. The next round does not offer it for that direction. The direction stays in the ranking. A follow-up is added only when `judgment` is `relevant`.

A direction closes when the human action is `reject`, or when the model sets `direction_decision` to `close` and writes a non-empty `direction_close_reason`. A close without a reason does not close the direction. `closed_direction_keys` is that set. `analytic_new_direction_keys` drops those keys and keeps a key whose passage was rejected. Quiet rounds read `analytic_new_direction_keys` when the field is present. A later round that offers an already seen open direction does not count it as new, so the quiet-round criterion can be met while the direction stays open. Older ledger rows that omit the field still use `new_direction_keys`.

`model_sampling_pairs` is the validated selection. When `pursuit` is `reread`, pairs beyond the remaining coding calls are `reread_unsent_pairs` with reason `call_allowance`, `scheduled_reread_pairs` is the subset that will be sent, and the stop is `call_allowance`. `brief_only` stores the full selection and does not set that stop. A selection repair sends the packed payload again, including only the passage proposals that fit the request. The note is [design-review-fixes-098.md](design-review-fixes-098.md).

## Stable question identity (0.9.9)

A research question keeps the key `kind:target` or `empty_paradigm_slot:target:role`. A reread note is stored on that key as `question_key`. Relevance does not create `reread_follow_up:<previous key>`.

A distinct question is a different key. The note must set `follow_up_kind` to one of the record-grounded direction kinds and `follow_up_target_id` to a code, relationship, or core id already on the run. `empty_paradigm_slot` also needs `follow_up_role`. The same fields always produce the same key. The note text is not compared. A target that is not on the run does not become a question.

`completed_pairs` records each `(question_key, passage_id)` that was reread. Retrieval does not offer that passage for that question again, and the reread loop does not send it. A pair withheld for that reason uses `completed_pair` and does not make the stage `budget_limited`.

Loading a ledger strips leading `reread_follow_up:` prefixes. `direction_keys`, `new_direction_keys`, `analytic_new_direction_keys`, closed keys, rejected passages, and notes are rewritten to the stable key. The original nested key is kept on `recorded_direction_key` when it differed. A nested key is not new when its stable question was already on an earlier round. A close stored against a nested key closes that stable question. The note is [design-review-fixes-099.md](design-review-fixes-099.md).

## Sampling spend and quiet rounds (0.9.10)

`sampling_token_baseline` records token estimates carried from earlier stages and sets this stage's `tokens_estimated` to 0. The first sampling round is not stopped because axial or focused coding already spent tokens. `max_sampling_review_tokens` and `sampling_token_budget` (default 6000) still bound what this stage spends. `calls_used` stays the carried total.

The saturation request is packed the same way the selection request is packed. Oldest ledger rows are dropped first, then `stopping_criteria`, then indicator fields. `n_documents` and the instruction stay. `saturation_pack` records the omitted fields. A request that still exceeds `sampling_token_budget` is not sent.

A human resume that lists some directions does not set `reject` on the rest. An unlisted direction stays open. An explicit `reject` still closes that direction.

`sampling_rounds_without_new_directions` is `not_evaluable`, and `observed` is null, when a recent quiet round left a selected pair unsent for `call_allowance`, or when `reread_unsent_pairs` was not recorded. The criterion can be met when those pairs were sent, or when each remaining unsent pair is in `closed_direction_keys`. The note is [design-review-fixes-0910.md](design-review-fixes-0910.md).

## Quiet rounds after a failed reread (0.9.11)

A quiet round is evaluable when every selected pair was completed or explicitly exempted. `completed_pair` is completed. A direction key, question key, or direction id in `closed_direction_keys` is exempt, including when the unsent reason is `request_budget` or `call_allowance`.

Any other unsent or failed reason leaves the criterion `not_evaluable` and `observed` null. That includes `request_budget`, `token_allowance`, `schema_unparsed`, `parse_failed`, `timeout`, `incomplete`, `missing_passage`, and a reason the ledger does not name. `reread_failed_pairs` are read the same way. An `reread_status` or `reread_stop` of `timeout` with no listed pair, or with a listed pair that was not exempted, is an unresolved timeout and is not silence. The note is [design-review-fixes-0911.md](design-review-fixes-0911.md).

## Recovered timeouts and question keys (0.9.12)

A reread that times out and then completes sets `reread_status` to `complete` and clears `reread_stop`. The timeout is appended to `reread_recovered_timeouts` and to `sampling_attempts` with status `recovered`. Those records are history. They are not the active failure the quiet criterion reads. An unresolved timeout, where `reread_status` or `reread_stop` is still `timeout`, stays `not_evaluable`.

`reread_unsent_pairs`, `reread_failed_pairs`, `reread_unattempted_pairs`, `sampling_selection_unsent_pairs`, and `scheduled_reread_pairs` store `question_key` and `direction_key` as the stable question, for example `thin_category:c-prep`. `direction_id` stays the round id, such as `dir-1-0001`. The exemption compares `question_key` first, then `direction_key`, then `direction_id`. `completed_pairs` already stored `question_key`. `model_sampling_pairs` and `selected_sampling_pairs` stay the model's direction and passage ids. `reread_active_pair` stays the in-flight direction id and passage id; the failed-pair copy made from it receives the question key. The note is [design-review-fixes-0912.md](design-review-fixes-0912.md).

## Per-pair timeout recovery (0.9.13)

A recovered pair is removed from `reread_failed_pairs` when its note is stored. That does not wait for the rest of the batch. The pair is appended to `reread_recovered_timeouts` and to `sampling_attempts` with status `recovered` and reason `timeout`. Pairs that still have no note stay failed. When no timeout failure remains, `reread_stop` is the next unresolved reason, such as `call_allowance`, rather than the recovered timeout. A pair is not both completed and failed. A batch in which every timed-out pair then succeeds still clears `reread_stop` and can meet the quiet-round criterion, as in 0.9.12. An unresolved timeout still leaves that criterion `not_evaluable`. The note is [design-review-fixes-0913.md](design-review-fixes-0913.md).

## Configured document minimum (0.9.14)

`min_documents` (default 3) is the saturation window the readiness check uses. A model saturation call is withheld when `n_documents` is below that configured value, when fewer than 2 sampling rounds are on the ledger, or when `saturation_window` is `not_interpretable`. The withheld record stores `minimum_documents` as the configured value, `status` `not_interpretable`, and `assessment` null. The decision call is not spent. A human checkpoint still pauses. Its payload carries `minimum_documents`, `window_ready`, and `window_reason`. The stopping-criteria rate rows already used this same minimum, so a run can show those rows as `undefined` while the model call stays unsent. The indicator label `saturation_window` is still computed from the constant 3 when the run has no curve of its own. That label still withholds the model call. The note is [design-review-fixes-0914.md](design-review-fixes-0914.md).

## Reread interpretation report (0.9.15)

`SamplingResult.completed_pairs` is `metadata.reread_completed_pairs`. It stays null when that list was not recorded, including a `brief_only` run. The ledger's `completed_pairs` are unchanged, and a completed pair is still not sent again.

`reread_interpretations` classifies notes after the run. It does not change which passages are selected or how a judgment is stored. A note is `unanchored` when `quote_unanchored` is set, including when `judgment` is `relevant`. Otherwise `relevant` and `irrelevant` follow the structured judgment, an empty or other judgment is `unjudged`, and a pair in `reread_failed_pairs` or an unsent pair with reason `schema_unparsed`, `timeout`, or `incomplete` is `failed` when it has no note. `call_allowance` is not a failed interpretation.

`relevance` is relevant notes divided by every classified interpretation, and null when that denominator is zero. The same block is reported for each direction `kind`, for `new_document` and `known_document`, and for `new_respondent` and `known_respondent`. A missing document or respondent id is left out of those novelty blocks. `relevant_reread_tokens` and `tokens_per_relevant_reread` stay null unless every relevant note has usage that can be tied to its passage. `cost_per_relevant_reread_usd` stays null unless the run records `token_prices.input_usd_per_million` and `token_prices.output_usd_per_million`. The note is [design-review-fixes-0915.md](design-review-fixes-0915.md).

## Step limit and reread expenditure (0.9.16)

The default recursion limit counts graph visits: five preparation nodes, one selection-repair visit, three visits per coding call (arm the pair, send it, and one repair), one visit that withholds pairs still pending, and three final nodes. It is not the call ceiling plus a fixed overhead. A caller can still pass `recursion_limit`.

`tokens_per_relevant_reread` is unchanged. `reread_expenditure_tokens` sums every sampling reread and every `SamplingRereadNote` repair, including irrelevant responses and failed parses. `expenditure_tokens_per_relevant_reread` divides that sum by `relevant_reread_count`. The rate is null when the count is zero, and the count is still reported. Cost fields stay null unless both token prices are recorded. `human_confirmed_useful_rereads` is null and is not derived from model judgments. The note is [design-review-fixes-0916.md](design-review-fixes-0916.md).

## Oversized rereads and expenditure coverage (0.9.17)

A reread that does not fit `sampling_token_budget`, or the remaining token allowance, is not a provider call. Those pairs are withheld in the visit that would have armed the first one that fits. The recursion limit stays the visit count from 0.9.16. It does not grow with the number of passages. A model selection is still cut to `max_sampling_review_calls` before that loop. A human assignment is not cut there, so every assigned passage is measured. When every assigned request is over the per-request budget, the stop is `request_budget` and every passage is on `reread_unsent_pairs`. A caller can still pass `recursion_limit`.

`reread_expenditure_tokens` is the known sum. A sent reread call with missing or all-zero usage is a call without usage. `usage_from_message` stores a missing provider report as zeros, and that record is not a known spend of zero. `reread_expenditure_calls` and `reread_expenditure_calls_with_usage` count sent calls. `reread_expenditure_usage_coverage` is their ratio, and null when no reread was sent. `reread_expenditure_complete` is true when every sent call reported a positive input or output, including when there were no reread calls. `expenditure_tokens_per_relevant_reread` and `expenditure_cost_per_relevant_reread_usd` stay null when usage is incomplete or when `relevant_reread_count` is zero. Cost stays null unless both prices are recorded and the usage is complete. A complete run with no reread calls reports cost 0 when prices are set. Prices are copied from source `metadata.token_prices`. An omitted request is not an unknown-usage call. `tokens_per_relevant_reread` is unchanged. The note is [design-review-fixes-0917.md](design-review-fixes-0917.md).

## Timed-out rereads in expenditure coverage (0.9.18)

A timeout during `pursuit_reread` keeps the operation that was in flight. An initial reread is `sampling_reread`. A repair, while `reread_repair_pending` is set, is `schema_repair` with `schema` `SamplingRereadNote`. The row has no usage. It is a sent reread call, so `reread_expenditure_calls` includes it and `reread_expenditure_complete` is false. The known token sum, `tokens_per_relevant_reread`, and the provider budget's `calls_used` are unchanged. Full expenditure rates and cost stay null. A selection or saturation timeout stays kind `provider` and is not reread spend. The note is [design-review-fixes-0918.md](design-review-fixes-0918.md).
