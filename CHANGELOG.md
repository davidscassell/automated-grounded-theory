# Changelog

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
