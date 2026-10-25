# automated-grounded-theory — Technical Design (0.6.0, corrections in 0.6.1, 0.6.2, 0.6.3, 0.6.4, 0.6.5, 0.6.6, 0.6.7, 0.6.8, 0.6.9, 0.6.10, 0.6.11, 0.6.12, 0.6.13, 0.6.14, 0.6.15, and 0.6.16, focused coding in 0.7.0, fixes in 0.7.1, 0.7.2, 0.7.3, 0.7.4, 0.7.5, 0.7.6, 0.7.7, 0.7.8, and 0.7.9, axial coding and saturation indicators in 0.8.0, fixes in 0.8.1)

Repo: https://github.com/davidscassell/automated-grounded-theory
Status: this file is the open-coding design, beginning with steps 1 and 2 of the v0.6 plan. Focused coding is in 0.7.0 and the 0.7.1 through 0.7.9 fixes. Axial coding, saturation indicators, and selective coding are separate experimental graphs from 0.8.0. Within-corpus theoretical sampling and a saturation decision that leaves `judgment` null are in 0.9.0 and later. Package `policy_version` is `0.9.28`. The current index is [README.md](README.md). The 0.8 axial slice is [design-axial-0800.md](design-axial-0800.md). 0.6.1 corrects measurement, batching, the alias gate, idle waits, and the shared client loop. It does not add those later steps.

The default procedure is Charmaz-informed automated analysis: initial coding against a frozen snapshot, constant comparison with distinct operations, and `build_themes` as a descriptive reporting view. It is not an axial-coding template.

## 1. What this release changes

| Choice in 0.5.4 / 0.5.5 | 0.6.0 |
| --- | --- |
| One code decision treated as the whole passage | Several assignments, each with its own evidence span. Spans may overlap. |
| Paragraph cuts that split answers | `unit="incident"` (default): interviewer question plus the full answer, then ~150–800 character incidents, never mid-sentence, cap ~1200. |
| Quote mismatch dropped the code | Kept from 0.5.5: `quote_status` is exact, normalized_match, fuzzy_match, possibly_modified, or unanchored. Possibly modified and unanchored are flagged. The code stays. |
| Opposite valence blocked a merge | Valence is context. `alias_duplicate` is not rejected for stance. `record_dimension` and `keep_distinct` keep the contrast. `link_to_category` can place both under one category label. |
| One theme per code, 30% share cap | Many-to-many membership. `max_theme_share` and `max_themes` are presentation budgets. Prevalence names its denominator. |
| Pairwise union-find was transitive | A component of three or more codes is merged only when every pair was judged. Otherwise the extra edge is `non_transitive` and listed under lost-distinction candidates. |
| Sequential coding node | `Send` workers read snapshot `S_t`. One commit writes `S_t+1`. `batch_size` defaults to 1 and can be set to 4. |

## 2. Stack

- Python ≥ 3.11, LangChain (`init_chat_model("provider:model")` or a `BaseChatModel`), LangGraph (`Send`, checkpointing).
- Pydantic v2 at model boundaries. One schema-repair retry (`REPAIR THIS OUTPUT`), separate from transport retries.
- Optional extras: provider SDKs, `checkpoint`, `plots`, `viz` (`grandalf`; `pygraphviz` is optional and not in the extra), `taguette` (`taguette>=1.5`, for a caller who imports Taguette itself), `dev`. The `taguette` extra does not add a SQLite project adapter.
- `draw_mermaid_png` posts to mermaid.ink. It is not an offline renderer.

## 3. Public API

```python
result = await aopen_code(
    corpus,
    context,
    model="xai:grok-4",          # or a BaseChatModel
    seed=42,
    unit="incident",             # default
    passages_per_request=1,      # alias: batch_size
    max_concurrent_requests=4,   # alias: concurrency; in-flight cap only
    passages_per_codebook_update=1,
    review_mode="document",      # "wave" is opt-in
    sweep_every=25,
    pipeline=True,
    # procedure="strict_sequential"  # one passage, commit each time, no pipeline
    review_below=0.5,
    min_merge_confidence=0.5,
    max_merges_per_sweep=12,
    token_budget=6000,
)
themes = await abuild_themes(result, context, model=model, max_themes=12, max_theme_share=0.30)
report = await aevaluate(result, compare_to=[other])
run = load_run(output_dir)
graph = get_graph("open_code")  # or "themes"
```

`context` guides attention. Keys named `codes` or `codebook` raise `PresetCodesError`.

`max_theme_share` does not block a fold. `abuild_themes` does not develop categories. There is no `arun_study` in this release.

## 4. Records

| Record | What is stored |
| --- | --- |
| Document | Stable id, source text hash, metadata, original order. |
| Incident | Question, exact start/end on the immutable source, `coverage_status`, `exclusion_reason`. |
| EvidenceSpan | Document id, exact offsets, `quote_status`, code id. Repeated quotes are disambiguated by occurrence or neighborhood and flagged `ambiguous_span` when neither is available. |
| CodeVersion | Stable `code_id`, `revision`, label, definition, inclusion/exclusion, anchor and contrast span ids, lifecycle. A hex id is never used as a label. |
| CodingAssignment | Span, code id and revision, snapshot id, rationale, optional `uncertainty_flags`. |
| Comparison | `alias_duplicate`, `refine_definition`, `split_code`, `link_to_category`, `record_dimension`, `keep_distinct`, `defer`. `relate_categories` is recorded and not executed. |

`coverage_status` is `coded`, `context_only`, `excluded`, `failed`, or `pending`. A missing output slot is `failed`, not an empty coding. `nearest_code_id` is null when the book is empty or no neighbor was named.

Uncertainty flags: `ambiguous_span`, `insufficient_context`, `competing_code`, `unsupported_inference`. Self-rated confidence is diagnostic only.

`metadata["code_versions"]` and `metadata["coverage"]` are materialized on the result. Taguette CSVs are unchanged: no confidence and no `quote_status` column. A future Taguette SQLite project adapter must pin the Taguette version it targets. This release does not include that adapter.

## 5. Graph

`get_graph("open_code")`:

`segment_and_shuffle` → `prepare_round` → `Send("code_worker")` → `commit` → `sweep` → `consolidate` → `review_evidence` → `export`.

`review_evidence` runs only while a code still needs review and the final-review allowance remains. Otherwise consolidation goes straight to export.

`prepare_round` freezes `snapshot_id`, `definition_revision`, `evidence_revision`, and `policy_version` (`0.6.5`). Workers do not mutate the codebook. `passages_per_codebook_update` is how many passages share that snapshot. `max_concurrent_requests` only caps in-flight calls. The proposal reducer dedupes by proposal id, drops conflicting payloads, sorts, and clears only on `{"__reset__": True}`. `commit` rejects a stale snapshot id, skips proposal ids already committed, and is the only writer of definitions.

`get_graph("themes")` fans chunk workers out with `Send` into `theme_worker`, then joins at `validate`. The reporting graph is separate from category development.

One `CallLimiter` is shared by coding, sweep, and evaluation calls in a run, including `abatch` and retries. Concurrent batches take slots from that same adaptive gate. Cache keys add snapshot, definition, evidence, and policy versions when those are set. Transport retries and the one schema repair are counted separately (`transport_retries`, `schema_repairs`). A repaired call's `usage` adds the failed response to the successful one.

Metrics on `metadata["metrics"]`: duplicates before reconciliation, reconciliation calls and seconds, coding wall time, per-request latency (`coding_seconds`, `coding_p50`, `coding_p95`), queue wait, tokens, retries, lost-distinction candidates, `merges_later_split`, `max_in_flight`, and `activity` (in-flight events, idle gaps of at least 0.05 seconds, and the same peak). `duplicate_rounds.json` and `lost_distinction_candidates.json` are written next to the codebook.

Checkpoint state is serializable (snapshot refs, passages, proposals). Models and the limiter stay on the runtime object.

## 6. Themes and saturation

Themes may share codes. Each theme reports `denominator_passages`, `denominator_documents`, and `denominator_respondents`. A presentation-budget note is a warning, not a rejected fold.

Saturation keeps the pre-alias and post-alias curves (`new_codes_pre_alias`, `new_codes_post_alias`). `judgment` is always null.

## 7. Evaluation and the three-way benchmark

`benchmarks/compare_versions.py --offline` runs the `v0.5.4` git tag (merged PR #4, not a 0.6 commit) in a detached worktree against the current package at `batch_size=1` and `batch_size=4` on the fake coder. `--model`, `--corpus`, and `--context` (a JSON object or a markdown notes file) run the current package live and pass the research context through. `--log-dir` appends every chat-model call to `calls.jsonl`. A tag older than 0.6 keeps that tag's shipped defaults. Do not treat that script as a download of Muse.

Benchmark scoring:

- Human codes are those tagged in the coded documents.
- Document ids `6` and `6.0` normalize to `6`.
- Alignment rows carry `grade` `full` / `partial` / `none` and may list several `llm_codes`. `audit_sample` exports 20% for hand review.
- Excerpt recall uses character-offset overlap (IoU or containment ≥ 0.3).
- Agreement on a fixed unit grid is per-code present/absent (`unit_grid_agreement`), not single-label kappa.
- `quote_and_coverage_rates` reports flag rates and uncoded share.
- `amap_codes` resolves a `provider:model` string through `init_chat_model`.
- Progress lines are flushed.

## 8. Results so far (live, xAI Grok, 150 comments)

| Version | Time | Final codes | Reuse | Singletons | Notes |
| --- | --- | --- | --- | --- | --- |
| v1 | 48.5 min | 132 | 13% | 90% | no merges |
| v6 (0.5.2) | 7.9 min | 68 | 56% | 29% | first real merges |
| v7 (0.5.3) | 21.2 min | 52 | 67% | 50% | safer merges; seeds diverge |
| v8 (0.5.4) | 49.5 min | 71 | 52% | 63% | sweep outputs of 8–12k tokens |

RAND Muse E2 under 0.5.4, as scored: precision 0.60, recall 0.19, F1 0.29 against 78 human codes. In-scope tags on the two coded documents are 23 codes, and much of the gap was the old quote drop plus one-label alignment. That figure is not the paper's codebook-application kappa.

0.6.0 has not been timed on that live corpus in this change. A later live three-way run of 0.6.0 (v0.5.4, batch 1, batch 4) is what 0.6.1 answers. 0.6.1 itself has not been live-timed.

## 9. Corrections in 0.6.1

- `duplicates_before_reconciliation` counts proposals born in the same snapshot that a later comparison aliases or judges as the same concept (`same_action` and `same_object`). It does not count identical labels. Exact proposal-id collisions stay in `exact_proposal_collisions`. Each round's pairs are in `duplicate_rounds.json`.
- Every accepted alias is exported with both definitions, up to three quotes from each side, and the rationale (`kind: accepted_alias` in `lost_distinction_candidates.json`). `merges_later_split` counts aliases a later quote split pulled apart.
- The shared limiter releases its slot before a retry sleep. `Retry-After` cannot exceed `max_backoff`. Pipeline overlap does not start a coding task when `concurrency` is 1, so a sweep is not stuck behind its own overlap. The run metadata includes the in-flight timeline.
- 0.6.1 filled a short document's round from the following documents, and kept a long document's passages together. 0.6.2 replaces that round shape when `batch_size` is above 1; see below. `batch_size` still defaults to 1.
- `run_sync` reuses one background event loop and does not close HTTP clients the caller still holds. `build_themes` can follow `open_code` on the same model instance.
- `alias_duplicate` requires the same action, the same object, a contrast check, no material contrast, and evidence that is not mixed. Labels that each name an object or aspect the other lacks stay `keep_distinct` (`aspects_differ`), including when they share a platform name. Mixed `quote_fits` or `evidence_same` false is `defer`. A missing rubric is `keep_distinct`. Valence is still not a reason to alias or to refuse. 0.6.1 auto-merged a trivial near-duplicate with the same stance. 0.6.4 replaces that rule; see below.
- `presentation_themes` folds reporting themes that share a content word. Analytic membership is unchanged. There is no theme quota. The prevalence chart uses that reporting view. `presentation_themes.json` is the list.

## 10. Corrections in 0.6.2

When `batch_size` is above 1, document mode builds each round by round-robin over the seeded document order. A round takes at most one passage from each document. Incident order inside a document stays as it was. Adjacent passages of one document share a round only when that document is the only one left. `batch_size` still defaults to 1, and that setting still reads a document through before the next.

Before the commit writes a round, one comparison call judges the new codes proposed in that round against each other. It uses the same alias, keep-distinct, link, and defer rules as a sweep. A pair that is not one concept stays two codes. Accepted aliases are written once. Those pairs are `in_batch` in `duplicate_rounds.json`. `cross_snapshot` is the later same-snapshot count, and `duplicates_before_reconciliation` still counts only that list.

Sweep, merge-check, definition-rewrite, and quote-fit calls use `sweep_timeout` (default 300 seconds) when that is longer than `request_timeout`. A zero on either side waits with no deadline.

## 11. Corrections in 0.6.3

A semicolon-joined label is split into separate candidate codes before validation. Each piece is checked on its own. A piece whose label matches a code already in the book is reused.

An empty label, a label over the length limit, and a new label that bundles several concepts are not dropped. Bundling means a comma list of three or more concepts, or three or more concepts joined by "and". The passage is sent back once with feedback that names the problem and asks for one concept. If the retry is still empty, too long, or bundled, or if it returns no codes, the code is kept and flagged (`empty_code`, `code_too_long`, or `bundled_concepts`). A blank label is stored as `(empty label)`. Generic labels are still rejected. `policy_version` was `0.6.3` in that release.

## 12. Corrections in 0.6.4

Lexical similarity proposes a comparison. It does not merge. Auto-merge is only a normalized spelling or inflection variant: case, punctuation, whitespace, and plural or gerund forms, with compatible definitions. Negation or a polarity difference never auto-merges. `trusting managers to share information` and `not trusting managers to share information` (Jaccard about 0.83, contradictory definitions) are proposed and then judged on definitions and evidence. Everything else uses that same full comparison.

`passages_per_request`, `max_concurrent_requests`, and `passages_per_codebook_update` are separate. `batch_size` and `concurrency` remain as aliases. Setting `batch_size=1` and `concurrency=4` without the new names warns: concurrency no longer widens the frozen window. `strict_sequential_settings()` and `procedure="strict_sequential"` are the documented sequential baseline: document mode, one passage per request, one request in flight, commit after every passage, pipeline off, parallel documents off. Later passages see earlier codes.

A cached pair judgment is invalidated when either definition's text or revision id changes, or when that code's examples or assignments change. There is no token-overlap threshold. Negating `Employees trust...` to `Employees do not trust...` reopens the pair.

Sweep, in-batch, and merge-check prompts share one stance sentence: stance is context, not a merge veto, and it is not by itself a reason to alias. The open-coding example maps "I like my team" to `expressing liking for the team`. Automatic notes that only record reading order are `provenance`. They do not claim reflexivity.

`adjusted_rand_index` stays and documents its limit: different singleton codes on every passage score 1.0. `aligned_assignment_agreement` reports mean set Jaccard per passage after greedy label alignment, and nominal Krippendorff's alpha on the aligned passage-by-code matrix. Those different singletons score low. `policy_version` was `0.6.4` in that release.

## 13. Corrections in 0.6.5

A sweep chunk carries definitions and examples for its own codes and for similar codes retrieved from other chunks (`cross_chunk_limit`, default 8). The rest of the book is an index of ids and short labels. The first example quote is not copied into a second field, and request JSON is compact. On the review fixture (chunk size 20), user-message characters for 20, 40, 80, and 160 codes drop from 9,347 / 54,028 / 178,806 / 642,142 to 5,467 / 14,846 / 36,922 / 103,334. Filling the cross-chunk cap on every chunk is 5,467 / 18,788 / 44,782 / 119,030. `scripts/benchmark_sweep_payload.py` rebuilds that table.

Each code stores `reviewed_revision`, a fingerprint of its label, definition, revision, and example quotes. Frequency does not change it. A later sweep sends only codes whose fingerprint is new or changed. Every `full_audit_every` sweeps (default 4; 0 means consolidation only) and the final consolidation pass read the whole book.

Embedding vectors are cached by embedding model and the sha256 of the exact text. A batch embeds a codebook row only when that text changed. Concurrent workers and sweeps that ask for the same snapshot share one vector list.

`token_budget` is the assembled coding request plus reserved output. Instructions, retrieved definitions, examples, neighbors, metadata, and schema framing count. When the model exposes `encode`, that tokenizer is used. Otherwise the estimate is characters / 3. A passage that still does not fit is split on whitespace, or sent alone with a warning when no slice fits. `policy_version` is `0.6.5`.

## 14. Corrections in 0.6.6

`abatch` takes one slot per call from the same AIMD gate as `ainvoke`. A batch does not create a semaphore of its own. Two concurrent batches with a ceiling of 2 stay at 2 in-flight calls. `max_in_flight` is that peak, on `metrics.max_in_flight` and `metrics.activity.max_in_flight`. Embeddings keep a separate semaphore so retrieval does not hold the model slot.

Coding time is three numbers. `stage_seconds.coding` and `metrics.coding_wall_seconds` are wall time: each batch interval is counted once, and overlapping batches in a round contribute their union. `metrics.coding_seconds` and `metrics.coding_request_latency_seconds` are the provider latency of each model call. `metrics.coding_queue_wait_seconds` is time waiting on the gate, on a token bucket, or on retry backoff. A batch duration is not copied onto each passage and summed. `coding_p50` and `coding_p95` are percentiles of the per-request latencies. `metrics.request_latency_seconds` and `metrics.queue_wait_seconds` are the same split for every model call in the run, including sweeps.

Schema repair adds the usage of the failed response to the successful one. When more than one response was billed, `usage.attempt_usage` lists them. Prompt-log totals and `input_tokens` / `output_tokens` include those attempts.

`tokens_per_minute` reserves estimated input tokens plus reserved output before the call, then replaces that reservation with the provider's actual input and output. It does not consume one unit per request. `requests_per_minute` does, one per attempt. Zero leaves a bucket open. Both values are stored on the run metadata. `policy_version` stays `0.6.5`.

## 15. Corrections in 0.6.7

A reservation larger than `tokens_per_minute` or `requests_per_minute` raises `OversizedReservation`. The bucket cannot hold it, so waiting cannot succeed. `allow_oversized=True` waits until the bucket is full and then drains it.

Cancelling the owner of an in-flight embedding removes that future. Other waiters use `asyncio.shield`, so one cancellation does not cancel the shared call. The cache key is the model, a fingerprint of dimensions, endpoint, deployment, base URL, API version, and model kwargs, and the sha256 of the text. Two clients with the same model name and different dimensions do not share vectors. A batch whose width is mixed, or not the configured width, is not stored.

`token_budget` (default 6000) is measured after retrieval. The neighbors of every passage in the call are unioned, the messages are built, and the JSON schema plus a short repair allowance are counted with reserved output. The previous estimate used `top_k_codes + frequent_codes` of the longest entries and omitted the schema. The review measured four passages at 8,759 estimated tokens against 19,250 actually sent (up to 48 retrieved codes plus 8 frequent codes, against an estimate of 20). A batch over the budget is repacked into fewer passages. Retrieved codes are trimmed from the end of the view until the assembled request fits. Whitespace splitting is only for a passage whose text still does not fit with an empty view. A schema repair is measured again, and the repair note is shortened until it fits. The default moved from 4000 to 6000 because the schema (1,686 tokens for `BatchCoding`) and the repair allowance are now inside the budget.

`aligned_assignment_agreement` aligns a label with the same code key or a spelling variant. `trusting managers to share information` and `not trusting managers to share information` do not align. `mapping={label: other}` is a reviewed equivalence and does align them. `align_threshold` adds a Jaccard fallback that still refuses a polarity flip.

Sweep chunks do not repeat the global label index. `cross_chunk` is a bounded neighborhood (`cross_chunk_limit`, default 8). One `CandidateDiscovery` call sends ids and short labels for the whole book and returns candidate pairs. On the review fixture, chunk size 20, user-message characters are:

| codes | 0.6.4 | 0.6.5 repeated index | 0.6.7 | 0.6.7 + 8 cross-chunk codes |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 9,347 | 5,467 | 5,467 | 5,467 |
| 40 | 54,028 | 14,846 | 13,355 | 17,297 |
| 80 | 178,806 | 36,922 | 26,577 | 34,437 |
| 160 | 642,142 | 103,334 | 53,261 | 68,957 |
| 320 | 2,423,837 | 324,781 | 106,812 | 138,180 |
| 640 | 9,404,813 | 1,120,941 | 213,900 | 276,612 |
| 1280 | 37,091,765 | 4,145,141 | 429,196 | 554,596 |

From 160 to 640 codes the 0.6.7 column grows 4.02x. From 20 to 1280 it grows 78.5x. The repeated-index column grows 10.85x from 160 to 640. `scripts/benchmark_sweep_payload.py` rebuilds the table. `policy_version` is `0.6.7`.

## 16. Corrections in 0.6.8

Retrieval for a coding round happens once per passage and codebook snapshot, on the async embedding path. The planner used to call `rank()` on the event loop, which embedded the passage as a query and bypassed the embedding semaphore, retries, and timeout. Coding then embedded the same passage as a document. Both paths now embed a passage as a query, and the rankings are reused for budgeting and for the coding call.

Validation feedback is capped. A retry shortens that note before it drops codes. Coding, the validation retry, and schema repair are measured immediately before the provider call. A request that is still over `token_budget` after that trimming is sent. The run logs a warning and records an `oversized_request` flag on the run metadata and the review list, with the passage id and the estimated size.

A sweep requests a decision only for pairs whose definitions and examples are both in the request. The neighborhood stays at `cross_chunk_limit` (default 8). Remaining hinted partners are sent in further bounded comparison requests, so a pair is not dropped and is not decided without its partner's evidence.

`policy_version` is `0.6.8`.

## 17. Corrections in 0.6.9

Oversized passages are split to completion. The packer used to stop at the first slice that had no ranking, embed that slice, and start the whole pack again. After 64 restarts it raised `coding planner could not rank passage slices` even when each restart had ranked another slice. Candidate slices are measured without that round trip. The slices that are kept are ranked once, together.

Query embeddings are not one sequential list under a single batch timeout. Each query uses the shared embedding semaphore, its own timeout, and its own retries. A success is written to the vector cache before the rest of the batch finishes, so a retry does not embed it again. `aembed_queries` or `embed_queries`, when the provider has one, embeds the chunk in one call.

`scripts/benchmark_review_modes.py` runs an embedding-latency row unless `--embedding-delay 0`. That row sleeps on embedding calls and ranks passages as queries.

`policy_version` is `0.6.9`.

## 18. Corrections in 0.6.10

Query batches no longer use `asyncio.gather` for a group of provider calls. Gather raises the first error and leaves the other calls running; the cache handler then removed their in-flight entries, and the next request for that text started a second provider call. The one-by-one path and the batch-query path wait for every sibling before that error is released, and they do not clear an entry whose task is still running. A sibling another caller is waiting on is not cancelled with the owner. It keeps running and keeps its entry.

`scripts/benchmark_review_modes.py --unique-passages` makes the embedding row's passage texts distinct. The default corpus reuses four stems (4 query calls). With the flag, `--delay 0.05` embeds 20 queries, peak 4, in 1.019 seconds (23.55 passages/s). The reused row on the same settings embeds 4 queries, peak 3, in 0.887 seconds (27.07 passages/s).

`policy_version` is `0.6.10`.

## 19. Corrections in 0.6.11

Quote anchoring still searches the passage, then the neighbors, then the document, and still never drops a code. An exact hit is not automatically evidence for the passage being coded. A hit that does not overlap that passage is `evidence_outside_passage`, `needs_review`, and attributed to the passage that contains the span. The segment stores `coded_passage_id` and `attributed_passage_id`. Coverage, agreement, and assignment counts follow the attributed id. The outside flag is not a quote retry.

A checkpoint stores a manifest: corpus hash (document id, name, source, and text), context, model config, procedure settings, and `policy_version`. Resume compares it with the call. The default is to raise `CheckpointMismatch`. `allow_changes` or `continuation` writes a continuation event with the old and new values, warns, and updates the checkpoint metadata to the settings that will actually run.

Codebook snapshots point at canonical vectors keyed by content hash. The process keeps a bounded LRU of snapshots (default 8, `configure_embedding_cache`) and drops a run's snapshots at the end of `embedding_run_scope`. The vectors stay. One to fifty codes retain about fifty vectors.

Query ownership now shields sibling settlement, not only the first wait, and detaches every registered waiter in an outer `finally`. Document batches publish each successful chunk before the group error, the same way queries do.

Nominal Krippendorff alpha no longer returns 1 when fewer than two units are usable. `[[1, 0]]` is 0. No paired ratings is undefined (`alpha` null, `undefined` true). The aligned metric still omits jointly-absent cells.

The Muse runner fills the unit grid from human excerpt spans and the reviewed code mapping. Without that alignment the agreement metric is unavailable.

The review fingerprint covers every assignment. Prompt examples are a separate diverse sample of bounded size.

`commit_passage` shallow-detaches codebook entries and shares segment dicts and history-event dicts. Counts update for the codes that commit touched. A later mutation still cannot rewrite the caller's book. Measured against 0.6.10, one commit on 400 codes and 3,200 segments dropped from about 43 ms to under 1 ms, and 800 successive new codes from about 11.4 s to about 0.9 s.

`policy_version` is `0.6.11`.

## 20. Corrections in 0.6.12

Resume distinguishes two kinds of manifest change. Research-context wording, model settings, and performance settings (concurrency, batch size, token budget, rate limits, `review_mode`, and `parallel_documents`) may continue with `allow_changes` or `continuation`. The continuation event stores the old and new values. Corpus identity, `unit`, and `seed` are not continuable. Those already determined the passages stored on the checkpoint and the order of that queue. `allow_changes` still raises `CheckpointMigrationRequired`. Start a new run. There is no procedure that inserts a document into a queue that was already segmented.

`review_mode` and `parallel_documents` do not change passage boundaries. `plan_coding` with the same corpus, unit, and seed returns the same passage ids, text, and offsets in document mode, wave mode, and with parallel documents. What changes is the initial queue and coding order, and parallel documents also build per-document branch queues. That plan is stored once, in `segment_and_shuffle`. Resume continues from the next node and does not run that node again, so a later change does not rewrite the stored passages, coding order, queue, or branch queues. `review_mode` is read again when the next batch is taken from the stored queue, so it can change how the remaining passages are grouped. The stored `parallel` flag and branch queues are what a parallel run keeps using. A continuation that flips `parallel_documents` records the new value and does not move the remaining passages onto new branch queues.

The model block of the manifest is the instances that will run, not only the `temperature=` argument. `chat_model_settings` and `embedding_model_settings` read model id, provider, temperature, `top_p`, max tokens, embedding dimensions, endpoint, and deployment. Names that look like keys, tokens, or secrets are omitted, and `model_kwargs` is passed through `redact`. Metadata stores those values, so a coder constructed with `temperature=1` is not recorded as `0.0`.

Prompt examples are chosen from the full assignment list. `_apply_counts` no longer stops at eight. When a sweep reviews a code it stores `reviewed_assignment_ids` and appends that list to `reviews_seen`. The next prompt includes every assignment added since then, then fills up to the requested limit with one example per remaining document, then quotes that overlap the chosen set the least, then passage order. Document diversity runs before that passage-order fill.

Quote attribution overlaps each matched span with each passage. The bounding box from the first span to the last is not used, so a long passage that only occupies the gap is not selected. Several overlapped passages are `attribution="multi"` and coverage counts the code on each of them. No overlapped passage is `attribution="unresolved"`: `attributed_passage_id` stays empty, and coverage, agreement, and the Adjusted Rand index do not place the code on the passage that was being coded. The `evidence_outside_passage` flag still keeps the code.

Muse `unit_grid_from_annotations` places an excerpt by exact match, then by whitespace collapsed to a single space with offsets mapped back to the original text. Repeated excerpts use `occurrence` or `order` (0-based) or `offset` / `start` / `start_offset` / `char_start`. Without one of those, the excerpt is `ambiguous` and is not aligned. `alignment_coverage` is aligned annotations divided by annotations considered. A passage touched by an unaligned or ambiguous annotation is an unknown unit and is excluded from the grid. Coverage below `ALIGNMENT_COVERAGE_MIN` (0.8) returns `available` false and `reason` `alignment_coverage_below_threshold` instead of a score.

`configure_embedding_cache(vector_limit=..., vector_bytes=...)` is an optional LRU over canonical vectors. `vector_bytes` is 8 bytes times the number of dimensions. Vectors referenced by a retained snapshot are kept even when that leaves the cache over the cap. Both limits default to none. `commit_passage` still copies the codebook dict and shallow-detaches every entry: the sweep mutates definitions in place on the new state, and those objects must not be the ones stored in the previous checkpoint. The segment list is a new list; unchanged segment dicts stay shared. Counts on an append-only commit extend the entry's assignment list instead of scanning every segment. `sync_counts`, used when a merge or split moves assignments, still scans.

`policy_version` is `0.6.12`.

## 21. Corrections in 0.6.13

Speaker labels match at the start of a line after optional indentation. A label also matches in the middle of a line when it follows sentence punctuation and the label is short or already known (`Interviewer`, `Participant`, and the other role names, or a label already used at the start of a line). ` Participant:` and `best.   Interviewer:` are separate turns. A line-start label is words without sentence punctuation, so the words before that mid-line label are not swallowed into it. If the last word of a line-start label is a known role, only that word is the label: `download the results   Interviewer:` is `Interviewer:`, and `download the results` stays on the previous turn. `Note:` after a period is not a speaker. Answer characters that still sit in no passage produce a warning and `metadata.uncovered_answers` (`doc_id`, `chars`, and the text). Non-whitespace text before the first speaker label, such as `Participant 02`, is attached to the first passage and recorded the same way on `metadata.attached_preambles`.

A coding reply that omits a passage is not left failed. That passage is requested alone, up to `missing_slot_retries` (default 2). Each attempt is logged and does not read or write the response cache, so a second identical solo prompt calls the model. Ids that stay failed are `metadata.summary.failed_passages` and a warning that names them.

`anchor_quote` searches the other passages in the same request before it returns `unanchored`. The hit is source `request`. It is kept and flagged `evidence_outside_passage`, and attribution is that passage. The quote policy is unchanged: flag, never drop, never block coding.

If the attributed passage already has that code and that quote, the commit does not append a second segment. `provenance.coded_passage_ids` lists every passage whose coding produced the quote, including the one being coded. Coverage counts the quote once. The coding prompt says to quote only the target passage, not the context around it.

`short_turn_chars` (default 60) merges a short speaker turn into the adjacent answer. `Participant: Ovid.` is the example. The merged passage is one slice of the source, so the characters between the turns stay in the passage. Zero disables the merge. Segmentation changes, which a new run picks up. Resume does not re-segment, so a checkpoint keeps the passages it stored.

The coding prompt, which document mode uses, tells the model to code each passage separately and not to write one code spanning several passages.

`policy_version` is `0.6.13`.

## 22. Corrections in 0.6.14

A speaker label is a word, a number (`Participant 02`), or initials (`J.D.`, `J. D.`). The initials alternative is tried first so `J.D.:` is not read as the word `J`. A numbered label whose other words are on the mid-line blocklist (`Table 1`, `Figure 2`, `Chapter 3`) is not a speaker. Dotted abbreviations (`e.g.`, `U.S.`, `i.e.`) are not speakers. A line-start label that ends in a known role still shrinks to that role, and a mid-line label still has to follow `.`, `!`, or `?`.

`_source_text_coverage` does not call the speaker parser. Every non-whitespace character of the document is compared with passage spans and with question context (the question string, plus the `Label:` that introduces it on that line). Passage text wins where they overlap. A looser speaker detector flags a turn whose label sits only inside a question body (`Participant 02:`, `J.D.:`, an extra `Interviewer:`). `Note:` and `Table 1:` inside a question stay ordinary prose. Uncovered characters are warned and stored as `source_uncovered` (`doc_id`, `chars`, `text`). Flagged turns are `question_context_turns`.

`anchor_quote` carries `located_doc_id` and `located_doc_name` from the passage that contained the hit. `commit_passage` writes that document id, name, and respondent metadata onto the segment. Provenance keeps the passage and document that were being coded: `coded_passage_id`, `coded_doc_id`, `coded_doc_name`, and `coded_documents`. Frequency `document_ids`, respondent counts, and `highlights.csv` `document` follow the located document.

Duplicate detection runs for every incoming assignment. Identity is the code key, the evidence document, and the matched spans (`quote_spans`, or the bounding `doc_start`/`doc_end` when there is one span). Quote text is not an identity. Spans that were never anchored do not merge. A hit copies the coded passage and document onto the existing segment's provenance. Inside-passage first and cross-passage first both leave one segment and frequency 1.

`configure_embedding_cache(vector_limit=...)` still evicts least-recently-used vectors. A request copies the vectors it will return before that eviction, on the batch-query path, the single-query path, and the document path. Eviction changes only what a later request can reuse. Two queries in one request with `vector_limit=1` return both vectors.

`select_prompt_examples` returns at most `limit` rows. Unsent assignments stay fresh. When there are more fresh rows than the limit, the batch prefers a new document and then the quote that overlaps the chosen set the least. `PROMPT_EXAMPLE_LIMIT` is 2, and coding prompts and sweeps use it. `record_review` stores the ids that were sent on `reviewed_assignment_ids` and on `reviews_seen.sent_assignment_ids`. `available_assignment_ids` is every id that existed when the review was built. `code_needs_review` stays true while any current assignment was not sent. A fingerprint with no per-assignment bookkeeping is unchanged.

Solo retries of a missing batch slot are tagged `missing_slot_retry` on the prompt log. `missing_slot_retry_cost` sums those calls into the run metrics. The batch call is not included.

`SHORT_TURN_CHARS` is 0. `I quit.` is a meaningful incident, and a character cap cannot tell it from filler such as `Ovid.`. The merge stays off until that has been evaluated. A positive `short_turn_chars` still merges a short turn into an adjacent answer, and `_gap_has_question` refuses the merge when an interviewer, moderator, or question label sits between the spans. A 0.6.13 checkpoint stored `short_turn_chars` 60. It is in the procedure manifest and not in the structural paths, so resume without passing 60 raises `CheckpointMismatch` unless `allow_changes` or `continuation` is set. Resume does not re-segment.

`policy_version` is `0.6.14`.

## 23. Corrections in 0.6.15

Consolidation used to have an unconditional edge to export. A code can still have unsent assignments after that pass, because each review shows at most `PROMPT_EXAMPLE_LIMIT` (2) examples. Eleven documents, a fake coder, `sweep_every=100`, and `pipeline=False` produced two sweeps and sent 4 of 10 assignments on `packing bags`, with `code_needs_review` still true and no warning about that.

After consolidation, `review_evidence` loops while `code_needs_review` is true. Each pass is an incremental review of the codes that still need it. It does not repeat label-only discovery. A sweep call in that loop is sent only when both allowances can take it:

- `max_final_review_calls` (default 8) counts follow-up sweep calls, not the consolidation pass and not coding calls. The call cap is checked first. The default covers the reviewer's case: 6 assignments remain after consolidation, and 3 calls send them at 2 per call, with room left. A code of about 20 assignments is the size the default is aimed at (the end sweep and consolidation send 4, and 8 follow-up calls send 16 when that code is alone in the chunk).
- `max_final_review_tokens` (default 24000, four times the default `token_budget`) is the estimated input of those prompts, about three characters per token plus a small per-message frame. A prompt whose estimate would pass what remains is not sent. Provider usage stays on the prompt log. When it is larger than the estimate, the charge is still the estimate that gated the call, so the cap and the gate use the same number.

Stopping criteria, in order: nothing still needs review (`review_complete` true, `review_stop` `complete`); the next call would exceed the call allowance (`call_allowance`); the next prompt would exceed the token allowance (`token_allowance`); a follow-up pass sends no new assignment id (`no_progress`). `max_final_review_calls=0` or `max_final_review_tokens=0` does not start the loop. The consolidation batch still runs when the book has at least two codes.

When the loop stops early, metadata `consolidated` is false. Warnings say the run is not fully consolidated and name the pending counts and ids. The result also records `review_complete`, `pending_counts`, `pending_assignment_ids`, `assignments_available`, `assignments_sent`, `assignments_pending`, `review_calls_used`, `review_tokens_used`, and `review_by_code` (available, sent, and pending per code). A sweep-log row with `consolidation: true` means that pass ran. It is not a claim that every assignment was sent.

A successful embedding is published to the shared future before `_store_vector` can evict it. Waiters do not read the vector back out of the cache, so a later sibling can fill `vector_limit` and a failed sibling can raise without turning the published success into `ConnectionError`. The batch-query, single-query, and document paths all publish from `_hold_stored`.

`policy_version` is `0.6.15`.

## 24. Corrections in 0.6.16

A coding action below `review_below` is applied. `needs_review` records `low_confidence`, and the review operation stays on `review.json`. The score is not a probability. In-vivo clearing, definition rewrites, theme-assignment confidence, and `min_merge_confidence` are unchanged.

`assign_codes` keeps every theme membership. `primary_codes` on a theme lists the codes for which that theme was first. `assignment_feedback` no longer says the first assignment was the only one kept. Theme prompts, including the codebook line, the request note, the review note, and the leftover note, say a code may belong to more than one theme and that the first theme is primary. The review prompt calls `max_theme_share` a soft presentation target. Exceeding it still only produces `_dominance_warning`. The protocol id is `theme-report-1`.

`apply_sweep` still groups a component with union-find. The label that remains is `_survivor_choice`: the model's target, or the lexicographically smaller key when the merge is trivial, the judgment is missing, or the component's directions conflict (the tie-break is the smaller target, then the smaller member). Retired labels become aliases. Segment provenance is copied through and not rewritten. `clustering.union_roots` is a separate helper and still returns the lexicographically smaller key.

`plan_coding(..., code_nonsubstantive=True)` queues uncertainty, refusal, `n/a`, and punctuation-only text and sets `nonsubstantive` and `nonsubstantive_kind`. Empty text stays `exclusion_reason=empty`. `code_nonsubstantive=False` restores exclusion. The flag is on run metadata and on the procedure manifest. A checkpoint that omits it compares as true. Incident rows carry the same two fields. Segments copy them from the passage.

`memo_review_reasons` returns `None` for a blank note or a placeholder, `["thin"]` under 20 words, and `[]` otherwise. `commit_passage` stores a real memo either way and sets `needs_review` from those reasons. A low memo confidence adds `low_confidence` and a review operation. The memo is still stored.

The README and [methodology.md](methodology.md) describe the package as automated initial coding plus a theme report. The speed table is a simulated fake-coder benchmark. Live accuracy against human coding for 0.6.x has not been measured. The last live numbers are 0.5.4. `pipeline=True` codes against the pre-merge snapshot.

`policy_version` is `0.6.16`.

## 25. Focused coding in 0.7.0

Focused coding is a new graph. It is not a node on `open_code`. Selection is deterministic and is a proposal. The default rule is top-k with ties broken by code id, and it does not raise every code unless `focused_raise_all` is set. A high raised fraction is warned. `researcher_checkpoint` and `claim_checkpoint` call `interrupt()` inside the node, resumed with `Command(resume=...)`. The reread allowance is split across developing assignments, competing interpretations, passages outside the selection, and a reserved claim-compete share that runs after properties are written. The four shares sum to 1 and default to one quarter. Each of the first three queues is sized to its reserved share of the call allowance. Neighbor, outside, and negative-case limits are optional caps. Competing readings are seeded from initial codings before claims exist. Compete output requires a keep, revise, or drop verdict for every initial code and a strongest alternative. Those codes are listed on equal footing. Revise and drop stay flagged proposals. A kept raised code is flagged `alternative_named_not_chosen` only when `alternative_fit` is `stronger` for a different code. A strongest alternative that names the kept code is `alternative_is_kept_code` and is invalid. The compete prompt excludes kept codes and codes already assigned on the passage (`allowed_alternatives`, `excluded_alternative_code_ids`). A code id and a label that name different codes are invalid. Claim compete searches the property's category, opposite poles, lexical or stored-embedding similarity, and low-confidence or flagged assignments, and still queues an unreread passage. That round uses a claim-test prompt that includes the specific claim and records supports, contradicts, qualifies, or irrelevant. contradicts and qualifies are negative cases and set `evidence_condition` to `disputed`. They do not set `claim_support`. A reallocated outside or compete call does not use that schema. Unused reserved calls are not described as reallocated. The warning denominator is the call allowance. Revise and drop proposals are consolidated per code. A revise together with a drop, or two revised definitions, is `proposal_conflict`, unapplied, and excluded. `proposal_stability` compares them across runs. The request timeout cancels the async wait, the sync future, and the event-loop bridge. A timed call runs off the package loop, and the provider client bounds connect, read, write, and pool. Schema token counts used while splitting a passage are cached. Provider-reported token usage is stored separately from the pre-call estimate. Properties merge on a normalized key, and property stability reports category-level and name-level overlap across runs. A near-duplicate new code is flagged. A property category that is not assigned on the cited passage is flagged beside the anchoring rate. Anchoring is not category fit and not claim support. Loading an open-coding folder continues segment and memo ids (`m00001`). The same span is a confirmation on the existing assignment. A different span is nested on that assignment, and a narrowed quote is flagged. `complete` means every corpus passage was reread. `complete_for_queue` means the queue finished and names passages never queued. `focused_request_timeout` bounds one call, including streaming and retries. The output folder writes segments, assignments, the prompt log with the raw model response, `focused_claims.csv`, and `downstream_claims.json`. Properties and dimensions are `category_development.json` (`state_version` 1, `relations` null). Each row has a disposition, an evidence condition, downstream eligibility, a category revision id, and an evidence revision id. `claim_support` stays undefined until a researcher sets it. Axial relations, a core category, and a saturation decision are not in this version. `judgment` stays null. Non-substantive detection treats Unicode letters and numbers as text. `seed` orders a queue that fits. `focused_sampling_seed` samples only when a list is longer than its slot.

Package 0.7.6 keeps timed model calls on one shared loop, enforces the request timeout on the waiting side and with a graph-step watchdog, and records a timeout as incomplete. A claim is tested only against a passage that claim retrieved. An absence-of-mention pole is flagged and not tested. `claim_test_counts` accumulates. `proposal_stability` is undefined when there are no proposals. A compete round that returns only keep is flagged `compete_keep_only`. The token estimate stays a ceiling and is reported beside provider usage.

Package 0.7.7 keeps a provider that accepts a connection and sends no body inside the request timeout. The reported error keeps that attempt's duration. A timeout is not retried unless the wall budget can hold another full attempt, and a retry names that shared deadline. The checkpoint records status `incomplete`, stop `timeout`, the step, and a partial export when an output directory is set. A same-code passage is a claim test only when it also shares content or an embedding; an empty quote is not sent; `claim_test_relevance` reports the rate. `supports` against a pole that states a shortfall the passage does not state is `claim_test_invalid`. `proposal_stability` compares revise and drop of the same code and lists `cross_seed_conflicts`. Property stability reports how many properties each run named. The token ceiling is unchanged; `focused_tokens_calibrated` is the ceiling divided by 1.28 and is not the stopper.

Package 0.7.8 accepts a claim-test judgment only for the requested passage and claim, and only when the quote anchors in that passage. Anything else is logged as `claim_test_invalid` and does not dispute the claim. A blocked model-loop thread is abandoned so the next call can run; killing that thread requires process isolation. httpx clients are built on the loop that uses them. A shortfall comparison is `possible_direction_mismatch`, not an automatic invalidation. Claim review packs into the remaining token allowance. The reviewer sees challenging passages within an evidence budget. The token figure is a heuristic. The 1.28 adjustment is empirical, and the ratio uses provider input tokens. Revise and drop proposals stay unapplied. A run that does not interrupt can leave a disputed claim pending.

Package 0.7.9 cuts off a provider that holds a connection open with a monotonic deadline thread that does not depend on an asyncio loop. On expiry the model loop is abandoned and the waiter is unblocked; `process_isolation` still kills a picklable child. httpx clients built during the call get a socket timeout. A timeout returns an incomplete result (`raise_on_timeout=True` still raises). `focused_calls_used` counts logged calls, and `focused_not_reread` lists queued passages that were not committed. A claim call needs overlap with the claim's evidence, not only its code. Counterevidence is preferred. Low-relevance pairs are skipped and the calls are reallocated. An irrelevant judgment with an empty quote is valid. `focused_synthetic_counterexample` injects one labeled passage for live evals and keeps it out of research exports. Keep-only compete rounds stay flagged. Property-name stability stays the measured score. The 1.28 token adjustment is the v23 total-token condition; the v24 input-token ratio was about 1.38. Neither is general.

`policy_version` was `0.7.9` in that release.

Package 0.8.0 adds `get_graph("axial")` and `get_graph("selective")`. They are not nodes on open coding, themes, or focused coding. Relationships use the five paradigm roles, cite a passage, and stay proposals until a researcher resumes `interrupt()`. `decision_mode="llm"` writes model fields and an audit row, and leaves human `claim_support`, `researcher_saturation`, and `researcher_core_id` unset. Saturation indicators are a curve. `judgment` stays null. A timeout returns an incomplete result unless `raise_on_timeout=True`. The design of this slice is [design-axial-0800.md](design-axial-0800.md).

`policy_version` was `0.8.0` in that release.

Package 0.8.1 measures code-frequency stabilization as total variation between consecutive documents' normalized assignment counts. Presence-bit reports from 0.8.0 are not comparable. A combined axial-then-selective run keeps the axial audit, and selective export does not rewrite the axial JSON. Call budgets count model calls: proposal calls are `ceil(pairs / batch_size)`, unused calls move, and the counterexample pair capacity is calls times batch size. `max_decision_calls=0` sends no reviewer call. The storyline names the core the decision mode actually selected. Proposal evidence is shared across both categories before the limit is filled. Provider HTTP clients are rebuilt only from `ProviderClientFactory` configuration. Axial and selective calls record provider token usage under `max_axial_review_tokens` and `axial_token_budget`. A stall before the first model call is on the same monotonic deadline as the request. A contradicts verdict that does not engage the claim's subject does not dispute it. Reallocation metadata matches the warning, and reallocation stays inside the token allowance.

`policy_version` was `0.8.1` in that release.

Package 0.8.2 keeps a paraphrased contradiction as a negative case. Missing subject overlap is `claim_subject_uncertain` and still disputes. Only explicit irrelevance language, not a phrase such as "is about", flags `rationale_indicates_irrelevance`, and that row stays as `disputed_pending_review` rather than being dropped. `max_axial_review_tokens` is one shared estimated-token allowance for coding, counterexamples, memos, the storyline, and decision review. Schema repair and retries acquire that allowance and a call from the same provider ceiling; a repair that does not fit keeps the failed output for review. An oversized batch is shrunk before its items are abandoned, and proposal trimming keeps evidence for both categories or defers the pair. A finished graph is not the same as analytical coverage: pairs that were not examined stay in `unexamined_pairs`, and a run that did no analysis is `budget_limited` rather than `complete`. Total variation between consecutive documents describes the difference between those documents. A low figure does not establish theoretical saturation, and it is separate from a human confirmation.

`policy_version` was `0.8.2` in that release.

Package 0.8.3 keeps the same budget across a human checkpoint. Consumed calls and estimated tokens are stored on the graph state and restored into a new runtime before any resumed node runs, including a resume in a new process. Coding calls and reviewer calls have separate ceilings. They share one token allowance. A repair spends the stage that originated it, so an LLM reviewer does not give coding a larger call budget. `complete` requires every requested stage to finish its work. A skipped review, an omitted storyline, or a selective failure after successful proposals is `budget_limited` or `incomplete`, with a status for each stage. Provider usage is recorded for every response, including a repair that fails to parse. A pair with no segment evidence is deferred as `missing_category_evidence` instead of raising inside the packer.

`policy_version` was `0.8.3` in that release.

Package 0.8.4 checkpoints each proposal request before the next one is sent. A timeout keeps the responses already written and the allowance already consumed, including the attempt that timed out. A resume continues inside that ceiling. Memos record how many were requested, produced, omitted, failed, and empty. A failed chunk stays unresolved, and a memo with no text does not make the stage `complete`. Usage is added once for each response the provider actually returned. A repair that does not fit the request budget is a preservation record: it does not reuse the failed response's token counts, and the warning says it was not sent. A decision-review response is counted when it fails to parse. Each requested pair is deferred when either category has no passage in that request, and the row names those category ids.

`policy_version` was `0.8.4` in that release.

Package 0.8.5 requires a quote to name both categories before `citation_anchoring_rate` is 1, and reports `endpoint_anchoring_rate` when the quote names only one side. The storyline packs categories and relationships into `axial_token_budget`, sends the passages those rows cite, and records what was omitted. Rejected relationships are excluded or flagged, and `storyline.json` carries the core and its provenance. An empty core is `unset`. Memos and the storyline have separate call counts inside `max_selective_review_calls`, with one call reserved for the storyline when both run. A memo that outruns its quote is flagged `memo_quote_unsupported`. A combined run keeps the coding calls on `provider_budget`. Negative-case search is the share of relationships that received a judgment. Axial counterexample verdicts use the same irrelevance check as focused claim tests. `code_frequency_shift` names the total variation sent to the saturation reviewer; 1 is maximal change, and the model assessment does not set `judgment`. A zero property or relationship rate is `undefined_partial_coverage` when pairs remain. Obvious pairs rotate with the seed, and selective calls use their own prompt and stage. Focused claim candidates need a subject stem when the claim has one. Reallocation counts the calls that moved. Focused metadata records `stage_outcomes` and a focused budget stage. An automated saturation resume is actor `automated`.

Package 0.8.6 does not accept a relationship unless both endpoints are evidenced. The match is a content stem against the category label, definition, or segment quote, or an embedding at or above 0.55. `from` and other stopwords are not a match, and `matched_tokens` records the stems. A one-sided row is kept as `provisional` with `acceptance_withheld` and is shown to the reviewer and the storyline. Counterexample retrieval requires both endpoints or an embedding; a round with no relevant judgment leaves the negative-case rate null. Storyline statements that cite an unsent passage, or that cite no relationship, are flagged. Memos state evidence strength, and `memo_accounting` separates memos produced in this run from pre-existing ones. An omitted seed is derived from the corpus. Pair coverage names the denominator. A finished focused queue is stage `complete`. The allowance comparison uses a padded observed input-token ratio and does not open the ceiling for a ratio above 3. A saturation window below 3 documents is not interpretable. The synthetic passage no longer announces itself in the model-facing text; the audit keeps the label and the limitation.

`policy_version` was `0.8.5` in that release.

Package 0.8.7 keeps a relationship whose quotes come from different respondents, and it does not treat that pair as citation support. `citation_anchoring_rate` is 1 only when both quotes sit in the cited passage and both endpoints are anchored. The split rates keep same-passage support separate from cross-passage quotes, and the cross-passage rate stays null. A causal or consequence accept on those quotes becomes `cross_respondent_pattern`. The reviewer is told whether the quotes share a passage, and a "same passage" rationale or memo is flagged when they do not. A generic stem does not anchor by itself. Blank proposals and empty statements are counted and are not relationships. A core with no relationship endpoint is `core_unsupported_by_graph`, and core stability across runs stays undefined with `n`. A counterexample or claim stage with no calls or no relevant judgment is `complete_no_negative_search`. A share of the token ceiling is reserved for the storyline. The model does not receive the synthetic flag, the audit label, or text that announces the injection. `tokens_estimated` is the raw heuristic and `tokens_estimated_compared` is what the ceiling uses. Temperature 0 is recorded and is not a determinism guarantee. The saturation reviewer sees window-wide rates labeled apart from the last document. `metadata.seed` matches the seed that planned the pairs.

`policy_version` was `0.8.6`. `policy_version` was `0.8.7`.

Package 0.9.0 adds a theoretical-sampling graph over the corpus already in hand, and an explicit saturation decision that still leaves `judgment` null. Directions are ranked from categories, relationships, indicators, coverage, and memos. Passage proposals are retrieved from passages that are not already evidence. A missing passage is a researcher action: the package does not collect new data. `decision_mode='llm'` writes `model_direction_ids`, `model_passage_ids`, and `model_saturation_judgment`, and does not write the researcher fields. A model assessment is withheld below 3 documents or below 2 sampling rounds, and also when pair coverage is too thin to read: 10 or more planned pairs, with coverage below 0.05 or fewer than 2 relationships. The stopping-criteria report names met, unmet, and undefined without declaring saturation, including a `pair_coverage` row. Axial and selective coding were not rebuilt. The same release records the v29 citation, memo, counterexample, and saturation-review fixes on those graphs. A limited Glaser family list can sensitize a sampling query. It is not an axial role. Constant comparison across sampling rounds is the indicator delta on the ledger. The design is [design-sampling-0900.md](design-sampling-0900.md).

`policy_version` was `0.9.0`.

Package 0.9.1 replaces the rule that a relationship is accepted only when both quotes sit in one cited passage. Each relationship now carries `evidence_tier`. Tier 1 is a hypothesis and is not accepted as supported. Tier 2, `pattern_supported`, needs the configured case floor (default 2), with a verbatim quote for both endpoints in each case. Accepting it does not verify it and does not license "leads to". Tier 3 is an attribution plus a rival check. Tier 4, `verified`, is a researcher promotion with a named scope. A model cannot set it. `citation_anchoring_rate` is null unless a tier of 2 or higher qualifies, and the per-tier rates are null when that tier is empty. The design is [design-relationship-tiers.md](design-relationship-tiers.md).

`policy_version` was `0.9.1`.

Package 0.9.2 makes negative-case search able to return a respondent who has one side of a link, an opposite pole, or a different outcome. The 2x2 counts stay descriptive. A contradicting judgment flags the relationship and a qualifying judgment adds a condition. Neither is dropped. The rate stays null when no relevant judgment was made. A core has to be an endpoint of a tier-2 relationship or it is `core_unsupported_by_graph` and is not the center of the storyline. Coverage of that core is null when the graph is empty. The design is [design-negative-cases-092.md](design-negative-cases-092.md).

`policy_version` was `0.9.2`.

Package 0.9.3 classifies a negative case for the respondent after that respondent's passages are combined, and it keeps a split interview from being queued as endpoint absence. A budget that leaves plausible candidates unqueued or unattempted is `budget_limited`. Sampling measures each initial request and does not send one that exceeds the per-request budget or the total allowance. A malformed sampling response is an incomplete result. Axial, focused, sampling, and selective graphs derive a step allowance from the planned requests and, if that limit is still hit, return the partial result. The design is [design-review-fixes-093.md](design-review-fixes-093.md).

`policy_version` was `0.9.3`.

Package 0.9.4 keeps a sampling reread on the direction the researcher or the model paired with that passage. A passage selected for two directions is two rereads, and each one spends a coding call. A quotation found in a complementary passage is stored with `quote_passage_id` and offsets, while `passage_id` stays the selected candidate. Older records that omit `quote_passage_id` are still read. Axial coding reconciles pairs that have no category passage before the proposal loop, and the derived step allowance includes a defer or split return for each planned pair. A step limit that is still hit returns the partial result. The design is [design-review-fixes-094.md](design-review-fixes-094.md).

`policy_version` was `0.9.4`.

Package 0.9.5 stores each sampling reread before the next request. A later timeout keeps the completed note, its usage, and its prompt row, and lists the failed and unattempted pairs. Resume does not send a reread that already has a note, and it does not reset the allowance. An explicit model selection is the set of rereads. A flat id list that disagrees is reported, and the stage is incomplete when any requested pair was dropped. The design is [design-review-fixes-095.md](design-review-fixes-095.md).

`policy_version` was `0.9.5`.

Package 0.9.6 checkpoints a malformed sampling reread before its repair, so a repair timeout keeps the original response and its usage. Resume sends the repair and does not repeat the initial request. The pair that is about to be sent is recorded first. A timeout names that pair, and a pair already withheld stays unsent. An omitted flat id list is derived from the selections. Only a nonempty list is compared. The design is [design-review-fixes-096.md](design-review-fixes-096.md).

`policy_version` was `0.9.6`.

Package 0.9.7 checks causal and consequence wording, including the role `consequence` or `cause`, before a tier 1 or 2 relationship is accepted, and checks a reviewer who says no causal language was used. `core_stability.n` counts named cores and includes the current run. One distinctive stem does not count as a same-passage anchor. An absence candidate cannot contradict a relationship. Negative-case rows and the aggregate share one record, and prompt rows are split into calls and unsent rows. Sampling call ceilings start at zero for the sampling stage. A selection request is packed to `sampling_token_budget`. Pairs beyond the coding allowance are unsent with reason `call_allowance`. Reread notes are written onto the ledger, and the next round drops a direction those notes closed. A focused claim round with plausible candidates left unsent is `budget_limited`. Memos that echo the prompt are flagged, and a rejected row is not sent for a memo. A non-causal storyline sentence does not flag an accepted relationship. The raw focused token estimate and the guarded ceiling figure are both reported. The axial token ceiling is unchanged, and a planning estimate warns when it is short. An irrelevant or missing synthetic check is `not_interpretable`. The design is [design-review-fixes-097.md](design-review-fixes-097.md).

`policy_version` was `0.9.7`.

Package 0.9.8 reads a reread judgment only from the structured field. An irrelevant passage is not offered again for that direction, and the direction stays open until a person rejects it or a model records a close with a reason. Quiet rounds still read `analytic_new_direction_keys` when that field is present, and that list omits only an explicitly closed key. `model_sampling_pairs` keeps the validated selection. The coding-call cap applies only when `pursuit` is `reread`, and the pairs that will be reread are `scheduled_reread_pairs`. `brief_only` does not become `budget_limited` because no reread was requested. A selection repair reuses the packed payload, so proposals removed to fit `sampling_token_budget` are not restored. The design is [design-review-fixes-098.md](design-review-fixes-098.md).

`policy_version` was `0.9.8`.

Package 0.9.9 keeps a sampling question on a stable key. A relevant reread is an observation of that question. It does not wrap the key in another `reread_follow_up:` direction. A new question is added only when the note's `follow_up_kind` and `follow_up_target_id` name a different recorded target. The note text is not matched. A completed `(question, passage)` pair is stored on the ledger and is not reread again. Older nested keys are normalized to the stable question when the ledger is loaded. `export_gate_relationships` lists accepted and withheld relationships with the rules already stored on each row, and `score_gate_adjudications` leaves a rate null when no adjudicated rows support it. The gates, `pattern_min_cases`, and the axial ceiling are unchanged. The design notes are [design-review-fixes-099.md](design-review-fixes-099.md) and [design-gate-evaluation.md](design-gate-evaluation.md).

`policy_version` was `0.9.9`.

Package 0.9.10 plans axial pairs that already have evidence before the seeded fill, and it reports that coverage. `max_axial_review_calls` still defaults to 8, so a default series examines a small fraction of the pairs. A corpus where no pair can reach `pattern_min_cases` records a warning. The minimum stays 2. `case_id_of` reads `metadata.respondent_id`, so question-documents from one respondent are one case. A one-stem lexical overlap is not offered as cross-case evidence. A tier-1 hypothesis may rest on one passage or one respondent, and a statement stitched from unrelated passages is still rejected. Causal wording below tier 3 is not downstream eligible. An absence qualification keeps its reason. A same-passage stem-only quote stays proposed, with scope `same_passage_stem_only`. 0.9.15 keeps that scope and does not treat it as supported evidence. Sampling baselines carried token estimates, packs the saturation request, and leaves an unlisted human direction open. Quiet rounds are not evaluable while selected pairs remain unsent. Memo counts balance. The token plan estimates the planned round. A synthetic check with no relationship is `not_run`. Proposed gate rows are listed. A focused timeout reports how long it ran after the last completed call. The axial ceiling stays 24000. The design is [design-review-fixes-0910.md](design-review-fixes-0910.md).

`policy_version` was `0.9.10`.

Package 0.9.11 keeps the full list of stem-only lexical matches in the run export and sends the proposal model a count. Those optional diagnostics are dropped before any passage is shortened. Quiet rounds stay `not_evaluable` when a selected pair was refused for a request budget, a token allowance, a parse failure, or an unresolved timeout, as well as for `call_allowance`. Pair evidence is classified from tokens and case memberships computed once, and that classification is skipped when no proposal is scheduled. A weak overlap does not hide a later pair's grounded passage. `pattern_min_cases` stays 2 and the axial ceiling stays 24000. The design is [design-review-fixes-0911.md](design-review-fixes-0911.md).

`policy_version` was `0.9.11`.

Package 0.9.12 clears `reread_stop` after a reread timeout is recovered, and keeps that timeout on `reread_recovered_timeouts` and in the attempt log. An unresolved timeout still leaves quiet rounds `not_evaluable`. Unsent, failed, and scheduled reread pairs carry the stable question key, and the quiet-round exemption reads that key before the round-specific direction id. `pattern_min_cases` stays 2 and the axial ceiling stays 24000. The design is [design-review-fixes-0912.md](design-review-fixes-0912.md). `policy_version` was `0.9.12`.

Package 0.9.13 retires a recovered pair from `reread_failed_pairs` when that pair succeeds, including when a later pair is withheld for `call_allowance`. The timeout stays on `reread_recovered_timeouts` and in the attempt log. The current `reread_stop` is the remaining blocker. Gate export keeps reviewer `disposition` and `downstream_eligible` apart and copies withholding flags. `downstream_precision` and `downstream_withheld_supported_recall` score downstream use. Disposition metrics keep their names. A missing denominator stays null, and an export without `downstream_eligible` leaves the new rates null. The design is [design-review-fixes-0913.md](design-review-fixes-0913.md).

`policy_version` was `0.9.13`.

Package 0.9.14 leaves a downstream full-population rate null unless every row in that rate's population has a boolean `downstream_eligible`. `downstream_precision_known` and `downstream_withheld_supported_recall_known` report the known rows, with the denominator and coverage beside them. An unknown supported row is not counted as withheld, and a known eligibility on an unsupported row does not turn the recall into 0. The sampling saturation check uses the configured `min_documents` rather than a hard-coded 3. Below that minimum the model decision call is withheld and not spent. A human checkpoint still pauses, and the payload carries `minimum_documents` and `window_ready`. `pattern_min_cases` stays 2 and the axial ceiling stays 24000. The design is [design-review-fixes-0914.md](design-review-fixes-0914.md).

`policy_version` was `0.9.14`.

Package 0.9.15 requires a nonempty relationship statement and keeps `nothing_to_propose` as the explicit empty batch. One repair is checked against the original evidence, and a statement that is only a quote is not a relationship. `downstream_eligible` means supported evidence. A weak stem stays hypothesis exploration: the core graph does not treat it as support, and the export rewrites a stale eligible flag. Memos and the storyline snapshot still receive the hypothesis row. `unspecified` is a valid association role. A causal role below tier 3 is recorded apart from sentence wording, and the causal gate stays on. The axial export adds an attrition funnel and fails when the installed distribution does not match the imported checkout. Temperature is unset unless the caller configured it. Sampling reports the reread interpretation breakdown and copies completed pairs onto the result. `pattern_min_cases` stays 2 and the axial ceiling stays 24000. The design is [design-review-fixes-0915.md](design-review-fixes-0915.md).

`policy_version` was `0.9.15`.

Package 0.9.16 accepts an editable install whose metadata names the imported checkout, and still raises `InstallationMismatch` for a version mismatch before axial or selective work calls a model. The sampling recursion limit is counted from graph visits, so a 32-call reread is not stopped early. Validation flags are recomputed from the effective statement and role, and the previous flags stay on the revision history. Until a human sets a role, a reviewer's role is the working role. A mixed proposal batch keeps the valid relationships and logs each invalid row. A batch with nothing valid still uses one repair. `expenditure_tokens_per_relevant_reread` divides all reread tokens by the relevant count and is null when that count is zero. `idle_timeout` stays recorded and unenforced. A returned timeout does not prove the provider request stopped. `pattern_min_cases` stays 2 and the axial ceiling stays 24000. The design is [design-review-fixes-0916.md](design-review-fixes-0916.md).

`policy_version` was `0.9.16`.

Package 0.9.17 withholds every reread that does not fit the per-request budget in the same visit, so a human assignment of oversized passages is not stopped by the recursion limit. A mixed proposal batch is recovered from tool-call arguments and from the repair response, and a batch with nothing valid still uses one repair. Reread expenditure reports the known sum, how many sent calls had usage, and whether that sum is complete. Full expenditure rates stay null when usage is missing. Validation refreshes `validation_basis` on every check, and a human accept validates the statement before the role. `pattern_min_cases` stays 2 and the axial ceiling stays 24000. The design is [design-review-fixes-0917.md](design-review-fixes-0917.md).

`policy_version` was `0.9.17`.

Package 0.9.18 keeps a timed-out reread in the expenditure count. An initial reread timeout is logged as `sampling_reread`. A reread-repair timeout is logged as `schema_repair` with schema `SamplingRereadNote`. The attempt has no usage, so coverage includes it and the full expenditure rates and cost stay null. Selection and saturation timeouts stay kind `provider`. Axial and focused timeout rows are unchanged, and the provider budget call count is still the limiter's `calls_used`. `pattern_min_cases` stays 2 and the axial ceiling stays 24000. The design is [design-review-fixes-0918.md](design-review-fixes-0918.md).

`policy_version` was `0.9.18`.

Package 0.9.19 keeps participant attribution on quoted participant text. An analyst memo does not raise tier 3 or use `participants attribute X to Y`. A labeled analyst interpretation stays tier 2, uses `analyst interprets X as related to Y`, and is not downstream eligible. Axial and selective resume reject a source whose code definitions, segment assignments, or case identity differ from the checkpoint. A 0.9.18 checkpoint cannot resume, because that hash did not record those fields. `pattern_min_cases` stays 2, the axial ceiling stays 24000, and tier 4 stays human-only. The design is [design-review-fixes-0919.md](design-review-fixes-0919.md).

`policy_version` was `0.9.19`.

Package 0.9.20 checks the relationship a participant quote actually states. A keyword hit does not raise tier 3. A denial, a reversed direction, an undirected association, and a hedge stay below tier 3. The pattern, the participant explanation, and the analyst interpretation are separate claims. A supported pattern stays downstream eligible and can support a core. The analyst claim stays a hypothesis and does not block that pattern. Selective resume fingerprints relationships, memos, properties, and dimensions. A 0.9.19 selective checkpoint cannot resume. Axial planning reserves slots for later evidence classes and records examined pairs. `pattern_min_cases` stays 2, the axial ceiling stays 24000, and tier 4 stays human-only. The design is [design-review-fixes-0920.md](design-review-fixes-0920.md).

`policy_version` was `0.9.20`.

Package 0.9.21 adds a discovery evaluation beside gate scoring. A frozen human file can record a relationship the run never proposed. Recall, precision, and distortion stay null when their denominators are empty. A wrong direction is distorted. A withheld but supported proposal is a gate false negative, using the same known-eligibility split as 0.9.14. Per-token and per-dollar figures stay null unless reread expenditure is complete. The example file is synthetic. `pattern_min_cases` stays 2, the axial ceiling stays 24000, `batch_size` stays 1, and tier 4 stays human-only. The design is [design-discovery-evaluation.md](design-discovery-evaluation.md) and [design-review-fixes-0921.md](design-review-fixes-0921.md).

`policy_version` was `0.9.21`.

Package 0.9.22 keeps completed axial examinations separate from pairs that were only planned or sent. A failed, timed-out, or unsent pair stays retryable. `unique_pairs_explored` counts completed examinations only. The quote pattern that authorizes tier 3 requires one clear clause and one unambiguous connector. Passive voice, negation, a second clause, and a hypothetical stay unvalidated. That pattern is not general semantic reliability. Discovery recall uses a maximum one-to-one assignment that does not depend on adjudication ids. A `system_proposal` row binds only by `system_relationship_id`. Structural recall and human claim correctness are separate rates. The schema is `discovery-eval-2`, and a `discovery-eval-1` file still verifies under its old hash. Recovered-per-token and recovered-per-dollar figures are named as reread expenditure only. The planner still builds the full pair space in memory. `pattern_min_cases` stays 2, the axial ceiling stays 24000, `batch_size` stays 1, and tier 4 stays human-only. The design is [design-review-fixes-0922.md](design-review-fixes-0922.md).

`policy_version` was `0.9.22`.

Package 0.9.23 keeps a proposal review from erasing a recovered discovery. Independent matching and `system_proposal` binding are separate, and the same proposal can serve both. Every gold item may stay unmatched. The assignment maximizes full-agreement recoveries, then structural matches, then canonical content order, and one full recovery beats any number of distorted matches. Claim reviews are bound to a run, a relationship, and a hash of the reviewed statement. A changed statement or a missing run binding is stale, not correct. The schema is `discovery-eval-3`. `discovery-eval-1` and `discovery-eval-2` files still verify under their old hashes, and an unbound claim judgment loads as stale or unjudged. An epistemic or existential-negation marker before a connector does not authorize tier 3. That rule is closed-pattern regression coverage, not general semantic accuracy. Matching is restricted to candidate-connected components. The axial planner still builds the full pair space in memory. That limit stays open. `pattern_min_cases` stays 2, the axial ceiling stays 24000, `batch_size` stays 1, and tier 4 stays human-only. The design is [design-review-fixes-0923.md](design-review-fixes-0923.md).

`policy_version` was `0.9.23`.

Package 0.9.24 packs focused selection, axial relationship review, and selective core selection into requests that each fit the per-request budget. `batch_size` stays 1 and is not that pack size. One item that alone exceeds the budget is omitted and the review continues. A truncated selection is not complete. Models still cannot set verified or authorize attribution wording. A checkpoint from 0.9.23 cannot resume into a chunked review. The axial planner still builds the full pair space in memory. That limit stays open. `pattern_min_cases` stays 2, the axial ceiling stays 24000, `batch_size` stays 1, and tier 4 stays human-only. The design is [design-review-fixes-0924.md](design-review-fixes-0924.md).

`policy_version` was `0.9.24`.

Package 0.9.25 withholds an incomplete focused selection and an incomplete core choice. Neither is applied, neither is downstream eligible, and the uncovered ids are reported. Chunk order does not decide which codes can be selected. Core candidates are the promoted focused categories. A missing or incomplete focused selection is reported and is not replaced by every open code. A direct selective call with no focused selection still scores the codebook the caller passed. A claim component is eligible only when the row is supported evidence. One quote serving both endpoints does not count as a both-endpoint case under `pattern_min_cases` 2. Denied means the participant rejects the from-to link. `unique_pairs` is the count of completed examinations, and `pair_space_size` is the full pair space. Reread `cost_status` follows whether both token prices are configured. A 0.9.24 decision-review checkpoint cannot resume. The axial planner still builds the full pair space in memory. That limit stays open. `pattern_min_cases` stays 2, the axial ceiling stays 24000, `batch_size` stays 1, and tier 4 stays human-only. The design is [design-review-fixes-0925.md](design-review-fixes-0925.md).

`policy_version` was `0.9.25`.

Package 0.9.26 reverses the 0.9.25 rule that dropped a quotation used for both endpoints. Each endpoint is supported when the located text grounds it, including an overlapping or identical span. Cases are still counted once, by respondent or document. An identical span is an inspect flag, `shared_quote_inspect`, and is not a disqualification. A frequency qualifier is read before a categorical denial, so `not always` and the same family are qualified and never denied. `do not lead` and `has nothing to do with` stay denied. Those deterministic readings are provisional. A focused-selection code that exceeds the request budget is compacted and retried. If it still does not fit, a human checkpoint can review it at a changed request budget or record an exclusion. Without a human the stop is `oversized_unrecovered`, and the model does not exclude the item. Completed chunk reviews are kept. A 0.9.25 decision-review checkpoint cannot resume. The axial planner still builds the full pair space in memory. That limit stays open. `pattern_min_cases` stays 2, the axial ceiling stays 24000, `batch_size` stays 1, and tier 4 stays human-only. The design is [design-review-fixes-0926.md](design-review-fixes-0926.md).

`policy_version` was `0.9.26`.

Package 0.9.27 puts the raised ids, the selection completeness, and the provisional ids on the focused export, and axial or selective coding loaded from that directory uses them. A withheld or provisional focused selection keeps a relationship exploratory when either endpoint was never reviewed (`endpoint_unreviewed_by_focus`). Withheld rows record one primary cause, and those counts sum to the withheld rows. Inspect-flag counts stay separate. `grounded_endpoints` counts rows at tier 2 or higher, the same rows as the tier counts. `endpoint_anchoring_complete` is the anchoring-list count and is not that stage. `if` is hypothetical only when it governs the from-to link. The LLM reviewer's per-row decision (`reviewer_disposition`), rationale, and `reviewer_decided` flag are exported. A focused selection that needs more chunks than `max_decision_calls` stops before the first request with `chunk_allowance_insufficient`. `auto_size_decision_calls` stays off unless set, and it only raises the call allowance. A 0.9.26 decision-review checkpoint cannot resume. The axial planner still builds the full pair space in memory. That limit stays open. `pattern_min_cases` stays 2, the axial ceiling stays 24000, `batch_size` stays 1, and tier 4 stays human-only. The design is [design-review-fixes-0927.md](design-review-fixes-0927.md).

`policy_version` was `0.9.27`.

Package 0.9.28 keeps an evidence tier when a row is ineligible, and it does not count that row as an accepted downstream result. `evidence_tier`, `reviewer_disposition`, `endpoint_review_status`, and `downstream_eligible` are separate fields. Withholding reasons overlap. `primary_withholding_cause` is a reporting convention whose counts sum to the explained rows, including an unreviewed endpoint and a proposed row with a known case or tier shortfall. A policy-only rejection of tier-1 evidence is `reviewer_policy_rejection`. Before the first focused-selection request, `chunk_allowance_insufficient` checks the remaining call allowance and the remaining token allowance and exports required against available for each. The token estimate includes the fixed prompt, payload, and schema, and excludes output tokens and a retry or repair margin. `auto_size_decision_calls` does not raise the token allowance and does not mean the review can finish. A validated chunk is fingerprinted and reused when a later resume only raises the call or token allowance. A changed input or policy drops only the affected chunk. `selection_chunk_order` stays `id` unless the caller asks for `seeded_shuffle` with a recorded seed. Shuffling removes the id-order preference and does not complete a partial selection. A chunk can miss a category that matters only across the corpus. `selection_consolidation` over provisionally raised and near-miss codes is off unless set. A partial selection in id order is the prefix of that order, not a sample, and the export says so. Every ineligible relationship, including proposed and unreviewed rows, has a cause. Reviewer acceptance is a separate field from downstream eligibility. A discourse aside is a parenthetical or comma-bounded hedge, not a growing phrase list, and it does not erase a real conditional. Identical statements on one directed pair are consolidated and the tier is reassessed. A complete focused selection limits later eligibility to the raised codes. A reviewed endpoint that was not raised is `endpoint_not_selected_by_focus`. When auto-sizing is on, the configured decision calls are reserved for claim review. A positive focused-review token ceiling is part of the selection preflight. The token estimate counts input only and ran about 16% above provider-reported input on a completed focused review. That gap is not a dollar bound, and the estimate is not rescaled. Relationship review, saturation review, and core selection each keep their own decision-call ceiling. A core stop after fewer calls than the allowance was earlier steps on one shared pool, not a double charge. `prior_step_calls` does not reduce the later step. Each step records starting usage, calls made, remaining allowance, and stop reason. The numeric allowances are not raised. A finished run that stopped on an allowance can resume on a larger budget. A finished complete stage cannot. `repeat_frozen_review` and `repeat_pipeline` are different experiments. Temperature 0 is configuration control, not a reproducibility guarantee. A live run with no denied links and no tier 3 does not show that those detectors work. A 0.9.27 decision-review checkpoint cannot resume. `pattern_min_cases` stays 2, the axial ceiling stays 24000, `batch_size` stays 1, and tier 4 stays human-only. The design is [design-review-fixes-0928.md](design-review-fixes-0928.md).

`policy_version` is `0.9.28`.

## 26. Out of scope

Category feedback, focused recoding of earlier negatives, relationship hypotheses, and a Taguette project-file adapter. See the v0.6 plan steps 3–5. Within-corpus theoretical sampling and a researcher or model-labeled saturation decision are in 0.9.0. Collecting new data, an automatic saturation flag, the conditional matrix, and the full 1978 coding-family catalog remain out of scope.
