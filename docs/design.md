# automated-grounded-theory — Technical Design (0.6.0, corrections in 0.6.1, 0.6.2, 0.6.3, 0.6.4, 0.6.5, 0.6.6, 0.6.7, 0.6.8, 0.6.9, 0.6.10, 0.6.11, and 0.6.12)

Repo: https://github.com/davidscassell/automated-grounded-theory
Status: steps 1 and 2 of the v0.6 plan. Category development, theoretical sampling, and theory integration are not in this release. 0.6.1 corrects measurement, batching, the alias gate, idle waits, and the shared client loop. It does not add those later steps.

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

`segment_and_shuffle` → `prepare_round` → `Send("code_worker")` → `commit` → `sweep` → `consolidate` → `export`.

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

Resume distinguishes two kinds of manifest change. Research-context wording, model settings, and performance settings (concurrency, batch size, token budget, rate limits) may continue with `allow_changes` or `continuation`. The continuation event stores the old and new values. Corpus identity and the segmentation plan are not continuable: `corpus_hash`, `unit`, `seed`, `review_mode`, and `parallel_documents`. Those fields already determined the passages stored on the checkpoint. `allow_changes` still raises `CheckpointMigrationRequired`. Start a new run. There is no procedure that inserts a document into a queue that was already segmented.

The model block of the manifest is the instances that will run, not only the `temperature=` argument. `chat_model_settings` and `embedding_model_settings` read model id, provider, temperature, `top_p`, max tokens, embedding dimensions, endpoint, and deployment. Names that look like keys, tokens, or secrets are omitted, and `model_kwargs` is passed through `redact`. Metadata stores those values, so a coder constructed with `temperature=1` is not recorded as `0.0`.

Prompt examples are chosen from the full assignment list. `_apply_counts` no longer stops at eight. When a sweep reviews a code it stores `reviewed_assignment_ids` and appends that list to `reviews_seen`. The next prompt includes every assignment added since then, then fills up to the requested limit with one example per remaining document, then quotes that overlap the chosen set the least, then passage order. Document diversity runs before that passage-order fill.

Quote attribution overlaps each matched span with each passage. The bounding box from the first span to the last is not used, so a long passage that only occupies the gap is not selected. Several overlapped passages are `attribution="multi"` and coverage counts the code on each of them. No overlapped passage is `attribution="unresolved"`: `attributed_passage_id` stays empty, and coverage, agreement, and the Adjusted Rand index do not place the code on the passage that was being coded. The `evidence_outside_passage` flag still keeps the code.

Muse `unit_grid_from_annotations` places an excerpt by exact match, then by whitespace collapsed to a single space with offsets mapped back to the original text. Repeated excerpts use `occurrence` or `order` (0-based) or `offset` / `start` / `start_offset` / `char_start`. Without one of those, the excerpt is `ambiguous` and is not aligned. `alignment_coverage` is aligned annotations divided by annotations considered. A passage touched by an unaligned or ambiguous annotation is an unknown unit and is excluded from the grid. Coverage below `ALIGNMENT_COVERAGE_MIN` (0.8) returns `available` false and `reason` `alignment_coverage_below_threshold` instead of a score.

`configure_embedding_cache(vector_limit=..., vector_bytes=...)` is an optional LRU over canonical vectors. `vector_bytes` is 8 bytes times the number of dimensions. Vectors referenced by a retained snapshot are kept even when that leaves the cache over the cap. Both limits default to none. `commit_passage` still copies the codebook dict and shallow-detaches every entry: the sweep mutates definitions in place on the new state, and those objects must not be the ones stored in the previous checkpoint. The segment list is a new list; unchanged segment dicts stay shared. Counts on an append-only commit extend the entry's assignment list instead of scanning every segment. `sync_counts`, used when a merge or split moves assignments, still scans.

`policy_version` is `0.6.12`.

## 21. Out of scope

Category feedback, focused recoding of earlier negatives, relationship hypotheses, sampling briefs, and a Taguette project-file adapter. See the v0.6 plan steps 3–5.
