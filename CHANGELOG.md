# Changelog

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
