# Changelog

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
