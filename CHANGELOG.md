# Changelog

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
