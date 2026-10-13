# Methodology

The default procedure is **Charmaz-informed automated analysis**: initial coding, constant comparison, and focused coding as a reporting view. It is not an axial-coding template and it does not claim theoretical saturation. Category development, theoretical sampling, and theory integration are later work. The engineering design is [docs/design.md](design.md).

## Sources

- Glaser, B. G., & Strauss, A. L. (1967). *The Discovery of Grounded Theory*. Aldine. https://doi.org/10.4324/9780203793206 — constant comparison; theory emerges from data; memos; theoretical saturation as a judgment about categories, not a quota.
- Glaser, B. G. (1978). *Theoretical Sensitivity*. Sociology Press — incident coding; ask what is happening and what category the incident indicates.
- Glaser, B. G. (1992). *Basics of Grounded Theory Analysis: Emergence vs. Forcing*. Sociology Press — do not force data into a preset frame.
- Strauss, A., & Corbin, J. (1990/1998). *Basics of Qualitative Research*. Sage — open coding, properties and dimensions, in-vivo terms. Axial and selective coding are out of scope here.
- Charmaz, K. (2006/2014). *Constructing Grounded Theory*. Sage — initial coding with gerunds, in-vivo codes, staying close to the data, memos, then focused coding. Saturation should be claimed cautiously.
- Chen, J., et al. Measuring open codes. https://arxiv.org/abs/2411.12142 — coverage, overlap, novelty, divergence across codebooks.
- Übellacker, T. (2024). AcademiaOS. https://arxiv.org/abs/2403.08844
- A Practical Guide and Assessment on Using ChatGPT to Conduct Grounded Theory. JMIR, 2025. https://www.jmir.org/2025/1/e70122 — LLM coding was weak on context, organization, and connections. Neighbors stay on each passage. Sweeps do the code-to-code comparison.

Hubert and Arabie (1985) is the adjustment used for the Adjusted Rand index on passage assignments. Taguette's codebook CSV importer is `taguette.import_codebook` (https://gitlab.com/remram44/taguette).

## What the default run does

`review_mode="document"` is the default.

| Step | Parameter | Practice |
| --- | --- | --- |
| Reject preset codes | `context` must not carry `codes`, `codebook`, `tags`, or a coding paradigm | Glaser 1992: emergence over forcing. The research question and sensitizing concepts only steer attention. |
| Segment | `unit` default `incident` (also paragraph, sentence, line, speaker turn) | Interviewer question plus the full answer, then incidents of about 150–800 characters, never cut mid-sentence, capped near 1,200. Short survey comments stay one incident. Neighbors and the question stay attached. |
| Order | `seed` shuffles documents; passages stay in reading order | Every document is read. The seed makes the document order repeatable. It is not theoretical sampling. |
| Code | `passages_per_request` default 1 (`batch_size` is the same setting); `max_concurrent_requests` default 4 (`concurrency` is the same setting); `passages_per_codebook_update` defaults to `passages_per_request` | Workers read one frozen snapshot (`Send`) and only the commit updates definitions. How many calls are in flight does not change how many passages share that snapshot. An incident may carry several overlapping spans. `nearest_code_id` may be empty. A quote that does not match is kept and flagged (`quote_status`), not dropped. A semicolon-joined label is split. An empty, too-long, or bundled new label is retried once and then kept with a review flag. |
| Strict sequential | `strict_sequential_settings()` or `procedure="strict_sequential"` | Document order, one passage per request, one request in flight, commit after every passage, no pipeline overlap, no parallel documents. The next passage sees codes committed from earlier passages. |
| Within-document batches | `passages_per_codebook_update` above 1, capped by `token_budget` | Each round takes the next incident from a different document, in the seeded document order, up to that commit window. A document contributes a second passage in that round only when it is the only document left. Incident order inside a document stays intact. Those passages share one frozen snapshot. `token_budget` (default 6000) bounds the request that will be sent: instructions, the codes retrieved for every passage in the call, neighbors, metadata, passages, the JSON schema, a short repair allowance, and reserved output. A batch that does not fit is repacked, and retrieved codes are trimmed. A passage is split only when its text still does not fit. One reconciliation follows the round, after the new codes in the round have been compared with each other. |
| Exclusions | empty text, `n/a`, `I don't know` | The row is kept with `coverage_status=excluded` and `exclusion_reason`. It is not sent to the model. |
| Provenance | one note per document reading | The note records that this document was read in seeded order and that the context was not used as codes. It is labeled provenance. It is not a reflexivity memo. A reflexivity memo is written only when the sweep model supplies one. |
| Synthesis | `sweep_every`, and a final consolidation pass | Code-to-code comparison. A chunk carries full evidence for its codes and for a bounded neighborhood of similar codes from other chunks. It does not repeat the whole book's label index. One compact label-only discovery call names candidates across the book. Later sweeps review codes that are new or whose label, definition, or examples changed. Every `full_audit_every` sweeps (default 4) and the consolidation pass read the whole book. Operations are `alias_duplicate`, `refine_definition`, `split_code`, `link_to_category`, `record_dimension`, `keep_distinct`, and `defer`. `relate_categories` is stored only. Stance is context, not a merge veto. A chain of pairs is not merged unless every pair in the component was judged. The survivor is the lexicographically smaller code key. The code id stays stable across renames; a raw 16-character hex string is never a label. |
| Overlap | `pipeline=True` | Synthesis of the finished step runs while the next document is coded against the pre-merge snapshot. The merge map is applied afterwards, in chunk order, not in completion order. The next document therefore does not yet see the merged labels. That is one step of lag, not a race. |
| Saturation | `result.saturation` | New codes before and after alias resolution (`new_codes_pre_alias`, `new_codes_post_alias`). `judgment` is always null. An absence of new labels is not meaning saturation. |
| Focused coding | `build_themes` | A descriptive reporting view. A code may sit in more than one theme. `max_theme_share` and `max_themes` are presentation budgets, not analytic quotas. Prevalence states its denominators; percentages may overlap. Leftover codes go to `Ungrouped incidents`. |
| Wave mode | `review_mode="wave"` | **Weakens constant comparison.** Passages from several documents can be coded together against one frozen snapshot, and a later passage in a document can be coded before an earlier one. New codes inside the wave are invisible to the other batches. Use it only when that tradeoff is acceptable. `parallel_documents=True` weakens comparison further: each document has its own codebook until a sweep. |

`concurrency` is a ceiling shared by every model call, including batched coding, sweeps, and evaluation. A rate-limit response (HTTP 429 or a message that says so) halves it; successes add one back up to the ceiling. Waits use full jitter. `Retry-After` is honored up to `max_backoff`, and the limiter releases its slot while it sleeps. Jitter does not change the codebook. An identical prompt is replayed from a response cache so a rerun with the same seed does not depend on provider sampling. Provider temperature 0 is still not a proof of bit-identical output; the cache is the rerun guarantee for prompts this process has already seen. Reported coding time is the wall clock of each batch once, plus per-request latency and queue wait. It is not the batch duration multiplied by the passages in that batch. `tokens_per_minute` budgets tokens. A reservation larger than that bucket raises `OversizedReservation` instead of waiting; `allow_oversized=True` waits for a full bucket and then drains it. `requests_per_minute` budgets requests.

Code ids stored on new codes are a hash of the normalized label, so the same spelling is the same id. The merge root is that label's sort order, not which chunk finished first.

## Speed and quality on a fake model

`scripts/benchmark_review_modes.py` sleeps 0.05 seconds per call. Corpus: 6 documents, 4 passages each (24 passages), seed 7, `sweep_every=8`, `unit="paragraph"`. The paragraph unit keeps each blank-line block as its own passage; the study default `unit="incident"` would pack these short documents. No API key. An embedding-latency row sleeps on each embedding call and ranks those passages as queries. `--embedding-delay 0` skips it. Measured on 0.6.0 in this environment. 0.6.1 was not remeasured. On 0.6.0, `concurrency` widened the frozen window. From 0.6.4 it does not: only `passages_per_codebook_update` does. The sequential row is the procedure `strict_sequential_settings()` now names.

| Mode | Seconds | Calls | Passages/s | Codes | Merges | Near-duplicate pairs left |
| --- | --- | --- | --- | --- | --- | --- |
| sequential (`document`, `pipeline=False`, `batch_size=1`, `concurrency=1`) | 1.690 | 28 | 14.20 | 3 | 3 | 0 |
| document (background synthesis, `batch_size=4`, `concurrency=4`) | 0.543 | 10 | 44.18 | 3 | 3 | 0 |
| wave (background synthesis, `batch_size=4`, `concurrency=4`) | 0.307 | 9 | 78.20 | 3 | 1 | 0 |

The fake coder reuses a label when that exact label is already in the snapshot, so identical sentences collapse at commit. All three finished with 3 codes and no leftover pair whose labels had Jaccard at least 0.5. Document mode matched the sequential code count and still recorded a merge on each synthesis. Wave mode was fastest and merged in a single pass, because mixed documents shared a snapshot. That is the constant-comparison cost, even when this particular fake did not leave extra codes behind. These times are not a provider benchmark.

## Taguette

Import `taguette_codebook.csv` or `taguette_themes.csv` from Project info → Import codebook. The theme file uses `Theme\Code` plus a parent row for the theme, which is the path form in Taguette's own tests (`interesting\places`). Taguette does not import highlights or `.qdc`. The `.qdc` files follow Taguette's export shape for other QDA packages. A future adapter for a Taguette SQLite project must pin the Taguette version it was tested against. This package does not implement that adapter.

`get_graph("open_code")` and `get_graph("themes")` draw the `Send` fan-out. `draw_mermaid_png` uses mermaid.ink. The `viz` extra installs grandalf; pygraphviz is optional.

## Evaluation

`evaluate_run` without `model` is offline: coverage, verbatim quotes, quote-anchor flag rates (`possibly_modified` and `unanchored` are kept and counted, not dropped), gerund and in-vivo rates, singletons, the saturation curve, and lexical Chen metrics when a second codebook is passed. `adjusted_rand_index` compares passage partitions (Hubert and Arabie 1985). It does not compare labels: two runs that give every passage a different singleton code score 1.0, because no two passages co-occur. `aligned_assignment_agreement` aligns codes by exact text or a spelling variant (a negation does not align; `mapping` records a reviewed equivalence), then scores the set of codes on each passage (mean Jaccard) and Krippendorff's alpha on the aligned passage-by-code matrix. That score is low for those different singletons. With `model`, near-duplicate pairs are judged concurrently and Chen metrics use those judgments. None of these scores assert saturation.
