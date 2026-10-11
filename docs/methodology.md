# Methodology

This note maps the pipeline onto grounded theory practice. It is an engineering account of what the code does, not a claim that a model run is a finished analysis. Saturation is reported and is not declared.

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
| Segment | `unit`: paragraph, sentence, line, or speaker turn | Charmaz: line-by-line or incident coding. Neighbors stay attached (JMIR 2025: context was a weak spot). |
| Order | `seed` shuffles documents; passages stay in reading order | Every document is read. The seed makes the document order repeatable. It is not theoretical sampling. |
| Code | one document at a time; `batch_size` 1 commits after each passage | Constant comparison: a new incident is compared with codes already earned. The model must name the nearest code and why it does not fit before minting a new one (Glaser 1978; Charmaz). |
| Within-document batches | `batch_size` above 1, capped by `token_budget` | Several passages of the *same* document share the codebook from the start of that step. That is faster and a milder break in comparison than mixing documents. Duplicates are left for synthesis. |
| Memos | incident memo only when the passage adds a property, dimension, condition, consequence, or comparison | Corbin and Strauss; Charmaz. A document memo copies those incident memos. Empty passages are not sent and do not get a passage memo. |
| Reflexivity | one note per document reading | Charmaz: the analyst is in the work. The note records that this document was read in seeded order and that the context was not used as codes. |
| Synthesis | `sweep_every`, and a final consolidation pass | Code-to-code comparison (Glaser and Strauss). The codebook is sorted alphabetically and chunked (default 20 codes). Those chunks run concurrently. One structured call per chunk proposes merges, checks that they share an action, an object, and a stance, and rewrites the survivor label and definition from the member quotes. A quote that does not fit is split back out. Near-duplicate labels with the same stance merge without a model call. Token overlap, and embeddings when supplied, are optional hints, not a gate. A positive code is not merged with a negative code. Each pair is decided once unless a definition changes in substance. At most `max_merges_per_sweep` merges are applied; the rest wait for review. Decisions below `min_merge_confidence` or `review_below` go to a review list and are not applied. Every hint and every model-proposed pair is logged with a rationale. A chunk that times out is skipped with a warning rather than failing the run. Union-find keeps the lexicographically smaller code id. Overloaded codes are flagged, not split, because splitting would reassign old segments without rereading them. |
| Overlap | `pipeline=True` | Synthesis of the finished step runs while the next document is coded against the pre-merge snapshot. The merge map is applied afterwards, in chunk order, not in completion order. The next document therefore does not yet see the merged labels. That is one step of lag, not a race. |
| Saturation | `result.saturation` | New codes per window, both before merges and as the net change in codebook size after merges. `judgment` is always null. Charmaz: do not let the software declare saturation. |
| Focused coding | `build_themes` | A separate call. It does not read the corpus again and it does not mint open codes (Charmaz's focused coding, operationalized as grouping codes). With `n_themes` unset, the prompt asks for about 6 to 12 themes. The model then reviews the themes in alphabetical order and proposes duplicate groups with a reason. Token overlap and optional embeddings are hints only. A fold that would hold more than about 30% of the codes is not kept. Every hinted or proposed pair is logged. Prevalence and quotes are counted and checked in ordinary code. Leftover codes get one reassignment call, then `Ungrouped incidents`. |
| Wave mode | `review_mode="wave"` | **Weakens constant comparison.** Passages from several documents can be coded together against one frozen snapshot, and a later passage in a document can be coded before an earlier one. New codes inside the wave are invisible to the other batches. Use it only when that tradeoff is acceptable. `parallel_documents=True` weakens comparison further: each document has its own codebook until a sweep. |

`concurrency` is a ceiling. A rate-limit response (HTTP 429 or a message that says so) halves it; successes add one back up to the ceiling. Waits use full jitter and `Retry-After` when the exception carries one. Jitter does not change the codebook. An identical prompt is replayed from a response cache so a rerun with the same seed does not depend on provider sampling. Provider temperature 0 is still not a proof of bit-identical output; the cache is the rerun guarantee for prompts this process has already seen.

Code ids stored on new codes are a hash of the normalized label, so the same spelling is the same id. The merge root is that label's sort order, not which chunk finished first.

## Speed and quality on a fake model

`scripts/benchmark_review_modes.py` sleeps 0.05 seconds per call. Corpus: 6 documents, 4 passages each (24 passages), seed 7, `sweep_every=8`. No API key. Measured in this environment:

| Mode | Seconds | Calls | Passages/s | Codes | Merges | Near-duplicate pairs left |
| --- | --- | --- | --- | --- | --- | --- |
| sequential (`document`, `pipeline=False`, `batch_size=1`, `concurrency=1`) | 1.588 | 28 | 15.11 | 3 | 3 | 0 |
| document (background synthesis, `batch_size=4`, `concurrency=4`) | 0.525 | 10 | 45.73 | 3 | 3 | 0 |
| wave (background synthesis, `batch_size=4`, `concurrency=4`) | 0.315 | 9 | 76.19 | 3 | 1 | 0 |

The fake coder reuses a label when that exact label is already in the snapshot, so identical sentences collapse at commit. All three finished with 3 codes and no leftover pair whose labels had Jaccard at least 0.5. Document mode matched the sequential code count and still recorded a merge on each synthesis. Wave mode was fastest and merged in a single pass, because mixed documents shared a snapshot. That is the constant-comparison cost, even when this particular fake did not leave extra codes behind. These times are not a provider benchmark.

## Taguette

Import `taguette_codebook.csv` or `taguette_themes.csv` from Project info → Import codebook. The theme file uses `Theme\Code` plus a parent row for the theme, which is the path form in Taguette's own tests (`interesting\places`). Taguette does not import highlights or `.qdc`. The `.qdc` files follow Taguette's export shape for other QDA packages.

## Evaluation

`evaluate_run` without `model` is offline: coverage, verbatim quotes, quote-anchor flag rates (`possibly_modified` and `unanchored` are kept and counted, not dropped), gerund and in-vivo rates, singletons, the saturation curve, and lexical Chen metrics when a second codebook is passed. `adjusted_rand_index` compares passage→code assignments across those codebooks. With `model`, near-duplicate pairs are judged concurrently and Chen metrics use those judgments. None of these scores assert saturation.
