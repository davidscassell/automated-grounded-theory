# automated-grounded-theory — Technical Design (0.6.0, corrections in 0.6.1, 0.6.2, 0.6.3, 0.6.4, and 0.6.5)

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
    token_budget=4000,
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

One `CallLimiter` is shared by coding and sweep calls in a run, including retries. Cache keys add snapshot, definition, evidence, and policy versions when those are set. Transport retries and the one schema repair are counted separately (`transport_retries`, `schema_repairs`).

Metrics on `metadata["metrics"]`: duplicates before reconciliation, reconciliation calls and seconds, coding p50/p95, tokens, retries, lost-distinction candidates, `merges_later_split`, and `activity` (in-flight events and idle gaps of at least 0.05 seconds). `duplicate_rounds.json` and `lost_distinction_candidates.json` are written next to the codebook.

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

## 14. Out of scope

Category feedback, focused recoding of earlier negatives, relationship hypotheses, sampling briefs, and a Taguette project-file adapter. See the v0.6 plan steps 3–5.
