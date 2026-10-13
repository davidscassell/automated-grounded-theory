# automated_grounded_theory

**Open coding**, then **focused coding**, in the grounded theory tradition (Glaser and Strauss 1967; Glaser 1978, 1992; Strauss and Corbin 1990/1998; Charmaz 2006/2014). By default the model reads every document in a seeded order, and every passage of that document in reading order, comparing each incident with the codebook earned so far. It does not decide that the data are saturated. How each step maps onto those sources is in [docs/methodology.md](docs/methodology.md). The 0.6 design, including the 0.6.1 corrections, is in [docs/design.md](docs/design.md). A short notebook with the same citations and the Taguette import steps is [docs/open_coding.ipynb](docs/open_coding.ipynb). A full walkthrough on a fictional employee survey, including an offline mode that needs no API key, is [examples/demo.ipynb](examples/demo.ipynb). The notebook writes its outputs to a temporary directory.

`build_themes` is the next round: it groups those open codes into themes. It does not do axial coding, selective coding, or theoretical sampling.

## Install

```bash
pip install automated_grounded_theory
```

Coding uses a LangChain chat model. Install the extra for the provider you actually call:

| Extra | Provider |
| --- | --- |
| `xai` | Grok / xAI (`langchain-xai`) |
| `openai` | OpenAI |
| `anthropic` | Anthropic |
| `google` | Google Gemini |
| `mistral` | Mistral |
| `groq` | Groq (the inference host, not Grok) |
| `ollama` | Ollama |
| `azure` | Azure OpenAI |
| `aws` | Bedrock |
| `cohere` | Cohere |
| `huggingface` | Hugging Face |
| `checkpoint` | SQLite resume (`langgraph-checkpoint-sqlite`) |
| `plots` | Theme prevalence chart (`matplotlib`) |
| `viz` | Graph drawing (`grandalf`). Optional `pygraphviz` needs a Graphviz install and is not in the extra. |
| `taguette` | The Taguette package (`taguette>=1.5`) for a caller who imports it. This package still does not write a Taguette SQLite project. |

```bash
pip install "automated_grounded_theory[xai]"
```

No provider is the default, and none is imported until you name it.

## API key

Pass `api_key` or leave it unset. If you leave it unset, the provider SDK reads its own environment variable. This package does not contain a key.

For Grok, create a key in your [xAI account](https://console.x.ai/) and export it:

```bash
export XAI_API_KEY="..."
```

`langchain-xai` reads `XAI_API_KEY` when you do not pass `api_key`. Other providers use their own variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and so on).

## Usage

```python
from automated_grounded_theory import open_code

result = open_code(
    corpus=["interviews/alex.txt", "interviews/blair.txt"],
    context={
        "research_question": "How do students describe the days around leaving home?",
        "population": "Undergraduates in their first term",
        "sensitizing_concepts": ["leaving home"],
    },
    model="xai:grok-4",
    sweep_model="xai:grok-4",
    seed=7,
    sweep_every=25,
    unit="incident",
    batch_size=1,
    concurrency=4,
    output_dir="coding_run",
    checkpoint_path="coding_run/checkpoints.sqlite",
)
```

`model` is a `provider:model-name` string, passed to LangChain `init_chat_model`, or a `BaseChatModel` you built yourself. `sweep_model` defaults to the same model. A blank string is the same as omitting it. `embeddings` is optional. When it is omitted, merge candidates are lexical and the sweep is still a chat-model call. You only need a chat model to run. Cached vectors are keyed by the model, a fingerprint of dimensions and endpoint, and the sha256 of the exact text, so the same model name at 3 and 5 dimensions does not share a vector. A returned vector whose width does not match that configuration is not stored.

Models this package builds from a `provider:model` string get `timeout=120` seconds (`request_timeout`). A `BaseChatModel` you pass in is not modified; the same deadline still wraps each coding call. Sweep calls use `sweep_timeout` (300 seconds) when that is longer than the coding deadline. `request_timeout=0` waits with no deadline, and the sweep then waits with no deadline too. Timeouts and connection errors are retried with full-jitter backoff, the same as rate limits. Only a rate limit lowers the concurrency ceiling.

```python
from langchain.chat_models import init_chat_model

coder = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)
result = open_code(corpus, context, model=coder, seed=7)
```

`context` may be a question string or a mapping with `research_question`, `population`, `setting`, `sensitizing_concepts`, and `notes`. Sensitizing concepts steer attention. They are not codes. A mapping that includes `codes`, `codebook`, `tags`, or a coding paradigm raises `PresetCodesError`.

A corpus item can be a mapping with `text` or `path`, optional `id` / `name`, and optional `metadata` (department, tenure, and anything else about the respondent). Metadata is copied onto each segment and into `highlights.csv`. It is not a code, and the model is told not to put it in labels or definitions.

`unit` defaults to `incident`: an interviewer question plus the full answer, split into incidents of about 150–800 characters on sentence boundaries, never mid-answer, and capped near 1,200 characters. `paragraph`, `sentence`, `line`, and `speaker_turn` remain available. Each incident keeps document id, character offsets, the question, and neighboring text. Empty rows and non-answers (`n/a`, `I don't know`) stay in the passage list with `exclusion_reason` and are not sent to the model. A quote that is not verbatim is kept and flagged (`quote_status`: exact, normalized_match, fuzzy_match, possibly_modified, unanchored).

`review_mode` defaults to `document`. Documents are shuffled with the seed. Passages inside a document stay in reading order and are coded with that document's neighbors. `review_mode="wave"` may code passages from several documents in one step against one frozen snapshot. That weakens constant comparison (Glaser and Strauss 1967): a later passage can be coded before an earlier one, and new codes are invisible to the rest of the wave. Use it only when you have accepted that tradeoff. `parallel_documents=True` is a second opt-in that gives each document its own codebook until a sweep. It is off by default.

Three settings control the schedule, and they do different jobs. `passages_per_request` (alias `batch_size`, default 1) is how many passages share one model call. `max_concurrent_requests` (alias `concurrency`, default 4) caps in-flight calls. It does not change which passages see which codebook. `token_budget` (default 6000) caps each coding call. The count is the request that will be sent: instructions, the codes retrieved for every passage in the call, neighbors, metadata, the passages, the JSON schema, a short repair allowance, and room for the answer. A batch that does not fit is split into fewer passages and retrieved codes are trimmed. A passage is split on whitespace only when the text itself still does not fit, or sent alone with a warning when no slice fits. A schema repair is measured again against the same budget. Validation feedback is bounded and shortened until the retry fits. A coding call or a repair that is still over the budget is sent, with a warning and an `oversized_request` flag (passage id and estimated size) on the run metadata and the review list. `passages_per_codebook_update` (default: the same as `passages_per_request`) is how many passages are coded before the codebook is committed. Above 1, each round takes the next incident from a different document, in the seeded order, and does not hold two adjacent passages from one document unless that document is the only one left. The round shares one frozen snapshot. Before the commit, one comparison call aliases new codes from that round when they are the same concept, and leaves the others distinct. `4` is the next request size to compare; do not start at 8.

`strict_sequential_settings()` or `procedure="strict_sequential"` is the sequential baseline: `review_mode="document"`, one passage per request, one request in flight, commit after every passage, `pipeline=False`, and `parallel_documents=False`. Later passages in a document see codes committed from earlier passages. Raising concurrency does not turn that into a wider frozen window.

A 429 halves the in-flight ceiling. Coding, sweeps, and evaluation share that one gate, including batched calls, so concurrent batches cannot each open their own ceiling. `metrics.max_in_flight` is the peak that held a slot. Transport retries use full jitter. `Retry-After` cannot exceed `max_backoff`, and the slot is released during the wait. Schema repair (`REPAIR THIS OUTPUT`) is a separate single attempt and is not a transport retry. Its token usage includes the failed response and the repaired one. Cache keys include the snapshot id, definition revision, evidence revision, and policy version. Run metadata includes an in-flight timeline (`metrics.activity`). Coding time is wall time (`stage_seconds.coding`), per-request latency (`metrics.coding_seconds`), and queue wait (`metrics.coding_queue_wait_seconds`). A batch duration is not added once per passage. `tokens_per_minute` reserves estimated input plus output and then reconciles to actual usage. A reservation larger than the bucket raises `OversizedReservation` instead of waiting. `allow_oversized=True` waits until the bucket is full and then drains it. `requests_per_minute` is the separate per-request bucket. Coding calls use `request_timeout` (120 seconds). Sweep calls use `sweep_timeout` (300 seconds) when that is longer.

`pipeline=True` (the default) runs synthesis of the passages just coded while the next document or wave is coded against the pre-merge snapshot. The merge map is applied when the sweep returns, in chunk order, and the surviving label is the lexicographically smaller code id. Async entry points are `aopen_code`, `abuild_themes`, `aevaluate`, and `amulti_seed_stability`. The sync wrappers reuse one background event loop and do not close HTTP clients the model still holds, so `build_themes` can follow `open_code` on the same model. Inside a notebook they still run on that loop rather than the notebook's.

`parallel_documents=True` codes each document on its own codebook and merges them at sweeps. It is off by default, because passages then do not see each other's new codes until the next sweep. `concurrency` caps how many documents are coded in one wave.

`interrupt_before_sweep=True` stops before a sweep so you can read the codebook. Resume with the same `checkpoint_path` or `checkpointer`, the same `thread_id`, and `resume=True`.

```python
from automated_grounded_theory import multi_seed_stability

report = multi_seed_stability(corpus, context, seeds=[1, 2, 3], model="xai:grok-4")
```

The report is label overlap across seeds, plus the Adjusted Rand index of passage partitions and `aligned_assignment_agreement` (mean Jaccard of each passage's code set after labels are aligned by exact text or a spelling variant, and Krippendorff's alpha on that matrix). A negation does not align. `mapping` records a reviewed equivalence. Adjusted Rand scores 1.0 when every passage has its own code, even if the codes differ. The aligned score does not. The report does not pick a winner.

A later decision model can be passed as `code_retriever` if it implements `rank(passage, codebook, k) -> labels`. Nothing like that ships in this version. Coding and sweeps are chat-model calls.

## What the model is asked to do

For each passage the model says which code each incident indicates, and whether that code is already in the book. One passage can receive several codes, one per distinct incident. Codes should be short gerunds, close to the quote (Charmaz), and they have to come from the participant's language. The research context only sensitizes attention. It is not a source of labels. Before settling on a gerund, the model is asked to look for a vivid participant phrase. `in_vivo` is true only when the label is a contiguous copy, the words are a metaphor, a repetition, or a coined phrase, and a gerund would lose that point. The prompt includes "Pay. Pay. Pay." and "feels like a number", and "I like my team" coded as "expressing liking for the team" with `in_vivo` false. The nearest existing codes, retrieved by embeddings when you pass them and otherwise by label overlap, are listed on the passage. Before a new code the model has to name one of those and why it does not fit. Reuse or refinement is preferred. Definitions stay at the level of the incident: respondent attributes and the question wording do not belong in them. A memo, when written, should be about 50 to 120 words and include a comparison, a property or dimension, and an emerging question. A restatement is not a memo, and a note that only says no property was recorded is dropped. `evaluate_run` reports `code_reuse_pct`, the share of segments that reuse a code already applied earlier.

A plain check then runs, with no model:

- a quote is anchored, never dropped: exact, normalized (whitespace, case, punctuation), fuzzy (small edits, with a similarity), possibly modified, or unanchored. `...` can match several spans. Search is the passage, then neighbors, then the document. Possibly modified and unanchored quotes are flagged and coding continues
- a semicolon-joined label is split into separate codes, and a piece that matches an existing code is reused
- an empty label, a label over the length limit, or a new label that bundles several concepts (a comma list of three or more, or several concepts joined by "and") is retried once with feedback. If it is still invalid, the code is kept and flagged for review. A blank label is stored as `(empty label)`
- generic labels (`experience`, `theme`, `discussion`, ...) are dropped
- an in-vivo flag is cleared when the label is not in the passage
- a non-gerund label is kept and flagged, unless `require_gerund=True`
- a new code with no nearest code, or a definition that repeats respondent metadata, is kept and flagged for review

Code-management calls send Pydantic JSON (stable ids for passages, codes, hints, and themes) and return Pydantic decisions. Every decision carries `confidence` from 0 to 1 and a `rationale` of one to three sentences. A coding action, in-vivo flag, definition rewrite, theme assignment, or tension below `review_below` (default 0.5) is written to `review.json` and is not applied. A merge below `min_merge_confidence` (default 0.5) is not applied either. One sweep call proposes the merge, checks the rubric (same action, same object, and a contrast check), and rewrites the survivor label and definition from the member quotes. Stance is recorded and is not a merge veto. A quote that does not fit is split back out. Labels that differ only by spelling or inflection (case, punctuation, whitespace, plural or gerund forms) and whose definitions are compatible merge without a model call. A negation or a polarity difference never does. Other lexical near-duplicates, including Jaccard at least 0.8, are only proposed and go through the definition and evidence comparison. Valence is stored and is not a merge veto. Both valences stay on the decision. A chain of similar pairs is merged only when every pair in that component was judged. Each pair is decided once per run, keyed by the sorted code ids. A later sweep revisits it when a definition's text or revision changes, or when that code's examples or assignments change. Token overlap is not a threshold. At most `max_merges_per_sweep` merges (default 12) are applied in one sweep; the rest wait on the review list. After the last document, a consolidation sweep reads the codebook again in sorted chunks. The saturation curve reports new codes before merges and the net change after merges. A new label whose content words mostly restate the research context is flagged for review. Pass `check_model` when you want a cheaper model to double-check quote fit and context leakage. `evaluate_run` reports the confidence distribution. `run_metadata.json` records seconds spent coding, sweeping, rewriting, and, from the later stages, building themes and evaluating. Taguette CSVs stay on Taguette's columns. Embeddings are still optional.

Every `sweep_every` passages (and once at the end) a sweep model compares codes with codes. The codebook is sorted alphabetically by label, then by definition, and sent in chunks of `sweep_chunk` (default 20) so later codes are not dropped and near-duplicate wording sits together. A chunk carries full definitions and examples for its own codes and for a bounded neighborhood of similar codes from other chunks (`cross_chunk_limit`, default 8). A hinted pair is included only when both codes' definitions and examples are in that request. Pairs past the neighborhood go into further bounded comparison requests. It does not repeat an index of the whole book. One compact label-only discovery call names candidates across the book. At chunk size 20 the user message is 53,261 characters for 160 codes, 213,900 for 640, and 429,196 for 1280. The repeated index was 103,334 / 1,120,941 / 4,145,141 at those sizes. `scripts/benchmark_sweep_payload.py` prints the table. Later sweeps review only codes that are new or whose label, definition, or examples changed. Every `full_audit_every` sweeps (default 4), and the final consolidation pass, read the whole book so a missed relationship can still be found. Chunks run concurrently and are reconciled by chunk index. The model reads every code in the chunk and proposes duplicate merges itself. Each merge needs a rationale. Token overlap is an optional hint. Embeddings, when you pass them, add further hints (cosine of label plus definition, default threshold 0.65) and never replace the lexical hints. If that embedding call fails, the hints are token overlap only. Merging works with no embeddings model. Each hint has a `pair_id`. The decision log has one row per hint: accepted, `model_rejected`, `model_omitted` when the model skipped it, or `chunk_skipped` when that chunk timed out. A merge the model proposes that was not a hint is logged as `model_proposed`. Rejections of pairs that were not hinted are ignored and counted in a warning. A sweep with no hints still asks the model to review the codebook, and says so in a warning. A chunk that times out or loses its connection is skipped with a warning. The rest of the run continues. The sweep model can sharpen definitions. It can flag an overloaded code. It does not split one, because that would reassign old segments without rereading them. Accepted merges become a union-find map whose root is the lexicographically smaller code id, then that map relabels segments. A response cache replays an identical prompt on a rerun. After each document the run stores a provenance note that the document was read in seeded order. That note is not a reflexivity memo. A reflexivity memo is written only when the sweep model supplies one. A document memo is written only when that reading produced a substantive passage memo. The placeholder sentence about no property or dimension is not stored.

`result.saturation` is new codes per batch. `result.saturation_report["judgment"]` is always `None`.

## Export

`output_dir` receives:

- `codebook.json` / `codebook.csv` — code, definition, examples, frequency
- `coded_segments.json` / `coded_segments.csv` — quote, document, offsets
- `memos.md` / `memos.json` / `memos.csv`
- `saturation.csv`
- `taguette_codebook.csv` — header `tag,description,number of highlights,number of documents`
- `codebook.qdc` — REFI-QDA codebook
- `highlights.csv` — columns `id,document,tag,content`, plus `metadata` (Taguette ignores extra columns)
- `run_metadata.json` and `prompt_log.json` — model names, seed, parameters, prompts, and token usage when the provider reports it

Taguette's importer (`taguette.import_codebook.list_tags`) looks for a column named `tag`, `name`, or `path`, plus an optional `description`. Other columns are ignored, so `taguette_codebook.csv` imports from Project info → Import codebook. The header matches Taguette's own codebook CSV export. A backslash in a path is the parent/child split (`interesting\places` in Taguette's tests and `.qdc` export). Backslashes inside a label are replaced so they do not add a level. Dots are kept.

`get_graph("open_code")` and `get_graph("themes")` return the LangGraph builders. Coding and theme chunks fan out with `Send`. `draw_mermaid_png` sends the diagram to mermaid.ink. `pip install "automated_grounded_theory[viz]"` adds grandalf for `draw_ascii`. `pygraphviz` is optional and is not installed by that extra.

Taguette writes REFI-QDA `.qdc` and does not import it. A future adapter that opens a Taguette SQLite project must pin the Taguette version it was tested against. This package does not ship that adapter. Codebook CSV export stays the supported Taguette path. `codebook.qdc` follows that export (`urn:QDA-XML:codebook:1.0`, `Code` with `guid`, `name`, `isCodable="true"`, and an empty `Sets`) and adds a `Description` child so other packages can read the definition. `highlights.csv` matches Taguette's highlight export shape (`id`, `document`, `tag`, `content`). Taguette does not import highlights.

Prompt caching: the instructions and research context sit in a stable system prefix. The codebook and the passages stay in the user message, which changes every call. Anthropic also gets an explicit `cache_control` breakpoint. Providers that cache a repeated prefix on their own (including xAI and OpenAI, when that model supports it) see the same stable prefix. Set `prompt_cache=True` or `False` to force the explicit breakpoint on or off.

## Focused coding

`build_themes` takes an `open_code` result and groups every open code into one theme. This is focused coding (Charmaz): the codes already exist, and the model categorizes them. It does not read the passages again and it does not mint codes.

```python
from automated_grounded_theory import build_themes

themes = build_themes(
    result,
    context,
    model="xai:grok-4",
    n_themes=None,
    seed=7,
    output_dir="coding_run/themes",
)
```

`model` is the same kind of argument as in `open_code`: a `provider:model-name` string or a `BaseChatModel`. `n_themes=None` asks for about 6 to `max_themes` themes (default 12) when the codebook is large enough, and fewer when it is small. One theme per code is not the goal. Set `n_themes` to ask for that exact number; it is not silently capped at 12. `seed` only shuffles the order of codes in the prompt. Representative quotes do not use it.

`build_themes` also accepts the `output_dir` from `open_code`. `load_run(directory)` (and `OpenCodeResult.load`) reads `run.json` when it is there, otherwise the split JSON files.

A small LangGraph calls the model, then checks the assignment. Every open code must be assigned, and every name must be a code from the codebook. Unknown names are dropped. A code named twice keeps the first theme. If anything is missing, the model is shown that list and called again (`validation_retries`, default 1). Codes still missing get one targeted reassignment call. Whatever that call does not place goes into `Ungrouped incidents`. A few leftovers do not raise `ThemeError`.

After that assignment the model reviews the themes in alphabetical order and proposes duplicate groups, with a reason on each fold. Token overlap is an optional hint. A code may belong to more than one theme. `max_theme_share` (default 0.30) and `max_themes` are presentation budgets: they do not block a fold or drop a membership. Prevalence fields name their denominators (`denominator_passages`, `denominator_documents`, `denominator_respondents`) because overlapping memberships mean the percentages can sum past 100. Every hinted pair and every model-proposed pair is logged on `metadata["duplicate_decisions"]`. A theme over the presentation budget is noted, not split.

Counts are computed from the coded segments, not by the model:

- a passage counts toward a theme when any segment in it carries a member code
- a document counts when any of its passages does (`n_comments` is that same document count, for a survey where each comment is a document)
- a respondent comes from metadata (`respondent_id`, `respondent`, `participant_id`, or `participant`), or from `respondent_key` if you pass one. With none of those, each document is its own respondent
- every metadata field on those passages is broken down (for example `question`), as document counts

Each theme keeps up to three representative quotes (`quotes_per_theme`). A quote has to be a verbatim substring of its passage. Shorter than `min_quote_chars` (default 15) is skipped. Only one quote is kept per respondent. The longer quote from the more frequent code is preferred. Tensions and negative cases come from the model; a quote on those is kept only when it is verbatim, and dropped otherwise. Passages with no segment, including ones skipped as empty or `n/a`, are listed as uncoded.

`output_dir` receives:

- `themes.md` — definition, properties, prevalence, metadata breakdowns, member codes, quotes, tensions, uncoded passages
- `themes.csv` — `theme,definition,codes,n_passages,pct_passages,n_documents,pct_documents,n_comments,pct_comments,n_respondents,pct_respondents`
- `themes.json` — the same records, plus tensions, negative cases, and uncoded passages
- `taguette_themes.csv` — Taguette codebook. One parent row per theme, then one row per code, path `Theme\Code` (backslash, which is what Taguette stores; backslashes inside a label are replaced)
- `taguette_themes.qdc` — the same paths in REFI-QDA codebook XML
- `theme_prevalence.png` — horizontal bars for percent of passages and percent of respondents

The chart needs matplotlib (`pip install "automated_grounded_theory[plots]"`). If it is not installed, the other files are still written and `themes.warnings` says the chart was skipped. Taguette imports `taguette_themes.csv` from Project info → Import codebook. After import, a code sits under its theme because the path is `Theme\Code` and the theme row is the prefix. Taguette does not import highlights or `.qdc`. Quote checks for each theme run concurrently and do not call the model.

`context` is the same sensitizing context as open coding. A mapping with preset codes raises `PresetCodesError` and does not call the model.

## Evaluate

`evaluate_run` scores a result or an output directory. With no `model`, it does not call a provider. Pass `model` and near-duplicate pairs are judged concurrently; Chen metrics then use those judgments. `adjusted_rand_index` compares passage partitions when `other_sources` is set (seed stability; Hubert and Arabie 1985). It scores 1.0 for different singleton codes on every passage, because it does not compare the labels. `aligned_assignment_agreement` and `aligned_assignment_alpha` score the set of codes on each passage after labels are aligned by exact text or a spelling variant. A negation does not align. `mapping` records a reviewed equivalence. The command-line `agt-evaluate` stays lexical and does not call a model.

```python
from automated_grounded_theory import evaluate_run

metrics = evaluate_run("coding_run", context="context.md", log_path="run.log")
```

The report includes coverage, codes per passage, verbatim-quote accuracy, quote-status rates (`quote_flag_rate` is the share flagged `possibly_modified` or `unanchored`), gerund and in-vivo rates, label length, singleton share, redundant pairs, the saturation curve and the first batch whose new-code rate is under 10%, merges, memo stats, context-leakage, calls, and tokens per passage.

Pass `other_sources` (more output directories or results on the same corpus) for Chen et al. coverage, overlap, novelty, and divergence (arXiv:2411.12142). The same scores are in `chen_metrics`. Without `model`, clustering is lexical Jaccard at 0.5, not the paper's model-based merge, so the default stays offline.

```bash
agt-evaluate coding_run context.md run.log --also coding_run/seed-1 coding_run/seed-2
```

## Method

- Glaser, B. G., & Strauss, A. L. (1967). *The Discovery of Grounded Theory*. Aldine. https://doi.org/10.4324/9780203793206
- Glaser, B. G. (1978). *Theoretical Sensitivity*. Sociology Press.
- Glaser, B. G. (1992). *Basics of Grounded Theory Analysis: Emergence vs. Forcing*. Sociology Press.
- Strauss, A., & Corbin, J. (1990/1998). *Basics of Qualitative Research*. Sage.
- Charmaz, K. (2006/2014). *Constructing Grounded Theory*. Sage.
- Übellacker, T. (2024). AcademiaOS. https://arxiv.org/abs/2403.08844
- A Practical Guide and Assessment on Using ChatGPT to Conduct Grounded Theory. JMIR, 2025. https://www.jmir.org/2025/1/e70122
- Chen, J., et al. Measuring open codes. https://arxiv.org/abs/2411.12142

The checks and the sweep exist because that JMIR tutorial found LLM coding weak on context, organization, and connections. Neighbors stay attached to each passage. Sweeps do the code-to-code comparison. Saturation is reported, not declared (Charmaz).

## Speed

`scripts/benchmark_review_modes.py` runs a fake coder that sleeps 0.05s per call on 6 documents × 4 passages (24 passages), seed 7, `sweep_every=8`, `unit="paragraph"`. No API key. It also runs an embedding-latency row that sleeps on each embedding call and ranks passages through the query API. `--embedding-delay 0` skips that row. `--unique-passages` gives that row a distinct query text per passage instead of four reused stems. Every number in this section is simulated: the coder, and the embedding model on that row, sleep instead of calling a provider. Remeasured on 0.6.11 at `--delay 0.05`. The reused embedding row embedded 4 queries (peak 3, 0.845s, 28.40 passages/s). `--unique-passages` embedded 20 (peak 4, 0.998s, 24.06 passages/s). On 0.6.0, `concurrency` widened the frozen window. From 0.6.4 only `passages_per_codebook_update` does that. The sequential row is what `strict_sequential_settings()` names.

| Mode | What it does | Seconds (simulated) | Calls | Passages/s | Codes | Merges | Near-duplicate pairs left |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sequential | `review_mode="document"`, `pipeline=False`, `batch_size=1`, `concurrency=1` | 1.659 | 28 | 14.47 | 3 | 3 | 0 |
| document | `review_mode="document"`, background synthesis, `batch_size=4`, `concurrency=4` | 0.563 | 10 | 42.66 | 3 | 1 | 0 |
| wave | `review_mode="wave"`, background synthesis, `batch_size=4`, `concurrency=4` | 0.593 | 11 | 40.47 | 3 | 1 | 0 |
| wave with unique-text embeddings | simulated; `review_mode="wave"`, distinct passage text, embedding calls sleep 0.05s (`--unique-passages`) | 0.998 | 11 | 24.06 | 3 | 1 | 0 |

On this fake coder, exact repeated labels collapse when they are committed, so none of the three left a near-duplicate pair behind. Document and wave each kept the sequential code count and recorded one merge. On this 0.05s sleep they took about the same time. The older 0.6.0 table, which showed wave much faster, was also simulated and is not this run. Wave mode still codes passages from different documents against one snapshot, so a later incident in a document can be coded before an earlier one. Re-run the script after you change the delay or the corpus. These times are simulated, not a provider benchmark.

## Benchmark

`benchmarks/run_benchmark.py` downloads the RAND Muse eval files and the BC Stats 2018 WES sample, runs `open_code` and `build_themes`, and asks the model to map each LLM code onto a human code. The report has per-code precision, recall, and F1, Cohen's kappa where the units line up, coverage of the human codes, novel LLM codes, and theme agreement for the BC Stats sample. Mapping rows carry a confidence and a rationale. Downloaded files go to `benchmarks/data/`, which is gitignored.

```bash
python benchmarks/run_benchmark.py --download
python benchmarks/run_benchmark.py --dataset muse-e10 --model xai:grok-4
python benchmarks/run_benchmark.py --dataset bcstats --model xai:grok-4
```

`--fixture` scores a tiny local directory and does not download anything.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`pytest` puts the repository root on `pythonpath`, so `tests` imports without an extra install step. The `dev` extra includes `langgraph-checkpoint-sqlite`. If that package is missing, the sqlite resume test is skipped and the in-memory resume test still runs.

Tests use a fake chat model and a four-document corpus in `tests/corpus.py`. Focused-coding tests use a second fake that groups those codes. Async tests cover a running event loop, rate-limit retry, and order-independent reconciliation. If Taguette is installed, `tests/test_taguette_roundtrip.py` imports the CSV through `taguette.import_codebook.list_tags` and checks that `Theme\Code` paths validate. They do not call a provider.

`tests/test_demo_notebook.py` executes [examples/demo.ipynb](examples/demo.ipynb) offline, in the same style as `nbmake`. The notebook in the repository is already executed in that offline mode. `pytest --nbmake examples/demo.ipynb` runs it the same way when the `dev` extra is installed.

## License

MIT. See [LICENSE](LICENSE).
