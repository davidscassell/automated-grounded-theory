# automated_grounded_theory

**Open coding**, then **focused coding**, in the grounded theory tradition (Glaser and Strauss 1967; Glaser 1978, 1992; Strauss and Corbin 1990/1998; Charmaz 2006/2014). By default the model reads every document in a seeded order, and every passage of that document in reading order, comparing each incident with the codebook earned so far. It does not decide that the data are saturated. How each step maps onto those sources is in [docs/methodology.md](docs/methodology.md). A short notebook with the same citations and the Taguette import steps is [docs/open_coding.ipynb](docs/open_coding.ipynb).

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
    unit="paragraph",
    batch_size=8,
    concurrency=2,
    output_dir="coding_run",
    checkpoint_path="coding_run/checkpoints.sqlite",
)
```

`model` is a `provider:model-name` string, passed to LangChain `init_chat_model`, or a `BaseChatModel` you built yourself. `sweep_model` defaults to the same model. `embeddings` is optional. When it is omitted, merge candidates are lexical and the sweep is still a chat-model call. You only need a chat model to run.

```python
from langchain.chat_models import init_chat_model

coder = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)
result = open_code(corpus, context, model=coder, seed=7)
```

`context` may be a question string or a mapping with `research_question`, `population`, `setting`, `sensitizing_concepts`, and `notes`. Sensitizing concepts steer attention. They are not codes. A mapping that includes `codes`, `codebook`, `tags`, or a coding paradigm raises `PresetCodesError`.

A corpus item can be a mapping with `text` or `path`, optional `id` / `name`, and optional `metadata` (department, tenure, and anything else about the respondent). Metadata is copied onto each segment and into `highlights.csv`. It is not a code, and the model is told not to put it in labels or definitions.

`unit` is `paragraph` (default), `sentence`, `line`, or `speaker_turn` (lines like `Name: ...`). Each passage keeps its document id, character offsets, and the neighboring passage. In document mode the seed shuffles documents and leaves passages in reading order. In wave mode the seed shuffles passages. `plan_coding` shows that order without calling a model. Empty passages and non-answers (`n/a`, `none`, punctuation) are kept in the passage list and are not sent to the model.

`review_mode` defaults to `document`. Documents are shuffled with the seed. Passages inside a document stay in reading order and are coded with that document's neighbors. The codebook is updated after the document (or after each passage when `batch_size` is 1). `review_mode="wave"` may code passages from several documents in one step against one frozen snapshot. That weakens constant comparison (Glaser and Strauss 1967): a later passage can be coded before an earlier one, and new codes are invisible to the rest of the wave. Use it only when you have accepted that tradeoff. `parallel_documents=True` is a second opt-in that gives each document its own codebook until a sweep. It is off by default.

`batch_size` (default 8, meant to stay in the 5–10 range) is how many passages share one model call. Batches are packed in coding order by an approximate token budget (`token_budget`, characters/4) so the split is a function of the seed, not of which call returns first. `concurrency` (default 2) is the ceiling on in-flight calls. A 429 halves that ceiling and a streak of successes raises it again (AIMD). Retries use full-jitter backoff and honor `Retry-After`. The system prompt and research context stay a stable prefix so a provider can cache them; the codebook snapshot and the passages come after.

`pipeline=True` (the default) runs synthesis of the passages just coded while the next document or wave is coded against the pre-merge snapshot. The merge map is applied when the sweep returns, in chunk order, and the surviving label is the lexicographically smaller code id. Set `pipeline=False` and `batch_size=1` for a fully sequential reading. Async entry points are `aopen_code`, `abuild_themes`, `aevaluate`, and `amulti_seed_stability`. The sync wrappers work inside a running event loop (a notebook) by using a worker thread.

`parallel_documents=True` codes each document on its own codebook and merges them at sweeps. It is off by default, because passages then do not see each other's new codes until the next sweep. `concurrency` caps how many documents are coded in one wave.

`interrupt_before_sweep=True` stops before a sweep so you can read the codebook. Resume with the same `checkpoint_path` or `checkpointer`, the same `thread_id`, and `resume=True`.

```python
from automated_grounded_theory import multi_seed_stability

report = multi_seed_stability(corpus, context, seeds=[1, 2, 3], model="xai:grok-4")
```

The report is label overlap across seeds. It does not pick a winner.

A later decision model can be passed as `code_retriever` if it implements `rank(passage, codebook, k) -> labels`. Nothing like that ships in this version. Coding and sweeps are chat-model calls.

## What the model is asked to do

For each passage the model says which code each incident indicates, and whether that code is already in the book. One passage can receive several codes, one per distinct incident. Codes should be short gerunds, close to the quote (Charmaz). When a vivid participant phrase does analytic work, the label should be that phrase and `in_vivo` should be true; the phrase has to occur in the passage. Before a new code, the model has to name the nearest existing code and why it does not fit. Definitions stay at the level of the incident: respondent attributes and the question wording do not belong in them. A memo is written only when the passage adds a property, a dimension, a condition, a consequence, or a comparison with a named code (Charmaz; Corbin and Strauss). A restatement is not a memo.

A plain check then runs, with no model:

- the quote must be found in the passage; the stored quote is that slice
- empty and generic labels (`experience`, `theme`, `discussion`, ...) are dropped
- an in-vivo flag is cleared when the label is not in the passage
- a non-gerund label is kept and flagged, unless `require_gerund=True`
- a new code with no nearest code, or a definition that repeats respondent metadata, is kept and flagged for review

Every `sweep_every` passages (and once at the end) a sweep model may merge codes. The whole codebook is sent, in chunks of 40 when it is larger than that, so later codes are not dropped. Chunks run concurrently and are reconciled by chunk index. When you pass `embeddings`, candidates are cosine similarity of label plus definition (default threshold 0.85) grouped by average-linkage agglomerative clustering. Without embeddings, candidates are label overlap. Those pairs are hints. The sweep model may also propose a duplicate pair itself. It can sharpen definitions. It can flag an overloaded code. It does not split one, because that would reassign old segments without rereading them. Accepted merges become a union-find map whose root is the lexicographically smaller code id, then that map relabels segments. A response cache replays an identical prompt on a rerun. After each document the run also stores a document memo (the incident memos from that reading) and a reflexivity note that the document was read in seeded order.

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

Taguette writes REFI-QDA `.qdc` and does not import it. `codebook.qdc` follows that export (`urn:QDA-XML:codebook:1.0`, `Code` with `guid`, `name`, `isCodable="true"`, and an empty `Sets`) and adds a `Description` child so other packages can read the definition. `highlights.csv` matches Taguette's highlight export shape (`id`, `document`, `tag`, `content`). Taguette does not import highlights.

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

`model` is the same kind of argument as in `open_code`: a `provider:model-name` string or a `BaseChatModel`. `n_themes=None` lets the model choose how many themes the codes support. Set it to ask for a specific number. `seed` only shuffles the order of codes in the prompt. Representative quotes do not use it.

A small LangGraph calls the model, then checks the assignment. Every open code must be assigned, and every name must be a code from the codebook. Unknown names are dropped. A code named twice keeps the first theme. If anything is missing, the model is shown that list and called again (`validation_retries`, default 1). `ThemeError` is raised when codes are still missing after those retries.

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

`evaluate_run` scores a result or an output directory. With no `model`, it does not call a provider. Pass `model` and near-duplicate pairs are judged concurrently; Chen metrics then use those judgments. `adjusted_rand_index` compares passage→code assignments when `other_sources` is set (seed stability; Hubert and Arabie 1985, as implemented for these labelings). The command-line `agt-evaluate` stays lexical and does not call a model.

```python
from automated_grounded_theory import evaluate_run

metrics = evaluate_run("coding_run", context="context.md", log_path="run.log")
```

The report includes coverage, codes per passage, verbatim-quote accuracy, gerund and in-vivo rates, label length, singleton share, redundant pairs, the saturation curve and the first batch whose new-code rate is under 10%, merges, memo stats, context-leakage, calls, and tokens per passage.

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

`scripts/benchmark_review_modes.py` runs a fake coder that sleeps 0.05s per call on 6 documents × 4 passages (24 passages), seed 7, `sweep_every=8`. No API key. Numbers below are from that script in this environment.

| Mode | What it does | Seconds | Calls | Passages/s | Codes | Merges | Near-duplicate pairs left |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sequential | `review_mode="document"`, `pipeline=False`, `batch_size=1`, `concurrency=1` | 1.455 | 27 | 16.50 | 3 | 3 | 0 |
| document | `review_mode="document"`, background synthesis, `batch_size=4`, `concurrency=4` | 0.389 | 9 | 61.64 | 3 | 3 | 0 |
| wave | `review_mode="wave"`, background synthesis, `batch_size=4`, `concurrency=4` | 0.177 | 8 | 135.34 | 3 | 1 | 0 |

On this fake coder, exact repeated labels collapse when they are committed, so none of the three left a near-duplicate pair behind. Document mode kept the same final code count as the sequential reading and still ran a merge on each synthesis. Wave mode was the fastest and needed only one merge pass, because passages from different documents shared a snapshot. That speed is the constant-comparison tradeoff described above: later incidents in a document can be coded before earlier ones. Re-run the script after you change the delay or the corpus; do not treat these times as a provider benchmark.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`pytest` puts the repository root on `pythonpath`, so `tests` imports without an extra install step. The `dev` extra includes `langgraph-checkpoint-sqlite`. If that package is missing, the sqlite resume test is skipped and the in-memory resume test still runs.

Tests use a fake chat model and a four-document corpus in `tests/corpus.py`. Focused-coding tests use a second fake that groups those codes. Async tests cover a running event loop, rate-limit retry, and order-independent reconciliation. If Taguette is installed, `tests/test_taguette_roundtrip.py` imports the CSV through `taguette.import_codebook.list_tags` and checks that `Theme\Code` paths validate. They do not call a provider.

## License

MIT. See [LICENSE](LICENSE).
