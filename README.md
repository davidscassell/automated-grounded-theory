# automated_grounded_theory

**Open coding**, then **focused coding**, in the grounded theory tradition. A chat model reads every passage, in a seeded random order, and builds a codebook as it goes. A second pass every N passages compares codes with codes. The package reports how many new codes each batch added. It does not decide that the data are saturated.

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

`unit` is `paragraph` (default), `sentence`, `line`, or `speaker_turn` (lines like `Name: ...`). Each passage keeps its document id, character offsets, and the neighboring passage. Only the coding order is shuffled. `plan_coding` shows that order without calling a model. Empty passages and non-answers (`n/a`, `none`, punctuation) are kept in the passage list and are not sent to the model.

`batch_size` (default 8) is how many passages share one model call. `concurrency` (default 2) is how many of those calls run together. The codebook is updated after each wave, and a wave stops at the next sweep, so later passages still see codes earned earlier in the seeded order. Raise either number when a run is dominated by call latency. Set both to 1 to code one passage at a time.

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

Every `sweep_every` passages (and once at the end) a sweep model may merge codes. The whole codebook is sent, in chunks of 40 when it is larger than that, so later codes are not dropped. Candidate pairs come from embedding cosine when you pass `embeddings`, and from label overlap otherwise. Those pairs are hints. The sweep model may also propose a duplicate pair itself. It can sharpen definitions. It can flag an overloaded code. It does not split one, because that would reassign old segments without rereading them. Accepted renames are applied to earlier segments with a merge map.

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

Taguette's importer looks for a column named `tag`, `name`, or `path`, plus an optional `description`. Other columns are ignored, so `taguette_codebook.csv` imports from Project info → Import codebook. The header matches Taguette's own codebook CSV export. Dots in labels are replaced, because Taguette treats `.` as a parent/child split.

Taguette writes REFI-QDA `.qdc` and does not import it. `codebook.qdc` follows that export (`urn:QDA-XML:codebook:1.0`, `Code` with `guid`, `name`, `isCodable`) and adds a `Description` child so other packages can read the definition. `highlights.csv` matches Taguette's highlight export shape. Taguette does not import highlights.

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
- `taguette_themes.csv` — Taguette codebook, one row per code, tag path `Theme.Code` (dots inside a label are replaced, so the only split is between theme and code)
- `theme_prevalence.png` — horizontal bars for percent of passages and percent of respondents

The chart needs matplotlib (`pip install "automated_grounded_theory[plots]"`). If it is not installed, the other files are still written and `themes.warnings` says the chart was skipped. Taguette imports `taguette_themes.csv` from Project info → Import codebook. It does not import highlights.

`context` is the same sensitizing context as open coding. A mapping with preset codes raises `PresetCodesError` and does not call the model.

## Evaluate

`evaluate_run` scores a result or an output directory. It does not call a model.

```python
from automated_grounded_theory import evaluate_run

metrics = evaluate_run("coding_run", context="context.md", log_path="run.log")
```

The report includes coverage, codes per passage, verbatim-quote accuracy, gerund and in-vivo rates, label length, singleton share, redundant pairs, the saturation curve and the first batch whose new-code rate is under 10%, merges, memo stats, context-leakage, calls, and tokens per passage.

Pass `other_sources` (more output directories or results on the same corpus) for Chen et al. coverage, overlap, novelty, and divergence. The same scores are in `chen_metrics`. Clustering is lexical Jaccard at 0.5, not the paper's model-based merge, so this stays offline.

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

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`pytest` puts the repository root on `pythonpath`, so `tests` imports without an extra install step. The `dev` extra includes `langgraph-checkpoint-sqlite`. If that package is missing, the sqlite resume test is skipped and the in-memory resume test still runs.

Tests use a fake chat model and a four-document corpus in `tests/corpus.py`. Focused-coding tests use a second fake that groups those codes. They do not call a provider.

## License

MIT. See [LICENSE](LICENSE).
