# automated_grounded_theory

First-round **open coding** in the grounded theory tradition. A chat model reads every passage, in a seeded random order, and builds a codebook as it goes. A second pass every N passages compares codes with codes. The package reports how many new codes each batch added. It does not decide that the data are saturated.

This round is open coding only. It does not do axial coding, selective coding, or theoretical sampling.

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

`unit` is `paragraph` (default), `sentence`, `line`, or `speaker_turn` (lines like `Name: ...`). Each passage keeps its document id, character offsets, and the neighboring passage. Only the coding order is shuffled. `plan_coding` shows that order without calling a model.

`parallel_documents=True` codes each document on its own codebook and merges them at sweeps. It is off by default, because passages then do not see each other's new codes until the next sweep.

`interrupt_before_sweep=True` stops before a sweep so you can read the codebook. Resume with the same `checkpoint_path` or `checkpointer`, the same `thread_id`, and `resume=True`.

```python
from automated_grounded_theory import multi_seed_stability

report = multi_seed_stability(corpus, context, seeds=[1, 2, 3], model="xai:grok-4")
```

The report is label overlap across seeds. It does not pick a winner.

A later decision model can be passed as `code_retriever` if it implements `rank(passage, codebook, k) -> labels`. Nothing like that ships in this version. Coding and sweeps are chat-model calls.

## What the model is asked to do

For each passage the model answers, in substance: what is happening here, which code does this incident indicate, and is that code already in the book? Codes should be short gerunds, close to the quote (Charmaz). An in-vivo label has to be the participant's own words and has to occur in the passage. The model may reuse a code, refine its definition or label, or add one.

A plain check then runs, with no model:

- the quote must be found in the passage; the stored quote is that slice
- empty and generic labels (`experience`, `theme`, `discussion`, ...) are dropped
- an in-vivo flag is cleared when the label is not in the passage
- a non-gerund label is kept and flagged, unless `require_gerund=True`

Every `sweep_every` passages (and once at the end) a sweep model may merge codes. Pairs come from embedding cosine, if you passed `embeddings`, or from token overlap otherwise. The model only confirms pairs on that list. It can sharpen definitions. It can flag an overloaded code. It does not split one, because that would reassign old segments without rereading them. Confirmed renames are applied to earlier segments with a merge map.

`result.saturation` is new codes per batch. `result.saturation_report["judgment"]` is always `None`.

## Export

`output_dir` receives:

- `codebook.json` / `codebook.csv` — code, definition, examples, frequency
- `coded_segments.json` / `coded_segments.csv` — quote, document, offsets
- `memos.md` / `memos.json` / `memos.csv`
- `saturation.csv`
- `taguette_codebook.csv` — header `tag,description,number of highlights,number of documents`
- `codebook.qdc` — REFI-QDA codebook
- `highlights.csv` — columns `id,document,tag,content`
- `run_metadata.json` and `prompt_log.json` — model names, seed, parameters, prompts

Taguette's importer looks for a column named `tag`, `name`, or `path`, plus an optional `description`. Other columns are ignored, so `taguette_codebook.csv` imports from Project info → Import codebook. The header matches Taguette's own codebook CSV export. Dots in labels are replaced, because Taguette treats `.` as a parent/child split.

Taguette writes REFI-QDA `.qdc` and does not import it. `codebook.qdc` follows that export (`urn:QDA-XML:codebook:1.0`, `Code` with `guid`, `name`, `isCodable`) and adds a `Description` child so other packages can read the definition. `highlights.csv` matches Taguette's highlight export shape. Taguette does not import highlights.

Prompt caching: the instructions and research context sit in a stable system prefix. Anthropic also gets an explicit `cache_control` breakpoint. Set `prompt_cache=True` or `False` to override.

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

Tests use a fake chat model and a four-document corpus in `tests/corpus.py`. They do not call a provider.

## License

MIT. See [LICENSE](LICENSE).
