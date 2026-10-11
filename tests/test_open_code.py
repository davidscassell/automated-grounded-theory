import json

import pytest

from automated_grounded_theory import open_code
from automated_grounded_theory.backends import prompt_cache_supported, resolve_chat_model
from automated_grounded_theory.context import PresetCodesError
from automated_grounded_theory.retrieval import EmbeddingCodeRetriever, cosine, merge_candidates
from automated_grounded_theory.stability import compare_runs
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver

from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder, FlakyCoder


def _run(model, **kwargs):
    kwargs.setdefault("seed", 7)
    kwargs.setdefault("sweep_every", 2)
    return open_code(CORPUS, CONTEXT, model=model, **kwargs)


def test_open_code_merges_and_keeps_verbatim_quotes(tmp_path):
    model = FakeCoder()
    result = _run(model, output_dir=tmp_path, api_key="sk-test-secret")
    assert not result.interrupted
    assert model.call_count == 6
    quotes = {item["passage_id"]: item["text"] for item in result.passages}
    assert len(result.segments) == 4
    for segment in result.segments:
        assert segment["quote"] in quotes[segment["passage_id"]]
    packing = [entry for entry in result.codebook if "packing" in entry["code"]]
    assert len(packing) == 1
    assert packing[0]["frequency"] == 2
    assert packing[0]["definition"].startswith("Bagging")
    assert sum(point["new_codes"] for point in result.saturation) == 4
    assert result.saturation_report["judgment"] is None
    assert "saturated" not in result.saturation_report
    assert any("Constant comparison" in memo["text"] for memo in result.memos)
    assert result.metadata["seed"] == 7
    assert "sk-test-secret" not in json.dumps(result.metadata)
    assert result.prompt_log
    assert "leaving home" in json.dumps(result.prompt_log).lower() or "Leaving" in json.dumps(result.prompt_log)
    assert (tmp_path / "taguette_codebook.csv").exists()
    assert (tmp_path / "codebook.qdc").exists()
    assert (tmp_path / "codebook.json").exists()


def test_same_seed_same_order():
    first = _run(FakeCoder())
    second = _run(FakeCoder())
    assert [item["passage_id"] for item in first.passages] == [item["passage_id"] for item in second.passages]


def test_parallel_documents_codes_every_document():
    result = _run(FakeCoder(), parallel_documents=True)
    assert {segment["doc_id"] for segment in result.segments} == {"alex", "blair", "casey", "drew"}
    assert len([entry for entry in result.codebook if "packing" in entry["code"]]) == 1


def test_retry_replaces_a_non_verbatim_quote():
    result = open_code(
        [CORPUS[0]],
        CONTEXT,
        model=FlakyCoder(),
        seed=1,
        sweep_every=25,
        validation_retries=1,
    )
    assert result.segments[0]["quote"] == "I spent the morning packing my suitcase."
    blob = json.dumps(result.validation_log)
    assert "quote_not_verbatim" in blob


def test_preset_codes_never_call_the_model():
    model = FakeCoder()
    with pytest.raises(PresetCodesError):
        open_code(CORPUS, {"research_question": "How?", "codebook": ["forced"]}, model=model)
    assert model.call_count == 0


def test_resume_from_memory_and_sqlite(tmp_path):
    saver = InMemorySaver()
    kwargs = dict(seed=3, sweep_every=2, checkpointer=saver, thread_id="run-1")
    paused = open_code(CORPUS, CONTEXT, model=FakeCoder(), interrupt_before_sweep=True, **kwargs)
    assert paused.interrupted
    assert len(paused.segments) == 2
    finished = open_code(CORPUS, CONTEXT, model=FakeCoder(), resume=True, **kwargs)
    assert not finished.interrupted
    assert len(finished.segments) == 4

    checkpoint = tmp_path / "run.sqlite"
    paused = open_code(
        CORPUS,
        CONTEXT,
        model=FakeCoder(),
        seed=3,
        sweep_every=2,
        checkpoint_path=checkpoint,
        thread_id="sqlite-1",
        interrupt_before_sweep=True,
    )
    assert paused.interrupted
    finished = open_code(
        CORPUS,
        CONTEXT,
        model=FakeCoder(),
        seed=3,
        sweep_every=2,
        checkpoint_path=checkpoint,
        thread_id="sqlite-1",
        resume=True,
    )
    assert len(finished.segments) == 4


def test_multi_seed_reports_overlap_without_a_winner():
    report = compare_runs([_run(FakeCoder(), seed=1), _run(FakeCoder(), seed=2)])
    assert report.mean_jaccard == 1
    assert report.codes_in_only_one_seed == []
    assert "does not" in report.note or "Neither" in report.note


def test_prompt_cache_and_provider_resolution(monkeypatch):
    class ChatAnthropic(FakeCoder):
        pass

    ChatAnthropic.__module__ = "langchain_anthropic.chat_models"
    assert prompt_cache_supported(ChatAnthropic(), "")
    assert not prompt_cache_supported(FakeCoder(), "xai:grok-4")

    cached = open_code([CORPUS[0]], CONTEXT, model=FakeCoder(), seed=1, sweep_every=25, prompt_cache=True)
    assert "cache_control" in json.dumps(cached.prompt_log)

    seen = {}

    def fake_init(model, **kwargs):
        seen["model"] = model
        seen["kwargs"] = kwargs
        return FakeCoder()

    import langchain.chat_models as chat_models

    monkeypatch.setattr(chat_models, "init_chat_model", fake_init)
    resolve_chat_model("xai:grok-4", api_key="sk-from-user", temperature=0)
    assert seen["model"] == "xai:grok-4"
    assert seen["kwargs"]["api_key"] == "sk-from-user"
    seen.clear()
    resolve_chat_model("xai:grok-4", temperature=0)
    assert "api_key" not in seen["kwargs"]
    model = FakeCoder()
    assert resolve_chat_model(model, api_key="ignored") is model


class _PackingEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        lowered = text.casefold()
        if "packing" in lowered:
            return [1.0, 0.05 if "bags up" in lowered else 0.0, 0.0]
        if "crying" in lowered or "station" in lowered:
            return [0.0, 0.0, 1.0]
        return [0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def test_embedding_candidates_are_optional():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    codes = [
        {"code": "packing bags", "definition": "packing", "frequency": 1, "examples": []},
        {"code": "packing bags up", "definition": "packing bags up", "frequency": 1, "examples": []},
        {"code": "crying at the station", "definition": "crying", "frequency": 1, "examples": []},
    ]
    pairs = merge_candidates(codes, embeddings=_PackingEmbeddings(), embedding_threshold=0.8)
    assert len(pairs) == 1
    assert {pairs[0]["source"], pairs[0]["target"]} == {"packing bags", "packing bags up"}
    ranked = EmbeddingCodeRetriever(_PackingEmbeddings()).rank("I was packing bags", codes, k=1)
    assert ranked == ["packing bags"]
