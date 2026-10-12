"""0.5.4: decide-once merges, one sweep call, rationale, timing, and the offline benchmark."""

import json
from copy import deepcopy
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory import evaluate_run, open_code
from automated_grounded_theory.async_support import run_sync
from automated_grounded_theory.benchmark import (
    amap_codes,
    bcstats_from_rows,
    binary_theme_kappa,
    cohens_kappa,
    excerpt_label_pairs,
    load_fixture_bundle,
    load_muse,
    load_theme_dictionary,
    read_xlsx,
    report_from_mapping,
    score_code_mapping,
    theme_agreement,
)
from automated_grounded_theory.codebook import (
    apply_sweep,
    frozen_pair_keys,
    remember_pair_decisions,
    trivial_merges,
    trivial_pair,
)
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.prompts import build_sweep_messages
from automated_grounded_theory.schemas import CodeMatch, CodeMatchBatch
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder

ROOT = Path(__file__).resolve().parents[1]
ROLLOUT = "criticizing rollout without training"
PROMO = "criticizing promotion without training"


def _entry(label: str, definition: str, valence: str = "negative") -> dict:
    return {
        "code": label,
        "key": label,
        "definition": definition,
        "valence": valence,
        "history": [],
    }


def _seg(label: str, quote: str) -> dict:
    return {
        "code_key": label,
        "code": label,
        "quote": quote,
        "passage_id": quote[:16],
        "doc_id": "d",
    }


def _pair_book() -> tuple[dict, list]:
    book = {
        ROLLOUT: _entry(ROLLOUT, "Criticizing a rollout that skipped training."),
        PROMO: _entry(PROMO, "Criticizing a promotion path that skipped training."),
    }
    segments = [
        _seg(ROLLOUT, "The rollout had no training."),
        _seg(PROMO, "Promotions skip the training."),
    ]
    return book, segments


def test_rejected_pair_stays_decided_until_the_definition_changes():
    book, segments = _pair_book()
    cache = remember_pair_decisions(
        [],
        book,
        [
            {
                "source": ROLLOUT,
                "target": PROMO,
                "decision": "reject",
                "confidence": 0.9,
                "reason": "verify_rejected",
                "rationale": "Both criticize missing training, but one is a rollout and the other is a promotion.",
                "source_valence": "negative",
                "target_valence": "negative",
            }
        ],
    )
    assert cache[0]["decision"] == "reject"
    assert cache[0]["source_valence"] == "negative"
    assert cache[0]["target_valence"] == "negative"
    frozen = frozen_pair_keys(cache, book)
    later = apply_sweep(
        book,
        segments,
        merges=[
            {
                "source": ROLLOUT,
                "target": PROMO,
                "confidence": 0.95,
                "rationale": "Both name missing training.",
            }
        ],
        candidates=[],
        frozen_pairs=frozen,
    )
    assert later["accepted_merges"] == []
    assert later["rejected_merges"][0]["reason"] == "already_decided"
    assert later["rejected_merges"][0]["source_valence"] == "negative"
    assert later["rejected_merges"][0]["target_valence"] == "negative"
    kept = remember_pair_decisions(
        cache,
        book,
        [
            {
                "source": ROLLOUT,
                "target": PROMO,
                "decision": "accept",
                "confidence": 0.95,
                "rationale": "A later sweep tried to accept the same pair.",
            }
        ],
    )
    assert kept[0]["decision"] == "reject"

    tweaked = deepcopy(book)
    tweaked[ROLLOUT]["definition"] = "Criticizing a rollout that skipped the training."
    assert frozen_pair_keys(cache, tweaked) == set()

    changed = deepcopy(book)
    changed[ROLLOUT]["definition"] = "Praising the cafeteria menu on Fridays."
    assert frozen_pair_keys(cache, changed) == set()
    merged = apply_sweep(
        changed,
        segments,
        merges=[
            {
                "source": ROLLOUT,
                "target": PROMO,
                "confidence": 0.9,
                "rationale": "The definitions now describe one incident.",
                "same_action": True,
                "same_object": True,
                "same_stance": True,
                "contrast_checked": True,
                "material_contrast": False,
            }
        ],
        candidates=[],
        frozen_pairs=set(),
    )
    assert merged["accepted_merges"] == []
    assert merged["rejected_merges"][0]["reason"] == "aspects_differ"


def test_quote_that_does_not_fit_is_split_back():
    book, segments = _pair_book()
    applied = apply_sweep(
        book,
        segments,
        merges=[
            {
                "source": ROLLOUT,
                "target": PROMO,
                "confidence": 0.9,
                "rationale": "Both criticize a change that skipped training.",
                "source_valence": "negative",
                "target_valence": "negative",
                "rewritten_label": "criticizing training gaps",
                "rewritten_definition": "Criticizing a change that went ahead without training.",
                "quote_fits": [
                    {
                        "quote": "Promotions skip the training.",
                        "fits": True,
                        "confidence": 0.9,
                        "rationale": "This quote is about promotion without training.",
                    },
                    {
                        "quote": "The rollout had no training.",
                        "fits": False,
                        "split_to": ROLLOUT,
                        "confidence": 0.85,
                        "rationale": "This quote is about a rollout, not a promotion.",
                    },
                ],
            }
        ],
        candidates=[],
    )
    assert applied["accepted_merges"] == []
    assert applied["rejected_merges"][0]["reason"] == "mixed_evidence"
    assert applied["rejected_merges"][0]["source_valence"] == "negative"
    assert applied["rejected_merges"][0]["target_valence"] == "negative"
    assert set(applied["codebook"]) == {ROLLOUT, PROMO}


def test_trivial_merge_skips_the_model_and_the_cap_defers():
    left = {
        "code": "criticizing the rollout",
        "valence": "negative",
        "definition": "Criticizing a rollout that skipped training.",
    }
    right = {
        "code": "criticizing the rollouts",
        "valence": "negative",
        "definition": "Criticizing a rollout that skipped training.",
    }
    assert trivial_pair(left, right)
    opposite = {**right, "valence": "positive"}
    assert not trivial_pair(left, opposite)
    extra = {**right, "code": "criticizing the rollout without any training"}
    assert not trivial_pair(left, extra)
    merges = trivial_merges([left, right])
    assert len(merges) == 1
    assert merges[0]["confidence"] == 0.95
    assert "inflection" in merges[0]["rationale"]

    labels = [
        "recording the alpha delay",
        "recording an alpha delay",
        "recording one alpha delay",
        "recording our alpha delay",
    ]
    book = {label: _entry(label, label, "neutral") for label in labels}
    segments = [_seg(label, label) for label in labels]
    flags = {
        "same_action": True,
        "same_object": True,
        "contrast_checked": True,
        "material_contrast": False,
        "evidence_same": True,
    }
    applied = apply_sweep(
        book,
        segments,
        merges=[
            {"source": "recording an alpha delay", "target": "recording the alpha delay", "confidence": 0.99, "rationale": "highest", **flags},
            {"source": "recording one alpha delay", "target": "recording the alpha delay", "confidence": 0.7, "rationale": "lower", **flags},
            {"source": "recording our alpha delay", "target": "recording the alpha delay", "confidence": 0.6, "rationale": "lowest", **flags},
        ],
        candidates=[],
        max_merges_per_sweep=1,
    )
    assert len(applied["accepted_merges"]) == 1
    assert {applied["accepted_merges"][0]["source"], applied["accepted_merges"][0]["target"]} == {
        "recording the alpha delay",
        "recording an alpha delay",
    }
    assert sum(item["reason"] == "merge_cap" for item in applied["review"]) == 2
    assert not any(item.get("reason") == "merge_cap" for item in applied["rejected_merges"])


def test_rubric_agreement_can_hold_a_confident_merge():
    book, segments = _pair_book()
    held = apply_sweep(
        book,
        segments,
        merges=[
            {
                "source": ROLLOUT,
                "target": PROMO,
                "confidence": 0.9,
                "rationale": "The labels share a verb and little else.",
                "same_action": False,
                "same_object": False,
                "same_stance": False,
            }
        ],
        candidates=[],
    )
    assert held["accepted_merges"] == []
    assert held["review"][0]["reason"] == "below_merge_confidence"
    assert held["review"][0]["confidence"] == 0.2
    applied = apply_sweep(
        book,
        segments,
        merges=[
            {
                "source": ROLLOUT,
                "target": PROMO,
                "confidence": 0.9,
                "rationale": "The model did not set the rubric, so the pair stays two codes.",
            }
        ],
        candidates=[],
    )
    assert applied["accepted_merges"] == []
    assert applied["rejected_merges"][0]["reason"] == "aspects_differ"


def test_run_sync_does_not_chain_a_closed_loop_under_no_running_loop():
    async def closed():
        raise RuntimeError("Event loop is closed")

    try:
        run_sync(closed())
    except RuntimeError as exc:
        assert str(exc) == "Event loop is closed"
        assert exc.__cause__ is None
        assert exc.__suppress_context__
        assert exc.__context__ is None or "no running event loop" not in str(exc.__context__)
    else:
        raise AssertionError("expected Event loop is closed")

    async def ok():
        return 3

    assert run_sync(ok()) == 3


def test_open_code_consolidates_and_keeps_rationale(tmp_path):
    result = open_code(CORPUS, CONTEXT, model=FakeCoder(), seed=7, sweep_every=2, output_dir=tmp_path)
    assert any(item.get("consolidation") for item in result.sweep_log)
    timing = result.metadata["stage_seconds"]
    assert "coding" in timing and "sweeps" in timing and "rewrites" in timing
    assert all(str(segment.get("rationale") or "").strip() for segment in result.segments)
    assert all(str(op.get("rationale") or "").strip() for op in result.ops)
    assert all(str(item.get("rationale") or "").strip() for item in result.review)
    coded = (tmp_path / "coded_segments.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "rationale" in coded.split(",")
    ops = json.loads((tmp_path / "ops.json").read_text(encoding="utf-8"))
    assert ops and all(str(op.get("rationale") or "").strip() for op in ops)
    taguette = (tmp_path / "taguette_codebook.csv").read_text(encoding="utf-8").splitlines()[0]
    assert taguette == "tag,description,number of highlights,number of documents"
    for record in json.loads((tmp_path / "sweep_log.json").read_text(encoding="utf-8")):
        for merge in record.get("accepted_merges") or []:
            assert str(merge.get("rationale") or "").strip()
            assert merge.get("source_valence")
            assert merge.get("target_valence")
    metrics = evaluate_run(result)
    assert "eval" in metrics["stage_seconds"]
    assert metrics["definition_not_survivor"] == 0

    piped = open_code(CORPUS, CONTEXT, model=FakeCoder(), seed=7, sweep_every=2, pipeline=True)
    serial = open_code(CORPUS, CONTEXT, model=FakeCoder(), seed=7, sweep_every=2, pipeline=False)
    assert len(piped.segments) == len(serial.segments)
    assert sorted(item["quote"] for item in piped.segments) == sorted(item["quote"] for item in serial.segments)


def test_sweep_system_prefix_stays_stable():
    context = ResearchContext(research_question="How do people leave?")
    first = build_sweep_messages(
        context=context,
        codes=[{"code": "packing bags", "definition": "Bagging.", "code_id": "a"}],
        candidates=[],
        prompt_cache=False,
    )
    second = build_sweep_messages(
        context=context,
        codes=[{"code": "crying at the station", "definition": "Tears.", "code_id": "b"}],
        candidates=[],
        prompt_cache=False,
        consolidation=True,
    )
    assert first[0].content == second[0].content
    assert first[1].content != second[1].content


def test_benchmark_fixture_scores_and_loaders():
    fixture = ROOT / "benchmarks" / "fixtures" / "tiny"
    bundle = load_fixture_bundle(fixture)
    mappings = json.loads((fixture / "mapping.json").read_text(encoding="utf-8"))
    scored = score_code_mapping(
        [row["code"] for row in bundle["human_codes"]],
        [row["llm_code"] for row in mappings],
        mappings,
    )
    assert scored["human_coverage"] == 1.0
    assert scored["novel_llm_codes"] == ["naming a small perk"]
    assert scored["missing_rationale"] == []
    assert scored["f1"] == 0.8
    assert cohens_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert cohens_kappa(["a"], ["b"]) == 0.0
    themes = theme_agreement(["Pay and benefits", "Leadership"], ["Pay and benefits", "Leadership", "Perks"])
    assert themes["recall"] == 1.0
    assert "Perks" in themes["novel_llm_themes"]
    blank = score_code_mapping(["pay"], ["pay"], [{"llm_code": "pay", "human_code": "pay", "confidence": 0.9, "rationale": ""}])
    assert blank["missing_rationale"] == ["pay"]

    muse = load_muse(ROOT / "benchmarks" / "fixtures" / "muse_tiny", "muse-e10")
    assert [item["id"] for item in muse["documents"]] == ["d1", "d2"]
    assert "wanting better pay" in muse["human_codes"]
    assert "Pay" not in muse["human_codes"]
    human_labels, llm_labels = excerpt_label_pairs(
        muse["tags"],
        [
            {"doc_id": "d1", "quote": "Pay is about 10 percent below market.", "code": "wanting better pay"},
            {"doc_id": "d2", "quote": "My manager actually listens when the shift runs long.", "code": "having a manager who listens"},
        ],
        mappings,
    )
    assert cohens_kappa(human_labels, llm_labels) == 1.0

    names = load_theme_dictionary(ROOT / "benchmarks" / "fixtures" / "bcstats_tiny" / "theme_subtheme_names.csv")
    table = bcstats_from_rows(
        [
            {"userid": "u1", "comment": "Pay is too low.", "Code 1": "24"},
            {"userid": "u2", "comment": "My supervisor never explains the change.", "Code 1": "83", "Code 2": "24"},
        ],
        names,
    )
    assert table["comments"][0]["human_themes"] == ["Compensation & Benefits"]
    assert "Increase salary" in table["human_codes"]
    kappa = binary_theme_kappa(
        table["comments"],
        {"u1": ["Compensation & Benefits"], "u2": ["Supervisors", "Compensation & Benefits"]},
    )
    assert kappa == 1.0
    report = report_from_mapping(
        table["human_codes"],
        ["Increase salary"],
        [
            {
                "llm_code": "Increase salary",
                "human_code": "Increase salary",
                "confidence": 0.9,
                "rationale": "Both name pay that should rise.",
            }
        ],
        human_themes=["Compensation & Benefits"],
        llm_themes=["Compensation & Benefits"],
        stage_seconds={"coding": 1.2, "sweeps": 0.4, "rewrites": 0.1, "themes": 0.2, "eval": 0.05},
    )
    assert report["stage_seconds"]["coding"] == 1.2
    assert report["theme_agreement"]["f1"] == 1.0


def test_mapping_judge_returns_rationale():
    class Mapper:
        def with_structured_output(self, schema, **kwargs):
            def _call(messages):
                parsed = CodeMatchBatch(
                    matches=[
                        CodeMatch(
                            llm_code="wanting better pay",
                            human_code="wanting better pay",
                            confidence=0.95,
                            rationale="Both codes name pay that is below what the work is worth.",
                        )
                    ]
                )
                if not kwargs.get("include_raw"):
                    return parsed
                return {"raw": AIMessage(content="{}"), "parsed": parsed, "parsing_error": None}

            return RunnableLambda(_call)

    rows = run_sync(
        amap_codes(
            Mapper(),
            [{"code": "wanting better pay", "definition": "Pay is low."}],
            [{"code": "wanting better pay", "definition": "Pay is below market.", "theme": "Pay"}],
        )
    )
    assert rows[0]["human_code"] == "wanting better pay"
    assert rows[0]["rationale"]
    assert rows[0]["confidence"] == 0.95


def test_read_xlsx_round_trip(tmp_path):
    path = tmp_path / "sample.xlsx"
    _write_tiny_xlsx(path, ["userid", "comment", "Code 1"], [["u1", "Pay is too low.", "24"]])
    rows = read_xlsx(path)
    assert rows == [{"userid": "u1", "comment": "Pay is too low.", "Code 1": "24"}]


def _write_tiny_xlsx(path: Path, headers: list[str], body: list[list[str]]) -> None:
    import zipfile

    strings = []
    for row in [headers, *body]:
        for cell in row:
            if cell not in strings:
                strings.append(cell)

    def col(index: int) -> str:
        return chr(ord("A") + index)

    def cell(row_number: int, index: int, value: str) -> str:
        return f'<c r="{col(index)}{row_number}" t="s"><v>{strings.index(value)}</v></c>'

    shared = "".join(f"<si><t>{value}</t></si>" for value in strings)
    rows = []
    for offset, row in enumerate([headers, *body], start=1):
        inside = "".join(cell(offset, index, value) for index, value in enumerate(row))
        rows.append(f'<row r="{offset}">{inside}</row>')
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(rows)}</sheetData></worksheet>"
    )
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{shared}</sst>"
    )
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
    workbook = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>"""
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("[Content_Types].xml", content_types)
        handle.writestr("_rels/.rels", rels)
        handle.writestr("xl/workbook.xml", workbook)
        handle.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        handle.writestr("xl/worksheets/sheet1.xml", sheet)
        handle.writestr("xl/sharedStrings.xml", shared_xml)
