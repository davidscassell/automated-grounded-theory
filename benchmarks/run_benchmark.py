"""Download public coded datasets and compare them with an open-coding run.

Data is written under ``benchmarks/data/`` and is not committed.

    python benchmarks/run_benchmark.py --download
    python benchmarks/run_benchmark.py --dataset muse-e10 --model xai:grok-4
    python benchmarks/run_benchmark.py --dataset bcstats --model xai:grok-4
    python benchmarks/run_benchmark.py --fixture benchmarks/fixtures/tiny --mapping benchmarks/fixtures/tiny/mapping.json

Muse files come from https://github.com/jdmatv/muse-eval-dataset.
The BC Stats 2018 sample comes from
https://github.com/aaronquinton/mds-capstone-bcstats (``2018_wes_qual_sample.xlsx``).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "data"
MUSE_ZIPS = {
    "Codebooks.zip": "https://github.com/jdmatv/muse-eval-dataset/raw/refs/heads/main/Codebooks.zip",
    "Documents.zip": "https://github.com/jdmatv/muse-eval-dataset/raw/refs/heads/main/Documents.zip",
    "Tags.zip": "https://github.com/jdmatv/muse-eval-dataset/raw/refs/heads/main/Tags.zip",
}
BC_XLSX_URLS = (
    "https://raw.githubusercontent.com/aaronquinton/mds-capstone-bcstats/master/data/raw/2018_wes_qual_sample.xlsx",
    "https://raw.githubusercontent.com/aaronquinton/mds-capstone-bcstats/main/data/raw/2018_wes_qual_sample.xlsx",
)
BC_NAMES_URL = (
    "https://raw.githubusercontent.com/aaronquinton/mds-capstone-bcstats/master/"
    "references/data-dictionaries/theme_subtheme_names.csv"
)
MUSE_QUESTIONS = {
    "e0": "What are people describing in these posts?",
    "e1": "How do researchers describe the work of curating data?",
    "e2": "How do people describe searching for information?",
    "e10": "How do people describe sustainability in their work and lives?",
}
BC_QUESTION = "What one thing would you like your organization to focus on to improve your work environment?"

sys.path.insert(0, str(ROOT))

from automated_grounded_theory.benchmark import (  # noqa: E402
    amap_codes,
    audit_sample,
    binary_theme_kappa,
    codes_tagged_in,
    excerpt_label_pairs,
    excerpt_offset_hits,
    load_bcstats,
    load_fixture_bundle,
    load_muse,
    quote_and_coverage_rates,
    report_from_mapping,
    unit_grid_from_annotations,
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            name = info.filename
            if name.startswith("__MACOSX") or "/._" in name or name.endswith(".DS_Store"):
                continue
            handle.extract(info, dest)


def download() -> None:
    """Fetch Muse and the BC Stats sample into ``benchmarks/data``. Does not commit them."""
    muse_root = DATA / "muse"
    for name, url in MUSE_ZIPS.items():
        archive = DATA / name
        _download(url, archive)
        _extract(archive, muse_root)
    errors = []
    for url in BC_XLSX_URLS:
        try:
            _download(url, DATA / "bcstats" / "2018_wes_qual_sample.xlsx")
            break
        except Exception as exc:  # noqa: BLE001 - try the other branch name
            errors.append(f"{url}: {exc}")
    else:
        raise RuntimeError("Could not download the BC Stats sample. " + "; ".join(errors))
    _download(BC_NAMES_URL, DATA / "bcstats" / "theme_subtheme_names.csv")


def load_fixture(directory: Path) -> dict:
    """Tiny offline corpus used by ``--fixture``."""
    return load_fixture_bundle(directory)


def _context(question: str) -> dict:
    return {"research_question": question, "population": "", "setting": ""}


def _stage_seconds(result, themes) -> dict:
    timing = dict((getattr(result, "metadata", {}) or {}).get("stage_seconds") or {})
    theme_meta = getattr(themes, "metadata", {}) or {}
    timing.update(dict(theme_meta.get("stage_seconds") or {}))
    return timing


def run_dataset(dataset: str, model: str, output: Path, limit: int | None) -> dict:
    """Open-code a downloaded dataset, build themes, and score the human codebook."""
    from automated_grounded_theory import build_themes, open_code
    from automated_grounded_theory.async_support import run_sync

    key = dataset.casefold()
    output.mkdir(parents=True, exist_ok=True)
    if key == "bcstats":
        bundle = load_bcstats(
            DATA / "bcstats" / "2018_wes_qual_sample.xlsx",
            DATA / "bcstats" / "theme_subtheme_names.csv",
        )
        question = BC_QUESTION
        documents = bundle["documents"]
        if limit:
            documents = documents[:limit]
            keep = {item["id"] for item in documents}
            comments = [item for item in bundle["comments"] if item["id"] in keep]
        else:
            comments = bundle["comments"]
        result = open_code(documents, _context(question), model=model, output_dir=output / "open_code")
        themes = build_themes(result, _context(question), model=model, output_dir=output / "themes")
        human = [{"code": item["code"], "definition": item["definition"], "theme": item["theme"]} for item in bundle["codebook"]]
        llm = [{"code": item["code"], "definition": item.get("definition") or ""} for item in result.codebook]
        mappings = run_sync(amap_codes(model, llm, human))
        by_subtheme = {item["code"]: item["theme"] for item in human}
        mapped_by_doc: dict[str, list[str]] = {}
        accepted = {
            str(row.get("llm_code") or ""): str(row.get("human_code") or "")
            for row in mappings
            if str(row.get("human_code") or "").strip() and float(row.get("confidence") or 0) >= 0.5
        }
        for segment in result.segments:
            human_code = accepted.get(str(segment.get("code") or ""))
            theme_name = by_subtheme.get(human_code or "")
            if not theme_name:
                continue
            doc_id = str(segment.get("doc_id") or "")
            bucket = mapped_by_doc.setdefault(doc_id, [])
            if theme_name not in bucket:
                bucket.append(theme_name)
        human_themes = [theme for comment in comments for theme in comment["human_themes"]]
        llm_themes = [theme for themes in mapped_by_doc.values() for theme in themes]
        report = report_from_mapping(
            bundle["human_codes"],
            [item["code"] for item in llm],
            mappings,
            human_themes=human_themes,
            llm_themes=llm_themes,
            stage_seconds=_stage_seconds(result, themes),
        )
        report["theme_kappa"] = binary_theme_kappa(comments, mapped_by_doc)
        report["llm_theme_names"] = [item.get("theme") or item.get("name") or "" for item in getattr(themes, "themes", [])]
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output / "mapping.json").write_text(json.dumps(mappings, indent=2), encoding="utf-8")
        return report

    bundle = load_muse(DATA / "muse", key)
    documents = bundle["documents"]
    if limit:
        documents = documents[:limit]
    keep = {item["id"] for item in documents}
    tags = [item for item in bundle["tags"] if item["document"] in keep]
    question = MUSE_QUESTIONS.get(bundle["dataset"], "What is happening in these documents?")
    _progress(f"{len(documents)} documents, {len(tags)} in-scope tags")
    result = open_code(documents, _context(question), model=model, output_dir=output / "open_code")
    themes = build_themes(result, _context(question), model=model, output_dir=output / "themes")
    in_scope = set(codes_tagged_in(tags, keep))
    human = [
        {"code": item["label"], "definition": item.get("definition") or "", "theme": ""}
        for item in bundle["codebook"]
        if item.get("label") and item["label"] in in_scope
    ]
    llm = [{"code": item["code"], "definition": item.get("definition") or ""} for item in result.codebook]
    mappings = _graded(run_sync(amap_codes(model, llm, human)))
    human_labels, llm_labels = excerpt_label_pairs(tags, result.segments, mappings)
    report = report_from_mapping(
        [item["code"] for item in human],
        [item["code"] for item in llm],
        mappings,
        human_labels=human_labels,
        llm_labels=llm_labels,
        stage_seconds=_stage_seconds(result, themes),
    )
    report["in_scope_human_codes"] = len(human)
    report["excerpt_offsets"] = excerpt_offset_hits(tags, result.segments, mappings, documents=documents)
    report["excerpt_offsets"].pop("rows", None)
    report["quote_and_coverage"] = quote_and_coverage_rates(result.passages, result.segments)
    report["unit_agreement"] = unit_grid_from_annotations(
        result.passages,
        result.segments,
        tags,
        mappings,
        documents,
    )
    sample = audit_sample(mappings, fraction=0.2, seed=0)
    report["audit_sample_size"] = len(sample)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "mapping.json").write_text(json.dumps(mappings, indent=2), encoding="utf-8")
    (output / "audit_sample.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    return report


def _progress(message: str) -> None:
    print(message, flush=True)


def _graded(mappings: list[dict]) -> list[dict]:
    graded = []
    for row in mappings:
        confidence = float(row.get("confidence") or 0)
        if row.get("grade"):
            grade = str(row["grade"])
        elif confidence >= 0.8:
            grade = "full"
        elif confidence >= 0.5:
            grade = "partial"
        else:
            grade = "none"
        llm_codes = list(row.get("llm_codes") or [])
        if not llm_codes and row.get("llm_code"):
            llm_codes = [row["llm_code"]]
        graded.append({**row, "grade": grade, "llm_codes": llm_codes})
    return graded


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Benchmark open coding against human codes.")
    parser.add_argument("--download", action="store_true", help="Download Muse and the BC Stats sample.")
    parser.add_argument("--fixture", type=Path, default=None, help="Score a tiny fixture directory and exit.")
    parser.add_argument("--mapping", type=Path, default=None, help="JSON list of llm_code/human_code/confidence/rationale.")
    parser.add_argument("--dataset", default="", help="muse-e1, muse-e2, muse-e10, or bcstats.")
    parser.add_argument("--model", default="", help="provider:model-name passed to open_code and the mapping judge.")
    parser.add_argument("--output", type=Path, default=DATA / "reports", help="Where the live run writes its report.")
    parser.add_argument("--limit", type=int, default=0, help="Code only the first N documents.")
    args = parser.parse_args()
    if args.download:
        download()
        _progress(f"Downloaded into {DATA}")
        return
    if args.fixture:
        bundle = load_fixture(args.fixture)
        mappings = json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else []
        human = [row.get("code") or row.get("label") or "" for row in bundle["human_codes"]]
        llm = [row.get("llm_code") or "" for row in mappings]
        human_labels = [row.get("human_code") or "" for row in bundle["segments"]]
        llm_labels = [row.get("llm_code") or "" for row in bundle["segments"]]
        themes_h = [row.get("theme") or "" for row in bundle["human_codes"] if row.get("theme")]
        themes_m = [row.get("theme") or "" for row in mappings if row.get("theme")]
        _progress(json.dumps(report_from_mapping(
            human,
            llm,
            mappings,
            human_labels=human_labels or None,
            llm_labels=llm_labels or None,
            human_themes=themes_h or None,
            llm_themes=themes_m or None,
        ), indent=2))
        return
    if args.dataset:
        if not args.model:
            parser.error("--dataset requires --model.")
        _progress(f"Coding {args.dataset} with {args.model}")
        report = run_dataset(args.dataset, args.model, args.output / args.dataset, args.limit or None)
        _progress(json.dumps(report, indent=2))
        return
    parser.error("Pass --download, --fixture, or --dataset.")


if __name__ == "__main__":
    main()
