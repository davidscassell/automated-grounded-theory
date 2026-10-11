"""Compare model codes with a human codebook.

The scoring functions are offline. ``benchmarks/run_benchmark.py`` downloads
the RAND Muse files and the BC Stats sample into ``benchmarks/data/``, which
is gitignored, and then calls these functions.
"""

from __future__ import annotations

import csv
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET


def cohens_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    """Unweighted Cohen's kappa for two equal-length label sequences."""
    if len(left) != len(right) or not left:
        return None
    total = len(left)
    agreed = sum(str(a) == str(b) for a, b in zip(left, right)) / total
    left_counts = Counter(str(item) for item in left)
    right_counts = Counter(str(item) for item in right)
    categories = set(left_counts) | set(right_counts)
    expected = sum((left_counts[item] / total) * (right_counts[item] / total) for item in categories)
    if expected == 1:
        return 1.0
    return round((agreed - expected) / (1 - expected), 4)


def score_code_mapping(
    human_codes: Sequence[str],
    llm_codes: Sequence[str],
    mappings: Sequence[Mapping[str, Any]],
    *,
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    """Per-code precision, recall, and F1 from an LLM-judged code mapping.

    A mapping counts when it names a human code and its confidence is at least
    ``min_confidence``. An LLM code with no such mapping is novel.
    """
    human = list(dict.fromkeys(str(code) for code in human_codes if str(code).strip()))
    model = list(dict.fromkeys(str(code) for code in llm_codes if str(code).strip()))
    human_set = set(human)
    accepted = []
    for row in mappings:
        human_code = str(row.get("human_code") or "").strip()
        llm_code = str(row.get("llm_code") or "").strip()
        confidence = float(row.get("confidence") or 0)
        if not llm_code or not human_code or human_code not in human_set or confidence < min_confidence:
            continue
        accepted.append(row)
    covered = {str(row.get("human_code")) for row in accepted}
    matched = {str(row.get("llm_code")) for row in accepted}
    per_code = []
    for code in human:
        hit = code in covered
        per_code.append(
            {
                "code": code,
                "precision": 1.0 if hit else 0.0,
                "recall": 1.0 if hit else 0.0,
                "f1": 1.0 if hit else 0.0,
                "matched_llm_codes": [str(row.get("llm_code")) for row in accepted if str(row.get("human_code")) == code],
            }
        )
    precision = (len(matched & set(model)) / len(model)) if model else 0.0
    recall = (len(covered & human_set) / len(human)) if human else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    novel = [code for code in model if code not in matched]
    missing_rationale = [
        str(row.get("llm_code") or "")
        for row in mappings
        if not str(row.get("rationale") or "").strip()
    ]
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "human_coverage": round(recall, 4),
        "novel_llm_codes": novel,
        "per_code": per_code,
        "mappings_used": len(accepted),
        "missing_rationale": missing_rationale,
    }


def theme_agreement(human_themes: Sequence[str], llm_themes: Sequence[str]) -> dict[str, Any]:
    """Set overlap of theme names. Used for the BC Stats theme check."""
    human = {str(item).strip() for item in human_themes if str(item).strip()}
    model = {str(item).strip() for item in llm_themes if str(item).strip()}
    overlap = human & model
    precision = (len(overlap) / len(model)) if model else 0.0
    recall = (len(overlap) / len(human)) if human else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "shared": sorted(overlap),
        "novel_llm_themes": sorted(model - human),
        "missing_human_themes": sorted(human - model),
    }


def report_from_mapping(
    human_codes: Sequence[str],
    llm_codes: Sequence[str],
    mappings: Sequence[Mapping[str, Any]],
    *,
    human_labels: Sequence[str] | None = None,
    llm_labels: Sequence[str] | None = None,
    human_themes: Sequence[str] | None = None,
    llm_themes: Sequence[str] | None = None,
    stage_seconds: Mapping[str, Any] | None = None,
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    """Metrics for one comparison. Mapping rows need confidence and rationale."""
    scored = score_code_mapping(human_codes, llm_codes, mappings, min_confidence=min_confidence)
    scored["cohens_kappa"] = None
    if human_labels is not None and llm_labels is not None:
        scored["cohens_kappa"] = cohens_kappa(human_labels, llm_labels)
    if human_themes is not None and llm_themes is not None:
        scored["theme_agreement"] = theme_agreement(human_themes, llm_themes)
    if stage_seconds:
        scored["stage_seconds"] = dict(stage_seconds)
    return scored


_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _column_index(letters: str) -> int:
    number = 0
    for char in letters:
        number = number * 26 + (ord(char.upper()) - 64)
    return number


def read_xlsx(path: str | Path) -> list[dict[str, str]]:
    """First worksheet as a list of header-keyed rows. No extra dependency."""
    with zipfile.ZipFile(path) as handle:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in handle.namelist():
            root = ET.fromstring(handle.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", _NS):
                shared.append("".join(node.text or "" for node in item.iter(f"{_MAIN}t")))
        sheet = next(
            (
                name
                for name in handle.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            ),
            "",
        )
        if not sheet:
            return []
        root = ET.fromstring(handle.read(sheet))
    grid: list[dict[int, str]] = []
    for row in root.findall("m:sheetData/m:row", _NS):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", _NS):
            ref = cell.get("r") or ""
            letters = "".join(char for char in ref if char.isalpha())
            kind = cell.get("t")
            node = cell.find("m:v", _NS)
            inline = cell.find("m:is", _NS)
            if kind == "inlineStr" and inline is not None:
                text = "".join(part.text or "" for part in inline.iter(f"{_MAIN}t"))
            elif node is None or node.text is None:
                text = ""
            elif kind == "s":
                text = shared[int(node.text)] if node.text.isdigit() else ""
            else:
                text = node.text
            if letters:
                values[_column_index(letters)] = text
        grid.append(values)
    if not grid:
        return []
    header = grid[0]
    columns = sorted(header)
    rows: list[dict[str, str]] = []
    for values in grid[1:]:
        item = {}
        for index in columns:
            name = str(header.get(index) or "").strip()
            if name:
                item[name] = str(values.get(index) or "")
        if any(str(value).strip() for value in item.values()):
            rows.append(item)
    return rows


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{str(key): value for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]


def _field(row: Mapping[str, Any], *names: str) -> str:
    folded = {str(key).strip().casefold(): value for key, value in row.items() if key is not None and str(key).strip()}
    for name in names:
        value = folded.get(name.casefold())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _find_csv(root: Path, folder: str, stem: str) -> Path:
    direct = root / folder / f"{stem}.csv"
    if direct.is_file():
        return direct
    matches = [path for path in root.rglob(f"{stem}.csv") if folder.casefold() in path.parent.name.casefold()]
    if not matches:
        matches = list(root.rglob(f"{stem}.csv"))
    if not matches:
        raise FileNotFoundError(f"No {stem}.csv under {root}")
    return sorted(matches)[0]


def load_fixture_bundle(directory: str | Path) -> dict[str, Any]:
    """Tiny offline corpus: documents plus human codes and, when present, segment labels."""
    place = Path(directory)
    documents = _rows(place / "documents.csv")
    codes = _rows(place / "human_codes.csv")
    segments = _rows(place / "segments.csv") if (place / "segments.csv").is_file() else []
    return {"documents": documents, "human_codes": codes, "segments": segments}


def load_muse(root: str | Path, dataset: str) -> dict[str, Any]:
    """Read one Muse split (``e1``, ``e10``, or ``muse-e10``) from an extracted tree."""
    key = dataset.removeprefix("muse-").lower()
    place = Path(root)
    codebook_rows = _rows(_find_csv(place, "Codebooks", f"{key}_codebook"))
    document_rows = _rows(_find_csv(place, "Documents", f"{key}_documents"))
    tag_rows = _rows(_find_csv(place, "Tags", f"{key}_tags"))
    by_id: dict[str, dict[str, str]] = {}
    codes: list[dict[str, str]] = []
    parents: set[str] = set()
    for row in codebook_rows:
        code_id = _field(row, "id")
        label = _field(row, "label", "code")
        parent = _field(row, "parent_id")
        if parent:
            parents.add(parent)
        entry = {
            "id": code_id,
            "label": label,
            "definition": _field(row, "definition"),
            "parent_id": parent,
        }
        if code_id:
            by_id[code_id] = entry
        if label:
            codes.append(entry)
    leaves = [entry for entry in codes if entry["id"] not in parents] or codes
    documents = []
    for row in document_rows:
        text = _field(row, "transcript text", "text", "document_text")
        doc_id = _field(row, "document", "doc_id", "id")
        if text and doc_id:
            documents.append({"id": doc_id, "text": text})
    tags = []
    for row in tag_rows:
        label_id = _field(row, "label_id", "code_id")
        label = by_id.get(label_id, {}).get("label") or _field(row, "label", "code")
        excerpt = _field(row, "excerpt", "quote", "text")
        doc_id = _field(row, "document", "doc_id")
        if excerpt and label:
            tags.append({"document": doc_id, "excerpt": excerpt, "label_id": label_id, "label": label})
    return {
        "dataset": key,
        "documents": documents,
        "human_codes": [entry["label"] for entry in leaves if entry["label"]],
        "codebook": codes,
        "tags": tags,
    }


def load_theme_dictionary(path: str | Path) -> dict[str, dict[str, str]]:
    """BC Stats ``theme_subtheme_names.csv`` keyed by the numeric subtheme code."""
    found: dict[str, dict[str, str]] = {}
    for row in _rows(Path(path)):
        code = _field(row, "code")
        if not code:
            continue
        found[str(int(float(code)))] = {
            "theme": _field(row, "theme"),
            "subtheme": _field(row, "subtheme_description", "subtheme"),
        }
    return found


def bcstats_from_rows(rows: Sequence[Mapping[str, Any]], names: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Comments plus the human subthemes named by Code 1–5."""
    comments = []
    subthemes: list[str] = []
    themes: list[str] = []
    seen_sub: set[str] = set()
    seen_theme: set[str] = set()
    for row in rows:
        text = _field(row, "comment", "text")
        if not text:
            continue
        coded: list[str] = []
        themed: list[str] = []
        for slot in range(1, 6):
            raw = _field(row, f"Code {slot}", f"code {slot}", f"code_{slot}")
            if not raw:
                continue
            try:
                key = str(int(float(raw)))
            except ValueError:
                continue
            info = names.get(key)
            if not info:
                continue
            subtheme = str(info.get("subtheme") or "")
            theme = str(info.get("theme") or "")
            if subtheme and subtheme not in coded:
                coded.append(subtheme)
            if theme and theme not in themed:
                themed.append(theme)
            if subtheme and subtheme not in seen_sub:
                seen_sub.add(subtheme)
                subthemes.append(subtheme)
            if theme and theme not in seen_theme:
                seen_theme.add(theme)
                themes.append(theme)
        comments.append(
            {
                "id": _field(row, "userid", "user_id", "id") or f"row-{len(comments) + 1}",
                "text": text,
                "human_codes": coded,
                "human_themes": themed,
            }
        )
    codebook = [
        {"code": info.get("subtheme") or "", "definition": info.get("subtheme") or "", "theme": info.get("theme") or ""}
        for info in names.values()
        if info.get("subtheme")
    ]
    return {
        "documents": [{"id": item["id"], "text": item["text"]} for item in comments],
        "comments": comments,
        "human_codes": [item["code"] for item in codebook],
        "human_themes": themes,
        "codebook": codebook,
        "subthemes_used": subthemes,
    }


def load_bcstats(xlsx_path: str | Path, dictionary_path: str | Path) -> dict[str, Any]:
    """Read the WES 2018 sample workbook and the public theme dictionary."""
    return bcstats_from_rows(read_xlsx(xlsx_path), load_theme_dictionary(dictionary_path))


def _token_overlap(left: str, right: str) -> float:
    from automated_grounded_theory.retrieval import jaccard

    left_text = " ".join(str(left or "").split())
    right_text = " ".join(str(right or "").split())
    if not left_text or not right_text:
        return 0.0
    if left_text.casefold() in right_text.casefold() or right_text.casefold() in left_text.casefold():
        return 1.0
    return jaccard(left_text, right_text)


def excerpt_label_pairs(
    tags: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    mappings: Sequence[Mapping[str, Any]],
    *,
    min_confidence: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Human tag label and mapped LLM label for each excerpt, for Cohen's kappa."""
    accepted: dict[str, str] = {}
    for row in mappings:
        human_code = str(row.get("human_code") or "").strip()
        llm_code = str(row.get("llm_code") or "").strip()
        if not human_code or not llm_code or float(row.get("confidence") or 0) < min_confidence:
            continue
        accepted[llm_code] = human_code
    human_labels: list[str] = []
    model_labels: list[str] = []
    for tag in tags:
        excerpt = str(tag.get("excerpt") or "")
        document = str(tag.get("document") or "")
        human_labels.append(str(tag.get("label") or ""))
        best_label = ""
        best_score = 0.0
        for segment in segments:
            doc_id = str(segment.get("doc_id") or "")
            doc_name = str(segment.get("doc_name") or "")
            if document and document not in {doc_id, doc_name}:
                continue
            score = _token_overlap(excerpt, str(segment.get("quote") or ""))
            if score > best_score:
                best_score = score
                best_label = accepted.get(str(segment.get("code") or ""), "")
        model_labels.append(best_label or "unmatched")
    return human_labels, model_labels


def binary_theme_kappa(
    comments: Sequence[Mapping[str, Any]],
    mapped_themes_by_doc: Mapping[str, Sequence[str]],
) -> float | None:
    """Mean Cohen's kappa across themes. Each comment is one paired yes/no."""
    themes = sorted(
        {
            theme
            for comment in comments
            for theme in (comment.get("human_themes") or [])
        }
        | {theme for themes in mapped_themes_by_doc.values() for theme in themes}
    )
    scores = []
    for theme in themes:
        left = ["yes" if theme in (comment.get("human_themes") or []) else "no" for comment in comments]
        right = [
            "yes" if theme in mapped_themes_by_doc.get(str(comment.get("id") or ""), ()) else "no"
            for comment in comments
        ]
        score = cohens_kappa(left, right)
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


async def amap_codes(
    model: Any,
    llm_codes: Sequence[Mapping[str, Any] | str],
    human_codes: Sequence[Mapping[str, Any] | str],
    *,
    limiter: Any = None,
    chunk_size: int = 40,
) -> list[dict[str, Any]]:
    """Judge which human code each LLM code matches. Chunks run concurrently."""
    import asyncio

    from automated_grounded_theory.async_support import CallLimiter, ainvoke_structured
    from automated_grounded_theory.prompts import build_code_match_messages
    from automated_grounded_theory.schemas import CodeMatchBatch

    prepared_human = [
        item if isinstance(item, Mapping) else {"code": str(item), "definition": "", "theme": ""}
        for item in human_codes
    ]
    prepared_llm = [item if isinstance(item, Mapping) else {"code": str(item), "definition": ""} for item in llm_codes]
    prepared_llm = [item for item in prepared_llm if str(item.get("code") or "").strip()]
    if not prepared_llm:
        return []
    size = max(1, int(chunk_size))
    chunks = [prepared_llm[index : index + size] for index in range(0, len(prepared_llm), size)]
    gate = limiter or CallLimiter(max(1, min(4, len(chunks))))

    async def one(chunk: list) -> list[dict]:
        messages = build_code_match_messages(llm_codes=chunk, human_codes=prepared_human)
        (payload, _usage), _messages = await ainvoke_structured(model, CodeMatchBatch, messages, gate)
        return list(payload.get("matches") or [])

    groups = await asyncio.gather(*(one(chunk) for chunk in chunks))
    return [row for group in groups for row in group]
