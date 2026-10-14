"""Write a run as CSV, JSON, Markdown, Taguette codebook CSV, and REFI-QDA QDC.

Taguette's codebook importer (``taguette.import_codebook.list_tags``) reads a
CSV with a name column called ``tag``, ``name``, or ``path``, and an optional
``description`` column. Other columns are ignored. The CSV written here uses
Taguette's own export header (``tag``, ``description``, ``number of highlights``,
``number of documents``). A backslash in the path is the parent/child split
(``interesting\\places`` in Taguette's tests). Dots are ordinary characters.

Taguette exports REFI-QDA codebook XML (``.qdc``) and does not import it.
The QDC file follows that export (``Code`` with ``guid``, ``name``,
``isCodable``) plus a ``Description`` child for packages that read one.

``highlights.csv`` uses Taguette's highlight-export columns (``id``,
``document``, ``tag``, ``content``). Taguette does not import highlights.
"""

from __future__ import annotations

import csv
import json
import re
import uuid
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET

from automated_grounded_theory.category_state import code_id_sidecar
from automated_grounded_theory.version import __version__

QDC_NAMESPACE = "urn:QDA-XML:codebook:1.0"
# Stable ids for a given tag name. Not Taguette's namespace.
_QDC_UUID_NAMESPACE = uuid.UUID("a6e0c1d2-7b34-4f0a-9c11-0b6e2a9d4f18")


# Taguette stores a tag path as one string. A backslash separates a parent from
# its child (``interesting\places`` in Taguette's own tests and QDC export).
# The codebook CSV importer (``taguette.import_codebook.list_tags``) keeps that
# path as written. Dots are ordinary characters.
TAGUETTE_SEPARATOR = "\\"


def taguette_tag(label: str, used: set[str] | None = None) -> str:
    """Make one path component safe for Taguette.

    Backslashes in a label would look like extra hierarchy levels, so they are
    replaced. The path itself is joined later with :data:`TAGUETTE_SEPARATOR`.
    """
    cleaned = label.replace("\\", "·").replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned or "untitled-code"
    if used is None:
        return cleaned
    if cleaned not in used:
        used.add(cleaned)
        return cleaned
    number = 2
    while f"{cleaned} ({number})" in used:
        number += 1
    unique = f"{cleaned} ({number})"
    used.add(unique)
    return unique


def _description(entry: Mapping[str, Any]) -> str:
    definition = str(entry.get("definition") or "").strip()
    examples = entry.get("examples") or []
    lines = [definition] if definition else []
    quotes = []
    for example in examples[:3]:
        quote = str(example.get("quote") or "").strip()
        doc_id = example.get("doc_id") or ""
        if quote:
            quotes.append(f'- "{quote}" ({doc_id})' if doc_id else f'- "{quote}"')
    if quotes:
        if lines:
            lines.append("")
        lines.append("Examples:")
        lines.extend(quotes)
    if entry.get("flagged_overloaded"):
        lines.append("")
        lines.append("Flagged during a sweep as possibly overloaded. Not split automatically.")
    return "\n".join(lines).strip()


def codebook_qdc(codes: list[Mapping[str, Any]], *, origin: str | None = None, preserve_separator: bool = False) -> str:
    """Serialize a REFI-QDA codebook (``.qdc``).

    The shape matches Taguette's export: ``CodeBook`` / ``Codes`` / ``Code``
    with ``guid``, ``name``, and ``isCodable``, plus an empty ``Sets``. A
    ``Description`` child carries the definition for tools that read it.
    Taguette writes ``.qdc`` and does not import it. ``name`` uses a backslash
    for hierarchy when ``preserve_separator`` is set (``Theme\\Code``).
    """
    ns = QDC_NAMESPACE
    ET.register_namespace("", ns)
    root = ET.Element(f"{{{ns}}}CodeBook", {"origin": origin or f"automated_grounded_theory {__version__}"})
    codes_el = ET.SubElement(root, f"{{{ns}}}Codes")
    used: set[str] = set()
    for entry in codes:
        raw = str(entry.get("code") or "")
        tag = raw if preserve_separator else taguette_tag(raw, used)
        if preserve_separator:
            if tag in used:
                tag = taguette_tag(tag, used)
            else:
                used.add(tag)
        guid = str(uuid.uuid5(_QDC_UUID_NAMESPACE, tag)).upper()
        code_el = ET.SubElement(
            codes_el,
            f"{{{ns}}}Code",
            {"guid": guid, "name": tag, "isCodable": "true"},
        )
        description = _description(entry)
        if description:
            desc_el = ET.SubElement(code_el, f"{{{ns}}}Description")
            desc_el.text = description
    ET.SubElement(root, f"{{{ns}}}Sets")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True).decode("utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    """Drop callables and repr anything else JSON cannot encode.

    Model callbacks are the usual offender. Core files are written before
    metadata so a metadata value cannot take the codebook down with it.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if callable(value):
        return None
    return repr(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, default=repr) + "\n",
        encoding="utf-8",
    )


def _examples_cell(entry: Mapping[str, Any]) -> str:
    parts = []
    for example in entry.get("examples") or []:
        quote = str(example.get("quote") or "").replace("\n", " ")
        doc_id = example.get("doc_id") or ""
        parts.append(f"{quote} [{doc_id}]" if doc_id else quote)
    return " || ".join(parts)


def memos_markdown(memos: list[Mapping[str, Any]]) -> str:
    lines = ["# Memos", ""]
    if not memos:
        lines.append("_No memos._")
        return "\n".join(lines) + "\n"
    for memo in memos:
        kind = memo.get("kind") or "memo"
        title = kind
        if memo.get("passage_id"):
            title = f"{kind} {memo.get('passage_id')}"
            if memo.get("doc_id"):
                title += f" ({memo.get('doc_id')})"
        elif memo.get("sweep_index") is not None:
            title = f"{kind} {memo.get('sweep_index')}"
        lines.append(f"## {title}")
        lines.append("")
        lines.append(str(memo.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_result(result, directory: str | Path) -> dict[str, str]:
    """Write the run into ``directory`` and return a map of names to paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    codes = list(result.codebook)
    segments = list(result.segments)
    memos = list(result.memos)
    paths: dict[str, Path] = {}

    def place(name: str) -> Path:
        path = directory / name
        paths[name] = path
        return path

    _write_json(place("codebook.json"), codes)
    _write_csv(
        place("codebook.csv"),
        [
            "code",
            "definition",
            "in_vivo",
            "gerund_ok",
            "frequency",
            "number_of_documents",
            "examples",
            "first_seen",
            "flagged_overloaded",
            "valence",
            "confidence_mean",
            "confidence_min",
        ],
        [
            [
                entry.get("code"),
                entry.get("definition"),
                entry.get("in_vivo"),
                entry.get("gerund_ok"),
                entry.get("frequency"),
                len(entry.get("document_ids") or []),
                _examples_cell(entry),
                entry.get("first_seen"),
                entry.get("flagged_overloaded"),
                entry.get("valence") or "",
                entry.get("confidence_mean"),
                entry.get("confidence_min"),
            ]
            for entry in codes
        ],
    )

    used_tags: set[str] = set()
    tag_rows = []
    sidecar_codes = []
    for entry in codes:
        tag = taguette_tag(str(entry.get("code") or ""), used_tags)
        tag_rows.append(
            [
                tag,
                _description(entry),
                entry.get("frequency") or 0,
                len(entry.get("document_ids") or []),
            ]
        )
        sidecar_codes.append({**dict(entry), "tag": tag})
    _write_csv(
        place("taguette_codebook.csv"),
        ["tag", "description", "number of highlights", "number of documents"],
        tag_rows,
    )
    _write_json(place("code_id_sidecar.json"), code_id_sidecar(sidecar_codes))
    place("codebook.qdc").write_text(
        codebook_qdc(codes, origin=f"automated_grounded_theory {__version__}"),
        encoding="utf-8",
    )

    _write_json(place("coded_segments.json"), segments)
    _write_csv(
        place("coded_segments.csv"),
        [
            "segment_id",
            "highlight_id",
            "doc_id",
            "doc_name",
            "passage_id",
            "code",
            "quote",
            "quote_status",
            "quote_similarity",
            "quote_source",
            "model_quote",
            "doc_start",
            "doc_end",
            "in_vivo",
            "action",
            "code_id",
            "confidence",
            "in_vivo_confidence",
            "valence",
            "needs_review",
            "review_reasons",
            "rationale",
            "nearest_code",
            "metadata",
        ],
        [
            [
                item.get("segment_id"),
                item.get("highlight_id"),
                item.get("doc_id"),
                item.get("doc_name"),
                item.get("passage_id"),
                item.get("code"),
                item.get("quote"),
                item.get("quote_status"),
                item.get("quote_similarity"),
                item.get("quote_source"),
                item.get("model_quote"),
                item.get("doc_start"),
                item.get("doc_end"),
                item.get("in_vivo"),
                item.get("action"),
                item.get("code_id"),
                item.get("confidence"),
                item.get("in_vivo_confidence"),
                item.get("valence"),
                item.get("needs_review"),
                ";".join(item.get("review_reasons") or []),
                item.get("rationale"),
                item.get("nearest_code"),
                json.dumps(item.get("metadata") or {}, ensure_ascii=False),
            ]
            for item in segments
        ],
    )
    _write_csv(
        place("highlights.csv"),
        ["id", "document", "tag", "content", "metadata"],
        [
            [
                item.get("highlight_id"),
                item.get("doc_name") or item.get("doc_id"),
                taguette_tag(str(item.get("code") or "")),
                item.get("quote"),
                json.dumps(item.get("metadata") or {}, ensure_ascii=False),
            ]
            for item in segments
        ],
    )

    _write_json(place("memos.json"), memos)
    _write_csv(
        place("memos.csv"),
        ["memo_id", "kind", "doc_id", "passage_id", "sweep_index", "text"],
        [
            [
                memo.get("memo_id"),
                memo.get("kind"),
                memo.get("doc_id"),
                memo.get("passage_id"),
                memo.get("sweep_index"),
                memo.get("text"),
            ]
            for memo in memos
        ],
    )
    place("memos.md").write_text(memos_markdown(memos), encoding="utf-8")

    _write_json(place("saturation.json"), result.saturation_report)
    _write_csv(
        place("saturation.csv"),
        [
            "batch",
            "passages_in_batch",
            "new_codes",
            "new_codes_per_passage",
            "cumulative_passages",
            "cumulative_codes",
            "new_codes_pre_merge",
            "new_codes_post_merge",
            "new_codes_per_passage_post_merge",
            "cumulative_codes_pre_merge",
            "cumulative_codes_post_merge",
        ],
        [
            [
                point.get("batch"),
                point.get("passages_in_batch"),
                point.get("new_codes"),
                point.get("new_codes_per_passage"),
                point.get("cumulative_passages"),
                point.get("cumulative_codes"),
                point.get("new_codes_pre_merge", point.get("new_codes")),
                point.get("new_codes_post_merge", point.get("new_codes")),
                point.get("new_codes_per_passage_post_merge", point.get("new_codes_per_passage")),
                point.get("cumulative_codes_pre_merge", point.get("cumulative_codes")),
                point.get("cumulative_codes_post_merge", point.get("cumulative_codes")),
            ]
            for point in result.saturation
        ],
    )
    _write_json(place("passages.json"), result.passages)
    _write_json(place("validation_log.json"), result.validation_log)
    _write_json(place("sweep_log.json"), result.sweep_log)
    _write_json(place("ops.json"), list(getattr(result, "ops", []) or []))
    _write_json(place("review.json"), list(getattr(result, "review", []) or []))
    _write_json(place("prompt_log.json"), result.prompt_log)
    metadata = dict(result.metadata)
    metadata["interrupted"] = result.interrupted
    metadata["saturation_note"] = result.saturation_report.get("note")
    metrics = metadata.get("metrics") or {}
    _write_json(
        place("duplicate_rounds.json"),
        {
            "in_batch": list(metrics.get("in_batch_rounds") or []),
            "cross_snapshot": list(metrics.get("duplicate_rounds") or []),
        },
    )
    _write_json(place("lost_distinction_candidates.json"), list(metadata.get("lost_distinction_audit") or []))
    _write_json(place("run_metadata.json"), metadata)
    _write_json(place("run.json"), result.to_dict())
    return {name: str(path) for name, path in paths.items()}
