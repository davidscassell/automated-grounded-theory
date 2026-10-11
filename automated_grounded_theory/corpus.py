"""Load a corpus from strings, mappings, or filesystem paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class LoadedDocument:
    doc_id: str
    name: str
    text: str
    source: str


CorpusItem = str | Path | Mapping[str, Any]


def normalize_newlines(text: str) -> str:
    """Return ``text`` with CR/LF folded to LF.

    Passage offsets refer to this normalized text.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _looks_like_path(value: str) -> bool:
    if not value or "\n" in value or len(value) > 4096:
        return False
    try:
        path = Path(value)
    except (OSError, ValueError):
        return False
    try:
        return path.is_file()
    except OSError:
        return False


def _read_path(path: Path) -> str:
    try:
        return normalize_newlines(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"Could not read corpus file {path}: {exc}") from exc


def _unique_id(base: str, used: set[str]) -> str:
    candidate = base or "document"
    if candidate not in used:
        used.add(candidate)
        return candidate
    number = 2
    while f"{candidate}-{number}" in used:
        number += 1
    unique = f"{candidate}-{number}"
    used.add(unique)
    return unique


def load_corpus(corpus: CorpusItem | Sequence[CorpusItem]) -> list[LoadedDocument]:
    """Load documents.

    Each item may be:

    * a string of document text (a string that is an existing file path is read)
    * a :class:`pathlib.Path`
    * a mapping with ``text`` or ``path`` and optional ``id`` / ``name``

    A single string or path is accepted as a one-document corpus.
    """
    if isinstance(corpus, (str, Path)) or isinstance(corpus, Mapping):
        items: Sequence[CorpusItem] = [corpus]
    else:
        items = list(corpus)
    if not items:
        raise ValueError("corpus must contain at least one document.")

    used: set[str] = set()
    documents: list[LoadedDocument] = []
    for index, item in enumerate(items, start=1):
        documents.append(_load_item(item, index, used))
    if not any(doc.text.strip() for doc in documents):
        raise ValueError("corpus documents are empty.")
    return documents


def _load_item(item: CorpusItem, index: int, used: set[str]) -> LoadedDocument:
    if isinstance(item, Path):
        text = _read_path(item)
        doc_id = _unique_id(item.stem or f"doc-{index}", used)
        return LoadedDocument(doc_id=doc_id, name=item.name, text=text, source=str(item))

    if isinstance(item, str):
        if _looks_like_path(item):
            path = Path(item)
            text = _read_path(path)
            doc_id = _unique_id(path.stem or f"doc-{index}", used)
            return LoadedDocument(doc_id=doc_id, name=path.name, text=text, source=str(path))
        text = normalize_newlines(item)
        doc_id = _unique_id(f"doc-{index}", used)
        return LoadedDocument(doc_id=doc_id, name=doc_id, text=text, source="inline")

    if isinstance(item, Mapping):
        name = str(item.get("name") or item.get("id") or f"doc-{index}")
        requested = str(item.get("id") or Path(name).stem or f"doc-{index}")
        if "text" in item and item["text"] is not None:
            text = normalize_newlines(str(item["text"]))
            source = str(item.get("source") or "inline")
        elif "path" in item and item["path"]:
            path = Path(item["path"])
            text = _read_path(path)
            source = str(path)
            name = str(item.get("name") or path.name)
        else:
            raise ValueError(f"Corpus item {index} needs 'text' or 'path'.")
        doc_id = _unique_id(requested, used)
        return LoadedDocument(doc_id=doc_id, name=name, text=text, source=source)

    raise TypeError(
        "Each corpus item must be a string, a path, or a mapping with 'text' or 'path'. "
        f"Got {type(item).__name__}."
    )
