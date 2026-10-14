"""Focused coding: a proposed code selection, then an allowance-bounded reread.

This is a separate graph from open coding and from the theme report. It reads
a finished open-coding run, ranks codes that are frequent, spread across
documents, or analytically heavy, and asks the researcher to accept or replace
that proposal. The reread tries the raised codes against passages and may add
assignments. It records properties and dimensions in their own store, each with
a citation. It does not write relations, a core category, or a saturation
judgment.

The researcher checkpoint calls :func:`langgraph.types.interrupt` inside the
node. Resume with :class:`langgraph.types.Command`. The default checkpointer
is :class:`langgraph.checkpoint.memory.InMemorySaver`.

Selection score
---------------
Each code is scored as

    frequency_component * focused_weight_frequency
    + document_component * focused_weight_documents
    + analytic_component * focused_weight_analytic

``frequency_component`` is the code's assignment count divided by the highest
count in the book (0 when the book is empty).

``document_component`` is the mean of the share of documents and the share of
respondents that carry the code. When every respondent id is the document id,
the two shares match.

``analytic_component`` is the mean of the enabled signals. It is not an
importance score. The memo term is this code's substantive-memo count divided
by the highest such count (0 when no code has one). Provenance notes do not
count. Memo count can reward a verbose memo; ablate ``memo`` to drop it. The
in-vivo term is 1 when the code is marked in vivo. The comparison term is 1
when a stored comparison names the code with ``refine_definition``,
``record_dimension``, ``link_to_category``, or ``keep_distinct``. A memo the
model wrote is the memo term, not a second score of importance.

The reread allowance is split across three purposes and interleaved:
developing existing assignments, testing competing interpretations, and
examining passages outside the initial selection. Shares are
``focused_share_develop``, ``focused_share_compete``, and
``focused_share_outside``. Coverage is reported per purpose.

A code is proposed by ``focused_selection_rule``. The default is ``top_k``:
about half the codes, ties broken by ``code_id``, and never every code unless
``focused_raise_all`` is set. ``quantile`` uses the same tie break and the
same ceiling. An explicit ``focused_min_score`` is a hard ``>=`` cutoff the
researcher chose. A researcher pin is proposed even when the score is low. A
contrast code, the only code on one pole of a dimension already recorded, is
proposed even when it is rare. Nothing is deleted from the codebook. The
lists are a proposal. The resume value can replace ``raised_code_ids``. A
selection that raises a high fraction of the book is warned on the result.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from functools import wraps
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from typing_extensions import TypedDict

from automated_grounded_theory.async_support import CallLimiter, ainvoke_structured, run_sync
from automated_grounded_theory.batching import request_input_tokens, resolve_tokenizer
from automated_grounded_theory.backends import chat_model_settings, describe_model
from automated_grounded_theory.category_state import (
    ANALYTIC_FEATURES,
    DEFAULT_SHARES,
    MEMO_COUNT_NOTE,
    REREAD_PURPOSES,
    STATE_VERSION,
    apply_category_edits,
    apply_claim_review,
    category_revision_id,
    citation_anchoring_report,
    claim_support_report,
    code_id_sidecar,
    evidence_condition_for,
    evidence_revision_id,
    interleave_queue,
    load_category_development,
    negative_case_candidates,
    normalize_analytic_features,
    normalize_shares,
    parse_incidents,
    proposal_record_fields,
    purpose_coverage,
    view_notes,
)
from automated_grounded_theory.codebook import (
    _next,
    codebook_list,
    commit_passage,
    empty_counters,
    interpret_coding,
    restore_id_counters,
)
from automated_grounded_theory.context import parse_context
from automated_grounded_theory.export import _write_csv, _write_json
from automated_grounded_theory.prompts import validation_feedback
from automated_grounded_theory.records import POLICY_VERSION, freeze_snapshot, proposal_buffer_reducer, select_committable
from automated_grounded_theory.results import OpenCodeResult, load_run
from automated_grounded_theory.retrieval import code_text, jaccard
from automated_grounded_theory.schemas import FocusedRecodeBatch
from automated_grounded_theory.validation import (
    anchor_quote,
    code_key,
)
from automated_grounded_theory.version import __version__

DEFAULT_FOCUSED_WEIGHT_FREQUENCY = 1.0
DEFAULT_FOCUSED_WEIGHT_DOCUMENTS = 1.0
DEFAULT_FOCUSED_WEIGHT_ANALYTIC = 1.0
DEFAULT_MAX_FOCUSED_REVIEW_CALLS = 8
DEFAULT_MAX_FOCUSED_REVIEW_TOKENS = 24000
DEFAULT_FOCUSED_NEIGHBOR_LIMIT = 1
DEFAULT_FOCUSED_OUTSIDE_LIMIT = 1
DEFAULT_FOCUSED_NEGATIVE_CASE_LIMIT = 1
DEFAULT_FOCUSED_SELECTION_RULE = "top_k"
DEFAULT_FOCUSED_SELECTION_QUANTILE = 0.5
DEFAULT_FOCUSED_RAISE_WARN_FRACTION = 0.75
SELECTION_RULES = ("top_k", "quantile")
DEFAULT_FOCUSED_BATCH_SIZE = 1
DEFAULT_FOCUSED_CONCURRENCY = 1
DEFAULT_FOCUSED_TOKEN_BUDGET = 6000

_JUDGMENT_OPS = frozenset(
    {"refine_definition", "record_dimension", "link_to_category", "keep_distinct"}
)
_MEMO_KINDS = frozenset({"passage", "document", "model_memo", "researcher_reflexivity", ""})
_CONTEXT_KEYS = ("research_question", "population", "setting", "sensitizing_concepts", "notes")
_PROCEDURE_KEYS = (
    "stage",
    "coding_lens",
    "seed",
    "focused_weight_frequency",
    "focused_weight_documents",
    "focused_weight_analytic",
    "focused_min_score",
    "focused_selection_rule",
    "focused_selection_k",
    "focused_selection_quantile",
    "focused_raise_all",
    "focused_raise_warn_fraction",
    "max_focused_review_calls",
    "max_focused_review_tokens",
    "focused_neighbor_limit",
    "focused_outside_limit",
    "focused_negative_case_limit",
    "focused_share_develop",
    "focused_share_compete",
    "focused_share_outside",
    "focused_analytic_features",
    "validation_retries",
)
_PERFORMANCE_KEYS = (
    "batch_size",
    "passages_per_request",
    "concurrency",
    "token_budget",
)
_STRUCTURAL_PATHS = frozenset({"corpus_hash", "source_hash", "procedure.seed"})
_RESUME_FORBIDDEN = frozenset(
    {"documents", "passages", "context", "model", "procedure", "corpus", "performance", "codebook"}
)

FOCUSED_SYSTEM = (
    "You are doing focused coding. The raised codes are a proposal the researcher can change. "
    "Reread only the passages in this request. Keep, add, or revise codes the passage supports. "
    "Quote the target passage. "
    "For each passage, name analytic properties and analytic dimensions of the raised categories: "
    "what varies in the incident, and the poles of that variation. "
    "Do not write a descriptive summary of the passage in place of a property. "
    "Each property and each dimension cites the document id, the passage id, and a quote copied from that passage. "
    "Also record a competing interpretation when the passage can be read with a different raised code "
    "or with an opposite pole of the same code. "
    "Do not write relations between categories. Do not name a core category. "
    "Do not decide that a category is saturated."
)


class FocusedError(RuntimeError):
    """The focused-coding run could not continue."""


class FocusedState(TypedDict, total=False):
    passages: list
    segments: list
    memos: list
    codebook: dict
    sweep_log: list
    documents: list
    metadata: dict
    warnings: list
    selection: list
    proposed_raised_code_ids: list
    proposed_not_raised_code_ids: list
    raised_code_ids: list
    not_raised_code_ids: list
    known_code_ids: list
    interrupt_enabled: bool
    researcher_decision: Any
    selection_source: str
    recode_queue: list
    recode_cursor: int
    queue_ready: bool
    round_jobs: list
    focused_calls_used: int
    focused_tokens_used: int
    focused_not_reread: list
    focused_reread: list
    recode_stop: str
    purpose_calls: dict
    purpose_tokens: dict
    purpose_coverage: dict
    revision_cycles: list
    properties: list
    dimensions: list
    prompt_log: list
    proposal_buffer: Annotated[dict, proposal_buffer_reducer]
    committed_ids: list
    counters: dict
    highlight_ids: dict
    snapshot_id: str
    snapshot_version: int
    export_paths: dict
    finished: bool


@dataclass
class FocusedRuntime:
    coder: BaseChatModel
    output_dir: str | None = None
    batch_size: int = DEFAULT_FOCUSED_BATCH_SIZE
    token_budget: int = DEFAULT_FOCUSED_TOKEN_BUDGET
    limiter: Any = None
    validation_retries: int = 1
    require_gerund: bool = False
    max_codes_per_passage: int = 6
    long_passage_chars: int = 2000
    review_below: float = 0.5

    def __post_init__(self) -> None:
        if self.limiter is None:
            self.limiter = CallLimiter(1)


@dataclass
class FocusedResult:
    """One focused-coding run. ``saturation_report['judgment']`` stays null."""

    raised_code_ids: list[str]
    not_raised_code_ids: list[str]
    selection: list[dict]
    codebook: list[dict]
    segments: list[dict]
    memos: list[dict]
    properties: list[dict]
    dimensions: list[dict]
    category_development: dict
    focused_not_reread: list[str]
    focused_reread: list[str]
    purpose_coverage: dict
    metadata: dict
    warnings: list[str] = field(default_factory=list)
    prompt_log: list[dict] = field(default_factory=list)
    export_paths: dict[str, str] = field(default_factory=dict)
    interrupted: bool = False
    interrupt_payload: dict | None = None
    thread_id: str = ""
    resumable: bool = False
    saturation_report: dict = field(default_factory=dict)
    checkpointer: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "raised_code_ids": self.raised_code_ids,
            "not_raised_code_ids": self.not_raised_code_ids,
            "selection": self.selection,
            "codebook": self.codebook,
            "segments": self.segments,
            "memos": self.memos,
            "properties": self.properties,
            "dimensions": self.dimensions,
            "category_development": self.category_development,
            "focused_not_reread": self.focused_not_reread,
            "focused_reread": self.focused_reread,
            "purpose_coverage": self.purpose_coverage,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "prompt_log": self.prompt_log,
            "export_paths": self.export_paths,
            "interrupted": self.interrupted,
            "interrupt_payload": self.interrupt_payload,
            "thread_id": self.thread_id,
            "resumable": self.resumable,
            "saturation_report": self.saturation_report,
        }
        return payload


class _IdleModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    @property
    def _llm_type(self) -> str:
        return "idle-focused"


def _as_result(source: OpenCodeResult | Mapping[str, Any] | str | Path) -> OpenCodeResult:
    if isinstance(source, OpenCodeResult):
        return source
    if isinstance(source, (str, Path)):
        return load_run(source)
    if isinstance(source, Mapping):
        return OpenCodeResult.from_dict(dict(source))
    raise TypeError("focused_code expects an open-coding result, a mapping, or a run directory.")


def _book_from_list(codes: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    book: dict[str, dict] = {}
    for entry in codes:
        row = dict(entry)
        key = str(row.get("key") or code_key(str(row.get("code") or "")))
        if not key:
            continue
        row["key"] = key
        row.setdefault("code_id", key)
        book[key] = row
    return book


def _code_rows(codebook: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> list[dict]:
    if isinstance(codebook, Mapping):
        return [dict(entry) for entry in codebook.values()]
    return [dict(entry) for entry in codebook]


def _respondent(passage: Mapping[str, Any]) -> str:
    meta = passage.get("metadata") or {}
    if isinstance(meta, Mapping):
        for key in ("respondent_id", "respondent", "participant_id", "participant"):
            value = meta.get(key)
            if value not in (None, ""):
                return str(value)
    return str(passage.get("doc_id") or "")


def _substantive_memo(memo: Mapping[str, Any]) -> bool:
    kind = str(memo.get("kind") or "")
    if kind == "provenance":
        return False
    return kind in _MEMO_KINDS


def _names_code(memo: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    code_id = str(entry.get("code_id") or "")
    label = str(entry.get("code") or "")
    linked = {str(item) for item in (memo.get("code_ids") or []) if str(item)}
    if memo.get("code_id"):
        linked.add(str(memo.get("code_id")))
    if code_id and code_id in linked:
        return True
    text = str(memo.get("text") or "").casefold()
    if label and label.casefold() in text:
        return True
    if code_id and code_id.casefold() in text:
        return True
    return False


def _walk_ops(sweep_log: Sequence[Any]) -> list[dict]:
    found: list[dict] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            found.append(dict(node))
            for key in ("ops", "merges", "recorded", "applied", "operations"):
                for child in node.get(key) or []:
                    visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for child in node:
                visit(child)

    for item in sweep_log or []:
        visit(item)
    return found


def _op_names(op: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    code_id = str(entry.get("code_id") or "")
    label = code_key(str(entry.get("code") or ""))
    values = [
        op.get("code"),
        op.get("code_id"),
        op.get("source"),
        op.get("source_id"),
        op.get("target"),
        op.get("target_id"),
    ]
    for value in values:
        text = str(value or "")
        if not text:
            continue
        if code_id and text == code_id:
            return True
        if label and code_key(text) == label:
            return True
    return False


def _model_judgments(entries: Sequence[Mapping[str, Any]], sweep_log: Sequence[Any]) -> set[str]:
    named: set[str] = set()
    for op in _walk_ops(sweep_log):
        kind = str(op.get("comparison_op") or op.get("op") or "")
        if kind not in _JUDGMENT_OPS:
            continue
        for entry in entries:
            if _op_names(op, entry):
                named.add(str(entry.get("code_id") or ""))
    return {item for item in named if item}


def contrast_code_ids(entries: Sequence[Mapping[str, Any]], sweep_log: Sequence[Any] | None = None) -> set[str]:
    """Codes that are the only one on a pole of a recorded dimension.

    A dimension with a single pole is not a contrast. The other pole has to
    exist, and this code has to be alone on its pole. Frequency is not required.
    """
    poles: dict[tuple[str, str], set[str]] = {}

    def add(code_id: str, name: str, value: str) -> None:
        if code_id and name and value:
            poles.setdefault((name, value), set()).add(code_id)

    by_id = {str(entry.get("code_id") or ""): entry for entry in entries}
    for entry in entries:
        code_id = str(entry.get("code_id") or "")
        for dim in entry.get("dimensions") or []:
            if not isinstance(dim, Mapping):
                continue
            add(code_id, str(dim.get("name") or ""), str(dim.get("value") or dim.get("pole") or ""))
    for op in _walk_ops(list(sweep_log or [])):
        if str(op.get("comparison_op") or op.get("op") or "") != "record_dimension":
            code_id = ""
            for entry in entries:
                if _op_names(op, entry):
                    code_id = str(entry.get("code_id") or "")
                    break
            if not code_id:
                continue
            add(
                code_id,
                str(op.get("dimension") or op.get("name") or ""),
                str(op.get("value") or op.get("pole") or op.get("valence") or ""),
            )
        else:
            code_id = ""
            for entry in entries:
                if _op_names(op, entry):
                    code_id = str(entry.get("code_id") or "")
                    break
            if not code_id and str(op.get("code_id") or "") in by_id:
                code_id = str(op.get("code_id") or "")
            add(
                code_id,
                str(op.get("dimension") or op.get("name") or "stance"),
                str(op.get("value") or op.get("pole") or op.get("valence") or ""),
            )
    by_name: dict[str, list[tuple[str, set[str]]]] = {}
    for (name, value), ids in poles.items():
        by_name.setdefault(name, []).append((value, ids))
    contrast: set[str] = set()
    for rows in by_name.values():
        if len(rows) < 2:
            continue
        for _value, ids in rows:
            if len(ids) == 1:
                contrast.add(next(iter(ids)))
    return contrast


def _pin_ids(entries: Sequence[Mapping[str, Any]], pins: Sequence[str] | None) -> set[str]:
    wanted = {str(item) for item in (pins or []) if str(item)}
    labels = {code_key(str(item)): str(item) for item in wanted}
    found: set[str] = set()
    for entry in entries:
        code_id = str(entry.get("code_id") or "")
        label = code_key(str(entry.get("code") or ""))
        pinned = bool(entry.get("pinned") or entry.get("researcher_pin"))
        if code_id in wanted or label in labels or pinned:
            if code_id:
                found.add(code_id)
    unresolved = []
    known_ids = {str(entry.get("code_id") or "") for entry in entries}
    known_labels = {code_key(str(entry.get("code") or "")) for entry in entries}
    for item in wanted:
        if item in known_ids or code_key(item) in known_labels:
            continue
        unresolved.append(item)
    if unresolved:
        raise FocusedError(f"Researcher pins are not in the codebook: {unresolved}")
    return found


def selection_target(
    n: int,
    *,
    rule: str = DEFAULT_FOCUSED_SELECTION_RULE,
    k: int | None = None,
    quantile: float = DEFAULT_FOCUSED_SELECTION_QUANTILE,
    raise_all: bool = False,
) -> int:
    """How many codes the score rule may raise.

    Ties are not included past this count. The default is the upper half,
    and it is at most ``n - 1`` when there is more than one code, unless
    ``raise_all`` is set.
    """
    if n <= 0:
        return 0
    if raise_all:
        return n
    if n == 1:
        return 1
    if rule == "top_k":
        chosen = (n + 1) // 2 if k is None else int(k)
    elif rule == "quantile":
        if not 0 < float(quantile) < 1:
            raise ValueError("focused_selection_quantile must be between 0 and 1.")
        chosen = math.floor((1.0 - float(quantile)) * n)
    else:
        raise ValueError(f"focused_selection_rule must be one of {SELECTION_RULES}.")
    return min(max(1, chosen), n - 1)


def select_focused_codes(
    codebook: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    passages: Sequence[Mapping[str, Any]] | None = None,
    segments: Sequence[Mapping[str, Any]] | None = None,
    memos: Sequence[Mapping[str, Any]] | None = None,
    sweep_log: Sequence[Any] | None = None,
    pins: Sequence[str] | None = None,
    weight_frequency: float = DEFAULT_FOCUSED_WEIGHT_FREQUENCY,
    weight_documents: float = DEFAULT_FOCUSED_WEIGHT_DOCUMENTS,
    weight_analytic: float = DEFAULT_FOCUSED_WEIGHT_ANALYTIC,
    min_score: float | None = None,
    analytic_features: Sequence[str] | None = None,
    selection_rule: str = DEFAULT_FOCUSED_SELECTION_RULE,
    selection_k: int | None = None,
    selection_quantile: float = DEFAULT_FOCUSED_SELECTION_QUANTILE,
    raise_all: bool = False,
    warn_fraction: float = DEFAULT_FOCUSED_RAISE_WARN_FRACTION,
) -> dict[str, Any]:
    """Propose ``raised_code_ids``. The codebook is not filtered.

    See the module docstring for the score. Pins and contrast codes are
    proposed even when the score is below the cutoff. ``min_score`` None uses
    ``selection_rule`` (top-k by default). A float is a hard ``>=`` threshold.
    Codes tied at the cutoff are ordered by ``code_id`` and only the target
    count is raised. ``analytic_features`` drops memo count, the in-vivo flag,
    or stored comparisons from the analytic term.
    """
    if weight_frequency < 0 or weight_documents < 0 or weight_analytic < 0:
        raise ValueError("Focused-coding weights must be zero or more.")
    rule = str(selection_rule or DEFAULT_FOCUSED_SELECTION_RULE)
    if rule not in SELECTION_RULES:
        raise ValueError(f"focused_selection_rule must be one of {SELECTION_RULES}.")
    if selection_k is not None and int(selection_k) < 1:
        raise ValueError("focused_selection_k must be at least 1.")
    if not 0 < float(selection_quantile) < 1:
        raise ValueError("focused_selection_quantile must be between 0 and 1.")
    if not 0 < float(warn_fraction) <= 1:
        raise ValueError("focused_raise_warn_fraction must be above 0 and at most 1.")
    enabled = normalize_analytic_features(analytic_features)
    entries = _code_rows(codebook)
    passage_rows = [dict(item) for item in (passages or [])]
    segment_rows = [dict(item) for item in (segments or [])]
    memo_rows = [dict(item) for item in (memos or []) if _substantive_memo(dict(item))]
    documents = {str(item.get("doc_id") or "") for item in passage_rows if item.get("doc_id")}
    for entry in entries:
        for doc_id in entry.get("document_ids") or []:
            if doc_id:
                documents.add(str(doc_id))
    respondents = {_respondent(item) for item in passage_rows if _respondent(item)}
    if not respondents:
        respondents = set(documents)
    memo_counts: dict[str, int] = {}
    for entry in entries:
        code_id = str(entry.get("code_id") or "")
        memo_counts[code_id] = sum(1 for memo in memo_rows if _names_code(memo, entry))
    max_memos = max(memo_counts.values(), default=0)
    judgments = _model_judgments(entries, list(sweep_log or []))
    contrasts = contrast_code_ids(entries, sweep_log)
    pinned = _pin_ids(entries, pins)
    frequencies = []
    for entry in entries:
        count = entry.get("frequency")
        if count is None or count == "":
            count = len(entry.get("assignments") or [])
        frequencies.append(int(count or 0))
    max_frequency = max(frequencies, default=0)
    rows = []
    for entry, frequency in zip(entries, frequencies):
        code_id = str(entry.get("code_id") or "")
        doc_ids = {str(item) for item in (entry.get("document_ids") or []) if str(item)}
        if not doc_ids:
            doc_ids = {
                str(segment.get("doc_id") or "")
                for segment in segment_rows
                if str(segment.get("code_id") or "") == code_id and segment.get("doc_id")
            }
        resp_ids = set()
        passage_by_id = {str(item.get("passage_id") or ""): item for item in passage_rows}
        for segment in segment_rows:
            if str(segment.get("code_id") or "") != code_id:
                continue
            passage = passage_by_id.get(str(segment.get("passage_id") or ""))
            if passage is not None:
                resp_ids.add(_respondent(passage))
            elif segment.get("doc_id"):
                resp_ids.add(str(segment.get("doc_id")))
        if not resp_ids:
            resp_ids = set(doc_ids)
        freq_component = (frequency / max_frequency) if max_frequency else 0.0
        doc_share = (len(doc_ids) / len(documents)) if documents else 0.0
        resp_share = (len(resp_ids) / len(respondents)) if respondents else doc_share
        document_component = (doc_share + resp_share) / 2
        memo_component = (memo_counts.get(code_id, 0) / max_memos) if max_memos else 0.0
        vivo_component = 1.0 if entry.get("in_vivo") else 0.0
        judgment_component = 1.0 if code_id in judgments else 0.0
        parts = []
        if "memo" in enabled:
            parts.append(memo_component)
        if "in_vivo" in enabled:
            parts.append(vivo_component)
        if "comparison" in enabled:
            parts.append(judgment_component)
        analytic_component = sum(parts) / len(parts) if parts else 0.0
        score = (
            float(weight_frequency) * freq_component
            + float(weight_documents) * document_component
            + float(weight_analytic) * analytic_component
        )
        rows.append(
            {
                "code_id": code_id,
                "code": str(entry.get("code") or ""),
                "frequency": frequency,
                "frequency_component": round(freq_component, 6),
                "document_share": round(doc_share, 6),
                "respondent_share": round(resp_share, 6),
                "document_component": round(document_component, 6),
                "substantive_memos": memo_counts.get(code_id, 0),
                "in_vivo": bool(entry.get("in_vivo")),
                "model_judgment": code_id in judgments,
                "analytic_component": round(analytic_component, 6),
                "analytic_features": list(enabled),
                "score": round(score, 6),
                "contrast": code_id in contrasts,
                "pin": code_id in pinned,
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), row["code_id"]))
    threshold = None
    target = None
    if raise_all:
        score_ids = {row["code_id"] for row in rows}
        target = len(rows)
    elif min_score is not None:
        threshold = float(min_score)
        score_ids = {row["code_id"] for row in rows if row["score"] >= threshold}
    else:
        target = selection_target(
            len(rows),
            rule=rule,
            k=selection_k,
            quantile=float(selection_quantile),
            raise_all=False,
        )
        score_ids = {row["code_id"] for row in rows[:target]}
    for row in rows:
        reasons = []
        if row["code_id"] in score_ids:
            reasons.append("raise_all" if raise_all else "score")
        if row["pin"]:
            reasons.append("pin")
        if row["contrast"]:
            reasons.append("contrast")
        row["raised"] = bool(reasons)
        row["raise_reason"] = reasons
    raised = [row["code_id"] for row in rows if row["raised"]]
    not_raised = [row["code_id"] for row in rows if not row["raised"]]
    fraction = (len(raised) / len(rows)) if rows else 0.0
    warnings: list[str] = []
    if rows and fraction + 1e-12 >= float(warn_fraction):
        detail = (
            f"Focused selection raised {len(raised)} of {len(rows)} codes "
            f"({fraction:.0%}), at or above {float(warn_fraction):.0%}."
        )
        if len(raised) == len(rows) and raise_all:
            detail += " focused_raise_all is set, so the score rule included every code."
        elif len(raised) == len(rows):
            detail += " Pins or contrast codes completed a set the score rule did not raise on its own."
        warnings.append(detail)
    return {
        "rows": rows,
        "raised_code_ids": raised,
        "not_raised_code_ids": not_raised,
        "threshold": None if threshold is None else round(float(threshold), 6),
        "selection_rule": "raise_all" if raise_all else ("min_score" if min_score is not None else rule),
        "selection_k": None if target is None else int(target),
        "selection_quantile": float(selection_quantile),
        "raise_all": bool(raise_all),
        "tie_policy": "code_id",
        "raised_fraction": round(fraction, 6),
        "warnings": warnings,
        "weights": {
            "focused_weight_frequency": float(weight_frequency),
            "focused_weight_documents": float(weight_documents),
            "focused_weight_analytic": float(weight_analytic),
        },
        "analytic_features": list(enabled),
        "analytic_ablation": [name for name in ANALYTIC_FEATURES if name not in enabled],
        "analytic_note": MEMO_COUNT_NOTE if "memo" in enabled else "Memo count is ablated and does not enter the score.",
    }


def apply_selection_override(
    known_ids: Sequence[str],
    proposed: Sequence[str],
    decision: Any,
) -> tuple[list[str], bool]:
    """Replace the proposal when the resume value lists ``raised_code_ids``.

    ``None``, ``{}``, ``True``, and ``{"accept": true}`` keep the proposal.
    A listed id that is not in the codebook raises. The codebook is not edited.
    """
    known = [str(item) for item in known_ids if str(item)]
    known_set = set(known)
    if decision is None or decision is True or decision == {}:
        return list(proposed), False
    if not isinstance(decision, Mapping):
        raise FocusedError(
            "Resume with {'accept': true} or {'raised_code_ids': [...]}. "
            "The value cannot replace the corpus, context, model, or procedure."
        )
    forbidden = sorted(set(decision) & _RESUME_FORBIDDEN)
    if forbidden:
        raise FocusedError(
            "The resume value cannot replace "
            + ", ".join(forbidden)
            + ". Those are checked on the focused_code call against the checkpoint manifest."
        )
    if "raised_code_ids" not in decision and (
        decision.get("accept") is True or "incidents" in decision or "incidents_csv" in decision
    ):
        return list(proposed), False
    if "raised_code_ids" not in decision:
        raise FocusedError("Resume with {'accept': true} or {'raised_code_ids': [...]}.")
    requested = [str(item) for item in (decision.get("raised_code_ids") or [])]
    unknown = [item for item in requested if item not in known_set]
    if unknown:
        raise FocusedError(f"raised_code_ids are not in the codebook: {unknown}")
    changed = requested != list(proposed)
    return requested, changed


def _queue_item(passage_id: str, purpose: str, source: str) -> dict[str, str]:
    return {"passage_id": passage_id, "purpose": purpose, "source": source}


def _dimension_claims(codebook: Sequence[Mapping[str, Any]], raised: set[str]) -> list[dict[str, Any]]:
    claims = []
    for entry in codebook:
        code_id = str(entry.get("code_id") or "")
        if code_id not in raised:
            continue
        for dimension in entry.get("dimensions") or []:
            claims.append(
                {
                    "entry_id": f"dim-{code_id}-{dimension.get('name')}-{dimension.get('value') or dimension.get('pole')}",
                    "kind": "dimension",
                    "category_id": code_id,
                    "name": dimension.get("name"),
                    "pole": dimension.get("value") or dimension.get("pole"),
                }
            )
    return claims


def _norm_pole(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _text_has_pole(text: str, pole: str) -> bool:
    token = _norm_pole(pole)
    return bool(token) and token in _norm_pole(text)


def _codes_by_passage(segments: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for segment in segments:
        passage_id = str(segment.get("passage_id") or "")
        code_id = str(segment.get("code_id") or "")
        if passage_id and code_id:
            found.setdefault(passage_id, set()).add(code_id)
    return found


def _pole_index(codebook: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, set[str]]], dict[str, set[str]]]:
    by_code: dict[str, dict[str, set[str]]] = {}
    all_poles: dict[str, set[str]] = {}
    for entry in codebook:
        code_id = str(entry.get("code_id") or "")
        for dimension in entry.get("dimensions") or []:
            if not isinstance(dimension, Mapping):
                continue
            name = _norm_pole(dimension.get("name"))
            pole = _norm_pole(dimension.get("value") or dimension.get("pole"))
            if name and pole:
                by_code.setdefault(code_id, {}).setdefault(name, set()).add(pole)
                all_poles.setdefault(name, set()).add(pole)
    return by_code, all_poles


def _has_opposite_pole(text: str, code_ids: set[str], poles: tuple[dict, dict]) -> bool:
    by_code, all_poles = poles
    for code_id in code_ids:
        for name, own in (by_code.get(code_id) or {}).items():
            others = set(all_poles.get(name) or ()) - set(own)
            if any(_text_has_pole(text, pole) for pole in others):
                return True
    return False


def _passage_confidence(passage_id: str, segments: Sequence[Mapping[str, Any]], raised: set[str]) -> float:
    scores = []
    for segment in segments:
        if str(segment.get("passage_id") or "") != passage_id:
            continue
        if str(segment.get("code_id") or "") not in raised:
            continue
        raw = segment.get("confidence")
        if raw in (None, ""):
            continue
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue
    return min(scores) if scores else 1.0


def _compete_fallback(
    by_id: Mapping[str, Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    codebook: Sequence[Mapping[str, Any]],
    raised: set[str],
    blocked: set[str],
) -> list[tuple[str, str]]:
    """Assigned passages that can test a competing reading.

    Order: another code already on the passage, an opposite pole in the text,
    then a raised code the passage was not given.
    """
    held = _codes_by_passage(segments)
    poles = _pole_index(codebook)
    assigned = [
        passage_id
        for passage_id in by_id
        if passage_id not in blocked and (held.get(passage_id) or set()) & raised
    ]
    alternative: list[str] = []
    opposite: list[str] = []
    untried: list[str] = []
    for passage_id in assigned:
        codes = held.get(passage_id) or set()
        if len(codes) > 1:
            alternative.append(passage_id)
            continue
        if _has_opposite_pole(str(by_id[passage_id].get("text") or ""), codes & raised, poles):
            opposite.append(passage_id)
            continue
        if raised - codes:
            untried.append(passage_id)
    return (
        [(passage_id, "alternative_code") for passage_id in alternative]
        + [(passage_id, "opposite_pole") for passage_id in opposite]
        + [(passage_id, "untried_code") for passage_id in untried]
    )


def _outside_fallback(
    by_id: Mapping[str, Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    raised: set[str],
    blocked: set[str],
    assigned_ids: set[str],
) -> list[tuple[str, str]]:
    """Passages outside the raised assignments, or the weakest assignments."""
    unassigned = [
        passage_id
        for passage_id in by_id
        if passage_id not in assigned_ids and passage_id not in blocked
    ]
    if unassigned:
        return [(passage_id, "unassigned") for passage_id in unassigned]
    ranked = sorted(
        (passage_id for passage_id in assigned_ids if passage_id in by_id and passage_id not in blocked),
        key=lambda passage_id: (_passage_confidence(passage_id, segments, raised), passage_id),
    )
    return [(passage_id, "low_confidence") for passage_id in ranked]


def _reallocate_shares(
    requested: Mapping[str, float],
    groups: Mapping[str, Sequence[Any]],
) -> tuple[dict[str, float], dict[str, Any], list[str]]:
    """Move a positive share off a purpose that still has no passages."""
    empty = [purpose for purpose in REREAD_PURPOSES if not groups.get(purpose) and float(requested[purpose]) > 0]
    active = [purpose for purpose in REREAD_PURPOSES if groups.get(purpose)]
    if not empty:
        return dict(requested), {"reallocated": False, "purposes": [], "note": ""}, []
    if not active:
        note = (
            "Focused reread purposes "
            + ", ".join(empty)
            + " had no passages, and no other purpose had a queue to receive their shares."
        )
        return (
            dict(requested),
            {
                "reallocated": True,
                "purposes": empty,
                "note": note,
                "shares_requested": dict(requested),
                "shares_applied": dict(requested),
            },
            [note],
        )
    applied = {purpose: 0.0 for purpose in REREAD_PURPOSES}
    freed = sum(float(requested[purpose]) for purpose in empty)
    active_total = sum(float(requested[purpose]) for purpose in active)
    for purpose in active:
        portion = (float(requested[purpose]) / active_total) if active_total else (1 / len(active))
        applied[purpose] = float(requested[purpose]) + freed * portion
    note = (
        "Focused reread "
        + ("purpose " if len(empty) == 1 else "purposes ")
        + ", ".join(f"{purpose} (share {float(requested[purpose]):.3f})" for purpose in empty)
        + " had no passages after search, so "
        + ("that share was" if len(empty) == 1 else "those shares were")
        + " reallocated to "
        + ", ".join(f"{purpose} {applied[purpose]:.3f}" for purpose in active)
        + "."
    )
    return (
        applied,
        {
            "reallocated": True,
            "purposes": empty,
            "note": note,
            "shares_requested": {purpose: float(requested[purpose]) for purpose in REREAD_PURPOSES},
            "shares_applied": applied,
        },
        [note],
    )


def build_recode_queue(
    passages: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    raised_code_ids: Sequence[str],
    codebook: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    neighbor_limit: int,
    outside_limit: int = DEFAULT_FOCUSED_OUTSIDE_LIMIT,
    negative_case_limit: int = DEFAULT_FOCUSED_NEGATIVE_CASE_LIMIT,
    shares: Mapping[str, float] | None = None,
    incidents: Sequence[Mapping[str, Any]] | None = None,
    claims: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Interleave three reserved reread purposes. The same seed replays the queue."""
    return plan_recode_queue(
        passages,
        segments,
        raised_code_ids,
        codebook,
        seed=seed,
        neighbor_limit=neighbor_limit,
        outside_limit=outside_limit,
        negative_case_limit=negative_case_limit,
        shares=shares,
        incidents=incidents,
        claims=claims,
    )["queue"]


def plan_recode_queue(
    passages: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    raised_code_ids: Sequence[str],
    codebook: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    neighbor_limit: int,
    outside_limit: int = DEFAULT_FOCUSED_OUTSIDE_LIMIT,
    negative_case_limit: int = DEFAULT_FOCUSED_NEGATIVE_CASE_LIMIT,
    shares: Mapping[str, float] | None = None,
    incidents: Sequence[Mapping[str, Any]] | None = None,
    claims: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the reread queue and say when a reserved purpose was empty.

    ``develop`` holds passages already assigned to a raised code.
    ``compete`` holds within-category negative cases and a bounded lexical
    neighborhood. When that list is empty, assigned passages fill it:
    another code on the passage, an opposite pole, or a raised code not yet
    applied. ``outside`` holds passages outside that selection. When none
    remain, the lowest-confidence raised assignments fill it. A purpose that
    is still empty gives its share back explicitly, and the plan carries a
    warning. Researcher incidents are not dropped to satisfy a limit.
    """
    share_values = normalize_shares(shares)
    raised = {str(item) for item in raised_code_ids}
    by_id = {str(item.get("passage_id") or ""): dict(item) for item in passages if item.get("passage_id")}
    assigned_ids = {
        str(segment.get("passage_id") or "")
        for segment in segments
        if str(segment.get("code_id") or "") in raised and segment.get("passage_id")
    }
    claim_rows = list(claims) if claims is not None else _dimension_claims(codebook, raised)
    hits = negative_case_candidates(list(by_id.values()), segments, claim_rows, codebook=codebook)
    within = []
    outside_hits = []
    for hit in hits:
        passage_id = hit["passage_id"]
        if passage_id not in by_id:
            continue
        if hit["reason"] == "within_category" and passage_id not in within:
            within.append(passage_id)
        elif hit["reason"] == "outside_support" and passage_id not in outside_hits:
            outside_hits.append(passage_id)
    researcher = {"develop": [], "compete": [], "outside": []}
    for incident in incidents or []:
        passage_id = str(incident.get("passage_id") or "")
        purpose = str(incident.get("purpose") or "")
        if passage_id in by_id and purpose in researcher:
            researcher[purpose].append(passage_id)
    reserved = {passage_id for group in researcher.values() for passage_id in group}
    compete_negative = [
        passage_id
        for passage_id in within
        if passage_id not in reserved
    ][: max(0, int(negative_case_limit))]
    within_selected = set(compete_negative)
    develop = [
        passage_id
        for passage_id in by_id
        if passage_id in assigned_ids and passage_id not in within_selected and passage_id not in reserved
    ]
    neighbor_ids: list[str] = []
    if neighbor_limit > 0 and raised:
        codes = [entry for entry in codebook if str(entry.get("code_id") or "") in raised]
        scored = []
        blocked = set(develop) | set(compete_negative) | reserved | assigned_ids
        for passage_id, passage in by_id.items():
            if passage_id in blocked:
                continue
            best = 0.0
            text = str(passage.get("text") or "")
            for entry in codes:
                best = max(best, jaccard(text, code_text(entry)))
            if best > 0:
                scored.append((-best, passage_id))
        scored.sort()
        neighbor_ids = [passage_id for _score, passage_id in scored[: int(neighbor_limit)]]
    outside_priority = [
        passage_id
        for passage_id in outside_hits
        if passage_id not in reserved
        and passage_id not in develop
        and passage_id not in compete_negative
        and passage_id not in neighbor_ids
        and passage_id not in assigned_ids
    ]
    outside_rest = [
        passage_id
        for passage_id in by_id
        if passage_id not in assigned_ids
        and passage_id not in reserved
        and passage_id not in neighbor_ids
        and passage_id not in compete_negative
        and passage_id not in outside_priority
    ]
    outside_auto = (outside_priority + outside_rest)[: max(0, int(outside_limit))]
    rng = random.Random(int(seed))
    rng.shuffle(develop)
    compete_auto = list(compete_negative)
    rng.shuffle(compete_auto)
    neighbor_shuffled = list(neighbor_ids)
    rng.shuffle(neighbor_shuffled)
    outside_shuffled = list(outside_auto)
    rng.shuffle(outside_shuffled)
    compete_fallback: list[tuple[str, str]] = []
    outside_fallback: list[tuple[str, str]] = []
    # Reserve outside first so a weak assignment is not swallowed by compete.
    if not outside_shuffled and int(outside_limit) > 0:
        blocked = reserved | set(compete_auto) | set(neighbor_shuffled)
        outside_fallback = _outside_fallback(by_id, segments, raised, blocked, assigned_ids)[: int(outside_limit)]
    if not compete_auto and not neighbor_shuffled:
        budget = max(0, int(negative_case_limit)) + max(0, int(neighbor_limit))
        if budget > 0:
            blocked = reserved | set(outside_shuffled) | {passage_id for passage_id, _source in outside_fallback}
            compete_fallback = _compete_fallback(by_id, segments, codebook, raised, blocked)[:budget]
    moved = {passage_id for passage_id, _source in compete_fallback + outside_fallback}
    develop = [passage_id for passage_id in develop if passage_id not in moved]
    groups = {
        "develop": [_queue_item(passage_id, "develop", "researcher") for passage_id in researcher["develop"]]
        + [_queue_item(passage_id, "develop", "assignment") for passage_id in develop],
        "compete": [_queue_item(passage_id, "compete", "researcher") for passage_id in researcher["compete"]]
        + [_queue_item(passage_id, "compete", "negative_case") for passage_id in compete_auto]
        + [_queue_item(passage_id, "compete", "neighbor") for passage_id in neighbor_shuffled]
        + [_queue_item(passage_id, "compete", source) for passage_id, source in compete_fallback],
        "outside": [_queue_item(passage_id, "outside", "researcher") for passage_id in researcher["outside"]]
        + [_queue_item(passage_id, "outside", "outside") for passage_id in outside_shuffled]
        + [_queue_item(passage_id, "outside", source) for passage_id, source in outside_fallback],
    }
    applied, reallocation, warnings = _reallocate_shares(share_values, groups)
    return {
        "queue": interleave_queue(groups, applied),
        "warnings": warnings,
        "shares_requested": share_values,
        "shares_applied": applied,
        "reallocation": reallocation,
        "purpose_counts": {purpose: len(groups[purpose]) for purpose in REREAD_PURPOSES},
    }


def cite_note(
    note: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    entry_id: str,
    coded_passage_id: str = "",
    codebook: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Keep a property or dimension and attach the passage that holds the quote.

    A quote found in a different passage than the one named is attributed to
    the passage that contains it and flagged. A quote found nowhere is kept,
    flagged ``unsupported``, and not assigned to the passage being coded.
    """
    claimed_passage = str(note.get("passage_id") or "")
    claimed_doc = str(note.get("document_id") or "")
    quote = str(note.get("quote") or "")
    by_id = {str(item.get("passage_id") or ""): item for item in passages}
    target = by_id.get(claimed_passage)
    target_text = str(target.get("text") or "") if target else ""
    anchor = anchor_quote(
        quote,
        target_text,
        document_text=str((target or {}).get("document_text") or ""),
        passage_start=int((target or {}).get("start_offset") or 0) if target else 0,
        request_passages=passages,
        exclude_passage_id=claimed_passage,
    )
    flags: list[str] = []
    passage_id = claimed_passage
    document_id = claimed_doc
    located = str(anchor.located_passage_id or "")
    if anchor.status in {"exact", "normalized_match", "fuzzy_match"} and anchor.source == "passage" and claimed_passage:
        passage_id = claimed_passage
        document_id = claimed_doc or str((target or {}).get("doc_id") or "")
    elif anchor.status in {"exact", "normalized_match", "fuzzy_match", "possibly_modified"} and (
        anchor.source in {"request", "neighbor", "document"} or located
    ):
        if located:
            passage_id = located
            document_id = str(anchor.located_doc_id or "")
            holder = by_id.get(located) or {}
            if not document_id:
                document_id = str(holder.get("doc_id") or "")
        if claimed_passage and passage_id != claimed_passage:
            flags.append("evidence_outside_passage")
        elif not claimed_passage and passage_id:
            flags.append("attributed_passage")
        elif anchor.source in {"neighbor", "document"} and claimed_passage:
            flags.append("evidence_outside_passage")
    elif anchor.status == "possibly_modified":
        flags.append("possibly_modified")
    quote_found = anchor.status in {"exact", "normalized_match", "fuzzy_match"}
    if not quote or not passage_id or not document_id or anchor.status == "unanchored":
        if "unsupported" not in flags:
            flags.append("unsupported")
    supported = "unsupported" not in flags and quote_found
    if kind == "dimension" and not str(note.get("pole") or "").strip():
        flags.append("dimension_pole_missing")
    quote_status = anchor.status if quote else "unanchored"
    stored_quote = anchor.quote if anchor.status not in {"unanchored"} and anchor.quote else quote
    category_id = str(note.get("category_id") or "")
    condition = evidence_condition_for(quote_status=quote_status, flags=flags, supported=supported)
    fields = proposal_record_fields(evidence_condition=condition)
    revision = _category_revision_for(category_id, codebook)
    return {
        "entry_id": entry_id,
        "kind": kind,
        "category_id": category_id,
        "category_revision_id": revision,
        "evidence_revision_id": evidence_revision_id(
            passage_id=passage_id,
            document_id=document_id,
            quote=stored_quote,
            quote_status=quote_status,
        ),
        "name": str(note.get("name") or ""),
        "pole": str(note.get("pole") or ""),
        "text": str(note.get("text") or ""),
        "document_id": document_id,
        "passage_id": passage_id,
        "cited_document_id": claimed_doc,
        "cited_passage_id": claimed_passage,
        "coded_passage_id": coded_passage_id,
        "quote": stored_quote,
        "model_quote": quote,
        "quote_status": quote_status,
        "flags": flags,
        "supported": supported,
        "needs_review": False,
        **fields,
    }


def _category_revision_for(
    category_id: str,
    codebook: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
) -> str:
    if not codebook or not category_id:
        return ""
    entries = codebook.values() if isinstance(codebook, Mapping) else codebook
    for entry in entries:
        if str(entry.get("code_id") or "") == category_id:
            return str(entry.get("category_revision_id") or category_revision_id(entry))
    return ""


def _hash_rows(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def corpus_hash(passages: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        {
            "passage_id": item.get("passage_id"),
            "doc_id": item.get("doc_id"),
            "text": item.get("text"),
            "start": item.get("start_offset"),
            "end": item.get("end_offset"),
        }
        for item in passages
    ]
    rows.sort(key=lambda item: (str(item["doc_id"]), str(item["passage_id"])))
    return _hash_rows(rows)


def source_hash(
    passages: Sequence[Mapping[str, Any]],
    codebook: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> str:
    return _hash_rows(
        {
            "passages": [
                {"passage_id": item.get("passage_id"), "text": item.get("text")} for item in passages
            ],
            "codes": [
                {
                    "code_id": item.get("code_id"),
                    "definition": item.get("definition"),
                    "revision": item.get("revision"),
                }
                for item in codebook
            ],
            "segments": [
                {
                    "segment_id": item.get("segment_id"),
                    "code_id": item.get("code_id"),
                    "passage_id": item.get("passage_id"),
                    "quote": item.get("quote"),
                }
                for item in segments
            ],
        }
    )


def _context_block(source: OpenCodeResult, context: Any) -> dict[str, Any]:
    stored = dict((source.metadata or {}).get("context") or {})
    if context is None:
        block = {key: stored.get(key) for key in _CONTEXT_KEYS}
        if block.get("sensitizing_concepts") is None:
            block["sensitizing_concepts"] = []
        return block
    parsed = parse_context(context)
    return {
        "research_question": parsed.research_question,
        "population": parsed.population,
        "setting": parsed.setting,
        "sensitizing_concepts": list(parsed.sensitizing_concepts),
        "notes": parsed.notes,
    }


def focused_manifest(
    source: OpenCodeResult,
    *,
    model: BaseChatModel | str,
    context: Any,
    procedure: Mapping[str, Any],
    performance: Mapping[str, Any],
) -> dict[str, Any]:
    passages = list(source.passages or [])
    codebook = list(source.codebook or [])
    segments = list(source.segments or [])
    return {
        "corpus_hash": corpus_hash(passages),
        "source_hash": source_hash(passages, codebook, segments),
        "context": _context_block(source, context),
        "model": {
            "model": describe_model(model),
            "model_settings": chat_model_settings(model),
        },
        "procedure": {key: procedure.get(key) for key in _PROCEDURE_KEYS},
        "performance": {key: performance.get(key) for key in _PERFORMANCE_KEYS},
        "policy_version": POLICY_VERSION,
    }


def _manifest_diff(saved: Any, current: Any) -> dict[str, dict[str, Any]]:
    from automated_grounded_theory.api import manifest_diff

    return manifest_diff(saved, current)


def _check_resume_manifest(saved: Mapping[str, Any], current: Mapping[str, Any], *, allow_changes: bool) -> list[dict]:
    from automated_grounded_theory.api import CheckpointMigrationRequired, CheckpointMismatch

    differences = _manifest_diff(saved, current)
    if not differences:
        return []
    structural = {path: change for path, change in differences.items() if path in _STRUCTURAL_PATHS}
    if structural:
        raise CheckpointMigrationRequired(structural)
    if not allow_changes:
        raise CheckpointMismatch(differences)
    return [{"event": "continuation", "differences": differences}]


def _decision_to_coding(decision: Mapping[str, Any], book: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    label = str(decision.get("code") or "")
    sent_id = str(decision.get("code_id") or "")
    by_id = {str(entry.get("code_id") or ""): key for key, entry in book.items()}
    by_label = {code_key(str(entry.get("code") or "")): key for key, entry in book.items()}
    known_key = by_id.get(sent_id) or by_label.get(code_key(label))
    action = str(decision.get("action") or "add")
    if known_key and action == "revise":
        mapped = "refine"
    elif known_key:
        mapped = "reuse"
    else:
        mapped = "new"
    if known_key:
        label = str(book[known_key].get("code") or label)
        sent_id = str(book[known_key].get("code_id") or "")
    else:
        sent_id = ""
    return {
        "action": mapped,
        "code": label,
        "code_id": sent_id if mapped != "new" else "",
        "definition": str(decision.get("definition") or ""),
        "quote": str(decision.get("quote") or ""),
        "rationale": str(decision.get("rationale") or ""),
        "confidence": decision.get("confidence", 1.0),
        "in_vivo": False,
    }


def _needs_label_retry(interpreted: Mapping[str, Any]) -> bool:
    return bool(interpreted.get("label_retries"))


def _span_pairs(item: Mapping[str, Any]) -> tuple[tuple[int, int], ...] | None:
    spans: list[tuple[int, int]] = []
    for span in item.get("quote_spans") or []:
        if isinstance(span, Mapping) and span.get("doc_start") is not None and span.get("doc_end") is not None:
            spans.append((int(span["doc_start"]), int(span["doc_end"])))
    if not spans and item.get("doc_start") is not None and item.get("doc_end") is not None:
        spans.append((int(item["doc_start"]), int(item["doc_end"])))
    if not spans:
        return None
    return tuple(spans)


def _same_quoted_span(existing: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    """True when the reread repeats the stored quote. A different span is not a match."""
    old, new = _span_pairs(existing), _span_pairs(item)
    if old is not None and new is not None:
        return old == new
    old_quote = " ".join(str(existing.get("quote") or "").split()).casefold()
    new_quote = " ".join(str(item.get("quote") or item.get("model_quote") or "").split()).casefold()
    if not new_quote:
        return True
    return bool(old_quote) and old_quote == new_quote


def _code_id_for_item(item: Mapping[str, Any], book: Mapping[str, Mapping[str, Any]]) -> str:
    key = str(item.get("key") or "")
    entry = book.get(key) or {}
    return str(entry.get("code_id") or item.get("code_id") or "")


def _assignments_for(segments: Sequence[Mapping[str, Any]], passage_id: str, code_id: str) -> list[Mapping[str, Any]]:
    found = []
    for segment in segments:
        if str(segment.get("code_id") or "") != code_id:
            continue
        homes = {str(segment.get("passage_id") or ""), str(segment.get("coded_passage_id") or "")}
        if passage_id in homes:
            found.append(segment)
    return found


def _confirm_assignment(segment: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    """Record a focused confirmation. The stored quote and span stay as they were."""
    row = dict(segment)
    quote = row.get("quote")
    start = row.get("doc_start")
    end = row.get("doc_end")
    spans = list(row.get("quote_spans") or [])
    provenance = dict(row.get("provenance") or {})
    focused = dict(provenance.get("focused") or {})
    focused.update(
        {
            "confirmed": True,
            "stage": "focused",
            "confidence": item.get("confidence"),
            "rationale": item.get("rationale") or "",
            "model_quote": item.get("model_quote") or item.get("quote") or "",
            "quote_status": item.get("quote_status") or "",
        }
    )
    provenance["focused"] = focused
    notes = [dict(note) for note in (row.get("confirmations") or []) if isinstance(note, Mapping)]
    notes.append(
        {
            "stage": "focused",
            "confidence": item.get("confidence"),
            "rationale": item.get("rationale") or "",
            "model_quote": item.get("model_quote") or item.get("quote") or "",
            "quote_status": item.get("quote_status") or "",
        }
    )
    row["provenance"] = provenance
    row["confirmations"] = notes
    row["focused_confirmation"] = True
    row["quote"] = quote
    row["doc_start"] = start
    row["doc_end"] = end
    row["quote_spans"] = spans
    return row


def apply_passage_recode(
    book: dict[str, dict],
    segments: list[dict],
    memos: list[dict],
    counters: dict,
    highlight_ids: dict,
    passage: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    *,
    passages: Sequence[Mapping[str, Any]],
    raised_code_ids: Sequence[str],
    require_gerund: bool = False,
    max_codes: int = 6,
    long_passage_chars: int = 2000,
    review_below: float = 0.5,
) -> dict[str, Any]:
    """Validate one reread and append assignments. Initial codes stay.

    An initial raised-code assignment that this reread does not repeat is kept
    and flagged ``focused_disagreement``. The same code and the same span is a
    confirmation on that assignment, with focused provenance. A different span
    is a separate proposal and does not rewrite the stored quote. Quotes follow
    the open-coding anchor.
    """
    coding = {
        "passage_id": str(passage.get("passage_id") or ""),
        "codes": [_decision_to_coding(item, book) for item in decisions],
        "memo": "",
    }
    interpreted = interpret_coding(
        coding,
        passage,
        book,
        max_codes=max_codes,
        require_gerund=require_gerund,
        long_passage_chars=long_passage_chars,
        review_below=review_below,
        passages=passages,
        request_passages=passages,
    )
    coded_id = str(passage.get("passage_id") or "")
    confirm_ids: dict[str, Mapping[str, Any]] = {}
    span_proposal_ids: set[str] = set()
    to_commit = []
    for item in interpreted.get("accepted") or []:
        code_id = _code_id_for_item(item, book)
        existing = _assignments_for(segments, coded_id, code_id) if code_id else []
        if existing and _same_quoted_span(existing[0], item):
            confirm_ids[str(existing[0].get("segment_id") or "")] = item
            continue
        if existing:
            span_proposal_ids.add(code_id)
        to_commit.append(item)
    confirmed_code_ids = set()
    for index, segment in enumerate(segments):
        sid = str(segment.get("segment_id") or "")
        if sid not in confirm_ids:
            continue
        segments[index] = _confirm_assignment(segment, confirm_ids[sid])
        confirmed_code_ids.add(str(segment.get("code_id") or ""))
    interpreted = {**interpreted, "accepted": to_commit}
    before_ids = {str(item.get("segment_id") or "") for item in segments}
    committed = commit_passage(
        book,
        segments,
        memos,
        highlight_ids,
        counters,
        passage,
        interpreted,
    )
    new_segments = []
    for segment in committed["segments"]:
        if str(segment.get("segment_id") or "") in before_ids:
            continue
        row = dict(segment)
        row["stage"] = "focused"
        reasons = [str(item) for item in (row.get("review_reasons") or [])]
        if "focused_recode" not in reasons:
            reasons.append("focused_recode")
        if str(row.get("code_id") or "") in span_proposal_ids:
            if "focused_span_proposal" not in reasons:
                reasons.append("focused_span_proposal")
            row["focused_span_proposal"] = True
            row["needs_review"] = True
        row["review_reasons"] = reasons
        new_segments.append(row)
    replaced = {str(item.get("segment_id") or ""): item for item in new_segments}
    merged = []
    for segment in committed["segments"]:
        sid = str(segment.get("segment_id") or "")
        merged.append(replaced.get(sid, segment))
    accepted_ids = set(confirmed_code_ids)
    for item in interpreted.get("accepted") or []:
        key = str(item.get("key") or "")
        entry = committed["codebook"].get(key) or {}
        if entry.get("code_id"):
            accepted_ids.add(str(entry.get("code_id")))
        if item.get("code_id"):
            accepted_ids.add(str(item.get("code_id")))
    raised = set(str(item) for item in raised_code_ids)
    disagreed = []
    flagged = []
    for segment in merged:
        sid = str(segment.get("segment_id") or "")
        if sid not in before_ids:
            flagged.append(segment)
            continue
        home = str(segment.get("coded_passage_id") or segment.get("passage_id") or "")
        if home != coded_id:
            flagged.append(segment)
            continue
        code_id = str(segment.get("code_id") or "")
        if code_id in raised and code_id not in accepted_ids:
            row = dict(segment)
            reasons = [str(item) for item in (row.get("review_reasons") or [])]
            if "focused_disagreement" not in reasons:
                reasons.append("focused_disagreement")
            row["review_reasons"] = reasons
            row["needs_review"] = True
            row["focused_disagreement"] = True
            flagged.append(row)
            disagreed.append(row)
        else:
            flagged.append(segment)
    return {
        "codebook": committed["codebook"],
        "segments": flagged,
        "memos": committed["memos"],
        "counters": committed["counters"],
        "highlight_ids": committed["highlight_ids"],
        "interpreted": interpreted,
        "disagreements": disagreed,
    }


def _recode_messages(
    passages: Sequence[Mapping[str, Any]],
    raised: Sequence[Mapping[str, Any]],
    *,
    feedback: str = "",
) -> list:
    payload = {
        "raised_codes": [
            {
                "code_id": entry.get("code_id"),
                "code": entry.get("code"),
                "definition": entry.get("definition"),
            }
            for entry in raised
        ],
        "passages": [
            {
                "passage_id": item.get("passage_id"),
                "doc_id": item.get("doc_id"),
                "text": item.get("text"),
            }
            for item in passages
        ],
        "feedback": feedback,
    }
    return [
        SystemMessage(content=FOCUSED_SYSTEM),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]


def _allowance_block(state: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    """Return a stop reason when the next prompt must not be sent.

    Calls are checked first, then tokens. Zero on either skips the model.
    """
    meta = state.get("metadata") or {}
    performance = meta.get("performance") or {}
    procedure = meta.get("procedure") or {}
    max_calls = int(procedure.get("max_focused_review_calls") or 0)
    max_tokens = int(procedure.get("max_focused_review_tokens") or 0)
    calls = int(state.get("focused_calls_used") or 0)
    tokens = int(state.get("focused_tokens_used") or 0)
    if calls >= max_calls:
        return "call_allowance", []
    queue = list(state.get("recode_queue") or [])
    cursor = int(state.get("recode_cursor") or 0)
    remaining = queue[cursor:]
    if not remaining:
        return None
    batch_size = max(1, int(performance.get("batch_size") or 1))
    batch_ids = _queue_passage_ids(remaining[:batch_size])
    by_id = {str(item.get("passage_id") or ""): item for item in (state.get("passages") or [])}
    batch = [by_id[pid] for pid in batch_ids if pid in by_id]
    raised_ids = set(state.get("raised_code_ids") or [])
    raised = [entry for entry in (state.get("codebook") or {}).values() if str(entry.get("code_id") or "") in raised_ids]
    messages = _recode_messages(batch, raised)
    estimate = request_input_tokens(messages, FocusedRecodeBatch, resolve_tokenizer(None))
    if tokens + estimate > max_tokens:
        return "token_allowance", _queue_passage_ids(remaining)
    return None


def _status_fields(state: Mapping[str, Any]) -> dict[str, Any]:
    procedure = (state.get("metadata") or {}).get("procedure") or {}
    stop = str(state.get("recode_stop") or "")
    not_reread = [str(item) for item in (state.get("focused_not_reread") or [])]
    if stop in {"call_allowance", "token_allowance", "no_progress"}:
        status = "incomplete"
        complete = False
    elif stop == "complete":
        status = "complete"
        complete = True
    else:
        status = "incomplete" if not_reread else "complete"
        complete = status == "complete"
        stop = stop or ("complete" if complete else "incomplete")
    return {
        "stage": "focused",
        "status": status,
        "complete": complete,
        "stop": stop,
        "focused_calls_used": int(state.get("focused_calls_used") or 0),
        "focused_tokens_used": int(state.get("focused_tokens_used") or 0),
        "max_focused_review_calls": int(procedure.get("max_focused_review_calls") or 0),
        "max_focused_review_tokens": int(procedure.get("max_focused_review_tokens") or 0),
        "focused_not_reread": not_reread,
        "focused_reread": [str(item) for item in (state.get("focused_reread") or [])],
    }


def _shares_from_procedure(procedure: Mapping[str, Any]) -> dict[str, float]:
    return normalize_shares(
        {
            "develop": procedure.get("focused_share_develop", DEFAULT_SHARES["develop"]),
            "compete": procedure.get("focused_share_compete", DEFAULT_SHARES["compete"]),
            "outside": procedure.get("focused_share_outside", DEFAULT_SHARES["outside"]),
        }
    )


def _queue_passage_ids(queue: Sequence[Any]) -> list[str]:
    ids = []
    for item in queue:
        if isinstance(item, Mapping):
            ids.append(str(item.get("passage_id") or ""))
        else:
            ids.append(str(item))
    return ids


def _category_development(state: Mapping[str, Any]) -> dict[str, Any]:
    properties = list(state.get("properties") or [])
    dimensions = list(state.get("dimensions") or [])
    records = properties + dimensions
    return {
        "state_version": STATE_VERSION,
        "stage": "focused",
        "relations": None,
        "relations_required": False,
        "properties": properties,
        "dimensions": dimensions,
        "revision_cycles": list(state.get("revision_cycles") or []),
        "views": view_notes(),
        "citation_anchoring": citation_anchoring_report(records),
        "claim_support": claim_support_report(records),
        "purpose_coverage": (state.get("metadata") or {}).get("purpose_coverage")
        or state.get("purpose_coverage")
        or {},
    }


def _incomplete_warning(fields: Mapping[str, Any]) -> str:
    ids = list(fields.get("focused_not_reread") or [])
    shown = ", ".join(ids[:30])
    if len(ids) > 30:
        shown = f"{shown}, and {len(ids) - 30} more"
    stop = str(fields.get("stop") or "")
    reasons = {
        "call_allowance": (
            f"the focused call allowance is {fields.get('max_focused_review_calls')} "
            f"and {fields.get('focused_calls_used')} calls were used"
        ),
        "token_allowance": (
            f"the focused token allowance is {fields.get('max_focused_review_tokens')} "
            f"and {fields.get('focused_tokens_used')} estimated tokens were used"
        ),
        "no_progress": "a focused reread returned no passage decisions",
    }
    why = reasons.get(stop, stop or "the reread stopped")
    return (
        "Focused coding is incomplete because "
        + why
        + ". Passages not reread: "
        + (shown or "(none)")
        + ". Initial assignments on those passages are unchanged."
    )


def _load_node(state: FocusedState) -> dict:
    book = {str(key): dict(value) for key, value in (state.get("codebook") or {}).items()}
    for key, entry in book.items():
        entry["category_revision_id"] = str(entry.get("category_revision_id") or category_revision_id(entry))
        book[key] = entry
    restored = restore_id_counters(
        state.get("segments") or [],
        state.get("memos") or [],
        highlight_ids=state.get("highlight_ids") or {},
    )
    counters = dict(state.get("counters") or {})
    for name, value in restored.items():
        counters[name] = max(int(counters.get(name) or 0), int(value or 0))
    return {"warnings": list(state.get("warnings") or []), "codebook": book, "counters": counters}


def _select_node(state: FocusedState) -> dict:
    meta = state.get("metadata") or {}
    procedure = meta.get("procedure") or {}
    weights = procedure
    minimum = procedure.get("focused_min_score")
    if minimum == "median":
        minimum = None
    selection = select_focused_codes(
        state.get("codebook") or {},
        passages=state.get("passages") or [],
        segments=state.get("segments") or [],
        memos=state.get("memos") or [],
        sweep_log=state.get("sweep_log") or [],
        pins=meta.get("pins") or [],
        weight_frequency=float(weights.get("focused_weight_frequency", DEFAULT_FOCUSED_WEIGHT_FREQUENCY)),
        weight_documents=float(weights.get("focused_weight_documents", DEFAULT_FOCUSED_WEIGHT_DOCUMENTS)),
        weight_analytic=float(weights.get("focused_weight_analytic", DEFAULT_FOCUSED_WEIGHT_ANALYTIC)),
        min_score=None if minimum is None else float(minimum),
        analytic_features=weights.get("focused_analytic_features"),
        selection_rule=str(procedure.get("focused_selection_rule") or DEFAULT_FOCUSED_SELECTION_RULE),
        selection_k=procedure.get("focused_selection_k"),
        selection_quantile=float(
            procedure.get("focused_selection_quantile")
            if procedure.get("focused_selection_quantile") is not None
            else DEFAULT_FOCUSED_SELECTION_QUANTILE
        ),
        raise_all=bool(procedure.get("focused_raise_all")),
        warn_fraction=float(
            procedure.get("focused_raise_warn_fraction")
            if procedure.get("focused_raise_warn_fraction") is not None
            else DEFAULT_FOCUSED_RAISE_WARN_FRACTION
        ),
    )
    known = [str(entry.get("code_id") or "") for entry in (state.get("codebook") or {}).values()]
    warnings = list(state.get("warnings") or [])
    for note in selection.get("warnings") or []:
        if note not in warnings:
            warnings.append(note)
    metadata = dict(meta)
    metadata["selection_summary"] = {
        "rule": selection.get("selection_rule"),
        "k": selection.get("selection_k"),
        "quantile": selection.get("selection_quantile"),
        "tie_policy": selection.get("tie_policy"),
        "raise_all": selection.get("raise_all"),
        "raised": len(selection.get("raised_code_ids") or []),
        "total": len(selection.get("rows") or []),
        "raised_fraction": selection.get("raised_fraction"),
    }
    return {
        "selection": selection["rows"],
        "proposed_raised_code_ids": list(selection["raised_code_ids"]),
        "proposed_not_raised_code_ids": list(selection["not_raised_code_ids"]),
        "known_code_ids": known,
        "warnings": warnings,
        "metadata": metadata,
    }


def _checkpoint_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "focused_selection",
        "stage": "focused",
        "raised_code_ids": list(state.get("proposed_raised_code_ids") or []),
        "not_raised_code_ids": list(state.get("proposed_not_raised_code_ids") or []),
        "scores": list(state.get("selection") or []),
        "instructions": (
            "This selection is a proposal. Resume with {\"accept\": true} or "
            "{\"raised_code_ids\": [...]}. Codes you leave out stay in the codebook "
            "and are not raised. A rare contrast stays eligible even when its score is low. "
            "Add incidents as {\"incidents\": [{\"passage_id\": \"...\", \"purpose\": \"develop|compete|outside\"}]} "
            "or {\"incidents_csv\": \"focused_incidents.csv\"}. A researcher incident can enter any purpose. "
            "Do not send the corpus, context, model, or procedure in the resume value."
        ),
    }


def _researcher_node(state: FocusedState) -> dict:
    proposed = list(state.get("proposed_raised_code_ids") or [])
    known = list(state.get("known_code_ids") or [])
    if not state.get("interrupt_enabled", True):
        raised = proposed
        decision = {"accept": True}
        source = "proposal"
        changed = False
    else:
        decision = interrupt(_checkpoint_payload(state))
        raised, changed = apply_selection_override(known, proposed, decision)
        source = "researcher" if changed else "proposal"
    not_raised = [code_id for code_id in known if code_id not in set(raised)]
    known_passages = {str(item.get("passage_id") or "") for item in (state.get("passages") or [])}
    try:
        incidents = parse_incidents(decision, known_passages)
    except ValueError as exc:
        raise FocusedError(str(exc)) from exc
    metadata = dict(state.get("metadata") or {})
    metadata["researcher_decision"] = decision if isinstance(decision, (dict, list, str, int, float, bool)) or decision is None else str(decision)
    metadata["researcher_edit_hash"] = _hash_rows(decision)
    metadata["selection_source"] = source
    metadata["researcher_incidents"] = incidents
    return {
        "raised_code_ids": raised,
        "not_raised_code_ids": not_raised,
        "researcher_decision": decision,
        "selection_source": source,
        "metadata": metadata,
    }


def _prepare_node(state: FocusedState) -> dict:
    updates: dict[str, Any] = {}
    codebook = state.get("codebook") or {}
    if not state.get("queue_ready"):
        meta = state.get("metadata") or {}
        procedure = meta.get("procedure") or {}
        plan = plan_recode_queue(
            state.get("passages") or [],
            state.get("segments") or [],
            state.get("raised_code_ids") or [],
            list(codebook.values()),
            seed=int(procedure.get("seed") or 0),
            neighbor_limit=int(procedure.get("focused_neighbor_limit") or 0),
            outside_limit=int(procedure.get("focused_outside_limit") or 0),
            negative_case_limit=int(procedure.get("focused_negative_case_limit") or 0),
            shares=_shares_from_procedure(procedure),
            incidents=(meta.get("researcher_incidents") or []),
            claims=list(state.get("properties") or []) + list(state.get("dimensions") or []),
        )
        queue = list(plan["queue"])
        warnings = list(state.get("warnings") or [])
        for note in plan.get("warnings") or []:
            if note not in warnings:
                warnings.append(note)
        metadata = dict(meta)
        metadata["reread_shares_applied"] = plan.get("shares_applied") or {}
        metadata["reread_reallocation"] = plan.get("reallocation") or {}
        metadata["reread_purpose_counts"] = plan.get("purpose_counts") or {}
        updates["warnings"] = warnings
        updates["metadata"] = metadata
        updates["recode_queue"] = queue
        updates["recode_cursor"] = 0
        updates["queue_ready"] = True
        updates["focused_not_reread"] = []
        updates["focused_reread"] = list(state.get("focused_reread") or [])
        state = {**state, **updates}
    if state.get("recode_stop") in {"call_allowance", "token_allowance", "no_progress", "complete"}:
        updates["round_jobs"] = []
        return updates
    queue = list(state.get("recode_queue") or [])
    cursor = int(state.get("recode_cursor") or 0)
    if cursor >= len(queue):
        updates["round_jobs"] = []
        updates["recode_stop"] = "complete"
        updates["focused_not_reread"] = []
        return updates
    blocked = _allowance_block(state)
    if blocked is not None:
        reason, remaining = blocked
        if not remaining:
            remaining = _queue_passage_ids(queue[cursor:])
        updates["round_jobs"] = []
        updates["recode_stop"] = reason
        updates["focused_not_reread"] = [str(item) for item in remaining]
        return updates
    performance = (state.get("metadata") or {}).get("performance") or {}
    procedure = (state.get("metadata") or {}).get("procedure") or {}
    batch_size = max(1, int(performance.get("batch_size") or 1))
    token_budget = int(performance.get("token_budget") or DEFAULT_FOCUSED_TOKEN_BUDGET)
    by_id = {str(item.get("passage_id") or ""): item for item in (state.get("passages") or [])}
    raised_ids = set(state.get("raised_code_ids") or [])
    raised = [entry for entry in codebook.values() if str(entry.get("code_id") or "") in raised_ids]
    chosen = []
    chosen_items = []
    for item in queue[cursor:]:
        passage_id = str(item.get("passage_id") or "") if isinstance(item, Mapping) else str(item)
        passage = by_id.get(passage_id)
        if passage is None:
            continue
        trial = chosen + [passage]
        messages = _recode_messages(trial, raised)
        estimate = request_input_tokens(messages, FocusedRecodeBatch, resolve_tokenizer(None))
        if chosen and estimate > token_budget:
            break
        chosen.append(passage)
        chosen_items.append(item if isinstance(item, Mapping) else {"passage_id": passage_id, "purpose": "develop"})
        if len(chosen) >= batch_size:
            break
    messages = _recode_messages(chosen, raised)
    estimate = request_input_tokens(messages, FocusedRecodeBatch, resolve_tokenizer(None))
    calls = int(state.get("focused_calls_used") or 0)
    tokens = int(state.get("focused_tokens_used") or 0)
    max_calls = int(procedure.get("max_focused_review_calls") or 0)
    max_tokens = int(procedure.get("max_focused_review_tokens") or 0)
    allow_retry = calls + 1 < max_calls and tokens + estimate * 2 <= max_tokens
    snapshot = freeze_snapshot(
        {"codebook": codebook, "segments": state.get("segments") or []},
        version=int(state.get("snapshot_version") or 0),
    )
    job_id = f"focused-{cursor}"
    updates.update(snapshot)
    updates["round_jobs"] = [
        {
            "job_id": job_id,
            "proposal_id": f"{snapshot['snapshot_id']}:{job_id}",
            "snapshot_id": snapshot["snapshot_id"],
            "passage_ids": [str(item.get("passage_id") or "") for item in chosen],
            "passages": chosen,
            "estimate": estimate,
            "allow_label_retry": allow_retry and int(procedure.get("validation_retries") or 0) > 0,
            "purposes": {
                str(item.get("passage_id") or ""): str(item.get("purpose") or "develop") for item in chosen_items
            },
        }
    ]
    purpose_calls = {purpose: int((state.get("purpose_calls") or {}).get(purpose) or 0) for purpose in REREAD_PURPOSES}
    purpose_tokens = {purpose: int((state.get("purpose_tokens") or {}).get(purpose) or 0) for purpose in REREAD_PURPOSES}
    seen_purposes = []
    for item in chosen_items:
        purpose = str(item.get("purpose") or "develop")
        if purpose not in seen_purposes:
            seen_purposes.append(purpose)
            purpose_calls[purpose] = purpose_calls.get(purpose, 0) + 1
            purpose_tokens[purpose] = purpose_tokens.get(purpose, 0) + estimate
    updates["recode_cursor"] = cursor + len(chosen)
    updates["focused_calls_used"] = calls + 1
    updates["focused_tokens_used"] = tokens + estimate
    updates["purpose_calls"] = purpose_calls
    updates["purpose_tokens"] = purpose_tokens
    return updates


def _route_prepare(state: FocusedState):
    jobs = list(state.get("round_jobs") or [])
    if not jobs:
        return "review_recode"
    sends = []
    for job in jobs:
        sends.append(
            Send(
                "recode_worker",
                {
                    "worker_job": job,
                    "worker_passages": state.get("passages") or [],
                    "worker_codebook": state.get("codebook") or {},
                    "worker_raised": list(state.get("raised_code_ids") or []),
                },
            )
        )
    return sends


def _worker(runtime: FocusedRuntime):
    async def recode_worker(payload: Mapping[str, Any]) -> dict:
        job = dict(payload.get("worker_job") or {})
        book = payload.get("worker_codebook") or {}
        raised_ids = set(payload.get("worker_raised") or [])
        raised = [entry for entry in book.values() if str(entry.get("code_id") or "") in raised_ids]
        passages = list(job.get("passages") or [])
        messages = _recode_messages(passages, raised)
        raw_parts: list[str] = []
        (parsed, usage), sent = await ainvoke_structured(
            runtime.coder,
            FocusedRecodeBatch,
            messages,
            runtime.limiter,
            use_cache=False,
            raw_out=raw_parts,
        )
        calls = 1
        interpreted_retry = False
        if job.get("allow_label_retry"):
            probe = _probe_labels(parsed, passages, book, runtime)
            if probe:
                feedback = "LABEL RETRY\n" + validation_feedback(probe)
                retry_messages = _recode_messages(passages, raised, feedback=feedback)
                (parsed, retry_usage), sent = await ainvoke_structured(
                    runtime.coder,
                    FocusedRecodeBatch,
                    retry_messages,
                    runtime.limiter,
                    use_cache=False,
                    raw_out=raw_parts,
                )
                usage = {
                    "input_tokens": int((usage or {}).get("input_tokens") or 0)
                    + int((retry_usage or {}).get("input_tokens") or 0),
                    "output_tokens": int((usage or {}).get("output_tokens") or 0)
                    + int((retry_usage or {}).get("output_tokens") or 0),
                }
                calls = 2
                interpreted_retry = True
        proposal = {
            "proposal_id": job.get("proposal_id"),
            "snapshot_id": job.get("snapshot_id"),
            "passage_index": 0,
            "coding": parsed,
            "passage_ids": list(job.get("passage_ids") or []),
            "calls": calls,
            "usage": usage,
            "label_retry": interpreted_retry,
            "response": parsed,
            "raw_response": "\n".join(part for part in raw_parts if part),
            "messages": [
                {"role": getattr(message, "type", "unknown"), "content": getattr(message, "content", "")}
                for message in sent
            ],
        }
        return {"proposal_buffer": {"items": [proposal]}}

    return recode_worker


def _probe_labels(parsed: Mapping[str, Any], passages: Sequence[Mapping], book: Mapping, runtime: FocusedRuntime) -> list:
    retries = []
    by_id = {str(item.get("passage_id") or ""): item for item in passages}
    for row in parsed.get("passages") or []:
        passage = by_id.get(str(row.get("passage_id") or ""))
        if passage is None:
            continue
        interpreted = interpret_coding(
            {
                "passage_id": passage.get("passage_id"),
                "codes": [_decision_to_coding(item, book) for item in (row.get("codes") or [])],
                "memo": "",
            },
            passage,
            book,
            max_codes=runtime.max_codes_per_passage,
            require_gerund=runtime.require_gerund,
            long_passage_chars=runtime.long_passage_chars,
            review_below=runtime.review_below,
            passages=passages,
            request_passages=passages,
        )
        retries.extend(interpreted.get("label_retries") or [])
    return retries


def _commit_node(runtime: FocusedRuntime):
    def commit_focused(state: FocusedState) -> dict:
        snapshot_id = str(state.get("snapshot_id") or "")
        fresh, _stale, _skipped = select_committable(
            (state.get("proposal_buffer") or {}).get("items") or [],
            snapshot_id=snapshot_id,
            committed_ids=state.get("committed_ids") or [],
        )
        book = {key: dict(value) for key, value in (state.get("codebook") or {}).items()}
        segments = [dict(item) for item in (state.get("segments") or [])]
        memos = [dict(item) for item in (state.get("memos") or [])]
        counters = dict(state.get("counters") or empty_counters())
        highlights = dict(state.get("highlight_ids") or {})
        properties = list(state.get("properties") or [])
        dimensions = list(state.get("dimensions") or [])
        prompt_log = list(state.get("prompt_log") or [])
        reread = list(state.get("focused_reread") or [])
        warnings = list(state.get("warnings") or [])
        calls = int(state.get("focused_calls_used") or 0)
        tokens = int(state.get("focused_tokens_used") or 0)
        passages = list(state.get("passages") or [])
        by_id = {str(item.get("passage_id") or ""): item for item in passages}
        raised = list(state.get("raised_code_ids") or [])
        addressed = False
        extra_calls = 0
        for proposal in fresh:
            parsed = proposal.get("coding") or {}
            sent_ids = [str(item) for item in (proposal.get("passage_ids") or [])]
            returned = [dict(item) for item in (parsed.get("passages") or []) if isinstance(item, Mapping)]
            returned_ids = {str(item.get("passage_id") or "") for item in returned}
            if any(pid and pid in sent_ids for pid in returned_ids):
                addressed = True
            extra_calls += max(0, int(proposal.get("calls") or 1) - 1)
            usage = proposal.get("usage") or {}
            prompt_log.append(
                {
                    "stage": "focused",
                    "proposal_id": proposal.get("proposal_id"),
                    "passage_ids": sent_ids,
                    "usage": usage,
                    "label_retry": bool(proposal.get("label_retry")),
                    "response": proposal.get("response"),
                    "raw_response": proposal.get("raw_response") or "",
                    "messages": list(proposal.get("messages") or []),
                }
            )
            actual = int(usage.get("input_tokens") or 0)
            if actual:
                tokens += 0
            for row in returned:
                passage_id = str(row.get("passage_id") or "")
                passage = by_id.get(passage_id)
                if passage is None or passage_id not in sent_ids:
                    warnings.append(
                        f"Focused recode named passage {passage_id or '(blank)'}, which was not in this request. "
                        "The decisions were kept in the prompt log and not assigned to another passage."
                    )
                    prompt_log[-1].setdefault("unapplied", []).append(row)
                    continue
                applied = apply_passage_recode(
                    book,
                    segments,
                    memos,
                    counters,
                    highlights,
                    passage,
                    list(row.get("codes") or []),
                    passages=passages,
                    raised_code_ids=raised,
                    require_gerund=runtime.require_gerund,
                    max_codes=runtime.max_codes_per_passage,
                    long_passage_chars=runtime.long_passage_chars,
                    review_below=runtime.review_below,
                )
                book = applied["codebook"]
                segments = applied["segments"]
                memos = applied["memos"]
                counters = applied["counters"]
                highlights = applied["highlight_ids"]
                if passage_id not in reread:
                    reread.append(passage_id)
                for kind, bucket, store in (
                    ("property", row.get("properties") or [], properties),
                    ("dimension", row.get("dimensions") or [], dimensions),
                ):
                    for note in bucket:
                        if not isinstance(note, Mapping):
                            continue
                        entry_id = f"{kind[:4]}-{len(store) + 1:04d}"
                        store.append(
                            cite_note(
                                note,
                                passages,
                                kind=kind,
                                entry_id=entry_id,
                                coded_passage_id=passage_id,
                                codebook=book,
                            )
                        )
                memo_text = str(row.get("memo") or "").strip()
                if memo_text:
                    memo_id = _next(counters, "memo", "m")
                    memos.append(
                        {
                            "memo_id": memo_id,
                            "kind": "model_memo",
                            "role": "focused_recode",
                            "passage_id": passage_id,
                            "doc_id": passage.get("doc_id"),
                            "text": memo_text,
                        }
                    )
        committed = list(state.get("committed_ids") or [])
        committed.extend(str(item.get("proposal_id") or "") for item in fresh)
        stop = state.get("recode_stop")
        not_reread = list(state.get("focused_not_reread") or [])
        if fresh and not addressed:
            queue = list(state.get("recode_queue") or [])
            cursor = int(state.get("recode_cursor") or 0)
            missed = []
            for proposal in fresh:
                missed.extend(str(item) for item in (proposal.get("passage_ids") or []))
            rest = _queue_passage_ids(queue[cursor:])
            not_reread = missed + [item for item in rest if item not in missed]
            stop = "no_progress"
            for pid in missed:
                if pid in reread:
                    reread.remove(pid)
        if extra_calls:
            calls += extra_calls
        version = int(state.get("snapshot_version") or 0) + 1
        return {
            "codebook": book,
            "segments": segments,
            "memos": memos,
            "counters": counters,
            "highlight_ids": highlights,
            "properties": properties,
            "dimensions": dimensions,
            "prompt_log": prompt_log,
            "warnings": warnings,
            "focused_reread": reread,
            "focused_not_reread": not_reread,
            "focused_calls_used": calls,
            "focused_tokens_used": tokens,
            "recode_stop": stop,
            "committed_ids": committed,
            "snapshot_version": version,
            "proposal_buffer": {"__reset__": True},
            "round_jobs": [],
        }

    return commit_focused


def _route_commit(state: FocusedState) -> str:
    if state.get("recode_stop") in {"call_allowance", "token_allowance", "no_progress", "complete"}:
        return "review_recode"
    return "prepare_recode"


def _review_node(state: FocusedState) -> dict:
    fields = _status_fields(state)
    warnings = list(state.get("warnings") or [])
    if fields["status"] == "incomplete":
        note = _incomplete_warning(fields)
        if note not in warnings:
            warnings.append(note)
    metadata = dict(state.get("metadata") or {})
    procedure = metadata.get("procedure") or {}
    requested = _shares_from_procedure(procedure)
    applied = metadata.get("reread_shares_applied") or requested
    reallocation = dict(metadata.get("reread_reallocation") or {})
    coverage = purpose_coverage(
        list(state.get("recode_queue") or []),
        list(state.get("focused_reread") or []),
        state.get("purpose_calls") or {},
        state.get("purpose_tokens") or {},
        applied,
    )
    emptied = set(reallocation.get("purposes") or [])
    for purpose in REREAD_PURPOSES:
        coverage[purpose]["requested_share"] = float(requested.get(purpose, 0))
        coverage[purpose]["applied_share"] = float(applied.get(purpose, 0))
        coverage[purpose]["reallocated_away"] = purpose in emptied
    if reallocation:
        coverage["reallocation"] = reallocation
    metadata.update(fields)
    metadata["purpose_coverage"] = coverage
    metadata["saturation_judgment"] = None
    return {
        "warnings": warnings,
        "metadata": metadata,
        "recode_stop": fields["stop"],
        "focused_not_reread": fields["focused_not_reread"],
        "purpose_coverage": coverage,
    }


def _claim_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "focused_claims",
        "stage": "focused",
        "properties": list(state.get("properties") or []),
        "dimensions": list(state.get("dimensions") or []),
        "purpose_coverage": (state.get("metadata") or {}).get("purpose_coverage") or {},
        "instructions": (
            "Claims stay proposed until you set a disposition. "
            "claim_support stays undefined until you set supported, partially_supported, "
            "contradicted, or indeterminate. "
            "Resume with {\"leave_claims\": true} to leave them, {\"accept_claims\": true} to accept anchored proposals, "
            "{\"claim_dispositions\": [{\"entry_id\": \"...\", \"disposition\": \"accepted\"}]}, "
            "{\"claim_reviews\": [{\"entry_id\": \"...\", \"claim_support\": \"supported\"}]}, "
            "or {\"category_edits\": [{\"code_id\": \"...\", \"definition\": \"...\"}]}. "
            "A category edit marks dependent claims and assignments needs_review and keeps their history."
        ),
    }


def _claim_node(state: FocusedState) -> dict:
    properties = list(state.get("properties") or [])
    dimensions = list(state.get("dimensions") or [])
    if not state.get("interrupt_enabled", True):
        return {}
    decision = interrupt(_claim_payload(state))
    try:
        reviewed = apply_claim_review(properties, dimensions, decision)
    except (ValueError, KeyError) as exc:
        raise FocusedError(str(exc)) from exc
    properties = reviewed["properties"]
    dimensions = reviewed["dimensions"]
    codebook = state.get("codebook") or {}
    segments = list(state.get("segments") or [])
    edits = []
    if isinstance(decision, Mapping):
        edits = list(decision.get("category_edits") or [])
    if edits:
        edited = apply_category_edits(codebook, segments, properties, dimensions, edits)
        codebook = edited["codebook"]
        segments = edited["segments"]
        properties = edited["properties"]
        dimensions = edited["dimensions"]
    metadata = dict(state.get("metadata") or {})
    metadata["claim_decision"] = decision if isinstance(decision, (dict, list, str, int, float, bool)) or decision is None else str(decision)
    return {
        "properties": properties,
        "dimensions": dimensions,
        "codebook": codebook,
        "segments": segments,
        "metadata": metadata,
    }


def _export_node(runtime: FocusedRuntime):
    def export_focused(state: FocusedState) -> dict:
        paths = dict(state.get("export_paths") or {})
        directory = runtime.output_dir or (state.get("metadata") or {}).get("output_dir")
        if directory:
            paths = export_focused_state(state, directory)
        return {"export_paths": paths, "finished": True}

    return export_focused


def export_focused_state(state: Mapping[str, Any], directory: str | Path) -> dict[str, str]:
    place = Path(directory)
    place.mkdir(parents=True, exist_ok=True)
    development = _category_development(state)
    selection = list(state.get("selection") or [])
    raised = set(state.get("raised_code_ids") or state.get("proposed_raised_code_ids") or [])
    _write_json(place / "category_development.json", development)
    _write_json(
        place / "focused_selection.json",
        {
            "raised_code_ids": list(state.get("raised_code_ids") or state.get("proposed_raised_code_ids") or []),
            "not_raised_code_ids": list(
                state.get("not_raised_code_ids") or state.get("proposed_not_raised_code_ids") or []
            ),
            "scores": selection,
            "selection_source": (state.get("metadata") or {}).get("selection_source"),
        },
    )
    _write_csv(
        place / "focused_selection.csv",
        [
            "code_id",
            "code",
            "score",
            "frequency",
            "document_share",
            "respondent_share",
            "analytic_component",
            "pin",
            "contrast",
            "raised",
            "raise_reason",
        ],
        [
            [
                row.get("code_id"),
                row.get("code"),
                row.get("score"),
                row.get("frequency"),
                row.get("document_share"),
                row.get("respondent_share"),
                row.get("analytic_component"),
                row.get("pin"),
                row.get("contrast"),
                row.get("code_id") in raised if state.get("raised_code_ids") is not None else row.get("raised"),
                " ".join(row.get("raise_reason") or []),
            ]
            for row in selection
        ],
    )
    metadata = dict(state.get("metadata") or {})
    metadata["saturation_judgment"] = None
    records = list(development.get("properties") or []) + list(development.get("dimensions") or [])
    eligible = [row for row in records if row.get("downstream_eligible")]
    _write_json(place / "focused_run.json", {"metadata": metadata, "warnings": list(state.get("warnings") or [])})
    _write_json(
        place / "downstream_claims.json",
        {
            "uses": view_notes()["downstream_claims.json"],
            "records": eligible,
        },
    )
    _write_csv(
        place / "focused_claims.csv",
        [
            "entry_id",
            "kind",
            "category_id",
            "category_revision_id",
            "evidence_revision_id",
            "name",
            "disposition",
            "evidence_condition",
            "downstream_eligible",
            "claim_support",
            "quote_status",
            "document_id",
            "passage_id",
            "quote",
            "needs_review",
        ],
        [
            [
                row.get("entry_id"),
                row.get("kind"),
                row.get("category_id"),
                row.get("category_revision_id"),
                row.get("evidence_revision_id"),
                row.get("name"),
                row.get("disposition"),
                row.get("evidence_condition"),
                row.get("downstream_eligible"),
                row.get("claim_support"),
                row.get("quote_status"),
                row.get("document_id"),
                row.get("passage_id"),
                row.get("quote"),
                row.get("needs_review"),
            ]
            for row in records
        ],
    )
    _write_json(place / "code_id_sidecar.json", code_id_sidecar(list((state.get("codebook") or {}).values())))
    segments = list(state.get("segments") or [])
    assignments = _assignment_rows(segments)
    _write_json(place / "coded_segments.json", segments)
    _write_json(place / "assignments.json", assignments)
    _write_json(place / "prompt_log.json", list(state.get("prompt_log") or []))
    return {
        "category_development.json": str(place / "category_development.json"),
        "focused_selection.json": str(place / "focused_selection.json"),
        "focused_selection.csv": str(place / "focused_selection.csv"),
        "focused_claims.csv": str(place / "focused_claims.csv"),
        "downstream_claims.json": str(place / "downstream_claims.json"),
        "code_id_sidecar.json": str(place / "code_id_sidecar.json"),
        "focused_run.json": str(place / "focused_run.json"),
        "coded_segments.json": str(place / "coded_segments.json"),
        "assignments.json": str(place / "assignments.json"),
        "prompt_log.json": str(place / "prompt_log.json"),
        "uses": view_notes(),
    }


def _assignment_rows(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per segment. Confirmations stay on the original span."""
    rows = []
    for segment in segments:
        rows.append(
            {
                "segment_id": segment.get("segment_id"),
                "code_id": segment.get("code_id"),
                "code": segment.get("code"),
                "passage_id": segment.get("passage_id"),
                "coded_passage_id": segment.get("coded_passage_id"),
                "doc_id": segment.get("doc_id"),
                "quote": segment.get("quote"),
                "doc_start": segment.get("doc_start"),
                "doc_end": segment.get("doc_end"),
                "focused_confirmation": bool(segment.get("focused_confirmation")),
                "focused_span_proposal": bool(segment.get("focused_span_proposal")),
                "needs_review": bool(segment.get("needs_review")),
                "review_reasons": list(segment.get("review_reasons") or []),
                "provenance": segment.get("provenance") or {},
            }
        )
    return rows


class _WorkerInput(TypedDict, total=False):
    worker_job: dict
    worker_passages: list
    worker_codebook: dict
    worker_raised: list


def build_focused_graph(runtime: FocusedRuntime | None = None):
    """Uncompiled focused-coding graph.

    ``load_open_coding`` → ``select_candidates`` → ``researcher_checkpoint`` →
    ``prepare_recode`` → ``recode_worker`` → ``commit_focused`` →
    ``review_recode`` → ``claim_checkpoint`` → ``export_focused``.

    ``researcher_checkpoint`` and ``claim_checkpoint`` call
    :func:`langgraph.types.interrupt` when interrupts are enabled.
    """
    runtime = runtime or FocusedRuntime(coder=_IdleModel())
    builder = StateGraph(FocusedState)
    builder.add_node("load_open_coding", _load_node)
    builder.add_node("select_candidates", _select_node)
    builder.add_node("researcher_checkpoint", _researcher_node)
    builder.add_node("prepare_recode", _prepare_node)
    builder.add_node("recode_worker", _worker(runtime), input_schema=_WorkerInput)
    builder.add_node("commit_focused", _commit_node(runtime))
    builder.add_node("review_recode", _review_node)
    builder.add_node("claim_checkpoint", _claim_node)
    builder.add_node("export_focused", _export_node(runtime))
    builder.add_edge(START, "load_open_coding")
    builder.add_edge("load_open_coding", "select_candidates")
    builder.add_edge("select_candidates", "researcher_checkpoint")
    builder.add_edge("researcher_checkpoint", "prepare_recode")
    builder.add_conditional_edges(
        "prepare_recode",
        _route_prepare,
        ["recode_worker", "review_recode"],
    )
    builder.add_edge("recode_worker", "commit_focused")
    builder.add_conditional_edges(
        "commit_focused",
        _route_commit,
        {"prepare_recode": "prepare_recode", "review_recode": "review_recode"},
    )
    builder.add_edge("review_recode", "claim_checkpoint")
    builder.add_edge("claim_checkpoint", "export_focused")
    builder.add_edge("export_focused", END)
    return builder


def _initial_state(source: OpenCodeResult, metadata: dict, *, interrupt_enabled: bool) -> FocusedState:
    book = _book_from_list(source.codebook)
    return {
        "passages": [dict(item) for item in source.passages],
        "segments": [dict(item) for item in source.segments],
        "memos": [dict(item) for item in source.memos],
        "codebook": book,
        "sweep_log": [dict(item) if isinstance(item, Mapping) else item for item in source.sweep_log],
        "documents": list((source.metadata or {}).get("documents") or []),
        "metadata": metadata,
        "warnings": [],
        "selection": [],
        "proposed_raised_code_ids": [],
        "proposed_not_raised_code_ids": [],
        "raised_code_ids": [],
        "not_raised_code_ids": [],
        "known_code_ids": [],
        "interrupt_enabled": interrupt_enabled,
        "recode_queue": [],
        "recode_cursor": 0,
        "queue_ready": False,
        "round_jobs": [],
        "focused_calls_used": 0,
        "focused_tokens_used": 0,
        "purpose_calls": {purpose: 0 for purpose in REREAD_PURPOSES},
        "purpose_tokens": {purpose: 0 for purpose in REREAD_PURPOSES},
        "purpose_coverage": {},
        "revision_cycles": [],
        "focused_not_reread": [],
        "focused_reread": [],
        "recode_stop": "",
        "properties": [],
        "dimensions": [],
        "prompt_log": [],
        "proposal_buffer": {"items": [], "duplicates": 0, "conflicts": []},
        "committed_ids": [],
        "counters": restore_id_counters(source.segments, source.memos),
        "highlight_ids": {},
        "snapshot_id": "s00000",
        "snapshot_version": 0,
        "export_paths": {},
        "finished": False,
    }


def _result_from_state(
    state: Mapping[str, Any],
    *,
    interrupted: bool,
    payload: dict | None,
    thread_id: str,
    resumable: bool,
    checkpointer: Any,
) -> FocusedResult:
    metadata = dict(state.get("metadata") or {})
    if not interrupted:
        metadata.update(_status_fields(state))
    else:
        metadata["status"] = "interrupted"
        metadata["complete"] = False
        metadata["stop"] = "researcher_checkpoint"
    metadata["saturation_judgment"] = None
    metadata["policy_version"] = POLICY_VERSION
    metadata["package_version"] = __version__
    development = _category_development(state)
    return FocusedResult(
        raised_code_ids=list(state.get("raised_code_ids") or state.get("proposed_raised_code_ids") or []),
        not_raised_code_ids=list(state.get("not_raised_code_ids") or state.get("proposed_not_raised_code_ids") or []),
        selection=list(state.get("selection") or []),
        codebook=codebook_list(state.get("codebook") or {}),
        segments=list(state.get("segments") or []),
        memos=list(state.get("memos") or []),
        properties=list(development["properties"]),
        dimensions=list(development["dimensions"]),
        category_development=development,
        focused_not_reread=[str(item) for item in (state.get("focused_not_reread") or [])],
        focused_reread=[str(item) for item in (state.get("focused_reread") or [])],
        purpose_coverage=dict((state.get("metadata") or {}).get("purpose_coverage") or state.get("purpose_coverage") or {}),
        metadata=metadata,
        warnings=list(state.get("warnings") or []),
        prompt_log=list(state.get("prompt_log") or []),
        export_paths=dict(state.get("export_paths") or {}),
        interrupted=interrupted,
        interrupt_payload=payload,
        thread_id=thread_id,
        resumable=resumable,
        saturation_report={
            "judgment": None,
            "note": (
                "Focused coding does not decide saturation. "
                "judgment stays null. This stage does not write it."
            ),
        },
        checkpointer=checkpointer,
    )


def _interrupt_payloads(snapshot: Any) -> list[Any]:
    payloads = []
    for item in tuple(getattr(snapshot, "interrupts", ()) or ()):
        payloads.append(getattr(item, "value", item))
    return payloads


def _procedure(
    *,
    seed: int,
    coding_lens: str,
    weight_frequency: float,
    weight_documents: float,
    weight_analytic: float,
    min_score: float | None,
    max_calls: int,
    max_tokens: int,
    neighbor_limit: int,
    outside_limit: int,
    negative_case_limit: int,
    shares: Mapping[str, float],
    analytic_features: Sequence[str] | None,
    validation_retries: int,
    selection_rule: str,
    selection_k: int | None,
    selection_quantile: float,
    raise_all: bool,
    warn_fraction: float,
) -> dict[str, Any]:
    if coding_lens != "charmaz":
        raise ValueError(
            f"coding_lens {coding_lens!r} is not part of focused coding. "
            "PR1 accepts only 'charmaz'. Axial lenses are not implemented."
        )
    if max_calls < 0 or max_tokens < 0:
        raise ValueError("max_focused_review_calls and max_focused_review_tokens must be zero or more.")
    if neighbor_limit < 0 or outside_limit < 0 or negative_case_limit < 0:
        raise ValueError("Focused passage limits must be zero or more.")
    if validation_retries < 0:
        raise ValueError("validation_retries must be zero or more.")
    rule = str(selection_rule or DEFAULT_FOCUSED_SELECTION_RULE)
    if rule not in SELECTION_RULES:
        raise ValueError(f"focused_selection_rule must be one of {SELECTION_RULES}.")
    if selection_k is not None and int(selection_k) < 1:
        raise ValueError("focused_selection_k must be at least 1.")
    if not 0 < float(selection_quantile) < 1:
        raise ValueError("focused_selection_quantile must be between 0 and 1.")
    if not 0 < float(warn_fraction) <= 1:
        raise ValueError("focused_raise_warn_fraction must be above 0 and at most 1.")
    share_values = normalize_shares(shares)
    features = list(normalize_analytic_features(analytic_features))
    return {
        "stage": "focused",
        "coding_lens": coding_lens,
        "seed": int(seed),
        "focused_weight_frequency": float(weight_frequency),
        "focused_weight_documents": float(weight_documents),
        "focused_weight_analytic": float(weight_analytic),
        "focused_min_score": None if min_score is None else float(min_score),
        "focused_selection_rule": rule,
        "focused_selection_k": None if selection_k is None else int(selection_k),
        "focused_selection_quantile": float(selection_quantile),
        "focused_raise_all": bool(raise_all),
        "focused_raise_warn_fraction": float(warn_fraction),
        "max_focused_review_calls": int(max_calls),
        "max_focused_review_tokens": int(max_tokens),
        "focused_neighbor_limit": int(neighbor_limit),
        "focused_outside_limit": int(outside_limit),
        "focused_negative_case_limit": int(negative_case_limit),
        "focused_share_develop": share_values["develop"],
        "focused_share_compete": share_values["compete"],
        "focused_share_outside": share_values["outside"],
        "focused_analytic_features": features,
        "validation_retries": int(validation_retries),
    }


def _performance(*, batch_size: int, concurrency: int, token_budget: int) -> dict[str, Any]:
    if batch_size < 1 or concurrency < 1 or token_budget < 1:
        raise ValueError("batch_size, concurrency, and token_budget must be at least 1.")
    return {
        "batch_size": int(batch_size),
        "passages_per_request": int(batch_size),
        "concurrency": int(concurrency),
        "token_budget": int(token_budget),
    }


async def afocused_code(
    open_coding_result: OpenCodeResult | Mapping[str, Any] | str | Path,
    *,
    model: BaseChatModel | str,
    seed: int = 0,
    coding_lens: str = "charmaz",
    pins: Sequence[str] | None = None,
    context: Any = None,
    focused_weight_frequency: float = DEFAULT_FOCUSED_WEIGHT_FREQUENCY,
    focused_weight_documents: float = DEFAULT_FOCUSED_WEIGHT_DOCUMENTS,
    focused_weight_analytic: float = DEFAULT_FOCUSED_WEIGHT_ANALYTIC,
    focused_min_score: float | None = None,
    focused_selection_rule: str = DEFAULT_FOCUSED_SELECTION_RULE,
    focused_selection_k: int | None = None,
    focused_selection_quantile: float = DEFAULT_FOCUSED_SELECTION_QUANTILE,
    focused_raise_all: bool = False,
    focused_raise_warn_fraction: float = DEFAULT_FOCUSED_RAISE_WARN_FRACTION,
    focused_neighbor_limit: int = DEFAULT_FOCUSED_NEIGHBOR_LIMIT,
    focused_outside_limit: int = DEFAULT_FOCUSED_OUTSIDE_LIMIT,
    focused_negative_case_limit: int = DEFAULT_FOCUSED_NEGATIVE_CASE_LIMIT,
    focused_share_develop: float = DEFAULT_SHARES["develop"],
    focused_share_compete: float = DEFAULT_SHARES["compete"],
    focused_share_outside: float = DEFAULT_SHARES["outside"],
    focused_analytic_features: Sequence[str] | None = None,
    max_focused_review_calls: int = DEFAULT_MAX_FOCUSED_REVIEW_CALLS,
    max_focused_review_tokens: int = DEFAULT_MAX_FOCUSED_REVIEW_TOKENS,
    batch_size: int = DEFAULT_FOCUSED_BATCH_SIZE,
    concurrency: int = DEFAULT_FOCUSED_CONCURRENCY,
    token_budget: int = DEFAULT_FOCUSED_TOKEN_BUDGET,
    validation_retries: int = 1,
    interrupt: bool = True,
    resume: Any = None,
    resuming: bool = False,
    checkpointer: Any = None,
    checkpoint_path: str | Path | None = None,
    thread_id: str | None = None,
    output_dir: str | Path | None = None,
    allow_changes: bool = False,
    continuation: bool = False,
    recursion_limit: int | None = None,
) -> FocusedResult:
    """Run focused coding over a finished open-coding result.

    ``interrupt`` defaults to true. The graph pauses inside
    ``researcher_checkpoint`` with :func:`langgraph.types.interrupt`. Resume by
    calling this function again with ``resuming=True`` and ``resume`` set to the
    value for :class:`~langgraph.types.Command`. The same checkpointer and
    ``thread_id`` are required. Corpus, context, model settings, and the
    analytic procedure are checked against the manifest stored at the start.
    ``batch_size`` (default 1), ``concurrency``, and ``token_budget`` are
    performance settings and are stored separately. Changing them, or the
    context or model, needs ``allow_changes`` or ``continuation``. Changing the
    corpus or ``seed`` does not continue.

    Zero ``max_focused_review_calls`` or ``max_focused_review_tokens`` still
    selects codes and, when ``interrupt`` is true, still pauses. The model loop
    is skipped and the status is ``incomplete``.
    """
    source = _as_result(open_coding_result)
    procedure = _procedure(
        seed=seed,
        coding_lens=coding_lens,
        weight_frequency=focused_weight_frequency,
        weight_documents=focused_weight_documents,
        weight_analytic=focused_weight_analytic,
        min_score=focused_min_score,
        max_calls=max_focused_review_calls,
        max_tokens=max_focused_review_tokens,
        neighbor_limit=focused_neighbor_limit,
        outside_limit=focused_outside_limit,
        negative_case_limit=focused_negative_case_limit,
        shares={
            "develop": focused_share_develop,
            "compete": focused_share_compete,
            "outside": focused_share_outside,
        },
        analytic_features=focused_analytic_features,
        validation_retries=validation_retries,
        selection_rule=focused_selection_rule,
        selection_k=focused_selection_k,
        selection_quantile=focused_selection_quantile,
        raise_all=focused_raise_all,
        warn_fraction=focused_raise_warn_fraction,
    )
    performance = _performance(batch_size=batch_size, concurrency=concurrency, token_budget=token_budget)
    manifest = focused_manifest(
        source,
        model=model,
        context=context,
        procedure=procedure,
        performance=performance,
    )
    metadata = {
        "package_version": __version__,
        "protocol": "focused-coding-1",
        "stage": "focused",
        "procedure": procedure,
        "performance": performance,
        "checkpoint_manifest": manifest,
        "pins": [str(item) for item in (pins or [])],
        "output_dir": None if output_dir is None else str(output_dir),
        "context": manifest["context"],
        "model": manifest["model"],
        "saturation_judgment": None,
    }
    limit = recursion_limit if recursion_limit is not None else max(250, 8 * max(1, len(source.passages)) + 40)
    runtime = FocusedRuntime(
        coder=model if not isinstance(model, str) else _resolve_model(model),
        output_dir=None if output_dir is None else str(output_dir),
        batch_size=performance["batch_size"],
        token_budget=performance["token_budget"],
        limiter=CallLimiter(performance["concurrency"]),
        validation_retries=validation_retries,
    )
    if isinstance(model, str):
        metadata["model"] = {
            "model": describe_model(runtime.coder),
            "model_settings": chat_model_settings(runtime.coder),
        }
        manifest = dict(manifest)
        manifest["model"] = metadata["model"]
        metadata["checkpoint_manifest"] = manifest

    own_saver = checkpointer is None and checkpoint_path is None
    if checkpoint_path is not None:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise FocusedError(
                "checkpoint_path requires the sqlite checkpointer. "
                "Install the checkpoint extra."
            ) from exc

        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            return await _invoke(
                runtime,
                saver,
                source,
                metadata,
                thread_id=thread_id or "focused",
                interrupt_enabled=interrupt,
                resume=resume,
                resuming=resuming,
                allow_changes=allow_changes or continuation,
                recursion_limit=limit,
                output_dir=output_dir,
            )
    saver = checkpointer if checkpointer is not None else _memory_saver()
    if thread_id is None:
        thread_id = "focused" if not own_saver else uuid.uuid4().hex
    return await _invoke(
        runtime,
        saver,
        source,
        metadata,
        thread_id=thread_id,
        interrupt_enabled=interrupt,
        resume=resume,
        resuming=resuming,
        allow_changes=allow_changes or continuation,
        recursion_limit=limit,
        output_dir=output_dir,
    )


def _resolve_model(model: str) -> BaseChatModel:
    from automated_grounded_theory.backends import resolve_chat_model

    return resolve_chat_model(model)


def _memory_saver():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


async def _invoke(
    runtime: FocusedRuntime,
    saver,
    source: OpenCodeResult,
    metadata: dict,
    *,
    thread_id: str,
    interrupt_enabled: bool,
    resume: Any,
    resuming: bool,
    allow_changes: bool,
    recursion_limit: int,
    output_dir: str | Path | None,
) -> FocusedResult:
    graph = build_focused_graph(runtime).compile(checkpointer=saver)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    snapshot = await graph.aget_state(config)
    has_state = bool(getattr(snapshot, "values", None))
    if resuming:
        if not has_state:
            raise FocusedError(
                f"No focused-coding checkpoint was found for thread {thread_id!r}. "
                "Start a run before resuming."
            )
        if not (getattr(snapshot, "interrupts", ()) or getattr(snapshot, "next", ())):
            raise FocusedError(f"Thread {thread_id!r} is not waiting on a researcher checkpoint.")
        saved_meta = dict((snapshot.values or {}).get("metadata") or {})
        continuation = _check_resume_manifest(
            saved_meta.get("checkpoint_manifest") or {},
            metadata.get("checkpoint_manifest") or {},
            allow_changes=allow_changes,
        )
        if continuation:
            metadata = dict(metadata)
            metadata["continuations"] = list(saved_meta.get("continuations") or []) + continuation
            await graph.aupdate_state(config, {"metadata": {**saved_meta, **metadata}})
        await graph.ainvoke(Command(resume=resume), config)
    else:
        if has_state:
            raise FocusedError(
                f"Checkpoint thread {thread_id!r} already has a focused run. "
                "Pass resuming=True to continue it, or a new thread_id to start another."
            )
        await graph.ainvoke(
            _initial_state(source, metadata, interrupt_enabled=interrupt_enabled),
            config,
        )
    snapshot = await graph.aget_state(config)
    payloads = _interrupt_payloads(snapshot)
    interrupted = bool(payloads) or bool(getattr(snapshot, "next", ()))
    state = dict(snapshot.values or {})
    if output_dir and interrupted and not state.get("export_paths"):
        state = dict(state)
        state["export_paths"] = export_focused_state(state, output_dir)
    result = _result_from_state(
        state,
        interrupted=interrupted,
        payload=payloads[0] if len(payloads) == 1 else ({"payloads": payloads} if payloads else None),
        thread_id=thread_id,
        resumable=interrupted,
        checkpointer=saver,
    )
    return result


@wraps(afocused_code)
def focused_code(*args: Any, **kwargs: Any) -> FocusedResult:
    """Synchronous wrapper for :func:`afocused_code`."""
    return run_sync(afocused_code(*args, **kwargs))


async def aread_focused_interrupt(checkpointer: Any, thread_id: str) -> dict[str, Any]:
    """Return the interrupt payload for a paused focused-coding thread."""
    graph = build_focused_graph().compile(checkpointer=checkpointer)
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    payloads = _interrupt_payloads(snapshot)
    return {
        "thread_id": thread_id,
        "interrupted": bool(payloads),
        "next": list(getattr(snapshot, "next", ()) or ()),
        "payload": payloads[0] if len(payloads) == 1 else None,
        "payloads": payloads,
    }


def read_focused_interrupt(checkpointer: Any, thread_id: str) -> dict[str, Any]:
    """Synchronous form of :func:`aread_focused_interrupt`."""
    return run_sync(aread_focused_interrupt(checkpointer, thread_id))


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for the researcher checkpoint.

    ``agt-focused interrupt`` prints the payload. ``agt-focused resume``
    continues with ``Command(resume=...)``.
    """
    parser = argparse.ArgumentParser(
        prog="agt-focused",
        description="View or resume the focused-coding researcher checkpoint.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("interrupt", help="Print the LangGraph interrupt payload.")
    show.add_argument("--checkpoint", required=True, help="SQLite checkpoint written by focused_code.")
    show.add_argument("--thread", required=True, help="thread_id of the paused run.")
    resume_parser = sub.add_parser("resume", help="Resume with Command(resume=...).")
    resume_parser.add_argument("--checkpoint", required=True)
    resume_parser.add_argument("--thread", required=True)
    resume_parser.add_argument("--resume", required=True, help="JSON value passed to Command(resume=...).")
    resume_parser.add_argument("--open-run", required=True, help="Open-coding output directory.")
    resume_parser.add_argument("--model", required=True, help="provider:model or the same model string as the run.")
    resume_parser.add_argument("--seed", type=int, default=0)
    resume_parser.add_argument("--allow-changes", action="store_true")
    resume_parser.add_argument("--continuation", action="store_true")
    resume_parser.add_argument("--output-dir")
    resume_parser.add_argument("--batch-size", type=int, default=DEFAULT_FOCUSED_BATCH_SIZE)
    resume_parser.add_argument("--max-focused-review-calls", type=int, default=DEFAULT_MAX_FOCUSED_REVIEW_CALLS)
    resume_parser.add_argument("--max-focused-review-tokens", type=int, default=DEFAULT_MAX_FOCUSED_REVIEW_TOKENS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "interrupt":
        payload = run_sync(_show_checkpoint(args.checkpoint, args.thread))
        _print_json(payload)
        return 0 if payload.get("interrupted") else 1
    decision = json.loads(args.resume)
    result = focused_code(
        args.open_run,
        model=args.model,
        seed=args.seed,
        resume=decision,
        resuming=True,
        checkpoint_path=args.checkpoint,
        thread_id=args.thread,
        allow_changes=args.allow_changes,
        continuation=args.continuation,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_focused_review_calls=args.max_focused_review_calls,
        max_focused_review_tokens=args.max_focused_review_tokens,
        interrupt=True,
    )
    _print_json(
        {
            "interrupted": result.interrupted,
            "status": result.metadata.get("status"),
            "stop": result.metadata.get("stop"),
            "raised_code_ids": result.raised_code_ids,
            "thread_id": result.thread_id,
        }
    )
    return 0


async def _show_checkpoint(path: str, thread_id: str) -> dict[str, Any]:
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:
        raise FocusedError(
            "agt-focused interrupt requires the sqlite checkpointer. Install the checkpoint extra."
        ) from exc
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        return await aread_focused_interrupt(saver, thread_id)


# Imported for callers that resume the compiled graph themselves.
__all__ = [
    "DEFAULT_FOCUSED_BATCH_SIZE",
    "DEFAULT_MAX_FOCUSED_REVIEW_CALLS",
    "DEFAULT_MAX_FOCUSED_REVIEW_TOKENS",
    "FocusedError",
    "FocusedResult",
    "afocused_code",
    "apply_selection_override",
    "build_focused_graph",
    "build_recode_queue",
    "cite_note",
    "contrast_code_ids",
    "focused_code",
    "read_focused_interrupt",
    "select_focused_codes",
]
