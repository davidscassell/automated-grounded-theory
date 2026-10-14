"""Result object returned by :func:`automated_grounded_theory.open_code`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automated_grounded_theory.codebook import codebook_list
from automated_grounded_theory.records import materialize_records, percentile
from automated_grounded_theory.retrieval import (
    DEFAULT_MAX_FINAL_REVIEW_CALLS,
    DEFAULT_MAX_FINAL_REVIEW_TOKENS,
    assignment_review_totals,
)
from automated_grounded_theory.saturation import saturation_report


@dataclass
class OpenCodeResult:
    """One open-coding run.

    ``saturation_report['judgment']`` is always null. The curve is for the
    researcher to read.
    """

    codebook: list[dict]
    segments: list[dict]
    memos: list[dict]
    passages: list[dict]
    saturation: list[dict]
    saturation_report: dict
    validation_log: list[dict]
    sweep_log: list[dict]
    prompt_log: list[dict]
    metadata: dict
    warnings: list[str] = field(default_factory=list)
    export_paths: dict[str, str] = field(default_factory=dict)
    interrupted: bool = False
    thread_id: str = ""
    resumable: bool = False
    ops: list[dict] = field(default_factory=list)
    review: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "codebook": self.codebook,
            "segments": self.segments,
            "memos": self.memos,
            "passages": self.passages,
            "saturation": self.saturation,
            "saturation_report": self.saturation_report,
            "validation_log": self.validation_log,
            "sweep_log": self.sweep_log,
            "prompt_log": self.prompt_log,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "export_paths": self.export_paths,
            "interrupted": self.interrupted,
            "thread_id": self.thread_id,
            "resumable": self.resumable,
            "ops": self.ops,
            "review": self.review,
        }

    @classmethod
    def load(cls, directory: str | Path) -> "OpenCodeResult":
        """Read a run written by ``open_code``.

        ``run.json`` is the whole result. When it is absent, the split files
        are used. ``saturation.json`` is the report dict (its ``points`` list
        is the curve).
        """
        place = Path(directory)
        if not place.is_dir():
            raise FileNotFoundError(f"No output directory at {place}")
        run_path = place / "run.json"
        if run_path.is_file():
            return cls.from_dict(json.loads(run_path.read_text(encoding="utf-8")))

        def read(name: str, default: Any) -> Any:
            path = place / name
            if not path.is_file():
                return default
            return json.loads(path.read_text(encoding="utf-8"))

        report = read("saturation.json", {})
        if isinstance(report, list):
            points = report
            report = {"points": points, "judgment": None}
        elif isinstance(report, dict):
            points = list(report.get("points") or [])
        else:
            points = []
            report = {}
        return cls.from_dict(
            {
                "codebook": read("codebook.json", []),
                "segments": read("coded_segments.json", []),
                "memos": read("memos.json", []),
                "passages": read("passages.json", []),
                "saturation": points,
                "saturation_report": report,
                "validation_log": read("validation_log.json", []),
                "sweep_log": read("sweep_log.json", []),
                "prompt_log": read("prompt_log.json", []),
                "metadata": read("run_metadata.json", {}),
                "warnings": [],
                "ops": read("ops.json", []),
                "review": read("review.json", []),
            }
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "OpenCodeResult":
        saturation = payload.get("saturation") or []
        report = payload.get("saturation_report")
        if isinstance(saturation, dict):
            report = saturation
            saturation = list(saturation.get("points") or [])
        if not isinstance(report, dict):
            report = {}
        return cls(
            codebook=list(payload.get("codebook") or []),
            segments=list(payload.get("segments") or []),
            memos=list(payload.get("memos") or []),
            passages=list(payload.get("passages") or []),
            saturation=list(saturation or []),
            saturation_report=report,
            validation_log=list(payload.get("validation_log") or []),
            sweep_log=list(payload.get("sweep_log") or []),
            prompt_log=list(payload.get("prompt_log") or []),
            metadata=dict(payload.get("metadata") or {}),
            warnings=list(payload.get("warnings") or []),
            export_paths=dict(payload.get("export_paths") or {}),
            interrupted=bool(payload.get("interrupted") or False),
            thread_id=str(payload.get("thread_id") or ""),
            resumable=bool(payload.get("resumable") or False),
            ops=list(payload.get("ops") or []),
            review=list(payload.get("review") or []),
        )


def load_run(directory: str | Path) -> OpenCodeResult:
    """Load an open-coding output directory. Same as :meth:`OpenCodeResult.load`."""
    return OpenCodeResult.load(directory)


def missing_slot_retry_cost(prompt_log: list) -> dict[str, int]:
    """Token cost of solo retries for passages a batch reply left out.

    The batch call is not included. Each retry is tagged ``missing_slot_retry``
    on the prompt log when it is sent.
    """
    calls = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for entry in prompt_log:
        if not isinstance(entry, dict) or not entry.get("missing_slot_retry"):
            continue
        calls += 1
        usage = entry.get("usage") or {}
        incoming = int(usage.get("input_tokens") or 0)
        outgoing = int(usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or 0)
        if total <= 0:
            total = incoming + outgoing
        input_tokens += incoming
        output_tokens += outgoing
        total_tokens += total
    return {
        "missing_slot_retry_calls": calls,
        "missing_slot_retry_input_tokens": input_tokens,
        "missing_slot_retry_output_tokens": output_tokens,
        "missing_slot_retry_tokens": total_tokens,
    }


def _attach_review_status(metadata: dict, state: dict, warnings: list[str]) -> None:
    """Record whether every assignment was sent, and say so when it was not.

    A finished run with ``review_complete`` false is not fully consolidated.
    The sweep log can still show that a consolidation pass ran. That pass is
    one bounded batch. It is not a claim that every assignment was in a prompt.
    """
    totals = assignment_review_totals(state.get("codebook") or {})
    max_calls = int(metadata.get("max_final_review_calls", DEFAULT_MAX_FINAL_REVIEW_CALLS))
    max_tokens = int(metadata.get("max_final_review_tokens", DEFAULT_MAX_FINAL_REVIEW_TOKENS))
    calls = int(metadata.get("review_calls_used") or 0)
    tokens = int(metadata.get("review_tokens_used") or 0)
    stop = str(metadata.get("review_stop") or "")
    needs = bool(totals["needs_review"])
    if not needs:
        stop = "complete"
    elif stop not in {"call_allowance", "token_allowance", "no_progress"}:
        if max_calls <= 0 or calls >= max_calls:
            stop = "call_allowance"
        elif max_tokens <= 0 or tokens >= max_tokens:
            stop = "token_allowance"
        else:
            stop = stop or "call_allowance"
    consolidation_ran = bool(state.get("consolidated")) or any(
        isinstance(item, dict) and item.get("consolidation") for item in (state.get("sweep_log") or [])
    )
    review_complete = not needs
    metadata["review_complete"] = review_complete
    metadata["review_stop"] = stop
    metadata["pending_counts"] = dict(totals["pending_counts"])
    metadata["pending_assignment_ids"] = {
        label: list(ids) for label, ids in totals["pending_assignment_ids"].items()
    }
    metadata["assignments_available"] = int(totals["assignments_available"])
    metadata["assignments_sent"] = int(totals["assignments_sent"])
    metadata["assignments_pending"] = int(totals["assignments_pending"])
    metadata["review_calls_used"] = calls
    metadata["review_tokens_used"] = tokens
    metadata["max_final_review_calls"] = max_calls
    metadata["max_final_review_tokens"] = max_tokens
    metadata["review_by_code"] = dict(totals["by_code"])
    metadata["consolidated"] = bool(consolidation_ran and review_complete)
    if review_complete:
        return
    note = _incomplete_review_warning(metadata)
    if note not in warnings:
        warnings.append(note)


def _incomplete_review_warning(metadata: dict) -> str:
    """Name the unsent evidence and the allowance that stopped the review."""
    counts = metadata.get("pending_counts") or {}
    pending_ids = metadata.get("pending_assignment_ids") or {}
    parts = []
    for label in sorted(counts):
        ids = list(pending_ids.get(label) or [])
        shown = ", ".join(ids[:30])
        if len(ids) > 30:
            shown = f"{shown}, and {len(ids) - 30} more"
        parts.append(f"{counts[label]} on {label} ({shown})")
    if not parts:
        parts.append("a code whose review fingerprint is not current")
    stop = str(metadata.get("review_stop") or "")
    calls = int(metadata.get("review_calls_used") or 0)
    tokens = int(metadata.get("review_tokens_used") or 0)
    max_calls = int(metadata.get("max_final_review_calls") or 0)
    max_tokens = int(metadata.get("max_final_review_tokens") or 0)
    reasons = {
        "call_allowance": (
            f"the final review call allowance is {max_calls} and {calls} follow-up calls were used"
        ),
        "token_allowance": (
            f"the final review token allowance is {max_tokens} and {tokens} estimated follow-up tokens were used"
        ),
        "no_progress": "a follow-up review sent no new assignment ids",
    }
    why = reasons.get(stop, "pending evidence remained")
    return (
        "Review stopped with evidence that was not sent: "
        + "; ".join(parts)
        + f". review_complete is false because {why}. "
        "The run is not fully consolidated."
    )


def result_from_state(
    state: dict,
    *,
    interrupted: bool,
    thread_id: str,
    resumable: bool,
    export_paths: dict[str, str] | None = None,
) -> OpenCodeResult:
    points = list(state.get("saturation") or [])
    passages = sorted(state.get("passages") or [], key=lambda item: item.get("coding_order", 0))
    metadata = dict(state.get("metadata") or {})
    if state.get("stage_seconds"):
        metadata["stage_seconds"] = {
            key: round(float(value), 4) for key, value in dict(state.get("stage_seconds") or {}).items()
        }
    metrics = dict(state.get("metrics") or {})
    metrics.update(missing_slot_retry_cost(list(state.get("prompt_log") or [])))
    samples = list(metrics.get("coding_request_latency_seconds") or metrics.get("coding_seconds") or [])
    if samples:
        metrics["coding_p50"] = percentile(samples, 50)
        metrics["coding_p95"] = percentile(samples, 95)
    prompt_log = list(state.get("prompt_log") or [])
    input_tokens = 0
    output_tokens = 0
    for entry in prompt_log:
        usage = entry.get("usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
    metrics["input_tokens"] = input_tokens
    metrics["output_tokens"] = output_tokens
    metadata["metrics"] = metrics
    metadata["lost_distinction_audit"] = list(state.get("lost_distinction_candidates") or [])
    metadata["policy_version"] = state.get("policy_version") or metadata.get("policy_version")
    records = materialize_records(
        documents=list(state.get("documents") or []),
        passages=passages,
        segments=list(state.get("segments") or []),
        codebook=state.get("codebook") or {},
    )
    metadata["records"] = {
        "documents": len(records["documents"]),
        "incidents": len(records["incidents"]),
        "evidence_spans": len(records["evidence_spans"]),
        "code_versions": len(records["code_versions"]),
    }
    metadata["code_versions"] = records["code_versions"]
    failed_ids = [
        str(item.get("passage_id") or "")
        for item in passages
        if item.get("coverage_status") == "failed" and str(item.get("passage_id") or "")
    ]
    metadata["failed_passages"] = failed_ids
    summary = dict(metadata.get("summary") or {})
    summary["failed_passages"] = failed_ids
    metadata["summary"] = summary
    warnings = list(state.get("warnings") or [])
    if not interrupted:
        _attach_review_status(metadata, state, warnings)
    metadata["coverage"] = [
        {
            "incident_id": item["incident_id"],
            "coverage_status": item["coverage_status"],
            "exclusion_reason": item["exclusion_reason"],
        }
        for item in records["incidents"]
    ]
    return OpenCodeResult(
        codebook=codebook_list(state.get("codebook") or {}),
        segments=list(state.get("segments") or []),
        memos=list(state.get("memos") or []),
        passages=passages,
        saturation=points,
        saturation_report=saturation_report(points),
        validation_log=list(state.get("validation_log") or []),
        sweep_log=list(state.get("sweep_log") or []),
        prompt_log=list(state.get("prompt_log") or []),
        metadata=metadata,
        warnings=warnings,
        ops=list(state.get("ops") or []),
        review=list(state.get("review") or []),
        export_paths=dict(export_paths or state.get("export_paths") or {}),
        interrupted=interrupted,
        thread_id=thread_id,
        resumable=resumable,
    )
