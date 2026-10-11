"""Result object returned by :func:`automated_grounded_theory.open_code`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from automated_grounded_theory.codebook import codebook_list
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
        }


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
        metadata=dict(state.get("metadata") or {}),
        warnings=list(state.get("warnings") or []),
        export_paths=dict(export_paths or state.get("export_paths") or {}),
        interrupted=interrupted,
        thread_id=thread_id,
        resumable=resumable,
    )
