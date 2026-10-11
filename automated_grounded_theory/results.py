"""Result object returned by :func:`automated_grounded_theory.open_code`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
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
        ops=list(state.get("ops") or []),
        review=list(state.get("review") or []),
        export_paths=dict(export_paths or state.get("export_paths") or {}),
        interrupted=interrupted,
        thread_id=thread_id,
        resumable=resumable,
    )
