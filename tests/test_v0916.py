"""0.9.16 review fixes. Fakes only. No network calls."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from automated_grounded_theory import axial_code, theoretical_sample
from automated_grounded_theory.backends import InstallationMismatch, installation_record
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder, _corpus
from tests.test_v090 import FakeSamplingCoder, _payload


def _axial(**overrides):
    fields = dict(
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    fields.update(overrides)
    return fields


class _Dist:
    def __init__(self, version: str, root: Path, direct_url: str | None = None):
        self.version = version
        self._root = root
        self._direct = direct_url

    def locate_file(self, rel: str):
        return self._root / rel

    def read_text(self, name: str) -> str:
        if name == "direct_url.json" and self._direct is not None:
            return self._direct
        raise FileNotFoundError(name)


def test_editable_install_is_checked_before_a_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An editable checkout is consistent from another working directory.

    Metadata may live in site-packages while the import comes from the
    checkout. A different distribution version still fails before the coder runs.
    """
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        coder = FakeAxialCoder()
        result = axial_code(_corpus(), coder, **_axial())
    finally:
        os.chdir(previous)
    assert coder.call_count >= 1
    provenance = result.metadata["provenance"]
    assert provenance["module_version"] == provenance["distribution_version"] == __version__
    assert provenance["layout"] in {"editable", "distribution"}
    assert Path(provenance["module_path"]).is_file()

    imported = Path(provenance["module_path"])
    project = imported.parents[1]
    site = tmp_path / "site-packages"
    site.mkdir()
    direct = json.dumps({"dir_info": {"editable": True}, "url": project.as_uri()})
    monkeypatch.setattr(
        "automated_grounded_theory.backends.importlib.metadata.distribution",
        lambda name: _Dist(__version__, site, direct),
    )
    record = installation_record()
    assert record["layout"] == "editable"
    assert Path(record["source_root"]) == project
    editable = axial_code(_corpus(), FakeAxialCoder(), **_axial())
    assert editable.metadata["provenance"]["layout"] == "editable"

    other = tmp_path / "other-checkout"
    other.mkdir()
    named = json.dumps({"dir_info": {"editable": True}, "url": other.as_uri()})
    monkeypatch.setattr(
        "automated_grounded_theory.backends.importlib.metadata.distribution",
        lambda name: _Dist(__version__, site, named),
    )
    with pytest.raises(InstallationMismatch, match="editable metadata"):
        axial_code(_corpus(), FakeAxialCoder(), **_axial())

    mismatch = FakeAxialCoder()
    monkeypatch.setattr(
        "automated_grounded_theory.backends.importlib.metadata.distribution",
        lambda name: _Dist("9.9.9", site, direct),
    )
    with pytest.raises(InstallationMismatch, match="9.9.9"):
        axial_code(_corpus(), mismatch, **_axial())
    assert mismatch.call_count == 0


def _sampling_visit_allowance(calls: int) -> int:
    """The same transitions ``_sampling_recursion_limit`` counts."""
    return 5 + 1 + (calls * 3) + 1 + 3


class _EveryPassage(FakeSamplingCoder):
    """Select every proposal on the thin category, then reread each one."""

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "ModelSamplingSelection":
                selections = []
                for row in body.get("directions") or []:
                    if str(row.get("kind") or "") != "thin_category":
                        continue
                    direction_id = str(row.get("direction_id") or "")
                    for group in body.get("passage_proposals") or []:
                        if str(group.get("direction_id") or "") != direction_id:
                            continue
                        for proposal in group.get("proposals") or []:
                            selections.append(
                                {
                                    "direction_id": direction_id,
                                    "passage_id": str(proposal.get("passage_id") or ""),
                                }
                            )
                parsed = schema(
                    selections=selections,
                    rationale="Every retrieved passage for the thin category is reread.",
                )
            elif name == "SamplingRereadNote":
                passage_id = str(body.get("passage_id") or "")
                text = str(body.get("text") or "")
                parsed = schema(
                    passage_id=passage_id,
                    judgment="relevant",
                    note="This passage bears on the thin category.",
                    quote=text[:48],
                )
            elif name == "ModelSaturationJudgment":
                parsed = schema(
                    assessment="not_saturated",
                    rationale="The window is interpretable and a sampling gap is still open.",
                )
            else:
                parsed = schema()
            if not include_raw:
                return parsed
            from langchain_core.messages import AIMessage

            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_call)


def test_default_recursion_limit_finishes_thirty_two_rereads():
    """Thirty-two rereads need a visit to arm each pair and a visit to send it.

    A limit of the call ceiling plus a fixed overhead stops before the last
    requests. The default limit is the graph's own transitions.
    """
    stem = "zephyrcrate"
    passages = [
        {
            "passage_id": f"extra-{index:02d}",
            "doc_id": f"d-{index:02d}",
            "respondent_id": f"r-{index:02d}",
            "text": f"Another {stem} account that was not coded yet.",
        }
        for index in range(32)
    ]
    corpus = {
        "codebook": {
            "c-pack": {
                "code_id": "c-pack",
                "code": stem,
                "definition": f"A {stem} that still needs another document.",
            }
        },
        "passages": [
            {
                "passage_id": "coded",
                "doc_id": "d-coded",
                "respondent_id": "r-coded",
                "text": f"The coded {stem} is already evidence.",
            },
            *passages,
        ],
        "segments": [{"segment_id": "s-coded", "code_id": "c-pack", "passage_id": "coded"}],
        "properties": [],
        "dimensions": [],
        "relationships": [],
        "memos": [],
    }
    calls = 32
    result = theoretical_sample(
        corpus,
        _EveryPassage(),
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        seed=3,
        evidence_limit=32,
        category_documents=2,
        max_sampling_review_calls=calls,
        max_decision_calls=4,
        sampling_token_budget=200000,
        max_sampling_review_tokens=200000,
        sampling_request_timeout=30,
    )
    assert result.metadata.get("stop") != "recursion_limit"
    assert result.metadata["performance"]["recursion_limit"] >= _sampling_visit_allowance(calls)
    completed = {row["passage_id"] for row in result.reread_notes}
    assert completed == {row["passage_id"] for row in passages}
    assert len(result.reread_notes) == calls
