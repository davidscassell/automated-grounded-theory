"""0.9.16 review fixes. Fakes only. No network calls."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from automated_grounded_theory import axial_code
from automated_grounded_theory.backends import InstallationMismatch, installation_record
from automated_grounded_theory.version import __version__
from tests.test_v080 import FakeAxialCoder, _corpus


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
