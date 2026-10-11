"""Exports must import through Taguette's own codebook reader.

Taguette's CSV importer is ``taguette.import_codebook.list_tags``. It expects
a ``tag``, ``name``, or ``path`` column and an optional ``description``.
Hierarchy is a backslash in the path (``interesting\\places`` in Taguette's
tests). Taguette writes REFI-QDA ``.qdc`` and does not import it; the file is
checked against the attributes that exporter writes.
"""

import io
from xml.etree import ElementTree as ET

import pytest

from automated_grounded_theory import build_themes, open_code
from automated_grounded_theory.export import QDC_NAMESPACE
from tests.corpus import CONTEXT, CORPUS
from tests.fake_llm import FakeCoder
from tests.fake_themer import FakeThemer

taguette_import = pytest.importorskip("taguette.import_codebook")
taguette_validate = pytest.importorskip("taguette.validate")


def _tags(path):
    payload = path.read_bytes()
    return taguette_import.list_tags(io.BytesIO(payload))


def test_open_code_csv_imports_and_qdc_matches_taguette_export_shape(tmp_path):
    result = open_code(
        CORPUS,
        CONTEXT,
        model=FakeCoder(),
        seed=7,
        sweep_every=25,
        output_dir=tmp_path,
        pipeline=False,
    )
    tags = _tags(tmp_path / "taguette_codebook.csv")
    assert tags
    assert len(tags) == len(result.codebook)
    for tag in tags:
        taguette_validate.tag_path(tag["path"])
        assert "\\" not in tag["path"]
        assert tag["path"]
        assert "description" in tag
    root = ET.fromstring((tmp_path / "codebook.qdc").read_text(encoding="utf-8"))
    assert root.tag == f"{{{QDC_NAMESPACE}}}CodeBook"
    codes = root.findall(f".//{{{QDC_NAMESPACE}}}Code")
    assert len(codes) == len(tags)
    for code in codes:
        assert code.get("isCodable") == "true"
        assert code.get("guid")
        assert code.get("name")
    assert root.find(f"{{{QDC_NAMESPACE}}}Sets") is not None


def test_theme_paths_import_as_parent_and_child(tmp_path):
    coded = open_code(CORPUS, CONTEXT, model=FakeCoder(), seed=7, sweep_every=25, pipeline=False)
    build_themes(coded, CONTEXT, model=FakeThemer(), output_dir=tmp_path, seed=1, chart=False)
    tags = _tags(tmp_path / "taguette_themes.csv")
    paths = [tag["path"] for tag in tags]
    children = [path for path in paths if "\\" in path]
    parents = [path for path in paths if "\\" not in path]
    assert parents
    assert children
    for child in children:
        taguette_validate.tag_path(child)
        parent, _code = child.split("\\", 1)
        assert parent in parents
    root = ET.fromstring((tmp_path / "taguette_themes.qdc").read_text(encoding="utf-8"))
    names = [code.get("name") for code in root.findall(f".//{{{QDC_NAMESPACE}}}Code")]
    assert any("\\" in name for name in names)
    assert any(name in parents for name in names)
