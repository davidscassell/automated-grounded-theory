"""The README diagram is the compiled open_code graph, not a hand-drawn sketch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _render_graph():
    path = ROOT / "scripts" / "render_graph.py"
    spec = importlib.util.spec_from_file_location("render_graph", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_readme_diagram_matches_compiled_graph():
    render = _render_graph()
    expected = render.diagram_mermaid()
    readme = render.README_PATH.read_text(encoding="utf-8")
    assert render.readme_mermaid(readme) == expected
    assert render.main(["--check"]) == 0


def test_coding_modes_share_the_default_graph():
    render = _render_graph()
    expected = render.diagram_mermaid()
    diagrams = render.mode_diagrams()
    assert set(diagrams) == {"document", "wave", "pipeline", "sequential"}
    assert set(diagrams.values()) == {expected}
    compiled = render.compiled_default()
    assert compiled.get_graph().draw_mermaid() == compiled.get_graph(xray=True).draw_mermaid()


def test_interrupt_before_sweep_keeps_the_same_edges():
    render = _render_graph()
    default = render.compiled_default()
    interrupted = render.compile_graph(render._runtime(), None, interrupt_before_sweep=True)
    assert render._edge_tuples(default) == render._edge_tuples(interrupted)
    annotated = render.interrupt_mermaid()
    assert annotated != render.diagram_mermaid()
    assert "__interrupt = before" in annotated


def test_open_code_png_is_a_png():
    png = ROOT / "docs" / "open_code_graph.png"
    assert png.is_file()
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
