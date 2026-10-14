"""Execute the demo notebook offline.

This is the CI-style check that ``nbmake`` would run: load
``examples/demo.ipynb``, run every cell with no API key, and require a
finished codebook, metrics, themes, and a Taguette export. The notebook
stored in the repository is already executed in that offline mode.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "demo.ipynb"


def _source(nb) -> str:
    return "\n".join(cell.source for cell in nb.cells)


def test_demo_notebook_is_saved_executed_offline():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(nb)
    assert "OFFLINE = True" in source
    assert "tempfile.mkdtemp" in source
    assert 'examples" / "run"' not in source
    assert "FICTIONAL" in source or "fictional" in source
    assert "aopen_code" in source
    assert "xai:grok-4" in source
    assert "openai:gpt-4o" in source
    assert "anthropic:claude-sonnet-4-5" in source
    assert "ollama:llama3.1" in source
    executed = [
        cell
        for cell in nb.cells
        if cell.cell_type == "code" and cell.get("outputs")
    ]
    assert len(executed) >= 8
    assert not any(
        output.get("output_type") == "error"
        for cell in nb.cells
        for output in cell.get("outputs", [])
    )


def test_demo_notebook_runs_offline(tmp_path):
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient

    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    errors = [
        output
        for cell in nb.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []
    chunks = []
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append(output.get("text", ""))
            data = output.get("data") or {}
            for key in ("text/plain", "text/markdown"):
                if key in data:
                    chunks.append(data[key])
    rendered = "\n".join(chunks)
    assert "review_mode document" in rendered
    # short_turn_chars defaults to 0, so each survey comment stays its own passage (150).
    # short_turn_chars=60 merges short filler and yields 100. "Thanks for asking." is kept.
    assert "passages 150" in rendered
    assert "coverage_pct" in rendered
    assert "mean Jaccard 1.0" in rendered
    assert "taguette_codebook.csv" in rendered
    assert "taguette_themes.csv" in rendered
    assert "examples/run" not in rendered
    images = [
        output
        for cell in nb.cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    assert len(images) >= 2
    # The executed copy stays in tmp so this test does not rewrite the notebook.
    written = tmp_path / "demo-executed.ipynb"
    nbformat.write(nb, written)
    assert written.is_file()
