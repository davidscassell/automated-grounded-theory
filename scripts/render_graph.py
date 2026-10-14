#!/usr/bin/env python3
"""Regenerate the open-coding diagram in README.md from the compiled graph.

The fenced block is the string returned by
``get_graph("open_code").compile().get_graph().draw_mermaid()``.
``get_graph(xray=True)`` is used only when it expands subgraphs. On this
graph it does not: there are no subgraphs, so the two drawings match.

Sequential, document, wave, pipeline, and parallel-document settings compile
the same nodes and edges. ``interrupt_before_sweep=True`` only annotates
``sweep`` with an interrupt-before breakpoint.

    python scripts/render_graph.py
    python scripts/render_graph.py --check
    python scripts/render_graph.py --png docs/open_code_graph.png

``--check`` compares the README block to the compiled graph and checks that
the mode variants still share that drawing. It does not write files.
A PNG is written when mermaid-cli (``mmdc``) is on ``PATH`` or passed with
``--mmdc``. ``draw_mermaid_png`` is not used: it posts the diagram to
mermaid.ink.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.graph import RunConfig, _IdleModel, build_graph, compile_graph, get_graph
from automated_grounded_theory.retrieval import LexicalCodeRetriever

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
DEFAULT_PNG = ROOT / "docs" / "open_code_graph.png"
BEGIN = "<!-- graph:open_code:begin -->"
END = "<!-- graph:open_code:end -->"
BLOCK_RE = re.compile(
    re.escape(BEGIN) + r"\n```mermaid\n(.*?)```\n" + re.escape(END),
    re.DOTALL,
)
FOCUSED_BEGIN = "<!-- graph:focused:begin -->"
FOCUSED_END = "<!-- graph:focused:end -->"
FOCUSED_RE = re.compile(
    re.escape(FOCUSED_BEGIN) + r"\n```mermaid\n(.*?)```\n" + re.escape(FOCUSED_END),
    re.DOTALL,
)


def _runtime(**overrides) -> RunConfig:
    fields = dict(
        coder=_IdleModel(),
        sweeper=_IdleModel(),
        retriever=LexicalCodeRetriever(),
        embeddings=None,
        context=ResearchContext(research_question="What is happening in this text?"),
        prompt_cache=False,
    )
    fields.update(overrides)
    return RunConfig(**fields)


def compiled_default():
    """Compile the graph ``get_graph("open_code")`` draws, with no interrupt."""
    return get_graph("open_code").compile()


def diagram_mermaid(compiled=None) -> str:
    """Return the mermaid source for the README.

    ``xray=True`` replaces the plain drawing only when it adds subgraph nodes.
    """
    compiled = compiled_default() if compiled is None else compiled
    plain = compiled.get_graph().draw_mermaid()
    xray = compiled.get_graph(xray=True).draw_mermaid()
    if xray != plain:
        return xray
    return plain


def mode_diagrams() -> dict[str, str]:
    """Mermaid for the scheduling modes. They compile one graph."""
    shared = {
        "document": {},
        "wave": {"review_mode": "wave", "pipeline": True, "concurrency": 4, "batch_size": 4},
        "pipeline": {"review_mode": "document", "pipeline": True, "concurrency": 4},
        "sequential": {
            "review_mode": "document",
            "pipeline": False,
            "concurrency": 1,
            "batch_size": 1,
            "passages_per_request": 1,
            "max_concurrent_requests": 1,
            "passages_per_codebook_update": 1,
        },
    }
    return {name: build_graph(_runtime(**options)).compile().get_graph().draw_mermaid() for name, options in shared.items()}


def interrupt_mermaid() -> str:
    """Drawing with ``interrupt_before_sweep``. Edges stay the same."""
    return compile_graph(_runtime(), None, interrupt_before_sweep=True).get_graph().draw_mermaid()


def focused_mermaid() -> str:
    """Mermaid for ``get_graph("focused")``, the separate focused-coding graph."""
    return get_graph("focused").compile().get_graph().draw_mermaid()


def readme_mermaid(readme: str) -> str:
    match = BLOCK_RE.search(readme)
    if match is None:
        raise ValueError(
            "README.md is missing the generated open_code mermaid block "
            f"between {BEGIN} and {END}."
        )
    return match.group(1)


def readme_focused_mermaid(readme: str) -> str:
    match = FOCUSED_RE.search(readme)
    if match is None:
        raise ValueError(
            "README.md is missing the generated focused mermaid block "
            f"between {FOCUSED_BEGIN} and {FOCUSED_END}."
        )
    return match.group(1)


def _replace_block(readme: str, pattern: re.Pattern[str], begin: str, end: str, mermaid: str) -> str:
    if not mermaid.endswith("\n"):
        mermaid = mermaid + "\n"
    block = f"{begin}\n```mermaid\n{mermaid}```\n{end}"

    def _swap(match: re.Match[str]) -> str:
        del match
        return block

    updated, count = pattern.subn(_swap, readme, count=1)
    if count != 1:
        raise ValueError(f"README.md is missing the generated mermaid block between {begin} and {end}.")
    return updated


def replace_readme_mermaid(readme: str, mermaid: str) -> str:
    # Return the block literally. A replacement string would treat
    # backslashes in a node id as re escapes.
    return _replace_block(readme, BLOCK_RE, BEGIN, END, mermaid)


def replace_focused_mermaid(readme: str, mermaid: str) -> str:
    return _replace_block(readme, FOCUSED_RE, FOCUSED_BEGIN, FOCUSED_END, mermaid)


def _edge_tuples(compiled) -> set[tuple[str, str, bool]]:
    return {(edge.source, edge.target, bool(edge.conditional)) for edge in compiled.get_graph().edges}


def check_variants(expected: str) -> None:
    diagrams = mode_diagrams()
    drifted = [name for name, source in diagrams.items() if source != expected]
    if drifted:
        raise SystemExit(
            "These coding modes no longer compile the default graph: "
            + ", ".join(drifted)
            + ". Update the README note before replacing the diagram."
        )
    interrupted = compile_graph(_runtime(), None, interrupt_before_sweep=True)
    default = compiled_default()
    if _edge_tuples(interrupted) != _edge_tuples(default):
        raise SystemExit(
            "interrupt_before_sweep changed the compiled edges. "
            "The README says it only annotates the sweep node."
        )
    if interrupt_mermaid() == expected or "__interrupt" not in interrupt_mermaid():
        raise SystemExit(
            "interrupt_before_sweep no longer annotates the sweep node. "
            "Update the README variant note."
        )
    plain = default.get_graph().draw_mermaid()
    xray = default.get_graph(xray=True).draw_mermaid()
    if plain != xray and expected != xray:
        raise SystemExit("xray=True expands subgraphs, but the README block is not that drawing.")


def write_png(mermaid: str, path: Path, mmdc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "graph.mmd"
        source.write_text(mermaid, encoding="utf-8")
        subprocess.run(
            [mmdc, "-i", str(source), "-o", str(path), "-e", "png", "-b", "white"],
            check=True,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the README block or the mode note is stale.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=DEFAULT_PNG,
        help="Where to write the PNG (default: docs/open_code_graph.png).",
    )
    parser.add_argument(
        "--mmdc",
        default=None,
        help="Path to mermaid-cli. Defaults to mmdc on PATH.",
    )
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Do not render the PNG.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mermaid = diagram_mermaid()
    check_variants(mermaid)
    focused = focused_mermaid()
    readme = README_PATH.read_text(encoding="utf-8")
    try:
        current = readme_mermaid(readme)
        current_focused = readme_focused_mermaid(readme)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.check:
        if current != mermaid:
            raise SystemExit("README mermaid block does not match graph.get_graph().draw_mermaid().")
        if current_focused != focused:
            raise SystemExit("README focused mermaid block does not match get_graph('focused').draw_mermaid().")
        print("README open_code and focused diagrams match the compiled graphs.")
        return 0
    updated = replace_focused_mermaid(replace_readme_mermaid(readme, mermaid), focused)
    if updated != readme:
        README_PATH.write_text(updated, encoding="utf-8")
        print(f"Updated {README_PATH.relative_to(ROOT)}")
    else:
        print("README mermaid blocks already match.")
    if not args.skip_png:
        mmdc = args.mmdc or shutil.which("mmdc")
        if not mmdc:
            print("mermaid-cli (mmdc) not found; left the PNG unchanged.", file=sys.stderr)
        else:
            write_png(mermaid, args.png, mmdc)
            print(f"Wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
