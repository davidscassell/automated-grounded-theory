"""Automated initial (open) coding, then a theme report.

Codes are produced passage by passage from the corpus. The research context
sensitizes attention and is not a codebook. The theme report groups those
codes afterwards for presentation. It does not reread the passages. Focused
coding is a separate graph that rereads passages against a proposed subset of
codes already earned. It is not the theme report.
"""

from automated_grounded_theory.api import (
    amulti_seed_stability,
    aopen_code,
    compare_runs,
    multi_seed_stability,
    open_code,
    strict_sequential_settings,
)
from automated_grounded_theory.evaluate import aevaluate, chen_metrics, evaluate_run
from automated_grounded_theory.context import PresetCodesError, ResearchContext
from automated_grounded_theory.focused import (
    FocusedError,
    FocusedResult,
    afocused_code,
    focused_code,
    read_focused_interrupt,
)
from automated_grounded_theory.graph import CodingError, get_graph
from automated_grounded_theory.results import OpenCodeResult, load_run
from automated_grounded_theory.retrieval import (
    CodeRetriever,
    EmbeddingCodeRetriever,
    LexicalCodeRetriever,
)
from automated_grounded_theory.segment import plan_coding
from automated_grounded_theory.stability import StabilityReport
from automated_grounded_theory.themes import ThemeError, ThemeResult, abuild_themes, build_themes
from automated_grounded_theory.version import __version__

__all__ = [
    "CodeRetriever",
    "CodingError",
    "EmbeddingCodeRetriever",
    "FocusedError",
    "FocusedResult",
    "LexicalCodeRetriever",
    "OpenCodeResult",
    "PresetCodesError",
    "ResearchContext",
    "StabilityReport",
    "ThemeError",
    "ThemeResult",
    "__version__",
    "abuild_themes",
    "aevaluate",
    "afocused_code",
    "amulti_seed_stability",
    "aopen_code",
    "build_themes",
    "chen_metrics",
    "compare_runs",
    "evaluate_run",
    "focused_code",
    "get_graph",
    "load_run",
    "read_focused_interrupt",
    "multi_seed_stability",
    "open_code",
    "plan_coding",
    "strict_sequential_settings",
]
