"""Cheap checks that the current docs name the code's defaults.

Historical design notes may describe an older release. The README and
docs/README.md are the current contract for the values below.
"""

from __future__ import annotations

from pathlib import Path

from automated_grounded_theory.axial import (
    DEFAULT_AXIAL_TOKEN_BUDGET,
    DEFAULT_MAX_AXIAL_REVIEW_TOKENS,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES
from automated_grounded_theory.sampling import (
    DEFAULT_MAX_SAMPLING_TOKENS,
    DEFAULT_SAMPLING_TOKEN_BUDGET,
)
from automated_grounded_theory.saturation import STOPPING_QUIET_ROUNDS
from automated_grounded_theory.version import __version__


def test_documented_defaults_match_the_code():
    readme = Path("README.md").read_text(encoding="utf-8")
    index = Path("docs/README.md").read_text(encoding="utf-8")
    assert __version__ == "0.9.17"
    assert POLICY_VERSION == "0.9.17"
    assert f"`policy_version` is `{POLICY_VERSION}`" in readme
    assert f"Package `policy_version` is `{POLICY_VERSION}`" in index
    assert f"`pattern_min_cases` {PATTERN_MIN_CASES}" in readme
    assert f"`pattern_min_cases` is {PATTERN_MIN_CASES}" in index
    assert f"`max_axial_review_calls` | {8} |" in readme
    assert f"`max_axial_review_calls` is 8" in index
    assert f"`max_axial_review_tokens` | {DEFAULT_MAX_AXIAL_REVIEW_TOKENS} |" in readme
    assert f"`max_axial_review_tokens` is {DEFAULT_MAX_AXIAL_REVIEW_TOKENS}" in index
    assert f"`axial_token_budget` | {DEFAULT_AXIAL_TOKEN_BUDGET} |" in readme
    assert f"`axial_token_budget` is {DEFAULT_AXIAL_TOKEN_BUDGET}" in index
    assert f"`sampling_token_budget` | {DEFAULT_SAMPLING_TOKEN_BUDGET} |" in readme
    assert f"`sampling_token_budget` is {DEFAULT_SAMPLING_TOKEN_BUDGET}" in index
    assert f"`max_sampling_review_tokens` | {DEFAULT_MAX_SAMPLING_TOKENS} |" in readme
    assert f"`max_sampling_review_tokens` is {DEFAULT_MAX_SAMPLING_TOKENS}" in index
    assert "`max_sampling_review_calls` | 4 |" in readme
    assert "`max_sampling_review_calls` is 4" in index
    assert f"`quiet_rounds` | {STOPPING_QUIET_ROUNDS} |" in readme
    assert f"`quiet_rounds` is {STOPPING_QUIET_ROUNDS}" in index
    assert "does not do theoretical sampling, axial coding, selective coding, or a saturation decision" not in readme
    assert "Theoretical sampling is not in this release" not in Path("docs/design.md").read_text(encoding="utf-8")
    later = Path("docs/design-later-stages.md").read_text(encoding="utf-8")
    assert "Theoretical sampling, Glaser families, and the conditional/consequential matrix in this note are not implemented" not in later
    assert "experimental" in readme
    assert "downstream_precision" in readme
    assert "downstream_withheld_supported_recall" in readme
    assert "export_gate_relationships" in readme
    assert "load_adjudications" in readme
    assert "score_gate_adjudications" in readme


def test_historical_review_notes_are_indexed():
    index = Path("docs/README.md").read_text(encoding="utf-8")
    notes = sorted(path.name for path in Path("docs").glob("design-review-fixes-*.md"))
    assert notes
    for name in notes:
        assert name in index
    assert "Historical" in index
    assert "Current" in index
    assert "design-review-fixes-0913.md" in index
