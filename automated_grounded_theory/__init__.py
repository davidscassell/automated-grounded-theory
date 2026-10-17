"""Automated initial (open) coding, then a theme report.

Codes are produced passage by passage from the corpus. The research context
sensitizes attention and is not a codebook. The theme report groups those
codes afterwards for presentation. It does not reread the passages. Focused
coding, axial coding, selective coding, and within-corpus theoretical sampling
are separate graphs. They are not the theme report. Saturation indicators do
not set ``judgment``. A saturation decision is a researcher record, or a
model judgment stored apart from that record. The package does not collect
new fieldwork data.
"""

from automated_grounded_theory.axial import (
    AxialError,
    AxialResult,
    aaxial_code,
    aselective_code,
    axial_code,
    read_axial_interrupt,
    relation_stability,
    selective_code,
)
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
from automated_grounded_theory.category_state import property_stability, proposal_stability
from automated_grounded_theory.decision_review import (
    ClaimDecision,
    ClaimDecisions,
    DecisionReviewer,
    SelectionDecision,
    apply_model_claim_decisions,
    claims_payload,
    selection_payload,
)
from automated_grounded_theory.gate_evaluation import (
    export_gate_relationships,
    load_adjudications,
    score_gate_adjudications,
)
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
from automated_grounded_theory.sampling import (
    SamplingError,
    SamplingResult,
    atheoretical_sample,
    read_sampling_interrupt,
    theoretical_sample,
)
from automated_grounded_theory.saturation import stopping_criteria_report, thematic_saturation_report
from automated_grounded_theory.segment import plan_coding
from automated_grounded_theory.stability import StabilityReport
from automated_grounded_theory.themes import ThemeError, ThemeResult, abuild_themes, build_themes
from automated_grounded_theory.version import __version__

__all__ = [
    "AxialError",
    "AxialResult",
    "ClaimDecision",
    "ClaimDecisions",
    "CodeRetriever",
    "CodingError",
    "DecisionReviewer",
    "EmbeddingCodeRetriever",
    "FocusedError",
    "FocusedResult",
    "LexicalCodeRetriever",
    "OpenCodeResult",
    "PresetCodesError",
    "ResearchContext",
    "SamplingError",
    "SamplingResult",
    "SelectionDecision",
    "StabilityReport",
    "ThemeError",
    "ThemeResult",
    "__version__",
    "aaxial_code",
    "abuild_themes",
    "aevaluate",
    "afocused_code",
    "aselective_code",
    "axial_code",
    "apply_model_claim_decisions",
    "claims_payload",
    "amulti_seed_stability",
    "aopen_code",
    "build_themes",
    "chen_metrics",
    "compare_runs",
    "evaluate_run",
    "export_gate_relationships",
    "load_adjudications",
    "score_gate_adjudications",
    "focused_code",
    "get_graph",
    "load_run",
    "read_axial_interrupt",
    "read_focused_interrupt",
    "read_sampling_interrupt",
    "relation_stability",
    "multi_seed_stability",
    "open_code",
    "plan_coding",
    "property_stability",
    "proposal_stability",
    "selective_code",
    "selection_payload",
    "stopping_criteria_report",
    "strict_sequential_settings",
    "thematic_saturation_report",
    "atheoretical_sample",
    "theoretical_sample",
]
