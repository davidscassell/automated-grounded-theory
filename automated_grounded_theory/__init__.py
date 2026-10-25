"""Open coding, a theme report, and separate later graphs.

Codes are produced passage by passage from the corpus. The research context
sensitizes attention and is not a codebook. The theme report groups those
codes afterwards for presentation. It does not reread the passages. Focused
coding, axial coding, selective coding, and within-corpus theoretical sampling
are separate graphs. Axial coding, selective coding, and theoretical sampling
are experimental. Each has a human checkpoint and an LLM-only decision mode.
They are not the theme report. Saturation indicators do not set ``judgment``.
A saturation decision is a researcher record, or a model judgment stored apart
from that record. The package does not collect new fieldwork data.
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
    AUTO_SIZE_DECISION_CALLS,
    CHUNK_ALLOWANCE_INSUFFICIENT,
    CHUNK_ALLOWANCE_RULE,
    CHUNK_REVIEW_RECOVERY,
    CONDITIONAL_SCOPE_RULE,
    CORE_CANDIDATE_RULE,
    DECISION_REVIEW_CHUNKING,
    ELIGIBILITY_RULE,
    FOCUS_ENDPOINT_RULE,
    INCOMPLETE_REVIEW_APPLICATION,
    LINK_DENIAL_RULE,
    OVERSIZED_RECOVERY,
    PAIR_COUNT_RULE,
    PATTERN_QUOTE_RULE,
    BUDGET_LEDGER_RULE,
    DUPLICATE_ASSERTION_RULE,
    EXPORT_ASSESSMENT_RULE,
    WITHHOLDING_CAUSE_RULE,
    ClaimDecision,
    ClaimDecisions,
    DecisionChunkingCheckpointError,
    DecisionReviewer,
    SelectionDecision,
    apply_model_claim_decisions,
    claims_payload,
    merge_core_nominations,
    merge_selection_ids,
    measure_chunk_context_sensitivity,
    pack_decision_chunks,
    selection_payload,
)
from automated_grounded_theory.discovery_evaluation import (
    DISCOVERY_SCHEMA_VERSION,
    DiscoveryEvaluationError,
    claim_statement_hash,
    freeze_discovery_set,
    load_discovery_file,
    score_discovery,
)
from automated_grounded_theory.gate_evaluation import (
    PRIMARY_WITHHOLDING_CAUSE,
    WITHHOLDING_CAUSES,
    export_gate_relationships,
    load_adjudications,
    primary_withholding_cause,
    relationship_assessment,
    score_gate_adjudications,
    stamp_relationship_assessment,
)
from automated_grounded_theory.repeatability import (
    compare_pipeline_runs,
    repeat_frozen_review,
    repeat_pipeline,
)
from automated_grounded_theory.focused import (
    FocusedError,
    FocusedResult,
    afocused_code,
    focused_code,
    load_focused_export,
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
from automated_grounded_theory.relationship_tiers import (
    score_attribution_accuracy,
    score_conditional_readings,
)
from automated_grounded_theory.version import __version__

__all__ = [
    "AxialError",
    "AxialResult",
    "ClaimDecision",
    "ClaimDecisions",
    "CodeRetriever",
    "CodingError",
    "DISCOVERY_SCHEMA_VERSION",
    "AUTO_SIZE_DECISION_CALLS",
    "CHUNK_ALLOWANCE_INSUFFICIENT",
    "CHUNK_ALLOWANCE_RULE",
    "CHUNK_REVIEW_RECOVERY",
    "BUDGET_LEDGER_RULE",
    "CONDITIONAL_SCOPE_RULE",
    "CORE_CANDIDATE_RULE",
    "DECISION_REVIEW_CHUNKING",
    "DUPLICATE_ASSERTION_RULE",
    "EXPORT_ASSESSMENT_RULE",
    "FOCUS_ENDPOINT_RULE",
    "ELIGIBILITY_RULE",
    "INCOMPLETE_REVIEW_APPLICATION",
    "LINK_DENIAL_RULE",
    "OVERSIZED_RECOVERY",
    "PAIR_COUNT_RULE",
    "PATTERN_QUOTE_RULE",
    "PRIMARY_WITHHOLDING_CAUSE",
    "WITHHOLDING_CAUSE_RULE",
    "WITHHOLDING_CAUSES",
    "DecisionChunkingCheckpointError",
    "DecisionReviewer",
    "DiscoveryEvaluationError",
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
    "merge_core_nominations",
    "measure_chunk_context_sensitivity",
    "merge_selection_ids",
    "pack_decision_chunks",
    "amulti_seed_stability",
    "aopen_code",
    "build_themes",
    "chen_metrics",
    "claim_statement_hash",
    "compare_pipeline_runs",
    "compare_runs",
    "evaluate_run",
    "export_gate_relationships",
    "freeze_discovery_set",
    "load_adjudications",
    "load_discovery_file",
    "score_attribution_accuracy",
    "score_conditional_readings",
    "score_discovery",
    "score_gate_adjudications",
    "stamp_relationship_assessment",
    "focused_code",
    "get_graph",
    "load_focused_export",
    "load_run",
    "primary_withholding_cause",
    "relationship_assessment",
    "repeat_frozen_review",
    "repeat_pipeline",
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
