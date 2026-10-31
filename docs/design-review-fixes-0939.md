# 0.9.39 review fixes

Status: current for package 0.9.39. The 0.9.38 note stays the record of inherited core stops, frozen chunk identity for an unfingerprinted match, and compared seats. This note does not change `pattern_min_cases` (2), the axial token number (24000 per step), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `auto_size_core_review` stays off. `allow_partial_core_review` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added. No LLM judge is added. No live evaluation was run. No adjudications are added.

## What was going wrong

A resume that finished every opening match still reported the wrong reuse account. `core_matches_reused` counted every validated match already on the bracket at the start of the last step, including matches this run had just sent. `core_unfingerprinted_reused` was counted only while the fingerprints were missing, so after the first step stamped them the count became 0.

The per-match `reuse` label had the same problem. A later step rewrote a match sent this run to the same label as a match loaded from the export. The last match, which had no later step, had no label.

The closed core budget moved `starting_usage` and `starting_tokens` forward on every step. At the end those fields held the totals from before the last match, and `calls_made` was only that last match. The preflight's `calls_used` is still the figure before the match that step is about to send. It is not the closed total.

## Decisions

1. `core_matches_reused` counts matches this run loaded and did not send. `core_unfingerprinted_reused` counts how many of those had no payload or policy fingerprint when they were loaded. Both stay at those numbers on later steps. The rule is `core_reuse_accounting_rule` `loaded_unsent_and_final_usage_v1`.

2. Each match has `provenance`, and `reuse` carries the same value. `reused_export` means the match was loaded without fingerprints. `reused_checkpoint` means it was loaded with fingerprints. `sent_this_run` means this run sent it. The label is written for the loaded matches once per run, on the runtime rather than from a label saved by the previous run. A later step does not change it.

3. `metadata["core_usage"]` is the closed account: `calls` and `tokens_estimated` are the final totals, including inherited work. `prior_calls` and `prior_tokens_estimated` are what this run loaded. `this_run_calls` and `this_run_tokens_estimated` are what it sent. The core budget step keeps `starting_usage` and `starting_tokens` at the values from when the step opened, so `calls_made` and `tokens_used` are this run's spend rather than the last match only. `core_bracket_preflight` stays the before-send check.

4. When the outcome is `no_core_supported`, every candidate's `pattern_supported_centrality` is 0, and no relationship is graph-supported or `downstream_eligible`, the run sets `core_thin_upstream_evidence` and `core_upstream_evidence_note`. A missing centrality counts as zero. The outcome stays `no_core_supported`. When that coincidence does not hold, both fields stay null. The rule is `core_upstream_evidence_rule` `note_thin_axial_evidence_v1`.

5. Both new rules are part of the procedure fingerprint, so a 0.9.38 thread checkpoint raises `DecisionChunkingCheckpointError` before `allow_changes` is considered. They are not part of a match's policy fingerprint. A validated match is not dropped as `policy_changed` because these rules were added. `resume_core_selection` continues a frozen export whose chunk plan still matches. The chunking version stays `request_budget_v6`.

The synthetic fixture is 54 candidates in six opening matches, two loaded without fingerprints and four sent on the resume. It is not a live export. `pattern_min_cases` stays 2. No LLM judge is added. A metric that was not computed stays null.
