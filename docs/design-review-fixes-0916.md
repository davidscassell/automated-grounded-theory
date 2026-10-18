# 0.9.16 review fixes

`policy_version` is `0.9.16`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. A metric that was not computed stays null. No live evaluation was run for this note. The 0.9.15 notes stay as history.

## Editable install

`installation_record` compares the imported module version with the distribution version first. A difference raises `InstallationMismatch` and names both versions.

An editable install may store metadata in site-packages while Python imports the checkout. That layout is valid when `direct_url.json` has `dir_info.editable` and a `file:` URL that contains the imported file, or, when that file is absent, when an editable finder's mapping contains the imported file. Metadata that names a different tree is a mismatch even if a finder matches the import. A regular install still has to sit beside its metadata.

`layout` is `editable` or `distribution`. `axial_code` and `selective_code` call this check after argument validation and before the graph, so a mismatch does not spend a model call. Result construction and export still call it.

## Sampling step limit

The default recursion limit counts visits, not a call ceiling plus a fixed overhead.

- Preparation: load, rank, retrieve, the selection checkpoint, and the ledger (5).
- The checkpoint can loop once for a selection repair (1).
- Each coding call can arm a pair, send the request, and repair it (3 per `max_sampling_review_calls`).
- One later visit withholds every pair still pending after that ceiling (1).
- Finalization: update the ledger, decide saturation, and export (3).

Saturation repairs stay inside that node. A caller can still pass `recursion_limit`. The recorded value is `metadata.performance.recursion_limit`.

## Validation flags

`effective_statement` is `human_statement` when that sentence is non-blank, otherwise `statement`. `model_statement` is not rechecked. `sync_validation_flags` rebuilds `wording_exceeds_tier`, `role_evidence_exceeds_tier`, `causal_wording_withheld`, `hypothesis_not_supported`, and `cross_case` from the effective statement, the effective role, and the tier. Other flags stay. When that set changes, the previous flags and the statement and role they used are appended to `validation_revision_history`. A second call with the same inputs does not append again. `needs_review` is set on a current overreach and is not cleared here.

## Effective role

Decision: a reviewer's role overrides the working role until a human sets one. A human role wins. Until then, `reviewer_paradigm_role` is the role validation, eligibility, export, and `validated_decision` read. `paradigm_role` is set to that working role when no human role is present. `proposal_paradigm_role`, `model_paradigm_role`, `reviewer_paradigm_role`, and `human_paradigm_role` stay as provenance. The alternative, keeping the reviewer role as a proposal until a human accepts it, was not taken: the recorded review was already an acceptance, and validation was ignoring the role it stored.

## Mixed proposal batches

Decision: keep every schema-valid relationship and drop the invalid ones. Each dropped row is logged with its index, reason, error, and draft on `proposal_diagnostics.dropped_relationships`, and the reason is counted in `rejection_reasons`. A batch with at least one valid row does not spend a repair. A repair replaces the whole response, so it can discard rows that were already usable if the repair fails or the allowance is gone.

A payload that is not a JSON object with a relationship list, and a batch in which every row is invalid, still use the single repair from 0.9.15. `nothing_to_propose` with an empty list stays valid.

## Reread expenditure

`tokens_per_relevant_reread` still counts only attributable tokens from successful relevant responses. It stays null when that usage cannot be tied to every relevant note.

`reread_expenditure_tokens` sums sampling reread calls and `SamplingRereadNote` repairs, including irrelevant responses, failed parses, and repairs. Selection repairs are not included. Zero spend is 0. `expenditure_tokens_per_relevant_reread` divides that sum by `relevant_reread_count`. The rate is null when the count is zero, and the count is reported. The same split applies to cost, which stays null unless both token prices are recorded.

`human_confirmed_useful_rereads` is null. `human_confirmed_useful_rereads_status` is `not_computed`. Model judgments do not fill it.

## Timeouts

Decision: enforcing `idle_timeout` is not feasible, and the behavior is unchanged. Structured calls do not report token or byte progress, so there is no inactivity clock to reset. `idle_timeout_enforced` stays false. `idle_timeout_enforcement` is `not_feasible`. `timeout_return_stops_remote_request` is false.

Returning after a timeout means this process stopped waiting. It does not prove that the provider request stopped. Cancelling the node or abandoning the provider await still does not stop a blocking synchronous call. Process isolation remains the boundary that can, and these graphs leave it off.
