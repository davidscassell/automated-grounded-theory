# 0.9.15 review fixes

`policy_version` is `0.9.15`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. A metric that was not computed stays null. Human input still goes through `interrupt()` and `Command(resume=...)`. A sampling timeout resume is still `ainvoke(None)`. No live evaluation was run for this note. The motivating observation, 54 of 65 same-passage proposals discarded as empty statements, is the reviewer's report of a v32 run. This package did not repeat that run.

## Statement contract

`AxialRelationshipDraft.statement` is a required string. `None` and whitespace fail validation with "A blank statement is not a relationship." `AxialProposalBatch.nothing_to_propose` true with an empty `relationships` list is still the explicit empty outcome. A batch that fails validation uses the existing single repair. The repair is logged, and grounding checks still apply to the repaired statement. A statement that is only the supplied quotes is discarded with reason `verbatim_quote_statement`. Quote text alone is not a relationship.

`proposal_diagnostics.rejection_reasons` counts discarded proposals by reason after a parsed batch, including an empty object when nothing was discarded. `schema_repairs` counts repair audits, including a repair that stays unparsed.

One blank row invalidates the whole batch rather than dropping only that row after a successful parse. That is a decision. A batch with both `nothing_to_propose` true and a non-empty list stays schema-valid; the loop treats the flag as the empty outcome only when the list is empty.

## Weak-stem eligibility

`downstream_eligible` is true only when `evidence_use` is `supported_evidence`. `relationship_evidence_use` is the shared rule.

`hypothesis_exploration` covers a tier below 2, a missing tier, a weak stem (`weak_stem_anchor`, `evidence_scope` `same_passage_stem_only`, or an endpoint reason `stem_only` or `generic_stem_only`), `acceptance_withheld`, and `evidence_status` `cross_respondent_pattern`. `withheld` covers a rejected or merged row, an empty statement, a synthetic row, `unanchored` or `disputed` evidence, `causal_wording_withheld`, `role_evidence_exceeds_tier`, and `wording_exceeds_tier` below tier 3.

Quote containment does not clear a weak stem. Export and source mapping recompute the rule, so a stored `downstream_eligible: true` on a weak stem does not ship. `graph_supported` requires tier 2 or higher and a usable row, and it returns false for a weak stem. It does not read `downstream_eligible`. A tier-2 row with `causal_wording_withheld` still graph-supports a core and is not downstream eligible.

Checked consumers: the core candidate is `core_unsupported_by_graph` when the only graph row is a weak stem, and `storyline_core_emphasis` is false. The memo prompt still includes that relationship, with `downstream_eligible` false and `evidence_use` `hypothesis_exploration`. The storyline graph snapshot still lists the relationship id. A tier-2 pattern without a weak stem exports as `supported_evidence` and `downstream_eligible` true.

## Role policy

The proposal prompt allows `unspecified` when the passages support only an association. It is not a missing-value flag. `consequence` and `causal_condition` are not assigned unless the passages support that role.

`statement_wording_overreach` reads the sentence and ignores the role. `role_evidence_overreach` is true for `consequence`, `cause`, or `causal_condition` below tier 3. The sentence flag remains `wording_exceeds_tier`. The role flag is `role_evidence_exceeds_tier`. Either one below tier 3 still sets `causal_wording_withheld`. `wording_overreach` with role `consequence` below tier 3 stays true, so the causal gate is not disabled. Gate export maps `role_evidence_exceeds_tier` to the rule `role_evidence`, apart from `consequence_wording`.

## Attrition and provenance

`metadata.attrition` records `unique_pairs`, `proposals`, `valid_statements`, `grounded_endpoints`, `evidence_tiers`, `accepted_rows`, and `downstream_eligible_rows`. `unique_pairs` is null when `pair_accounting.planned` is not a list. `evidence_tiers` is null when no row has an integer tier. `deduplicated_relationships` is null unless `comparison_runs` is present. The dedup key is `(from_id, to_id, whitespace-normalized casefolded statement)` across the comparison runs and the current run.

`metadata.provenance` records the commit, the imported module path, and the module and distribution versions. A missing git commit is null. `InstallationMismatch` is raised when the imported `automated_grounded_theory/__init__.py` is not the distribution's package, or when the distribution version differs from `__version__`. `generation_parameters.temperature` is null and `temperature_status` is `unset` unless the caller set a numeric temperature before the automatic determinism record. The model instance can still receive temperature 0 for resume checkpoints. That automatic value is not reported as the configured temperature.

## Repair visits

A paid axial or focused call returns before its repair, so the original response and its usage are already checkpointed. The repair is the next visit. A repair that is still over the request budget after the repair note is shortened is `schema_repair` with status `not_sent`, `received` false, and empty usage. The warning says the request was not sent. A repair the call allowance refuses is status `refused`, and the stage stays `incomplete` with reason `schema_unparsed`. The pair counts as attempted. It is not reclassified as `budget_limited`. Neither path calls the model again or copies the original usage onto the repair row.

## Timeouts

A stalled initial axial call is cut off by `axial_request_timeout`. The partial result is `incomplete`, `stop` is `timeout`, and `axial_run.json` is written when an output directory was set. Resume with `resume={"accept": True}` on the same checkpoint can finish, and the timeout row stays in the prompt log. A stalled repair keeps the failed parse on the call audit and raises `TimeoutError`. A later call with the stall cleared can return a statement.

`timeout_policy` separates two clocks. `run_timeout_seconds` is the elapsed wall clock already named `axial_request_timeout`, `focused_request_timeout`, or `sampling_request_timeout`. `idle_timeout_seconds` is an inactivity limit. Structured calls do not report progress, so `idle_timeout` is stored and `idle_timeout_enforced` is false. That limit is not enforced.

Cancellation is cooperative. The provider timeout, the graph-node timeout, and process isolation are different boundaries. The default node watchdog is the elapsed run timeout times the sequential calls in the step, plus two seconds, so a stalled provider hits the provider timeout first. `node_timeout` replaces that watchdog when the caller sets it. The default is unchanged. Abandoning the provider await, or cancelling the node, does not stop a blocking synchronous call such as `time.sleep`. `node_timeout_stops_remote_request` is false. `CallLimiter(process_isolation=True)` can terminate a picklable provider in a child process. Axial, focused, and sampling graphs leave process isolation off.

## Replay

A human resume sends `Command(resume=...)` on the same thread. The interrupted node starts again from its first line. Work before `interrupt()` is recomputed from the checkpointed input: stage notes, allowance copies, and candidate lists are derived again rather than appended. `interrupt()` is not wrapped in `except Exception`. The pause order is the graph order: axial relationships, then thematic saturation.

A node that raises before it returns is not an interrupt. The next invoke on that thread with `resume=None` retries the node from the last checkpoint. A timeout resume stays that same `None` invoke when the thread has no interrupt.

Human and model relation reviews both write `metadata.validated_decision`. The relationship rows share `relationship_id`, `disposition`, `paradigm_role`, and `needs_review`. `provenance.actor` is `human` or `llm`, and `provenance.channel` is `interrupt` or `model`.

## Persistence

`metadata.persistence` names the checkpointer and the durability mode. The default checkpointer is `InMemorySaver`. It drops every checkpoint when the process exits. `checkpoint_path` opens a SQLite saver. That file is still readable after the process exits, and a later process can resume the same thread without repeating a proposal that was already checkpointed. `durability` defaults to `async`, which is LangGraph's own default. `sync` and `exit` are available. This does not change the default.

## Pair records

`metadata.pair_records` holds one record per direction and passage. The record stores the question key, the direction id, the passage id, the current status, the result, and the attempt history. `reread_completed_pairs`, `reread_failed_pairs`, `reread_unsent_pairs`, `reread_status`, `reread_stop`, and `reread_stage_summary` are derived from those records. A completed pair is not also failed. Ledger `completed_pairs` stay question key plus passage id. The metadata list stays direction id plus passage id. This does not change which passages are reread.

## LangGraph floor

Node timeouts and `RetryPolicy` are passed into `add_node` on the axial, focused, and sampling graphs. Those arguments need LangGraph 1.2 or newer. The dependency floor is `langgraph>=1.2,<2`. The previous floor was `>=1.0,<2`. This is a declared raise, not a silent one. Schema validity, evidence eligibility, and token accounting stay application rules.

## Sampling report

This does not change which passages are reread or how a judgment is stored. `reread_interpretations` breaks non-relevant notes into `irrelevant`, `unjudged`, `unanchored`, and `failed`. An unanchored quote is `unanchored` even when `judgment` is `relevant`. `relevance` is null when nothing was classified. The same counts are grouped by direction kind and by document and respondent novelty. A missing id is not treated as new or known. `tokens_per_relevant_reread` is null unless every relevant note has attributable usage. `cost_per_relevant_reread_usd` is null unless `token_prices` records both per-million rates.

`SamplingResult.completed_pairs` is the list on `metadata.reread_completed_pairs`, or null when that key is absent.

The tier note is [design-relationship-tiers.md](design-relationship-tiers.md). The gate note is [design-gate-evaluation.md](design-gate-evaluation.md). The sampling note is [design-sampling-0900.md](design-sampling-0900.md).
