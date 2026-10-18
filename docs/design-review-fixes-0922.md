# Review fixes 0.9.22

`policy_version` is `0.9.22`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays a human `verify`. A metric that was not computed stays null. No live or paid model call was run for this note. Matching stays deterministic. There is no LLM judge.

The 0.9.20 decisions stay as shipped: deterministic quote validation, pattern support for a core with the analyst component left as a hypothesis, a validated quote without a rival at tier 2, contradictory quotes read as `contested`, and no resume of a 0.9.19 selective checkpoint. The 0.9.21 discovery evaluation stays a separate scorer from gate evaluation. This note changes exploration accounting, the quote pattern, and how discovery scores are assigned.

## Completed examinations are the exclusion history

Decision: planning no longer writes the round into `examined_pairs`. That list is the default exclusion history. It contains completed examinations only, including a valid `nothing_to_propose` response with an empty relationship list.

Decision: a pair that was refused, omitted for the token allowance, left unparsed, or timed out is not completed. It stays out of `examined_pairs` and can be sent on a later run. `unique_pairs_explored` is the size of the completed history, including pairs the caller already passed in. It is not the planner's projection of history plus this round.

Decision: `within_run_examined` stays the planned round. `pairs_planned`, `pairs_attempted`, and `pairs_completed` are separate counts on the result and on `pair_coverage`. `pairs_planned` is the round length, not the size of the full pair space.

Decision: a timeout aborts the proposal node before that node publishes attempted and completed counts. The returned `pairs_attempted` is the last committed value. When the first request times out, that value is 0. The pair is still absent from `examined_pairs`, and a funded rerun can send it. This note does not add a second write of the in-flight pair during the timeout handler, because that handler does not know which request was in flight once the node has been discarded.

Decision: an `examined_pairs` list written by 0.9.20 or 0.9.21 was filled when the round was planned. It cannot be split into completed and unsent pairs. Passing that list to 0.9.22 still excludes every pair in it. `pair_accounting.completed` on those old runs is not a substitute: `examined_pairs` was filled even when `completed` stayed empty. To retry an unsent pair from an old run, omit it from the list passed in.

## The quote pattern is not general semantic reliability

Decision: tier 3 still requires a participant quote, a relevant rival, and a reading that authorizes the from-to link. The reading is a closed pattern, not a general parse of the sentence. One clause and one recognized connector have to be unambiguous. Anything else stays unvalidated and does not authorize tier 3.

Decision: the longest connector phrase at a position wins. Passive `caused by`, `led to by`, `lead to by`, and `leads to by` point backward. Active `caused`, `causes`, `cause`, and `led to` stay forward. `cause` does not match inside `because`.

Decision: negation that governs the claim clause denies the link. The recognized forms include `not what`, `are not`, `doesn't`, `never`, and `no relation`, plus the existing do-not family. `not only` is not a denial. `not sure` stays a hedge.

Decision: two claim-bearing clauses, or two connectors in one clause that disagree, are `ambiguous`. `if`, `would`, `someone said`, and the related hypothetical markers are `hypothetical`. A hypothetical is not tier 3 even when the connector itself would be from-to. An ordinary conditional that uses `if` is therefore unvalidated. That is the closed pattern, not a claim that every conditional sentence is unreal.

Decision: the auditable set is `tests/fixtures/attribution_adversarial_1.json`, fixture version `attribution-adversarial-1`. The quotes are synthetic. The expected outcomes are validated forward, validated reverse, denied, hedged, hypothetical, and ambiguous, plus the interviewer `not_a_link` and an undirected association. Validated reverse means the direction was recognized as `to_from` and was not authorized.

## Discovery assignment does not follow adjudication ids

Decision: independent gold is assigned by maximum-weight one-to-one matching. Full agreement is maximized first, then any structural match, then canonical content order. Adjudication ids and relationship ids are not tie-breaks. Renaming or reordering gold rows and proposals does not change the rates. The 0.9.21 greedy pass in adjudication-id order is gone.

Decision: a `system_proposal` row binds only by `system_relationship_id`. Duplicate ids in the discovery file raise `DiscoveryEvaluationError`. A missing id, an id that is not in the export, or two export rows with the same id are `unmatched_system_proposal` and are listed in `unmatched_system_relationship_ids`. They are not moved onto another proposal that shares endpoints and a passage. A bound row has `discovery_status` `linked`. It is outside structural recall, outside the precision numerator, and outside distortion. The bound proposal remains in the precision denominator because precision's population is still every non-synthetic proposal with both endpoint ids. Gate false negatives still use the bound row's eligibility, or `system_outcome` when the id did not bind.

Decision: structural recall compares endpoint ids, passage overlap, and optional direction, role, and wording. It does not read the proposed statement. A denying statement with matching metadata can score structural recall 1. Structural recall means those structural fields were recovered. It does not mean the human-supported interpretation was recovered.

Decision: `claim_judgment` records a human reading of that statement: `direction`, `qualification`, `support`, and `verdict` (`correct`, `incorrect`, `partly_correct`, `not_judged`). `claim_correct_rate` is correct over judged matched rows. `partly_correct` is judged and is not correct. `not_judged` is not judged. The rate is null when none of the matched rows were judged. Coverage is judged over matched and is null when nothing matched. The rate is never folded into structural recall.

Decision: the schema written by `freeze_discovery_set` is `discovery-eval-2`. A `discovery-eval-1` file still loads. Its hash is checked against the eval-1 body, which omits `claim_judgment`. The loaded object reports `schema_version` `discovery-eval-2` and `source_schema_version` `discovery-eval-1`, and the eval-1 hash is kept. Freezing again writes a new hash because the schema version and the claim objects enter the body, even when every verdict is `not_judged`. An eval-1 file that already carries a real claim judgment raises and has to be frozen as eval-2. The synthetic example file is still the eval-1 artifact and is not rewritten.

Decision: `discovery_recovered_per_reread_token` and `discovery_recovered_per_reread_dollar` replace the 0.9.21 names. The numerator is independent rows recovered. The denominators are `reread_expenditure_tokens` and `reread_expenditure_cost_usd` only. Other run spend is not included. Incomplete reread expenditure still leaves both rates null.

## Open: the planner still materializes the pair space

Decision, left open: `plan_relation_pairs` still builds every unordered pair and shuffles the less-obvious band in memory. The saved output already omits the unexamined list when the space is larger than 256 and keeps the count plus `selection_procedure`. An equivalent selection still needs that shuffle, or an index of the same size. No cheaper selection was added. A later change would have to show the same round for the same seed, examined set, and evidence classes.

## What this release does not change

`pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays a human `verify`. Temperature stays unset unless the caller set it. No live or paid call is part of this package's tests. Gate scoring is unchanged. The synthetic discovery example is not a study result.
