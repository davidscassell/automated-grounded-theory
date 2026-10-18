# Discovery evaluation

`policy_version` is `0.9.22`. This note is the discovery scorer. It is not [design-gate-evaluation.md](design-gate-evaluation.md).

`score_gate_adjudications` scores a verdict only when its relationship id is already in the accepted or withheld export. An id that is absent is `unmatched` and is excluded. That scorer cannot lower recall for a relationship the run never proposed. Discovery starts from a frozen file of relationships a person identified, including rows the export does not contain.

No live model call is used. Matching does not use embeddings or an LLM judge. A metric that was not computed stays null. The JSON file under `examples/` is synthetic example data only. It is not a study result, and this package does not ship a real adjudication.

## File

Decision: the schema version written by `freeze_discovery_set` is `discovery-eval-2`. It is not `policy_version`. A frozen file can outlive a package bump. `load_discovery_file` accepts `discovery-eval-1` and `discovery-eval-2` and rejects any other schema version.

Decision: a `discovery-eval-1` file still loads. Its stored hash is checked against the eval-1 body, which omits `claim_judgment`. After that check the in-memory `schema_version` is `discovery-eval-2`, `source_schema_version` stays `discovery-eval-1`, and every claim verdict is `not_judged`. Freezing that object writes `discovery-eval-2` and a new hash, even when every verdict is `not_judged`, because the schema version and the claim objects are part of the body. An eval-1 file that already contains `claim_judgment`, or a claim verdict other than `not_judged`, raises and must be frozen as `discovery-eval-2`.

The frozen artifact is JSON or JSONL. CSV is an adjudicator template, `examples/discovery_adjudication_template.csv`. A CSV loads with `frozen_hash_status` `absent`. `score_discovery` refuses it until `freeze_discovery_set` has written a hash. `score_discovery(proposals, None)` does not raise. Every rate on that call is null.

An adjudication has `adjudication_id`, `kind` (`independent` or `system_proposal`), `adjudicator_id`, and `rationale`. Endpoints are `from_id` and `to_id`. `direction` is `from_to`, `to_from`, `undirected`, or omitted. Omitted and `unspecified` are the same: direction is not scored. Optional `role` and `wording` are scored only when the adjudicator set them. `supporting_passages` holds `passage_id` and `quote`. `conceptual_distinctions` is a list of labels. The flags are `unsupported_interpretation`, `over_merging`, `loss_of_minority_account`, and `loss_of_contradictory_account`. `evidence_supported` defaults to true. `system_relationship_id` and `system_outcome` (`rejected`, `withheld`, `repair_failed`, `proposed`, `accepted`, or empty) record a system item the adjudicator reviewed.

Decision: an independent row with `evidence_supported` true needs both code ids and at least one passage id. The loader rejects it otherwise.

Decision: `unsupported_interpretation` forces `evidence_supported` false. That row is not in the recall denominator. The count is still reported. Over-merging and the two loss-of-account flags are counts only. They do not change recovered, missed, or distorted.

Decision: a duplicate `adjudication_id` in the gold list raises. Disagreement is a separate list: `adjudication_id`, `adjudicator_id`, `field`, `value`, `rationale`. Scoring uses the primary row only. A disagreement does not change recall or precision. The result reports `adjudicator_disagreement_count`.

Decision: on the CSV template, several passages share one row as pipe-separated `passage_id` and `quote` lists of equal length. A quote that contains `|` belongs in JSON, not in that column. `conceptual_distinctions` uses the same separator. A row with `disagreement_adjudicator_id` is a disagreement, not a second gold row. The template file is a header only.

JSONL may start with one header object whose `record_type` is `header`. That header carries `schema_version`, `label`, `note`, `frozen_hash`, and any disagreements. Later lines are adjudications. A line with `record_type` `disagreement` is a disagreement.

## Freeze

Decision: `freeze_discovery_set` writes `frozen_hash`, a SHA-256 of the canonical JSON body. The body is `schema_version`, `label`, `note`, adjudications sorted by `adjudication_id`, and disagreements sorted by adjudication id, adjudicator id, field, value, and rationale. Passage lists are sorted by passage id and quote. Flags are booleans. `frozen_hash` is not part of the body. Encoding is UTF-8, keys are sorted, and separators are compact. Quote text is inside the hash even though matching ignores it, so an edit to the recorded evidence is a different set.

`load_discovery_file` recomputes the hash when one is stored. A mismatch raises `DiscoveryEvaluationError` and the message includes `frozen_hash`. A missing hash loads with `frozen_hash_status` `absent`. `score_discovery` raises on that status. An empty frozen list loads. Recall, distortion, and the gate rate are null because those denominators are empty. Precision is also null when the run proposed no scoreable relationship. Precision is 0 when the frozen set supports nothing and the run still proposed one or more relationships: that 0 has a denominator.

## Matching

Decision: endpoint identity is a code id. Matching casefolds it and collapses whitespace. The pair is unordered. Labels are not matched. An empty id does not match.

Decision: the passage anchor is the intersection of passage ids. The system ids are read from `passage_id`, `passage_ids`, `supporting_passage_ids`, `supporting_cases`, `case_quotes`, `quotes`, `supporting_passages`, and `endpoint_anchoring`. Quote text is stored for the reader and is not a match signal.

Decision: a candidate match needs both the same code-id pair and a non-empty passage-id overlap. The same endpoints on a different passage are not a match.

Decision: direction, role, and wording are scored only when the gold row sets them. `stated_direction`, or else `direction`, is in the system's endpoint order. Swapped endpoints flip `from_to` and `to_from`. When the system names no direction, the same order is `from_to` and the swapped order is `to_from`. Gold `undirected` against a directional system row is a distortion. Role is compared with `paradigm_role`, then `role`, after casefold, and only when gold role is set. Wording is compared with `allowed_wording` after casefold and whitespace collapse, and only when gold wording is set. The free-text `statement` is not compared.

Decision: recovered means the endpoint and passage match and every scored field agrees. Distorted means that match and at least one scored field disagrees. Distorted is not recovered. It stays in the recall denominator and out of the numerator. Missed means an evidence-supported independent row with no candidate. An unmatched proposal is a non-synthetic system row with both endpoint ids that was assigned to no gold row.

Decision: independent gold uses one global one-to-one assignment. The primary objective maximizes full-agreement recoveries, then any structural match (including distortions). Ties break by canonical content order of the gold row and the proposal: code ids, passage ids, direction, role, wording, rationale, and statement. `adjudication_id` and `relationship_id` are not part of that order, so renaming or reordering those ids does not change the rates. 0.9.21 assigned greedily in adjudication-id order. That order is not used.

Decision: a `system_proposal` adjudication binds only by `system_relationship_id`. It is not endpoint-matched. Duplicate `system_relationship_id` values in the discovery file raise. A missing id, an id absent from the export, or more than one export row with that id is `unmatched_system_proposal` and is listed in `unmatched_system_relationship_ids`. It is never attached to a different proposal that shares endpoints. A successful bind is `discovery_status` `linked`. That row is not recovered, not distorted, and not an independent recall gold. The bound proposal is removed from the independent pool and is not an unmatched proposal. It stays in the precision denominator because it is still a non-synthetic proposal, and it is not a structural true positive.

Decision: structural matching does not read the proposed `statement`. A statement that denies the gold relationship can still be structurally recovered. That is not a claim that the human-supported interpretation was recovered. `claim_correct_rate` is a separate human measure.

Decision: `evidence_tier`, `evidence_label`, model judgments, and `attribution_source` are not read. A tier 3 label does not turn a wrong direction into a recovery. The result says `metric_source` `human_adjudication`, `model_judgments_used` false, and `tier_labels_used` false. Those fields are not tier labels and they are not model judgments.

## Rates

Recall denominator: independent adjudications with `evidence_supported` true. `system_proposal` rows are not in that denominator. `discovery_recall` is independent rows recovered, over that count. Null when the count is 0 or when no discovery file was passed. `discovery_recall_denominator` is the count. `discovery_recall_coverage` is how many of those rows have both code ids and a passage id, over the same count. The loader already requires those fields, so coverage is 1 when the denominator is positive, and null when it is 0.

Precision denominator: non-synthetic system rows with both endpoint ids. `discovery_precision` is recovered matches, over that count. Distorted matches and unmatched proposals stay in the denominator and out of the numerator. Null when that count is 0 or when no discovery file was passed. A real 0 is 0. `discovery_precision_coverage` is how many of those rows carry a passage id. A row with no passage id cannot match. Coverage is null when the precision denominator is 0.

`discovery_distortion_rate` is distorted matches over recovered matches plus distorted matches. Null when that sum is 0. `discovery_distortion_denominator` is the sum. These three rates do not read `downstream_eligible`.

## Gate false negatives

Decision: a gate false negative is an evidence-supported item the system proposed and then withheld. Proposed here means the row matched a system relationship, or the adjudication `kind` is `system_proposal`. An independent row that never matched is a discovery miss, not a gate false negative.

Withheld means the export boolean `downstream_eligible` is false, or `disposition` is `rejected`, `provisional`, `merged`, or `withheld`, or the export flags include `repair_failed`, or `system_outcome` is `rejected`, `withheld`, or `repair_failed`. `schema_unparsed` is not `repair_failed`. When the export has a boolean `downstream_eligible`, that boolean wins over `system_outcome`. Disposition and `repair_failed` are export facts and still count as withheld.

Decision: eligibility is known when the export has a boolean `downstream_eligible`. Otherwise an explicit `system_outcome` makes it known: `rejected`, `withheld`, and `repair_failed` are not eligible; `accepted` and `proposed` are not withheld by that field. Anything else is unknown.

`gate_false_negative_rate` is withheld-but-supported over the population, and only when every population member has known eligibility. Otherwise it is null. `withheld_but_supported_count` can still be positive. `gate_false_negative_rate_known` uses only the known members. Its denominator is `gate_false_negative_rate_known_denominator`. `gate_false_negative_rate_coverage` is known members over the population, and is null when the population is 0. This is the 0.9.14 split. A recovered match with `downstream_eligible` false is both recovered and a gate false negative. The two statuses are separate.

Per relationship, `discovery_status` is `recovered`, `missed`, `distorted`, `linked`, `unmatched_system_proposal`, `unmatched_proposal`, or `not_scored`. `gate_status` is `withheld_but_supported`, `passed`, `unknown`, or `not_applicable`.

## Claim correctness

Decision: `claim_judgment` is a human reading of the actual proposed statement. It has `verdict` (`correct`, `incorrect`, `partly_correct`, or `not_judged`), plus optional `direction`, `qualification`, and `support`. The default is `not_judged`. Structural recall does not read it.

`claim_correct_rate` is `correct` over judged matched rows. Judged means `correct`, `incorrect`, or `partly_correct`. `partly_correct` is judged and is not correct. `not_judged` is not judged. The denominator `claim_correct_denominator` is that judged count. `claim_matched_count` is independent structural matches, recovered or distorted. `claim_correct_coverage` is judged over matched, and is null when nothing matched. The rate is null when nothing was judged. It is never added to structural recall, precision, or distortion.

## Expenditure

`discovery_recovered_per_reread_token` is independent rows recovered, divided by `reread_expenditure_tokens`. `discovery_recovered_per_reread_dollar` divides the same numerator by `reread_expenditure_cost_usd`. Both names are scoped to reread expenditure. Other run spend is not in the denominator. Both are null unless `reread_expenditure_complete` is true and the denominator is a positive number. An incomplete report stays null even when it already carries a token count or a dollar figure. Zero recoveries over a positive complete reread spend is 0. The 0.9.21 keys `discovery_recovered_per_token` and `discovery_recovered_per_dollar` are not emitted.

Decision: a mapping with `prompt_log` or `reread_notes` is passed to `reread_interpretation_report`. A mapping that already has `reread_expenditure_complete` and is not a run state is used as given. Omitted expenditure leaves both rates null.

## What this release does not change

`pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays a human `verify`. The 0.9.20 claim components and selective fingerprint stay as shipped. 0.9.22 changes which axial pairs count as examined, and it tightens the quote pattern that can authorize tier 3. Those decisions are in [design-review-fixes-0922.md](design-review-fixes-0922.md). This module does not rescore tiers. Structural recall is recovery of the specified structural fields. It is not recovery of the human-supported interpretation.
