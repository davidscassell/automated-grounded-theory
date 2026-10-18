# 0.9.20 review fixes

`policy_version` is `0.9.20`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays a human `verify`. A metric that was not computed stays null. No live evaluation was run for this note. The 0.9.19 notes stay as history.

The frozen human-adjudicated discovery-recall set is not in this package. This note does not add a schema or a loader for it.

## What a participant quote states

Decision: quote validation is a deterministic rule. It does not call a model, so it does not spend a paid call. A human confirmation is not required for a quote the rule accepts, and this release does not collect one. `validated_by` is `deterministic_rule`.

Decision: an attribution keyword, shared vocabulary, and a located quote only nominate a candidate. They do not authorize tier 3 or `participants attribute X to Y`. The rule reads direction, polarity, and hedging. Tier 3 still needs a relevant rival check. A validated quote with no rival stays tier 2 and may set `attribution_source` to `participant`, which is the 0.9.1 rival gate.

Decision: the stated relationship direction is from the row's `from_id` to its `to_id`. An affirmative, unhedged quote that places the from-endpoint as the cause and the to-endpoint as the effect is `validated`. A denial stays below tier 3 and is flagged `participant_denies_link`. A reversed quote is flagged `participant_direction_reversed` and is not accepted for the stated direction. A hedge is flagged `participant_link_hedged`, stays below tier 3, and keeps the qualification on the claim. An undirected association (`related to` and the same family) is `direction_unspecified` and does not authorize the directed claim. A `because` that explains why the topics were discussed, with both endpoints on the same side of the connector, is `not_a_link`. If one quote would authorize tier 3 and another denies the link, the reading is `contested` and tier 3 is not authorized.

The quote text, passage, stated direction, polarity, hedging, status, qualifications, and `validated_by` are stored on `claims.participant_explanation` and on `participant_link`.

## Claim components

Decision: one row no longer switches between mutually exclusive evidentiary roles. It keeps three components.

- `pattern` is the co-occurrence. When both endpoints are anchored in at least `pattern_min_cases` cases, this component is downstream eligible and `supports_core` is true. `core_support_component` is `pattern`. A weak stem still blocks core support.
- `participant_explanation` records the quote reading. It never supports a core, including when it authorizes tier 3 wording.
- `analyst_interpretation` stays hypothesis-level. `downstream_eligible` and `supports_core` are false. It uses `analyst interprets X as related to Y`. It does not change the row wording and it does not remove the pattern's eligibility. A model memo still cannot create this component.

Decision: 0.9.19 made the analyst flag a row-level exclusion. 0.9.20 does not. A stored 0.9.19 row that has the flag and no `claims` still fails `graph_supported` until it is reassessed. Reassessment appends that exclusion to `claim_revision_history` with `policy_version` `0.9.19`, then derives eligibility from the current pattern.

Decision: claim flags are recomputed on every reassessment. A flag that the current evidence does not support is removed. The previous tier, attribution, wording, participant reading, and analyst presence stay on `claim_revision_history`. A second reassessment with the same evidence does not append again.

## Selective fingerprint

Decision: resume rejects a selective source whose fingerprint differs. `allow_changes` does not bypass it. The selective hash covers the axial evidence fingerprint plus relationships (evidence, dispositions, and interpretation fields), memos, properties, and dimensions. An unchanged source still resumes.

Decision: a 0.9.19 selective checkpoint hashed only the axial evidence. Resume cannot tell whether relationships, memos, properties, or dimensions changed, so that checkpoint does not continue. The error says those inputs were not fingerprinted and a new selective run is required. Axial resume still uses the evidence fingerprint only. The default seed and sampling `source_hash` stay on the older passage-and-code-id hash.

## Axial exploration

Decision: the planner reserves a minimum of one same-respondent slot and one lexical cross-case slot when that class still has unexamined pairs. Same-passage pairs take the remaining slots. The quotas are `exploration_quotas` on `axial_code`. The default is conservative and does not raise `max_axial_review_calls` or the token ceiling. When the reserved minimums are already filled by the old order, the selected pairs match the 0.9.19 planner. Seed 0 also matches that planner when a class is not rotated. Another seed rotates a class only when the class has more pending pairs than slots.

Decision: examined pairs are an argument to `axial_code` and an input to the planner. The plan stores them on the result metadata, so the checkpoint keeps them. `within_run_examined` is this round. `unique_pairs_explored` counts this round together with the history that was passed in.

Decision: the unexamined pair list is stored when the unordered pair count is 256 or fewer. Above that, the list is empty, `unexamined_materialized` is false, and `unexamined_pair_count` plus `selection_procedure` are the record. A positive count still makes saturation coverage partial. Classification intersects precomputed endpoint passage hits instead of scanning every passage for every pair.
