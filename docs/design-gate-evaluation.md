# Gate adjudication (0.9.9)

Fake-model tests check that a gate fires. They do not show whether a fired gate withheld a relationship a reader would still call supported, or accepted one the reader would not. This note is the evaluation design for that question. No adjudicated study set is included. The JSON below is an example of the file shape, not a finding.

The gates are unchanged. `pattern_min_cases` stays 2. The axial token ceiling stays 24000.

## What an evaluation run exports

`export_gate_relationships(relationships)` reads each non-synthetic relationship and copies it into `accepted` or `withheld`. It does not re-run a gate and it does not edit the row.

- `accepted`: `disposition` is `accepted` or `verified`.
- `withheld`: `disposition` is `rejected`, `provisional`, or `merged`, or the row already carries `acceptance_withheld`.
- A row that is still `proposed`, with none of those dispositions, is left out of both lists.
- `accepted_downstream` (0.9.28) is the accepted rows whose `downstream_eligible` is true. An ineligible accepted row stays in `accepted` and is not an accepted downstream result.

0.9.28 copies `evidence_tier`, `reviewer_disposition`, `endpoint_review_status`, `downstream_eligible`, `withholding_reasons`, and `primary_withholding_cause` from `relationship_assessment`. The run JSON uses that same object. `primary_withholding_cause` is a reporting convention: substantive limits outrank a reviewer decision, the primary counts sum to the explained rows, and `withholding_reason_counts` overlaps. An explained row is disposition-withheld, `downstream_eligible` false, unreviewed at an endpoint, or still proposed with a known case or tier shortfall. A proposed row with enough cases and unknown eligibility is not explained. `reviewer_policy_rejection` marks a policy-only rejection of evidence below tier 2. A missing `downstream_eligible` is null, not true.

Each copied row has `relationship_id`, `disposition`, `rules`, and `withholding_flags`. `disposition` is the reviewer disposition. `downstream_eligible` is copied only when the source row stores a boolean. A missing field is not filled in as true. `withholding_flags` lists `causal_wording_withheld`, `acceptance_withheld`, `verification_withheld`, `relationship_rejected`, and `relationship_merged` when those flags are already on the row. An accepted row can still have `downstream_eligible` false. That split was added in 0.9.13. The rules are the labels already stored on the row:

| Rule | Where it is read |
| --- | --- |
| `stem_only` | `endpoint_anchoring.reason` `stem_only`, or flag `weak_stem_anchor` |
| `generic_stem_only` | `endpoint_anchoring.reason` `generic_stem_only` |
| `absence_reclassified` | a non-synthetic counterexample whose `reclassification_reason` or `reason` is `absence_is_not_contradiction` |
| `consequence_wording` | flags `wording_exceeds_tier`, `reviewer_causal_claim_mismatch`, `memo_wording_exceeds_tier`, `storyline_wording_exceeds_tier` |
| `role_evidence` | flag `role_evidence_exceeds_tier` |
| `acceptance_withheld` | flag `acceptance_withheld` |
| other | any other flag or anchor reason, under that flag or reason name |

An accepted row can still list a rule. A wording flag does not by itself move the row to `withheld`. 0.9.15 adds `role_evidence` for the flag `role_evidence_exceeds_tier`. `consequence_wording` still reads the sentence flags. The causal gate is not turned off.

## Adjudication record

`load_adjudications` reads a JSON list, or an object with an `adjudications` list. An optional `downstream_eligible` boolean and an optional `withholding_flags` list are kept when the file includes them. Scoring does not read those optional notes. It reads the export. Each object needs:

| Field | Meaning |
| --- | --- |
| `relationship_id` | The id on the exported row |
| `rules` | Optional list of rule names the person was looking at. Scoring uses the rules on the export, not this list |
| `verdict` | `supported` or `unsupported` |
| `adjudicator` | Who made the verdict. A non-empty string |
| `timestamp` | When it was made. A non-empty string. The loader does not check the clock |
| `reason` | Why, in the adjudicator's words. A non-empty string |

A missing field or an unknown verdict raises `GateEvaluationError`. The loader does not fill a verdict. Two records for one relationship id are reported on `duplicate_relationship_ids` and are not scored.

Example, not a study result:

```json
{
  "adjudications": [
    {
      "relationship_id": "rel-example",
      "rules": ["stem_only"],
      "verdict": "supported",
      "adjudicator": "example-adjudicator",
      "timestamp": "2026-01-01T00:00:00Z",
      "reason": "Example only. This is not an adjudicated finding."
    }
  ]
}
```

## Metrics

`score_gate_adjudications(exported, adjudications)` returns:

- `accept_precision`: adjudicated accepted rows whose verdict is `supported`, divided by adjudicated accepted rows. Null when no accepted row was adjudicated. This follows `disposition`.
- `withheld_supported_recall`: withheld rows whose verdict is `supported`, divided by every adjudicated row whose verdict is `supported`. Null when no adjudicated row is `supported`. This follows `disposition`.
- `downstream_precision`: supported verdicts among adjudicated rows whose export says `downstream_eligible` is true. This is the full-population rate. It stays null unless every adjudicated row has a boolean `downstream_eligible`, because an unknown row might be eligible. It is also null when no adjudicated row is eligible, and null when the export does not carry the field.
- `downstream_precision_known`: the same fraction among rows already known to be eligible. Null when that denominator is zero. `downstream_precision_known_denominator` is the count of adjudicated rows with `downstream_eligible` true. `downstream_precision_coverage` is adjudicated rows with known eligibility divided by adjudicated rows, and null when nothing was adjudicated.
- `downstream_withheld_supported_recall`: supported rows whose export says `downstream_eligible` is false, divided by every adjudicated supported row. This is the full-population rate. It stays null unless every adjudicated supported row has a boolean `downstream_eligible`. An unsupported row with known eligibility does not identify it, and it does not turn the rate into 0. Null when no adjudicated row is `supported`. A missing field is not treated as eligible and is not treated as withheld.
- `downstream_withheld_supported_recall_known`: the same fraction among supported rows whose eligibility is known. Null when that denominator is zero. `downstream_withheld_supported_recall_known_denominator` is the count of those rows. `downstream_withheld_supported_recall_coverage` is that count divided by every adjudicated supported row, and null when no adjudicated row is `supported`.
- `by_rule`: the same rates, denominators, and coverage for rows whose export lists that rule. A rule that fired on no exported row is omitted. A rule with no adjudicated rows keeps the rates null.
- Counts of adjudicated accepted, withheld, supported, downstream-eligible, and downstream-withheld rows sit beside the rates. A count can be zero. A rate cannot be stored as zero when its denominator is zero.
- `unmatched_adjudication_ids`: adjudication ids that are not in the export.

A 0.9.9 export that has `relationship_id`, `disposition`, and `rules` only still scores `accept_precision` and `withheld_supported_recall`. Its full-population and known-only downstream rates stay null. Coverage of supported rows is 0 when those rows were adjudicated and none of them carry `downstream_eligible`. Coverage is null when nothing was adjudicated.

These rates describe agreement with the adjudication file. They do not change `disposition`, a flag, or a tier.
