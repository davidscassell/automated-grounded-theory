# Gate adjudication (0.9.9)

Fake-model tests check that a gate fires. They do not show whether a fired gate withheld a relationship a reader would still call supported, or accepted one the reader would not. This note is the evaluation design for that question. No adjudicated study set is included. The JSON below is an example of the file shape, not a finding.

The gates are unchanged. `pattern_min_cases` stays 2. The axial token ceiling stays 24000.

## What an evaluation run exports

`export_gate_relationships(relationships)` reads each non-synthetic relationship and copies it into `accepted` or `withheld`. It does not re-run a gate and it does not edit the row.

- `accepted`: `disposition` is `accepted` or `verified`.
- `withheld`: `disposition` is `rejected`, `provisional`, or `merged`, or the row already carries `acceptance_withheld`.
- A row that is still `proposed`, with none of those dispositions, is left out of both lists.

Each copied row has `relationship_id`, `disposition`, and `rules`. The rules are the labels already stored on the row:

| Rule | Where it is read |
| --- | --- |
| `stem_only` | `endpoint_anchoring.reason` `stem_only`, or flag `weak_stem_anchor` |
| `generic_stem_only` | `endpoint_anchoring.reason` `generic_stem_only` |
| `absence_reclassified` | a non-synthetic counterexample whose `reclassification_reason` or `reason` is `absence_is_not_contradiction` |
| `consequence_wording` | flags `wording_exceeds_tier`, `reviewer_causal_claim_mismatch`, `memo_wording_exceeds_tier`, `storyline_wording_exceeds_tier` |
| `acceptance_withheld` | flag `acceptance_withheld` |
| other | any other flag or anchor reason, under that flag or reason name |

An accepted row can still list a rule. A wording flag does not by itself move the row to `withheld`.

## Adjudication record

`load_adjudications` reads a JSON list, or an object with an `adjudications` list. Each object needs:

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

- `accept_precision`: adjudicated accepted rows whose verdict is `supported`, divided by adjudicated accepted rows. Null when no accepted row was adjudicated.
- `withheld_supported_recall`: withheld rows whose verdict is `supported`, divided by every adjudicated row whose verdict is `supported`. Null when no adjudicated row is `supported`.
- `by_rule`: the same two rates for rows whose export lists that rule. A rule that fired on no exported row is omitted. A rule with no adjudicated rows keeps both rates null.
- Counts of adjudicated accepted, withheld, and supported rows sit beside the rates. A count can be zero. A rate cannot be stored as zero when its denominator is zero.
- `unmatched_adjudication_ids`: adjudication ids that are not in the export.

These rates describe agreement with the adjudication file. They do not change `disposition`, a flag, or a tier.
