# 0.9.21 review fixes

`policy_version` is `0.9.21`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays a human `verify`. A metric that was not computed stays null. No live evaluation was run for this note. The 0.9.20 decisions stay as shipped: deterministic quote validation, pattern support for a core with the analyst component left as a hypothesis, a validated quote without a rival at tier 2, contradictory quotes read as `contested`, and no resume of a 0.9.19 selective checkpoint.

The discovery scorer is specified in [design-discovery-evaluation.md](design-discovery-evaluation.md). This note records the decisions. The example file is synthetic. It is not an adjudication of a corpus.

## Discovery file

Decision: discovery evaluation is a separate module from gate evaluation. Gate scoring still excludes an adjudication id that is not in the export. Discovery can count a relationship that was never proposed.

Decision: the frozen schema version is `discovery-eval-1`, independent of `policy_version`. JSON and JSONL are the frozen artifact. The CSV file is a header-only template for adjudicators. Pipe-separated passage ids and quotes share one CSV row, and a row that names `disagreement_adjudicator_id` is a disagreement rather than another gold row. A quote that contains `|` is written in JSON.

Decision: an independent evidence-supported row needs both code ids and a passage id. `unsupported_interpretation` clears `evidence_supported` and removes that row from recall. Over-merging and loss of a minority or contradictory account are counted and do not change the match status.

Decision: duplicate gold ids raise. Disagreement is stored beside the gold row and does not change recall or precision.

Decision: `freeze_discovery_set` hashes the normalized body with SHA-256. The hash does not include itself. Load checks a stored hash and raises when the body changed. Scoring refuses a file that has no hash. Passing no discovery file does not raise, and every rate is null. An empty frozen set leaves recall, distortion, and the gate rate null. Precision is 0 when that empty set is compared with one or more proposals, because the denominator is those proposals.

## Matching and rates

Decision: a match is an unordered pair of casefolded code ids plus a shared passage id. Labels, quote text, embeddings, and a model judge are not used. Direction is scored only when the gold row sets it, in the gold endpoint order. Role uses `paradigm_role` when gold role is set. Wording uses `allowed_wording` when gold wording is set. A tier label cannot repair a wrong direction.

Decision: one gold row takes one system row. Independent rows are matched first. A full agreement is preferred to a distortion, then the lowest relationship id. Distorted is not recovered and does lower recall. A missed independent row is not a gate false negative.

Decision: recall's denominator is independent evidence-supported rows. Precision's denominator is non-synthetic proposals with both endpoint ids. The distortion rate's denominator is recovered matches plus distorted matches. Each rate is null when its denominator is empty. `downstream_eligible` is not an input to those three rates.

Decision: the gate false-negative population is evidence-supported items the system proposed, either by a match or by `kind` `system_proposal`. Withheld means eligibility false, a withheld disposition, `repair_failed`, or a withheld `system_outcome`. The export boolean wins over `system_outcome`. `schema_unparsed` is not treated as a failed repair. The full rate stays null unless every population member has known eligibility, and the known rate and coverage are reported beside it, as in 0.9.14. A recovered row can also be a gate false negative.

Decision: recovered independent rows per token and per dollar are reported only when `reread_expenditure_complete` is true and the spend denominator is positive. Incomplete expenditure stays null even if a dollar figure is already filled in. A run state is scored through `reread_interpretation_report`. Human-adjudicated fields stay apart from model judgments and from tier labels.

## Left as shipped

Decision: this release does not change quote validation, claim components, tier numbers, the selective fingerprint, axial exploration, `pattern_min_cases`, the axial ceiling, or `batch_size`.
