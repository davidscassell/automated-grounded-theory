# Review fixes 0.9.23

`policy_version` is `0.9.23`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays a human `verify`. A metric that was not computed stays null. No live or paid model call was run for this note. Matching stays deterministic. There is no LLM judge.

## Proposal reviews and structural recall

Decision: independent discovery matching and `system_proposal` binding are separate tasks. `score_discovery` does not reserve a bound proposal before the independent assignment. The same proposal is a candidate for both. A review that names the proposal does not turn a recovered independent row into `missed`, and it does not change `discovery_recall` or `discovery_precision`.

Decision: the gate false-negative population is still one member per evidence-supported adjudication that the system proposed, including a `system_proposal` row. Adding a review can change `gate_false_negative_rate`. That is not structural recall.

Decision: a linked proposal is not also listed as `unmatched_proposal`. It stays in the precision denominator. It enters the precision numerator only when independent gold structurally recovers it.

## Assignment

Decision: every gold item may stay unmatched. The unmatched option weighs 0. A missing structural edge is negative and is never preferred to unmatched. 0.9.22 built a square matrix with no unmatched column when the gold count equalled the proposal count, and a large negative missing edge made two distorted matches beat one full recovery.

Decision: the objective is lexicographic. First maximize full-agreement recoveries. Then maximize structural matches. Then break ties by canonical content order of the gold row and the proposal: code ids, passage ids, direction, role, wording, rationale, and statement. `adjudication_id` and `relationship_id` are not part of that order. Renaming or reordering those ids does not change the pairing.

Decision: one full recovery beats any number of distorted matches. The recovery weight inside a component is one more than the number of gold rows in that component. A distortion weighs 1. Unmatched weighs 0.

Decision: only structural edges are in the matrix. Connected components of those edges are solved independently, once each. Isolated proposals stay in the precision denominator and are not given matrix columns. The content bonus is smaller than one primary unit. For a component of at most 12 structural edges the bonus is geometric, so the content order is exact. A larger component keeps the first two objectives and uses a rank bonus that is not a full ordering of every tied set.

## Claim reviews

Decision: independent discovery gold and run-specific claim reviews are separate. The schema written by `freeze_discovery_set` is `discovery-eval-3`. `claim_reviews` is the bound list. A real `claim_judgment` on a new adjudication raises. Flat CSV claim columns are not a bound review.

Decision: each review stores `review_id`, `run_id`, `relationship_id`, the statement, direction, qualification, support, verdict, `statement_hash`, and `binding_status`. `claim_statement_hash` is SHA-256 of canonical JSON for the whitespace-collapsed statement, the casefolded direction, and the casefolded qualification. A duplicate `review_id` raises. A duplicate bound `(run_id, relationship_id)` raises. A supplied `statement_hash` that does not match those three fields raises.

Decision: `score_discovery` takes `run_id`. When it is omitted and `proposals` is a mapping, `proposals["run_id"]` is used. A review is fresh only when the relationship id matches, the run id equals the scored run, and the hash recomputed from the proposal statement, the proposal direction (`stated_direction`, else `direction`), and the review's qualification equals the stored hash. Qualification is taken from the review both times because a proposal has no qualification field. A changed statement, a changed direction, a missing run id, or a different run is stale. The effective verdict is `not_judged`. Stale is not judged and is not carried forward as correct.

Decision: `claim_correct_rate` is correct over judged fresh reviews. The population is unique relationship ids that are an independent structural match or an id-bound `system_proposal`. The same relationship counts once. `partly_correct` is judged and is not correct. `claim_correct_denominator` is the judged count. `claim_matched_count` is the population. `claim_correct_coverage` is judged over matched. `claim_stale_count` counts stale reviews on that population and is not folded into the rate. The rate is null when nothing was judged. Rows report `claim_binding` as `fresh`, `stale`, or `unjudged`.

Decision: the `discovery-eval-1` hash is unchanged. It omits `claim_judgment` and has no `claim_reviews`. The `discovery-eval-2` hash is unchanged. It uses the schema string `discovery-eval-2`, includes `claim_judgment`, and has no `claim_reviews` key. After a successful check the in-memory `schema_version` is `discovery-eval-3` and `source_schema_version` stays the file's version. A source other than `discovery-eval-3` does not score a claim as correct. A verdict other than `not_judged` is stale. `not_judged` on those files is unjudged, not stale. Freezing an old file rewrites it to `discovery-eval-3`, drops the unbound judgment, and writes a new hash even when every verdict was `not_judged`. The dropped judgment is not copied into `claim_reviews`.

## Attribution

Decision: tier numbers are unchanged. A marker of negation, epistemic uncertainty, or existential negation in the clause text before the first direction-bearing connector sets the reading to `unsupported`. It does not authorize tier 3. The marker may sit anywhere in that prefix. A marker that appears only after the connector does not by itself fail the clause. A clause with no connector is left to the existing denial rules.

Decision: the check is not a short list of example sentences. The closed pattern includes `no evidence`, `not clear`, `unclear`, `doubt`, `there is no`, `nothing to suggest`, `nothing that`, `I don't know`, `I do not know`, `no sign`, `no indication`, `no proof`, `no reason to`, `not established`, `not obvious`, `cannot tell`, `can't tell`, `no data`, `no support`, `there are no`, `no doubt`, `without evidence`, `lack of evidence`, `uncertain`, and `hard to say whether`, among the same class. Direct `do not lead to` still matches the denial pattern when the epistemic pattern does not, and stays `denied`. Neither status authorizes tier 3.

Decision: no affirmative exception is recognized. `there is no doubt that` and `I have no doubt` fail closed. They are not read as participant-supported attribution.

Decision: `tests/fixtures/attribution_adversarial_2.json` (`attribution-adversarial-2`) is synthetic regression coverage for these patterns, including the quote `There is no evidence that tools for the shift cause coworkers on the crew` and other constructions phrased differently. Passing the fixture is not general semantic accuracy. `attribution-adversarial-1` still stands. `score_attribution_accuracy(cases)` scores only the cases the caller passes, through `apply_evidence_tier` on a synthetic tools/crew corpus. It does not load the packaged fixture. `None` or an empty list leaves `accuracy` and `correct_count` null, with denominator 0. No model is called.

## What stays open

Decision: `plan_relation_pairs` still builds the full pair space and shuffles the less-obvious band. Saved output still omits the unexamined list above 256 pairs. That quadratic memory limit is open. This release does not change the planner.

Decision: 0.9.22 examined-pair accounting is unchanged. A 0.9.20 or 0.9.21 `examined_pairs` list was written at plan time and still excludes every pair in it. A timeout still leaves the pair retryable. `nothing_to_propose` is still a completed examination.
