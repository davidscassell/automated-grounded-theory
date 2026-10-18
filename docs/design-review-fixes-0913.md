# 0.9.13 review fixes

`policy_version` is `0.9.13`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. A metric that was not computed stays null. Human input still goes through `interrupt()` and `Command(resume=...)`. A sampling timeout resume is still `ainvoke(None)`. No live evaluation was run for this note.

## Per-pair timeout recovery

0.9.12 cleared `reread_stop` only when the whole reread batch finished. A pair could time out, succeed on retry, and still sit in `reread_failed_pairs` when a later pair was withheld. `_finish_reread` returned on that outstanding refusal before it recorded the recovery, and `stop` still said `timeout`, so the current blocker was stored as `timeout` rather than `call_allowance`.

A pair leaves `reread_failed_pairs` when a note for that pair is stored. If the active stop is a timeout, the pair is appended to `reread_recovered_timeouts` and to `sampling_attempts` with status `recovered` and reason `timeout`. Pairs that still have no note stay failed. The active timeout stop is cleared only when no failed pair remains. The stage status then follows whatever is still unresolved. In the three-passage case the third pair's `call_allowance` is `reread_stop`. A pair is not both completed and failed. That split is a decision.

When one of two timeouts recovers and the other does not, only the unrecovered pair stays in `reread_failed_pairs`, and `reread_stop` stays `timeout`. A batch in which every timed-out pair succeeds still records the history and clears the active stop, so the quiet-round criterion can be met as in 0.9.12. An unresolved timeout, with `reread_status` or `reread_stop` still `timeout`, stays `not_evaluable`.

The history row's `reread_status` is `complete`, `skipped`, or `budget_limited` when the stage already has that status, and `recovered` otherwise. That label is a decision. It is history. It is not the active failure.

If an unresolved timeout pair and a later allowance refusal are both still open, `reread_stop` stays `timeout`. The allowance becomes the stop only after the timeout pair has been recovered. That precedence is a decision.

## Downstream withholding

`_tiered_accept` can set `disposition` to `accepted`, `downstream_eligible` to false, and `flags` to include `causal_wording_withheld`. The exporter used to classify that row as accepted from disposition alone, so a supported verdict scored `accept_precision` 1.0 and `withheld_supported_recall` 0.0 and did not show that the row was blocked from downstream use.

Each exported row now keeps `disposition` and, when the source row stores a boolean, `downstream_eligible`. `withholding_flags` copies `causal_wording_withheld`, `acceptance_withheld`, `verification_withheld`, `relationship_rejected`, and `relationship_merged` when those flags are already on the row. Those fields are separate. A wording flag still does not by itself move the row from `accepted` to `withheld`. That separation is a decision.

`downstream_precision` is supported verdicts among adjudicated rows whose export says `downstream_eligible` is true. `downstream_withheld_supported_recall` is supported rows whose export says `downstream_eligible` is false, divided by every adjudicated supported row. `accept_precision` and `withheld_supported_recall` keep their names and still follow disposition. A denominator of zero stays null.

An export that does not carry `downstream_eligible`, including a 0.9.9 export, scores both new rates as null. A missing field is not treated as eligible. A non-boolean value is also unknown. Those choices are a decision. `load_adjudications` keeps an optional `downstream_eligible` or `withholding_flags` on an adjudication record. Scoring reads the export, not those optional notes. That source is a decision.

The gate note is [design-gate-evaluation.md](design-gate-evaluation.md).
