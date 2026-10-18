# 0.9.14 review fixes

`policy_version` is `0.9.14`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. A metric that was not computed stays null. Human input still goes through `interrupt()` and `Command(resume=...)`. A sampling timeout resume is still `ainvoke(None)`. No live evaluation was run for this note.

## Downstream rates and unknown eligibility

0.9.13 divided `downstream_withheld_supported_recall` by every adjudicated supported row. A supported row with no boolean `downstream_eligible` stayed in that denominator and added nothing to the numerator, so the fraction was a lower bound stored as a point estimate. One supported blocked row and one supported unknown row scored 0.5. The true share is 0.5 if the unknown row is eligible and 1.0 if it is blocked. A second case scored 0.0 when no supported row had known eligibility, because some other adjudicated row did. Per-rule buckets used the same flag.

`downstream_withheld_supported_recall` and `downstream_precision` are now the full-population rates. Each stays null unless every row in its population has a boolean `downstream_eligible`. Recall's population is the adjudicated supported rows. An unsupported row does not identify that rate. Precision's population is every adjudicated row, because an unknown row might be eligible. That scope for precision is a decision.

`downstream_withheld_supported_recall_known` and `downstream_precision_known` are the fractions on the rows whose eligibility is already known. Each has a denominator count and a coverage. A known-only rate with a zero denominator stays null. Coverage is null when its own denominator is zero, and 0 when rows were adjudicated and none of them had known eligibility. The same fields are on `by_rule`. Disposition metrics keep their names. An export with no `downstream_eligible` field still scores the downstream rates as null.

## Sampling document minimum

`_window_ready` compared `n_documents` with the constant 3 and wrote that same constant on the withheld judgment. `min_documents` (default 3) was already the minimum on the stopping-criteria rate rows. A 3-document corpus, two sampling rounds, and `min_documents` 10 could store a fake-model `sufficient_for_this_question` as `recorded` while those rate rows were `undefined`, and the review call was spent.

The readiness check and the withheld record now use the configured `min_documents`. Below it, `model_saturation_judgment` is `not_interpretable`, `assessment` is null, `minimum_documents` is the configured value, and the saturation call is not sent. A human run still interrupts. The payload carries `minimum_documents`, `window_ready`, and `window_reason`, and the stored decision records the same minimum. `judgment` stays null.

`saturation_window` of `not_interpretable` still withholds the model call. That label is still computed from the constant 3 when the run has no saturation curve. A configured minimum below 3 does not by itself relabel the indicator window. That limit is a decision. Two sampling rounds remain the round minimum.

The gate note is [design-gate-evaluation.md](design-gate-evaluation.md). The sampling note is [design-sampling-0900.md](design-sampling-0900.md).
