# 0.9.40 review fixes

Status: current for package 0.9.40. The 0.9.39 note stays the record of loaded-match provenance and closed core usage. This note does not change core selection. It does not change `pattern_min_cases` (2), `one_document_one_respondent` (off), the axial token number (24000 per step), `batch_size` (1), the 6000 request budgets, or the 8-call axial default. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. No paid call is added. No LLM judge is added. No adjudications are added. No live evaluation was run.

## What was going wrong

Saved runs showed raised-code counts of 57, 55, and 54. The focused-selection requests were the same bytes. A few borderline decisions still flipped at temperature 0. That is model variation. The package did not hash the response or name the codes.

Axial planning sent 4 of 1,386 plausible pairs. Eight calls split in half leave 4 proposal calls at batch 1. The token shortfall was stored and not warned. Counterexample requests were about 10,000 tokens against the 6,000 cap and were dropped without returning the calls. The planner ranked pairs by evidence class and ignored how many documents the pair shared. A statement that copied its quote was rejected with no retry. Raised-code document reach and reread-queue coverage were not reported. The export did not carry a relationship funnel that reconciles.

## Decisions

1. `response_hash` is stored on a validated selection response. `compare_selection_replays` pairs logs by request hash and lists `flipped_code_ids`. A request that is not in both logs is not a flip.

2. When the round plans fewer pairs than `pairs_needed`, the run warns. The preflight records that number. The allowance and the split are unchanged.

3. Counterexample passage text and quotes are shortened until the request fits. If it still does not, `counterexample_reallocation` records the reason and the calls return to proposals.

4. Within a class, pairs with a higher `document_cooccurrence` are planned first. The threshold the investigation used is `pattern_min_cases` documents. Two documents are not two respondents.

5. One statement-copy retry uses the existing proposal call. It counts against `max_axial_review_calls`. A second copy is still rejected.

6. `raised_code_document_reach` and `reread_queue_coverage` are reports. They do not change which codes are raised or which passages are queued. A missing raised list leaves the reach counts null.

7. `relationship_funnel` has a pair sequence and a row sequence. Each drop is the sum of that stage's reasons. Rows carry every applicable reason and evidence exposure. `one_document_one_respondent` stays off, so documents are not counted as respondents.

The chunking fingerprint is unchanged, so a 0.9.39 core checkpoint can resume. These checks are engineering and evidence-traceability coverage. Analytic accuracy is unmeasured.
