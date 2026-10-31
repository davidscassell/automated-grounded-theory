# 0.9.40 review fixes

Status: current for package 0.9.40. The 0.9.39 note stays the record of loaded-match provenance and closed core usage. This note does not change core selection. It does not change `pattern_min_cases` (2), `one_document_one_respondent` (off), the axial token number (24000 per step), `batch_size` (1), the 6000 request budgets, or the 8-call axial default. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. No paid call is added. No LLM judge is added. No adjudications are added. No live evaluation was run.

## What was going wrong

Saved runs showed raised-code counts of 57, 55, and 54. The focused-selection requests were the same bytes. A few borderline decisions still flipped at temperature 0. That is model variation. The package did not hash the response or name the codes.

Axial planning sent 4 of 1,386 plausible pairs. Eight calls split in half leave 4 proposal calls at batch 1. The token shortfall was stored and not warned. Counterexample requests were about 10,000 tokens against the 6,000 cap and were dropped without returning the calls. The planner ranked pairs by evidence class and ignored how many documents the pair shared. A statement that copied its quote was rejected with no retry. Raised-code document reach and reread-queue coverage were not reported. The export did not carry a relationship funnel that reconciles.

## Decisions

1. `response_hash` is stored on a validated selection response. `compare_selection_replays` pairs logs by request hash and lists `flipped_code_ids`. A request that is not in both logs is not a flip.

2. When the round plans fewer pairs than `pairs_needed`, the run warns. The preflight records that number. The allowance and the split are unchanged.

3. Surplus passage text is shortened until the counterexample request fits. The proposed claim, category definitions, relevant passage context, and provenance stay. `omitted_context` records what was shortened. `counterexample_adequacy` is `unassessed`. Fitting the cap is not evidence that the test is still adequate. If the request still cannot be sent, `counterexample_reallocation` records the reason and the calls that fit the remaining token allowance, and the dollar allowance when one is set, return to proposals. An unset dollar allowance stays null and does not raise the token cap.

4. Within a class, pairs with a higher `document_cooccurrence` are planned first. A pair coded together in at least `pattern_min_cases` documents is a qualifying pair. That is planning priority, not a ceiling. One pair can support several distinct assertions. Coded co-occurrence does not set the corpus's total relationship potential. Two documents are not two respondents.

5. The preflight reserves one statement-copy retry per proposal request, so every budget calculation includes it. The retry counts against `max_axial_review_calls` and against the remaining token and dollar allowances. The rejected response and the retry lineage are kept. The retry asks for a grounded statement, or an explicit `nothing_to_propose` when no supported relationship is present. That outcome is `no_supported_relationship`. A second copy is still rejected.

6. `raised_code_document_reach` and `reread_queue_coverage` are reports. They do not change which codes are raised or which passages are queued. A missing raised list leaves the reach counts null.

7. `relationship_funnel` has a pair sequence and a row sequence. Each drop is the sum of that stage's reasons. Rows carry every applicable reason and evidence exposure. `reviewer_rejected` is a reviewer's no. `insufficient_case_coverage` is used only when a recorded case count is below `pattern_min_cases`. Every other eligibility failure is `deterministic_ineligible`. A failed row is not described as failing to reach two cases unless that count was recorded and is short. `one_document_one_respondent` stays off, so documents are not counted as respondents.

8. `pair_coverage.denominator_note` separates three counts. `candidate_universe` is every unordered pair among this round's codes. `plausible_pairs` is the evidence-class subset, recomputed for this round, and it is not cumulative. `unexamined_pairs_this_series` is the universe minus pairs already completed and minus this round. That count can be larger than `plausible_pairs` because pairs with no evidence class are still unexamined. A saved investigation had 1,386 plausible pairs and 1,423 unexamined pairs for that reason. `pairs_planned` is this round only.

9. `unique_pairs_attempted` and `unique_pairs_completed` are distinct pairs across rounds. A repeated request does not raise those counts.

10. `pair_exploration_share` defaults to 0.25 and is configurable on `axial_code`. It is on the procedure dict and is not part of the chunking fingerprint. The share keeps slots for rare, contrasting, or incompletely coded pairs. Multi-document-first ordering does not permanently exclude them. Evidence-class pairs fill those slots before adjacency pairs. A round of one pair rounds the share to zero slots, so a single planned pair stays the qualifying pair.

11. A response hash identifies a changed response. `analytic_consequences` says what a parsed difference does: a code raised only in one log would enter that run's raised set, focused reread, and axial pair pool, and would be absent from the other. A hash change with the same raised set is `response_changed_without_parsed_difference`. The comparison does not say which decision is correct.

`pair_exploration_share` is not a procedure-fingerprint key. The chunking fingerprint is unchanged, so a 0.9.39 core checkpoint can resume. These checks are engineering and evidence-traceability coverage. Analytic accuracy is unmeasured.
