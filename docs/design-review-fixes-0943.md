# 0.9.43 review fixes

Status: current for package 0.9.43. The 0.9.42 note stays the record of the slim counterexample request and proposal lineage. This note does not change core selection, reviewer standards, `pattern_min_cases` (2), `one_document_one_respondent` (off), the 6,000 request cap, or the 24,000 token ceiling. No LLM judge is added. Undefined metrics stay null. No Muse or TA data is in this package.

A live axial run on frozen v46 input spent about $0.115 and 26 calls. Four multi-document pairs, each in two documents, stayed unplanned with no reason while two of six proposal slots went to single-document pairs. No calls were transferred to those pairs. Spare retries went to counterexamples. All nine counterexamples targeted relationships the reviewer had already rejected. The usage summary omitted six core calls (29,243 input tokens) and still said complete. The pair funnel counted six proposal requests and not the seventh call, which was a retry. Four of five raw proposals for one pair were invalid because the statement repeated the quote, and the retry did not recover. Kept rows carried `quotes not in cited passage`. Review-history entries had `stage` null.

## Decisions

1. Examining every multi-document pair precedes a single-document pair and the counterexample allowance. Counterexample calls and unused quote-copy retries move onto proposals until those pairs fit or `max_axial_review_calls` is exhausted. The ceiling is not raised. A pair that still does not fit is `budget_blocked` with that reason. The terminal statuses are `proposed`, `nothing_to_propose`, `failed`, and `budget_blocked`. An unplanned row without a reason is closed as `budget_blocked`. `pair_exploration_share` fills only slots that remain after every qualifying pair is seated or the round is full.

2. Counterexamples are not applicable to a row the reviewer has already rejected. The rejection stands. The call is not spent and does not reopen the row. The export records that targeting decision. Counterexample calls are also not spent while a multi-document pair is still unexamined.

3. `call_usage` includes every stage in the prompt log and the allowance ledger, including `axial_core_review`. `complete` is true only when `usage_completeness` is `complete`. A ledger call that is not in the log makes the summary partial. A missing provider count stays null.

4. `pair_funnel.proposal_calls` counts proposal requests and statement-copy retries. Pair stages stay pair counts, so a retry does not look like a dropped pair. Each relationship row lists that pair's retry outcomes.

5. A statement that only repeats its quote is `invalid_restated_quote` on the raw proposal. The retry tells the model to write words that remain after the quote is removed, or to set `nothing_to_propose`. A restated quote is still invalid. `retry_still_invalid` is the outcome when the retry copies the quote again.

6. `quotes_not_in_cited_passage` is not removed. `quote_scope_cause` is `quote_found_in_other_passage` when the quote is verbatim in a different passage, and `quote_absent_from_corpus` when it is not a verbatim span of any passage. Curly quotes and dashes are folded before that check, so a punctuation-only difference is not flagged.

7. A model relation review writes `stage` `axial_relation_review` on the review-history entry.

These checks are engineering and evidence-traceability coverage. Analytic accuracy is unmeasured.
