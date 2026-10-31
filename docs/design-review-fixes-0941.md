# 0.9.41 review fixes

Status: current for package 0.9.41. The 0.9.40 note stays the record of upstream evidence, pair denominators, and the first relationship funnel. This note does not change core selection. It does not change `pattern_min_cases` (2), `one_document_one_respondent` (off), the axial token number (24000 per step), `batch_size` (1), the 6000 request budgets, the 8-call axial default, or `max_decision_calls` (4). `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. No paid call is added. No LLM judge is added. No adjudications are added. No live evaluation was run.

## What was going wrong

A live axial run on 0.9.40 planned the quote-copy retry on top of a 12/12 split, so 36 calls were planned against a 24-call allowance. `axial_token_plan.warning` was empty at a 168k shortfall. A counterexample request of about 9.8k tokens was dropped as `token_allowance` before it was shortened, and no omitted context or reallocation was recorded. Twelve proposals used about 42.6k of a 48k shared pool, so no counterexample ran. The planner ordered by evidence class first, so single-document same-passage pairs came before remaining multi-document pairs. Those unplanned multi-document pairs had no individual status. The row funnel reported 9→4→4→4→0 with `not_carried_forward` and said it reconciled, while the rows were 9 consolidated, 4 reviewed, and 5 unreviewed. A reviewed row that also failed the document count showed only `reviewer_rejected`. Relation review recorded 8 calls for 4 rows, and a 9th was dropped. The coding stage reported `tokens_used` 0.

## Decisions

1. Retries fit inside the call allowance. Default 8 calls at share 0.5 is 2 proposal requests, 2 reserved retries, and 4 counterexample calls. A 24-call half split is 6 requests, 6 retries, and 12 counterexample calls, not 36. A leftover single slot stays a proposal and does not reserve a retry. That non-reservation is recorded. The 8-call default and the half split stay.

2. Counterexample tokens are reserved before proposals spend the coding pool. The share follows the call split. The coding-plan shortfall is a warning. The ceiling is not raised. Decision steps are not in that estimate.

3. An oversized counterexample is shortened first, including when the first refusal is `token_allowance`. The claim, category definitions, relevant passage context, and provenance stay. If it still cannot be sent, `counterexample_unsent` names `request_budget` or `token_allowance`. Adequacy stays `unassessed`. Returned calls are an `allowance_transfers` row. A dollar allowance that is unset stays null.

4. Document spread outranks evidence class. Each class keeps its minimum. Every pair coded together in at least `pattern_min_cases` documents has a terminal status. `nothing_to_propose` is not `unplanned`, `budget_blocked`, or `failed`.

5. The funnel stages are relationship ids: consolidated, reviewed, eligible. Reasons stay independent: `reviewer_rejected`, `insufficient_document_count`, `hypothetical_only_evidence`, and `incomplete_review`. Document counts are documents. Reconciliation fails when a count, an id list, or a status disagrees. `not_carried_forward` is not added to force a pass.

6. Relation review is one disposition assessment of every relationship in the chunk. The code does not send a second assessment and does not send a schema repair on this step. `max_decision_calls` stays 4 and is not raised. A live count of 8 calls for 4 reviewed rows is not a required second pass, so it is not budgeted and the duplication is not added. Calls on this step equal chunks sent.

7. Coding-stage `tokens_used` is the coding-stage ledger. The token plan copies that figure after calls are sent.

The chunking fingerprint is unchanged, so a 0.9.39 core checkpoint can resume. These checks are engineering and evidence-traceability coverage. Analytic accuracy is unmeasured. Eligibility and reviewer standards are unchanged.
