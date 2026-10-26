# 0.9.32 review fixes

Status: record of package 0.9.32. Package 0.9.38 is current. The 0.9.31 note stays the record of condition scope and consolidation accounting. This note does not change `pattern_min_cases` (2), the axial token number (24000), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added.

A decision review that reports every item reviewed, including a count such as 355 of 355, establishes processing coverage. A passing synthetic fixture does the same. Neither establishes that a selection is valid. Human-adjudicated examples are not in this package.

## Decision

`max_axial_review_tokens` stays 24000. From this version it is the token allowance of each decision step: relation review, saturation review, and core selection, in both the axial graph and the selective graph. It is not one pool those steps share. A run can spend one 24000 allowance on each step. Coding, counterexamples, memos, and the storyline keep a separate pool of the same number, and a decision step does not spend that pool. The number is not raised. Call ceilings are unchanged.

`decision_budget_steps` records, for each step, the scope, starting call usage, calls used, remaining calls, the token allowance, starting tokens, tokens required, tokens available, tokens used, remaining tokens, and the stop reason. The required-versus-available token figures are written before the step spends. An earlier step's tokens do not reduce a later step.

## Supporting cases

A case is the first present identity among `respondent_id`, `participant_id`, `case_id`, `respondent`, and `participant`, on the passage or on its metadata. A top-level value wins. Otherwise the case is the document. Several quotes from one person are one case. That count is made after a duplicate merge, and swapping relationship ids does not change it. One person cannot meet `pattern_min_cases` or raise a tier. Two respondents can.

## Reread

Before selection, the package may record a worst-case reread estimate that assumes every codebook code is raised. That figure is labeled and does not block a selection that fits its own call and token budget.

After selection is applied, the reread is estimated again from the raised codes and the actual queue. If that estimate exceeds `max_focused_review_tokens`, the selection stays and the reread does not start. The stop reason is `reread_allowance_insufficient`.

`token_budget` (default 6000) is the per-request cap for a reread as well as for other focused requests. A passage that alone exceeds it is not sent. The export records the estimate and `reread_request_exceeds_token_budget`. The default stays 6000.

Coverage reports `passages_covered` of `passages_in_corpus` and `documents_covered` of `documents_in_corpus`. `complete_for_queue` means the queue finished. `complete_for_queue_is_corpus_coverage` is false. It is not a claim that the corpus was reread.

## Assessment object

Each relationship row stores `relationship_assessment` under that name. The gate export does too. The same fields remain copied onto the row.

## Resume

The fingerprint stays `request_budget_v6`. It also requires `decision_token_scope` `per_step_v1`, `case_count_rule` `respondent_identity_v1`, and `reread_estimate_rule` `post_selection_v1`. A 0.9.31 checkpoint is missing those keys and raises `DecisionChunkingCheckpointError` before `allow_changes` is considered.
