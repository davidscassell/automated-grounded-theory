# 0.9.38 review fixes

Status: record of package 0.9.38. Package 0.9.42 is current. The 0.9.37 note stays the record of the separate core allowance, the bracket preflight, partial execution, the resource stop names, validated-match reuse, and bounded auto-size. This note does not change `pattern_min_cases` (2), the axial token number (24000 per step), `batch_size` (1), or the 6000 request budgets. `auto_size_decision_calls` stays off. `auto_size_core_review` stays off. `allow_partial_core_review` stays off. `selection_consolidation` stays off. Tier 4 stays human-only. Undefined metrics stay null. Human input still arrives through `interrupt()` and `Command(resume=...)`. No paid call is added. No LLM judge is added. No live evaluation was run. No adjudications are added.

## What was going wrong

A saved core checkpoint can already hold `core_pair_exceeds_request_budget` in `decision_chunk_state.stop_reasons`. That string was the plan-time label for a pair the packer could not split. On a later run the opening matches can each fit the 6000 cap, and the preflight can say so. Resume still loaded the old stop. `_drop_cleared_allowance_stops` removes call, token, and request stops only after `decision_stop` has already been cleared, and it does not remove `core_pair_exceeds_request_budget`. After the first new match the loop saw a stop reason and finished. A bracket of six opening matches therefore stopped after three.

The same resume, once that stop was cleared by hand, did not keep the two matches that had no payload fingerprint. The first trust built the fingerprint with `chunk_count` equal to the number of stored records and did not write that count back. The next match made the record list longer, the fingerprint no longer matched, and the earlier matches were sent again.

A call that was sent and did not return a decision, and the call that only schedules a richer `insufficient_evidence` retry, were added to `core_candidates_compared`. The recorded match then added the same seats again when the retry completed.

## Decisions

1. At the start of each core visit, stops this visit can derive again are cleared from the saved plan and from `decision_stop`. That set includes `core_pair_exceeds_request_budget`, `core_bracket_incomplete`, `core_request_too_large`, `core_call_allowance_insufficient`, `core_token_allowance_insufficient`, `call_allowance`, `token_allowance`, `request_budget`, `invalid_decision`, and `review_error`. The current plan adds `core_pair_exceeds_request_budget` back when a pair still cannot be split, and nothing is sent. The preflight adds a request, call, or token stop back when the remaining openings still do not fit. A failure during this visit adds `review_error` or `invalid_decision` again. A saved copy of those strings is not evidence that they still apply. The rule is `core_resume_stop_rule` `rederive_inherited_stops_v1`.

2. `core_resume_stop_rule` is part of the procedure fingerprint, so a 0.9.37 thread checkpoint raises `DecisionChunkingCheckpointError` before `allow_changes` is considered. It is not part of a match's policy fingerprint. Clearing an inherited stop does not change a decision that was already validated, so a match stamped under 0.9.37 is not dropped as `policy_changed`. `resume_core_selection` continues a frozen export whose chunk plan still matches. The chunking version stays `request_budget_v6`.

3. An export written before match fingerprints stores neither `chunk_index` nor `chunk_count`. The first trust freezes the index, the count, and the round role it used to build the payload fingerprint. A later match does not change those fields. The match is not sent again, and its seats are not counted a second time. `insufficient_evidence` and an incomplete match stay non-reusable.

4. `core_candidates_compared` counts seats in a match that recorded a substantive decision: a select, or an abstention of `no_suitable_candidate` or `insufficient_evidence`. A failed call, a parse failure, an invalid decision, and the scheduling of a richer retry do not add seats. A sent failure still counts against the call and token allowance. The blocked `insufficient_evidence` path, where the richer retry cannot be sent, records the unresolved match once.

The synthetic fixture uses 54 candidates in six opening matches. Each match's preflight estimate on this character heuristic is under the 6000 cap. That is the same stop and the same reuse failure as a live bracket whose opening matches were about 5595 to 5648 tokens. It is not that live export, and it is not a validity result. A low figure on this model is not a safe assumption for every provider.

`pattern_min_cases` stays 2. No LLM judge is added. A metric that was not computed stays null.
