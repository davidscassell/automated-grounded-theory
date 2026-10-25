# Review fixes 0.9.24

`policy_version` is `0.9.24`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `axial_token_budget` stays 6000. Focused `decision_token_budget` stays 6000. `batch_size` stays 1. Tier 4 stays a human `verify`. A metric that was not computed stays null. No live or paid model call was run for this note. The axial planner still builds the full pair space in memory. That limit stays open.

A live LLM-only evaluation on 0.9.23 sent no focused selection call on a TA 1% sample or on three Muse documents. The first selection request held every code, its definition, and two quotes. It was checked against the decision allowance and the per-request budget before any call. TA stopped on `decision_request_budget` at the 6000 default. Muse stopped on `decision_token_allowance` with a payload of about 57600 tokens. Raising the allowance did not help Muse, because the request was still one payload. Axial decision review and core selection stopped the same way in five of six runs. One TA run with a 24000 decision budget finished them.

## Decision

Decision: focused selection, axial relationship review, and selective core selection are packed into requests that each fit the per-request budget. That budget is `decision_token_budget` for focused selection and `axial_token_budget` for axial and selective review. `batch_size` is the coding, memo, and counterexample pack size. It is not the review chunk size, and it stays 1.

Decision: packing is deterministic. Focused codes are ordered by `code_id` before packing. Relationships are ordered by `relationship_id`. Core candidates stay in the evidence order from the candidate score: graph support, then pattern-supported centrality, accepted centrality, proposed centrality, and `code_id`. Codebook insertion order does not change the chunks. Chunk index is not a merge key.

Decision: a prefix that fits stays in one request. One code, relationship, or candidate whose own request exceeds the per-request budget is omitted and packing continues. The omission reasons are `single_code_exceeds_request_budget`, `single_relationship_exceeds_request_budget`, and `single_candidate_exceeds_request_budget`. That omission does not by itself set `decision_stop` and does not stop the stage.

Decision: focused selection merges the raised ids from validated chunks. The union is returned in code-id order. An id outside its chunk invalidates that chunk. Unreviewed codes are not raised. If at least one chunk validates, that partial set is applied and the reread can continue. `selection_review_complete` is false, and the run status is `incomplete`. A later ceiling sets `decision_stop`. Omissions without a ceiling use stop `decision_selection_truncated`. If every code is individually over the request budget, the deterministic proposal is kept, `decision_stop` is not set, and `selection_review_fallback` is `proposal_unreviewed_over_request_budget`. The selection is still not complete. If a ceiling or `invalid_decision` happens before any validated chunk, nothing is applied, `decision_stop` is set, and the reread is skipped.

Decision: an invalid selection, relationship, or core chunk stops the remaining chunks. Validated chunks already stored are still applied when any exist.

Decision: relationship decisions from validated chunks are applied together by the existing tier gate. A model `verified` disposition still cannot set tier 4. It is flagged `model_cannot_verify`. The model still cannot authorize attribution wording. Core nominations are merged after the chunks. A graph-supported nomination beats any nomination that is not graph-supported. Among the eligible nominations the order is pattern-supported centrality, accepted centrality, proposed centrality, then `code_id`. A truncated core still stores `model_core_id` and sets `core_selection_truncated`. An unknown core id is kept only when the review was one chunk and nothing was omitted, which is the previous single-request behavior.

Decision: claim review packing is unchanged. One claim larger than the request budget still stops claim review. Saturation review stays one call.

Decision: `max_decision_calls` is not raised. Each sent chunk spends one decision call. A review that needs more chunks than the call ceiling, or more estimated tokens than the allowance, is truncated and is not complete. Relationship review and saturation share the axial decision-call counter. A fresh `selective_code` starts that counter at 0.

Decision: the chunking fingerprint is `request_budget_v1`, stored as `procedure.decision_review_chunking` on focused manifests and on axial and selective procedures. A checkpoint saved before that field existed fails resume with `DecisionChunkingCheckpointError`, including when `allow_changes` is true. An open chunk plan whose version, kind, or plan hash does not match the rebuilt plan fails the same way. The hash covers the chunking version, kind, per-request budget, token ceiling, call ceiling, ordered ids, chunks, and omissions. A finished review is not re-entered.

Decision: each review report is `metadata.decision_reviews[kind]`. It records `chunk_count`, `chunks_sent`, `chunks_validated`, `item_count`, `items_reviewed`, `items_omitted`, `omissions`, `calls`, and `sized_by`. `stop_reasons` is null when there are none. `tokens_estimated` and `expenditure_complete` are null when no chunk was sent. `expenditure_complete` is false when a sent chunk has no provider input-token count. `complete` is true only when every item was in a validated chunk, nothing was omitted, no chunk was invalid, and there is no stop reason. A truncated selection is not a complete run.

Decision: in LLM mode the axial and selective recursion limit grows by `max_decision_calls`, one visit per chunk the call ceiling can send. Human mode adds nothing. An explicit `recursion_limit` still wins. The focused recursion default is unchanged.

Decision: the estimate used while packing includes a wide chunk index, so the smaller index on the sent request still fits the budget that packing accepted.
