# 0.9.11 review fixes

`policy_version` is `0.9.11`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. A metric that was not computed stays null. Human input still goes through `interrupt()` and `Command(resume=...)`. No live evaluation was run for this note.

## Proposal diagnostics

`candidate_cross_case_passages` still records every passage that shares fewer than two distinctive stems with either endpoint. That complete list is appended to `metadata["lexical_only_withheld"]` and is written to `axial_run.json`.

The proposal request does not carry that list. It carries `lexical_only_withheld_summary`: a count, `support_cap` 1, `lexical_only` true, and a note that the full list is exported. That is a decision. The passages are not evidence, and the list grows with the corpus. The default `axial_token_budget` is still 6000.

When a request is over that budget, `lexical_only_withheld` and `lexical_only_withheld_summary` are removed before `_trim_surplus_passages` shortens passage evidence. The removed keys are recorded on `omitted_diagnostics.dropped_before_evidence`. A one-hundred weak-match corpus with one passage that contains both categories still sends the proposal call. The export still lists every weak passage.

## Quiet rounds

`sampling_rounds_without_new_directions` is evaluable only when `reread_unsent_pairs` is present and every selected pair was completed or explicitly exempted. `completed_pair` is completed. A `direction_key`, `question_key`, or `direction_id` in `closed_direction_keys` is exempt for every failure reason. A closed direction withheld for `request_budget` can still leave the criterion met. That preserves the 0.9.10 rule for `call_allowance`.

Any other reason is not silence. `request_budget`, `token_allowance`, `schema_unparsed`, `parse_failed`, `timeout`, `incomplete`, `missing_passage`, and a reason the ledger does not name all leave the criterion `not_evaluable` with `observed` null. `missing_passage` and an unnamed reason are blocking. That is a decision. `reread_failed_pairs` are read the same way. An `reread_status` or `reread_stop` of `timeout` is unresolved when no pair is listed, and it stays unresolved when a listed pair was not exempted. The status string stays `not_evaluable`. The reason stays `selected_pairs_unsent_or_unrecorded`.

Three successive sampling rounds that each withhold the same long passage for `request_budget` do not meet the quiet criterion.

## Pair classification

`classify_pair_evidence` tokenizes each passage once, builds endpoint terms once per code, and indexes tokens to passage ids. A pair's lexical candidates are the passages that contain a stem from each endpoint. Occupied cases are still skipped, and the first grounded overlap in passage order still wins. The class, the lexical-only count, the reaching count, and the single-document count match the 0.9.10 nested loop.

Classification walks the full pair space, so it is not run when `propose_calls` is 0. `pair_evidence` then records `classification_skipped`, reason `no_proposal_requests`, and null counts. `pattern_reach_warning` is a product of classification. It is not emitted when classification was skipped. That is a decision. A scheduled proposal still classifies, and a corpus where no pair can reach `pattern_min_cases` still warns. The minimum is not changed.

## Cross-case passage order

A passage is added to `seen` only after both endpoints share at least `_GROUNDED_STEMS` (2) distinctive stems. A one-stem overlap is appended to `lexical_only` and does not reserve the passage, so a later pair in the same batch can still offer it. The offered passage and its code ids do not depend on pair order.

`lexical_only` rows stay per pair. A passage that has already qualified for an earlier pair is not also listed as `lexical_only` for a later pair, so that diagnostic list can still depend on order. A qualifying passage is still selected at most once. When `limit` is tight, pair order can still decide which qualifying pair receives a shared passage. Those two points are decisions. The bug fixed here is the weak match that hid a later grounded match.
