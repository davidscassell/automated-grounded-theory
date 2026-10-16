# Review fixes (0.9.4)

This note records three corrections after the 0.9.3 review. It does not add a graph node. The graphs stay the ones in [design-axial-0800.md](design-axial-0800.md) and [design-sampling-0900.md](design-sampling-0900.md). Tiers stay the rule in [design-relationship-tiers.md](design-relationship-tiers.md). Respondent classification stays the rule in [design-review-fixes-093.md](design-review-fixes-093.md) except where a quotation's passage is now stored separately from the selected candidate.

Nothing here is quoted from a source that was not already used in those notes. Strauss and Corbin (1990, 1998), Charmaz (2006), Glaser (1978, 1992), and Glaser and Strauss on constant comparison are the sources those notes already name. This note does not add a page citation.

## Sampling pairs

A reread is one `(direction_id, passage_id)` pair. The passage id is not resolved to a direction afterwards.

A human resume assigns passages on the direction row, through `interrupt()` and `Command(resume=...)`. Those assignments are the selection. A retrieval proposal that listed the same passage under another direction does not move the reread. Accepting a direction without naming a passage does not sample the passages that retrieval proposed for it.

A passage named on two accepted or edited directions is two rereads. Each prompt carries that direction's sampling question and target. Each note stores that direction. Each reread spends one coding call. When the allowance has one call left, the second pair is not sent. `reread_unsent_pairs` names its direction and its passage, and the stage is `budget_limited`.

`decision_mode="llm"` may return `selections`, a list of those pairs. An empty list still parses, so an older response that only sent `direction_ids` and `passage_ids` is accepted. The pairs are then derived: a passage proposed under two selected directions becomes two pairs. A passage that no selected proposal contains is paired only when exactly one direction was selected. Otherwise it is flagged on `model_passages_without_direction` and is not given to the first proposal. An explicit `selections` list is used as written and is not expanded by the proposal order.

`researcher_sampling.pairs` and `model_sampling_pairs` keep the pairs. The flat id lists stay.

## Quote passage

`passage_id` on a counterexample is still the candidate that was selected. `quote_passage_id` is the passage whose text contains the quotation. `quote_start` and `quote_end` are character offsets in that passage. They are null when the quotation is not anchored. They are not zero.

The selected passage is searched first, then the complementary passages. A quotation in both is attributed to the selected passage. A quotation only in a complementary passage stays `valid_test` true and records that complementary id. `complementary_passage_ids` is unchanged.

The same fields are written on endpoint anchoring and on the quote locations used for `evidence_scope`. The cited `passage_id` stays the relationship's cited passage. `quote_passage_id` is where the quotation was found. A quote in another respondent's passage still sets `cross_respondent_pattern` when the located quotations sit in more than one document. Supporting-case quotes already stored the located passage; they now also store `quote_passage_id` and the offsets. The storyline passage list includes `quote_passage_id` as well as the selected passage.

A record that has no `quote_passage_id` is still readable. Readers treat the missing field as the selected `passage_id`. Focused coding already attributes a quote found in another request passage through `located_passage_id` and `evidence_outside_passage`. That path is unchanged.

## Graph steps

Pairs whose category has no assigned passage are recorded once, before the proposal loop, as `missing_category_evidence` and omitted. They are not one graph return per batch. A window that still has to be deferred or split can return to the node. The derived step allowance is `propose_calls` plus `planned_pair_count` plus 24 steps of overhead, unless the caller passes `recursion_limit`. `planned_pair_count` is the unordered pair count. It is a planning figure stored beside `propose_calls`. It is not a new analytic setting, and `batch_size` stays a performance setting at its default of 1.

A mixed corpus can therefore finish the pairs that have segment evidence without an explicit step limit. Pairs that were omitted for missing evidence stay in the remaining count, so the proposal stage stays `budget_limited` with reason `missing_category_evidence`. They are not dropped and they are not reported as a finished search.

If a caller still sets `recursion_limit` below the work, the run returns the partial result. The stage is `incomplete` with reason `recursion_limit`. Usage, the prompt log, and the records so far are kept.
