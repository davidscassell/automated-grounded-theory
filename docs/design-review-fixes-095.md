# Review fixes (0.9.5)

This note records two corrections after the 0.9.4 review. The sampling graph now returns to `pursuit_reread` once per pair, and to `sampling_checkpoint` when a model selection needs a repair. Axial and selective graphs are unchanged. Tiers stay the rule in [design-relationship-tiers.md](design-relationship-tiers.md). Pairs stay the rule in [design-review-fixes-094.md](design-review-fixes-094.md) except where a completed reread is now stored before the next request, and except where an explicit `selections` list is no longer filtered by the flat id lists.

Nothing here is quoted from a source that was not already used in those notes. Strauss and Corbin (1990, 1998), Charmaz (2006), Glaser (1978, 1992), and Glaser and Strauss on constant comparison are the sources those notes already name. This note does not add a page citation.

## One reread, then a checkpoint

0.9.4 sent every selected pair inside one `pursuit_reread` visit. The notes and the prompt rows stayed in that visit until it returned. A timeout on a later pair left the checkpoint at the previous node, so a response that had already come back was discarded. The call allowance had already counted both attempts.

0.9.5 follows the same shape as axial proposals in 0.8.4. One visit sends one `(direction_id, passage_id)` pair and returns. The conditional edge comes back while another pair has not been stored and has not been withheld. The note, its usage, and its prompt row are in the checkpoint before the next request starts.

A timeout writes three lists on the partial result:

- `reread_completed_pairs` are the pairs that already have a note.
- `reread_failed_pairs` is the pair whose request timed out, when that request had acquired a call.
- `reread_unattempted_pairs` are the pairs that had not been sent.

The stage is `incomplete` and `stop` is `timeout`. The partial result is resumable. Resume does not use `Command(resume=...)` for this pause, because the graph is not inside `interrupt()`. It continues the node that did not return. A human pause still uses `interrupt()` and `Command(resume=...)`. An empty resume is still an error.

Resume restores `provider_budget` before the next request. A pair that already has a note is not sent again. The failed pair is sent again when a coding call remains, and then the unattempted pairs are sent. If no coding call remains, those pairs are withheld, `stop` stays `timeout`, and the stage stays `incomplete`. It is not reported complete. When resume does store every remaining pair, `stop` is cleared and `timeout_recovered` records that a timeout happened.

A selection repair has the same shape. The first selection call does not repair inside that call. A response that does not parse is written, with its usage, and the visit returns with `selection_repair_pending`. The next visit sends only the repair. A timeout on the repair leaves the first response in the checkpoint. Resume sends the repair again and does not send the first request again. The consumed review calls stay consumed.

`batch_size` stays a performance setting and stays 1 by default. The derived step allowance is still the planned review calls, the planned decision calls, and 24 steps of overhead, unless the caller passes `recursion_limit`. One step per reread, and one extra visit for a repair, sit inside that allowance. The allowance is not a new analytic setting.

## Explicit selections

`selections` is the pairing when the response names any pair. Each pair is checked against the direction ids in this round and the passage ids in the corpus. A pair that names an unknown direction or an unknown passage is stored on `model_rejected_selections` with the reason. It is not dropped without a record. Pairs that pass are the rereads.

`model_direction_ids` and `model_passage_ids` are derived from those pairs. Empty flat lists mean the response did not send them. That is not a mismatch. A non-empty flat list whose set differs from the derived ids is `model_flat_list_mismatch`. The selections still decide which pairs are reread. The mismatch does not.

When `selections` is empty, the 0.9.4 derivation stays: a passage proposed under two selected directions becomes two pairs, and a passage that no selected proposal contains is paired only when one direction was selected. Otherwise it is `model_passages_without_direction`.

If a requested pair is rejected, a flat list disagrees, a passage cannot be paired, or an id is unknown, `model_sampling_status` is `incomplete` and the stage is not `complete`. A warning is added only in that case. A passage that is in the corpus but outside the retrieval list is still kept. It is recorded on `model_passages_outside_proposal` and does not, by itself, make the stage incomplete.
