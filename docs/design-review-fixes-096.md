# Review fixes (0.9.6)

This note records three corrections after the 0.9.5 review. The sampling graph does not gain a node. `pursuit_reread` returns once to record the pair, once to send it, and once more when that response needs a repair. Axial and selective graphs are unchanged. Tiers stay the rule in [design-relationship-tiers.md](design-relationship-tiers.md). Pairs stay the rule in [design-review-fixes-094.md](design-review-fixes-094.md) and [design-review-fixes-095.md](design-review-fixes-095.md) except where a reread repair is now its own visit and an omitted flat list is no longer a mismatch.

Nothing here is quoted from a source that was not already used in those notes. Strauss and Corbin (1990, 1998), Charmaz (2006), Glaser (1978, 1992), and Glaser and Strauss on constant comparison are the sources those notes already name. This note does not add a page citation.

## The reread repair is its own visit

0.9.5 made each `(direction_id, passage_id)` pair its own graph step, and it split a selection repair the same way. The reread repair still ran inside the call that sent the original request. A timeout on that repair discarded the original response and its usage, and a resume had no pending repair to send.

The original request is now sent with repair deferred. When the response does not parse, the visit returns. The prompt row, the usage, and `reread_repair_pending` are in the checkpoint, with `reread_active_pair` still naming that pair. The next visit sends only the repair. A timeout on the repair leaves the original response in the audit trail. Resume continues `pursuit_reread`, sends the repair, and does not send the original request again. The calls already consumed stay consumed.

If the repair itself does not parse, the stage is `incomplete`. The original response remains. If no coding call remains, the repair is not sent and the stage stays `incomplete`. It is not reported complete.

## The active pair

The visit before a provider call writes `reread_active_pair` and returns without calling the model. The next visit sends that pair. A timeout reads the active pair from the checkpoint and stores it on `reread_failed_pairs`.

`reread_unsent_pairs` are pairs that were not sent: the request did not fit, the passage is missing, or the call allowance was already spent. Those rows stay withheld. They are not copied into `reread_failed_pairs` and they are not listed as unattempted. `reread_unattempted_pairs` are pairs that had not been started. A withheld row may carry `reason`. A row from before this field has no `reason`. It is still a withheld pair.

A checkpoint from 0.9.5 has no `reread_active_pair`. A timeout on that older shape still treats the first pair that is neither stored nor withheld as the in-flight pair. Missing keys stay readable.

## Flat lists, one at a time

Explicit `selections` stay the rereads. `model_direction_ids` and `model_passage_ids` are still derived from the pairs that name a known direction and a corpus passage.

An empty `direction_ids` or `passage_ids` list was not sent. It is not compared. The derived list fills it. A nonempty list is compared with the derived ids for that side only. One nonempty list that matches, with the other list omitted, is not `model_flat_list_mismatch`, and the stage can be `complete`. A nonempty list that differs is still a mismatch. The selections are still the rereads, and the stage is `incomplete`. `model_flat_list_mismatch.compared` names the lists that were sent. `disagreed` names the ones that differed.

`batch_size` stays a performance setting and stays 1 by default. The derived step allowance is still the planned review calls, the planned decision calls, and 24 steps of overhead. Recording a pair and repairing a reread are extra visits inside that overhead. The allowance is not a new analytic setting.
