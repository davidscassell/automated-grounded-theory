# 0.9.17 review fixes

`policy_version` is `0.9.17`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. A metric that was not computed stays null. No live evaluation was run for this note. The 0.9.16 notes stay as history.

## Oversized rereads

Decision: pairs that are not sent are withheld together in the visit that would arm the next pair that fits. The recursion limit stays the 0.9.16 visit count. It is not increased by `evidence_limit` or by the number of selected passages.

A request over `sampling_token_budget`, or over the remaining token allowance, does not use a provider call. The 0.9.16 limit assumed that the only non-call withhold was the one visit after the call ceiling. That is true for `call_allowance`. It is not true for a per-request budget failure, which used to return after one pair and come back for the next. A human can assign more passages than `max_sampling_review_calls`. Those pairs are not cut at selection. With the default four-call allowance, 32 oversized assignments used to stop at `recursion_limit` after 22 unsent passages and zero model calls. They now finish in that arming visit. Every passage is on `reread_unsent_pairs` with reason `request_budget`, and the finished stop is `request_budget`.

A model selection is still cut to the coding-call allowance before the reread loop. Pairs beyond that allowance stay `call_allowance` and are not measured against the request budget. The first pair that fits is armed for the next visit. The 32-call reread from 0.9.16 still completes inside the same visit limit.

## Mixed proposal batches

Decision: keep the 0.9.16 rule. Valid rows stay. Invalid rows are logged. A repair is sent only when nothing valid remains, and it is still one repair.

The raw text used for salvage now includes tool-call arguments when the message content has no JSON object. Content that already contains an object is left as it is. The repair response is salvaged before a failed repair is closed, so a mixed repair keeps its valid rows and does not clear `repair_pending` first. Grounding checks still reject an unrequested pair, an empty statement, a verbatim quote, an unknown passage, and stitched passages after that salvage.

## Reread expenditure

Decision: a sent call with no reported tokens is unknown usage, not a spend of zero. `usage_from_message` stores a missing provider report as zeros. That all-zero record, and a total with no input and no output, does not count as usage. An omitted or unsent request is not a call in this total.

`reread_expenditure_tokens` is the known sum, including irrelevant responses, failed parses, and `SamplingRereadNote` repairs that reported a positive input or output. `reread_expenditure_calls` is the number of sent reread calls. `reread_expenditure_calls_with_usage` is how many of those reported usage. `reread_expenditure_usage_coverage` is that ratio, and null when there were no reread calls. `reread_expenditure_complete` is true when every sent call reported usage. No reread calls is complete, and the known sum is 0.

`expenditure_tokens_per_relevant_reread` and `expenditure_cost_per_relevant_reread_usd` stay null when usage is incomplete or when `relevant_reread_count` is zero. Cost stays null unless source `metadata.token_prices` has both `input_usd_per_million` and `output_usd_per_million`, and the usage is complete. A complete run with no reread calls reports cost 0 when those prices are set. There is no new price argument. `tokens_per_relevant_reread` still counts only attributable tokens from successful relevant responses. `human_confirmed_useful_rereads` stays null and is not computed.

## Validation basis

Decision: refresh `validation_basis` on every validation. Append `validation_revision_history` only when the owned flag set changes, and archive the previous basis. A statement edit that leaves the flags the same does not add a history row.

A human accept applies the statement and refreshes validation before it applies the role. Those are two validations inside one resume. The role change then archives the updated statement. The alternative, archiving the pre-resume statement when the sentence and the role change together, was not taken: the sentence was already the statement those next flags were about.
