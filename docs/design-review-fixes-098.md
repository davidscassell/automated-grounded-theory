# Review fixes (0.9.8)

These notes record the review of 0.9.7. They do not add a graph node. No new source is cited. `pattern_min_cases` stays 2. The axial token ceiling stays 24000.

## A passage rejection is not a closed direction

`judgment` on a reread note is the structured field. The note text is not searched for `irrelevant`. A note can say the evidence is not irrelevant and still be `judgment` `relevant`. That direction stays in the next ranking, and a follow-up is added only when the judgment is `relevant`.

`judgment` `irrelevant` stores the passage on `rejected_passages`. Retrieval does not offer that passage for that direction again. The direction stays open, because one passage that does not bear on the question does not answer it.

A direction is closed only by an explicit decision. A human `reject` is `rejected_direction_keys`. A model close is `direction_decision` `close` together with a non-empty `direction_close_reason`. A close flag without a reason does not close the direction. `closed_direction_keys` is that set. The next round does not issue a closed key.

`analytic_new_direction_keys` drops an explicitly closed key and keeps a key whose passage was rejected. Quiet rounds read that field when it is present. A later round that offers an already seen open direction has an empty `new_direction_keys` list, so the quiet-round criterion can be met while the direction stays open and can still receive a different passage. Older rows that omit the field still use `new_direction_keys`. `judgment` on the saturation record stays null.

This narrows the 0.9.7 rule that dropped a direction after an irrelevant note. The passage is what is not reissued. The direction is reissued until someone closes it.

## Selected pairs and scheduled rereads

`model_sampling_pairs` and `selected_sampling_pairs` are the validated selection. The coding-call ceiling is applied only when `pursuit` is `reread`. `scheduled_reread_pairs` is the subset that ceiling can pay for, and the reread loop sends that subset. Pairs beyond it stay on `reread_unsent_pairs` with reason `call_allowance`, and `sampling_stop` is `call_allowance`. A flat-list mismatch is still recorded and does not replace that stop.

`brief_only` requests no reread. The full selection is kept. The stage is not `budget_limited` because the coding allowance is zero.

## Selection repair uses the packed payload

The initial selection request drops lowest-ranked directions, and the passage proposals for the directions that were dropped, until the request fits `sampling_token_budget`. That payload is stored. The repair sends it again and appends the repair note. It does not put the omitted proposals back. If the stored payload is missing, the repair filters proposals with the same rule as the initial request.

## Not changed

`pattern_min_cases` stays 2. The axial review-token ceiling stays 24000. A count of supported relationships that a rejection rule withheld is not added in this version. The three 0.9.7 rules that can withhold a relationship a reader might still treat as supported are `stem_only` anchoring, an absence candidate reclassified from `contradicts` to `qualifies`, and the consequence or cause wording gate below tier 3. Whether evaluation output should tally those withheld rows is a decision, recorded in the pull request and not implemented here.
