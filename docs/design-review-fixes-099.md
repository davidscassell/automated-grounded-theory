# Review fixes (0.9.9)

These notes record the review of 0.9.8. They do not add a graph node. No new source is cited. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. The 0.9.8 rules are unchanged: the structured `judgment` decides relevance, a rejected passage is not a closed direction, and `model_sampling_pairs` stays separate from `scheduled_reread_pairs`.

## Stable question identity

A question key is `kind:target`, or `empty_paradigm_slot:target:role`. `stable_question_key` removes leading `reread_follow_up:` prefixes until that form remains. `thin_category:c-prep`, `reread_follow_up:thin_category:c-prep`, and `reread_follow_up:reread_follow_up:thin_category:c-prep` are one question.

A relevant reread stores `question_key` on the note and does not create a direction whose identity is the previous key wrapped again. The observation stays on the question that retrieved the passage.

## What "distinct" means

A new direction is created only when all of these hold:

- `follow_up_kind` is one of the record-grounded direction kinds (`thin_category`, `one_sided_category`, `no_negative_case`, `unexplored_property`, `unexplored_dimension`, `partial_relationship`, `cross_respondent_relationship`, `empty_paradigm_slot`, `open_memo`, `weak_core`).
- `follow_up_target_id` is a code id, relationship id, or core id already stored on the run. The kind selects which of those sets is allowed.
- For `empty_paradigm_slot`, `follow_up_role` is one of the paradigm roles.
- The resulting key is different from the question the note is attached to.

The note text is not read and is not compared. The same structured fields on the same run always produce the same key. A target that is not on the run does not become a question. If the ranker already emitted that key, the note does not add a second copy.

## Completed pairs and older ledgers

Each reread writes `completed_pairs` as `question_key` and `passage_id`. Retrieval skips that passage for that question. The reread loop withholds it with reason `completed_pair` and does not call the model. That reason does not set `budget_limited`. Changing a follow-up identifier cannot make the pair new, because the pair is stored under the stable key.

`normalize_sampling_ledger` runs when a sampling round loads. It rewrites `direction_keys`, `new_direction_keys`, `analytic_new_direction_keys`, closed and rejected keys, rejected passages, and notes onto the stable key. A key is new only when that stable question was absent from earlier rounds. When the stored key differed, `recorded_direction_key` keeps the nested form. A close that was stored on a nested key closes the stable question, because the nested key was not a separate question.

Quiet rounds still read `analytic_new_direction_keys` when that field is present. After normalization, a round whose only new key was a wrapper of an already seen question has an empty list, so `sampling_rounds_without_new_directions` can be met.

## Gate adjudication export

The export and scorer are specified in [design-gate-evaluation.md](design-gate-evaluation.md). They do not change a gate.
