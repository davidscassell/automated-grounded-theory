# Review fixes (0.9.3)

This note records the 0.9.3 corrections to the 0.9.2 negative-case search and to sampling and graph execution. It does not add a graph node. The graphs stay the ones in [design-axial-0800.md](design-axial-0800.md) and [design-sampling-0900.md](design-sampling-0900.md). Tiers stay the rule in [design-relationship-tiers.md](design-relationship-tiers.md). The 0.9.2 search and core rule stay in [design-negative-cases-092.md](design-negative-cases-092.md) except where this note says the search now classifies the respondent and reports a truncated queue as `budget_limited`.

Nothing here is quoted from a source that was not already used in those notes. Strauss and Corbin (1990, 1998), Charmaz (2006), Glaser (1978, 1992), and Glaser and Strauss on constant comparison are the sources those notes already name. This note does not add a page citation.

## Respondent classification

Passages that share a respondent are read together before a candidate type is chosen. The respondent has an endpoint when any of their passages, including the cited supporting passage, shows it. The type is then one of `contrasting_outcome`, `opposite_pole`, `a_without_b`, `b_without_a`, `both_endpoints`, or `embedding`. A respondent who shows both endpoints is not also queued as `a_without_b` and `b_without_a`.

One representative passage is queued. When that passage does not itself show both endpoints, and another passage from the same respondent does show the missing one, that passage is sent as `complementary_passages`. A quote may be anchored in either passage. The cited passage can be complementary. It is not itself a second candidate.

## Budget and slots

`plausible`, `queued`, `attempted`, and `remaining` are stored separately, on the search and on each relationship. `attempted` counts a candidate only after a request was sent. The stage is `budget_limited` when `remaining` is greater than zero. Judging one relationship does not make the stage complete while another relationship still has a plausible candidate.

The first queued slot is offered to each relationship that has a candidate before any relationship receives a second slot. With one slot and two relationships, the first relationship is queued and the second stays `budget`. With two slots, each relationship gets one before either gets another.

## Sampling requests

`sampling_token_budget` is the per-request ceiling. `max_sampling_review_tokens` is the total allowance. Both are performance settings. The initial selection, reread, and saturation requests are measured with the same token estimate axial coding uses. A request over either ceiling is not sent. The stage is `incomplete`, with reason `request_budget` or `token_allowance`. A repair that would overflow the per-request budget is not sent. A response that still does not parse is preserved, each attempt and its usage is stored on `sampling_attempts` and in the prompt log, and the stage is `incomplete`.

A reread payload carries the selected direction, its sampling query, the target statement or category, and the comparison the note should resolve. The stored note keeps `direction_id` and `target_id`.

## Graph steps

Each axial proposal is one graph step. The step allowance is `propose_calls` plus 24 steps of overhead, unless the caller passes `recursion_limit`. Focused coding already derived its limit from the corpus. Sampling uses the planned review and decision calls plus the same overhead. Selective coding uses the same helper; its graph does not loop, so the overhead covers it. `recursion_limit` is a performance setting.

If a run still reaches the step limit, the exception is caught. The checkpoint's relationships, prompt log, and provider usage are kept. The stage is `incomplete` with reason `recursion_limit`.
