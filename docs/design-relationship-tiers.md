# Relationship evidence tiers (0.9.1)

This note is the 0.9.1 rule for accepting a relationship. It sits on the axial graph from 0.8.7 and the v29 fixes recorded in 0.9.0. It does not add a node. Sampling stays the graph in [design-sampling-0900.md](design-sampling-0900.md).

The file `literature-cross-respondent-relationships.md` was not on this machine. Nothing below is quoted from it. The sources named here are the ones already used in [design-later-stages.md](design-later-stages.md), which this change re-read: Strauss and Corbin (1990, 1998), Charmaz (2006), Glaser (1978, 1992), Glaser and Strauss on the constant comparative method as that note states it, and Kelle (2005, 2007) on theoretical scaffolds. Miles, Huberman, and Saldaña were not in those pages. They are not cited here as works that were read. The request that produced this rule said those traditions require a negative-case search, and that Glaser rejects that search and rejects counting participants. This package does not pretend to quote Glaser for that rejection.

## What the package does

One incident is enough to show that a category is present. [design-later-stages.md](design-later-stages.md) already says Glaser and Strauss treat a single modifying incident as analytically real, and that two documents are a review threshold for this package rather than a rule from the method literature. A relationship is not that incident. It is a hypothesis built by comparing incidents across cases. Strauss and Corbin (1990, 1998) put links among categories at this stage. Charmaz (2006) asks the relational questions without requiring the paradigm slots. Glaser (1978, 1992), as that note uses him, treats a required paradigm as forcing and would use a family as a theoretical code, which is a hypothesis, not a finding.

No source read for this note requires both category quotes to sit in one passage. The 0.8.7 rule did. It withheld every causal accept whose quotes came from different passages, and live runs accepted no relationships. That rule is retired. A quote from respondent A for one category and a quote from respondent B for the other is kept. It is a hypothesis. It is not one instance of the link.

`pattern_min_cases` defaults to 2. The procedure and the row record that number and a note: it is a design choice for this package, not a quotation from the method literature. The same passage inside a case is the strongest within-case evidence. Two question-segments from one respondent are one case, and a weaker one. Survey data often has one short answer per question, so co-occurrence is computed across a respondent's segments, not by treating each question as a respondent.

Negative-case search stays. It was already in the counterexample step. Strauss and Corbin, as this package has used them, are the reason it stays. Counts are description. The 2x2 (`both`, `a_only`, `b_only`, `neither`) does not validate the relationship and does not raise the tier. When the corpus cannot be read, those counts are null. They are not stored as zero. Variation search and the rival check are `undefined` when they were not run.

## Tiers

| Tier | Label | What it takes | Wording the row may use |
| --- | --- | --- | --- |
| 0 | `category_evidence` | One verbatim quote for one endpoint | a category is present |
| 1 | `hypothesis` | Each endpoint anchored somewhere, possibly in different cases | may be related |
| 2 | `pattern_supported` | At least `pattern_min_cases` cases, each with a verbatim quote for both endpoints | co-occurs in N respondents |
| 3 | `attribution` | Tier 2, plus a participant quote that links the two or an analyst memo that says why, plus a rival check that returned a relevant judgment | participants attribute X to Y |
| 4 | `verified` | Tier 3, plus a researcher `verify` with a named scope | verified within the recorded scope |

A category label is not a quote. The quote has to be a verbatim substring of a located passage and share a distinctive stem with that endpoint. Tier 1 is flagged `cross_case` when the two endpoint quotes are not in the same case, and `hypothesis_not_supported`. Accepting it does not happen. The row stays.

Accepting tier 2 sets `evidence_status` to `pattern_supported` and the disposition to `accepted`. It does not set `verified`. A sentence that says "leads to" on that row is kept and flagged `wording_exceeds_tier`. The storyline flag is `storyline_wording_exceeds_tier`. The memo flag is `memo_wording_exceeds_tier`. The same style of flag already used for an unaccepted cause, `storyline_unaccepted_relationship`, still fires when the cited tier is below 3.

Tier 3 is the first tier that may use causal wording. The rival check is a counterexample judgment of supports, contradicts, or qualifies that passed the usual validity checks. An irrelevant round does not count. A search that was not run stays `undefined`, and the tier stays 2.

Tier 4 is only the human checkpoint. Resume with `action` `verify` and a non-empty `scope`. If the row is not yet tier 3, the flag is `verification_withheld` and the computed tier stays. After counterexamples, commit promotes a requested verification that now meets tier 3, and records the scope. `decision_mode='llm'` may ask for `verified`. The request is kept on the review history, `model_cannot_verify` is set, and the tier is not raised. `judgment` stays null.

## What the reviewer sees

The relation payload includes `evidence_tier`, `evidence_label`, `allowed_wording`, `supporting_cases`, `cooccurrence_2x2`, the within-case strength, and whether the negative-case and variation searches were reported. The instruction says to accept tier 2 as `pattern_supported` and not to set `verified`. Proposal requests still prefer `candidate_same_passage_cooccurrence`. Remaining slots in `axial_evidence_limit` may list `candidate_cross_case` passages from other respondents whose text contains both endpoints' distinctive terms. The model cites those passage ids. The list does not push a same-passage passage out of a full budget.

`citation_anchoring_rate` is the share of tier 2 or higher rows whose supporting cases each quote both endpoints. It is null when no such row qualifies. `citation_anchoring_by_tier` reports `n` and `citation_rate` for each tier. Below tier 2, and when `n` is 0, `citation_rate` is null. `citation_anchoring_same_passage` is that share among tier 2 or higher rows whose strongest within-case evidence is the same passage, and it is null when there are none. `citation_anchoring_cross_passage` stays null. `endpoint_anchoring_rate` is unchanged: it is still the share of endpoints the quote matches, and a missing denominator stays null.

Human and model fields stay separate. An empty resume is still an error. `provider_budget` still survives a pause. A category with no passage in the request is still deferred. Undefined stays undefined.

0.9.2 does not change these tiers. It changes who is sent to the negative-case judgment and what a contradicting or qualifying judgment does to the row, the memo, and the storyline. A core now needs tier 2 before it is graph-supported. That rule is [design-negative-cases-092.md](design-negative-cases-092.md). 0.9.3 does not change the tiers either. It classifies the respondent before choosing a passage and reports a truncated search as `budget_limited`. That note is [design-review-fixes-093.md](design-review-fixes-093.md). 0.9.4 does not change the tiers. A supporting quote records `quote_passage_id` and offsets for the passage that contains it. That note is [design-review-fixes-094.md](design-review-fixes-094.md). Package 0.9.13 does not change these tier numbers or `pattern_min_cases`. `downstream_eligible` stays separate from `disposition`. Package 0.9.15 keeps the tier numbers. `downstream_eligible` is supported evidence only. A weak stem is hypothesis exploration. `unspecified` is a valid association role, and a causal role below tier 3 is `role_evidence_exceeds_tier`, apart from sentence wording. The scoring note is [design-gate-evaluation.md](design-gate-evaluation.md). The 0.9.15 note is [design-review-fixes-0915.md](design-review-fixes-0915.md). The index is [README.md](README.md).
