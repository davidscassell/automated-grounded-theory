# Negative-case search and a graph-grounded core (0.9.2)

This note is the 0.9.2 rule. It does not add a graph node. Axial coding is still the graph in [design-axial-0800.md](design-axial-0800.md). Selective coding is still the graph after it. Tiers stay the rule in [design-relationship-tiers.md](design-relationship-tiers.md).

`design-audit.md` was not on this machine. Nothing below is quoted from it. The sources named here are the ones already used in [design-later-stages.md](design-later-stages.md): Strauss and Corbin (1990, 1998), Charmaz (2006), Glaser (1978, 1992), and Glaser and Strauss on constant comparison as that note states it. The request for this release said Strauss and Corbin treat a negative case as a dimensional extreme or a variation the theory has to account for, and that Glaser objects to a negative-case procedure. Eisenhardt and Miles and Huberman were named in that request for replication logic. They were not read for this note and are not cited as sources that were read.

## Negative-case search

The search is a procedure setting, `negative_case_search`, default on. Turning it off skips the stage. It is not a performance setting. `axial_evidence_limit` and the call ceilings stay the performance and budget settings.

For each relationship whose tier is missing or at least 1, and for each claim, the candidate set is built at respondent level:

- a respondent with evidence for one endpoint and not the other
- a respondent whose text is an opposite pole, or states a different outcome while mentioning both topics
- a passage that hits both endpoints, or whose embedding is at or above 0.55

Tier 0 is category evidence and is not searched. A missing tier is still searched, so a row that has not been scored yet is not skipped.

Respondents are ranked by contrast and by how strongly a passage hits each endpoint. The first passage from each plausible respondent is queued before a second passage from the same respondent. Passages with no endpoint signal, and passages with only a generic stem or an embedding below 0.55, are filtered. They are not sent. The search record stores candidates considered, how many were filtered, why, and the 2x2 (`both`, `a_only`, `b_only`, `neither`). Those counts are descriptive. They do not validate or invalidate a link, and they do not move a tier.

A model call happens only for a queued candidate. The judgment is `supports`, `contradicts`, `qualifies`, or `irrelevant`, with a verbatim quote from the located passage for the first three. A rationale that says the passage is irrelevant makes the judgment invalid. That guard is unchanged.

There is no second-wave fallback. When nothing plausible remains, the reason is `no_plausible_candidates` and the stage is `complete_no_negative_search`. When plausible candidates exist and the call budget is exhausted, the reason is `budget` and the stage is `budget_limited`. An all-irrelevant round stays `complete_no_negative_search`. `relationships_with_negative_case_search` and the relevance rate stay null when no relevant judgment exists. They are not stored as zero.

A valid `contradicts` still disputes the row, flags `relationship_contradicted`, and stores `contradiction_note`. The row is kept. The tier number is not lowered. Storyline and memo text that ignore the note are flagged. A valid `qualifies` stores `variation_note` and `condition_note`, flags `relationship_qualified`, and reports the variation search. It stays a relevant judgment for the rival check. On an axial relationship it does not, by itself, set `evidence_condition` to `disputed`. Focused claim tests still treat a valid `qualifies` as a dispute.

Claim retrieval keeps the sources it already used (`claim_opposite_pole`, `claim_negative_case`, `claim_same_code`, `claim_similarity`). The respondent search is recorded on the claim report. A respondent candidate is queued in a remaining slot, and a category-without-outcome passage is queued only when it also passes the existing claim relevance gate. Passages that fail that gate stay in the record.

## Core and storyline

`score_candidates` shows counts: pattern-supported centrality, cases covered, related categories, contradicted relationships, and qualified relationships. `graph_supported` is true only for an endpoint of a non-rejected relationship at tier 2 or higher. A proposed link or a tier-1 hypothesis is visible and does not clear the flag.

The human or the model still chooses. A model core that is not graph-supported is kept as `model_core_id`, flagged `core_unsupported_by_graph`, and not copied into `researcher_core_id`. `core_emphasis` is false, so the storyline prompt does not center that category. The id and its provenance stay on the payload. A human may still name an unsupported core. It is flagged and kept.

Integration reports categories linked to the core directly, categories linked by a path, orphan categories, and contradicted relationship ids. `core_category_coverage` and `core_respondent_coverage` are the shares of categories and respondents linked to the core. Both are null when the graph has no relationships. They are not stored as zero in that case.

A storyline statement cites relationship ids and their tiers, or it is `statement_kind` `description` and still flagged `storyline_no_relationship`. Wording stays inside the lowest cited tier. A contradicted citation has to carry a qualifier. A qualified citation has to carry the condition. Provenance stays `human`, `model`, or `unset`.

`core_stability` reports `n`. The score is pairwise agreement of the core id each run named. It is null when fewer than `core_stability_min_n` runs name a core. That minimum defaults to 2 and is a design choice for this package, not a quotation from the method literature. Two runs that name different cores score 0. They are not averaged into one core.

Core selection, memos, and the storyline each store a `graph_snapshot` of the relationship ids and tiers they read. Later edits do not rewrite that snapshot.

`judgment` stays null. Human and model fields stay separate. An empty resume is still an error. `provider_budget` still survives a pause.
