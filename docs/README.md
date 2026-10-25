# Design notes

Package `policy_version` is `0.9.25`. This index says which notes describe the package now and which notes are the record of an earlier version. A historical note can disagree with the code. The code, the README, and the current notes below win.

Axial coding, selective coding, and theoretical sampling are experimental. No study adjudication file is included. The discovery example is synthetic. Live evaluation runs through v31, on package 0.9.9, did not measure analytic accuracy.

## Current

| Note | What it describes now |
| --- | --- |
| [../README.md](../README.md) | The pipeline, the defaults, the evidence tiers, quiet rounds, timeouts, and gate scoring. |
| [methodology.md](methodology.md) | How the open-coding default maps onto the sources already named there. |
| [design.md](design.md) | The open-coding engineering design, plus a version log through `0.9.25`. The early sections are the 0.6 run. |
| [design-relationship-tiers.md](design-relationship-tiers.md) | Evidence tiers. `pattern_min_cases` is 2. Tier labels are `category_evidence`, `hypothesis`, `pattern_supported`, `attribution`, and `verified`. |
| [design-negative-cases-092.md](design-negative-cases-092.md) | Negative-case search at respondent level, and a core that needs tier 2. Later search corrections are in the 0.9.3 review note. |
| [design-gate-evaluation.md](design-gate-evaluation.md) | `export_gate_relationships`, `load_adjudications`, and `score_gate_adjudications`, including `downstream_precision` and `downstream_withheld_supported_recall`. |
| [design-discovery-evaluation.md](design-discovery-evaluation.md) | `load_discovery_file`, `freeze_discovery_set`, and `score_discovery`. Recall, precision, distortion, and gate false negatives for a frozen human discovery file. |
| [design-sampling-0900.md](design-sampling-0900.md) | The sampling graph. Later sections through 0.9.18 are part of the current behavior. 0.9.23 does not change that graph. |
| [focused-evaluation.md](focused-evaluation.md) | The offline focused comparison and its live-cost estimate. It does not call an API. |
| [design-review-fixes-0925.md](design-review-fixes-0925.md) | An incomplete selection or core choice is withheld. Core candidates are the promoted focused categories. Row eligibility gates components. Denied means an affirmative rejection of the link. This is the latest review note. |

## Historical

These notes stay so a reader can see what a version changed. They are not the current contract. `design-review-fixes-*` through 0.9.24 are in this list. `design-review-fixes-0925.md` is the latest review note, listed above.

| Note | Version it records |
| --- | --- |
| [design-later-stages.md](design-later-stages.md) | Literature design written against 0.6.16. The opening paragraph says what 0.9.13 implemented. |
| [design-axial-0800.md](design-axial-0800.md) | Axial and selective slice through 0.8.7, with pointers forward. |
| [design-review-fixes-093.md](design-review-fixes-093.md) | 0.9.3 |
| [design-review-fixes-094.md](design-review-fixes-094.md) | 0.9.4 |
| [design-review-fixes-095.md](design-review-fixes-095.md) | 0.9.5 |
| [design-review-fixes-096.md](design-review-fixes-096.md) | 0.9.6 |
| [design-review-fixes-097.md](design-review-fixes-097.md) | 0.9.7 |
| [design-review-fixes-098.md](design-review-fixes-098.md) | 0.9.8 |
| [design-review-fixes-099.md](design-review-fixes-099.md) | 0.9.9 |
| [design-review-fixes-0910.md](design-review-fixes-0910.md) | 0.9.10 |
| [design-review-fixes-0911.md](design-review-fixes-0911.md) | 0.9.11 |
| [design-review-fixes-0912.md](design-review-fixes-0912.md) | 0.9.12 |
| [design-review-fixes-0913.md](design-review-fixes-0913.md) | 0.9.13. Its downstream-rate sentence is the formula that version shipped. 0.9.14 replaces that denominator. |
| [design-review-fixes-0914.md](design-review-fixes-0914.md) | 0.9.14. The coverage split and `min_documents` check still stand. |
| [design-review-fixes-0915.md](design-review-fixes-0915.md) | 0.9.15. The statement contract, eligibility rule, and reread report still stand. 0.9.16 changes the editable-install check, mixed batches, and the sampling step limit. |
| [design-review-fixes-0916.md](design-review-fixes-0916.md) | 0.9.16. The editable-install check, reviewer role, and timeout wording still stand. 0.9.17 changes oversized rereads, mixed-batch salvage, expenditure coverage, and the validation basis. |
| [design-review-fixes-0917.md](design-review-fixes-0917.md) | 0.9.17. The oversized-reread visit, mixed-batch salvage, and validation basis still stand. 0.9.18 counts a timed-out reread or reread repair as unknown usage. |
| [design-review-fixes-0918.md](design-review-fixes-0918.md) | 0.9.18. Timed-out rereads stay in expenditure coverage. 0.9.19 separates participant attribution from analyst memos and fingerprints axial evidence. |
| [design-review-fixes-0919.md](design-review-fixes-0919.md) | 0.9.19. Participant attribution stays on a quote, and axial resume fingerprints definitions, segments, and case identity. 0.9.20 checks the relationship the quote states, splits claim components, and fingerprints selective inputs. |
| [design-review-fixes-0920.md](design-review-fixes-0920.md) | 0.9.20. Quote validation, claim components, the selective fingerprint, and axial exploration stay as shipped. 0.9.21 adds a frozen discovery evaluation beside gate scoring. |
| [design-review-fixes-0921.md](design-review-fixes-0921.md) | 0.9.21. The frozen discovery evaluation stays beside gate scoring. 0.9.22 separates completed axial examinations from unsent pairs and tightens discovery scoring. |
| [design-review-fixes-0922.md](design-review-fixes-0922.md) | 0.9.22. Completed axial examinations, the closed quote pattern, and discovery-eval-2 stay as shipped. 0.9.23 keeps proposal reviews off structural recall and binds claim reviews to the statement. |
| [design-review-fixes-0923.md](design-review-fixes-0923.md) | 0.9.23. Discovery matching, claim binding, and the attribution fail-closed rule stay as shipped. 0.9.24 chunks decision-review requests that do not fit the per-request budget. |
| [design-review-fixes-0924.md](design-review-fixes-0924.md) | 0.9.24. Decision-review requests are chunked to the per-request budget. 0.9.25 withholds an incomplete selection or core choice instead of applying the partial set. |

`design-review-fixes-0924.md` records the 0.9.24 chunking. `design-review-fixes-0925.md` is the current review note.

## Defaults the tests check

These match the function defaults. `pattern_min_cases` is 2. `max_axial_review_calls` is 8. `max_axial_review_tokens` is 24000. `axial_token_budget` is 6000. `sampling_token_budget` is 6000. `max_sampling_review_tokens` is 24000. `max_sampling_review_calls` is 4. `quiet_rounds` is 2.
