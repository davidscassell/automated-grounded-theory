# 0.9.42 review fixes

Status: current for package 0.9.42. The 0.9.41 note stays the record of retries inside the call allowance and the counterexample token reserve. This note does not change core selection, `pattern_min_cases` (2), `one_document_one_respondent` (off), the 6,000 request cap, or the 24,000 token ceiling. No LLM judge is added. Undefined metrics stay null. No Muse or TA data is in this package.

## Why a compacted counterexample was 9,466 tokens

The cap stayed 6,000. Passage text had already been shortened to 40 characters. The request still serialized the whole relationship row: supporting cases, claims, co-occurrence, review history, flags, and other derived metadata.

A fixture with that shape, and a derived-metadata field long enough to match the observed total, measures:

| Part | Tokens |
| --- | ---: |
| Instructions | 407 |
| Schema | 241 |
| Claim | 17 |
| Category definitions | 0 |
| Evidence (compacted passage) | 14 |
| Provenance | 27 |
| Other context | 8,760 |
| Total | 9,466 |

Other context is the defect. The evidence is not. Sending the full row cannot be fixed by compacting the passage again.

## Decisions

1. The counterexample request sends the claim, the two endpoint definitions, readable passage context, and provenance. It does not send the codebook, review history, or derived metadata. Passage text stops at the quote plus surrounding context. It is not cut to a meaningless stub. If that required content still exceeds 6,000 tokens, the stage is `required_but_blocked` and the breakdown is stored. The stage is not `skipped` or `not_required`.

2. Identical statements on the same directed pair are consolidated before relation review. Evidence, provenance, and review history are unioned. `prior_decisions` keeps every member's disposition. The merged row is `proposed`. An accepted member does not win, and a rejected member does not win. The review sees the combined evidence.

3. Each returned draft receives `proposal_id` before it is accepted or rejected. The export says `invalid`, `retained`, or `merged_into:<relationship id>`. `proposal_lineage` fails when those dispositions do not account for the raw rows and the relationships.

4. Calls moved off a blocked counterexample are queued. Unplanned multi-document pairs come first. The unused counterexample token reserve is released only as far as those calls need, and the call is not queued when the remaining allowance still refuses it. The graph routes back to proposals and keeps sending until `calls_executed` equals `calls_transferred` or the coding-stage ceiling is reached. A counterexample that already ran does not consume a transferred proposal slot. `calls_unsent` counts calls that were neither sent as counterexamples nor transferred.

5. `call_usage` keeps `estimated_input_tokens`, `provider_input_tokens`, `provider_output_tokens`, and `usage_completeness` separate. A missing provider count stays null.

6. Funnel rows keep four summary reasons and the full reason list. `partial_endpoint_evidence` stays on the full list when an endpoint is unanchored.

These checks are engineering and evidence-traceability coverage. Analytic accuracy is unmeasured.
