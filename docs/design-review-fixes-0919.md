# 0.9.19 review fixes

`policy_version` is `0.9.19`. Axial coding, selective coding, and theoretical sampling stay experimental. `pattern_min_cases` stays 2. The axial token ceiling stays 24000. `batch_size` stays 1. Tier 4 stays a human `verify`. A metric that was not computed stays null. No live evaluation was run for this note. The 0.9.18 notes stay as history.

## Participant attribution and analyst interpretation

Decision: tier 3 and the wording `participants attribute X to Y` require a participant quote. The quote has to state the link, share distinctive stems with both endpoints, and sit inside a located passage. A relevant rival check is still required before the tier rises. An analyst memo cannot supply that quote.

Decision: a memo counts as an analyst interpretation only when it is explicitly labeled (`analyst interprets`, `analyst interpretation`, `I interpret`, or `my interpretation`) and its text names both endpoints. A shared token plus `because` is not enough, even when both endpoints happen to be named. The row stays tier 2. `allowed_wording` is `analyst interprets X as related to Y`. `attribution_source` is `analyst_interpretation`. The flag is `analyst_interpretation`. `evidence_use` is `hypothesis_exploration`, so `downstream_eligible` is false, and the row does not support a core. It is not a new tier, and a human `verify` still requires tier 3.

Decision: a memo with `actor` `model` or `kind` `model_memo` is the model's text. It cannot authorize participant attribution and it cannot authorize an analyst interpretation. LLM-only mode therefore cannot use a memo it just wrote to reach tier 3.

## Axial evidence fingerprint

Decision: resume rejects a source that does not match the saved evidence, rather than continuing on the checkpoint's old definitions. `source_hash` for axial and selective coding now covers passage text, document and respondent ids, code definitions and labels, and segment assignments. Those are the fields focused coding already hashes, plus the case ids the tier count reads. An unchanged source still resumes. `allow_changes` does not bypass this hash.

Decision: a checkpoint saved by 0.9.18 used a hash of passage text and code ids only. Resume cannot tell whether definitions, segments, or case ids changed. That checkpoint does not continue. The error says the evidence fingerprint is missing and a new run is required. The default seed still uses the older passage-and-code-id hash, so a definition edit does not reshuffle an unseeded run. Sampling keeps that older hash.
