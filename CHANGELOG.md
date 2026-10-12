# Changelog

## 0.6.4

- Token overlap only proposes a comparison. Auto-merge is limited to spelling and inflection variants (case, punctuation, whitespace, plural and gerund forms) whose definitions are compatible. A negation or a polarity difference is never auto-merged. `trusting managers to share information` and `not trusting managers to share information` (Jaccard about 0.83, contradictory definitions) are proposed, not merged.
- The coding schedule is three settings: `passages_per_request`, `max_concurrent_requests`, and `passages_per_codebook_update`. Concurrency no longer widens the frozen codebook. `strict_sequential_settings()` and `procedure="strict_sequential"` read one passage at a time in document order, commit after every passage, and turn pipeline overlap and parallel documents off. `batch_size` and `concurrency` remain as aliases and warn when the old coupling would have changed the procedure.
- A cached pair decision is reopened when either code's definition text or revision changes, or when its examples or assignments change. Token overlap is not a threshold. Negating a definition (`Employees trust...` to `Employees do not trust...`) reopens the pair.
- Sweep prompts treat stance as context, not a merge veto, matching the coding prompt and the docs.
- The open-coding example for "I like my team" is `expressing liking for the team`. Automatic reading-order notes are `provenance`, not reflexivity memos. A reflexivity memo is still the one a sweep model writes.
- `adjusted_rand_index` stays. It compares passage partitions, so different singleton codes on every passage score 1.0. `aligned_assignment_agreement` scores the set of codes on each passage after label alignment (mean Jaccard, and Krippendorff's alpha on the aligned passage-by-code matrix).

## 0.6.3

- Semicolon-joined labels are split into separate candidate codes. Each piece is validated on its own, and a piece that matches an existing code is reused.
- An empty label, a label over the length limit, or a new label that bundles several concepts is retried once. The feedback names the problem. A bundle is a comma list of three or more concepts, or three or more concepts joined by "and". The retry asks for a single concept.
- If that retry is still empty, too long, or bundled, or if it returns no codes, the code is kept and flagged for review (`empty_code`, `code_too_long`, `bundled_concepts`). A blank label is stored as `(empty label)`. Generic labels are still rejected.
