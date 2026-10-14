"""Apply passage coding and sweep decisions to the running codebook.

Counts are rebuilt from segments after every commit so a merge cannot
double-count applications.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from automated_grounded_theory.retrieval import jaccard
from automated_grounded_theory.validation import (
    is_code_id_text,
    LABEL_HOLD_REASONS,
    LENIENT_LABEL_REASONS,
    QUOTE_FLAG_STATUSES,
    assess_label,
    attribute_leak,
    bounded_unit,
    bundled_concepts,
    code_key,
    display_label,
    gerund_ok,
    label_restates_context,
    lenient_label_fields,
    anchor_quote,
    attribute_evidence,
    attributed_passage_ids,
    evidence_wholly_outside,
    memo_review_reasons,
    review_passage_flags,
    split_joined_label,
)

_ACTION_ALIASES = {
    "reuse": "reuse",
    "refine": "refine",
    "new": "new",
    "create": "new",
    "add": "new",
    "update": "refine",
}


def _span_rows(anchor) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Pair local spans with document spans. Missing document offsets stay unset."""
    local = list(anchor.spans)
    docs = list(anchor.doc_spans)
    if len(docs) != len(local):
        docs = [(None, None)] * len(local)
    return list(zip(local, docs))


def _quote_flag_rationale(status: str, similarity: float | None) -> str:
    if status == "possibly_modified":
        score = f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "unknown"
        return (
            f"The closest source span is only similar to the quote (similarity {score}), "
            "so the wording may have been modified. The code was kept."
        )
    return (
        "The quote could not be anchored in the passage, its neighbors, or the document. "
        "The code was kept and flagged for review."
    )


def _code_id(key: str) -> str:
    """Stable id for a normalized label. The same spelling always hashes the same."""
    import hashlib

    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def empty_counters() -> dict[str, int]:
    return {"segment": 0, "memo": 0, "highlight": 0, "sweep": 0}


def _next(counters: dict[str, int], name: str, prefix: str) -> str:
    counters[name] = int(counters.get(name, 0)) + 1
    return f"{prefix}{counters[name]:05d}"


def _fallback_rationale(op: str, reason: str) -> str:
    """A one-sentence decision text when the model did not supply one."""
    text = {
        "context_restatement": (
            "The label mostly restates the research context, so it needs wording grounded in the participant's passage."
        ),
        "already_decided": (
            "This pair was already decided in this run, and neither definition nor its evidence has changed."
        ),
        "opposite_valence": "These codes take opposite stances, so they stay separate.",
        "below_merge_confidence": "The agreement confidence is below the merge threshold, so the merge waits for review.",
        "below_review_threshold": "The confidence is below the review threshold, so this decision waits for a person.",
        "low_confidence": (
            "The model's confidence is below the review threshold. The action was applied and flagged. "
            "The score is a diagnostic, not a probability."
        ),
        "thin": "The memo is analytic and shorter than the comfortable range. It was kept and flagged thin.",
        "in_vivo_below_review_threshold": "The in-vivo flag is not confident enough to apply, so the code stays a composed label.",
        "merge_cap": "This merge was past the cap for this sweep, so it was not applied.",
        "quote_split": "This quote does not fit the rewritten code, so it was split back out.",
        "quote_possibly_modified": (
            "The closest source span only partly matches the quote, so the wording may have been modified."
        ),
        "quote_unanchored": (
            "The quote could not be anchored in the passage, its neighbors, or the document, so the code was kept and flagged."
        ),
        "definition_not_survivor": "A definition for a retired code was ignored so it would not replace the survivor.",
        "empty_code": "The label was empty, so the code was kept under a review flag instead of being dropped.",
        "code_too_long": "The label is too long, so the code was kept and flagged for review.",
        "bundled_concepts": "The label bundles several concepts, so the code was kept and flagged for review.",
        "evidence_outside_passage": (
            "The quote is outside the passage being coded, so the code was kept and flagged. "
            "One overlapping passage is the attribution. Spans in several passages stay on "
            "each of those passages. An unresolved quote is not assigned to the passage that was coded."
        ),
    }.get(reason)
    if text:
        return text
    if op == "merge_accept":
        return "These codes name the same incident, so they were merged."
    if op == "merge_reject":
        return "These codes do not name the same incident, so they stay separate."
    if op == "memo":
        return "The memo records a comparison, a property, and a question."
    return f"This {op or 'decision'} was recorded."


def _op(
    *,
    op: str,
    status: str,
    confidence: float,
    ids: Mapping[str, Any] | None = None,
    reason: str = "",
    rationale: str = "",
) -> dict[str, Any]:
    text = str(rationale or "").strip() or _fallback_rationale(op, reason)
    return {
        "op_id": "",
        "op": op,
        "decision": op,
        "status": status,
        "confidence": confidence,
        "ids": dict(ids or {}),
        "reason": reason,
        "rationale": text,
    }


def _index_ids(codebook: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for key, entry in codebook.items():
        index[str(entry.get("code_id") or _code_id(key))] = key
    return index


def expand_joined_labels(proposals: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Split semicolon-joined labels into one proposal per candidate code.

    Each piece is validated on its own. A later piece does not keep the
    original code id, so it can match an existing label or become its own
    code. A label with no semicolon is unchanged.
    """
    expanded: list[dict] = []
    for proposal in proposals:
        incoming = dict(proposal)
        label = str(incoming.get("code") or "")
        if ";" not in label:
            expanded.append(incoming)
            continue
        parts = split_joined_label(label)
        multi = len(parts) > 1
        for index, part in enumerate(parts):
            row = dict(incoming)
            row["code"] = part
            if multi:
                row["code_id"] = ""
                row["refined_label"] = None
                row["_joined_part"] = True
                if index:
                    row["action"] = "new"
            expanded.append(row)
    return expanded


def _fields_for_known(entry: Mapping[str, Any], label: str, key: str) -> dict[str, Any]:
    """Reuse the stored label when a candidate matches a code already in the book."""
    shown = str(entry.get("code") or display_label(label) or label)
    return {
        "code": shown,
        "key": key,
        "in_vivo": bool(entry.get("in_vivo", False)),
        "gerund_ok": bool(entry.get("gerund_ok", gerund_ok(shown))),
    }


def _mark_label_review(entry: dict, reasons: Sequence[str]) -> None:
    """Keep empty, over-long, and bundled labels visible on the code itself."""
    flags = [reason for reason in reasons if reason in LABEL_HOLD_REASONS]
    if not flags:
        return
    entry["needs_review"] = True
    stored = [str(reason) for reason in (entry.get("review_reasons") or [])]
    for flag in flags:
        if flag not in stored:
            stored.append(flag)
    entry["review_reasons"] = stored


def opposite_valence(left: Any, right: Any) -> bool:
    """True when one code is positive and the other is negative."""
    pair = {str(left or "").strip().casefold(), str(right or "").strip().casefold()}
    return pair == {"positive", "negative"}


def definitions_changed_materially(previous: str, current: str) -> bool:
    """True when the definition text is not the same string.

    Any edit reopens a pair. Token overlap is not a threshold: adding "not"
    keeps most of the words and still counts as a change.
    """
    return str(previous or "") != str(current or "")


def agreement_confidence(merge: Mapping[str, Any]) -> float | None:
    """Rubric agreement as a second confidence, using the 0.2 / 0.5 / 0.95 anchors.

    ``None`` means the model did not set the rubric, so the caller's own
    confidence stands. The three flags are action, object, and stance. Stance
    is recorded and is not a veto: two of the three still score 0.5. A reject
    is confident when that rubric fails.
    """
    flags = [merge.get("same_action"), merge.get("same_object"), merge.get("same_stance")]
    if all(flag is None for flag in flags):
        return None
    hits = sum(bool(flag) for flag in flags)
    action = str(merge.get("action") or "accept")
    if action == "reject":
        if hits == 0:
            return 0.95
        if hits == 1:
            return 0.5
        return 0.2
    if hits == 3:
        return 0.95
    if hits == 2:
        return 0.5
    return 0.2


def decision_confidence(merge: Mapping[str, Any]) -> float:
    """Confidence that drives the review gate. Prefer the rubric agreement."""
    agreed = merge.get("agreement_confidence")
    if agreed is None:
        agreed = agreement_confidence(merge)
    if agreed is not None:
        return min(bounded_unit(merge.get("confidence"), 1.0), bounded_unit(agreed, 1.0))
    return bounded_unit(merge.get("confidence"), 1.0)


TRIVIAL_LABEL_JACCARD = 0.8

# Closed-class negation. These tokens are not stemmed, so "not trusting" does
# not collapse into "trusting" and a polarity flip cannot auto-merge.
_NEGATION_TOKENS = frozenset(
    {
        "not",
        "no",
        "never",
        "without",
        "neither",
        "nor",
        "nobody",
        "nothing",
        "nowhere",
        "cannot",
        "cant",
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hasnt",
        "havent",
        "hadnt",
        "wouldnt",
        "shouldnt",
        "couldnt",
        "mustnt",
        "aint",
    }
)


def _surface_tokens(text: str) -> list[str]:
    """Case-folded words. Punctuation and extra whitespace are separators."""
    folded = str(text or "").casefold().replace("'", "").replace("’", "").replace("`", "")
    return re.findall(r"[a-z0-9]+", folded)


def _token_forms(token: str) -> set[str]:
    """Spelling plus plural and gerund forms of one word. Negation stays put."""
    if token in _NEGATION_TOKENS:
        return {token}
    forms = {token}
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if stem:
            forms.add(stem)
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                forms.add(stem[:-1])
            if stem[-1] not in "aeiou":
                forms.add(stem + "e")
    if token.endswith("ies") and len(token) > 4:
        forms.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 4:
        forms.add(token[:-2])
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        forms.add(token[:-1])
    return forms


def _negation_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _surface_tokens(text) if token in _NEGATION_TOKENS)


def negation_differs(left: str, right: str) -> bool:
    """True when one text has a negation the other does not, in a different place."""
    return _negation_tokens(left) != _negation_tokens(right)


def _forms_match(left: str, right: str) -> bool:
    left_tokens = _surface_tokens(left)
    right_tokens = _surface_tokens(right)
    if len(left_tokens) != len(right_tokens) or not left_tokens:
        return False
    return all(_token_forms(a) & _token_forms(b) for a, b in zip(left_tokens, right_tokens))


def spelling_variant(left: str, right: str) -> bool:
    """True for case, punctuation, whitespace, plural, or gerund variants.

    An added or moved negation is not a spelling variant.
    """
    if negation_differs(left, right):
        return False
    return _forms_match(left, right)


def definitions_compatible(left: str, right: str) -> bool:
    """True when two definitions do not contradict and differ only by inflection.

    An empty definition does not contradict a definition that has no negation.
    A negation on one side only is not compatible. Any other wording difference
    waits for the full comparison.
    """
    if negation_differs(left, right):
        return False
    if not _surface_tokens(left) or not _surface_tokens(right):
        return True
    return _forms_match(left, right)


def _polarity_differs(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if opposite_valence(left.get("valence"), right.get("valence")):
        return True
    if negation_differs(str(left.get("code") or ""), str(right.get("code") or "")):
        return True
    return negation_differs(str(left.get("definition") or ""), str(right.get("definition") or ""))


def trivial_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Spelling or inflection variants with compatible definitions, and no polarity flip.

    Token overlap is not enough. ``trusting managers to share information`` and
    ``not trusting managers to share information`` are similar and stay two codes.
    """
    if _polarity_differs(left, right):
        return False
    if not spelling_variant(str(left.get("code") or ""), str(right.get("code") or "")):
        return False
    return definitions_compatible(str(left.get("definition") or ""), str(right.get("definition") or ""))


def lexical_comparison_proposals(
    codes: Sequence[Mapping[str, Any]],
    *,
    threshold: float = TRIVIAL_LABEL_JACCARD,
) -> list[dict]:
    """Label overlap at or above ``threshold`` proposes a comparison. It does not merge.

    Pairs that are spelling variants are left to :func:`trivial_merges`. Every
    other pair, including contradictory near-duplicates, is only a hint.
    """
    ordered = [entry for entry in codes if str(entry.get("code") or "").strip()]
    proposals: list[dict] = []
    for index, left in enumerate(ordered):
        left_label = str(left.get("code") or "")
        left_key = code_key(left_label)
        if not left_key:
            continue
        for right in ordered[index + 1 :]:
            right_label = str(right.get("code") or "")
            right_key = code_key(right_label)
            if not right_key or left_key == right_key or trivial_pair(left, right):
                continue
            score = jaccard(left_label, right_label)
            if score < threshold:
                continue
            proposals.append(
                {
                    "source": left_label,
                    "target": right_label,
                    "score": float(score),
                    "method": "lexical",
                    "proposed": True,
                }
            )
    proposals.sort(key=lambda item: (-float(item["score"]), str(item["source"]).casefold(), str(item["target"]).casefold()))
    return proposals


def trivial_merges(codes: Sequence[Mapping[str, Any]], frozen: set[frozenset[str]] | None = None) -> list[dict]:
    """Auto-merges for spelling and inflection variants. Everything else is only proposed."""
    blocked = frozen or set()
    ordered = sorted(codes, key=lambda entry: code_key(str(entry.get("code") or "")))
    merges: list[dict] = []
    for index, left in enumerate(ordered):
        left_key = code_key(str(left.get("code") or ""))
        if not left_key:
            continue
        for right in ordered[index + 1 :]:
            right_key = code_key(str(right.get("code") or ""))
            if not right_key or frozenset((left_key, right_key)) in blocked:
                continue
            if not trivial_pair(left, right):
                continue
            merges.append(
                {
                    "source": left.get("code"),
                    "target": right.get("code"),
                    "source_valence": str(left.get("valence") or ""),
                    "target_valence": str(right.get("valence") or ""),
                    "rationale": (
                        "These labels differ only by spelling or inflection and their definitions "
                        "are compatible, so the merge did not need a model call."
                    ),
                    "confidence": 0.95,
                    "agreement_confidence": 0.95,
                    "same_action": True,
                    "same_object": True,
                    "same_stance": str(left.get("valence") or "") == str(right.get("valence") or ""),
                    "trivial": True,
                    "action": "accept",
                    "contrast_checked": True,
                    "material_contrast": False,
                    "evidence_same": True,
                    "comparison_op": "alias_duplicate",
                }
            )
    return merges


_ASPECT_STOP = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "for",
    "and",
    "or",
    "in",
    "on",
    "with",
    "after",
    "before",
    "from",
    "without",
    "that",
    "this",
    "into",
    "over",
    "under",
    "than",
    "then",
}
_ASPECT_MODIFIERS = {
    "better",
    "more",
    "less",
    "higher",
    "lower",
    "small",
    "large",
    "very",
    "really",
    "up",
    "any",
    "much",
    "many",
}


def aspect_tokens(text: str) -> set[str]:
    """Content words of a label or definition, without light verbs or modifiers."""
    tokens = set()
    for token in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", str(text or "").casefold()):
        if token in _ASPECT_STOP or token in _ASPECT_MODIFIERS or len(token) < 2:
            continue
        if token.endswith("ing") and len(token) > 5:
            continue
        tokens.add(token)
    return tokens


def aspects_differ(left: str, right: str) -> bool:
    """True when each label names an object or aspect the other does not.

    A longer label that only adds detail to the same words does not differ.
    Sharing one platform name is not enough when each side still has its own object.
    """
    left_tokens = aspect_tokens(left)
    right_tokens = aspect_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return False
    return bool(left_tokens - right_tokens) and bool(right_tokens - left_tokens)


def _definitions_differ(left_label: str, left_def: str, right_label: str, right_def: str) -> bool:
    """True when two long concept statements share no content word.

    The statement is the label plus the definition. A later sharpening can
    drop the object from one definition while the label still names it, and
    that paraphrase is not a difference. Short definitions are not compared.
    """
    left_definition = str(left_def or "").strip()
    right_definition = str(right_def or "").strip()
    if len(left_definition.split()) < 4 or len(right_definition.split()) < 4:
        return False
    left_text = f"{left_label} {left_definition}".strip()
    right_text = f"{right_label} {right_definition}".strip()
    if jaccard(left_text, right_text) >= 0.15:
        return False
    return not (aspect_tokens(left_text) & aspect_tokens(right_text))


def _entry_for_merge(merge: Mapping[str, Any], codebook: Mapping[str, Mapping[str, Any]], side: str) -> dict:
    label = str(merge.get(side) or "")
    entry = dict(codebook.get(code_key(label)) or {})
    entry.setdefault("code", label)
    valence = merge.get(f"{side}_valence")
    if valence and not entry.get("valence"):
        entry["valence"] = valence
    definition = merge.get(f"{side}_definition")
    if definition and not entry.get("definition"):
        entry["definition"] = definition
    return entry


def classify_comparison(merge: Mapping[str, Any], codebook: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """Decide whether an alias is the same concept in the definitions and the quotes.

    ``alias_duplicate`` requires the same action, the same object, a contrast
    check that found no material difference, and evidence that is not mixed.
    Anything else stays ``keep_distinct``, ``defer``, or the operation the
    model already named. Valence is not part of this gate.

    A ``trivial`` flag auto-merges only a spelling or inflection variant whose
    definitions are compatible. Lexical similarity does not skip these checks,
    and a negation or polarity difference never auto-merges.
    """
    named = str(merge.get("comparison_op") or merge.get("op") or "").strip()
    if named in {"keep_distinct", "defer", "link_to_category", "record_dimension", "split_code", "refine_definition"}:
        return named, named
    if merge.get("trivial"):
        source_entry = _entry_for_merge(merge, codebook, "source")
        target_entry = _entry_for_merge(merge, codebook, "target")
        if trivial_pair(source_entry, target_entry):
            return "alias_duplicate", "trivial_pair"
        if _polarity_differs(source_entry, target_entry):
            return "keep_distinct", "polarity_differs"
        left_def = str(source_entry.get("definition") or "")
        right_def = str(target_entry.get("definition") or "")
        if spelling_variant(str(source_entry.get("code") or ""), str(target_entry.get("code") or "")) and not definitions_compatible(
            left_def, right_def
        ):
            return "keep_distinct", "definitions_incompatible"
        return "keep_distinct", "not_spelling_variant"
    fits = list(merge.get("quote_fits") or [])
    if any(isinstance(item, Mapping) and item.get("fits") is False for item in fits):
        return "defer", "mixed_evidence"
    if merge.get("evidence_same") is False:
        return "defer", "mixed_evidence"
    source = str(merge.get("source") or "")
    target = str(merge.get("target") or "")
    if aspects_differ(source, target):
        return "keep_distinct", "aspects_differ"
    source_def = str((codebook.get(code_key(source)) or {}).get("definition") or "")
    target_def = str((codebook.get(code_key(target)) or {}).get("definition") or "")
    if _definitions_differ(source, source_def, target, target_def):
        return "keep_distinct", "definitions_differ"
    if merge.get("same_action") is False or merge.get("same_object") is False:
        return "keep_distinct", "different_action_or_object"
    if merge.get("same_action") is not True or merge.get("same_object") is not True:
        return "keep_distinct", "concept_not_shown"
    if merge.get("contrast_checked") is not True:
        return "defer", "contrast_unchecked"
    if merge.get("material_contrast") is True:
        return "defer", "material_contrast"
    return "alias_duplicate", "same_concept"


def definition_fingerprint(entry: Mapping[str, Any]) -> str:
    """Hash of the definition text. Any edit, including a negation, changes it."""
    return hashlib.sha256(str(entry.get("definition") or "").encode("utf-8")).hexdigest()[:16]


def evidence_fingerprint(entry: Mapping[str, Any]) -> str:
    """Hash of examples and assignments. Frequency is not included."""
    rows: list[str] = []
    for item in list(entry.get("examples") or []) + list(entry.get("assignments") or []):
        if isinstance(item, str):
            rows.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        rows.append(
            "|".join(
                (
                    str(item.get("segment_id") or ""),
                    str(item.get("passage_id") or ""),
                    str(item.get("doc_id") or ""),
                    "" if item.get("doc_start") is None else str(item.get("doc_start")),
                    "" if item.get("doc_end") is None else str(item.get("doc_end")),
                    str(item.get("quote") or ""),
                )
            )
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def _pair_evidence_current(
    row: Mapping[str, Any],
    codebook: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    by_id: Mapping[str, str],
) -> bool:
    """False when either code's definition revision, text, or evidence changed.

    There is no lexical threshold. Adding "not" to a definition, bumping the
    revision, or replacing an example reopens the pair.
    """
    stored_defs = {str(key): str(value or "") for key, value in (row.get("definitions") or {}).items()}
    stored_hashes = {str(key): str(value) for key, value in (row.get("definition_hashes") or {}).items()}
    stored_revs = {str(key): str(value) for key, value in (row.get("definition_revisions") or {}).items()}
    stored_evidence = {str(key): str(value) for key, value in (row.get("evidence_hashes") or {}).items()}
    for item in ids:
        entry = codebook[by_id[item]]
        if stored_hashes:
            if stored_hashes.get(item) != definition_fingerprint(entry):
                return False
        elif definitions_changed_materially(stored_defs.get(item, ""), str(entry.get("definition") or "")):
            return False
        stored_rev = stored_revs.get(item, "")
        current_rev = str(entry.get("revision") or "")
        if stored_rev and current_rev and stored_rev != current_rev:
            return False
        if "evidence_hashes" in row and stored_evidence.get(item, "") != evidence_fingerprint(entry):
            return False
    return True


def _pair_still_current(
    row: Mapping[str, Any],
    codebook: Mapping[str, Mapping[str, Any]],
    by_id: Mapping[str, str],
) -> tuple[str, str] | None:
    """Code keys for a cached pair that still exists with the same definition and evidence."""
    ids = [str(item) for item in (row.get("ids") or []) if str(item)]
    if len(ids) != 2:
        return None
    keys = [by_id.get(item) for item in ids]
    if any(key is None for key in keys) or keys[0] == keys[1]:
        return None
    if not _pair_evidence_current(row, codebook, ids, by_id):
        return None
    return str(keys[0]), str(keys[1])


def frozen_pair_keys(cache: Sequence[Mapping[str, Any]], codebook: Mapping[str, Mapping[str, Any]]) -> set[frozenset[str]]:
    """Pairs already decided this run whose definitions and evidence have not changed.

    The key is the two code keys. A later sweep skips asking the model again
    until a definition revision, the definition text, or the examples or
    assignments of either code change. Token overlap does not keep a pair frozen.
    """
    by_id = _index_ids(codebook)
    frozen: set[frozenset[str]] = set()
    for row in cache:
        keys = _pair_still_current(row, codebook, by_id)
        if keys is not None:
            frozen.add(frozenset(keys))
    return frozen


def replay_accepted_pairs(
    cache: Sequence[Mapping[str, Any]],
    codebook: Mapping[str, Mapping[str, Any]],
) -> list[dict]:
    """Merges to apply again when an accepted pair's codes both exist.

    A later passage can recreate a code an earlier sweep retired. The accept
    stands, so the merge is applied again without another model call while
    both definitions are unchanged. A rejection is not replayed.
    """
    by_id = _index_ids(codebook)
    merges: list[dict] = []
    seen: set[frozenset[str]] = set()
    for row in cache:
        if str(row.get("decision") or "") != "accept":
            continue
        keys = _pair_still_current(row, codebook, by_id)
        if keys is None:
            continue
        left, right = keys
        pair = frozenset((left, right))
        if pair in seen:
            continue
        seen.add(pair)
        merges.append(
            {
                "source": codebook[left].get("code") or left,
                "target": codebook[right].get("code") or right,
                "source_id": str(codebook[left].get("code_id") or ""),
                "target_id": str(codebook[right].get("code_id") or ""),
                "rationale": str(row.get("rationale") or row.get("reason") or ""),
                "confidence": bounded_unit(row.get("confidence"), 1.0),
                "pair_id": "",
                "same_action": row.get("same_action", True),
                "same_object": row.get("same_object", True),
                "same_stance": row.get("same_stance"),
                "contrast_checked": row.get("contrast_checked", True),
                "material_contrast": row.get("material_contrast", False),
                "evidence_same": row.get("evidence_same", True),
                "comparison_op": "alias_duplicate",
            }
        )
    return merges


def remember_pair_decisions(
    cache: Sequence[Mapping[str, Any]],
    codebook: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict]:
    """Record one decision per pair, keyed by the sorted code ids."""
    by_key = _index_ids(codebook)
    id_by_key = {key: code_id for code_id, key in by_key.items()}
    stored = {tuple(sorted(str(item) for item in (row.get("ids") or []))): dict(row) for row in cache}
    for row in rows:
        source = code_key(str(row.get("source") or ""))
        target = code_key(str(row.get("target") or ""))
        if not source or not target or source == target:
            continue
        if source not in id_by_key or target not in id_by_key:
            continue
        ids = tuple(sorted((id_by_key[source], id_by_key[target])))
        decision = row.get("decision") or "reject"
        previous = stored.get(ids)
        current_defs = {
            ids[0]: str(codebook[by_key[ids[0]]].get("definition") or ""),
            ids[1]: str(codebook[by_key[ids[1]]].get("definition") or ""),
        }
        current_hashes = {item: definition_fingerprint(codebook[by_key[item]]) for item in ids}
        current_revs = {item: str(codebook[by_key[item]].get("revision") or "") for item in ids}
        current_evidence = {item: evidence_fingerprint(codebook[by_key[item]]) for item in ids}
        if previous and str(previous.get("decision")) == "reject" and str(decision) == "accept":
            if _pair_evidence_current(previous, codebook, list(ids), by_key):
                continue
        stored[ids] = {
            "ids": list(ids),
            "decision": decision,
            "confidence": bounded_unit(row.get("confidence"), 1.0),
            "reason": str(row.get("reason") or ""),
            "rationale": str(row.get("rationale") or ""),
            "source_valence": str(row.get("source_valence") or codebook[source].get("valence") or ""),
            "target_valence": str(row.get("target_valence") or codebook[target].get("valence") or ""),
            "definitions": current_defs,
            "definition_hashes": current_hashes,
            "definition_revisions": current_revs,
            "evidence_hashes": current_evidence,
            "same_action": row.get("same_action"),
            "same_object": row.get("same_object"),
            "same_stance": row.get("same_stance"),
            "contrast_checked": row.get("contrast_checked"),
            "material_contrast": row.get("material_contrast"),
            "evidence_same": row.get("evidence_same"),
        }
    return [stored[key] for key in sorted(stored)]


def interpret_coding(
    coding: Mapping[str, Any] | Any,
    passage: Mapping[str, Any],
    codebook: Mapping[str, Mapping[str, Any]],
    *,
    max_codes: int,
    require_gerund: bool,
    long_passage_chars: int,
    review_below: float = 0.5,
    context_terms: set[str] | None = None,
    document_text: str = "",
    passages: Sequence[Mapping[str, Any]] | None = None,
    request_passages: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one model response. Does not mutate the codebook.

    A coding action whose confidence is below ``review_below`` is applied and
    flagged ``needs_review`` with reason ``low_confidence``. The same decision
    stays on ``review``. The score is the model's diagnostic, not a probability.
    Unknown code ids are rejected.
    A quote that cannot be anchored does not reject the code. The quote is
    flagged ``possibly_modified`` or ``unanchored`` and coding continues.
    ``quote_retries`` lists those flags so the caller can ask for a closer
    quote once, then keep the code anyway.     A quote located wholly outside
    this passage is also kept. It is flagged ``evidence_outside_passage``.
    One passage that overlaps the matched spans is the attribution. Spans in
    several passages stay on each of those passages. Unresolved attribution
    does not fall back to the passage being coded. That flag does not
    retry or block coding.

    A semicolon-joined label is split first. Each piece is validated on its
    own, and a piece that matches an existing code is reused. An empty label,
    a label over the length limit, or a new label that bundles several
    concepts is listed on ``label_retries`` and still accepted with a review
    flag. Generic labels are rejected.
    """
    payload = coding.model_dump() if hasattr(coding, "model_dump") else dict(coding)
    proposals = expand_joined_labels(list(payload.get("codes") or []))
    rejections: list[dict] = []
    accepted: list[dict] = []
    ops: list[dict] = []
    review: list[dict] = []
    quote_retries: list[dict] = []
    label_retries: list[dict] = []
    seen_keys: set[str] = set()
    low_confidence = bool(payload.get("low_confidence"))
    passage_text = str(passage.get("text") or "")
    passage_id = str(passage.get("passage_id") or "")
    id_to_key = _index_ids(codebook)
    model_passage = str(payload.get("passage_id") or "")
    if model_passage and passage_id and model_passage != passage_id:
        review.append(
            _op(
                op="assign_quote",
                status="review",
                confidence=1.0,
                ids={"passage_id": passage_id, "model_passage_id": model_passage},
                reason="passage_id_mismatch",
            )
        )

    if len(proposals) > max_codes:
        for extra in proposals[max_codes:]:
            rejections.append(
                {
                    "code": extra.get("code", ""),
                    "reason": "over_max_codes",
                    "quote": extra.get("quote", ""),
                }
            )
        proposals = proposals[:max_codes]

    known = {code_key(str(entry.get("code") or entry.get("key") or "")): entry for entry in codebook.values()}
    same_request = list(request_passages) if request_passages is not None else _request_passages(payload, passages)

    for proposal in proposals:
        label = str(proposal.get("code") or "")
        quote = str(proposal.get("quote") or "")
        if is_code_id_text(label):
            if label in id_to_key:
                label = str(codebook[id_to_key[label]].get("code") or label)
            else:
                rejections.append({"code": label, "reason": "code_id_as_label", "quote": quote})
                continue
        passage_start = int(passage.get("start_offset") or 0)
        passage_end = int(passage.get("end_offset") or 0) or (passage_start + len(passage_text))
        anchor = anchor_quote(
            quote,
            passage_text,
            prev_text=str(passage.get("prev_text") or ""),
            next_text=str(passage.get("next_text") or ""),
            document_text=str(passage.get("document_text") or document_text or ""),
            passage_start=passage_start,
            passage_end=passage_end,
            request_passages=same_request,
            exclude_passage_id=passage_id,
        )
        verbatim = anchor.quote
        quote_flag = f"quote_{anchor.status}" if anchor.status in QUOTE_FLAG_STATUSES else ""
        outside = evidence_wholly_outside(anchor, passage_start=passage_start, passage_end=passage_end)
        attribution = attribute_evidence(
            anchor,
            passages,
            doc_id=str(passage.get("doc_id") or ""),
            coded_passage_id=passage_id,
            passage_start=passage_start,
            passage_end=passage_end,
        )
        attributed_passage_id = attribution.primary
        attributed_passage_ids = list(attribution.passage_ids)
        raw_action = str(proposal.get("action") or "").strip().casefold()
        action = _ACTION_ALIASES.get(raw_action)
        provisional_key = code_key(label)
        if action is None:
            action = "reuse" if provisional_key in known or provisional_key in seen_keys else "new"
        exists = provisional_key in known or provisional_key in seen_keys
        if action == "new" and exists:
            action = "reuse"
        if action in {"reuse", "refine"} and not exists:
            action = "new"

        sent_id = str(proposal.get("code_id") or "").strip()
        if action in {"reuse", "refine"} and sent_id:
            if sent_id not in id_to_key:
                rejections.append({"code": label, "reason": "unknown_code_id", "quote": verbatim})
                ops.append(
                    _op(
                        op=action,
                        status="rejected",
                        confidence=bounded_unit(proposal.get("confidence"), 1.0),
                        ids={"code_id": sent_id, "passage_id": passage_id},
                        reason="unknown_code_id",
                    )
                )
                continue
            id_key = id_to_key[sent_id]
            if provisional_key and provisional_key != id_key:
                label = str(codebook[id_key].get("code") or label)
                provisional_key = id_key

        in_vivo = bool(proposal.get("in_vivo"))
        in_vivo_confidence = bounded_unit(proposal.get("in_vivo_confidence"), 1.0)
        ok, reasons, fields = assess_label(
            label,
            in_vivo=in_vivo,
            passage=passage_text,
            require_gerund=require_gerund,
        )
        if not ok:
            reason = reasons[0] if reasons else "rejected"
            matched = provisional_key in known or provisional_key in seen_keys
            if reason in LENIENT_LABEL_REASONS and matched:
                fields = _fields_for_known(known.get(provisional_key) or {}, label, provisional_key)
                reasons = []
            elif reason in LENIENT_LABEL_REASONS:
                fields = lenient_label_fields(label, in_vivo=in_vivo, passage=passage_text)
                reasons = [reason]
            else:
                rejections.append({"code": label, "reason": reason, "quote": verbatim})
                continue
        if quote_flag:
            quote_retries.append(
                {
                    "code": label,
                    "reason": "quote_not_verbatim",
                    "quote": quote,
                    "quote_status": anchor.status,
                    "quote_similarity": anchor.similarity,
                }
            )
        if in_vivo and in_vivo_confidence < review_below:
            fields = dict(fields)
            fields["in_vivo"] = False
            reasons.append("in_vivo_below_review_threshold")

        refined = proposal.get("refined_label")
        refined_label = str(refined).strip() if refined else ""
        if action == "refine" and refined_label:
            refined_ok, refined_reasons, refined_fields = assess_label(
                refined_label,
                in_vivo=in_vivo,
                passage=passage_text,
                require_gerund=require_gerund,
            )
            if not refined_ok:
                reason = refined_reasons[0] if refined_reasons else "rejected"
                if reason in LENIENT_LABEL_REASONS:
                    refined_fields = lenient_label_fields(
                        refined_label,
                        in_vivo=in_vivo,
                        passage=passage_text,
                    )
                    refined_reasons = [reason]
                else:
                    rejections.append({"code": refined_label, "reason": reason, "quote": verbatim})
                    continue
            fields = refined_fields
            reasons = [reason for reason in reasons if reason not in LABEL_HOLD_REASONS]
            reasons.extend(reason for reason in refined_reasons if reason not in reasons)

        prospective_key = str(fields.get("key") or "")
        if prospective_key in seen_keys and action == "new":
            action = "reuse"
        if action == "new" and bundled_concepts(str(fields.get("code") or "")):
            if "bundled_concepts" not in reasons:
                reasons.append("bundled_concepts")
        hold_reasons = [name for name in LABEL_HOLD_REASONS if name in reasons]
        if hold_reasons:
            label_retries.append(
                {
                    "code": label,
                    "source_code": str(proposal.get("code") or ""),
                    "reason": hold_reasons[0],
                    "reasons": hold_reasons,
                    "quote": quote,
                }
            )

        confidence = bounded_unit(proposal.get("confidence"), 1.0)
        if proposal.get("confidence") is not None:
            try:
                float(proposal.get("confidence"))
            except (TypeError, ValueError):
                reasons.append("confidence_unparsed")
        flags = review_passage_flags(
            passage_text,
            quote=verbatim,
            long_passage_chars=long_passage_chars,
            low_confidence=low_confidence,
            confidence=confidence,
            review_below=review_below,
        )
        flags.extend(reasons)
        if quote_flag:
            flags.append(quote_flag)
        if outside:
            flags.append("evidence_outside_passage")
        review_flags = flags
        definition = str(proposal.get("definition") or "").strip()
        if action == "new" and not definition:
            definition = str(proposal.get("rationale") or "").strip() or "Definition not yet specified."
            review_flags.append("missing_definition")
        if action == "new" and known and not str(proposal.get("nearest_code") or "").strip():
            review_flags.append("missing_nearest_code")
        if attribute_leak(definition, passage.get("metadata") or {}):
            review_flags.append("definition_contains_respondent_attribute")
        nearest_ids = [str(item).strip() for item in (proposal.get("nearest_code_ids") or []) if str(item).strip()]
        known_nearest = [item for item in nearest_ids if item in id_to_key]
        if len(known_nearest) != len(nearest_ids):
            review_flags.append("unknown_nearest_code_id")
        if action == "new" and sent_id and sent_id != _code_id(code_key(str(fields.get("code") or label))):
            review_flags.append("code_id_ignored")
        if (
            action == "new"
            and context_terms
            and not proposal.get("context_leak_cleared")
            and label_restates_context(str(fields.get("code") or label), context_terms)
        ):
            review_flags.append("context_restatement")
        op_name = {"new": "create", "reuse": "reuse", "refine": "refine"}[action]
        if confidence < review_below and "low_confidence" not in review_flags:
            review_flags.append("low_confidence")

        key = fields["key"]
        if key in seen_keys and action == "new":
            action = "reuse"
        seen_keys.add(key)
        uncertainty: list[str] = []
        for flag in proposal.get("uncertainty_flags") or []:
            name = str(flag)
            if name in {"ambiguous_span", "insufficient_context", "competing_code", "unsupported_inference"}:
                if name not in uncertainty:
                    uncertainty.append(name)
        if "ambiguous_span" in getattr(anchor, "flags", ()) and "ambiguous_span" not in uncertainty:
            uncertainty.append("ambiguous_span")
        for flag in uncertainty:
            if flag not in review_flags:
                review_flags.append(flag)
        nearest_code_id = known_nearest[0] if known_nearest else None
        evidence_doc_id, evidence_doc_name, evidence_metadata = _evidence_source(
            anchor,
            passage,
            list(same_request) + list(passages or []),
        )
        accepted.append(
            {
                "action": action,
                "key": key,
                "code": fields["code"],
                "previous_key": provisional_key if action == "refine" and key != provisional_key else None,
                "definition": definition,
                "quote": verbatim,
                "model_quote": anchor.model_quote,
                "quote_status": anchor.status,
                "quote_similarity": anchor.similarity,
                "quote_source": anchor.source,
                "quote_spans": [
                    {"start": start, "end": end, "doc_start": doc_start, "doc_end": doc_end}
                    for (start, end), (doc_start, doc_end) in _span_rows(anchor)
                ],
                "local_start": anchor.spans[0][0] if anchor.spans else None,
                "local_end": anchor.spans[-1][1] if anchor.spans else None,
                "doc_start": anchor.doc_start,
                "doc_end": anchor.doc_end,
                "in_vivo": fields["in_vivo"],
                "gerund_ok": fields["gerund_ok"],
                "rationale": str(proposal.get("rationale") or ""),
                "nearest_code": str(proposal.get("nearest_code") or "").strip(),
                "nearest_code_id": nearest_code_id,
                "why_not_nearest": str(proposal.get("why_not_nearest") or "").strip(),
                "nearest_code_ids": known_nearest,
                "uncertainty_flags": uncertainty,
                "confidence": confidence,
                "in_vivo_confidence": in_vivo_confidence,
                "valence": str(proposal.get("valence") or "neutral"),
                "in_vivo_rationale": str(proposal.get("in_vivo_rationale") or proposal.get("rationale") or ""),
                "code_id": _code_id(key),
                "needs_review": bool(review_flags),
                "review_reasons": review_flags,
                "coded_passage_id": passage_id,
                "attributed_passage_id": attributed_passage_id,
                "attributed_passage_ids": attributed_passage_ids,
                "attribution": attribution.status,
                "evidence_outside_passage": outside,
                "evidence_doc_id": evidence_doc_id,
                "evidence_doc_name": evidence_doc_name,
                "evidence_metadata": evidence_metadata,
            }
        )
        applied = _op(
            op=op_name if action != "reuse" else "reuse",
            status="applied",
            confidence=confidence,
            ids={"passage_id": passage_id, "code_id": _code_id(key)},
            reason="",
            rationale=str(proposal.get("rationale") or ""),
        )
        ops.append(applied)
        if confidence < review_below:
            held_confidence = _op(
                op=op_name,
                status="review",
                confidence=confidence,
                ids={"passage_id": passage_id, "code_id": _code_id(key)},
                reason="low_confidence",
                rationale=str(proposal.get("rationale") or ""),
            )
            ops.append(held_confidence)
            review.append(held_confidence)
        if quote_flag:
            held_quote = _op(
                op="assign_quote",
                status="review",
                confidence=confidence,
                ids={
                    "passage_id": passage_id,
                    "code_id": _code_id(key),
                    "quote_status": anchor.status,
                    "quote_similarity": anchor.similarity,
                    "doc_start": anchor.doc_start,
                    "doc_end": anchor.doc_end,
                },
                reason=quote_flag,
                rationale=_quote_flag_rationale(anchor.status, anchor.similarity),
            )
            held_quote["quote_status"] = anchor.status
            held_quote["quote_similarity"] = anchor.similarity
            held_quote["quote"] = verbatim
            held_quote["model_quote"] = anchor.model_quote
            ops.append(held_quote)
            review.append(held_quote)
        if "evidence_outside_passage" in review_flags:
            if attribution.status == "multi":
                outside_rationale = (
                    "The quote matches spans in more than one passage "
                    f"({', '.join(attributed_passage_ids)}). The code was kept and flagged. "
                    "It is not attributed to the passage being coded."
                )
            elif attribution.status == "unresolved":
                outside_rationale = (
                    "The quote is outside this passage and could not be attributed. "
                    "The code was kept and flagged. It was not assigned to the passage being coded."
                )
            else:
                outside_rationale = (
                    "The quote matches text outside this passage. The code was kept and "
                    "attributed to the passage that contains the evidence."
                )
            held_outside = _op(
                op="assign_quote",
                status="review",
                confidence=confidence,
                ids={
                    "passage_id": passage_id,
                    "coded_passage_id": passage_id,
                    "attributed_passage_id": attributed_passage_id,
                    "attributed_passage_ids": attributed_passage_ids,
                    "attribution": attribution.status,
                    "code_id": _code_id(key),
                    "doc_start": anchor.doc_start,
                    "doc_end": anchor.doc_end,
                },
                reason="evidence_outside_passage",
                rationale=outside_rationale,
            )
            ops.append(held_outside)
            review.append(held_outside)
        if "context_restatement" in review_flags:
            held_context = _op(
                op="create",
                status="review",
                confidence=confidence,
                ids={"passage_id": passage_id, "code_id": _code_id(key)},
                reason="context_restatement",
                rationale=str(proposal.get("context_leak_rationale") or proposal.get("rationale") or ""),
            )
            ops.append(held_context)
            review.append(held_context)
        for hold_reason in LABEL_HOLD_REASONS:
            if hold_reason not in review_flags:
                continue
            held_label = _op(
                op=op_name,
                status="review",
                confidence=confidence,
                ids={"passage_id": passage_id, "code_id": _code_id(key)},
                reason=hold_reason,
            )
            ops.append(held_label)
            review.append(held_label)
        if "in_vivo_below_review_threshold" in review_flags:
            held_flag = _op(
                op="in_vivo",
                status="review",
                confidence=in_vivo_confidence,
                ids={"passage_id": passage_id, "code_id": _code_id(key)},
                reason="in_vivo_below_review_threshold",
                rationale=str(proposal.get("in_vivo_rationale") or proposal.get("rationale") or ""),
            )
            ops.append(held_flag)
            review.append(held_flag)
        elif fields["in_vivo"]:
            ops.append(
                _op(
                    op="in_vivo",
                    status="applied",
                    confidence=in_vivo_confidence,
                    ids={"passage_id": passage_id, "code_id": _code_id(key)},
                    rationale=str(proposal.get("in_vivo_rationale") or proposal.get("rationale") or ""),
                )
            )

    memo_body = str(payload.get("memo") or "").strip()
    memo_confidence = bounded_unit(payload.get("memo_confidence"), 1.0)
    memo_reasons = memo_review_reasons(memo_body)
    if memo_reasons is None:
        memo_body = ""
        memo_reasons = []
    elif memo_body and memo_confidence < review_below and "low_confidence" not in memo_reasons:
        memo_reasons.append("low_confidence")
    if memo_body:
        for reason in memo_reasons:
            held_memo = _op(
                op="memo",
                status="review",
                confidence=memo_confidence,
                ids={"passage_id": passage_id},
                reason=reason,
                rationale=memo_body,
            )
            ops.append(held_memo)
            review.append(held_memo)
        ops.append(
            _op(
                op="memo",
                status="applied",
                confidence=memo_confidence,
                ids={"passage_id": passage_id},
                reason="",
                rationale=memo_body,
            )
        )

    if not accepted and len(passage_text) >= long_passage_chars:
        rejections.append({"code": "", "reason": "uncoded_long_passage", "quote": ""})

    return {
        "accepted": accepted,
        "rejections": rejections,
        "memo": memo_body,
        "memo_confidence": memo_confidence,
        "memo_reasons": memo_reasons,
        "low_confidence": low_confidence,
        "ops": ops,
        "review": review,
        "quote_retries": quote_retries,
        "label_retries": label_retries,
        "expanded_codes": proposals,
    }


def commit_passage(
    codebook: dict[str, dict],
    segments: list[dict],
    memos: list[dict],
    highlight_ids: dict[str, str],
    counters: dict[str, int],
    passage: Mapping[str, Any],
    interpreted: Mapping[str, Any],
    *,
    existing_keys: set[str] | None = None,
    origin_snapshot_id: str = "",
) -> dict[str, Any]:
    """Append segments for accepted codes. ``existing_keys`` tracks globally new labels.

    Unchanged codebook entries and existing segment dicts are shared with the
    caller until this function copies what it is about to change. A rename
    copies a segment dict before it rewrites the label. Counts are recomputed
    only for codes this commit touches.

    The codebook container is still a new dict, and every entry is still
    shallow-detached. A later sweep edits definitions on the state this commit
    returns. Those edits must not rewrite the checkpoint that holds the
    previous entry objects, so the detach stays on every entry even when the
    commit does not change that code. Nested history events stay shared.
    The segment list is a new list; segment dicts that this commit does not
    rewrite stay shared.

    An append-only commit updates counts from the new segments plus the
    assignment list already stored on the entry. It does not walk the whole
    segment list. :func:`sync_counts` still scans every segment, because a
    merge or a split moves assignments between codes.
    """
    book = {key: _detach_entry(entry) for key, entry in codebook.items()}
    new_segments = list(segments)
    new_memos = list(memos)
    highlights = dict(highlight_ids)
    counters = dict(counters)
    known_before = set(existing_keys) if existing_keys is not None else set(book)
    new_codes = 0
    touched: set[str] = set()
    appended: list[dict] = []

    memo_body = str(interpreted.get("memo") or "").strip()
    memo_reasons = interpreted.get("memo_reasons")
    if memo_reasons is None:
        reviewed = memo_review_reasons(memo_body)
        if reviewed is None:
            memo_body = ""
            memo_reasons = []
        else:
            memo_reasons = reviewed
    else:
        memo_reasons = [str(reason) for reason in memo_reasons if str(reason)]
        if not memo_body:
            memo_reasons = []
    memo_id = None
    if memo_body:
        memo_id = _next(counters, "memo", "m")
        new_memos.append(
            {
                "memo_id": memo_id,
                "kind": "passage",
                "doc_id": passage.get("doc_id"),
                "passage_id": passage.get("passage_id"),
                "sweep_index": None,
                "text": memo_body,
                "needs_review": bool(memo_reasons),
                "review_reasons": list(memo_reasons),
            }
        )

    for item in interpreted.get("accepted") or []:
        key = item["key"]
        previous = item.get("previous_key")
        if previous and previous in book and previous != key:
            book = _rename_code(book, new_segments, previous, key, item["code"])
            touched.add(previous)
            touched.add(key)
        if key not in book:
            if key not in known_before:
                new_codes += 1
                known_before.add(key)
            book[key] = {
                "code": item["code"],
                "key": key,
                "code_id": _code_id(key),
                "definition": item["definition"],
                "in_vivo": item["in_vivo"],
                "valence": item.get("valence") or "neutral",
                "gerund_ok": item["gerund_ok"],
                "frequency": 0,
                "document_ids": [],
                "examples": [],
                "assignments": [],
                "revision": 1,
                "inclusion": "",
                "exclusion": "",
                "lifecycle": "active",
                "anchor_span_ids": [],
                "contrast_span_ids": [],
                "category_links": [],
                "first_seen": passage.get("passage_id"),
                "origin_snapshot_id": str(origin_snapshot_id or ""),
                "history": [
                    {
                        "event": "created",
                        "passage_id": passage.get("passage_id"),
                        "definition": item["definition"],
                        "action": item["action"],
                    }
                ],
                "flagged_overloaded": False,
                "relation_hints": [],
            }
        elif item["action"] == "refine" and item["definition"]:
            book[key]["definition"] = item["definition"]
            book[key]["revision"] = int(book[key].get("revision") or 1) + 1
            book[key]["history"].append(
                {
                    "event": "refined",
                    "passage_id": passage.get("passage_id"),
                    "definition": item["definition"],
                }
            )
            book[key]["in_vivo"] = bool(book[key].get("in_vivo") or item["in_vivo"])
            book[key]["gerund_ok"] = item["gerund_ok"]
        elif item["in_vivo"]:
            book[key]["in_vivo"] = True
        _mark_label_review(book[key], item.get("review_reasons") or [])

        evidence_doc_id = str(item.get("evidence_doc_id") or passage.get("doc_id") or "")
        evidence_doc_name = item.get("evidence_doc_name")
        if evidence_doc_name in (None, ""):
            evidence_doc_name = passage.get("doc_name")
        evidence_metadata = item.get("evidence_metadata")
        if not isinstance(evidence_metadata, Mapping):
            evidence_metadata = dict(passage.get("metadata") or {})
        else:
            evidence_metadata = dict(evidence_metadata)
        if item.get("doc_start") is None or item.get("doc_end") is None:
            span_key = (
                f"{evidence_doc_id}:{passage.get('passage_id')}:"
                f"{item.get('quote_status')}:{item.get('model_quote')}"
            )
        else:
            span_key = f"{evidence_doc_id}:{item['doc_start']}:{item['doc_end']}"
        if span_key not in highlights:
            highlights[span_key] = _next(counters, "highlight", "h")
        coded_passage_id = str(passage.get("passage_id") or "")
        attribution = str(item.get("attribution") or "")
        attributed_ids = [str(value) for value in (item.get("attributed_passage_ids") or []) if str(value)]
        if attribution in {"multi", "unresolved"}:
            # Do not count the quote on the passage that was being coded.
            attributed_passage_id = ""
            passage_id_for_row = ""
        elif attribution in {"inside", "single"}:
            attributed_passage_id = str(item.get("attributed_passage_id") or "") or (
                attributed_ids[0] if len(attributed_ids) == 1 else ""
            )
            passage_id_for_row = attributed_passage_id
            if attributed_passage_id and attributed_passage_id not in attributed_ids:
                attributed_ids = [attributed_passage_id]
        else:
            attributed_passage_id = str(item.get("attributed_passage_id") or "") or coded_passage_id
            passage_id_for_row = attributed_passage_id
            if not attributed_ids and attributed_passage_id:
                attributed_ids = [attributed_passage_id]
        segment = {
            "segment_id": _next(counters, "segment", "s"),
            "highlight_id": highlights[span_key],
            "passage_id": passage_id_for_row,
            "coded_passage_id": coded_passage_id,
            "attributed_passage_id": attributed_passage_id,
            "attributed_passage_ids": attributed_ids,
            "attribution": attribution,
            "doc_id": evidence_doc_id,
            "doc_name": evidence_doc_name,
            "code": book[key]["code"],
            "code_key": key,
            "quote": item["quote"],
            "model_quote": item.get("model_quote") or item["quote"],
            "quote_status": item.get("quote_status") or "",
            "quote_similarity": item.get("quote_similarity"),
            "quote_source": item.get("quote_source") or "",
            "quote_spans": list(item.get("quote_spans") or []),
            "doc_start": item["doc_start"],
            "doc_end": item["doc_end"],
            "code_id": book[key].get("code_id") or _code_id(key),
            "in_vivo": item["in_vivo"],
            "in_vivo_confidence": item.get("in_vivo_confidence", 1.0),
            "in_vivo_rationale": item.get("in_vivo_rationale") or "",
            "valence": item.get("valence") or book[key].get("valence") or "neutral",
            "action": item["action"],
            "rationale": item["rationale"],
            "nearest_code": item.get("nearest_code") or "",
            "nearest_code_id": item.get("nearest_code_id"),
            "why_not_nearest": item.get("why_not_nearest") or "",
            "uncertainty_flags": list(item.get("uncertainty_flags") or []),
            "metadata": evidence_metadata,
            "provenance": _coded_provenance(passage),
            "confidence": item["confidence"],
            "needs_review": item["needs_review"],
            "review_reasons": list(item["review_reasons"]),
            "nonsubstantive": bool(passage.get("nonsubstantive")),
            "nonsubstantive_kind": str(passage.get("nonsubstantive_kind") or ""),
            "memo_id": memo_id,
        }
        targets = [pid for pid in attributed_ids if pid]
        if passage_id_for_row and passage_id_for_row not in targets:
            targets.append(passage_id_for_row)
        # Every incoming assignment, including one that lands on the passage
        # being coded. Identity is the code, the source document, and the
        # matched spans. The same words at another offset stay a second quote.
        duplicate_at = _find_same_evidence(new_segments, key, evidence_doc_id, item)
        if duplicate_at is not None:
            new_segments[duplicate_at] = _merge_cross_passage(
                new_segments[duplicate_at],
                coded_passage_id=coded_passage_id,
                coded_doc_id=str(passage.get("doc_id") or ""),
                coded_doc_name=str(passage.get("doc_name") or ""),
                targets=targets,
                item=item,
            )
            continue
        new_segments.append(segment)
        appended.append(segment)
        touched.add(key)

    book = _refresh_counts(book, new_segments, touched, added=appended)
    ops = []
    for op in interpreted.get("ops") or []:
        row = dict(op)
        row["op_id"] = _next(counters, "op", "o")
        ops.append(row)
    return {
        "codebook": book,
        "segments": new_segments,
        "memos": new_memos,
        "highlight_ids": highlights,
        "counters": counters,
        "new_codes": new_codes,
        "ops": ops,
    }


def _request_passages(
    payload: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Passages that shared the model request, when the coding recorded their ids."""
    ids = [str(item) for item in (payload.get("request_passage_ids") or []) if str(item)]
    if not ids or not passages:
        return []
    wanted = set(ids)
    return [item for item in passages if str(item.get("passage_id") or "") in wanted]


def _evidence_source(
    anchor: Any,
    passage: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
) -> tuple[str, str, dict]:
    """Document identity of the located evidence, not of the passage being coded.

    A same-request hit can sit in another document. The segment's document,
    name, and respondent metadata come from that passage. The coded passage
    stays on provenance.
    """
    located_id = str(getattr(anchor, "located_passage_id", "") or "")
    located = None
    if located_id and getattr(anchor, "source", "") == "request":
        for item in passages:
            if str(item.get("passage_id") or "") == located_id:
                located = item
                break
    if located is not None:
        return (
            str(located.get("doc_id") or getattr(anchor, "located_doc_id", "") or ""),
            str(located.get("doc_name") or getattr(anchor, "located_doc_name", "") or ""),
            dict(located.get("metadata") or {}),
        )
    if getattr(anchor, "source", "") == "request" and str(getattr(anchor, "located_doc_id", "") or ""):
        return (
            str(anchor.located_doc_id),
            str(getattr(anchor, "located_doc_name", "") or ""),
            {},
        )
    return (
        str(passage.get("doc_id") or ""),
        str(passage.get("doc_name") or ""),
        dict(passage.get("metadata") or {}),
    )


def _coded_provenance(passage: Mapping[str, Any]) -> dict:
    """The passage and document that were being coded, kept apart from the evidence."""
    passage_id = str(passage.get("passage_id") or "")
    doc_id = str(passage.get("doc_id") or "")
    doc_name = str(passage.get("doc_name") or "")
    return {
        "coded_passage_id": passage_id,
        "coded_doc_id": doc_id,
        "coded_doc_name": doc_name,
        "coded_documents": [
            {"passage_id": passage_id, "doc_id": doc_id, "doc_name": doc_name},
        ],
    }


def _span_identity(item: Mapping[str, Any]) -> tuple[tuple[int, int], ...] | None:
    """Matched document spans. The quote text is not an identity."""
    spans: list[tuple[int, int]] = []
    for span in item.get("quote_spans") or []:
        if not isinstance(span, Mapping):
            continue
        if span.get("doc_start") is None or span.get("doc_end") is None:
            continue
        spans.append((int(span["doc_start"]), int(span["doc_end"])))
    if not spans and item.get("doc_start") is not None and item.get("doc_end") is not None:
        spans.append((int(item["doc_start"]), int(item["doc_end"])))
    if not spans:
        return None
    return tuple(spans)


def _norm_quote(quote: Any) -> str:
    return re.sub(r"\s+", " ", str(quote or "").strip()).casefold()


def _is_cross_passage(item: Mapping[str, Any], coded_passage_id: str, targets: Sequence[str]) -> bool:
    """True when this quote is evidence for a passage other than the one being coded."""
    if not targets:
        return False
    if item.get("evidence_outside_passage"):
        return True
    return coded_passage_id not in targets


def _find_same_evidence(
    segments: Sequence[Mapping[str, Any]],
    key: str,
    doc_id: str,
    item: Mapping[str, Any],
) -> int | None:
    """An existing assignment of this code, document, and span.

    Text equality is not enough. The same words at a different offset are a
    different quote and stay a second assignment.
    """
    spans = _span_identity(item)
    source = str(doc_id or "")
    if spans is None or not source:
        return None
    for index, segment in enumerate(segments):
        if _segment_key(segment) != key:
            continue
        if str(segment.get("doc_id") or "") != source:
            continue
        if _span_identity(segment) == spans:
            return index
    return None


def _merge_cross_passage(
    segment: Mapping[str, Any],
    *,
    coded_passage_id: str,
    coded_doc_id: str = "",
    coded_doc_name: str = "",
    targets: Sequence[str],
    item: Mapping[str, Any],
) -> dict:
    """Keep one segment. Record the passage that was being coded on its provenance."""
    updated = dict(segment)
    provenance = dict(updated.get("provenance") or {})
    coded_ids = [str(value) for value in (provenance.get("coded_passage_ids") or []) if str(value)]
    original = str(updated.get("coded_passage_id") or provenance.get("coded_passage_id") or "")
    if original and original not in coded_ids:
        coded_ids.append(original)
    if coded_passage_id and coded_passage_id not in coded_ids:
        coded_ids.append(coded_passage_id)
    provenance["coded_passage_ids"] = coded_ids
    documents = [dict(row) for row in (provenance.get("coded_documents") or []) if isinstance(row, Mapping)]

    def _add_doc(passage_id: str, doc_id: str, doc_name: str) -> None:
        if not passage_id and not doc_id:
            return
        marker = (passage_id, doc_id)
        if any((str(row.get("passage_id") or ""), str(row.get("doc_id") or "")) == marker for row in documents):
            return
        documents.append({"passage_id": passage_id, "doc_id": doc_id, "doc_name": doc_name})

    _add_doc(
        original,
        str(provenance.get("coded_doc_id") or ""),
        str(provenance.get("coded_doc_name") or ""),
    )
    _add_doc(coded_passage_id, coded_doc_id, coded_doc_name)
    provenance["coded_documents"] = documents
    if coded_passage_id and not provenance.get("coded_passage_id"):
        provenance["coded_passage_id"] = coded_passage_id
    updated["provenance"] = provenance
    current = [str(value) for value in (updated.get("attributed_passage_ids") or []) if str(value)]
    if not current:
        single = str(updated.get("attributed_passage_id") or updated.get("passage_id") or "")
        if single:
            current = [single]
    for target in targets:
        if target and target not in current:
            current.append(target)
    updated["attributed_passage_ids"] = current
    if len(current) == 1:
        updated["attributed_passage_id"] = current[0]
        if not updated.get("passage_id"):
            updated["passage_id"] = current[0]
    elif len(current) > 1 and str(updated.get("attribution") or "") == "multi":
        updated["attributed_passage_id"] = ""
    reasons = list(updated.get("review_reasons") or [])
    if item.get("evidence_outside_passage") and "evidence_outside_passage" not in reasons:
        reasons.append("evidence_outside_passage")
    updated["review_reasons"] = reasons
    updated["needs_review"] = bool(reasons)
    return updated


def _rename_code(codebook: dict[str, dict], segments: list[dict], old: str, new: str, display: str) -> dict[str, dict]:
    book = codebook
    if old not in book:
        return book
    entry = book.pop(old)
    entry["key"] = new
    entry["code"] = display
    entry.setdefault("history", []).append({"event": "renamed", "from": old, "to": new})
    if new in book:
        survivor = book[new]
        survivor.setdefault("history", []).extend(entry.get("history") or [])
        survivor["history"].append({"event": "merged_from", "code": entry.get("code"), "key": old})
        if entry.get("definition") and not survivor.get("definition"):
            survivor["definition"] = entry["definition"]
    else:
        # The id stays with the concept. A new label is a new revision, not a new code.
        entry["revision"] = int(entry.get("revision") or 1) + 1
        book[new] = entry
    for index, segment in enumerate(segments):
        if segment.get("code_key") != old:
            continue
        # Share the caller's segment dict until a rename actually rewrites it.
        segment = dict(segment)
        segments[index] = segment
        segment["code_key"] = new
        segment["code"] = book[new]["code"] if new in book else display
        segment["code_id"] = book[new].get("code_id") or _code_id(new)
    return book


_ENTRY_LISTS = (
    "history",
    "examples",
    "assignments",
    "reviewed_assignment_ids",
    "available_assignment_ids",
    "reviews_seen",
    "document_ids",
    "review_reasons",
    "relation_hints",
    "anchor_span_ids",
    "contrast_span_ids",
    "category_links",
)


def _detach_entry(entry: Mapping[str, Any]) -> dict:
    """Shallow copy an entry and the lists a later commit or sweep appends to.

    Nested history events stay shared. Those dicts are not edited in place.
    """
    copied = dict(entry)
    for field in _ENTRY_LISTS:
        value = copied.get(field)
        if isinstance(value, list):
            copied[field] = list(value)
    return copied


def _assignment_digest(applications: Sequence[Mapping[str, Any]]) -> str:
    """Hash every assignment, not a prefix of the stored examples."""
    rows = []
    for item in applications:
        rows.append(
            "|".join(
                (
                    str(item.get("segment_id") or ""),
                    str(item.get("passage_id") or ""),
                    str(item.get("doc_id") or ""),
                    "" if item.get("doc_start") is None else str(item.get("doc_start")),
                    "" if item.get("doc_end") is None else str(item.get("doc_end")),
                    str(item.get("quote") or ""),
                    str(item.get("code_key") or ""),
                )
            )
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def _assignment_row(item: Mapping[str, Any]) -> dict:
    """One stored assignment. Every application is kept, not a prefix of eight."""
    ids = [str(value) for value in (item.get("attributed_passage_ids") or []) if str(value)]
    return {
        "segment_id": str(item.get("segment_id") or ""),
        "quote": item.get("quote"),
        "doc_id": item.get("doc_id"),
        "passage_id": item.get("passage_id"),
        "coded_passage_id": item.get("coded_passage_id") or "",
        "attributed_passage_id": item.get("attributed_passage_id") or "",
        "attributed_passage_ids": ids,
        "attribution": item.get("attribution") or "",
        "doc_start": item.get("doc_start"),
        "doc_end": item.get("doc_end"),
        "quote_status": item.get("quote_status") or "",
        "quote_similarity": item.get("quote_similarity"),
        "confidence": item.get("confidence"),
        "code_key": item.get("code_key") or "",
    }


def _apply_counts(entry: Mapping[str, Any], applications: Sequence[Mapping[str, Any]], key: str) -> dict:
    """Frequency, documents, the full assignment list, and its digest.

    ``assignments`` is every application, in order. ``examples`` is that same
    history with duplicate passage/quote pairs removed, and it is not cut at
    eight. Prompt selection reads ``assignments`` and keeps its own bound.
    """
    updated = _detach_entry(entry)
    updated["key"] = key
    updated["frequency"] = len(applications)
    updated["document_ids"] = sorted({str(item.get("doc_id")) for item in applications if item.get("doc_id")})
    assignments = [_assignment_row(item) for item in applications]
    examples = []
    seen_quotes: set[tuple] = set()
    for item in assignments:
        marker = (item.get("passage_id"), item.get("quote"), item.get("segment_id"))
        if marker in seen_quotes:
            continue
        seen_quotes.add(marker)
        examples.append(item)
    updated["assignments"] = assignments
    updated["examples"] = examples
    updated["evidence_revision"] = _assignment_digest(applications)
    scores = []
    for item in applications:
        if item.get("confidence") is None or item.get("confidence") == "":
            continue
        try:
            scores.append(float(item["confidence"]))
        except (TypeError, ValueError):
            continue
    if scores:
        updated["confidence_mean"] = round(sum(scores) / len(scores), 4)
        updated["confidence_min"] = min(scores)
    if applications and not updated.get("first_seen"):
        updated["first_seen"] = applications[0].get("passage_id")
    label = str(updated.get("code") or "")
    evidence = assignments or examples
    if updated.get("in_vivo") and not any(
        code_key(label) and label.casefold() in str(item.get("quote") or "").casefold() for item in evidence
    ):
        from automated_grounded_theory.validation import in_vivo_supported

        updated["in_vivo"] = any(in_vivo_supported(label, str(item.get("quote") or "")) for item in evidence)
    return updated


def _segment_key(segment: Mapping[str, Any]) -> str:
    return str(segment.get("code_key") or code_key(str(segment.get("code") or "")))


def _refresh_counts(
    book: dict[str, dict],
    segments: Sequence[Mapping[str, Any]],
    touched: set[str],
    *,
    added: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict]:
    """Recompute counts for ``touched`` codes. Other entries stay shared.

    ``added`` is the segments this commit appended. When every touched entry
    already stores ``assignments``, those lists are extended and the rest of
    ``segments`` is not scanned. A missing list falls back to a scan of the
    touched codes. Merges use :func:`sync_counts`, which always scans.
    """
    if not touched:
        return book
    if added is not None and all("assignments" in book[key] for key in touched if key in book):
        grouped: dict[str, list] = {key: [] for key in touched}
        for segment in added:
            key = _segment_key(segment)
            if key in grouped:
                grouped[key].append(segment)
        for key in touched:
            if key not in book:
                continue
            previous = [item for item in (book[key].get("assignments") or []) if isinstance(item, Mapping)]
            known = {str(item.get("segment_id")) for item in previous if item.get("segment_id")}
            extra = []
            for segment in grouped.get(key) or []:
                segment_id = str(segment.get("segment_id") or "")
                if segment_id and segment_id in known:
                    continue
                extra.append(segment)
            applications = previous + extra
            if not applications:
                book.pop(key, None)
                continue
            book[key] = _apply_counts(book[key], applications, key)
        return book
    grouped = {key: [] for key in touched}
    for segment in segments:
        key = _segment_key(segment)
        if key in grouped:
            grouped[key].append(segment)
    for key in touched:
        if key not in book:
            continue
        applications = grouped.get(key) or []
        if not applications:
            book.pop(key, None)
            continue
        book[key] = _apply_counts(book[key], applications, key)
    return book


def sync_counts(codebook: dict[str, dict], segments: list[dict]) -> dict[str, dict]:
    """Rebuild frequency, documents, and examples from ``segments``."""
    grouped: dict[str, list[dict]] = {}
    for segment in segments:
        grouped.setdefault(_segment_key(segment), []).append(segment)
    synced: dict[str, dict] = {}
    for key, entry in codebook.items():
        applications = grouped.get(key) or []
        if not applications:
            continue
        synced[key] = _apply_counts(deepcopy(entry), applications, key)
    return synced


def union_codebooks(books: list[dict[str, dict]]) -> dict[str, dict]:
    """Combine per-document codebooks. Counts are left for :func:`sync_counts`."""
    united: dict[str, dict] = {}
    for book in books:
        for key, entry in book.items():
            if key not in united:
                united[key] = deepcopy(entry)
                continue
            current = united[key]
            current.setdefault("history", []).extend(entry.get("history") or [])
            if len(str(entry.get("definition") or "")) > len(str(current.get("definition") or "")):
                current["definition"] = entry["definition"]
            current["in_vivo"] = bool(current.get("in_vivo") or entry.get("in_vivo"))
            current["flagged_overloaded"] = bool(current.get("flagged_overloaded") or entry.get("flagged_overloaded"))
            hints = list(current.get("relation_hints") or [])
            for hint in entry.get("relation_hints") or []:
                if hint not in hints:
                    hints.append(hint)
            current["relation_hints"] = hints
    return united


def _audit_alias(record: Mapping[str, Any], codebook: Mapping[str, Mapping[str, Any]], segments: Sequence[Mapping[str, Any]]) -> dict:
    """One accepted alias, with both definitions and a few member quotes."""
    source = code_key(str(record.get("source") or ""))
    target = code_key(str(record.get("target") or ""))

    def quotes(key: str) -> list[str]:
        found = []
        for segment in segments:
            seg_key = segment.get("code_key") or code_key(str(segment.get("code") or ""))
            quote = str(segment.get("quote") or "").strip()
            if seg_key == key and quote:
                found.append(quote)
        return found[:3]

    return {
        "kind": "accepted_alias",
        "source": record.get("source"),
        "target": record.get("target"),
        "source_definition": str((codebook.get(source) or {}).get("definition") or ""),
        "target_definition": str((codebook.get(target) or {}).get("definition") or ""),
        "source_quotes": quotes(source),
        "target_quotes": quotes(target),
        "rationale": str(record.get("rationale") or ""),
        "comparison_op": "alias_duplicate",
        "later_split": False,
    }


def mark_later_splits(candidates: list[dict], split_merges: Sequence[Mapping[str, Any]]) -> int:
    """Mark accepted aliases that a later split pulled apart. Returns how many."""
    if not split_merges:
        return 0
    keys = set()
    for item in split_merges:
        left = code_key(str(item.get("source") or ""))
        right = code_key(str(item.get("target") or ""))
        if left and right:
            keys.add(frozenset((left, right)))
    count = 0
    for row in candidates:
        if row.get("kind") != "accepted_alias" or row.get("later_split"):
            continue
        pair = frozenset(
            (code_key(str(row.get("source") or "")), code_key(str(row.get("target") or "")))
        )
        if pair in keys:
            row["later_split"] = True
            count += 1
    return count


def apply_post_merge(
    codebook: dict[str, dict],
    segments: list[dict],
    accepted: Sequence[Mapping[str, Any]],
    pre_merge_book: Mapping[str, Mapping[str, Any]],
    pre_segments: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict], list[dict], list[dict], dict[str, int]]:
    """Rewrite each survivor from the merge, then split quotes that do not fit.

    ``rewritten_label`` and ``rewritten_definition`` come from the same sweep
    response as the merge. A quote with ``fits`` false moves back to ``split_to``
    or to the code it had before the merge.
    """
    book = codebook
    rows = [dict(segment) for segment in segments]
    original: dict[str, str] = {}
    for segment in pre_segments:
        quote = str(segment.get("quote") or "").strip()
        key = str(segment.get("code_key") or code_key(str(segment.get("code") or "")))
        if quote and key and quote not in original:
            original[quote] = key
    ops: list[dict] = []
    checked = 0
    split = 0
    split_merges: list[dict] = []
    for merge in accepted:
        source = code_key(str(merge.get("source") or ""))
        target = code_key(str(merge.get("target") or ""))
        survivor = source if source in book else target if target in book else ""
        if not survivor:
            continue
        label = str(merge.get("rewritten_label") or "").strip()
        definition = str(merge.get("rewritten_definition") or "").strip()
        rationale = str(merge.get("rationale") or "")
        confidence = float(merge.get("confidence") or 0)
        if label and code_key(label) != survivor:
            new_key = code_key(label)
            book = _rename_code(book, rows, survivor, new_key, label)
            survivor = new_key if new_key in book else survivor
            ops.append(
                _op(
                    op="refine_definition",
                    status="applied",
                    confidence=confidence,
                    ids={"code_id": str(book.get(survivor, {}).get("code_id") or "")},
                    rationale=rationale or f"The survivor label was rewritten to {label}.",
                )
            )
        if definition and survivor in book:
            book[survivor]["definition"] = definition
            book[survivor].setdefault("history", []).append({"event": "definition_rewritten", "definition": definition})
            ops.append(
                _op(
                    op="refine_definition",
                    status="applied",
                    confidence=confidence,
                    ids={"code_id": str(book[survivor].get("code_id") or "")},
                    rationale=rationale or definition,
                )
            )
        moved_any = False
        for check in merge.get("quote_fits") or []:
            checked += 1
            fits = check.get("fits", True)
            if isinstance(check, Mapping) and fits:
                continue
            quote = str(check.get("quote") or "").strip()
            if not quote:
                continue
            split_to = str(check.get("split_to") or "").strip()
            dest = code_key(split_to) if split_to else original.get(quote, "")
            if not dest or dest == survivor:
                dest = original.get(quote, dest)
            if not dest or dest == survivor:
                continue
            if dest not in book:
                previous = pre_merge_book.get(dest)
                if previous:
                    restored = deepcopy(dict(previous))
                    restored["key"] = dest
                    book[dest] = restored
                elif split_to:
                    book[dest] = {
                        "code": split_to,
                        "key": dest,
                        "code_id": _code_id(dest),
                        "definition": f"Incidents of {split_to}.",
                        "valence": "neutral",
                        "history": [{"event": "split_out", "quote": quote}],
                        "frequency": 0,
                        "document_ids": [],
                        "examples": [],
                    }
                else:
                    continue
            moved = False
            for segment in rows:
                if str(segment.get("quote") or "").strip() != quote:
                    continue
                if segment.get("code_key") != survivor:
                    continue
                segment["code_key"] = dest
                segment["code"] = book[dest].get("code") or split_to or dest
                segment["code_id"] = book[dest].get("code_id") or _code_id(dest)
                moved = True
                break
            if not moved:
                continue
            split += 1
            moved_any = True
            ops.append(
                _op(
                    op="split_quote",
                    status="applied",
                    confidence=bounded_unit(check.get("confidence"), confidence or 1.0),
                    ids={"code_id": str(book[dest].get("code_id") or ""), "quote": quote},
                    reason="quote_split",
                    rationale=str(check.get("rationale") or ""),
                )
            )
        if moved_any:
            split_merges.append({"source": merge.get("source"), "target": merge.get("target")})
    book = sync_counts(book, rows)
    return book, rows, ops, {
        "quotes_checked": checked,
        "quotes_split": split,
        "merges_later_split": len(split_merges),
        "split_merges": split_merges,
    }


def plan_in_batch_aliases(
    proposals: Sequence[Mapping[str, Any]],
    merges: Sequence[Mapping[str, Any]],
    *,
    min_merge_confidence: float = 0.5,
) -> dict[str, Any]:
    """Decide which new codes from one round are the same concept.

    The proposals are not in the frozen codebook. ``merges`` are the model's
    structured decisions. The same alias gate used by a sweep applies, so a
    mixed or different concept stays two codes. Returns label rewrites for the
    commit and the pair rows to export apart from cross-snapshot duplicates.
    """
    book: dict[str, dict] = {}
    segments: list[dict] = []
    for item in proposals:
        label = str(item.get("code") or "").strip()
        key = code_key(label)
        if not key or key in book:
            continue
        book[key] = {
            "code": label,
            "key": key,
            "definition": str(item.get("definition") or ""),
            "valence": str(item.get("valence") or "neutral"),
            "code_id": "",
            "history": [],
        }
        for quote in item.get("quotes") or []:
            text = str(quote or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "code": label,
                    "code_key": key,
                    "quote": text,
                    "passage_id": str(item.get("passage_id") or ""),
                }
            )
    if len(book) < 2:
        return {"rewrites": {}, "definitions": {}, "pairs": []}
    applied = apply_sweep(
        book,
        segments,
        merges=list(merges),
        candidates=[],
        min_merge_confidence=min_merge_confidence,
        max_merges_per_sweep=None,
    )
    rewrites: dict[str, str] = {}
    for retired, survivor in (applied.get("merge_map") or {}).items():
        retired_key = code_key(str(retired))
        if retired_key and survivor:
            rewrites[retired_key] = str(survivor)
    definitions: dict[str, str] = {}
    for merge in applied.get("accepted_merges") or []:
        definition = str(merge.get("rewritten_definition") or "").strip()
        if not definition:
            continue
        source = code_key(str(merge.get("source") or ""))
        target = code_key(str(merge.get("target") or ""))
        post = applied["codebook"]
        survivor = target if target in post else source if source in post else ""
        if survivor:
            definitions[survivor] = definition
    pairs = []
    for merge in list(applied.get("accepted_merges") or []) + list(applied.get("rejected_merges") or []):
        pairs.append(
            {
                "scope": "in_batch",
                "source": merge.get("source"),
                "target": merge.get("target"),
                "comparison_op": merge.get("comparison_op") or merge.get("decision") or "",
                "decision": merge.get("decision") or "",
                "reason": merge.get("reason") or "",
                "confidence": merge.get("confidence"),
                "rationale": merge.get("rationale") or "",
                "applied": str(merge.get("decision") or "") == "accept",
            }
        )
    return {"rewrites": rewrites, "definitions": definitions, "pairs": pairs}


def _survivor_choice(
    codebook: Mapping[str, Mapping[str, Any]],
    find,
    components: Mapping[str, set[str]],
    preferred: Mapping[str, str],
    conflicted: set[str],
    directed_edges: Sequence[tuple[str, str]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Map every code key to the label that should remain, plus the rule used.

    A model merge keeps the target side. A component with no model direction
    keeps the lexicographically smaller key. Directions that disagree inside
    one component use that same lexicographic tie-break among the labels a
    model named as a target.
    """
    edges_by_root: dict[str, list[tuple[str, str]]] = {}
    for source, target in directed_edges:
        if source not in codebook or target not in codebook:
            continue
        edges_by_root.setdefault(find(source), []).append((source, target))
    survivor_of: dict[str, str] = {}
    rules: dict[str, str] = {}
    seen: set[str] = set()
    for key in codebook:
        root = find(key)
        if root in seen:
            continue
        seen.add(root)
        members = set(components.get(root) or {root})
        edges = edges_by_root.get(root, [])
        if root in conflicted:
            targets = {target for _source, target in edges if target in members}
            choice = min(targets or members)
            rule = "tie_break"
        elif edges:
            choice = preferred.get(root, min(members))
            if choice not in members:
                choice = min(members)
            rule = "target"
        else:
            choice = min(members)
            rule = "lexicographic"
        for member in members:
            survivor_of[member] = choice
        rules[choice] = rule
    for key in codebook:
        survivor_of.setdefault(key, key)
    return survivor_of, rules


def apply_sweep(
    codebook: dict[str, dict],
    segments: list[dict],
    *,
    merges: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]],
    definitions: list[Mapping[str, Any]] | None = None,
    overloaded: list[str] | None = None,
    relation_hints: list[Mapping[str, Any]] | None = None,
    min_merge_confidence: float = 0.5,
    review_below: float = 0.5,
    frozen_pairs: set[frozenset[str]] | None = None,
    max_merges_per_sweep: int | None = 12,
) -> dict[str, Any]:
    """Apply model-proposed merges, sharpen definitions, and relabel segments.

    ``candidates`` are hints that were shown to the model. A merge is applied
    when both labels exist. The model may propose a pair that was not a candidate.
    Accepted merges record ``from_candidate`` so a later reader can see which
    pairs the model added on its own.
    """
    hinted: set[frozenset[str]] = set()
    for item in candidates:
        source_key = code_key(str(item.get("source") or ""))
        target_key = code_key(str(item.get("target") or ""))
        if source_key and target_key and source_key != target_key:
            hinted.add(frozenset((source_key, target_key)))
    parent = {key: key for key in codebook}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    accepted: list[dict] = []
    rejected: list[dict] = []
    eligible: list[tuple[dict, str, str]] = []
    ignored: list[dict] = []
    review: list[dict] = []
    ops: list[dict] = []
    id_to_key = _index_ids(codebook)
    frozen = frozen_pairs or set()
    for merge in merges:
        source_id = str(merge.get("source_id") or "").strip()
        target_id = str(merge.get("target_id") or "").strip()
        if source_id and source_id not in id_to_key:
            rejected.append(
                {
                    "source": merge.get("source"),
                    "target": merge.get("target"),
                    "pair_id": merge.get("pair_id") or "",
                    "source_id": source_id,
                    "reason": "unknown_code_id",
                    "confidence": bounded_unit(merge.get("confidence"), 1.0),
                    "from_candidate": False,
                }
            )
            continue
        if target_id and target_id not in id_to_key:
            rejected.append(
                {
                    "source": merge.get("source"),
                    "target": merge.get("target"),
                    "pair_id": merge.get("pair_id") or "",
                    "target_id": target_id,
                    "reason": "unknown_code_id",
                    "confidence": bounded_unit(merge.get("confidence"), 1.0),
                    "from_candidate": False,
                }
            )
            continue
        source = id_to_key[source_id] if source_id else code_key(str(merge.get("source") or ""))
        target = id_to_key[target_id] if target_id else code_key(str(merge.get("target") or ""))
        offered = frozenset((source, target)) in hinted if source and target else False
        confidence = decision_confidence(merge)
        record = {
            "source": codebook.get(source, {}).get("code", merge.get("source")),
            "target": codebook.get(target, {}).get("code", merge.get("target")),
            "source_id": source_id or str(codebook.get(source, {}).get("code_id") or ""),
            "target_id": target_id or (codebook.get(target, {}).get("code_id") or ""),
            "pair_id": merge.get("pair_id") or "",
            "rationale": str(merge.get("rationale") or "").strip() or _fallback_rationale("merge_accept", ""),
            "confidence": confidence,
            "model_confidence": bounded_unit(merge.get("model_confidence", merge.get("confidence")), 1.0),
            "agreement_confidence": merge.get("agreement_confidence", agreement_confidence(merge)),
            "source_valence": str(
                merge.get("source_valence") or codebook.get(source, {}).get("valence") or ""
            ),
            "target_valence": str(
                merge.get("target_valence") or codebook.get(target, {}).get("valence") or ""
            ),
            "from_candidate": offered,
            "decision": "reject",
            "rewritten_label": str(merge.get("rewritten_label") or ""),
            "rewritten_definition": str(merge.get("rewritten_definition") or ""),
            "quote_fits": list(merge.get("quote_fits") or []),
        }
        # Stance is context. Opposite valence does not reject the comparison.
        if source and target and frozenset((source, target)) in frozen:
            rejected.append(
                {
                    **record,
                    "reason": "already_decided",
                    "decision": "reject",
                    "rationale": str(merge.get("rationale") or "").strip() or _fallback_rationale("merge_reject", "already_decided"),
                }
            )
            continue
        if confidence < min_merge_confidence:
            held = {**record, "reason": "below_merge_confidence", "status": "review"}
            review.append(held)
            rejected.append({**record, "reason": "below_merge_confidence"})
            ops.append(
                _op(
                    op="merge_accept",
                    status="review",
                    confidence=confidence,
                    ids={"source_id": record["source_id"], "target_id": record["target_id"], "pair_id": record["pair_id"]},
                    reason="below_merge_confidence",
                    rationale=record["rationale"],
                )
            )
            continue
        if source not in parent or target not in parent or source == target:
            if offered:
                rejected.append({**record, "reason": "unknown_or_same"})
            else:
                ignored.append({**record, "reason": "not_offered"})
            continue
        for flag in (
            "same_action",
            "same_object",
            "same_stance",
            "contrast_checked",
            "material_contrast",
            "evidence_same",
            "quote_fits",
        ):
            if flag in merge:
                record[flag] = merge.get(flag)
        decision, gate_reason = classify_comparison(merge, codebook)
        record["comparison_op"] = decision
        record["trivial"] = bool(merge.get("trivial"))
        if decision != "alias_duplicate":
            rejected.append({**record, "reason": gate_reason, "decision": decision})
            ops.append(
                _op(
                    op=decision,
                    status="deferred" if decision == "defer" else "applied",
                    confidence=confidence,
                    ids={"source_id": record["source_id"], "target_id": record["target_id"], "pair_id": record["pair_id"]},
                    reason=gate_reason,
                    rationale=record["rationale"],
                )
            )
            continue
        eligible.append((record, source, target))

    eligible.sort(key=lambda item: (-float(item[0].get("confidence") or 0), str(item[0].get("source") or ""), str(item[0].get("target") or "")))
    from automated_grounded_theory.records import component_aliases_are_direct

    eligible_pairs = {frozenset((source, target)) for _record, source, target in eligible}
    components: dict[str, set[str]] = {key: {key} for key in codebook}
    preferred = {key: key for key in codebook}
    conflicted: set[str] = set()
    directed_edges: list[tuple[str, str]] = []
    lost: list[dict] = []
    cap = None if max_merges_per_sweep is None else max(0, int(max_merges_per_sweep))
    applied_count = 0
    for record, source, target in eligible:
        if cap is not None and applied_count >= cap:
            held = {**record, "reason": "merge_cap", "status": "review", "decision": "defer"}
            review.append(held)
            ops.append(
                _op(
                    op="merge_accept",
                    status="review",
                    confidence=float(record.get("confidence") or 0),
                    ids={"source_id": record["source_id"], "target_id": record["target_id"], "pair_id": record["pair_id"]},
                    reason="merge_cap",
                    rationale=record.get("rationale") or "",
                )
            )
            continue
        root_source, root_target = find(source), find(target)
        if root_source == root_target:
            # A later judgment that would retire the label already chosen is a
            # conflict. The component still merges; the survivor tie-breaks.
            current = preferred.get(root_source, root_source)
            if not record.get("trivial") and source == current and target != current:
                conflicted.add(root_source)
                directed_edges.append((source, target))
            rejected.append({**record, "reason": "already_merged"})
            continue
        group = set(components.get(root_source) or {root_source}) | set(components.get(root_target) or {root_target})
        if not component_aliases_are_direct(group, eligible_pairs):
            rejected.append({**record, "reason": "non_transitive", "comparison_op": "alias_duplicate"})
            lost.append(
                {
                    "kind": "non_transitive",
                    "codes": sorted(group),
                    "edge": [source, target],
                    "reason": "non_transitive",
                    "rationale": record.get("rationale") or "",
                }
            )
            ops.append(
                _op(
                    op="alias_duplicate",
                    status="rejected",
                    confidence=float(record.get("confidence") or 0),
                    ids={"source_id": record["source_id"], "target_id": record["target_id"]},
                    reason="non_transitive",
                    rationale=record.get("rationale") or "",
                )
            )
            continue
        # Group membership stays a union-find. The label that remains is chosen
        # afterwards: the model's target, or a lexicographic tie-break when the
        # judgment is missing or the component's directions conflict.
        if root_target < root_source:
            parent[root_source] = root_target
            group_root, retired_root = root_target, root_source
        else:
            parent[root_target] = root_source
            group_root, retired_root = root_source, root_target
        if record.get("trivial"):
            left, right = preferred[root_source], preferred[root_target]
            chosen = left if left <= right else right
        else:
            chosen = preferred[root_target]
            directed_edges.append((source, target))
        preferred[group_root] = chosen
        preferred.pop(retired_root, None)
        if retired_root in conflicted or group_root in conflicted:
            conflicted.add(group_root)
        conflicted.discard(retired_root)
        components[group_root] = group
        components.pop(retired_root, None)
        record = {**record, "decision": "accept", "comparison_op": "alias_duplicate"}
        accepted.append(record)
        lost.append(_audit_alias(record, codebook, segments))
        applied_count += 1
        ops.append(
            _op(
                op="alias_duplicate",
                status="applied",
                confidence=float(record.get("confidence") or 0),
                ids={"source_id": record["source_id"], "target_id": record["target_id"], "pair_id": record["pair_id"]},
                reason="",
                rationale=record.get("rationale") or "",
            )
        )

    survivor_of, survivor_rule = _survivor_choice(codebook, find, components, preferred, conflicted, directed_edges)
    book: dict[str, dict] = {}
    merge_map: dict[str, str] = {}
    for key, entry in codebook.items():
        choice = survivor_of[key]
        merge_map[key] = choice
        if choice not in book:
            survivor = deepcopy(codebook[choice])
            survivor["key"] = choice
            survivor["code"] = codebook[choice]["code"]
            survivor["history"] = list(survivor.get("history") or [])
            survivor["aliases"] = [
                str(alias)
                for alias in (survivor.get("aliases") or [])
                if str(alias).strip() and str(alias) != survivor["code"]
            ]
            book[choice] = survivor
            members = [member for member, chosen in survivor_of.items() if chosen == choice]
            if len(members) > 1:
                survivor["history"].append(
                    {
                        "event": "survivor_chosen",
                        "rule": survivor_rule.get(choice, "target"),
                        "code": survivor["code"],
                    }
                )
        if key != choice:
            aliases = book[choice].setdefault("aliases", [])
            label = str(entry.get("code") or key)
            if label and label != book[choice]["code"] and label not in aliases:
                aliases.append(label)
            for prior in entry.get("aliases") or []:
                prior_label = str(prior).strip()
                if prior_label and prior_label != book[choice]["code"] and prior_label not in aliases:
                    aliases.append(prior_label)
            book[choice].setdefault("history", []).extend(entry.get("history") or [])
            book[choice]["history"].append(
                {
                    "event": "merged_from",
                    "code": entry.get("code"),
                    "key": key,
                    "rationale": next(
                        (item.get("rationale", "") for item in accepted if code_key(str(item.get("source") or "")) == key),
                        "",
                    ),
                }
            )
            book[choice]["aliases"] = sorted(book[choice].get("aliases") or [], key=code_key)

    for update in definitions or []:
        update_id = str(update.get("code_id") or "").strip()
        if update_id and update_id not in id_to_key:
            rejected.append(
                {
                    "source": update.get("code"),
                    "target": None,
                    "code_id": update_id,
                    "reason": "unknown_code_id",
                    "confidence": bounded_unit(update.get("confidence"), 1.0),
                }
            )
            continue
        key = id_to_key[update_id] if update_id else code_key(str(update.get("code") or ""))
        confidence = bounded_unit(update.get("confidence"), 1.0)
        if key not in parent:
            rejected.append({"source": update.get("code"), "target": None, "reason": "unknown_definition", "confidence": confidence})
            continue
        root = survivor_of.get(key, find(key))
        definition = str(update.get("definition") or "").strip()
        if key != root:
            # The post-merge rewrite owns the survivor wording. A definition
            # aimed at a retired code is dropped so it does not become review noise.
            continue
        if confidence < review_below:
            held = {
                "source": update.get("code"),
                "target": None,
                "reason": "below_review_threshold",
                "confidence": confidence,
                "status": "review",
            }
            review.append(held)
            ops.append(
                _op(
                    op="refine_definition",
                    status="review",
                    confidence=confidence,
                    ids={"code_id": update_id or str(codebook.get(key, {}).get("code_id") or "")},
                    reason="below_review_threshold",
                    rationale=str(update.get("rationale") or ""),
                )
            )
            continue
        if root in book and definition:
            book[root]["definition"] = definition
            book[root].setdefault("history", []).append({"event": "definition_sharpened", "definition": definition})
            ops.append(
                _op(
                    op="refine_definition",
                    status="applied",
                    confidence=confidence,
                    ids={"code_id": str(book[root].get("code_id") or "")},
                    reason="",
                    rationale=str(update.get("rationale") or update.get("definition") or ""),
                )
            )

    for label in overloaded or []:
        key = code_key(str(label))
        if key not in parent:
            continue
        root = survivor_of.get(key, find(key))
        if root in book:
            book[root]["flagged_overloaded"] = True
            book[root].setdefault("history", []).append({"event": "flagged_overloaded", "code": label})

    new_segments = []
    for segment in segments:
        updated = dict(segment)
        key = updated.get("code_key") or code_key(str(updated.get("code") or ""))
        if key in parent:
            root = survivor_of.get(key, find(key))
            updated["code_key"] = root
            updated["code"] = book[root]["code"]
        new_segments.append(updated)

    book = sync_counts(book, new_segments)
    hints_applied = []
    for hint in relation_hints or []:
        source = code_key(str(hint.get("source") or ""))
        target = code_key(str(hint.get("target") or ""))
        if source not in parent or target not in parent:
            continue
        source_root, target_root = survivor_of.get(source, find(source)), survivor_of.get(target, find(target))
        if source_root not in book or target_root not in book:
            continue
        record = {
            "source": book[source_root]["code"],
            "target": book[target_root]["code"],
            "relation": str(hint.get("relation") or "related"),
            "note": str(hint.get("note") or ""),
        }
        for root in {source_root, target_root}:
            current = book[root].setdefault("relation_hints", [])
            if record not in current:
                current.append(record)
        hints_applied.append(record)

    display_map = {}
    for key, root in merge_map.items():
        if key != root and root in book and key in codebook:
            display_map[codebook[key]["code"]] = book[root]["code"]

    return {
        "codebook": book,
        "segments": new_segments,
        "accepted_merges": accepted,
        "rejected_merges": rejected,
        "ignored_decisions": ignored,
        "review": review,
        "ops": ops,
        "merge_map": display_map,
        "relation_hints": hints_applied,
        "lost_distinction_candidates": lost,
    }


def _pair_key(source: Any, target: Any) -> frozenset[str] | None:
    left = code_key(str(source or ""))
    right = code_key(str(target or ""))
    if not left or not right or left == right:
        return None
    return frozenset((left, right))


def finalize_candidate_decisions(
    applied: dict[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    rejections: Sequence[Mapping[str, Any]] | None = None,
    skipped: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Log exactly one decision for each offered candidate pair.

    A pair the model never mentions is ``model_omitted``. A pair whose sweep
    chunk timed out is ``chunk_skipped``. A decision whose pair_id or labels
    are not in the candidate list is stored on ``ignored_decisions`` and is
    not counted as a candidate decision.
    """
    hinted: dict[frozenset[str], dict] = {}
    by_id: dict[str, frozenset[str]] = {}
    for item in candidates:
        pair = _pair_key(item.get("source"), item.get("target"))
        if pair is None or pair in hinted:
            continue
        hinted[pair] = dict(item)
        pair_id = str(item.get("pair_id") or "").strip()
        if pair_id and pair_id not in by_id:
            by_id[pair_id] = pair
    for item in skipped or []:
        pair = _pair_key(item.get("source"), item.get("target"))
        if pair is None or pair in hinted:
            continue
        hinted[pair] = dict(item)
        pair_id = str(item.get("pair_id") or "").strip()
        if pair_id and pair_id not in by_id:
            by_id[pair_id] = pair

    decided: set[frozenset[str]] = set()
    for item in list(applied.get("accepted_merges") or []) + list(applied.get("rejected_merges") or []):
        pair = _offered_pair(item, hinted, by_id)
        if pair:
            decided.add(pair)

    ignored = list(applied.get("ignored_decisions") or [])

    def add(item: Mapping[str, Any], *, reason: str, rationale: str) -> None:
        pair = _offered_pair(item, hinted, by_id)
        if pair is None:
            ignored.append(
                {
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "pair_id": item.get("pair_id") or "",
                    "rationale": rationale,
                    "from_candidate": False,
                    "reason": "not_offered",
                }
            )
            return
        if pair in decided:
            return
        decided.add(pair)
        offered = hinted[pair]
        applied.setdefault("rejected_merges", []).append(
            {
                "source": offered.get("source"),
                "target": offered.get("target"),
                "pair_id": offered.get("pair_id") or "",
                "rationale": rationale,
                "from_candidate": True,
                "reason": reason,
            }
        )

    for item in rejections or []:
        add(
            item,
            reason="model_rejected",
            rationale=str(item.get("rationale") or ""),
        )
    skipped_rows = sorted(
        list(skipped or []),
        key=lambda item: (str(item.get("source") or "").casefold(), str(item.get("target") or "").casefold()),
    )
    for item in skipped_rows:
        add(
            item,
            reason="chunk_skipped",
            rationale=str(item.get("rationale") or "This sweep chunk hit a deadline and was skipped."),
        )
    omitted = [
        item
        for item in candidates
        if (pair := _pair_key(item.get("source"), item.get("target"))) is not None and pair not in decided
    ]
    omitted.sort(key=lambda item: (str(item.get("source") or "").casefold(), str(item.get("target") or "").casefold()))
    for item in omitted:
        add(
            item,
            reason="model_omitted",
            rationale="The model did not accept or reject this candidate.",
        )
    applied["ignored_decisions"] = ignored
    applied["candidate_decisions"] = _candidate_decision_rows(applied, hinted)
    return applied


def _offered_pair(
    item: Mapping[str, Any],
    hinted: Mapping[frozenset[str], Mapping[str, Any]],
    by_id: Mapping[str, frozenset[str]],
) -> frozenset[str] | None:
    """The candidate this decision names, or None when it was not offered."""
    pair = _pair_key(item.get("source"), item.get("target"))
    pair_id = str(item.get("pair_id") or "").strip()
    if pair_id:
        expected = by_id.get(pair_id)
        if expected is None:
            return None
        if pair is not None and pair != expected:
            return None
        return expected
    if pair in hinted:
        return pair
    return None


def _candidate_decision_rows(applied: Mapping[str, Any], hinted: Mapping[frozenset[str], Mapping[str, Any]]) -> list[dict]:
    """One row per offered pair, in candidate order."""
    rows = []
    for pair, offered in hinted.items():
        decision = "reject"
        reason = "model_omitted"
        for item in applied.get("accepted_merges") or []:
            if _pair_key(item.get("source"), item.get("target")) == pair:
                decision = "accept"
                reason = "accepted"
                break
        else:
            for item in applied.get("rejected_merges") or []:
                if _pair_key(item.get("source"), item.get("target")) == pair:
                    reason = str(item.get("reason") or "model_rejected")
                    break
        matched = None
        for item in list(applied.get("accepted_merges") or []) + list(applied.get("rejected_merges") or []):
            if _pair_key(item.get("source"), item.get("target")) == pair:
                matched = item
                break
        rows.append(
            {
                "pair_id": offered.get("pair_id") or "",
                "source": offered.get("source"),
                "target": offered.get("target"),
                "decision": decision,
                "reason": reason,
                "confidence": None if matched is None else matched.get("confidence", 1.0),
            }
        )
    for item in applied.get("accepted_merges") or []:
        if item.get("from_candidate"):
            continue
        pair = _pair_key(item.get("source"), item.get("target"))
        if pair is None or pair in hinted:
            continue
        rows.append(
            {
                "pair_id": item.get("pair_id") or "",
                "source": item.get("source"),
                "target": item.get("target"),
                "decision": "accept",
                "reason": "model_proposed",
                "rationale": item.get("rationale") or "",
                "confidence": item.get("confidence", 1.0),
            }
        )
    return rows


def codebook_list(codebook: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = [dict(entry) for entry in codebook.values()]
    rows.sort(key=lambda entry: (-int(entry.get("frequency") or 0), str(entry.get("code") or "").casefold()))
    return rows
