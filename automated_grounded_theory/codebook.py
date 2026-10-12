"""Apply passage coding and sweep decisions to the running codebook.

Counts are rebuilt from segments after every commit so a merge cannot
double-count applications.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from automated_grounded_theory.retrieval import jaccard
from automated_grounded_theory.validation import (
    is_code_id_text,
    QUOTE_FLAG_STATUSES,
    assess_label,
    attribute_leak,
    bounded_unit,
    code_key,
    label_restates_context,
    anchor_quote,
    review_passage_flags,
    substantive_memo,
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
            "This pair was already decided in this run, and neither definition has changed in substance."
        ),
        "opposite_valence": "These codes take opposite stances, so they stay separate.",
        "below_merge_confidence": "The agreement confidence is below the merge threshold, so the merge waits for review.",
        "below_review_threshold": "The confidence is below the review threshold, so this decision waits for a person.",
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


def opposite_valence(left: Any, right: Any) -> bool:
    """True when one code is positive and the other is negative."""
    pair = {str(left or "").strip().casefold(), str(right or "").strip().casefold()}
    return pair == {"positive", "negative"}


def definitions_changed_materially(previous: str, current: str) -> bool:
    """True when a definition change is large enough to reopen a pair decision.

    Wording tweaks stay frozen. A change is material when the token overlap
    of the two definitions falls below 0.5.
    """
    old = str(previous or "").strip()
    new = str(current or "").strip()
    if old == new:
        return False
    return jaccard(old, new) < 0.5


def agreement_confidence(merge: Mapping[str, Any]) -> float | None:
    """Rubric agreement as a second confidence, using the 0.2 / 0.5 / 0.95 anchors.

    ``None`` means the model did not set the rubric, so the caller's own
    confidence stands. An accept needs the same action, the same object, and
    the same stance. A reject is confident when that rubric fails.
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


def trivial_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Near-identical labels with the same stance can merge without a model call."""
    if opposite_valence(left.get("valence"), right.get("valence")):
        return False
    left_stance = str(left.get("valence") or "neutral").strip().casefold() or "neutral"
    right_stance = str(right.get("valence") or "neutral").strip().casefold() or "neutral"
    if left_stance != right_stance:
        return False
    return jaccard(str(left.get("code") or ""), str(right.get("code") or "")) >= TRIVIAL_LABEL_JACCARD


def trivial_merges(codes: Sequence[Mapping[str, Any]], frozen: set[frozenset[str]] | None = None) -> list[dict]:
    """Deterministic merges for near-duplicate labels. These skip the sweep model."""
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
                        "These labels are near-duplicates and share the same stance, "
                        "so the merge did not need a model call."
                    ),
                    "confidence": 0.95,
                    "agreement_confidence": 0.95,
                    "same_action": True,
                    "same_object": True,
                    "same_stance": True,
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


def classify_comparison(merge: Mapping[str, Any], codebook: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """Decide whether an alias is the same concept in the definitions and the quotes.

    ``alias_duplicate`` requires the same action, the same object, a contrast
    check that found no material difference, and evidence that is not mixed.
    Anything else stays ``keep_distinct``, ``defer``, or the operation the
    model already named. Valence is not part of this gate.
    """
    named = str(merge.get("comparison_op") or merge.get("op") or "").strip()
    if named in {"keep_distinct", "defer", "link_to_category", "record_dimension", "split_code", "refine_definition"}:
        return named, named
    if merge.get("trivial"):
        return "alias_duplicate", "trivial_pair"
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


def _pair_still_current(
    row: Mapping[str, Any],
    codebook: Mapping[str, Mapping[str, Any]],
    by_id: Mapping[str, str],
) -> tuple[str, str] | None:
    """Code keys for a cached pair that still exists with the same definitions."""
    ids = [str(item) for item in (row.get("ids") or []) if str(item)]
    if len(ids) != 2:
        return None
    keys = [by_id.get(item) for item in ids]
    if any(key is None for key in keys) or keys[0] == keys[1]:
        return None
    stored = {str(key): str(value or "") for key, value in (row.get("definitions") or {}).items()}
    current = {item: str(codebook[by_id[item]].get("definition") or "") for item in ids}
    if any(definitions_changed_materially(stored.get(item, ""), current[item]) for item in ids):
        return None
    return str(keys[0]), str(keys[1])


def frozen_pair_keys(cache: Sequence[Mapping[str, Any]], codebook: Mapping[str, Mapping[str, Any]]) -> set[frozenset[str]]:
    """Pairs already decided this run whose definitions have not changed.

    The key is the two code keys. A later sweep skips asking the model again
    until one definition is different from the definition stored with the decision.
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
        if previous and str(previous.get("decision")) == "reject" and str(decision) == "accept":
            stored_defs = previous.get("definitions") or {}
            if not any(
                definitions_changed_materially(str(stored_defs.get(item) or ""), current_defs[item]) for item in ids
            ):
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
) -> dict[str, Any]:
    """Validate one model response. Does not mutate the codebook.

    A decision whose confidence is below ``review_below`` is not applied.
    It is returned on ``review``. Unknown code ids are rejected.
    A quote that cannot be anchored does not reject the code. The quote is
    flagged ``possibly_modified`` or ``unanchored`` and coding continues.
    ``quote_retries`` lists those flags so the caller can ask for a closer
    quote once, then keep the code anyway.
    """
    payload = coding.model_dump() if hasattr(coding, "model_dump") else dict(coding)
    proposals = list(payload.get("codes") or [])
    rejections: list[dict] = []
    accepted: list[dict] = []
    ops: list[dict] = []
    review: list[dict] = []
    quote_retries: list[dict] = []
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

    for proposal in proposals:
        label = str(proposal.get("code") or "")
        quote = str(proposal.get("quote") or "")
        if is_code_id_text(label):
            if label in id_to_key:
                label = str(codebook[id_to_key[label]].get("code") or label)
            else:
                rejections.append({"code": label, "reason": "code_id_as_label", "quote": quote})
                continue
        anchor = anchor_quote(
            quote,
            passage_text,
            prev_text=str(passage.get("prev_text") or ""),
            next_text=str(passage.get("next_text") or ""),
            document_text=str(passage.get("document_text") or document_text or ""),
            passage_start=int(passage.get("start_offset") or 0),
            passage_end=int(passage.get("end_offset") or 0) or None,
        )
        verbatim = anchor.quote
        quote_flag = f"quote_{anchor.status}" if anchor.status in QUOTE_FLAG_STATUSES else ""
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
            rejections.append({"code": label, "reason": reasons[0] if reasons else "rejected", "quote": verbatim})
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
                rejections.append(
                    {
                        "code": refined_label,
                        "reason": refined_reasons[0] if refined_reasons else "rejected",
                        "quote": verbatim,
                    }
                )
                continue
            fields = refined_fields
            reasons = list(reasons) + [reason for reason in refined_reasons if reason not in reasons]

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
        )
        flags.extend(reasons)
        if quote_flag:
            flags.append(quote_flag)
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
        if confidence < review_below:
            rejections.append(
                {
                    "code": fields.get("code") or label,
                    "reason": "below_review_threshold",
                    "quote": verbatim,
                    "confidence": confidence,
                }
            )
            held = _op(
                op=op_name,
                status="review",
                confidence=confidence,
                ids={"passage_id": passage_id, "code_id": sent_id},
                reason="below_review_threshold",
                rationale=str(proposal.get("rationale") or ""),
            )
            ops.append(held)
            review.append(held)
            continue

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
    if memo_body and memo_confidence < review_below:
        held_memo = _op(
            op="memo",
            status="review",
            confidence=memo_confidence,
            ids={"passage_id": passage_id},
            reason="below_review_threshold",
            rationale=memo_body,
        )
        ops.append(held_memo)
        review.append(held_memo)
        memo_body = ""
    elif memo_body:
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
        "low_confidence": low_confidence,
        "ops": ops,
        "review": review,
        "quote_retries": quote_retries,
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
    """Append segments for accepted codes. ``existing_keys`` tracks globally new labels."""
    book = deepcopy(codebook)
    new_segments = [dict(segment) for segment in segments]
    new_memos = list(memos)
    highlights = dict(highlight_ids)
    counters = dict(counters)
    known_before = set(existing_keys) if existing_keys is not None else set(book)
    new_codes = 0

    memo_body = str(interpreted.get("memo") or "").strip()
    if memo_body and not substantive_memo(memo_body):
        memo_body = ""
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
            }
        )

    for item in interpreted.get("accepted") or []:
        key = item["key"]
        previous = item.get("previous_key")
        if previous and previous in book and previous != key:
            book = _rename_code(book, new_segments, previous, key, item["code"])
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

        if item.get("doc_start") is None or item.get("doc_end") is None:
            span_key = (
                f"{passage.get('doc_id')}:{passage.get('passage_id')}:"
                f"{item.get('quote_status')}:{item.get('model_quote')}"
            )
        else:
            span_key = f"{passage.get('doc_id')}:{item['doc_start']}:{item['doc_end']}"
        if span_key not in highlights:
            highlights[span_key] = _next(counters, "highlight", "h")
        segment = {
            "segment_id": _next(counters, "segment", "s"),
            "highlight_id": highlights[span_key],
            "passage_id": passage.get("passage_id"),
            "doc_id": passage.get("doc_id"),
            "doc_name": passage.get("doc_name"),
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
            "metadata": dict(passage.get("metadata") or {}),
            "confidence": item["confidence"],
            "needs_review": item["needs_review"],
            "review_reasons": list(item["review_reasons"]),
            "memo_id": memo_id,
        }
        new_segments.append(segment)

    book = sync_counts(book, new_segments)
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
    for segment in segments:
        if segment.get("code_key") == old:
            segment["code_key"] = new
            segment["code"] = book[new]["code"] if new in book else display
            segment["code_id"] = book[new].get("code_id") or _code_id(new)
    return book


def sync_counts(codebook: dict[str, dict], segments: list[dict]) -> dict[str, dict]:
    """Rebuild frequency, documents, and examples from ``segments``."""
    grouped: dict[str, list[dict]] = {}
    for segment in segments:
        key = segment.get("code_key") or code_key(str(segment.get("code") or ""))
        grouped.setdefault(key, []).append(segment)
    synced: dict[str, dict] = {}
    for key, entry in codebook.items():
        applications = grouped.get(key) or []
        if not applications:
            continue
        updated = deepcopy(entry)
        updated["key"] = key
        updated["frequency"] = len(applications)
        updated["document_ids"] = sorted({str(item.get("doc_id")) for item in applications if item.get("doc_id")})
        examples = []
        seen_quotes: set[tuple] = set()
        for item in applications:
            marker = (item.get("passage_id"), item.get("quote"))
            if marker in seen_quotes:
                continue
            seen_quotes.add(marker)
            examples.append(
                {
                    "quote": item.get("quote"),
                    "doc_id": item.get("doc_id"),
                    "passage_id": item.get("passage_id"),
                    "doc_start": item.get("doc_start"),
                    "doc_end": item.get("doc_end"),
                    "quote_status": item.get("quote_status") or "",
                    "quote_similarity": item.get("quote_similarity"),
                }
            )
            if len(examples) >= 8:
                break
        updated["examples"] = examples
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
        if updated.get("in_vivo") and not any(
            code_key(label) and label.casefold() in str(item.get("quote") or "").casefold() for item in examples
        ):
            # Keep the flag only when some stored quote still contains the label.
            from automated_grounded_theory.validation import in_vivo_supported

            updated["in_vivo"] = any(in_vivo_supported(label, str(item.get("quote") or "")) for item in examples)
        synced[key] = updated
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
        survivor = source if source <= target else target
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
        # The survivor is the lexicographically smaller code key. The model's
        # source/target direction does not change which label remains.
        if root_target < root_source:
            parent[root_source] = root_target
            survivor_root, retired_root = root_target, root_source
        else:
            parent[root_target] = root_source
            survivor_root, retired_root = root_source, root_target
        components[survivor_root] = group
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

    book: dict[str, dict] = {}
    merge_map: dict[str, str] = {}
    for key, entry in codebook.items():
        root = find(key)
        merge_map[key] = root
        if root not in book:
            survivor = deepcopy(codebook[root])
            survivor["key"] = root
            survivor["code"] = codebook[root]["code"]
            survivor["history"] = list(survivor.get("history") or [])
            book[root] = survivor
        if key != root:
            book[root].setdefault("history", []).append(
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
        root = find(key)
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
        root = find(key)
        if root in book:
            book[root]["flagged_overloaded"] = True
            book[root].setdefault("history", []).append({"event": "flagged_overloaded", "code": label})

    new_segments = []
    for segment in segments:
        updated = dict(segment)
        key = updated.get("code_key") or code_key(str(updated.get("code") or ""))
        if key in parent:
            root = find(key)
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
        source_root, target_root = find(source), find(target)
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
