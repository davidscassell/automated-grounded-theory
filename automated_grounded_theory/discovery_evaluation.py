"""Score a frozen human discovery set against a run's relationship export.

This is not gate evaluation. ``score_gate_adjudications`` only scores
adjudications whose relationship id is already in the accepted or withheld
export, so a relationship the run never proposed cannot lower that recall.
Discovery starts from relationships a person identified independently, then
asks which of those the run recovered, distorted, or missed.

Matching is deterministic. Endpoint identity is a normalized code id. A
passage anchor is a shared passage id. Quotes, embeddings, and model
judgments are not match signals. Tier labels are not match signals.
The proposed statement is not a structural match signal. Structural recall
measures recovery of the specified structural fields. It does not mean the
human-supported interpretation was recovered. A separate human
``claim_judgment`` scores that statement.

Independent gold is assigned by a lexicographic one-to-one matching:
full agreement first, then any structural match, then a content ordering
that does not use adjudication ids or relationship ids. Each gold item may
stay unmatched. A proposal review does not remove a candidate from that
assignment. ``system_proposal`` rows bind only by ``system_relationship_id``.
Claim reviews are a separate list, bound to a run, a relationship, and a
hash of the reviewed statement.

A rate whose denominator is empty stays null. A full gate false-negative
rate stays null unless eligibility is known for every row in that population,
the same rule as the 0.9.14 downstream rates. Recovered-per-token and
recovered-per-dollar figures use reread expenditure only, and they stay
null unless that expenditure is complete.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

DISCOVERY_SCHEMA_VERSION = "discovery-eval-3"
DISCOVERY_SCHEMA_VERSION_V2 = "discovery-eval-2"
DISCOVERY_SCHEMA_VERSION_V1 = "discovery-eval-1"
_ACCEPTED_SCHEMAS = frozenset(
    {DISCOVERY_SCHEMA_VERSION, DISCOVERY_SCHEMA_VERSION_V2, DISCOVERY_SCHEMA_VERSION_V1}
)

_KINDS = frozenset({"independent", "system_proposal"})
_DIRECTIONS = frozenset({"from_to", "to_from", "undirected"})
_OUTCOMES = frozenset({"rejected", "withheld", "repair_failed", "proposed", "accepted"})
_CLAIM_VERDICTS = frozenset({"correct", "incorrect", "partly_correct", "not_judged"})
_WITHHELD_OUTCOMES = frozenset({"rejected", "withheld", "repair_failed"})
_WITHHELD_DISPOSITIONS = frozenset({"rejected", "provisional", "merged", "withheld"})
_FLAG_NAMES = (
    "unsupported_interpretation",
    "over_merging",
    "loss_of_minority_account",
    "loss_of_contradictory_account",
)

DISCOVERY_CSV_COLUMNS = (
    "adjudication_id",
    "kind",
    "adjudicator_id",
    "rationale",
    "from_id",
    "to_id",
    "direction",
    "role",
    "wording",
    "passage_id",
    "quote",
    "conceptual_distinctions",
    "unsupported_interpretation",
    "over_merging",
    "loss_of_minority_account",
    "loss_of_contradictory_account",
    "evidence_supported",
    "system_relationship_id",
    "system_outcome",
    "claim_verdict",
    "claim_direction",
    "claim_qualification",
    "claim_support",
    "disagreement_adjudicator_id",
    "disagreement_field",
    "disagreement_value",
    "disagreement_rationale",
)

_RATE_NAMES = (
    "discovery_recall",
    "discovery_precision",
    "discovery_distortion_rate",
    "gate_false_negative_rate",
    "gate_false_negative_rate_known",
    "discovery_recovered_per_reread_token",
    "discovery_recovered_per_reread_dollar",
    "claim_correct_rate",
)


class DiscoveryEvaluationError(ValueError):
    """A discovery file is the wrong version, unfrozen, or internally inconsistent."""


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _code_id(value: Any) -> str:
    """Casefold a code id. Labels are not ids and are not matched."""
    return _text(value).casefold()


def _passage_id(value: Any) -> str:
    return _text(value)


def _bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    raise DiscoveryEvaluationError(f"Expected a boolean, got {value!r}.")


def _direction(value: Any) -> str | None:
    """Return a scored direction, or None when direction is not scored."""
    if value is None:
        return None
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if text in {"", "unspecified", "none", "unknown"}:
        return None
    if text in _DIRECTIONS:
        return text
    raise DiscoveryEvaluationError(f"Unknown direction {value!r}.")


def _optional_token(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _labels(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = value.replace(";", "|").split("|")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
    else:
        raise DiscoveryEvaluationError("conceptual_distinctions must be a list or a pipe-separated string.")
    found: list[str] = []
    for part in parts:
        text = _text(part)
        if text and text not in found:
            found.append(text)
    return sorted(found)


def _passages_from_value(value: Any) -> list[dict[str, str]]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = [{"passage_id": value, "quote": ""}]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DiscoveryEvaluationError("supporting_passages must be a list.")
    rows: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            passage_id = _passage_id(item)
            quote = ""
        elif isinstance(item, Mapping):
            passage_id = _passage_id(item.get("passage_id"))
            quote = str(item.get("quote") or "")
        else:
            raise DiscoveryEvaluationError("A supporting passage must be an object or a passage id.")
        if not passage_id:
            continue
        rows.append({"passage_id": passage_id, "quote": quote})
    rows.sort(key=lambda row: (row["passage_id"], row["quote"]))
    return rows


def _passages_from_csv(row: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_ids = str(row.get("passage_id") or "")
    raw_quotes = str(row.get("quote") or "")
    if raw_ids == "" and raw_quotes == "":
        return []
    ids = raw_ids.split("|")
    quotes = raw_quotes.split("|") if raw_quotes != "" else []
    if quotes and len(quotes) != len(ids):
        raise DiscoveryEvaluationError(
            "CSV passage_id and quote lists must be the same length when both use '|'."
        )
    paired = []
    for index, passage_id in enumerate(ids):
        quote = quotes[index] if index < len(quotes) else ""
        paired.append({"passage_id": passage_id, "quote": quote})
    return _passages_from_value(paired)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _empty_claim_judgment() -> dict[str, Any]:
    return {"verdict": "not_judged", "direction": None, "qualification": None, "support": None}


def claim_statement_hash(statement: Any, direction: Any, qualification: Any) -> str:
    """Fingerprint the statement a person reviewed, plus direction and qualification.

    Whitespace is collapsed. Direction and qualification are casefolded.
    The hash does not include a run id or a relationship id. Those are stored
    beside it. A later score recomputes this from the proposal's statement and
    direction and from the review's qualification.
    """
    body = {
        "statement": " ".join(str(statement or "").split()),
        "direction": " ".join(str(direction or "").casefold().split()),
        "qualification": " ".join(str(qualification or "").casefold().split()),
    }
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _claim_judgment(row: Mapping[str, Any], *, version: str, rewrite_old: bool = False) -> dict[str, Any]:
    """A legacy reading stored on the gold row. It is not a structural match.

    discovery-eval-1 has no place for this judgment. discovery-eval-3 does not
    score it: a bound reading lives in ``claim_reviews``. A real verdict on
    either of those schemas raises. discovery-eval-2 still stores the object
    so its hash stays valid, and scoring treats that verdict as stale.
    """
    legacy = version == DISCOVERY_SCHEMA_VERSION_V1
    current = version == DISCOVERY_SCHEMA_VERSION
    raw = row.get("claim_judgment")
    flat = any(key in row for key in ("claim_verdict", "claim_direction", "claim_qualification", "claim_support"))
    if legacy:
        # An upgraded in-memory row carries not_judged. A real judgment cannot
        # stay on a discovery-eval-1 hash.
        if flat:
            raise DiscoveryEvaluationError(
                "discovery-eval-1 cannot store claim_judgment. Freeze the file as discovery-eval-3."
            )
        if isinstance(raw, Mapping):
            verdict = str(raw.get("verdict") or "not_judged").strip().casefold().replace(" ", "_").replace("-", "_")
            extras = any(str(raw.get(key) or "").strip() for key in ("direction", "qualification", "support"))
            if verdict == "not_judged" and not extras:
                return _empty_claim_judgment()
            raise DiscoveryEvaluationError(
                "discovery-eval-1 cannot store claim_judgment. Freeze the file as discovery-eval-3."
            )
        if raw not in (None, "", {}):
            raise DiscoveryEvaluationError(
                "discovery-eval-1 cannot store claim_judgment. Freeze the file as discovery-eval-3."
            )
        return _empty_claim_judgment()
    if flat and not isinstance(raw, Mapping):
        raw = {
            "verdict": row.get("claim_verdict"),
            "direction": row.get("claim_direction"),
            "qualification": row.get("claim_qualification"),
            "support": row.get("claim_support"),
        }
    if raw is None or raw == "":
        return _empty_claim_judgment()
    if not isinstance(raw, Mapping):
        raise DiscoveryEvaluationError("claim_judgment must be an object.")
    verdict = str(raw.get("verdict") or "not_judged").strip().casefold().replace(" ", "_").replace("-", "_")
    if verdict not in _CLAIM_VERDICTS:
        raise DiscoveryEvaluationError(f"Unknown claim verdict {raw.get('verdict')!r}.")

    def _optional(key: str) -> str | None:
        value = raw.get(key)
        if value is None or str(value).strip() == "":
            return None
        return str(value).strip()

    parsed = {
        "verdict": verdict,
        "direction": _optional("direction"),
        "qualification": _optional("qualification"),
        "support": _optional("support"),
    }
    carried = parsed["verdict"] != "not_judged" or any(parsed[key] for key in ("direction", "qualification", "support"))
    if current and carried:
        if rewrite_old:
            # An eval-1 or eval-2 judgment is not a bound review. Drop it on
            # freeze so it cannot be scored as correct under discovery-eval-3.
            return _empty_claim_judgment()
        raise DiscoveryEvaluationError(
            "discovery-eval-3 stores a claim review in claim_reviews, bound to a run, "
            "a relationship, and a statement hash. claim_judgment on an adjudication is not that review."
        )
    return parsed


def _one_claim_review(row: Mapping[str, Any]) -> dict[str, Any]:
    """One human reading of one proposed statement on one run."""
    if not isinstance(row, Mapping):
        raise DiscoveryEvaluationError("A claim review must be an object.")
    review_id = _text(row.get("review_id"))
    if not review_id:
        raise DiscoveryEvaluationError("A claim review needs review_id.")
    verdict = str(row.get("verdict") or "not_judged").strip().casefold().replace(" ", "_").replace("-", "_")
    if verdict not in _CLAIM_VERDICTS:
        raise DiscoveryEvaluationError(f"Unknown claim verdict {row.get('verdict')!r}.")
    run_id = _optional_token(row.get("run_id"))
    relationship_id = _optional_token(row.get("relationship_id"))
    statement = " ".join(str(row.get("statement") or "").split())
    direction = _optional_token(row.get("direction"))
    if direction is not None:
        direction = " ".join(direction.casefold().split())
    qualification = _optional_token(row.get("qualification"))
    if qualification is not None:
        qualification = " ".join(qualification.casefold().split())
    support = row.get("support")
    if support is None or str(support).strip() == "":
        support_text = None
    else:
        support_text = str(support).strip()
    digest = claim_statement_hash(statement, direction, qualification)
    supplied = row.get("statement_hash")
    if supplied not in (None, "") and not _same_hash(supplied, digest):
        raise DiscoveryEvaluationError(
            f"Claim review {review_id} statement_hash does not match the reviewed statement."
        )
    bound = bool(run_id and relationship_id and statement)
    return {
        "review_id": review_id,
        "run_id": run_id,
        "relationship_id": relationship_id,
        "statement": statement,
        "direction": direction,
        "qualification": qualification,
        "support": support_text,
        "verdict": verdict,
        "statement_hash": digest,
        "binding_status": "bound" if bound else "unbound",
    }


def _claim_reviews(value: Any, *, version: str) -> list[dict[str, Any]]:
    if value is None or value == "":
        value = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DiscoveryEvaluationError("claim_reviews must be a list.")
    if value and version != DISCOVERY_SCHEMA_VERSION:
        raise DiscoveryEvaluationError(
            f"{version} cannot store claim_reviews. Freeze the file as {DISCOVERY_SCHEMA_VERSION}."
        )
    reviews = [_one_claim_review(row) for row in value]
    seen: set[str] = set()
    bindings: set[tuple[str, str]] = set()
    for review in reviews:
        if review["review_id"] in seen:
            raise DiscoveryEvaluationError(f"Duplicate claim review_id {review['review_id']}.")
        seen.add(review["review_id"])
        if review["binding_status"] != "bound":
            continue
        key = (str(review["run_id"]), str(review["relationship_id"]))
        if key in bindings:
            raise DiscoveryEvaluationError(
                "Duplicate claim review for run "
                f"{review['run_id']!r} and relationship {review['relationship_id']!r}."
            )
        bindings.add(key)
    reviews.sort(key=lambda review: review["review_id"])
    return reviews


def _one_adjudication(
    row: Mapping[str, Any],
    *,
    passages: list[dict[str, str]] | None = None,
    version: str = DISCOVERY_SCHEMA_VERSION,
    rewrite_old: bool = False,
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise DiscoveryEvaluationError("An adjudication must be an object.")
    adjudication_id = _text(row.get("adjudication_id"))
    if not adjudication_id:
        raise DiscoveryEvaluationError("An adjudication needs adjudication_id.")
    kind = _text(row.get("kind")).casefold()
    if kind not in _KINDS:
        raise DiscoveryEvaluationError(
            f"kind must be independent or system_proposal, not {row.get('kind')!r}."
        )
    adjudicator_id = _text(row.get("adjudicator_id"))
    if not adjudicator_id:
        raise DiscoveryEvaluationError(f"Adjudication {adjudication_id} needs adjudicator_id.")
    if "rationale" not in row or not isinstance(row.get("rationale"), str):
        raise DiscoveryEvaluationError(f"Adjudication {adjudication_id} needs a string rationale.")
    flags = {name: _bool(row.get(name), False) for name in _FLAG_NAMES}
    evidence_supported = _bool(row.get("evidence_supported"), True)
    if flags["unsupported_interpretation"]:
        evidence_supported = False
    direction = _direction(row.get("direction"))
    role = _optional_token(row.get("role"))
    if role is not None:
        role = role.casefold()
    wording = _optional_token(row.get("wording"))
    if wording is not None:
        wording = wording.casefold()
    stored_passages = passages if passages is not None else _passages_from_value(row.get("supporting_passages"))
    from_id = _text(row.get("from_id"))
    to_id = _text(row.get("to_id"))
    outcome = row.get("system_outcome")
    if outcome is None or str(outcome).strip() == "":
        system_outcome = None
    else:
        system_outcome = str(outcome).strip().casefold()
        if system_outcome not in _OUTCOMES:
            raise DiscoveryEvaluationError(f"Unknown system_outcome {outcome!r}.")
    if kind == "independent" and evidence_supported:
        if not _code_id(from_id) or not _code_id(to_id):
            raise DiscoveryEvaluationError(
                f"Independent adjudication {adjudication_id} needs from_id and to_id."
            )
        if not stored_passages:
            raise DiscoveryEvaluationError(
                f"Independent adjudication {adjudication_id} needs a supporting passage id."
            )
    system_relationship_id = _optional_token(row.get("system_relationship_id"))
    return {
        "adjudication_id": adjudication_id,
        "kind": kind,
        "adjudicator_id": adjudicator_id,
        "rationale": row.get("rationale"),
        "from_id": from_id,
        "to_id": to_id,
        "direction": direction,
        "role": role,
        "wording": wording,
        "supporting_passages": stored_passages,
        "conceptual_distinctions": _labels(row.get("conceptual_distinctions")),
        "unsupported_interpretation": flags["unsupported_interpretation"],
        "over_merging": flags["over_merging"],
        "loss_of_minority_account": flags["loss_of_minority_account"],
        "loss_of_contradictory_account": flags["loss_of_contradictory_account"],
        "evidence_supported": evidence_supported,
        "system_relationship_id": system_relationship_id,
        "system_outcome": system_outcome,
        "claim_judgment": _claim_judgment(row, version=version, rewrite_old=rewrite_old),
    }


def _one_disagreement(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise DiscoveryEvaluationError("A disagreement must be an object.")
    adjudication_id = _text(row.get("adjudication_id"))
    adjudicator_id = _text(row.get("adjudicator_id") or row.get("disagreement_adjudicator_id"))
    field = _text(row.get("field") or row.get("disagreement_field"))
    if "value" in row:
        value = _json_value(row.get("value"))
    else:
        value = _json_value(row.get("disagreement_value"))
    rationale = row.get("rationale", row.get("disagreement_rationale"))
    if not adjudication_id or not adjudicator_id or not field:
        raise DiscoveryEvaluationError(
            "A disagreement needs adjudication_id, adjudicator_id, and field."
        )
    if not isinstance(rationale, str):
        raise DiscoveryEvaluationError("A disagreement needs a string rationale.")
    try:
        _stable(value)
    except TypeError as exc:
        raise DiscoveryEvaluationError("A disagreement value must be JSON data.") from exc
    return {
        "adjudication_id": adjudication_id,
        "adjudicator_id": adjudicator_id,
        "field": field,
        "value": value,
        "rationale": rationale,
    }


def _schema_of(payload: Mapping[str, Any]) -> str:
    source = str(payload.get("source_schema_version") or "")
    version = str(payload.get("schema_version") or "")
    if source in _ACCEPTED_SCHEMAS:
        return source
    if version in _ACCEPTED_SCHEMAS:
        return version
    if not version:
        return DISCOVERY_SCHEMA_VERSION
    raise DiscoveryEvaluationError(f"schema_version must be {DISCOVERY_SCHEMA_VERSION}.")


def _normalize_payload(payload: Mapping[str, Any] | Sequence[Any], *, force_current: bool = False) -> dict[str, Any]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, Mapping)):
        payload = {"adjudications": list(payload), "schema_version": DISCOVERY_SCHEMA_VERSION}
    if not isinstance(payload, Mapping):
        raise DiscoveryEvaluationError("A discovery file must be an object or a list of adjudications.")
    version = DISCOVERY_SCHEMA_VERSION if force_current else _schema_of(payload)
    incoming = ""
    if isinstance(payload, Mapping):
        incoming = str(payload.get("source_schema_version") or payload.get("schema_version") or "")
    rewrite_old = force_current and incoming in {DISCOVERY_SCHEMA_VERSION_V1, DISCOVERY_SCHEMA_VERSION_V2}
    raw_rows = payload.get("adjudications")
    if raw_rows is None:
        raw_rows = []
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise DiscoveryEvaluationError("adjudications must be a list.")
    adjudications = [_one_adjudication(row, version=version, rewrite_old=rewrite_old) for row in raw_rows]
    seen: set[str] = set()
    for row in adjudications:
        if row["adjudication_id"] in seen:
            raise DiscoveryEvaluationError(f"Duplicate adjudication_id {row['adjudication_id']}.")
        seen.add(row["adjudication_id"])
    adjudications.sort(key=lambda row: row["adjudication_id"])
    raw_disagreements = payload.get("disagreements") or []
    if not isinstance(raw_disagreements, Sequence) or isinstance(raw_disagreements, (str, bytes)):
        raise DiscoveryEvaluationError("disagreements must be a list.")
    disagreements = [_one_disagreement(row) for row in raw_disagreements]
    known_ids = {row["adjudication_id"] for row in adjudications}
    for row in disagreements:
        if row["adjudication_id"] not in known_ids:
            raise DiscoveryEvaluationError(
                f"Disagreement {row['adjudication_id']} does not name an adjudication."
            )
    disagreements.sort(
        key=lambda row: (
            row["adjudication_id"],
            row["adjudicator_id"],
            row["field"],
            _stable(row["value"]),
            row["rationale"],
        )
    )
    system_ids = [
        row["system_relationship_id"]
        for row in adjudications
        if row["kind"] == "system_proposal" and row.get("system_relationship_id")
    ]
    if len(system_ids) != len(set(system_ids)):
        raise DiscoveryEvaluationError("Duplicate system_relationship_id in system_proposal adjudications.")
    return {
        "schema_version": version,
        "label": str(payload.get("label") or ""),
        "note": str(payload.get("note") or ""),
        "adjudications": adjudications,
        "disagreements": disagreements,
        "claim_reviews": _claim_reviews(payload.get("claim_reviews"), version=version),
    }


def _hash_mode(version: str) -> str:
    if version == DISCOVERY_SCHEMA_VERSION_V1:
        return "v1"
    if version == DISCOVERY_SCHEMA_VERSION_V2:
        return "v2"
    return "v3"


def _hash_adjudication(row: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    item = dict(row)
    # eval-1 never stored the judgment. eval-3 stores reviews beside the gold.
    # eval-2 keeps claim_judgment and must not gain keys.
    if mode in {"v1", "v3"}:
        item.pop("claim_judgment", None)
    item.pop("claim_reviews", None)
    return item


def _schema_string(mode: str) -> str:
    if mode == "v1":
        return DISCOVERY_SCHEMA_VERSION_V1
    if mode == "v2":
        return DISCOVERY_SCHEMA_VERSION_V2
    return DISCOVERY_SCHEMA_VERSION


def _canonical_body(doc: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    body = {
        "schema_version": _schema_string(mode),
        "label": doc.get("label") or "",
        "note": doc.get("note") or "",
        "adjudications": [_hash_adjudication(row, mode=mode) for row in doc["adjudications"]],
        "disagreements": doc["disagreements"],
    }
    if mode == "v3":
        body["claim_reviews"] = list(doc.get("claim_reviews") or [])
    return body


def _hash(doc: Mapping[str, Any], *, mode: str) -> str:
    raw = json.dumps(
        _canonical_body(doc, mode=mode),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _same_hash(stored: Any, computed: str) -> bool:
    left = str(stored or "").strip().casefold()
    right = computed.casefold()
    return left == right or left == right.removeprefix("sha256:")


def freeze_discovery_set(payload: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    """Normalize a discovery set as discovery-eval-3 and attach its frozen hash.

    The hash covers schema version, label, note, adjudications, disagreements,
    and ``claim_reviews``. It does not cover ``frozen_hash`` or a leftover
    ``claim_judgment`` on a gold row. Gold stays separate from claim reviews.
    A discovery-eval-1 or discovery-eval-2 body is accepted as input and
    rewritten. That rewrite changes the hash even when there are no claim
    reviews, because the schema version and the review list are part of the
    body. Quote text is included so an edit to the recorded evidence changes
    the hash. Structural matching later ignores that text.
    """
    normalized = _normalize_payload(payload, force_current=True)
    normalized["schema_version"] = DISCOVERY_SCHEMA_VERSION
    normalized["source_schema_version"] = DISCOVERY_SCHEMA_VERSION
    normalized["frozen_hash"] = _hash(normalized, mode="v3")
    normalized["frozen_hash_status"] = "verified"
    return normalized


def _finalize(payload: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    stored = payload.get("frozen_hash") if isinstance(payload, Mapping) else None
    version = DISCOVERY_SCHEMA_VERSION
    if isinstance(payload, Mapping):
        version = _schema_of(payload)
    normalized = _normalize_payload(payload, force_current=False)
    if stored:
        computed = _hash(normalized, mode=_hash_mode(version))
        if not _same_hash(stored, computed):
            raise DiscoveryEvaluationError(
                "frozen_hash does not match the discovery set. The file changed after it was frozen."
            )
        normalized["frozen_hash"] = computed
        normalized["frozen_hash_status"] = "verified"
    else:
        normalized["frozen_hash"] = None
        normalized["frozen_hash_status"] = "absent"
    # The checked hash stays the source schema's hash. The in-memory schema
    # is the current one. Scoring treats a source other than discovery-eval-3
    # as an unbound claim and does not count it correct.
    normalized["source_schema_version"] = version
    normalized["schema_version"] = DISCOVERY_SCHEMA_VERSION
    return normalized


def _parse_jsonl(text: str) -> dict[str, Any]:
    header: dict[str, Any] = {}
    adjudications: list[Any] = []
    disagreements: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiscoveryEvaluationError(f"JSONL line {line_number} is not JSON.") from exc
        if not isinstance(row, Mapping):
            raise DiscoveryEvaluationError(f"JSONL line {line_number} must be an object.")
        record_type = str(row.get("record_type") or "")
        if record_type == "header":
            header = dict(row)
            continue
        if record_type == "disagreement":
            disagreements.append(dict(row))
            continue
        adjudications.append(dict(row))
    version = header.get("schema_version") or DISCOVERY_SCHEMA_VERSION
    return {
        "schema_version": version,
        "label": header.get("label") or "",
        "note": header.get("note") or "",
        "adjudications": adjudications,
        "disagreements": list(header.get("disagreements") or []) + disagreements,
        "claim_reviews": list(header.get("claim_reviews") or []),
        "frozen_hash": header.get("frozen_hash"),
    }


def _parse_csv(text: str) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise DiscoveryEvaluationError("A discovery CSV needs a header row.")
    adjudications: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for row in reader:
        if not any(str(value or "").strip() for value in row.values()):
            continue
        if _text(row.get("disagreement_adjudicator_id")):
            disagreements.append(
                {
                    "adjudication_id": row.get("adjudication_id"),
                    "adjudicator_id": row.get("disagreement_adjudicator_id"),
                    "field": row.get("disagreement_field"),
                    "value": row.get("disagreement_value"),
                    "rationale": row.get("disagreement_rationale") or "",
                }
            )
            continue
        item = dict(row)
        item["supporting_passages"] = _passages_from_csv(row)
        item["rationale"] = str(row.get("rationale") or "")
        adjudications.append(item)
    return {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "label": "",
        "note": "",
        "adjudications": adjudications,
        "disagreements": disagreements,
    }


def load_discovery_file(source: str | Path | Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    """Load JSON, JSONL, or CSV into a normalized discovery set.

    A stored ``frozen_hash`` is recomputed. A mismatch raises
    ``DiscoveryEvaluationError``. A file with no hash loads, and
    ``frozen_hash_status`` is ``absent``. Scoring refuses that file.
    CSV is the adjudicator template. Freeze the normalized JSON before a run
    is scored. This function does not invent adjudications.
    """
    if isinstance(source, Mapping) or (
        isinstance(source, Sequence) and not isinstance(source, (str, bytes))
    ):
        return _finalize(source)
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        payload: Any = _parse_jsonl(text)
    elif suffix == ".csv":
        payload = _parse_csv(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DiscoveryEvaluationError(f"{path.name} is not JSON.") from exc
    return _finalize(payload)


def _proposal_rows(proposals: Any) -> list[Mapping[str, Any]]:
    if proposals is None:
        return []
    if isinstance(proposals, Mapping):
        if "relationships" in proposals:
            rows = proposals.get("relationships") or []
        elif any(key in proposals for key in ("accepted", "withheld", "proposed")):
            rows = []
            for key in ("accepted", "withheld", "proposed"):
                rows.extend(list(proposals.get(key) or []))
        else:
            raise DiscoveryEvaluationError(
                "proposals must be a list or a mapping with relationships."
            )
    elif isinstance(proposals, Sequence) and not isinstance(proposals, (str, bytes)):
        rows = proposals
    else:
        raise DiscoveryEvaluationError("proposals must be a list or a mapping with relationships.")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise DiscoveryEvaluationError("relationships must be a list.")
    return [row for row in rows if isinstance(row, Mapping)]


def _system_passage_ids(row: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()

    def add(value: Any) -> None:
        text = _passage_id(value)
        if text:
            found.add(text)

    add(row.get("passage_id"))
    for key in ("passage_ids", "supporting_passage_ids"):
        value = row.get(key)
        if isinstance(value, str):
            add(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                add(item)
    for case in row.get("supporting_cases") or []:
        if not isinstance(case, Mapping):
            continue
        ids = case.get("passage_ids")
        if isinstance(ids, str):
            add(ids)
        elif isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
            for item in ids:
                add(item)
        add(case.get("passage_id"))
    for key in ("case_quotes", "quotes", "supporting_passages"):
        for item in row.get(key) or []:
            if isinstance(item, Mapping):
                add(item.get("passage_id"))
    for anchor in row.get("endpoint_anchoring") or []:
        if isinstance(anchor, Mapping):
            add(anchor.get("passage_id"))
    return found


def _pair(from_id: Any, to_id: Any) -> tuple[str, str] | None:
    left = _code_id(from_id)
    right = _code_id(to_id)
    if not left or not right:
        return None
    return tuple(sorted((left, right)))


def _gold_passages(gold: Mapping[str, Any]) -> set[str]:
    return {row["passage_id"] for row in gold.get("supporting_passages") or [] if row.get("passage_id")}


def _system_direction_in_gold_order(gold: Mapping[str, Any], system: Mapping[str, Any]) -> str | None:
    """Translate the system's direction into the gold endpoint order.

    ``stated_direction`` or ``direction`` is in the system's from-to order.
    Swapped endpoints flip ``from_to`` and ``to_from``. When the system names
    no direction, the same endpoint order is ``from_to`` and the swapped
    order is ``to_from``.
    """
    gold_from = _code_id(gold.get("from_id"))
    gold_to = _code_id(gold.get("to_id"))
    system_from = _code_id(system.get("from_id"))
    system_to = _code_id(system.get("to_id"))
    same = system_from == gold_from and system_to == gold_to
    swapped = system_from == gold_to and system_to == gold_from
    raw = system.get("stated_direction")
    if raw is None or str(raw).strip() == "":
        raw = system.get("direction")
    stated: str | None
    if raw is None or str(raw).strip() == "":
        stated = None
    else:
        text = str(raw).strip().casefold().replace("-", "_").replace(" ", "_")
        if text in {"", "unspecified", "none", "unknown"}:
            stated = None
        elif text in _DIRECTIONS:
            stated = text
        else:
            stated = "unrecognized"
    if stated is None:
        if same:
            return "from_to"
        if swapped:
            return "to_from"
        return None
    if stated == "undirected":
        return "undirected"
    if stated == "unrecognized":
        return "unrecognized"
    if swapped:
        return "to_from" if stated == "from_to" else "from_to"
    return stated


def _system_role(row: Mapping[str, Any]) -> str | None:
    for key in ("paradigm_role", "role"):
        if key in row and _optional_token(row.get(key)):
            return _optional_token(row.get(key)).casefold()
    return None


def _system_wording(row: Mapping[str, Any]) -> str | None:
    return _optional_token(row.get("allowed_wording"))


def _field_report(gold: Mapping[str, Any], system: Mapping[str, Any]) -> dict[str, Any]:
    """Compare only the fields the adjudicator specified.

    Tier labels, ``attribution_source``, and the free-text statement are not
    read. A wrong direction stays a distortion even when ``evidence_tier`` is 3.
    """
    disagreements: list[str] = []
    direction_agrees = None
    role_agrees = None
    wording_agrees = None
    if gold.get("direction"):
        system_direction = _system_direction_in_gold_order(gold, system)
        direction_agrees = system_direction == gold.get("direction")
        if not direction_agrees:
            disagreements.append("direction")
    if gold.get("role"):
        role_agrees = _system_role(system) == gold.get("role")
        if not role_agrees:
            disagreements.append("role")
    if gold.get("wording"):
        system_wording = _system_wording(system)
        wording_agrees = system_wording is not None and system_wording.casefold() == gold.get("wording")
        if not wording_agrees:
            disagreements.append("wording")
    return {
        "direction_agrees": direction_agrees,
        "role_agrees": role_agrees,
        "wording_agrees": wording_agrees,
        "disagreements": disagreements,
    }


def _eligible(system: Mapping[str, Any] | None, gold: Mapping[str, Any]) -> tuple[bool, bool | None]:
    """Known eligibility from the export boolean, else from ``system_outcome``.

    The export boolean wins when both are present. A missing boolean is
    unknown. ``schema_unparsed`` does not decide eligibility.
    """
    if system is not None and isinstance(system.get("downstream_eligible"), bool):
        return True, system.get("downstream_eligible")
    outcome = gold.get("system_outcome")
    if outcome in _WITHHELD_OUTCOMES:
        return True, False
    if outcome in {"proposed", "accepted"}:
        return True, True
    return False, None


def _withheld(system: Mapping[str, Any] | None, gold: Mapping[str, Any]) -> bool:
    export_bool = system is not None and isinstance(system.get("downstream_eligible"), bool)
    if export_bool and system.get("downstream_eligible") is False:
        return True
    disposition = ""
    flags: set[str] = set()
    if system is not None:
        disposition = str(system.get("disposition") or "").strip().casefold()
        flags = {str(flag) for flag in (system.get("flags") or []) if str(flag or "")}
    if "repair_failed" in flags:
        return True
    if disposition in _WITHHELD_DISPOSITIONS:
        return True
    if not export_bool and gold.get("system_outcome") in _WITHHELD_OUTCOMES:
        return True
    return False


def _expenditure_view(expenditure: Any) -> Mapping[str, Any] | None:
    if expenditure is None:
        return None
    if not isinstance(expenditure, Mapping):
        raise DiscoveryEvaluationError("expenditure must be a mapping or omitted.")
    if "prompt_log" in expenditure or "reread_notes" in expenditure:
        from automated_grounded_theory.sampling import reread_interpretation_report

        return reread_interpretation_report(expenditure)
    if "reread_expenditure_complete" in expenditure:
        return expenditure
    return {"reread_expenditure_complete": False}


def _value_rates(recovered: int, expenditure: Any, *, scored: bool) -> tuple[float | None, float | None]:
    """Recovered independent relationships per reread token and per reread dollar.

    The denominators are ``reread_expenditure_tokens`` and
    ``reread_expenditure_cost_usd`` only. Other run spend is not included.
    Both stay null when adjudications are absent, when reread expenditure is
    incomplete, or when the token or dollar denominator is not positive.
    Zero recoveries over a positive complete reread spend is 0, not null.
    """
    if not scored:
        return None, None
    report = _expenditure_view(expenditure)
    if report is None or report.get("reread_expenditure_complete") is not True:
        return None, None
    per_token = None
    tokens = report.get("reread_expenditure_tokens")
    if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) and tokens > 0:
        per_token = round(recovered / float(tokens), 4)
    per_dollar = None
    cost = report.get("reread_expenditure_cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
        per_dollar = round(recovered / float(cost), 4)
    return per_token, per_dollar


def _null_rates() -> dict[str, Any]:
    return {name: None for name in _RATE_NAMES}


def _absent_report() -> dict[str, Any]:
    return {
        "evaluation": "discovery",
        "metric_source": None,
        "model_judgments_used": False,
        "tier_labels_used": False,
        "adjudications_absent": True,
        "schema_version": None,
        "frozen_hash": None,
        "frozen_hash_status": None,
        "adjudicator_disagreement_count": None,
        "disagreements": [],
        "conceptual_distinction_labels": [],
        "conceptual_distinction_count": None,
        "unsupported_interpretation_count": None,
        "over_merging_count": None,
        "loss_of_minority_account_count": None,
        "loss_of_contradictory_account_count": None,
        "recovered_count": None,
        "missed_count": None,
        "distorted_count": None,
        "unmatched_proposal_count": None,
        "withheld_but_supported_count": None,
        "discovery_recall_denominator": None,
        "discovery_recall_coverage": None,
        "discovery_precision_denominator": None,
        "discovery_precision_coverage": None,
        "discovery_distortion_denominator": None,
        "gate_false_negative_population": None,
        "gate_false_negative_rate_known_denominator": None,
        "gate_false_negative_rate_coverage": None,
        **_null_rates(),
        "claim_correct_denominator": None,
        "claim_correct_coverage": None,
        "claim_matched_count": None,
        "claim_stale_count": None,
        "source_schema_version": None,
        "relationships": [],
        "unmatched_proposals": [],
        "unmatched_system_relationship_ids": None,
    }


def _candidate_indexes(
    gold: Mapping[str, Any],
    prepared: Sequence[dict[str, Any]],
) -> list[int]:
    """Indexes of system rows that share the code-id pair and a passage id."""
    gold_pair = _pair(gold.get("from_id"), gold.get("to_id"))
    gold_passages = _gold_passages(gold)
    if gold_pair is None or not gold_passages or not gold.get("evidence_supported"):
        return []
    found: list[int] = []
    for index, item in enumerate(prepared):
        if item["synthetic"] or item["pair"] != gold_pair:
            continue
        if gold_passages.isdisjoint(item["passages"]):
            continue
        found.append(index)
    return found


def _gold_content_key(gold: Mapping[str, Any]) -> tuple[Any, ...]:
    """Canonical gold content. Adjudication ids are not part of the key."""
    return (
        _code_id(gold.get("from_id")),
        _code_id(gold.get("to_id")),
        tuple(sorted(_gold_passages(gold))),
        str(gold.get("direction") or ""),
        str(gold.get("role") or ""),
        str(gold.get("wording") or ""),
        " ".join(str(gold.get("rationale") or "").casefold().split()),
    )


def _proposal_content_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Canonical proposal content. Relationship ids are not part of the key."""
    direction = row.get("stated_direction")
    if direction is None or str(direction).strip() == "":
        direction = row.get("direction")
    role = row.get("paradigm_role")
    if role is None or str(role).strip() == "":
        role = row.get("role")
    return (
        _code_id(row.get("from_id")),
        _code_id(row.get("to_id")),
        tuple(sorted(_system_passage_ids(row))),
        " ".join(str(direction or "").casefold().split()),
        " ".join(str(role or "").casefold().split()),
        " ".join(str(row.get("allowed_wording") or "").casefold().split()),
        " ".join(str(row.get("statement") or "").casefold().split()),
    )


def _hungarian_maximize(weight: list[list[int]]) -> list[int]:
    """Column assigned to each row of a square matrix, maximizing the sum.

    Missing structural edges use a large negative weight so a zero-weight
    padding column is preferred. The result is one column per row.
    """
    n = len(weight)
    if n == 0:
        return []
    peak = 0
    for row in weight:
        for value in row:
            magnitude = abs(int(value))
            if magnitude > peak:
                peak = magnitude
    inf = max(10**18, peak * (n + 2) + 1)
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = -weight[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def _lex_weight(primary: int, rank: int, edge_count: int, recovery_unit: int) -> int:
    """One integer whose maximum is the documented lexicographic objective.

    Full agreement is ``recovery_unit`` and a distortion is 1, with
    ``recovery_unit`` equal to one more than the gold rows in the component.
    One full recovery therefore beats every distorted match in that component.
    The content bonus is smaller than one primary unit. For a component of at
    most 12 structural edges the bonus is geometric, so the canonical content
    order is exact. A larger component keeps the first two objectives and uses
    a rank bonus that is not a full ordering of every tied set.
    """
    if edge_count <= 12:
        base = edge_count + 1
        scale = base**edge_count
        bonus = base ** (edge_count - 1 - rank)
    else:
        scale = edge_count * recovery_unit + 1
        bonus = edge_count - rank
    return primary * scale + bonus


def _solve_component(
    gold_rows: Sequence[Mapping[str, Any]],
    edges: Sequence[dict[str, Any]],
) -> dict[str, tuple[int, dict[str, Any]]]:
    """One lexicographic assignment for a candidate-connected component.

    Each gold row has an unmatched column of weight 0. A missing structural
    edge is negative and is never chosen over staying unmatched.
    """
    ordered = sorted(edges, key=lambda edge: edge["content"])
    gold_ids: list[int] = []
    prop_ids: list[int] = []
    for edge in ordered:
        if edge["gold_index"] not in gold_ids:
            gold_ids.append(edge["gold_index"])
        if edge["prop_index"] not in prop_ids:
            prop_ids.append(edge["prop_index"])
    local_gold = {gold_index: local for local, gold_index in enumerate(gold_ids)}
    local_prop = {prop_index: local for local, prop_index in enumerate(prop_ids)}
    n_gold = len(gold_ids)
    n_prop = len(prop_ids)
    recovery_unit = n_gold + 1
    edge_count = len(ordered)
    negative = -(10**15)
    width = n_prop + n_gold
    matrix = [[0 for _ in range(width)] for _ in range(width)]
    for row in range(n_gold):
        for col in range(n_prop):
            matrix[row][col] = negative
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for rank, edge in enumerate(ordered):
        primary = recovery_unit if edge["recovered"] else 1
        matrix[local_gold[edge["gold_index"]]][local_prop[edge["prop_index"]]] = _lex_weight(
            primary, rank, edge_count, recovery_unit
        )
        by_pair[(edge["gold_index"], edge["prop_index"])] = edge
    assignment = _hungarian_maximize(matrix)
    chosen: dict[str, tuple[int, dict[str, Any]]] = {}
    for local_g, local_p in enumerate(assignment):
        if local_g >= n_gold or local_p < 0 or local_p >= n_prop:
            continue
        if matrix[local_g][local_p] <= 0:
            continue
        gold_index = gold_ids[local_g]
        prop_index = prop_ids[local_p]
        edge = by_pair[(gold_index, prop_index)]
        chosen[gold_rows[gold_index]["adjudication_id"]] = (prop_index, edge["fields"])
    return chosen


def _assign_independent(
    gold_rows: Sequence[Mapping[str, Any]],
    prepared: Sequence[dict[str, Any]],
) -> dict[str, tuple[int, dict[str, Any]]]:
    """Lexicographic structural assignment for independent gold only.

    Only pairs with a structural match are edges. Each connected component
    is solved once. Proposals with no structural edge are not in the matrix.
    A proposal review is not consulted here and cannot block a candidate.
    """
    gold_indexes = [index for index, gold in enumerate(gold_rows) if gold.get("kind") == "independent"]
    if not gold_indexes:
        return {}
    edges: list[dict[str, Any]] = []
    for gold_index in gold_indexes:
        gold = gold_rows[gold_index]
        for prop_index in _candidate_indexes(gold, prepared):
            fields = _field_report(gold, prepared[prop_index]["row"])
            edges.append(
                {
                    "gold_index": gold_index,
                    "prop_index": prop_index,
                    "recovered": not fields["disagreements"],
                    "content": (
                        _gold_content_key(gold),
                        _proposal_content_key(prepared[prop_index]["row"]),
                    ),
                    "fields": fields,
                }
            )
    if not edges:
        return {}
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: tuple[str, int], right: tuple[str, int]) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for edge in edges:
        union(("g", edge["gold_index"]), ("p", edge["prop_index"]))
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for edge in edges:
        groups.setdefault(find(("g", edge["gold_index"])), []).append(edge)
    assigned: dict[str, tuple[int, dict[str, Any]]] = {}
    for group in groups.values():
        assigned.update(_solve_component(gold_rows, group))
    return assigned


def _proposal_claim_direction(row: Mapping[str, Any]) -> str:
    raw = row.get("stated_direction")
    if raw is None or str(raw).strip() == "":
        raw = row.get("direction")
    return " ".join(str(raw or "").casefold().split())


def _claim_resolution(
    *,
    source_schema: str,
    reviews: Sequence[Mapping[str, Any]],
    proposal: Mapping[str, Any] | None,
    scored_run: str,
    legacy_verdicts: Sequence[str],
) -> tuple[str, str, int]:
    """Return effective verdict, binding, and how many stale reviews attach here.

    ``fresh`` is a review whose run, relationship, and statement hash all
    match. Anything else that tried to judge this relationship is ``stale``
    and counts as not judged. No review is ``unjudged``.
    """
    if source_schema != DISCOVERY_SCHEMA_VERSION:
        if any(verdict != "not_judged" for verdict in legacy_verdicts):
            return "not_judged", "stale", 1
        return "not_judged", "unjudged", 0
    fresh: list[Mapping[str, Any]] = []
    stale = 0
    statement = "" if proposal is None else proposal.get("statement")
    direction = "" if proposal is None else _proposal_claim_direction(proposal)
    for review in reviews:
        bound = review.get("binding_status") == "bound"
        run_ok = bool(scored_run) and review.get("run_id") == scored_run
        if bound and run_ok and proposal is not None:
            digest = claim_statement_hash(statement, direction, review.get("qualification"))
            if _same_hash(review.get("statement_hash"), digest):
                fresh.append(review)
                continue
        stale += 1
    if fresh:
        return str(fresh[0].get("verdict") or "not_judged"), "fresh", stale
    if stale:
        return "not_judged", "stale", stale
    return "not_judged", "unjudged", 0


def score_discovery(
    proposals: Any,
    discovery: str | Path | Mapping[str, Any] | Sequence[Any] | None = None,
    expenditure: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compare system relationships with a frozen human discovery set.

    ``discovery`` None leaves every rate null and does not raise. A discovery
    mapping or file with no ``frozen_hash`` raises. Passing a run state as
    ``expenditure`` (a mapping with ``prompt_log`` or ``reread_notes``) reuses
    ``reread_interpretation_report``. A report that already has
    ``reread_expenditure_complete`` is used as given.

    Recall uses independent, evidence-supported adjudications only.
    ``system_proposal`` rows bind by ``system_relationship_id`` and are
    gate-false-negative evidence, not extra recall gold. That bind does not
    remove the proposal from independent matching. A missing or duplicate
    export id is ``unmatched_system_proposal`` and is not attached to a
    different proposal. Distorted matches stay in the recall denominator
    and out of the numerator. Precision uses non-synthetic proposals that
    have both endpoint ids. A bound system proposal is not a structural true
    positive unless independent gold also recovers it. The distortion rate
    uses recovered and distorted matches only. ``downstream_eligible`` does
    not enter those three rates. Structural recall does not read the proposed
    statement or a claim review.

    ``run_id`` selects which claim reviews bind. When it is omitted and
    ``proposals`` is a mapping, ``proposals["run_id"]`` is used. A review
    whose run, relationship, or statement hash does not match is stale.

    The gate false-negative population is evidence-supported items the system
    proposed: a matched row, or a ``system_proposal`` adjudication. An
    independent relationship that was never proposed is missed, not a gate
    false negative. The full rate is null unless every population member has
    known eligibility. The known rate uses only those members.
    """
    if discovery is None:
        return _absent_report()
    if isinstance(discovery, (str, Path)):
        loaded = load_discovery_file(discovery)
    else:
        loaded = _finalize(discovery)
    if loaded.get("frozen_hash_status") != "verified":
        raise DiscoveryEvaluationError(
            "discovery set has no frozen_hash. Freeze the file before scoring."
        )

    scored_run = _text(run_id)
    if not scored_run and isinstance(proposals, Mapping):
        scored_run = _text(proposals.get("run_id"))

    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(_proposal_rows(proposals)):
        pair = _pair(row.get("from_id"), row.get("to_id"))
        prepared.append(
            {
                "index": index,
                "row": row,
                "synthetic": bool(row.get("synthetic")),
                "pair": pair,
                "passages": _system_passage_ids(row),
                "relationship_id": str(row.get("relationship_id") or ""),
            }
        )

    gold_rows = list(loaded["adjudications"])
    by_relationship: dict[str, list[int]] = {}
    for item in prepared:
        if item["relationship_id"]:
            by_relationship.setdefault(item["relationship_id"], []).append(item["index"])
    assignment: dict[str, int] = {}
    field_by_gold: dict[str, dict[str, Any]] = {}
    linked: set[str] = set()
    unmatched_system_ids: list[str] = []
    used: set[int] = set()
    # Independent matching does not see system_proposal binds. The same
    # proposal can be recovered for gold and linked for a review.
    structural = _assign_independent(gold_rows, prepared)
    for adjudication_id, (index, fields) in structural.items():
        used.add(index)
        assignment[adjudication_id] = index
        field_by_gold[adjudication_id] = fields
    for gold in gold_rows:
        if gold["kind"] != "system_proposal":
            continue
        named = str(gold.get("system_relationship_id") or "")
        hits = by_relationship.get(named) or []
        if not named or len(hits) != 1:
            if named:
                unmatched_system_ids.append(named)
            continue
        used.add(hits[0])
        assignment[gold["adjudication_id"]] = hits[0]
        linked.add(gold["adjudication_id"])

    relationships: list[dict[str, Any]] = []
    recovered = missed = distorted = 0
    matched_recovered = matched_distorted = 0
    population = 0
    population_known = 0
    withheld_count = 0
    withheld_known = 0
    recall_denominator = 0
    recall_scoreable = 0
    claim_members: dict[str, dict[str, Any]] = {}

    for gold in gold_rows:
        in_recall = gold["kind"] == "independent" and gold["evidence_supported"]
        if in_recall:
            recall_denominator += 1
            if _pair(gold.get("from_id"), gold.get("to_id")) and _gold_passages(gold):
                recall_scoreable += 1
        matched_index = assignment.get(gold["adjudication_id"])
        system = prepared[matched_index]["row"] if matched_index is not None else None
        fields = field_by_gold.get(gold["adjudication_id"])
        if gold["adjudication_id"] in linked:
            status = "linked"
            direction_agrees = role_agrees = wording_agrees = None
            overlap = sorted(prepared[matched_index]["passages"])
        elif matched_index is None:
            if in_recall:
                status = "missed"
                missed += 1
            elif gold["kind"] == "system_proposal":
                status = "unmatched_system_proposal"
            else:
                status = "not_scored"
            direction_agrees = role_agrees = wording_agrees = None
            overlap = []
        else:
            assert fields is not None
            direction_agrees = fields["direction_agrees"]
            role_agrees = fields["role_agrees"]
            wording_agrees = fields["wording_agrees"]
            overlap = sorted(_gold_passages(gold) & prepared[matched_index]["passages"])
            if fields["disagreements"]:
                status = "distorted"
                matched_distorted += 1
                if in_recall:
                    distorted += 1
            else:
                status = "recovered"
                matched_recovered += 1
                if in_recall:
                    recovered += 1
        rel_id = None if system is None else str(system.get("relationship_id") or "") or None
        in_claim = bool(rel_id) and (
            (status in {"recovered", "distorted"} and in_recall) or gold["adjudication_id"] in linked
        )
        if in_claim and rel_id is not None:
            legacy = gold.get("claim_judgment") if isinstance(gold.get("claim_judgment"), Mapping) else {}
            member = claim_members.setdefault(rel_id, {"proposal": system, "legacy": [], "rows": []})
            member["legacy"].append(str(legacy.get("verdict") or "not_judged"))
        in_population = gold["evidence_supported"] and (
            matched_index is not None or gold["kind"] == "system_proposal"
        )
        known, eligible = _eligible(system, gold)
        withheld = _withheld(system, gold) if in_population else False
        if not in_population:
            gate_status = "not_applicable"
        elif withheld:
            gate_status = "withheld_but_supported"
        elif known:
            gate_status = "passed"
        else:
            gate_status = "unknown"
        if in_population:
            population += 1
            if withheld:
                withheld_count += 1
            if known:
                population_known += 1
                if withheld:
                    withheld_known += 1
        relationships.append(
            {
                "adjudication_id": gold["adjudication_id"],
                "kind": gold["kind"],
                "relationship_id": None if system is None else str(system.get("relationship_id") or "") or None,
                "discovery_status": status,
                "gate_status": gate_status,
                "from_id": gold.get("from_id") or None,
                "to_id": gold.get("to_id") or None,
                "direction_agrees": direction_agrees,
                "role_agrees": role_agrees,
                "wording_agrees": wording_agrees,
                "passage_ids": overlap,
                "claim_verdict": "not_judged",
                "claim_binding": "unjudged",
                "eligibility_known": known if in_population else False,
                "downstream_eligible": eligible if in_population else None,
                "withheld": withheld,
            }
        )
        if in_claim and rel_id is not None:
            claim_members[rel_id]["rows"].append(len(relationships) - 1)

    reviews_by_rel: dict[str, list[Mapping[str, Any]]] = {}
    for review in loaded.get("claim_reviews") or []:
        named = str(review.get("relationship_id") or "")
        if named:
            reviews_by_rel.setdefault(named, []).append(review)
    claim_matched = claim_judged = claim_correct = claim_stale = 0
    source_schema = str(loaded.get("source_schema_version") or "")
    for rel_id, member in claim_members.items():
        verdict, binding, stale = _claim_resolution(
            source_schema=source_schema,
            reviews=reviews_by_rel.get(rel_id) or [],
            proposal=member["proposal"],
            scored_run=scored_run,
            legacy_verdicts=member["legacy"],
        )
        claim_stale += stale
        claim_matched += 1
        if binding == "fresh" and verdict in {"correct", "incorrect", "partly_correct"}:
            claim_judged += 1
            if verdict == "correct":
                claim_correct += 1
        for index in member["rows"]:
            relationships[index]["claim_verdict"] = verdict
            relationships[index]["claim_binding"] = binding

    unmatched: list[dict[str, Any]] = []
    precision_denominator = 0
    precision_anchored = 0
    for item in prepared:
        if item["synthetic"] or item["pair"] is None:
            continue
        precision_denominator += 1
        if item["passages"]:
            precision_anchored += 1
        if item["index"] in used:
            continue
        row = item["row"]
        record = {
            "adjudication_id": None,
            "kind": None,
            "relationship_id": item["relationship_id"] or None,
            "discovery_status": "unmatched_proposal",
            "gate_status": "not_applicable",
            "from_id": _text(row.get("from_id")) or None,
            "to_id": _text(row.get("to_id")) or None,
            "direction_agrees": None,
            "role_agrees": None,
            "wording_agrees": None,
            "passage_ids": sorted(item["passages"]),
            "claim_verdict": None,
            "claim_binding": None,
            "eligibility_known": False,
            "downstream_eligible": None,
            "withheld": False,
        }
        unmatched.append(record)
        relationships.append(record)

    identified = population > 0 and population_known == population
    per_token, per_dollar = _value_rates(recovered, expenditure, scored=True)
    labels: list[str] = []
    for gold in gold_rows:
        for label in gold["conceptual_distinctions"]:
            if label not in labels:
                labels.append(label)
    labels.sort()
    return {
        "evaluation": "discovery",
        "metric_source": "human_adjudication",
        "model_judgments_used": False,
        "tier_labels_used": False,
        "adjudications_absent": False,
        "schema_version": loaded["schema_version"],
        "frozen_hash": loaded["frozen_hash"],
        "frozen_hash_status": loaded["frozen_hash_status"],
        "adjudicator_disagreement_count": len(loaded["disagreements"]),
        "disagreements": list(loaded["disagreements"]),
        "conceptual_distinction_labels": labels,
        "conceptual_distinction_count": len(labels),
        "unsupported_interpretation_count": sum(1 for row in gold_rows if row["unsupported_interpretation"]),
        "over_merging_count": sum(1 for row in gold_rows if row["over_merging"]),
        "loss_of_minority_account_count": sum(1 for row in gold_rows if row["loss_of_minority_account"]),
        "loss_of_contradictory_account_count": sum(
            1 for row in gold_rows if row["loss_of_contradictory_account"]
        ),
        "recovered_count": recovered,
        "missed_count": missed,
        "distorted_count": distorted,
        "unmatched_proposal_count": len(unmatched),
        "withheld_but_supported_count": withheld_count,
        "discovery_recall": _rate(recovered, recall_denominator),
        "discovery_recall_denominator": recall_denominator,
        "discovery_recall_coverage": _rate(recall_scoreable, recall_denominator),
        "discovery_precision": _rate(matched_recovered, precision_denominator),
        "discovery_precision_denominator": precision_denominator,
        "discovery_precision_coverage": _rate(precision_anchored, precision_denominator),
        "discovery_distortion_rate": _rate(matched_distorted, matched_recovered + matched_distorted),
        "discovery_distortion_denominator": matched_recovered + matched_distorted,
        "gate_false_negative_population": population,
        "gate_false_negative_rate": _rate(withheld_count, population) if identified else None,
        "gate_false_negative_rate_known": _rate(withheld_known, population_known),
        "gate_false_negative_rate_known_denominator": population_known,
        "gate_false_negative_rate_coverage": _rate(population_known, population),
        "discovery_recovered_per_reread_token": per_token,
        "discovery_recovered_per_reread_dollar": per_dollar,
        "claim_correct_rate": _rate(claim_correct, claim_judged),
        "claim_correct_denominator": claim_judged,
        "claim_correct_coverage": _rate(claim_judged, claim_matched),
        "claim_matched_count": claim_matched,
        "claim_stale_count": claim_stale,
        "source_schema_version": loaded.get("source_schema_version"),
        "relationships": relationships,
        "unmatched_proposals": unmatched,
        "unmatched_system_relationship_ids": unmatched_system_ids,
    }
