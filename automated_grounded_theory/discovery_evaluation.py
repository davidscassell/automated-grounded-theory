"""Score a frozen human discovery set against a run's relationship export.

This is not gate evaluation. ``score_gate_adjudications`` only scores
adjudications whose relationship id is already in the accepted or withheld
export, so a relationship the run never proposed cannot lower that recall.
Discovery starts from relationships a person identified independently, then
asks which of those the run recovered, distorted, or missed.

Matching is deterministic. Endpoint identity is a normalized code id. A
passage anchor is a shared passage id. Quotes, embeddings, and model
judgments are not match signals. Tier labels are not match signals.

A rate whose denominator is empty stays null. A full gate false-negative
rate stays null unless eligibility is known for every row in that population,
the same rule as the 0.9.14 downstream rates. Per-token and per-dollar
figures stay null unless reread expenditure is complete.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

DISCOVERY_SCHEMA_VERSION = "discovery-eval-1"

_KINDS = frozenset({"independent", "system_proposal"})
_DIRECTIONS = frozenset({"from_to", "to_from", "undirected"})
_OUTCOMES = frozenset({"rejected", "withheld", "repair_failed", "proposed", "accepted"})
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
    "discovery_recovered_per_token",
    "discovery_recovered_per_dollar",
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


def _one_adjudication(row: Mapping[str, Any], *, passages: list[dict[str, str]] | None = None) -> dict[str, Any]:
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


def _normalize_payload(payload: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, Mapping)):
        payload = {"adjudications": list(payload), "schema_version": DISCOVERY_SCHEMA_VERSION}
    if not isinstance(payload, Mapping):
        raise DiscoveryEvaluationError("A discovery file must be an object or a list of adjudications.")
    version = payload.get("schema_version") or DISCOVERY_SCHEMA_VERSION
    if version != DISCOVERY_SCHEMA_VERSION:
        raise DiscoveryEvaluationError(f"schema_version must be {DISCOVERY_SCHEMA_VERSION}.")
    raw_rows = payload.get("adjudications")
    if raw_rows is None:
        raw_rows = []
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise DiscoveryEvaluationError("adjudications must be a list.")
    adjudications = [_one_adjudication(row) for row in raw_rows]
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
    return {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "label": str(payload.get("label") or ""),
        "note": str(payload.get("note") or ""),
        "adjudications": adjudications,
        "disagreements": disagreements,
    }


def _canonical_body(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": doc["schema_version"],
        "label": doc.get("label") or "",
        "note": doc.get("note") or "",
        "adjudications": doc["adjudications"],
        "disagreements": doc["disagreements"],
    }


def _hash(doc: Mapping[str, Any]) -> str:
    raw = json.dumps(
        _canonical_body(doc),
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
    """Normalize a discovery set and attach its frozen hash.

    The hash covers schema version, label, note, adjudications, and
    disagreements. It does not cover ``frozen_hash`` itself. Quote text is
    included so an edit to the recorded evidence changes the hash. Matching
    later ignores that text.
    """
    normalized = _normalize_payload(payload)
    normalized["frozen_hash"] = _hash(normalized)
    normalized["frozen_hash_status"] = "verified"
    return normalized


def _finalize(payload: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    stored = payload.get("frozen_hash") if isinstance(payload, Mapping) else None
    normalized = _normalize_payload(payload)
    if stored:
        computed = _hash(normalized)
        if not _same_hash(stored, computed):
            raise DiscoveryEvaluationError(
                "frozen_hash does not match the discovery set. The file changed after it was frozen."
            )
        normalized["frozen_hash"] = computed
        normalized["frozen_hash_status"] = "verified"
    else:
        normalized["frozen_hash"] = None
        normalized["frozen_hash_status"] = "absent"
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
    """Recovered independent relationships per token and per dollar.

    Both stay null when adjudications are absent, when expenditure is
    incomplete, or when the token or dollar denominator is not positive.
    Zero recoveries over a positive complete spend is 0, not null.
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
        "relationships": [],
        "unmatched_proposals": [],
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


def score_discovery(
    proposals: Any,
    discovery: str | Path | Mapping[str, Any] | Sequence[Any] | None = None,
    expenditure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare system relationships with a frozen human discovery set.

    ``discovery`` None leaves every rate null and does not raise. A discovery
    mapping or file with no ``frozen_hash`` raises. Passing a run state as
    ``expenditure`` (a mapping with ``prompt_log`` or ``reread_notes``) reuses
    ``reread_interpretation_report``. A report that already has
    ``reread_expenditure_complete`` is used as given.

    Recall uses independent, evidence-supported adjudications only.
    ``system_proposal`` rows are gate-false-negative evidence, not extra
    recall gold. Distorted matches stay in the recall denominator and out of
    the numerator. Precision uses non-synthetic proposals that have both
    endpoint ids. The distortion rate uses recovered and distorted matches
    only. ``downstream_eligible`` does not enter those three rates.

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
    assignment: dict[str, int] = {}
    field_by_gold: dict[str, dict[str, Any]] = {}
    used: set[int] = set()
    ordered = sorted(gold_rows, key=lambda row: (0 if row["kind"] == "independent" else 1, row["adjudication_id"]))
    for gold in ordered:
        choices: list[tuple[int, int, str, dict[str, Any]]] = []
        for index in _candidate_indexes(gold, prepared):
            if index in used:
                continue
            fields = _field_report(gold, prepared[index]["row"])
            rank = 0 if not fields["disagreements"] else 1
            choices.append((rank, index, prepared[index]["relationship_id"], fields))
        if not choices:
            continue
        choices.sort(key=lambda item: (item[0], item[2], item[1]))
        _rank, index, _relationship_id, fields = choices[0]
        used.add(index)
        assignment[gold["adjudication_id"]] = index
        field_by_gold[gold["adjudication_id"]] = fields

    relationships: list[dict[str, Any]] = []
    recovered = missed = distorted = 0
    matched_recovered = matched_distorted = 0
    population = 0
    population_known = 0
    withheld_count = 0
    withheld_known = 0
    recall_denominator = 0
    recall_scoreable = 0

    for gold in gold_rows:
        in_recall = gold["kind"] == "independent" and gold["evidence_supported"]
        if in_recall:
            recall_denominator += 1
            if _pair(gold.get("from_id"), gold.get("to_id")) and _gold_passages(gold):
                recall_scoreable += 1
        matched_index = assignment.get(gold["adjudication_id"])
        system = prepared[matched_index]["row"] if matched_index is not None else None
        fields = field_by_gold.get(gold["adjudication_id"])
        if matched_index is None:
            if in_recall:
                status = "missed"
                missed += 1
            elif gold["kind"] == "system_proposal":
                status = "unlinked"
            else:
                status = "not_scored"
            direction_agrees = role_agrees = wording_agrees = None
            overlap: list[str] = []
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
                "eligibility_known": known if in_population else False,
                "downstream_eligible": eligible if in_population else None,
                "withheld": withheld,
            }
        )

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
        "discovery_recovered_per_token": per_token,
        "discovery_recovered_per_dollar": per_dollar,
        "relationships": relationships,
        "unmatched_proposals": unmatched,
    }
