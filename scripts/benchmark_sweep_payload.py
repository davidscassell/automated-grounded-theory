#!/usr/bin/env python3
"""Character counts for a sweep from 20 codes through 1280.

The review measured 0.6.4 user messages at chunk size 20: about 9.2k, 54k,
179k, and 644k characters. That request pretty-printed every chunk plus a
label index that repeated every code's definition and examples, and it copied
the first quote into a separate ``example`` field.

0.6.5 kept full evidence on the chunk and on at most ``cross_chunk_limit``
similar codes, but still repeated an id-and-label index of the whole book on
every chunk. That stays near-linear only while the book is small. At 160
codes it is about 103k characters; at 640 it is about 1.12M.

0.6.7 sends a bounded neighborhood with each chunk and one compact label-only
discovery request for the whole book. The table runs through 1280 codes so
the near-linear claim covers the range where the repeated index bends.

    python scripts/benchmark_sweep_payload.py

No API key. Exit status is 0 when the 0.6.4 column stays within 3% of the
published counts, the 0.6.7 column stays near-linear across the whole range,
and the repeated-index column does not.
"""

from pydantic import BaseModel, Field

from automated_grounded_theory.codebook import _code_id
from automated_grounded_theory.context import ResearchContext
from automated_grounded_theory.prompts import STANCE_POLICY, aggregate_sweep_chars
from automated_grounded_theory.validation import code_key

SIZES = (20, 40, 80, 160, 320, 640, 1280)
CHUNK_SIZE = 20
# Rounded counts from the review, user-message characters, chunk size 20.
PUBLISHED = {20: 9200, 40: 54000, 80: 179000, 160: 644000}

# The 0.6.4 sweep note. Later audit wording is not part of that payload.
LEGACY_NOTE = (
    "Merge only when the definitions name the same concept and the quotes show it: "
    "same action, same object, contrast checked, and no material contrast. "
    + STANCE_POLICY
    + " Record same_stance, but do not require it. "
    "Rewrite the survivor label and definition from the member quotes in this same response."
    " No similarity hints. Read every code and propose duplicates yourself."
)

CONTEXT = ResearchContext(research_question="How do people leave?")


class _LegacyCode(BaseModel):
    """0.6.4 codebook row, including the duplicated first quote."""

    code_id: str
    label: str
    definition: str = ""
    frequency: int = 0
    example: str = ""
    examples: list[dict] = Field(default_factory=list)
    valence: str = ""


class _LegacySweep(BaseModel):
    chunk_index: int = 1
    chunk_count: int = 1
    note: str = ""
    codes: list[_LegacyCode] = Field(default_factory=list)
    label_index: list[_LegacyCode] = Field(default_factory=list)
    hints: list = Field(default_factory=list)


def review_codes(n: int) -> list[dict]:
    """Distinct short labels, a 35-character definition, and two 24-character quotes.

    Pretty-printed with the 0.6.4 note, this is the fixture behind the
    published 9.2k / 54k / 179k / 644k table.
    """
    rows = []
    for index in range(n):
        label = f"c{index}"
        quote = "q" * 24
        rows.append(
            {
                "code": label,
                "key": label,
                "code_id": _code_id(code_key(label)),
                "definition": "d" * 35,
                "frequency": 1,
                "valence": "",
                "examples": [
                    {"quote": quote, "passage_id": f"p{index}"},
                    {"quote": quote, "passage_id": f"p{index}"},
                ],
            }
        )
    return rows


def _legacy_code(entry: dict) -> _LegacyCode:
    examples = [
        {"quote": str(item.get("quote") or ""), "passage_id": str(item.get("passage_id") or "")}
        for item in (entry.get("examples") or [])[:2]
        if item.get("quote")
    ]
    return _LegacyCode(
        code_id=str(entry["code_id"]),
        label=str(entry["code"]),
        definition=str(entry.get("definition") or ""),
        frequency=int(entry.get("frequency") or 0),
        example=examples[0]["quote"] if examples else "",
        examples=examples,
        valence=str(entry.get("valence") or ""),
    )


def legacy_user_chars(codes: list[dict], *, chunk_size: int = CHUNK_SIZE) -> int:
    """User-message characters of the 0.6.4 sweep (pretty JSON, full label index)."""
    size = max(1, int(chunk_size))
    chunks = [codes[start : start + size] for start in range(0, len(codes), size)] or [[]]
    total = 0
    for index, chunk in enumerate(chunks, start=1):
        request = _LegacySweep(
            chunk_index=index,
            chunk_count=len(chunks),
            note=LEGACY_NOTE,
            codes=[_legacy_code(entry) for entry in chunk],
            label_index=[_legacy_code(entry) for entry in codes] if len(chunks) > 1 else [],
        )
        total += len(request.model_dump_json(indent=2))
    return total


def _filled_cross_chunks(codes: list[dict], *, chunk_size: int, limit: int) -> list[list[dict]]:
    """Worst case of the cap: each chunk also carries ``limit`` full codes from outside it."""
    size = max(1, int(chunk_size))
    chunks = [codes[start : start + size] for start in range(0, len(codes), size)] or [[]]
    filled = []
    for chunk in chunks:
        inside = {id(entry) for entry in chunk}
        others = [entry for entry in codes if id(entry) not in inside]
        filled.append(others[: max(0, int(limit))])
    return filled


def payload_table(*, cross_chunk_limit: int = 8) -> list[dict]:
    """One row per codebook size.

    ``indexed`` repeats the 0.6.5 label index on every chunk. ``after`` is the
    0.6.7 request: bounded neighborhoods plus one label-only discovery pass.
    ``after_capped`` also fills ``cross_chunk_limit`` full codes on each chunk.
    ``published`` is set only for the sizes the 0.6.4 review published.
    """
    rows = []
    for n in SIZES:
        codes = review_codes(n)
        before = legacy_user_chars(codes)
        indexed = aggregate_sweep_chars(
            codes,
            context=CONTEXT,
            chunk_size=CHUNK_SIZE,
            audit="full",
            user_only=True,
            full_label_index=True,
            discover=False,
        )
        after = aggregate_sweep_chars(
            codes,
            context=CONTEXT,
            chunk_size=CHUNK_SIZE,
            audit="full",
            user_only=True,
        )
        after_capped = aggregate_sweep_chars(
            codes,
            context=CONTEXT,
            chunk_size=CHUNK_SIZE,
            cross_chunks=_filled_cross_chunks(codes, chunk_size=CHUNK_SIZE, limit=cross_chunk_limit),
            audit="full",
            user_only=True,
        )
        rows.append(
            {
                "codes": n,
                "published": PUBLISHED.get(n),
                "before": before,
                "indexed": indexed,
                "after": after,
                "after_capped": after_capped,
            }
        )
    return rows


def _doubling_ok(values: list[int], *, ceiling: float) -> bool:
    return all(nxt / prev < ceiling for prev, nxt in zip(values, values[1:]) if prev)


def _row(rows: list[dict], codes: int, key: str) -> int:
    for row in rows:
        if row["codes"] == codes:
            return int(row[key])
    raise KeyError(codes)


def main() -> None:
    rows = payload_table()
    print(f"{'codes':>6}  {'published':>10}  {'0.6.4':>10}  {'0.6.5':>10}  {'0.6.7':>10}  {'0.6.7+8':>10}")
    for row in rows:
        published = "" if row["published"] is None else f"{row['published']:10d}"
        print(
            f"{row['codes']:6d}  {published:>10}  {row['before']:10d}  "
            f"{row['indexed']:10d}  {row['after']:10d}  {row['after_capped']:10d}"
        )
    before = [row["before"] for row in rows]
    indexed = [row["indexed"] for row in rows]
    after = [row["after"] for row in rows]
    capped = [row["after_capped"] for row in rows]
    print(f"0.6.4 growth 1280/20: {before[-1] / before[0]:.1f}x")
    print(f"0.6.5 repeated index 640/160: {_row(rows, 640, 'indexed') / _row(rows, 160, 'indexed'):.2f}x")
    print(f"0.6.7 growth 1280/20: {after[-1] / after[0]:.1f}x")
    print(f"0.6.7 growth 640/160: {_row(rows, 640, 'after') / _row(rows, 160, 'after'):.2f}x")
    print(f"0.6.7 with 8 cross-chunk codes, 1280/20: {capped[-1] / capped[0]:.1f}x")
    for row in rows:
        if row["published"] is None:
            continue
        gap = abs(row["before"] - row["published"]) / row["published"]
        if gap > 0.03:
            raise SystemExit(
                f"{row['codes']} codes: 0.6.4 reconstruction {row['before']} is {gap:.1%} from {row['published']}"
            )
    # Adjacent doublings of a linear payload stay near 2x across 20..1280.
    # 20 codes is one chunk and has no discovery request. The step to 40 adds it.
    if after[1] / after[0] >= 3.5 or not _doubling_ok(after[1:], ceiling=2.5):
        raise SystemExit(f"0.6.7 payload is not near-linear across {SIZES}: {after}")
    span = SIZES[-1] / SIZES[0]
    if after[-1] / after[0] >= span * 2.5:
        raise SystemExit(f"0.6.7 payload grows faster than linear: {after}")
    # The same range shows the repeated index is not linear. 640/160 is 4x codes.
    if _row(rows, 640, "indexed") / _row(rows, 160, "indexed") < 8:
        raise SystemExit(f"repeated-index payload no longer shows the quadratic bend: {indexed}")
    if capped[1] / capped[0] >= 3.5 or not _doubling_ok(capped[1:], ceiling=2.5):
        raise SystemExit(f"capped neighborhood payload is not near-linear: {capped}")
    if before[3] / before[0] < 50:
        raise SystemExit(f"0.6.4 reconstruction is no longer quadratic at 160: {before}")


if __name__ == "__main__":
    main()
