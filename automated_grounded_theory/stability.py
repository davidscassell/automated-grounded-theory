"""Compare open-coding runs started from different shuffle seeds.

Label overlap is descriptive. Open codes are not expected to match string
for string, and a high or low score is not a validity coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from automated_grounded_theory.evaluate import aligned_assignment_agreement, assignment_ari
from automated_grounded_theory.retrieval import jaccard
from automated_grounded_theory.validation import code_key


@dataclass
class StabilityReport:
    seeds: list[int]
    pairwise: list[dict]
    mean_jaccard: float
    mean_soft_jaccard: float
    stable_codes: list[str]
    codes_in_only_one_seed: list[str]
    per_seed_codes: dict[str, list[str]]
    note: str
    runs: list = field(default_factory=list)

    def to_dict(self, *, include_runs: bool = False) -> dict:
        payload = {
            "seeds": self.seeds,
            "pairwise": self.pairwise,
            "mean_jaccard": self.mean_jaccard,
            "mean_soft_jaccard": self.mean_soft_jaccard,
            "stable_codes": self.stable_codes,
            "codes_in_only_one_seed": self.codes_in_only_one_seed,
            "per_seed_codes": self.per_seed_codes,
            "note": self.note,
        }
        if include_runs:
            payload["runs"] = [run.to_dict() if hasattr(run, "to_dict") else run for run in self.runs]
        return payload


def _labels(result) -> list[str]:
    return [str(entry.get("code") or "") for entry in result.codebook if entry.get("code")]


def _definitions(result) -> dict[str, str]:
    found = {}
    for entry in result.codebook:
        label = str(entry.get("code") or "")
        if label:
            found[code_key(label)] = str(entry.get("definition") or "")
    return found


def _soft_jaccard(left_labels: Sequence[str], right_labels: Sequence[str], left_defs: dict, right_defs: dict) -> float:
    if not left_labels and not right_labels:
        return 1.0
    if not left_labels or not right_labels:
        return 0.0
    pairs = []
    for left in left_labels:
        for right in right_labels:
            label_score = jaccard(left, right)
            definition_score = jaccard(
                f"{left} {left_defs.get(code_key(left), '')}",
                f"{right} {right_defs.get(code_key(right), '')}",
            )
            pairs.append((max(label_score, definition_score), code_key(left), code_key(right)))
    pairs.sort(reverse=True)
    used_left: set[str] = set()
    used_right: set[str] = set()
    matches = 0
    for score, left_key, right_key in pairs:
        if score < 0.5 or left_key in used_left or right_key in used_right:
            continue
        used_left.add(left_key)
        used_right.add(right_key)
        matches += 1
    return (2 * matches) / (len(left_labels) + len(right_labels))


def compare_runs(runs: Sequence, *, soft_threshold_note: str = "0.5") -> StabilityReport:
    """Pairwise label stability across completed open-coding runs."""
    if len(runs) < 2:
        raise ValueError("compare_runs needs at least two runs.")
    label_sets = []
    definitions = []
    per_seed: dict[str, list[str]] = {}
    seeds: list[int] = []
    for index, run in enumerate(runs):
        labels = _labels(run)
        label_sets.append({code_key(label): label for label in labels})
        definitions.append(_definitions(run))
        seed = run.metadata.get("seed", index) if getattr(run, "metadata", None) else index
        seeds.append(seed)
        per_seed[str(seed)] = sorted(labels, key=str.casefold)

    pairwise = []
    jaccards = []
    soft = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            left = set(label_sets[i])
            right = set(label_sets[j])
            union = left | right
            exact = (len(left & right) / len(union)) if union else 1.0
            soft_score = _soft_jaccard(
                list(label_sets[i].values()),
                list(label_sets[j].values()),
                definitions[i],
                definitions[j],
            )
            jaccards.append(exact)
            soft.append(soft_score)
            agreement = aligned_assignment_agreement(runs[i], runs[j])
            pairwise.append(
                {
                    "seed_a": seeds[i],
                    "seed_b": seeds[j],
                    "jaccard": exact,
                    "soft_jaccard": soft_score,
                    "soft_match_threshold": soft_threshold_note,
                    "shared_codes": sorted(label_sets[i][key] for key in sorted(left & right)),
                    "adjusted_rand_index": round(assignment_ari(runs[i], runs[j]), 4),
                    "aligned_assignment_jaccard": agreement["mean_jaccard"],
                    "aligned_assignment_alpha": agreement["krippendorff_alpha"],
                }
            )

    intersection = set(label_sets[0])
    for labels in label_sets[1:]:
        intersection &= set(labels)
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    for labels in label_sets:
        for key, label in labels.items():
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, label)
    only_one = sorted((display[key] for key, count in counts.items() if count == 1), key=str.casefold)
    stable = sorted((label_sets[0][key] for key in intersection), key=str.casefold)
    return StabilityReport(
        seeds=seeds,
        pairwise=pairwise,
        mean_jaccard=(sum(jaccards) / len(jaccards)) if jaccards else 1.0,
        mean_soft_jaccard=(sum(soft) / len(soft)) if soft else 1.0,
        stable_codes=stable,
        codes_in_only_one_seed=only_one,
        per_seed_codes=per_seed,
        note=(
            "Exact Jaccard uses case-folded code labels. Soft Jaccard greedily matches "
            "labels, or label-plus-definition text, at token overlap 0.5. "
            "adjusted_rand_index compares passage partitions, not code labels: "
            "different singleton codes on every passage score 1.0. "
            "aligned_assignment_jaccard is the mean Jaccard of the code set on each "
            "passage after labels are aligned by exact text or spelling variant "
            "(a negation does not align), and aligned_assignment_alpha is "
            "Krippendorff's alpha on that passage-by-code matrix. "
            "None of these scores decides which seed is correct, and none is a validity coefficient. "
            "The report does not pick a winning seed. Open-coding labels are expected to move."
        ),
        runs=list(runs),
    )
