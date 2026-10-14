"""Deterministic grouping for synthesis.

Average-linkage agglomerative clustering on cosine distance. No random seed:
ties break toward the lower index. :func:`union_roots` keeps the
lexicographically smaller key. Sweep merge survivors are chosen separately,
from the model's target, in the codebook module.
"""

from __future__ import annotations

from typing import Sequence

from automated_grounded_theory.retrieval import cosine


def average_linkage(vectors: Sequence[Sequence[float]], *, max_distance: float) -> list[list[int]]:
    """Cluster row indexes. Merge while the average pairwise distance is within the cap.

    ``max_distance`` is ``1 - cosine_threshold`` when vectors are compared by cosine.
    Every index appears in exactly one cluster. Singletons stay singletons.
    """
    count = len(vectors)
    if count == 0:
        return []
    clusters: list[list[int]] = [[index] for index in range(count)]
    if max_distance < 0:
        return clusters

    def distance(left: int, right: int) -> float:
        return 1.0 - cosine(vectors[left], vectors[right])

    while len(clusters) > 1:
        best: tuple[float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                total = 0.0
                pairs = 0
                for left in clusters[i]:
                    for right in clusters[j]:
                        total += distance(left, right)
                        pairs += 1
                score = total / pairs
                candidate = (score, i, j)
                if best is None or candidate < best:
                    best = candidate
        if best is None or best[0] > max_distance:
            break
        _, left_at, right_at = best
        merged = clusters[left_at] + clusters[right_at]
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in {left_at, right_at}]
        clusters.append(merged)
    for cluster in clusters:
        cluster.sort()
    clusters.sort(key=lambda cluster: cluster[0])
    return clusters


def cluster_pairs(
    labels: Sequence[str],
    vectors: Sequence[Sequence[float]] | None,
    *,
    cosine_threshold: float,
) -> list[tuple[int, int, float]]:
    """Pairs that share an agglomerative cluster, highest cosine first."""
    if vectors is None or len(labels) < 2:
        return []
    groups = average_linkage(vectors, max_distance=max(0.0, 1.0 - cosine_threshold))
    found: list[tuple[int, int, float]] = []
    for group in groups:
        if len(group) < 2:
            continue
        for offset, left in enumerate(group):
            for right in group[offset + 1 :]:
                score = cosine(vectors[left], vectors[right])
                found.append((left, right, score))
    found.sort(key=lambda item: (-item[2], labels[item[0]].casefold(), labels[item[1]].casefold()))
    return found


def union_roots(pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    """Map every id in ``pairs`` to the lexicographically smallest member of its set."""
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for left, right in pairs:
        if not left or not right or left == right:
            continue
        parent.setdefault(left, left)
        parent.setdefault(right, right)
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        if right_root < left_root:
            parent[left_root] = right_root
        else:
            parent[right_root] = left_root
    return {key: find(key) for key in list(parent)}


def adjusted_rand_index(left: Sequence[str], right: Sequence[str]) -> float:
    """Adjusted Rand index of two labelings of the same items, in index order.

    1 is identical partitions. 0 is the chance level. Items are paired by position.
    """
    if len(left) != len(right):
        raise ValueError("adjusted_rand_index requires two labelings of the same length.")
    if not left:
        return 1.0
    counts: dict[tuple[str, str], int] = {}
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for a, b in zip(left, right):
        counts[(a, b)] = counts.get((a, b), 0) + 1
        left_counts[a] = left_counts.get(a, 0) + 1
        right_counts[b] = right_counts.get(b, 0) + 1

    def comb(n: int) -> int:
        return n * (n - 1) // 2

    contingency = sum(comb(value) for value in counts.values())
    left_sum = sum(comb(value) for value in left_counts.values())
    right_sum = sum(comb(value) for value in right_counts.values())
    total = comb(len(left))
    if total == 0:
        return 1.0
    expected = (left_sum * right_sum) / total
    maximum = (left_sum + right_sum) / 2
    if maximum == expected:
        return 1.0
    return (contingency - expected) / (maximum - expected)
