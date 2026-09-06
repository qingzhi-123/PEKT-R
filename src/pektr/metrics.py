"""Recommendation metrics for PEKT-R."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from math import log2


def hit_rate_at_k(
    recommendations: Mapping[str, Sequence[str]],
    truth: Mapping[str, Set[str]],
    k: int = 10,
) -> float:
    if not truth:
        return 0.0
    hits = 0
    total = 0
    for user_id, relevant in truth.items():
        if not relevant:
            continue
        total += 1
        top_k = set(recommendations.get(user_id, [])[:k])
        hits += int(bool(top_k & set(relevant)))
    return hits / total if total else 0.0


def dcg_at_k(ranked_items: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    score = 0.0
    for idx, item in enumerate(ranked_items[:k], start=1):
        if item in relevant:
            score += 1.0 / log2(idx + 1)
    return score


def ndcg_at_k(
    recommendations: Mapping[str, Sequence[str]],
    truth: Mapping[str, Set[str]],
    k: int = 10,
) -> float:
    if not truth:
        return 0.0
    total = 0
    score = 0.0
    for user_id, relevant in truth.items():
        if not relevant:
            continue
        total += 1
        ranked = recommendations.get(user_id, [])
        ideal_hits = min(len(relevant), k)
        ideal = sum(1.0 / log2(idx + 1) for idx in range(1, ideal_hits + 1))
        if ideal > 0:
            score += dcg_at_k(ranked, relevant, k) / ideal
    return score / total if total else 0.0

