"""Pure-Python PEKT-R recommendation scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterable, Sequence


@dataclass
class CandidateScore:
    problem_index: int
    problem_id: str
    title: str
    score: float
    knowledge_match: float
    difficulty_match: float
    error_correction: float
    difficulty: float
    tags: list[int]
    tag_names: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _scaled_error_weakness(values: Sequence[float]) -> list[float]:
    total = sum(abs(float(v)) for v in values)
    if total <= 0:
        return [0.0 for _ in values]
    return [float(v) / total for v in values]


def score_one_candidate(
    knowledge_state: Sequence[float],
    error_weakness: Sequence[float],
    candidate: dict[str, object],
    alpha: float = 0.45,
    beta: float = 0.25,
    gamma: float = 0.30,
    normalize_knowledge: bool = True,
    normalize_error: bool = True,
) -> CandidateScore:
    tags = [int(tag) for tag in candidate.get("tags", [])]
    difficulty = float(candidate.get("difficulty", 0.5))
    error_cover = [float(v) for v in candidate.get("error_cover", [])]

    if tags:
        gaps = [1.0 - float(knowledge_state[tag]) for tag in tags if 0 <= tag < len(knowledge_state)]
        knowledge_match = _mean(gaps) if normalize_knowledge else sum(gaps)
        ability = _mean([float(knowledge_state[tag]) for tag in tags if 0 <= tag < len(knowledge_state)])
    else:
        knowledge_match = 1.0 - _mean([float(v) for v in knowledge_state])
        ability = _mean([float(v) for v in knowledge_state])

    difficulty_match = max(0.0, min(1.0, 1.0 - abs(ability - difficulty)))
    weakness = _scaled_error_weakness(error_weakness) if normalize_error else [float(v) for v in error_weakness]
    error_correction = sum(a * b for a, b in zip(weakness, error_cover))

    score = alpha * knowledge_match + beta * difficulty_match + gamma * error_correction
    return CandidateScore(
        problem_index=int(candidate["problem_index"]),
        problem_id=str(candidate["problem_id"]),
        title=str(candidate.get("title", "")),
        score=float(score),
        knowledge_match=float(knowledge_match),
        difficulty_match=float(difficulty_match),
        error_correction=float(error_correction),
        difficulty=float(difficulty),
        tags=tags,
        tag_names=[str(x) for x in candidate.get("tag_names", [])],
    )


def recommend_from_state(
    knowledge_state: Sequence[float],
    error_weakness: Sequence[float],
    candidates: Iterable[dict[str, object]],
    top_n: int = 10,
    exclude_problem_indices: Iterable[int] | None = None,
    alpha: float = 0.45,
    beta: float = 0.25,
    gamma: float = 0.30,
) -> list[CandidateScore]:
    excluded = set(exclude_problem_indices or [])
    scored = [
        score_one_candidate(
            knowledge_state,
            error_weakness,
            candidate,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        for candidate in candidates
        if int(candidate["problem_index"]) not in excluded
    ]
    scored.sort(key=lambda item: (item.score, item.knowledge_match, item.error_correction), reverse=True)
    return scored[:top_n]
