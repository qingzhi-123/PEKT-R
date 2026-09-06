"""Feature builders for PEKT-R error representation."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections.abc import Iterable, Sequence


ERROR_LABELS: tuple[str, ...] = (
    "syntax_structure_error",
    "data_type_error",
    "uninitialized_variable",
    "array_string_out_of_bounds",
    "condition_judgement_error",
    "loop_boundary_error",
    "recursion_termination_error",
    "algorithm_complexity_too_high",
    "dynamic_programming_state_error",
)

ERROR_INDEX = {name: idx for idx, name in enumerate(ERROR_LABELS)}

DEFAULT_ACCEPTED_STATUSES = frozenset({"0"})

# Judge status names follow common online-judge conventions used by BePKT-like
# systems. The model keeps the raw judge id too, so this weak labeling can be
# replaced by supervised error labels without changing model code.
JUDGE_STATUS_HINTS: dict[str, tuple[str, ...]] = {
    "-2": ("syntax_structure_error",),
    "-1": ("condition_judgement_error",),
    "1": ("algorithm_complexity_too_high",),
    "2": ("algorithm_complexity_too_high",),
    "3": ("algorithm_complexity_too_high",),
    "4": ("array_string_out_of_bounds",),
    "8": ("condition_judgement_error",),
}

TAG_HINTS: dict[str, tuple[str, ...]] = {
    "data_type_error": (
        "type",
        "\u6570\u636e\u7c7b\u578b",
        "\u57fa\u672c\u6570\u636e\u7c7b\u578b",
        "\u57fa\u672c\u7c7b\u578b",
    ),
    "array_string_out_of_bounds": (
        "array",
        "string",
        "\u6570\u7ec4",
        "\u5b57\u7b26\u4e32",
        "\u6307\u9488",
        "\u94fe\u8868",
        "\u961f\u5217",
        "\u6808",
    ),
    "condition_judgement_error": (
        "if",
        "\u6761\u4ef6",
        "\u5224\u65ad",
        "\u5206\u652f",
        "\u6a21\u62df",
    ),
    "loop_boundary_error": (
        "for",
        "while",
        "\u5faa\u73af",
        "\u679a\u4e3e",
    ),
    "recursion_termination_error": (
        "dfs",
        "bfs",
        "\u9012\u5f52",
        "\u56de\u6eaf",
    ),
    "algorithm_complexity_too_high": (
        "sort",
        "\u6392\u5e8f",
        "\u8d2a\u5fc3",
        "\u641c\u7d22",
        "\u56fe",
        "\u6811",
        "\u7b97\u6cd5",
    ),
    "dynamic_programming_state_error": (
        "dp",
        "\u52a8\u6001\u89c4\u5212",
        "\u80cc\u5305",
        "\u9012\u63a8",
    ),
}

CODE_HINTS: dict[str, tuple[re.Pattern[str], ...]] = {
    "array_string_out_of_bounds": (
        re.compile(r"\[[^\]]+\]"),
        re.compile(r"\b(strlen|strcpy|gets|scanf)\b"),
    ),
    "loop_boundary_error": (
        re.compile(r"\b(for|while)\s*\("),
    ),
    "recursion_termination_error": (
        re.compile(r"\b(dfs|backtrack|solve)\s*\("),
    ),
    "dynamic_programming_state_error": (
        re.compile(r"\bdp\s*\["),
        re.compile(r"\bmem(?:o|set)?\s*\["),
    ),
}


def set_large_csv_field_limit() -> None:
    """Raise csv field limit enough for source-code columns on Windows too."""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def is_accepted_status(
    judge_status: str | int | None,
    accepted_statuses: Iterable[str] = DEFAULT_ACCEPTED_STATUSES,
) -> bool:
    return str(judge_status) in {str(x) for x in accepted_statuses}


def empty_error_vector() -> list[float]:
    return [0.0] * len(ERROR_LABELS)


def _mark(vector: list[float], labels: Iterable[str]) -> None:
    for label in labels:
        idx = ERROR_INDEX.get(label)
        if idx is not None:
            vector[idx] = 1.0


def _lower_blob(values: Iterable[str | None]) -> str:
    return " ".join(v for v in values if v).lower()


def _safe_json_text(raw_info: str | None) -> str:
    if not raw_info:
        return ""
    text = str(raw_info)
    if len(text) > 20000:
        text = text[:20000]
    try:
        parsed = json.loads(text)
    except Exception:
        return text.lower()
    return json.dumps(parsed, ensure_ascii=False).lower()


def tag_based_error_vector(tag_names: Sequence[str], failed: bool = True) -> list[float]:
    vector = empty_error_vector()
    if not failed:
        return vector
    tag_blob = _lower_blob(tag_names)
    for label, hints in TAG_HINTS.items():
        if any(hint.lower() in tag_blob for hint in hints):
            _mark(vector, (label,))
    return vector


def build_error_vector(
    *,
    judge_status: str | int | None,
    info: str | None,
    code: str | None,
    language: str | None,
    tag_names: Sequence[str],
    accepted_statuses: Iterable[str] = DEFAULT_ACCEPTED_STATUSES,
) -> list[float]:
    """Build PEKT-R multi-label error vector for one submission.

    BePKT has judge status and raw code but no human fine-grained error labels.
    This function builds weak labels aligned to the error taxonomy in the paper.
    Replace its output with supervised CodeT5+/PLCodeBERT predictions if those
    labels are available.
    """

    status = "" if judge_status is None else str(judge_status)
    failed = not is_accepted_status(status, accepted_statuses)
    vector = empty_error_vector()
    if not failed:
        return vector

    _mark(vector, JUDGE_STATUS_HINTS.get(status, ()))

    info_blob = _safe_json_text(info)
    if any(s in info_blob for s in ("compile", "syntax", "expected", "missing", "undeclared")):
        _mark(vector, ("syntax_structure_error",))
    if any(s in info_blob for s in ("type", "incompatible", "conversion")):
        _mark(vector, ("data_type_error",))
    if any(s in info_blob for s in ("uninitialized", "uninitialised", "may be used")):
        _mark(vector, ("uninitialized_variable",))
    if any(s in info_blob for s in ("segmentation", "signal\": 11", "out of bounds", "index")):
        _mark(vector, ("array_string_out_of_bounds",))
    if any(s in info_blob for s in ("time", "tle", "cpu_time", "real_time")) and status in {"1", "2"}:
        _mark(vector, ("algorithm_complexity_too_high",))

    code_text = code or ""
    for label, patterns in CODE_HINTS.items():
        if any(pattern.search(code_text) for pattern in patterns):
            if label == "algorithm_complexity_too_high" or status not in {"1", "2", "3"}:
                _mark(vector, (label,))

    _mark(vector, _labels_from_problem_tags(tag_names))

    if not any(vector):
        _mark(vector, ("condition_judgement_error",))
    return vector


def _labels_from_problem_tags(tag_names: Sequence[str]) -> list[str]:
    tag_blob = _lower_blob(tag_names)
    labels: list[str] = []
    for label, hints in TAG_HINTS.items():
        if any(hint.lower() in tag_blob for hint in hints):
            labels.append(label)
    return labels


def temporal_decay_error_weakness(
    error_vectors: Sequence[Sequence[float]],
    rho: float = 0.85,
) -> list[float]:
    """Compute C_i,t = sum_tau rho^(t-tau) c_tau."""

    if not error_vectors:
        return empty_error_vector()
    dim = len(error_vectors[0])
    weakness = [0.0] * dim
    n = len(error_vectors)
    for pos, vector in enumerate(error_vectors):
        weight = rho ** (n - pos - 1)
        for idx, value in enumerate(vector):
            weakness[idx] += float(value) * weight
    return weakness


def normalize_vector(values: Sequence[float]) -> list[float]:
    total = sum(float(v) for v in values)
    if total <= 0:
        return [0.0 for _ in values]
    return [float(v) / total for v in values]

