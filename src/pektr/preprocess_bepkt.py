"""Preprocess BePKT into PEKT-R sequence artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .features import (
    DEFAULT_ACCEPTED_STATUSES,
    ERROR_LABELS,
    build_error_vector,
    set_large_csv_field_limit,
    tag_based_error_vector,
)


def _sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        yield from csv.DictReader(f)


def read_problem_metadata(data_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], list[str]]:
    raw_dir = data_dir / "raw_data"
    if not raw_dir.exists():
        raw_dir = data_dir / "used_in_pdkt"

    tag_name_by_id: dict[str, str] = {}
    for row in _read_csv(raw_dir / "problem_tag.csv"):
        tag_name_by_id[row["id"]] = row["name"]

    sorted_tag_ids = sorted(tag_name_by_id, key=_sort_key)
    tag_id_to_index = {tag_id: idx for idx, tag_id in enumerate(sorted_tag_ids)}
    tag_names = [tag_name_by_id[tag_id] for tag_id in sorted_tag_ids]

    problem_tags: dict[str, list[str]] = defaultdict(list)
    for row in _read_csv(raw_dir / "problem_tags.csv"):
        problem_id = row["problem_id"]
        tag_id = row["problemtag_id"]
        if tag_id in tag_id_to_index:
            problem_tags[problem_id].append(tag_id)

    problems: dict[str, dict[str, Any]] = {}
    for row in _read_csv(raw_dir / "problem.csv"):
        problem_id = row["id"]
        source_tag_ids = problem_tags.get(problem_id, [])
        problems[problem_id] = {
            "id": problem_id,
            "title": row.get("title", ""),
            "source_tag_ids": source_tag_ids,
            "tag_indices": [tag_id_to_index[tag_id] for tag_id in source_tag_ids],
            "tag_names": [tag_name_by_id[tag_id] for tag_id in source_tag_ids],
        }
    return problems, tag_id_to_index, tag_names


def scan_submission_stats(
    submission_path: Path,
    accepted_statuses: set[str],
) -> tuple[Counter[str], dict[str, list[int]], int]:
    status_counts: Counter[str] = Counter()
    acceptance: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    row_count = 0
    for row in _read_csv(submission_path):
        problem_id = row.get("problem_id", "")
        status = str(row.get("result", ""))
        if not problem_id or status == "":
            continue
        row_count += 1
        status_counts[status] += 1
        acceptance[problem_id][1] += 1
        if status in accepted_statuses:
            acceptance[problem_id][0] += 1
    return status_counts, acceptance, row_count


def preprocess_bepkt(
    data_dir: str | Path,
    out_dir: str | Path,
    accepted_statuses: set[str] | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    min_sequence_len: int = 3,
) -> dict[str, Any]:
    set_large_csv_field_limit()
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    raw_dir = data_dir / "raw_data"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Expected BePKT raw_data directory: {raw_dir}")
    submission_path = raw_dir / "submission.csv"
    if not submission_path.exists():
        raise FileNotFoundError(f"Expected submission file: {submission_path}")

    accepted_statuses = accepted_statuses or set(DEFAULT_ACCEPTED_STATUSES)
    problems, tag_id_to_index, tag_names = read_problem_metadata(data_dir)
    problem_ids = sorted(problems, key=_sort_key)
    problem_id_to_index = {problem_id: idx + 1 for idx, problem_id in enumerate(problem_ids)}

    status_counts, acceptance, row_count = scan_submission_stats(submission_path, accepted_statuses)
    judge_statuses = sorted(status_counts, key=_sort_key)
    judge_status_to_index = {status: idx + 1 for idx, status in enumerate(judge_statuses)}

    sequences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    problem_error_sum: dict[int, list[float]] = defaultdict(lambda: [0.0] * len(ERROR_LABELS))
    problem_error_count: Counter[int] = Counter()
    skipped = 0

    for order, row in enumerate(_read_csv(submission_path)):
        user_id = row.get("user_id", "")
        problem_id = row.get("problem_id", "")
        status = str(row.get("result", ""))
        timestamp = row.get("create_time", "")
        if not user_id or not problem_id or not timestamp or problem_id not in problem_id_to_index:
            skipped += 1
            continue

        problem = problems[problem_id]
        error_vector = build_error_vector(
            judge_status=status,
            info=row.get("info"),
            code=row.get("code"),
            language=row.get("language"),
            tag_names=problem["tag_names"],
            accepted_statuses=accepted_statuses,
        )
        response = 1 if status in accepted_statuses else 0
        problem_index = problem_id_to_index[problem_id]
        if response == 0 and any(error_vector):
            sums = problem_error_sum[problem_index]
            for idx, value in enumerate(error_vector):
                sums[idx] += float(value)
            problem_error_count[problem_index] += 1

        sequences[user_id].append(
            {
                "timestamp": timestamp,
                "order": order,
                "problem_id": problem_index,
                "response": response,
                "judge_status_id": judge_status_to_index[status],
                "knowledge": problem["tag_indices"],
                "error_vector": error_vector,
            }
        )

    for events in sequences.values():
        events.sort(key=lambda item: (item["timestamp"], item["order"]))

    eligible_users = [user_id for user_id, events in sequences.items() if len(events) >= min_sequence_len]
    rng = random.Random(seed)
    rng.shuffle(eligible_users)
    n_train = int(len(eligible_users) * train_ratio)
    n_val = int(len(eligible_users) * val_ratio)
    splits = {
        "train": eligible_users[:n_train],
        "val": eligible_users[n_train : n_train + n_val],
        "test": eligible_users[n_train + n_val :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sequences.jsonl").open("w", encoding="utf-8") as f:
        for user_id in sorted(sequences, key=_sort_key):
            events = sequences[user_id]
            payload = {
                "user_id": user_id,
                "problem_ids": [event["problem_id"] for event in events],
                "responses": [event["response"] for event in events],
                "judge_status_ids": [event["judge_status_id"] for event in events],
                "knowledge": [event["knowledge"] for event in events],
                "error_vectors": [event["error_vector"] for event in events],
                "timestamps": [event["timestamp"] for event in events],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    problem_features = []
    for problem_id in problem_ids:
        problem = problems[problem_id]
        problem_index = problem_id_to_index[problem_id]
        accepted, total = acceptance.get(problem_id, [0, 0])
        difficulty = 0.5 if total == 0 else 1.0 - (accepted / total)
        if problem_error_count[problem_index] > 0:
            count = problem_error_count[problem_index]
            error_cover = [value / count for value in problem_error_sum[problem_index]]
        else:
            error_cover = tag_based_error_vector(problem["tag_names"], failed=True)
        problem_features.append(
            {
                "problem_index": problem_index,
                "problem_id": problem_id,
                "title": problem["title"],
                "tags": problem["tag_indices"],
                "tag_names": problem["tag_names"],
                "difficulty": max(0.0, min(1.0, difficulty)),
                "error_cover": error_cover,
                "submission_count": total,
                "accepted_count": accepted,
            }
        )

    vocab = {
        "problem_id_to_index": problem_id_to_index,
        "index_to_problem_id": [None] + problem_ids,
        "tag_id_to_index": tag_id_to_index,
        "tag_index_to_name": tag_names,
        "judge_status_to_index": judge_status_to_index,
        "index_to_judge_status": [None] + judge_statuses,
        "error_labels": list(ERROR_LABELS),
        "accepted_statuses": sorted(accepted_statuses, key=_sort_key),
    }

    stats = {
        "rows": row_count,
        "skipped_rows": skipped,
        "users": len(sequences),
        "eligible_users": len(eligible_users),
        "problems": len(problem_ids),
        "knowledge_concepts": len(tag_names),
        "judge_status_counts": dict(status_counts),
        "splits": {name: len(users) for name, users in splits.items()},
    }

    for file_name, payload in (
        ("vocab.json", vocab),
        ("problem_features.json", {"problems": problem_features}),
        ("splits.json", splits),
        ("preprocess_stats.json", stats),
    ):
        with (out_dir / file_name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BePKT for PEKT-R.")
    parser.add_argument("--data-dir", default="data/BePKT")
    parser.add_argument("--out-dir", default="artifacts/bepkt")
    parser.add_argument("--accepted-status", action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-sequence-len", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted = set(args.accepted_status or DEFAULT_ACCEPTED_STATUSES)
    stats = preprocess_bepkt(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        accepted_statuses=accepted,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_sequence_len=args.min_sequence_len,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

