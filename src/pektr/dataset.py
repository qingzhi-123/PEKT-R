"""Dataset and collation utilities for PEKT-R."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_sequences(artifact_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(artifact_dir) / "sequences.jsonl"
    sequences = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sequences.append(json.loads(line))
    return sequences


def load_problem_features(artifact_dir: str | Path) -> list[dict[str, Any]]:
    return load_json(Path(artifact_dir) / "problem_features.json")["problems"]


def load_split_users(artifact_dir: str | Path, split: str | None) -> set[str] | None:
    if split is None:
        return None
    splits = load_json(Path(artifact_dir) / "splits.json")
    return set(str(user_id) for user_id in splits[split])


class SequenceWindowDataset(Dataset):
    """Fixed-length windows over per-user learning sequences."""

    def __init__(
        self,
        artifact_dir: str | Path,
        split: str | None = "train",
        max_seq_len: int = 100,
        window_stride: int | None = None,
        min_seq_len: int = 2,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.max_seq_len = max_seq_len
        self.window_stride = window_stride or max(1, max_seq_len - 1)
        allowed_users = load_split_users(self.artifact_dir, split)
        self.sequences = [
            seq
            for seq in load_sequences(self.artifact_dir)
            if allowed_users is None or str(seq["user_id"]) in allowed_users
        ]
        self.windows: list[tuple[int, int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            n_items = len(seq["problem_ids"])
            if n_items < min_seq_len:
                continue
            start = 0
            while start < n_items - 1:
                end = min(n_items, start + max_seq_len)
                if end - start >= min_seq_len:
                    self.windows.append((seq_idx, start, end))
                if end == n_items:
                    break
                start += self.window_stride

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        seq_idx, start, end = self.windows[index]
        seq = self.sequences[seq_idx]
        return {
            "user_id": seq["user_id"],
            "problem_ids": seq["problem_ids"][start:end],
            "responses": seq["responses"][start:end],
            "judge_status_ids": seq["judge_status_ids"][start:end],
            "knowledge": seq["knowledge"][start:end],
            "error_vectors": seq["error_vectors"][start:end],
        }


def collate_pektr(
    batch: list[dict[str, Any]],
    n_knowledge: int,
    error_dim: int,
) -> dict[str, Any]:
    max_len = max(len(item["problem_ids"]) for item in batch)
    batch_size = len(batch)
    problem_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    responses = torch.zeros((batch_size, max_len), dtype=torch.float32)
    judge_status_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    knowledge = torch.zeros((batch_size, max_len, n_knowledge), dtype=torch.float32)
    error_vectors = torch.zeros((batch_size, max_len, error_dim), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for row_idx, item in enumerate(batch):
        length = len(item["problem_ids"])
        problem_ids[row_idx, :length] = torch.tensor(item["problem_ids"], dtype=torch.long)
        responses[row_idx, :length] = torch.tensor(item["responses"], dtype=torch.float32)
        judge_status_ids[row_idx, :length] = torch.tensor(item["judge_status_ids"], dtype=torch.long)
        attention_mask[row_idx, :length] = True

        for time_idx, tags in enumerate(item["knowledge"]):
            for tag in tags:
                tag_idx = int(tag)
                if 0 <= tag_idx < n_knowledge:
                    knowledge[row_idx, time_idx, tag_idx] = 1.0

        for time_idx, vector in enumerate(item["error_vectors"]):
            trimmed = [float(value) for value in vector[:error_dim]]
            if trimmed:
                error_vectors[row_idx, time_idx, : len(trimmed)] = torch.tensor(trimmed, dtype=torch.float32)

    return {
        "user_ids": [str(item["user_id"]) for item in batch],
        "problem_ids": problem_ids,
        "responses": responses,
        "judge_status_ids": judge_status_ids,
        "knowledge": knowledge,
        "error_vectors": error_vectors,
        "attention_mask": attention_mask,
    }


def build_inference_batch(
    sequence: dict[str, Any],
    n_knowledge: int,
    error_dim: int,
    max_seq_len: int,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    length = len(sequence["problem_ids"])
    start = max(0, length - max_seq_len)
    item = {
        "user_id": sequence["user_id"],
        "problem_ids": sequence["problem_ids"][start:],
        "responses": sequence["responses"][start:],
        "judge_status_ids": sequence["judge_status_ids"][start:],
        "knowledge": sequence["knowledge"][start:],
        "error_vectors": sequence["error_vectors"][start:],
    }
    batch = collate_pektr([item], n_knowledge=n_knowledge, error_dim=error_dim)
    if device is not None:
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)
    return batch


def find_sequence_by_user(artifact_dir: str | Path, user_id: str) -> dict[str, Any] | None:
    user_id = str(user_id)
    for sequence in load_sequences(artifact_dir):
        if str(sequence["user_id"]) == user_id:
            return sequence
    return None

