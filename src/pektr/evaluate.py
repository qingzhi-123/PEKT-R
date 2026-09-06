"""Evaluate PEKT-R recommendation quality."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from .config import PEKTRConfig
from .dataset import build_inference_batch, load_json, load_problem_features, load_sequences
from .metrics import hit_rate_at_k, ndcg_at_k
from .model import PEKTRModel
from .scoring import recommend_from_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PEKT-R with leave-one-out ranking.")
    parser.add_argument("--artifact-dir", default="artifacts/bepkt")
    parser.add_argument("--checkpoint", default="checkpoints/pektr.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.30)
    parser.add_argument("--rho", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def trim_sequence(sequence: dict[str, Any], end: int) -> dict[str, Any]:
    return {
        "user_id": sequence["user_id"],
        "problem_ids": sequence["problem_ids"][:end],
        "responses": sequence["responses"][:end],
        "judge_status_ids": sequence["judge_status_ids"][:end],
        "knowledge": sequence["knowledge"][:end],
        "error_vectors": sequence["error_vectors"][:end],
    }


def evaluate_model(
    model: PEKTRModel,
    config: PEKTRConfig,
    artifact_dir: str | Path,
    split: str = "test",
    k: int = 10,
    num_negatives: int = 100,
    alpha: float = 0.45,
    beta: float = 0.25,
    gamma: float = 0.30,
    rho: float = 0.85,
    seed: int = 42,
    device: torch.device | str | None = None,
    max_users: int | None = None,
) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    artifact_dir = Path(artifact_dir)
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device)
    model.eval()

    split_users = set(str(x) for x in load_json(artifact_dir / "splits.json")[split])
    sequences = [seq for seq in load_sequences(artifact_dir) if str(seq["user_id"]) in split_users]
    sequences.sort(key=lambda item: str(item["user_id"]))
    if max_users is not None and max_users > 0:
        sequences = sequences[:max_users]
    problem_features = load_problem_features(artifact_dir)
    by_problem_index = {int(item["problem_index"]): item for item in problem_features}
    all_problem_indices = set(by_problem_index)

    recommendations: dict[str, list[str]] = {}
    truth: dict[str, set[str]] = {}
    evaluated = 0
    skipped = 0

    for sequence in sequences:
        if len(sequence["problem_ids"]) < 2:
            skipped += 1
            continue
        target_problem = int(sequence["problem_ids"][-1])
        if target_problem not in by_problem_index:
            skipped += 1
            continue
        history = trim_sequence(sequence, end=len(sequence["problem_ids"]) - 1)
        seen = set(history["problem_ids"])
        negatives = list(all_problem_indices - seen - {target_problem})
        if len(negatives) > num_negatives:
            negatives = rng.sample(negatives, num_negatives)
        candidates = [by_problem_index[idx] for idx in negatives + [target_problem]]

        batch = build_inference_batch(
            history,
            n_knowledge=config.n_knowledge,
            error_dim=config.error_dim,
            max_seq_len=config.max_seq_len,
            device=device,
        )
        with torch.no_grad():
            outputs = model(
                problem_ids=batch["problem_ids"],
                responses=batch["responses"],
                judge_status_ids=batch["judge_status_ids"],
                knowledge=batch["knowledge"],
                error_vectors=batch["error_vectors"],
                attention_mask=batch["attention_mask"],
            )
            state = model.current_knowledge_state(outputs, batch["attention_mask"])[0]
            valid_len = int(batch["attention_mask"][0].sum().item())
            weakness = PEKTRModel.temporal_error_weakness(
                batch["error_vectors"][0, :valid_len, :],
                rho=rho,
            )

        ranked = recommend_from_state(
            knowledge_state=state.detach().cpu().tolist(),
            error_weakness=weakness.detach().cpu().tolist(),
            candidates=candidates,
            top_n=k,
            exclude_problem_indices=None,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        user_id = str(sequence["user_id"])
        recommendations[user_id] = [str(item.problem_index) for item in ranked]
        truth[user_id] = {str(target_problem)}
        evaluated += 1

    return {
        "split": split,
        "k": k,
        "num_negatives": num_negatives,
        "evaluated_users": evaluated,
        "skipped_users": skipped,
        f"HR@{k}": hit_rate_at_k(recommendations, truth, k=k),
        f"NDCG@{k}": ndcg_at_k(recommendations, truth, k=k),
    }


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = PEKTRConfig.from_dict(checkpoint["config"])
    device = resolve_device(args.device)
    model = PEKTRModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    metrics = evaluate_model(
        model=model,
        config=config,
        artifact_dir=artifact_dir,
        split=args.split,
        k=args.k,
        num_negatives=args.num_negatives,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        rho=args.rho,
        seed=args.seed,
        device=device,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
