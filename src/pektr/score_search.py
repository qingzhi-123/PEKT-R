"""Search PEKT-R recommendation-score weights for a trained checkpoint."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any

import torch

from .config import PEKTRConfig
from .dataset import build_inference_batch, load_json, load_problem_features, load_sequences
from .evaluate import trim_sequence
from .metrics import hit_rate_at_k, ndcg_at_k
from .model import PEKTRModel
from .scoring import recommend_from_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search alpha/beta/gamma/rho for PEKT-R ranking.")
    parser.add_argument("--artifact-dir", default="artifacts/bepkt")
    parser.add_argument("--checkpoint", default="checkpoints/pektr.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def weight_grid() -> list[tuple[float, float, float]]:
    values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    grid = []
    for alpha, beta in itertools.product(values, values):
        gamma = round(1.0 - alpha - beta, 2)
        if gamma < 0.05 or gamma > 0.80:
            continue
        grid.append((alpha, beta, gamma))
    return grid


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[PEKTRModel, PEKTRConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = PEKTRConfig.from_dict(checkpoint["config"])
    model = PEKTRModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def build_eval_cache(
    model: PEKTRModel,
    config: PEKTRConfig,
    artifact_dir: Path,
    split: str,
    num_negatives: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    split_users = set(str(x) for x in load_json(artifact_dir / "splits.json")[split])
    sequences = [seq for seq in load_sequences(artifact_dir) if str(seq["user_id"]) in split_users]
    sequences.sort(key=lambda item: str(item["user_id"]))
    problem_features = load_problem_features(artifact_dir)
    by_problem_index = {int(item["problem_index"]): item for item in problem_features}
    all_problem_indices = set(by_problem_index)
    cache = []

    for sequence in sequences:
        if len(sequence["problem_ids"]) < 2:
            continue
        target_problem = int(sequence["problem_ids"][-1])
        if target_problem not in by_problem_index:
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
            state = model.current_knowledge_state(outputs, batch["attention_mask"])[0].detach().cpu().tolist()
            valid_len = int(batch["attention_mask"][0].sum().item())
            error_history = batch["error_vectors"][0, :valid_len, :].detach().cpu()
        cache.append(
            {
                "user_id": str(sequence["user_id"]),
                "target": str(target_problem),
                "state": state,
                "error_history": error_history,
                "candidates": candidates,
            }
        )
    return cache


def evaluate_cached(
    cache: list[dict[str, Any]],
    k: int,
    alpha: float,
    beta: float,
    gamma: float,
    rho: float,
) -> dict[str, float]:
    recommendations: dict[str, list[str]] = {}
    truth: dict[str, set[str]] = {}
    for item in cache:
        weakness = PEKTRModel.temporal_error_weakness(item["error_history"], rho=rho).tolist()
        ranked = recommend_from_state(
            item["state"],
            weakness,
            item["candidates"],
            top_n=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        recommendations[item["user_id"]] = [str(score.problem_index) for score in ranked]
        truth[item["user_id"]] = {item["target"]}
    return {
        f"HR@{k}": hit_rate_at_k(recommendations, truth, k=k),
        f"NDCG@{k}": ndcg_at_k(recommendations, truth, k=k),
    }


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    device = resolve_device(args.device)
    model, config = load_model(args.checkpoint, device)
    cache = build_eval_cache(
        model=model,
        config=config,
        artifact_dir=artifact_dir,
        split=args.split,
        num_negatives=args.num_negatives,
        seed=args.seed,
        device=device,
    )
    results = []
    for alpha, beta, gamma in weight_grid():
        for rho in [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
            metrics = evaluate_cached(cache, args.k, alpha, beta, gamma, rho)
            results.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "rho": rho,
                    "evaluated_users": len(cache),
                    **metrics,
                }
            )
    results.sort(key=lambda item: (item[f"NDCG@{args.k}"], item[f"HR@{args.k}"]), reverse=True)
    summary = {
        "split": args.split,
        "k": args.k,
        "checkpoint": str(args.checkpoint),
        "best": results[0] if results else None,
        "top10": results[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

