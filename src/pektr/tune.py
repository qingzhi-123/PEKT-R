"""GPU-friendly PEKT-R hyperparameter search."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import PEKTRConfig
from .dataset import SequenceWindowDataset, collate_pektr, load_json
from .evaluate import evaluate_model
from .model import PEKTRModel
from .train import move_batch, resolve_device, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune PEKT-R hyperparameters on GPU.")
    parser.add_argument("--artifact-dir", default="artifacts/bepkt")
    parser.add_argument("--output-dir", default="artifacts/tuning")
    parser.add_argument("--best-checkpoint", default="checkpoints/pektr_best.pt")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--max-val-users", type=int, default=82)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=["quick", "balanced"], default="quick")
    return parser.parse_args()


def trial_space(preset: str, rng: random.Random) -> list[dict[str, Any]]:
    if preset == "quick":
        return [
            {
                "max_seq_len": 64,
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.10,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "alpha": 0.45,
                "beta": 0.25,
                "gamma": 0.30,
                "rho": 0.85,
            },
            {
                "max_seq_len": 96,
                "d_model": 96,
                "n_heads": 4,
                "n_layers": 2,
                "dim_feedforward": 192,
                "dropout": 0.15,
                "lr": 7e-4,
                "weight_decay": 1e-4,
                "alpha": 0.40,
                "beta": 0.20,
                "gamma": 0.40,
                "rho": 0.90,
            },
            {
                "max_seq_len": 128,
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.10,
                "lr": 5e-4,
                "weight_decay": 5e-5,
                "alpha": 0.50,
                "beta": 0.20,
                "gamma": 0.30,
                "rho": 0.85,
            },
            {
                "max_seq_len": 128,
                "d_model": 128,
                "n_heads": 8,
                "n_layers": 2,
                "dim_feedforward": 384,
                "dropout": 0.20,
                "lr": 5e-4,
                "weight_decay": 1e-4,
                "alpha": 0.35,
                "beta": 0.25,
                "gamma": 0.40,
                "rho": 0.90,
            },
            {
                "max_seq_len": 160,
                "d_model": 160,
                "n_heads": 8,
                "n_layers": 3,
                "dim_feedforward": 320,
                "dropout": 0.15,
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "alpha": 0.45,
                "beta": 0.15,
                "gamma": 0.40,
                "rho": 0.92,
            },
            {
                "max_seq_len": 192,
                "d_model": 192,
                "n_heads": 8,
                "n_layers": 2,
                "dim_feedforward": 384,
                "dropout": 0.10,
                "lr": 3e-4,
                "weight_decay": 5e-5,
                "alpha": 0.50,
                "beta": 0.15,
                "gamma": 0.35,
                "rho": 0.88,
            },
        ]

    combos = []
    for max_seq_len, d_model, n_layers, dropout, lr in itertools.product(
        [96, 128, 160, 192],
        [96, 128, 160, 192],
        [2, 3],
        [0.10, 0.15, 0.20],
        [7e-4, 5e-4, 3e-4],
    ):
        heads = 8 if d_model % 8 == 0 else 4
        combos.append(
            {
                "max_seq_len": max_seq_len,
                "d_model": d_model,
                "n_heads": heads,
                "n_layers": n_layers,
                "dim_feedforward": d_model * 2,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": rng.choice([5e-5, 1e-4, 2e-4]),
                "alpha": rng.choice([0.35, 0.40, 0.45, 0.50]),
                "beta": rng.choice([0.15, 0.20, 0.25]),
                "gamma": rng.choice([0.30, 0.35, 0.40, 0.45]),
                "rho": rng.choice([0.85, 0.88, 0.90, 0.92]),
            }
        )
    rng.shuffle(combos)
    return combos


def build_loader(
    artifact_dir: Path,
    split: str,
    config: PEKTRConfig,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = SequenceWindowDataset(
        artifact_dir=artifact_dir,
        split=split,
        max_seq_len=config.max_seq_len,
        window_stride=max(1, config.max_seq_len // 2),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_pektr(batch, config.n_knowledge, config.error_dim),
    )


def one_trial(
    trial_id: int,
    artifact_dir: Path,
    params: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], PEKTRModel]:
    config = PEKTRConfig.from_artifact_dir(
        artifact_dir,
        max_seq_len=params["max_seq_len"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        dim_feedforward=params["dim_feedforward"],
        dropout=params["dropout"],
    )
    model = PEKTRModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    train_loader = build_loader(artifact_dir, "train", config, args.batch_size, shuffle=True)
    val_loader = build_loader(artifact_dir, "val", config, args.batch_size, shuffle=False)

    start = time.time()
    last_train = {}
    last_val_loss = {}
    for _ in range(args.epochs):
        last_train = run_epoch(model, train_loader, optimizer, device)
        last_val_loss = run_epoch(model, val_loader, None, device)

    val_rank = evaluate_model(
        model=model,
        config=config,
        artifact_dir=artifact_dir,
        split="val",
        k=args.eval_k,
        num_negatives=args.num_negatives,
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=params["gamma"],
        rho=params["rho"],
        seed=args.seed,
        device=device,
        max_users=args.max_val_users,
    )
    elapsed = time.time() - start
    result = {
        "trial": trial_id,
        "params": params,
        "train": last_train,
        "val_loss": last_val_loss,
        "val_rank": val_rank,
        "score": float(val_rank[f"NDCG@{args.eval_k}"]),
        "elapsed_sec": round(elapsed, 2),
        "device": str(device),
    }
    return result, model


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "pektr_tuning_results.jsonl"
    best_checkpoint = Path(args.best_checkpoint)
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    candidates = trial_space(args.preset, rng)[: args.trials]
    best_result: dict[str, Any] | None = None
    best_model: PEKTRModel | None = None
    with results_path.open("a", encoding="utf-8") as f:
        for trial_id, params in enumerate(candidates, start=1):
            result, model = one_trial(trial_id, artifact_dir, params, args, device)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(result, ensure_ascii=False))
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result
                best_model = model
                torch.save(
                    {
                        "config": best_model.config.to_dict(),
                        "model_state": best_model.state_dict(),
                        "tuning_result": best_result,
                        "vocab_summary": {
                            "n_problems": len(load_json(artifact_dir / "vocab.json")["problem_id_to_index"]),
                            "n_knowledge": best_model.config.n_knowledge,
                        },
                    },
                    best_checkpoint,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "best_checkpoint": str(best_checkpoint),
        "best_result": best_result,
        "results_path": str(results_path),
    }
    with (output_dir / "best_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

