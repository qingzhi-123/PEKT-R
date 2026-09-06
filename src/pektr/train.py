"""Training entry point for PEKT-R."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None

from .config import PEKTRConfig
from .dataset import SequenceWindowDataset, collate_pektr, load_json
from .model import PEKTRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PEKT-R.")
    parser.add_argument("--artifact-dir", default="artifacts/bepkt")
    parser.add_argument("--checkpoint", default="checkpoints/pektr.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(
    model: PEKTRModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "kt_loss": 0.0, "contrastive_loss": 0.0, "batches": 0}
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, leave=False, desc="train" if training else "eval")
    for batch in iterator:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(
                problem_ids=batch["problem_ids"],
                responses=batch["responses"],
                judge_status_ids=batch["judge_status_ids"],
                knowledge=batch["knowledge"],
                error_vectors=batch["error_vectors"],
                attention_mask=batch["attention_mask"],
            )
            losses = model.compute_loss(outputs, batch["responses"], batch["attention_mask"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        for key in ("loss", "kt_loss", "contrastive_loss"):
            totals[key] += float(losses[key].detach().cpu())
        totals["batches"] += 1
    denom = max(1, totals.pop("batches"))
    return {key: value / denom for key, value in totals.items()}


def build_loader(
    artifact_dir: str | Path,
    split: str,
    config: PEKTRConfig,
    batch_size: int,
    window_stride: int | None,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = SequenceWindowDataset(
        artifact_dir=artifact_dir,
        split=split,
        max_seq_len=config.max_seq_len,
        window_stride=window_stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_pektr(batch, config.n_knowledge, config.error_dim),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    artifact_dir = Path(args.artifact_dir)
    if not (artifact_dir / "vocab.json").exists():
        raise FileNotFoundError(f"Run pektr-preprocess-bepkt first: {artifact_dir}")

    vocab = load_json(artifact_dir / "vocab.json")
    config = PEKTRConfig.from_artifact_dir(
        artifact_dir,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        contrastive_weight=args.contrastive_weight,
    )
    device = resolve_device(args.device)
    model = PEKTRModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = build_loader(
        artifact_dir,
        "train",
        config,
        batch_size=args.batch_size,
        window_stride=args.window_stride,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = build_loader(
        artifact_dir,
        "val",
        config,
        batch_size=args.batch_size,
        window_stride=args.window_stride,
        num_workers=args.num_workers,
        shuffle=False,
    )

    best_val = float("inf")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["loss"] <= best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "vocab_summary": {
                        "n_problems": len(vocab["problem_id_to_index"]),
                        "n_knowledge": len(vocab["tag_index_to_name"]),
                        "error_labels": vocab["error_labels"],
                    },
                    "history": history,
                },
                checkpoint_path,
            )
    print(f"saved_best_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()

