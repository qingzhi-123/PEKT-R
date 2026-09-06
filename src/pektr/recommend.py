"""Generate PEKT-R recommendations for one learner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .config import PEKTRConfig
from .dataset import (
    build_inference_batch,
    find_sequence_by_user,
    load_json,
    load_problem_features,
)
from .model import PEKTRModel
from .scoring import recommend_from_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend exercises with PEKT-R.")
    parser.add_argument("--artifact-dir", default="artifacts/bepkt")
    parser.add_argument("--checkpoint", default="checkpoints/pektr.pt")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.30)
    parser.add_argument("--rho", type=float, default=0.85)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-seen", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_candidate_knowledge(
    candidates: list[dict[str, Any]],
    n_knowledge: int,
    device: torch.device,
) -> torch.Tensor:
    knowledge = torch.zeros((len(candidates), n_knowledge), dtype=torch.float32, device=device)
    for row, candidate in enumerate(candidates):
        for tag in candidate.get("tags", []):
            tag_idx = int(tag)
            if 0 <= tag_idx < n_knowledge:
                knowledge[row, tag_idx] = 1.0
    return knowledge


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = PEKTRConfig.from_dict(checkpoint["config"])
    device = resolve_device(args.device)
    model = PEKTRModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    sequence = find_sequence_by_user(artifact_dir, args.user_id)
    if sequence is None:
        raise ValueError(f"Unknown user_id={args.user_id}")
    if len(sequence["problem_ids"]) < 1:
        raise ValueError(f"user_id={args.user_id} has no history")

    batch = build_inference_batch(
        sequence,
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
        error_history = batch["error_vectors"][0, :valid_len, :]
        weakness = PEKTRModel.temporal_error_weakness(error_history, rho=args.rho)

    problem_features = load_problem_features(artifact_dir)
    seen = set(sequence["problem_ids"]) if not args.include_seen else set()
    ranked = recommend_from_state(
        knowledge_state=state.detach().cpu().tolist(),
        error_weakness=weakness.detach().cpu().tolist(),
        candidates=problem_features,
        top_n=args.top_n,
        exclude_problem_indices=seen,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    top_candidates = [item.to_dict() for item in ranked]
    candidate_ids = torch.tensor([item["problem_index"] for item in top_candidates], dtype=torch.long, device=device)
    candidate_knowledge = make_candidate_knowledge(top_candidates, config.n_knowledge, device)
    with torch.no_grad():
        pred_correct = model.predict_candidates(state, candidate_ids, candidate_knowledge).detach().cpu().tolist()
    for item, pred in zip(top_candidates, pred_correct):
        item["predicted_correct_probability"] = float(pred)

    payload = {
        "user_id": str(args.user_id),
        "top_n": args.top_n,
        "recommendations": top_candidates,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

