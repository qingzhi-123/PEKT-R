"""Configuration helpers for PEKT-R."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any


@dataclass
class PEKTRConfig:
    n_problems: int
    n_knowledge: int
    n_judge_status: int
    error_dim: int
    max_seq_len: int = 100
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    contrastive_weight: float = 0.0
    contrastive_margin: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PEKTRConfig":
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        **overrides: Any,
    ) -> "PEKTRConfig":
        artifact_dir = Path(artifact_dir)
        with (artifact_dir / "vocab.json").open("r", encoding="utf-8") as f:
            vocab = json.load(f)

        config = cls(
            n_problems=len(vocab["problem_id_to_index"]),
            n_knowledge=len(vocab["tag_index_to_name"]),
            n_judge_status=len(vocab["judge_status_to_index"]),
            error_dim=len(vocab["error_labels"]),
        )
        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        return config

