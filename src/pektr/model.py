"""PEKT-R model implementation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .config import PEKTRConfig


class PEKTRModel(nn.Module):
    """Programming Error-aware Knowledge Tracing Recommendation model.

    The model implements the PEKT-R equations from the supplied manuscript:
    z_t = concat(e_t, a_t, k_t, lambda_t * g_t), where g_t is the nonlinear
    error embedding and lambda_t is an error influence gate. A causal
    Transformer encoder plays the DTransformer-style sequence encoder role.
    """

    def __init__(self, config: PEKTRConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.problem_embedding = nn.Embedding(config.n_problems + 1, d_model, padding_idx=0)
        self.answer_embedding = nn.Embedding(3, d_model, padding_idx=0)
        self.judge_embedding = nn.Embedding(config.n_judge_status + 1, d_model, padding_idx=0)
        self.knowledge_projection = nn.Linear(config.n_knowledge, d_model, bias=False)
        self.error_projection = nn.Sequential(
            nn.Linear(config.error_dim, d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.error_gate = nn.Sequential(
            nn.Linear(config.error_dim + 2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(5 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout),
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.knowledge_head = nn.Linear(d_model, config.n_knowledge)
        self.response_head = nn.Sequential(
            nn.Linear(config.n_knowledge + 2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self,
        problem_ids: torch.Tensor,
        responses: torch.Tensor,
        judge_status_ids: torch.Tensor,
        knowledge: torch.Tensor,
        error_vectors: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = problem_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.config.max_seq_len}")

        problem_emb = self.problem_embedding(problem_ids)
        answer_ids = responses.long() + 1
        answer_ids = answer_ids.masked_fill(~attention_mask, 0)
        answer_emb = self.answer_embedding(answer_ids)
        judge_emb = self.judge_embedding(judge_status_ids)
        knowledge_emb = self.knowledge_projection(knowledge.float())
        error_emb = self.error_projection(error_vectors.float())

        gate_input = torch.cat([error_vectors.float(), knowledge_emb, answer_emb], dim=-1)
        lambda_t = self.error_gate(gate_input)
        fused = torch.cat([problem_emb, answer_emb, judge_emb, knowledge_emb, lambda_t * error_emb], dim=-1)
        z = self.fusion(fused)

        positions = torch.arange(seq_len, device=problem_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        z = z + self.position_embedding(positions)
        z = z.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=problem_ids.device),
            diagonal=1,
        )
        hidden = self.encoder(
            z,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask,
        )
        hidden = hidden.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        knowledge_state = torch.sigmoid(self.knowledge_head(hidden))

        return {
            "hidden": hidden,
            "knowledge_state": knowledge_state,
            "error_gate": lambda_t.squeeze(-1),
        }

    def forward(
        self,
        problem_ids: torch.Tensor,
        responses: torch.Tensor,
        judge_status_ids: torch.Tensor,
        knowledge: torch.Tensor,
        error_vectors: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(
            problem_ids=problem_ids,
            responses=responses,
            judge_status_ids=judge_status_ids,
            knowledge=knowledge,
            error_vectors=error_vectors,
            attention_mask=attention_mask,
        )

        if problem_ids.size(1) < 2:
            encoded["next_logits"] = problem_ids.new_zeros((problem_ids.size(0), 0), dtype=torch.float32)
            encoded["next_prob"] = encoded["next_logits"]
            return encoded

        state_t = encoded["knowledge_state"][:, :-1, :]
        next_problem_emb = self.problem_embedding(problem_ids[:, 1:])
        next_knowledge_emb = self.knowledge_projection(knowledge[:, 1:, :].float())
        pred_features = torch.cat([state_t, next_problem_emb, next_knowledge_emb], dim=-1)
        logits = self.response_head(pred_features).squeeze(-1)
        encoded["next_logits"] = logits
        encoded["next_prob"] = torch.sigmoid(logits)
        return encoded

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = outputs["next_logits"]
        if logits.numel() == 0:
            zero = outputs["hidden"].sum() * 0.0
            return {"loss": zero, "kt_loss": zero, "contrastive_loss": zero}

        targets = responses[:, 1:].float()
        valid = attention_mask[:, :-1] & attention_mask[:, 1:]
        if valid.any():
            kt_loss = F.binary_cross_entropy_with_logits(logits[valid], targets[valid])
        else:
            kt_loss = logits.sum() * 0.0

        if self.config.contrastive_weight > 0:
            contrastive = self._adjacent_contrastive_loss(outputs["hidden"], attention_mask)
        else:
            contrastive = kt_loss.new_zeros(())
        loss = kt_loss + self.config.contrastive_weight * contrastive
        return {"loss": loss, "kt_loss": kt_loss, "contrastive_loss": contrastive}

    def _adjacent_contrastive_loss(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden.size(1) < 2:
            return hidden.sum() * 0.0
        left = F.normalize(hidden[:, :-1, :], dim=-1)
        right = F.normalize(hidden[:, 1:, :], dim=-1)
        positives = F.cosine_similarity(left, right, dim=-1)
        negatives = F.cosine_similarity(left, torch.roll(right, shifts=1, dims=0), dim=-1)
        valid = attention_mask[:, :-1] & attention_mask[:, 1:]
        margin_loss = F.relu(self.config.contrastive_margin - positives + negatives)
        return margin_loss[valid].mean() if valid.any() else hidden.sum() * 0.0

    def current_knowledge_state(self, outputs: dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        return outputs["knowledge_state"][batch_indices, lengths]

    def predict_candidates(
        self,
        current_state: torch.Tensor,
        candidate_problem_ids: torch.Tensor,
        candidate_knowledge: torch.Tensor,
    ) -> torch.Tensor:
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0).expand(candidate_problem_ids.size(0), -1)
        problem_emb = self.problem_embedding(candidate_problem_ids.long())
        knowledge_emb = self.knowledge_projection(candidate_knowledge.float())
        logits = self.response_head(torch.cat([current_state, problem_emb, knowledge_emb], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)

    @staticmethod
    def temporal_error_weakness(error_vectors: torch.Tensor, rho: float = 0.85) -> torch.Tensor:
        if error_vectors.numel() == 0:
            return error_vectors.new_zeros((0,))
        length = error_vectors.size(0)
        powers = torch.arange(length - 1, -1, -1, device=error_vectors.device, dtype=error_vectors.dtype)
        weights = torch.pow(torch.tensor(rho, device=error_vectors.device, dtype=error_vectors.dtype), powers)
        return (error_vectors * weights.unsqueeze(-1)).sum(dim=0)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "PEKTRModel":
        config = PEKTRConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state"])
        return model
