"""Tiny export-friendly text transformer for contextual QSO semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn

from .semantic_data import (
    ACT_NAMES,
    CHAR_TO_ID,
    MAX_SEMANTIC_LENGTH,
    PENDING_QUESTION_NAMES,
    PHASE_NAMES,
    PROCEDURE_GRADES,
    REGISTER_NAMES,
    SLOT_NAMES,
    TOPIC_NAMES,
)


SEMANTIC_MODEL_VERSION = "qso-semanticformer-0.4"
SEMANTIC_INPUT_NAMES = (
    "input_ids", "valid_mask", "context_ids", "self_callsign_ids",
    "peer_callsign_ids", "known_slots",
)
SEMANTIC_OUTPUT_NAMES = (
    "act_logits", "topic_logits", "slot_logits", "register_logits", "procedure_logits", "safe_to_commit_logits",
)


@dataclass(frozen=True)
class SemanticModelConfig:
    vocab_size: int = max(CHAR_TO_ID.values()) + 1
    d_model: int = 128
    layers: int = 6
    heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.10
    max_length: int = MAX_SEMANTIC_LENGTH

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextBlock(nn.Module):
    def __init__(self, config: SemanticModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.d_model // config.heads
        self.norm1 = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ff_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.d_model), nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = values.shape
        normalized = self.norm1(values)
        qkv = self.qkv(normalized).reshape(batch, steps, 3, self.heads, self.head_dim)
        query, key, value = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid_mask[:, None, None, :], -10_000.0)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value).transpose(1, 2).reshape(batch, steps, channels)
        mask = valid_mask.unsqueeze(-1).to(values.dtype)
        values = (values + self.dropout(self.projection(context))) * mask
        values = (values + self.ffn(self.norm2(values))) * mask
        return values


class QsoSemanticFormer(nn.Module):
    def __init__(self, config: SemanticModelConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            config = SemanticModelConfig()
        elif isinstance(config, dict):
            config = SemanticModelConfig(**config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.position = nn.Embedding(config.max_length, config.d_model)
        self.phase_embedding = nn.Embedding(len(PHASE_NAMES), config.d_model)
        self.pending_embedding = nn.Embedding(len(PENDING_QUESTION_NAMES), config.d_model)
        self.known_slots_projection = nn.Sequential(
            nn.Linear(len(TOPIC_NAMES), config.d_model), nn.SiLU(),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(config.d_model * 5, config.d_model),
            nn.SiLU(),
            nn.LayerNorm(config.d_model),
        )
        self.blocks = nn.ModuleList(TextBlock(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.slot_context_projection = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.SiLU(),
            nn.LayerNorm(config.d_model),
        )
        self.act_head = nn.Linear(config.d_model, len(ACT_NAMES))
        self.topic_head = nn.Linear(config.d_model, len(TOPIC_NAMES))
        self.slot_head = nn.Linear(config.d_model, len(SLOT_NAMES))
        self.register_head = nn.Linear(config.d_model, len(REGISTER_NAMES))
        self.procedure_head = nn.Linear(config.d_model, len(PROCEDURE_GRADES))
        self.safe_to_commit_head = nn.Linear(config.d_model, 1)

    def _pool_identifier(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).unsqueeze(-1).to(torch.float32)
        embedded = self.embedding(ids)
        return (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        context_ids: torch.Tensor,
        self_callsign_ids: torch.Tensor,
        peer_callsign_ids: torch.Tensor,
        known_slots: torch.Tensor,
    ):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        mask = valid_mask.unsqueeze(-1).to(torch.float32)
        phase = self.phase_embedding(context_ids[:, 0])
        pending = self.pending_embedding(context_ids[:, 1])
        self_call = self._pool_identifier(self_callsign_ids)
        peer_call = self._pool_identifier(peer_callsign_ids)
        known = self.known_slots_projection(known_slots.to(torch.float32))
        dialogue_context = self.context_projection(torch.cat(
            (phase, pending, self_call, peer_call, known), dim=-1,
        ))
        cls_selector = (positions == 0).unsqueeze(-1).to(torch.float32)
        values = self.embedding(input_ids) + self.position(positions)
        values = (values + dialogue_context.unsqueeze(1) * cls_selector) * mask
        for block in self.blocks:
            values = block(values, valid_mask)
        values = self.norm(values) * mask
        pooled = values[:, 0]
        expanded_context = dialogue_context.unsqueeze(1).expand(-1, values.shape[1], -1)
        slot_values = self.slot_context_projection(torch.cat(
            (values, expanded_context), dim=-1,
        )) * mask
        return (
            self.act_head(pooled),
            self.topic_head(pooled),
            self.slot_head(slot_values),
            self.register_head(pooled),
            self.procedure_head(pooled),
            self.safe_to_commit_head(pooled).squeeze(-1),
        )


def semantic_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
