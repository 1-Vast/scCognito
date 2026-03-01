from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SERSignals:
    token_vocab: list[str]
    c: torch.Tensor  # (N, K)


def load_ser_signals(pt_path: str, device: str = "cuda") -> SERSignals:
    obj = torch.load(pt_path, map_location=device)
    token_vocab = obj["token_vocab"]
    c = obj["c"].to(device)
    return SERSignals(token_vocab=token_vocab, c=c)


class TrainablePrototypes(nn.Module):
    """Trainable prototype vectors p_k in R^d."""

    def __init__(self, K: int, d: int):
        super().__init__()
        self.P = nn.Embedding(K, d)
        nn.init.normal_(self.P.weight, mean=0.0, std=0.02)

    def forward(self):
        return self.P.weight  # (K, d)


def semantic_energy(
    z: torch.Tensor,  # (N, d)
    c: torch.Tensor,  # (N, K)
    P: torch.Tensor,  # (K, d)
    w_proto: float = 1.0,
    valid_mask: Optional[torch.Tensor] = None,  # (N,), include in SER loss when True
) -> torch.Tensor:
    """
    Prototype alignment energy.
    target_i = sum_k c[i,k] * p_k
    E_i = 1 - cos(z_i, target_i)
    """
    target = c @ P
    e_cell = 1.0 - F.cosine_similarity(z, target, dim=1)

    if valid_mask is None:
        valid = torch.ones_like(e_cell, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=z.device)
        if valid.dtype != torch.bool:
            valid = valid > 0
        valid = valid.reshape(-1)
        if valid.numel() != e_cell.numel():
            raise ValueError(f"valid_mask size mismatch: got {valid.numel()} vs N={e_cell.numel()}")

    if not torch.any(valid):
        return torch.zeros((), device=z.device, dtype=z.dtype)

    return w_proto * e_cell[valid].mean()
