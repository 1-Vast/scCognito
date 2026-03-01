from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
    """
    Prototype vectors p_k ∈ R^d are trainable, enabling SER without external p init.
    """
    def __init__(self, K: int, d: int):
        super().__init__()
        self.P = nn.Embedding(K, d)
        nn.init.normal_(self.P.weight, mean=0.0, std=0.02)

    def forward(self):
        return self.P.weight  # (K, d)

def semantic_energy(
    z: torch.Tensor,              # (N, d)
    c: torch.Tensor,              # (N, K)
    P: torch.Tensor,              # (K, d)
    w_proto: float = 1.0,
) -> torch.Tensor:
    """
    E_semantic: prototype alignment energy.
      target_i = Σ_k c[i,k] * p_k
      E = 1 - cos(z_i, target_i)
    """
    target = c @ P
    e = 1.0 - F.cosine_similarity(z, target, dim=1).mean()
    return w_proto * e