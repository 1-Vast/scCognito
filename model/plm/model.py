from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = scatter_add(src, index, dim_size)
    cnt = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    ones = torch.ones(index.size(0), 1, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, ones)
    return out / (cnt + 1e-12)

class SAGEConv(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_nei = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        nei = scatter_mean(x[col], row, dim_size=x.size(0))
        h = self.lin_self(x) + self.lin_nei(nei)
        h = self.norm(h)
        h = F.gelu(h)
        return self.dropout(h)

class DecoderMLP(nn.Module):
    def __init__(self, d_z: int, d_x: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_z),
            nn.LayerNorm(d_z),
            nn.GELU(),
            nn.Linear(d_z, d_x),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class TalkingHeadAttention(nn.Module):
    """
    Talking-Heads self-attention:
    - Linear projection across the head dimension before softmax (logits mixing)
    - Linear projection across the head dimension after softmax (weights mixing)

    Engineering:
    - Query chunking reduces peak memory (avoid full H*N*N materialization).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        chunk_q: int = 1024,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads: d_model={d_model}, n_heads={n_heads}")

        self.n_heads = int(n_heads)
        self.d_k = int(d_model // n_heads)
        self.chunk_q = int(chunk_q)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.pre_softmax_proj = nn.Linear(self.n_heads, self.n_heads)
        self.post_softmax_proj = nn.Linear(self.n_heads, self.n_heads)

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, d_model)
        n = int(x.size(0))
        if n <= 1:
            return x

        residual = x
        x = self.norm(x)

        q = self.q_proj(x).view(n, self.n_heads, self.d_k).transpose(0, 1)  # (H, N, d_k)
        k = self.k_proj(x).view(n, self.n_heads, self.d_k).transpose(0, 1)
        v = self.v_proj(x).view(n, self.n_heads, self.d_k).transpose(0, 1)

        k_t = k.transpose(-2, -1)  # (H, d_k, N)
        scale = 1.0 / math.sqrt(float(self.d_k))

        out = x.new_empty((n, self.n_heads * self.d_k))
        step = n if self.chunk_q <= 0 else max(1, self.chunk_q)

        for q0 in range(0, n, step):
            q1 = min(n, q0 + step)
            qb = q[:, q0:q1, :]  # (H, B, d_k)

            scores = torch.matmul(qb, k_t) * scale  # (H, B, N)

            # logits talking-heads (pre-softmax head mixing)
            scores = scores.permute(1, 2, 0)  # (B, N, H)
            scores = self.pre_softmax_proj(scores)
            scores = scores.permute(2, 0, 1)  # (H, B, N)

            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)

            # weights talking-heads (post-softmax head mixing)
            attn = attn.permute(1, 2, 0)  # (B, N, H)
            attn = self.post_softmax_proj(attn)
            attn = attn.permute(2, 0, 1)  # (H, B, N)

            out_b = torch.matmul(attn, v)  # (H, B, d_k)
            out_b = out_b.transpose(0, 1).contiguous().view(q1 - q0, -1)  # (B, d_model)
            out[q0:q1] = out_b

        return residual + self.out_proj(out)


class DualGraphEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        n_layers: int = 2,
        dropout: float = 0.1,
        global_attn: bool = True,
        global_attn_heads: int = 4,
        global_attn_chunk_q: int = 1024,
        global_attn_max_n: int = 8192,
        global_attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.d_hid = int(d_hid)
        self.d_out = int(d_out)

        self.s_layers = nn.ModuleList()
        self.a_layers = nn.ModuleList()

        din = int(d_in)
        for _ in range(self.n_layers - 1):
            self.s_layers.append(SAGEConv(din, int(d_hid), dropout))
            self.a_layers.append(SAGEConv(din, int(d_hid), dropout))
            din = int(d_hid)

        self.s_layers.append(SAGEConv(din, int(d_out), dropout))
        self.a_layers.append(SAGEConv(din, int(d_out), dropout))

        # Multi-scale feature dim across all encoder layers.
        self.multi_scale_dim = int(d_hid) * max(0, self.n_layers - 1) + int(d_out)

        # Stabilize gate input to avoid sigmoid saturation.
        self.norm_concat = nn.LayerNorm(int(self.multi_scale_dim) * 2)
        self.fusion_gate = nn.Sequential(
            nn.Linear(int(self.multi_scale_dim) * 2, max(16, int(d_hid) // 2)),
            nn.ReLU(),
            nn.Linear(max(16, int(d_hid) // 2), 1),
        )
        self.reduce_proj = nn.Linear(int(self.multi_scale_dim), int(d_out))

        self.enable_global_attn = bool(global_attn)
        self.global_attn_max_n = int(global_attn_max_n)
        self.global_attention = TalkingHeadAttention(
            d_model=int(d_out),
            n_heads=int(global_attn_heads),
            dropout=float(global_attn_dropout),
            chunk_q=int(global_attn_chunk_q),
        )

    def encode_streams(
        self,
        x: torch.Tensor,
        edge_spatial: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        hs = x
        hs_scales: list[torch.Tensor] = []
        for i in range(self.n_layers):
            hs = self.s_layers[i](hs, edge_spatial)
            hs_scales.append(hs)
        hs_multi = torch.cat(hs_scales, dim=-1)

        if edge_attr is None:
            return hs_multi, None

        ha = x
        ha_scales: list[torch.Tensor] = []
        for i in range(self.n_layers):
            ha = self.a_layers[i](ha, edge_attr)
            ha_scales.append(ha)
        ha_multi = torch.cat(ha_scales, dim=-1)
        return hs_multi, ha_multi

    def fuse_streams(self, hs: torch.Tensor, ha: Optional[torch.Tensor]) -> torch.Tensor:
        if ha is None:
            fused = hs
        else:
            concat = torch.cat([hs, ha], dim=-1)
            w = torch.sigmoid(self.fusion_gate(self.norm_concat(concat)))  # (N, 1)
            fused = w * hs + (1.0 - w) * ha

        z = self.reduce_proj(fused)

        if self.enable_global_attn and int(z.size(0)) <= self.global_attn_max_n:
            z = self.global_attention(z)
        return F.normalize(z, p=2, dim=-1)

    def forward(self, x: torch.Tensor, edge_spatial: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        hs, ha = self.encode_streams(x, edge_spatial, edge_attr)
        return self.fuse_streams(hs, ha)
