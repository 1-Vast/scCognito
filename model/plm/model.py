from __future__ import annotations

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index  # message col -> row
        m = x[col]
        nei = scatter_mean(m, row, dim_size=x.size(0))
        h = self.lin_self(x) + self.lin_nei(nei)
        h = F.relu(h)
        return self.dropout(h)


class DualGraphEncoder(nn.Module):
    """
    Two-stream encoder: spatial graph and attribute graph.
    Fix: if edge_attr is None, return hs directly (avoid redundant fusion).
    """
    def __init__(self, d_in: int, d_hid: int, d_out: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        # logit-space fair cold start: sigmoid(0.0)=0.5
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.s_layers = nn.ModuleList()
        self.a_layers = nn.ModuleList()

        din = d_in
        for _ in range(n_layers - 1):
            self.s_layers.append(SAGEConv(din, d_hid, dropout))
            self.a_layers.append(SAGEConv(din, d_hid, dropout))
            din = d_hid

        self.s_layers.append(SAGEConv(din, d_out, dropout))
        self.a_layers.append(SAGEConv(din, d_out, dropout))

    def forward(self, x: torch.Tensor, edge_spatial: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        hs = x
        for i in range(self.n_layers):
            hs = self.s_layers[i](hs, edge_spatial)

        if edge_attr is None:
            return hs

        ha = x
        for i in range(self.n_layers):
            ha = self.a_layers[i](ha, edge_attr)

        w = torch.sigmoid(self.alpha)
        return w * hs + (1.0 - w) * ha


class DecoderMLP(nn.Module):
    def __init__(self, d_z: int, d_x: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_z),
            nn.ReLU(),
            nn.Linear(d_z, d_x),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
