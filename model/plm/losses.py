from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_gene_blocks(x: torch.Tensor, mask_ratio: float):
    """
    Mask a subset of genes per cell (row-wise gene mask).
    """
    N, D = x.shape
    mask = (torch.rand((N, D), device=x.device) < mask_ratio)
    x_m = x.clone()
    x_m[mask] = 0.0
    return x_m, mask


def masked_recon_loss(x_true: torch.Tensor, x_pred: torch.Tensor, mask: torch.Tensor):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_true.device)
    return F.mse_loss(x_pred[mask], x_true[mask])


def per_cell_masked_mse(x_true: torch.Tensor, x_pred: torch.Tensor, mask: torch.Tensor):
    """
    Per-cell reconstruction error on masked positions.
    """
    diff2 = (x_pred - x_true).pow(2) * mask.float()
    denom = mask.float().sum(dim=1).clamp_min(1.0)
    return diff2.sum(dim=1) / denom


def spatial_neighbor_recon_loss(
    x_hat_center: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    edge_spatial: torch.Tensor,
    max_edges: int = 50000,
):
    """
    Predict masked gene values of neighbors using center reconstruction.

    For each edge (i <- j): use x_hat_center[i] to predict x_j(masked).

    Memory fix:
    - Sample edges to cap the (E, D) materialization and prevent OOM on large graphs.
    """
    if edge_spatial.numel() == 0:
        return torch.tensor(0.0, device=x_true.device)

    row, col = edge_spatial  # row=i, col=j
    E = row.numel()
    if E == 0:
        return torch.tensor(0.0, device=x_true.device)

    if E > max_edges:
        perm = torch.randperm(E, device=x_true.device)[:max_edges]
        row = row[perm]
        col = col[perm]

    m = mask[col]  # (E_sample, D)
    if m.sum() == 0:
        return torch.tensor(0.0, device=x_true.device)

    pred = x_hat_center[row][m]
    true = x_true[col][m]
    return F.mse_loss(pred, true)


def spatial_smoothness_loss(z: torch.Tensor, edge_spatial: torch.Tensor):
    """
    Encourage adjacent nodes in the spatial graph to stay close in latent space.
    """
    if edge_spatial.numel() == 0:
        return torch.tensor(0.0, device=z.device)

    row, col = edge_spatial
    if row.numel() == 0:
        return torch.tensor(0.0, device=z.device)

    diff = (z[row] - z[col]).float()
    return (diff.pow(2).sum(dim=-1)).mean()
