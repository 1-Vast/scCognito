from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PLMConfig:
    # ---------- io ----------
    h5ad_path: Path
    ser_pt_path: Path
    out_dir: Path

    # ---------- runtime ----------
    device: str = "cuda"
    emb_key: str = "X_plm"
    save_h5ad: bool = True

    # ---------- representation ----------
    use_rep: str = "X"           # "X" or key in adata.obsm (e.g., "X_pca")
    pca_dim: int = 128
    use_hvg: bool = True
    hvg_top: int = 2000

    # ---------- graphs ----------
    spatial_graph: str = "knn"   # "knn" | "radius"
    spatial_k: int = 12
    spatial_radius: Optional[float] = None
    spatial_max_edges: int = 50000  # max sampled edges for spatial neighbor loss

    attribute_graph: bool = True
    attr_k: int = 12

    # ---------- model ----------
    d_hid: int = 256
    d_out: int = 128
    n_layers: int = 2
    dropout: float = 0.1

    # ---------- optimization ----------
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    log_every: int = 10
    mask_ratio: float = 0.25
    grad_clip: float = 5.0

    # ---------- losses ----------
    w_recon: float = 1.0
    w_spatial_pred: float = 1.0
    lam_ser: float = 1.0
    ser_w_proto: float = 1.0
