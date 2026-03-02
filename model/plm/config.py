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
    mode: str = "finetune"

    # ---------- representation ----------
    use_rep: str = "X"
    pca_dim: int = 128
    use_hvg: bool = True
    hvg_top: int = 2000

    # ---------- graphs ----------
    spatial_graph: str = "knn"
    spatial_k: int = 12
    spatial_radius: Optional[float] = None
    spatial_max_edges: int = 50000

    attribute_graph: bool = True
    attr_k: int = 12

    # ---------- model ----------
    d_hid: int = 512
    d_out: int = 256
    n_layers: int = 3
    dropout: float = 0.1

    # ---------- optimization ----------
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1500
    log_every: int = 10
    mask_ratio: float = 0.25
    grad_clip: float = 5.0

    # ---------- losses ----------
    w_recon: float = 1.0
    w_spatial_pred: float = 1.0
    w_spatial_smooth: float = 0.5
    lam_ser: float = 1.0
    lam_ser_warmup_ratio: float = 0.15
    ser_w_proto: float = 1.0
    
    # ---------- contrastive ----------
    w_contrast: float = 0.1
    contrast_temp: float = 0.07

    # ---------- global attention ----------
    global_attn: bool = True
    global_attn_heads: int = 4
    global_attn_chunk_q: int = 1024
    global_attn_max_n: int = 8192
    global_attn_dropout: float = 0.0
    
    # --- smooth auto-calibration (recommended) ---
    smooth_auto: bool = True
    smooth_target_ratio: float = 0.08   
    smooth_update_every: int = 25      
    smooth_scale_init: float = 1.0   
    smooth_scale_clip: tuple = (1e0, 1e10)  