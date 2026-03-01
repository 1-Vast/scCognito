from __future__ import annotations

from pathlib import Path
import numpy as np
import scanpy as sc
import torch

from .model import DualGraphEncoder
from .data import load_plm_batch
from .config import PLMConfig


def export_embeddings(cfg: PLMConfig, ckpt_path: Path):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    obj = torch.load(str(ckpt_path), map_location=device)

    batch = load_plm_batch(
        h5ad_path=str(cfg.h5ad_path),
        use_rep=cfg.use_rep,
        pca_dim=cfg.pca_dim,
        use_hvg=cfg.use_hvg,
        hvg_top=cfg.hvg_top,
        spatial_graph=cfg.spatial_graph,
        spatial_k=cfg.spatial_k,
        spatial_radius=cfg.spatial_radius,
        attribute_graph=cfg.attribute_graph,
        attr_k=cfg.attr_k,
    )

    x = batch.x.to(device)
    edge_s = batch.edge_spatial.to(device)
    edge_a = batch.edge_attr.to(device) if batch.edge_attr is not None else None

    encoder = DualGraphEncoder(
        d_in=x.size(1),
        d_hid=cfg.d_hid,
        d_out=cfg.d_out,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)
    encoder.load_state_dict(obj["encoder"])
    encoder.eval()

    with torch.no_grad():
        z = encoder(x, edge_s, edge_a).detach().cpu().numpy()

    adata = sc.read_h5ad(str(cfg.h5ad_path))
    adata.obsm[cfg.emb_key] = z.astype(np.float32)

    # Write per-cell metrics from training
    last_metrics = obj.get("last_metrics", {})
    if "recon_err_cell" in last_metrics:
        adata.obs["plm_recon_err"] = last_metrics["recon_err_cell"]
    if "ser_energy_cell" in last_metrics:
        adata.obs["plm_ser_energy"] = last_metrics["ser_energy_cell"]

    # Write semantic coverage/strength from SER signals
    try:
        ser = torch.load(str(cfg.ser_pt_path), map_location="cpu")
        c = ser["c"].numpy()
        if c.shape[0] == adata.n_obs:
            adata.obs["ser_coverage"] = (c > 0).sum(axis=1)
            adata.obs["ser_strength"] = c.sum(axis=1)
        else:
            print("[WARN] SER c N mismatch, skip coverage/strength.")
    except Exception as e:
        print("[WARN] Failed to load SER pt:", e)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_h5ad = cfg.out_dir / "plm_embedded.h5ad"
    if cfg.save_h5ad:
        adata.write_h5ad(str(out_h5ad))
        print("[OK] saved:", out_h5ad)

    return out_h5ad