from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scanpy as sc
import torch

from .graphs import build_knn_graph, build_radius_graph, normalize_adj, coo_to_edge_index


@dataclass
class PLMBatch:
    x: torch.Tensor
    spatial: torch.Tensor
    edge_spatial: torch.Tensor
    edge_attr: Optional[torch.Tensor]


def _to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def load_plm_batch(
    h5ad_path: str,
    use_rep: str = "X",
    pca_dim: int = 128,
    use_hvg: bool = True,
    hvg_top: int = 2000,
    spatial_graph: str = "knn",
    spatial_k: int = 12,
    spatial_radius: Optional[float] = None,
    attribute_graph: bool = True,
    attr_k: int = 12,
) -> PLMBatch:
    adata = sc.read_h5ad(h5ad_path)

    # ---------- biological tokenization (continuous) ----------
    if use_rep == "X":
        if use_hvg and adata.n_vars > hvg_top:
            # seurat_v3 expects raw counts; use robust fallbacks for preprocessed inputs.
            selected = False
            try:
                sc.pp.highly_variable_genes(adata, n_top_genes=hvg_top, flavor="seurat_v3")
                hv = adata.var["highly_variable"].fillna(False).to_numpy()
                if hv.sum() > 0:
                    adata = adata[:, hv].copy()
                    selected = True
            except Exception:
                pass

            if not selected:
                try:
                    ad_tmp = adata.copy()
                    sc.pp.normalize_total(ad_tmp, target_sum=1e4)
                    sc.pp.log1p(ad_tmp)
                    sc.pp.highly_variable_genes(ad_tmp, n_top_genes=hvg_top, flavor="seurat")
                    hv = ad_tmp.var["highly_variable"].fillna(False).to_numpy()
                    if hv.sum() > 0:
                        adata = adata[:, hv].copy()
                        selected = True
                except Exception as e:
                    print(f"[WARN] HVG failed; fallback to all genes. reason={e}")

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        X = _to_dense(adata.X).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    else:
        if use_rep not in adata.obsm:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            # After log1p, use seurat-style HVG (not seurat_v3).
            if use_hvg and adata.n_vars > hvg_top:
                sc.pp.highly_variable_genes(adata, n_top_genes=hvg_top, flavor="seurat")
                ad = adata[:, adata.var["highly_variable"]].copy()
            else:
                ad = adata

            sc.pp.pca(ad, n_comps=pca_dim)
            adata.obsm["X_pca"] = ad.obsm["X_pca"]

        X = np.asarray(adata.obsm[use_rep], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- spatial coordinates ----------
    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        has_true_spatial = True
    else:
        print("[WARN] No 'spatial' found in adata.obsm. Spatial graph will be disabled.")
        spatial = np.zeros((adata.n_obs, 2), dtype=np.float32)
        has_true_spatial = False

    # ---------- graphs ----------
    # Spatial graph: disabled if no true spatial coordinates
    if has_true_spatial:
        if spatial_graph == "knn":
            A_s = build_knn_graph(spatial, k=spatial_k)
        elif spatial_graph == "radius":
            if spatial_radius is None:
                raise ValueError("spatial_radius must be set for radius graph")
            A_s = build_radius_graph(spatial, radius=float(spatial_radius))
        else:
            raise ValueError(f"Unknown spatial_graph: {spatial_graph}")

        A_s = normalize_adj(A_s)
        edge_spatial = coo_to_edge_index(A_s)
    else:
        edge_spatial = torch.empty((2, 0), dtype=torch.long)

    # Attribute graph: KNN on expression tokens (optional)
    edge_attr = None
    if attribute_graph:
        A_a = build_knn_graph(X, k=attr_k)
        A_a = normalize_adj(A_a)
        edge_attr = coo_to_edge_index(A_a)

    return PLMBatch(
        x=torch.tensor(X),
        spatial=torch.tensor(spatial),
        edge_spatial=edge_spatial,
        edge_attr=edge_attr,
    )
