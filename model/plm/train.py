from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.optim import AdamW

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x

from .config import PLMConfig
from .model import DualGraphEncoder, DecoderMLP
from .ser import load_ser_signals, TrainablePrototypes, semantic_energy


# =========================
# Graph builders (merged)
# =========================

def _empty_graph(n: int) -> sp.coo_matrix:
    return sp.coo_matrix((n, n), dtype=np.float32)


def build_knn_graph(X: np.ndarray, k: int = 12) -> sp.coo_matrix:
    N = int(X.shape[0])
    if N < 3:
        return _empty_graph(N)

    k_eff = int(max(1, min(int(k), N - 1)))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]

    rows = np.repeat(np.arange(N), k_eff)
    cols = idx.reshape(-1)
    data = np.ones_like(rows, dtype=np.float32)

    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocoo()


def build_radius_graph(X: np.ndarray, radius: float) -> sp.coo_matrix:
    nn = NearestNeighbors(radius=float(radius), metric="euclidean").fit(X)
    graph = nn.radius_neighbors_graph(X, mode="connectivity")
    A = graph.tocoo()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocoo()


def normalize_adj(A: sp.coo_matrix) -> sp.coo_matrix:
    A = A.tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
    D_inv = sp.diags(deg_inv_sqrt)
    An = D_inv @ A @ D_inv
    return An.tocoo()


def coo_to_edge_index(A: sp.coo_matrix) -> torch.Tensor:
    return torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)


def build_graph_edge_index(
    X: np.ndarray,
    kind: str,
    k: int = 12,
    radius: Optional[float] = None,
    normalize: bool = True,
) -> torch.Tensor:
    kind = str(kind).strip().lower()
    if kind == "knn":
        A = build_knn_graph(X, k=int(k))
    elif kind == "radius":
        if radius is None:
            raise ValueError("radius must be set for radius graph")
        A = build_radius_graph(X, radius=float(radius))
    else:
        raise ValueError(f"Unknown graph kind: {kind}")

    if normalize:
        A = normalize_adj(A)
    return coo_to_edge_index(A)


# =========================
# Data loader (merged)
# =========================

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


def _sample_values(X, max_values: int = 20000) -> np.ndarray:
    if sp.issparse(X):
        vals = np.asarray(X.data, dtype=np.float32).reshape(-1)
    else:
        vals = np.asarray(X, dtype=np.float32).reshape(-1)
    if vals.size <= max_values:
        return vals
    idx = np.linspace(0, vals.size - 1, num=max_values, dtype=np.int64)
    return vals[idx]


def _looks_like_raw_counts(X) -> bool:
    vals = _sample_values(X)
    if vals.size == 0:
        return False
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False
    if float(vals.min()) < 0.0:
        return False
    frac_integer = float(np.mean(np.isclose(vals, np.round(vals), atol=1e-3)))
    p99 = float(np.quantile(vals, 0.99))
    return frac_integer >= 0.90 and p99 >= 1.0


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

    if use_rep == "X":
        raw_count_like = _looks_like_raw_counts(adata.X)
        if use_hvg and adata.n_vars > hvg_top:
            selected = False
            if raw_count_like:
                try:
                    sc.pp.highly_variable_genes(adata, n_top_genes=hvg_top, flavor="seurat_v3")
                    hv = adata.var["highly_variable"].fillna(False).to_numpy()
                    if hv.sum() > 0:
                        adata = adata[:, hv].copy()
                        selected = True
                except Exception as e:
                    print(f"[WARN][HVG] seurat_v3 failed; fallback to seurat. reason={e}")

            if not selected:
                try:
                    ad_tmp = adata.copy()
                    if raw_count_like:
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

            if use_hvg and adata.n_vars > hvg_top:
                sc.pp.highly_variable_genes(adata, n_top_genes=hvg_top, flavor="seurat")
                ad = adata[:, adata.var["highly_variable"]].copy()
            else:
                ad = adata

            sc.pp.pca(ad, n_comps=pca_dim)
            adata.obsm["X_pca"] = ad.obsm["X_pca"]

        X = np.asarray(adata.obsm[use_rep], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        has_true_spatial = True
    else:
        spatial = np.zeros((adata.n_obs, 2), dtype=np.float32)
        has_true_spatial = False
        print("[WARN] Missing adata.obsm['spatial']; spatial graph disabled.", flush=True)

    if has_true_spatial:
        edge_spatial = build_graph_edge_index(
            X=spatial,
            kind=spatial_graph,
            k=int(spatial_k),
            radius=spatial_radius,
            normalize=False,
        )
    else:
        edge_spatial = torch.empty((2, 0), dtype=torch.long)

    edge_attr = None
    if attribute_graph:
        A_a = build_knn_graph(X, k=int(attr_k))
        A_a = normalize_adj(A_a)
        edge_attr = coo_to_edge_index(A_a)

    return PLMBatch(
        x=torch.tensor(X),
        spatial=torch.tensor(spatial),
        edge_spatial=edge_spatial,
        edge_attr=edge_attr,
    )


# =========================
# Losses (merged)
# =========================

def mask_gene_blocks(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    mr = float(max(0.0, min(1.0, mask_ratio)))
    if mr <= 0.0:
        mask = torch.zeros_like(x, dtype=torch.bool)
        return x, mask
    
    mask = torch.rand(x.shape, device=x.device) < mr
    x_m = x.clone()
    x_m[mask] = 0.0
    return x_m, mask

def masked_recon_loss(x_true: torch.Tensor, x_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x_true = x_true.float()
    x_pred = x_pred.float()
    if mask is None or mask.numel() == 0 or (mask.sum() == 0):
        return F.mse_loss(x_pred, x_true)
    diff = (x_pred - x_true).pow(2)
    return diff[mask].mean()


def spatial_neighbor_recon_loss(
    x_hat_center: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    edge_spatial: torch.Tensor,
    max_edges: int = 50000,
) -> torch.Tensor:
    if edge_spatial.numel() == 0:
        return torch.tensor(0.0, device=x_true.device)

    row, col = edge_spatial
    if row.numel() == 0:
        return torch.tensor(0.0, device=x_true.device)

    if int(row.numel()) > int(max_edges):
        idx = torch.randperm(int(row.numel()), device=row.device)[: int(max_edges)]
        row = row[idx]
        col = col[idx]

    x_hat_n = x_hat_center[col]
    x_true_c = x_true[row]

    if mask is None or mask.numel() == 0 or (mask.sum() == 0):
        return F.mse_loss(x_hat_n, x_true_c)

    mask_c = mask[row]
    diff = (x_hat_n - x_true_c).pow(2)
    if mask_c.sum() == 0:
        return torch.tensor(0.0, device=x_true.device)
    return diff[mask_c].mean()


def spatial_smoothness_loss(z: torch.Tensor, edge_spatial: torch.Tensor) -> torch.Tensor:
    if edge_spatial.numel() == 0:
        return torch.tensor(0.0, device=z.device)

    row, col = edge_spatial
    if row.numel() == 0:
        return torch.tensor(0.0, device=z.device)

    diff = (z[row] - z[col]).float()
    return (diff.pow(2).mean(dim=-1)).mean()


def cross_view_infonce(hs: torch.Tensor, ha: torch.Tensor, scale, eps: float = 1e-6, max_samples: int = 8192) -> torch.Tensor:
    N = int(hs.size(0))
    if N > max_samples:
        idx = torch.randperm(N, device=hs.device)[:max_samples]
        hs = hs[idx]
        ha = ha[idx]
        
    hs = F.normalize(hs, p=2, dim=-1)
    ha = F.normalize(ha, p=2, dim=-1)
    sim = hs @ ha.t()

    if isinstance(scale, torch.Tensor):
        s = scale.to(device=hs.device, dtype=hs.dtype)
    else:
        s = torch.tensor(float(scale), device=hs.device, dtype=hs.dtype)
    s = s.clamp_min(eps)

    logits = sim * s
    labels = torch.arange(hs.size(0), device=hs.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)

# =========================
# Utilities
# =========================

def _should_enable_tqdm() -> bool:
    flag = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _init_amp(device: torch.device):
    device_type = device.type
    use_amp = (device_type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    return device_type, use_amp, scaler


def _schedule_weights(ep: int, total_epochs: int, w_contrast: float, lam_ser: float) -> Tuple[float, float]:
    e = int(max(1, total_epochs))
    s0 = int(max(1, round(0.30 * e)))
    s1 = int(max(s0 + 1, round(0.60 * e)))
    s2 = int(max(s1 + 1, round(0.85 * e)))

    # contrast: 0 -> warmup -> hold -> decay
    if ep <= s0:
        w_ctr = 0.0
    elif ep <= s1:
        t = float(ep - s0) / float(max(1, s1 - s0))
        w_ctr = float(w_contrast) * t
    elif ep <= s2:
        w_ctr = float(w_contrast)
    else:
        t = float(ep - s2) / float(max(1, e - s2))
        w_ctr = float(w_contrast) * (1.0 - 0.70 * t)

    ser_start = int(max(1, round(0.20 * e)))
    if ep <= ser_start:
        lam = 0.0
    elif ep <= s2:
        t = float(ep - ser_start) / float(max(1, s2 - ser_start))
        lam = float(lam_ser) * (t ** 2)
    else:
        lam = float(lam_ser)

    return w_ctr, lam


def _save_ckpt(cfg: PLMConfig, encoder, decoder, prototypes, token_vocab, last_metrics: dict, last_good_ep: int, last_loss: Optional[float]) -> Path:
    ckpt = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "prototypes": prototypes.state_dict() if prototypes is not None else {},
        "token_vocab": token_vocab,
        "config": cfg.__dict__,
        "last_metrics": last_metrics,
        "last_good_ep": int(last_good_ep),
        "last_loss": float(last_loss) if last_loss is not None else None,
    }
    out = cfg.out_dir / "plm_ckpt.pt"
    torch.save(ckpt, out)
    return out


# =========================
# Training (merged full/minibatch)
# =========================

def run_train(cfg: PLMConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    device_type, use_amp, scaler = _init_amp(device)

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

    ser_obj = load_ser_signals(str(cfg.ser_pt_path), device=str(device))
    c = ser_obj.c
    if int(c.size(0)) != int(x.size(0)):
        raise ValueError(
            f"SER c has N={int(c.size(0))} but data has N={int(x.size(0))}. Bridge must match the dataset."
        )

    coverage_mask = (c.sum(dim=1) > 0)
    covered_n = int(coverage_mask.sum().item())
    total_n = int(coverage_mask.numel())
    
    print(f"\n[DEBUG] SER Coverage: {covered_n} out of {total_n} cells have text signals!", flush=True)
    if covered_n == 0:
        print(f"[ERROR] Semantic label mapping failed completely! groupby={cfg.groupby}.", flush=True)
        raise ValueError(
            "Detected label mismatch: all cells have zero semantic signal. Remove --skip_teacher to rebuild Teacher dict, or check the h5ad cluster labels."
        )

    encoder = DualGraphEncoder(
        d_in=int(x.size(1)),
        d_hid=int(cfg.d_hid),
        d_out=int(cfg.d_out),
        n_layers=int(cfg.n_layers),
        dropout=float(cfg.dropout),
        global_attn=bool(cfg.global_attn),
        global_attn_heads=int(cfg.global_attn_heads),
        global_attn_chunk_q=int(cfg.global_attn_chunk_q),
        global_attn_max_n=int(cfg.global_attn_max_n),
        global_attn_dropout=float(cfg.global_attn_dropout),
        contrastive_init_temp=float(cfg.contrast_temp),
        learnable_contrastive_temp=True,
    ).to(device)

    decoder = DecoderMLP(d_z=int(cfg.d_out), d_x=int(x.size(1))).to(device)

    K = len(ser_obj.token_vocab)
    token_texts = ser_obj.token_texts if ser_obj.token_texts is not None else ser_obj.token_vocab
    prototypes = TrainablePrototypes(
        K=K,
        d=int(cfg.d_out),
        init_texts=token_texts,
        text_encoder_name=getattr(cfg, "text_encoder_name", None),
        text_batch_size=int(getattr(cfg, "text_batch_size", 16)),
        text_max_length=int(getattr(cfg, "text_max_length", 32)),
        text_pooling=str(getattr(cfg, "text_pooling", "mean")),
        text_device=str(getattr(cfg, "text_device", "cpu")),
        enable_text_init=getattr(cfg, "enable_text_init", None),
    ).to(device)

    all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(prototypes.parameters())
    opt = AdamW(all_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    pbar = tqdm(
        range(1, int(cfg.epochs) + 1),
        total=int(cfg.epochs),
        desc="PLM Train",
        unit="ep",
        ascii=True,
        ncols=140,
        colour="green",
        disable=(not _should_enable_tqdm()),
    )

    mode = str(getattr(cfg, "mode", "finetune") or "finetune").strip().lower()
    if mode not in {"pretrain", "finetune"}:
        mode = "finetune"

    ser_w_neg = float(getattr(cfg, "ser_w_neg", 0.5))
    ser_neg_samples = int(getattr(cfg, "ser_neg_samples", 128))
    ser_neg_temp = float(getattr(cfg, "ser_neg_temp", 0.1))

    smooth_scale = float(getattr(cfg, "smooth_scale_init", 1.0))
    smooth_ma_recon = None
    smooth_ma_smooth = None
    step = 0

    last_good_ep = 0
    last_loss: Optional[float] = None
    last_good_ckpt: Optional[Path] = None

    for ep in pbar:
        encoder.train()
        decoder.train()
        prototypes.train()
        opt.zero_grad(set_to_none=True)

        x_m, mask = mask_gene_blocks(x, float(cfg.mask_ratio))

        w_contrast_now, lam_ser_now = _schedule_weights(
            ep=int(ep),
            total_epochs=int(cfg.epochs),
            w_contrast=float(cfg.w_contrast),
            lam_ser=float(cfg.lam_ser),
        )

        with torch.autocast(device_type=device_type, enabled=use_amp):
            hs, ha = encoder.encode_streams(x_m, edge_s, edge_a)
            z, z_raw = encoder.fuse_streams_raw(hs, ha)

            x_hat = decoder(z_raw)

            l_recon = masked_recon_loss(x_true=x, x_pred=x_hat, mask=mask)
            l_spatial = spatial_neighbor_recon_loss(
                x_hat_center=x_hat,
                x_true=x,
                mask=mask,
                edge_spatial=edge_s,
                max_edges=int(cfg.spatial_max_edges),
            )

            l_smooth_raw = spatial_smoothness_loss(z, edge_s)

            if bool(getattr(cfg, "smooth_auto", True)):
                recon_val = float(l_recon.detach().item())
                smooth_val = float(l_smooth_raw.detach().item())

                if smooth_ma_recon is None:
                    smooth_ma_recon = recon_val
                    smooth_ma_smooth = max(smooth_val, 1e-12)
                else:
                    beta = 0.9
                    smooth_ma_recon = beta * smooth_ma_recon + (1.0 - beta) * recon_val
                    smooth_ma_smooth = beta * smooth_ma_smooth + (1.0 - beta) * max(smooth_val, 1e-12)

                update_every = max(1, int(getattr(cfg, "smooth_update_every", 25)))
                if (step % update_every) == 0:
                    target_ratio = float(getattr(cfg, "smooth_target_ratio", 0.08))
                    desired = target_ratio * float(smooth_ma_recon)
                    denom = float(getattr(cfg, "w_spatial_smooth", 0.5)) * float(smooth_ma_smooth)
                    new_scale = desired / max(denom, 1e-12)

                    lo, hi = getattr(cfg, "smooth_scale_clip", (1e-4, 1e4))
                    lo, hi = float(lo), float(hi)
                    if hi < lo:
                        lo, hi = hi, lo
                    smooth_scale = float(max(lo, min(hi, new_scale)))

            l_smooth = l_smooth_raw * smooth_scale

            e_sem = torch.zeros((), device=device, dtype=z.dtype)
            if lam_ser_now > 0.0 and covered_n > 0:
                P = prototypes()

                cov = coverage_mask.to(dtype=z.dtype, device=z.device)
                strength = c.sum(dim=1).to(dtype=z.dtype)
                
                if bool((cov > 0).any()):
                    s_med = torch.median(strength[cov > 0])
                else:
                    s_med = torch.tensor(1.0, device=z.device, dtype=z.dtype)
                
                s_med = torch.clamp(s_med, min=1e-6)

                gate = torch.clamp(strength / (2.0 * s_med), 0.0, 1.0) * cov
                c_eff = c * gate.unsqueeze(1)

                e_sem = semantic_energy(
                    z=z,
                    c=c_eff,
                    P=P,
                    w_proto=float(cfg.ser_w_proto),
                    valid_mask=coverage_mask,
                    w_neg=float(ser_w_neg),
                    neg_samples=int(ser_neg_samples),
                    neg_temperature=float(ser_neg_temp),
                )

            l_contrast = torch.zeros((), device=device, dtype=z.dtype)
            if ha is not None and w_contrast_now > 0.0:
                hs_p, ha_p = encoder.project_contrastive(hs, ha)
                scale = encoder.get_contrastive_scale(max_scale=50.0)
                l_contrast = cross_view_infonce(hs_p, ha_p, scale=scale)

            loss = (
                float(cfg.w_recon) * l_recon
                + float(cfg.w_spatial_pred) * l_spatial
                + float(getattr(cfg, "w_spatial_smooth", 0.0)) * l_smooth
                + float(lam_ser_now) * e_sem
                + float(w_contrast_now) * l_contrast
            )

        if not torch.isfinite(loss):
            dump = {
                "epoch": int(ep),
                "l_recon": float(l_recon.detach().cpu().item()),
                "l_spatial": float(l_spatial.detach().cpu().item()),
                "l_smooth_raw": float(l_smooth_raw.detach().cpu().item()),
                "smooth_scale": float(smooth_scale),
                "l_contrast": float(l_contrast.detach().cpu().item()),
                "e_sem": float(e_sem.detach().cpu().item()),
                "w_contrast_now": float(w_contrast_now),
                "lam_ser_now": float(lam_ser_now),
            }
            (cfg.out_dir / "nan_dump.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")

            if last_good_ckpt is not None and last_good_ckpt.exists():
                obj = torch.load(str(last_good_ckpt), map_location=device, weights_only=False) 
                encoder.load_state_dict(obj["encoder"], strict=False)
                decoder.load_state_dict(obj["decoder"], strict=False)
                prototypes.load_state_dict(obj.get("prototypes", {}), strict=False)
                for g in opt.param_groups:
                    g["lr"] = float(g["lr"]) * 0.5
                step += 1
                continue
            break

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(all_params, float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, float(cfg.grad_clip))
            opt.step()

        step += 1
        last_loss = float(loss.detach().item())

        actual_log_every = max(int(cfg.log_every), int(cfg.epochs) // 20, 1)
        if ep % actual_log_every == 0 or ep == 1 or ep == int(cfg.epochs):
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                recon=f"{l_recon.item():.3f}",
                ctr=f"{l_contrast.item():.3f}",
                ser=f"{e_sem.item():.3f}",
            )

            last_good_ep = int(ep)
            last_good_ckpt = cfg.out_dir / "plm_ckpt_last_good.pt"
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "prototypes": prototypes.state_dict(),
                    "token_vocab": ser_obj.token_vocab,
                    "config": cfg.__dict__,
                    "last_metrics": {},
                },
                last_good_ckpt,
            )

    encoder.eval()
    decoder.eval()
    prototypes.eval()

    with torch.no_grad():
        hs_eval, ha_eval = encoder.encode_streams(x, edge_s, edge_a)
        z_eval, z_eval_raw = encoder.fuse_streams_raw(hs_eval, ha_eval)
        x_hat_eval = decoder(z_eval_raw)
        recon_cell = F.mse_loss(x_hat_eval, x, reduction="none").mean(dim=1).cpu()

        if covered_n > 0:
            P_eval = prototypes()
            target_eval = c @ P_eval
            ser_cell = 1.0 - torch.cosine_similarity(z_eval, target_eval, dim=1)
            ser_cell = ser_cell.masked_fill(~coverage_mask, float("nan")).cpu()
        else:
            ser_cell = torch.full((int(x.size(0)),), float("nan"), dtype=torch.float32).cpu()

        last_metrics = {
            "recon_err_cell": recon_cell.numpy(),
            "ser_energy_cell": ser_cell.numpy(),
        }

    out_ckpt = _save_ckpt(
        cfg=cfg,
        encoder=encoder,
        decoder=decoder,
        prototypes=prototypes,
        token_vocab=ser_obj.token_vocab,
        last_metrics=last_metrics,
        last_good_ep=last_good_ep,
        last_loss=last_loss,
    )

    report = {
        "version": 4,
        "final_loss": float(last_loss) if last_loss is not None else None,
        "ser_coverage": f"{covered_n}/{total_n}",
        "mode": mode,
    }
    (cfg.out_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] saved:", out_ckpt, flush=True)
    return out_ckpt


# =========================
# Inference/export (merged)
# =========================

def export_embeddings(cfg: PLMConfig, ckpt_path: Path) -> Path:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    obj = torch.load(str(ckpt_path), map_location=device, weights_only=False) 

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
        d_in=int(x.size(1)),
        d_hid=int(cfg.d_hid),
        d_out=int(cfg.d_out),
        n_layers=int(cfg.n_layers),
        dropout=float(cfg.dropout),
        global_attn=bool(cfg.global_attn),
        global_attn_heads=int(cfg.global_attn_heads),
        global_attn_chunk_q=int(cfg.global_attn_chunk_q),
        global_attn_max_n=int(cfg.global_attn_max_n),
        global_attn_dropout=float(cfg.global_attn_dropout),
        contrastive_init_temp=float(cfg.contrast_temp),
        learnable_contrastive_temp=True,
    ).to(device)

    encoder.load_state_dict(obj["encoder"], strict=False)
    encoder.eval()

    with torch.no_grad():
        z = encoder(x, edge_s, edge_a).detach().cpu().numpy()

    adata = sc.read_h5ad(str(cfg.h5ad_path))
    adata.obsm[cfg.emb_key] = z.astype(np.float32)

    last_metrics = obj.get("last_metrics", {})
    if "recon_err_cell" in last_metrics:
        adata.obs["plm_recon_err"] = last_metrics["recon_err_cell"]
    if "ser_energy_cell" in last_metrics:
        adata.obs["plm_ser_energy"] = last_metrics["ser_energy_cell"]

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_h5ad = cfg.out_dir / "plm_embedded.h5ad"
    if bool(getattr(cfg, "save_h5ad", True)):
        adata.write_h5ad(str(out_h5ad))
        print("[OK] saved:", out_h5ad, flush=True)

    return out_h5ad
