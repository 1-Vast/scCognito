from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def _to_dense(x: Any) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _safe_sample_indices(n: int, max_n: int, seed: int = 0) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _safe_silhouette(z: np.ndarray, labels: np.ndarray, max_n: int = 5000) -> float | None:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return None
    idx = _safe_sample_indices(z.shape[0], max_n=max_n, seed=0)
    return float(silhouette_score(z[idx], labels[idx]))


def _spatial_coherence(labels: np.ndarray, spatial: np.ndarray, k: int = 6) -> float | None:
    n = spatial.shape[0]
    if n < 3:
        return None
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(spatial)
    ids = nn.kneighbors(return_distance=False)
    ids = ids[:, 1:]  # remove self
    same = (labels[:, None] == labels[ids]).astype(np.float32)
    return float(np.mean(same))


def _knn_entropy(z: np.ndarray, groups: np.ndarray, k: int = 15) -> float | None:
    uniq = np.unique(groups)
    if len(uniq) < 2 or z.shape[0] < 3:
        return None
    n = z.shape[0]
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(z)
    ids = nn.kneighbors(return_distance=False)[:, 1:]
    mapper = {g: i for i, g in enumerate(uniq)}
    gidx = np.array([mapper[g] for g in groups], dtype=np.int64)
    c = len(uniq)
    ent = []
    for i in range(n):
        bins = np.bincount(gidx[ids[i]], minlength=c).astype(np.float64)
        p = bins / max(1.0, bins.sum())
        p = p[p > 0]
        h = -np.sum(p * np.log(p))
        ent.append(h / np.log(c))
    return float(np.mean(ent))


def _distance_corr(z: np.ndarray, other: np.ndarray, max_n: int = 1500) -> dict[str, float]:
    n = min(z.shape[0], other.shape[0])
    idx = _safe_sample_indices(n, max_n=max_n, seed=1)
    d1 = pdist(z[idx], metric="euclidean")
    d2 = pdist(other[idx], metric="euclidean")
    rho, p = spearmanr(d1, d2)
    return {"rho": float(rho), "pvalue": float(p)}


def _to_numeric_time(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce")
    if np.isfinite(x).sum() >= max(3, len(s) // 2):
        med = np.nanmedian(x.values.astype(float))
        x = np.where(np.isfinite(x), x, med)
        return x.astype(float)
    return s.astype("category").cat.codes.to_numpy(dtype=float)


def _top_markers_by_domain(adata, groupby: str, topk: int = 10) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    try:
        sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", n_genes=topk)
        df = sc.get.rank_genes_groups_df(adata, group=None)
        if "group" not in df.columns or "names" not in df.columns:
            return out
        for g, gdf in df.groupby("group", observed=False):
            genes = [str(x) for x in gdf["names"].head(topk).tolist()]
            out[str(g)] = genes
    except Exception:
        return out
    return out


def _perturb_gene_score(adata, genes: list[str]) -> dict[str, Any]:
    var_index = pd.Index(adata.var_names.astype(str))
    keep = [g for g in genes if g in var_index]
    miss = [g for g in genes if g not in var_index]
    if not keep:
        return {"ok": False, "message": "no requested genes found", "missing": miss}
    cols = [int(var_index.get_loc(g)) for g in keep]
    x = adata.X[:, cols]
    x = _to_dense(x).astype(np.float32)
    score = x.mean(axis=1)
    return {
        "ok": True,
        "genes_used": keep,
        "genes_missing": miss,
        "score_mean": float(np.mean(score)),
        "score_p95": float(np.quantile(score, 0.95)),
        "score_p99": float(np.quantile(score, 0.99)),
        "_score_vec": score.tolist(),
    }


def run_downstream(
    embedded_h5ad: str,
    out_dir: str,
    emb_key: str = "X_plm",
    n_domains: int = 8,
    label_col: str = "",
    batch_col: str = "",
    time_col: str = "",
    perturb_genes: str = "",
    marker_topk: int = 10,
) -> dict[str, Any]:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(Path(embedded_h5ad).resolve()))

    if emb_key in adata.obsm:
        z = np.asarray(adata.obsm[emb_key], dtype=np.float32)
    else:
        z = _to_dense(adata.X).astype(np.float32)
        emb_key = "X"
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    # -------- Task 1: spatial domain identification --------
    n = int(adata.n_obs)
    if n_domains <= 1:
        if label_col and label_col in adata.obs.columns:
            n_domains_eff = int(max(2, adata.obs[label_col].astype(str).nunique()))
        else:
            n_domains_eff = 8
    else:
        n_domains_eff = int(n_domains)
    n_domains_eff = max(2, min(n_domains_eff, max(2, n - 1)))

    km = KMeans(n_clusters=n_domains_eff, random_state=0, n_init=20)
    dom = km.fit_predict(z).astype(int)
    adata.obs["demo_domain"] = pd.Categorical(dom.astype(str))

    domain_metrics: dict[str, Any] = {
        "n_domains": int(n_domains_eff),
        "silhouette": _safe_silhouette(z, dom),
        "davies_bouldin": float(davies_bouldin_score(z, dom)),
    }

    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        domain_metrics["spatial_coherence_k6"] = _spatial_coherence(dom, spatial, k=6)

    if label_col and label_col in adata.obs.columns:
        y = adata.obs[label_col].astype(str).to_numpy()
        domain_metrics["ari_vs_label"] = float(adjusted_rand_score(y, dom.astype(str)))
        domain_metrics["nmi_vs_label"] = float(normalized_mutual_info_score(y, dom.astype(str)))

    # -------- Task 2: cross-modal / cross-representation alignment --------
    align: dict[str, Any] = {"emb_key": emb_key, "pairs": {}}
    for k in sorted(adata.obsm.keys()):
        if k == emb_key:
            continue
        m = np.asarray(adata.obsm[k], dtype=np.float32)
        if m.ndim != 2 or m.shape[0] != z.shape[0] or m.shape[1] < 2:
            continue
        try:
            align["pairs"][k] = _distance_corr(z, np.nan_to_num(m), max_n=1500)
        except Exception:
            continue

    # -------- Task 3: large-scale integration metrics --------
    integration: dict[str, Any] = {}
    if batch_col and batch_col in adata.obs.columns:
        b = adata.obs[batch_col].astype(str).to_numpy()
        integration["batch_col"] = batch_col
        integration["knn_batch_entropy_k15"] = _knn_entropy(z, b, k=15)
        integration["batch_silhouette"] = _safe_silhouette(z, b)

    # -------- Task 4: temporal evolution --------
    temporal: dict[str, Any] = {}
    if time_col and time_col in adata.obs.columns:
        t = _to_numeric_time(adata.obs[time_col])
        pc1 = PCA(n_components=1, random_state=0).fit_transform(z).reshape(-1)
        rho, p = spearmanr(pc1, t)
        temporal = {
            "time_col": time_col,
            "pseudo_time_rho": float(rho),
            "pseudo_time_pvalue": float(p),
        }

    # -------- Task 5: perturbation simulation (simple score proxy) --------
    perturb: dict[str, Any] = {}
    genes = [g.strip() for g in perturb_genes.split(",") if g.strip()]
    if genes:
        perturb = _perturb_gene_score(adata, genes)
        if perturb.get("ok"):
            s = np.asarray(perturb.pop("_score_vec"), dtype=np.float32)
            by_dom = (
                pd.DataFrame({"domain": adata.obs["demo_domain"].astype(str).to_numpy(), "score": s})
                .groupby("domain")["score"]
                .mean()
                .sort_values(ascending=False)
            )
            perturb["top_sensitive_domains"] = by_dom.head(5).to_dict()

    # -------- Task 6: mechanism interpretability --------
    markers = _top_markers_by_domain(adata, groupby="demo_domain", topk=max(3, int(marker_topk)))

    # -------- Task 7/8: cognitive feedback & autonomous suggestions --------
    notes: list[str] = []
    if domain_metrics.get("silhouette") is not None and domain_metrics["silhouette"] < 0.05:
        notes.append("Domain separation is weak. Consider tuning lam_ser/mask_ratio or graph k.")
    if integration.get("knn_batch_entropy_k15") is not None and integration["knn_batch_entropy_k15"] < 0.4:
        notes.append("Batch mixing is low. Consider stronger integration strategy or batch-aware alignment.")
    if temporal and abs(temporal.get("pseudo_time_rho", 0.0)) < 0.2:
        notes.append("Temporal monotonicity is weak. Consider trajectory-specific modeling.")
    if not notes:
        notes.append("Embedding quality is acceptable for downstream exploration.")

    payload = {
        "summary": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "emb_key": emb_key,
        },
        "tasks": {
            "spatial_domain_identification": domain_metrics,
            "cross_modal_alignment": align,
            "large_scale_integration": integration,
            "temporal_evolution": temporal,
            "perturbation_simulation": perturb,
            "mechanistic_interpretability": {
                "marker_topk": int(marker_topk),
                "markers_by_domain": markers,
            },
            "cognitive_feedback_for_experiment_design": {"notes": notes},
            "autonomous_scientific_exploration": {"next_actions": notes},
        },
    }

    metrics_json = out / "downstream_metrics.json"
    metrics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    assign_csv = out / "downstream_domain_assignments.csv"
    pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "demo_domain": adata.obs["demo_domain"].astype(str).to_numpy(),
        }
    ).to_csv(assign_csv, index=False)

    out_h5ad = out / "plm_embedded_with_downstream.h5ad"
    adata.write_h5ad(str(out_h5ad))

    report_html = out / "downstream_report.html"
    report_html.write_text(
        "\n".join(
            [
                "<html><head><meta charset='utf-8'/>",
                "<style>body{font-family:Arial,Helvetica,sans-serif;margin:18px;}pre{background:#f6f6f6;padding:10px;overflow:auto;}</style>",
                "</head><body>",
                "<h2>Downstream Report</h2>",
                f"<p><b>embedded_h5ad:</b> {embedded_h5ad}</p>",
                "<h3>Metrics</h3>",
                "<pre>" + json.dumps(payload, indent=2, ensure_ascii=False) + "</pre>",
                f"<p><b>metrics_json:</b> {metrics_json}</p>",
                f"<p><b>domain_assignments:</b> {assign_csv}</p>",
                f"<p><b>h5ad_with_downstream:</b> {out_h5ad}</p>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "metrics_json": str(metrics_json),
        "report_html": str(report_html),
        "h5ad_with_downstream": str(out_h5ad),
        "domain_assignments": str(assign_csv),
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="demo.run_downstream", description="Run downstream tasks on PLM embedding")
    ap.add_argument("--embedded_h5ad", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--emb_key", type=str, default="X_plm")
    ap.add_argument("--n_domains", type=int, default=8)
    ap.add_argument("--label_col", type=str, default="")
    ap.add_argument("--batch_col", type=str, default="")
    ap.add_argument("--time_col", type=str, default="")
    ap.add_argument("--perturb_genes", type=str, default="")
    ap.add_argument("--marker_topk", type=int, default=10)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    print("[PROGRESS][DOWNSTREAM] stage=start pct=0.0", flush=True)
    out = run_downstream(
        embedded_h5ad=args.embedded_h5ad,
        out_dir=args.out_dir,
        emb_key=args.emb_key,
        n_domains=args.n_domains,
        label_col=args.label_col.strip(),
        batch_col=args.batch_col.strip(),
        time_col=args.time_col.strip(),
        perturb_genes=args.perturb_genes.strip(),
        marker_topk=args.marker_topk,
    )
    print("[PROGRESS][DOWNSTREAM] stage=done pct=100.0", flush=True)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
