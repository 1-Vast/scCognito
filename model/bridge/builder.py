from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import scanpy as sc
import torch

from .config import BridgeConfig
from .io import load_teacher_tokens, save_ser_pt


def _extract_cluster_tokens_from_teacher(
    teacher: Dict[str, Any],
    conf_floor: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse cluster tokens from teacher JSON.

    Compatible with current teacher JSON schema:
      info["constraints"]["prototype_constraints"]
    """
    out: Dict[str, List[Dict[str, Any]]] = {}

    for cluster_name, info in teacher.items():
        pcs = (info.get("constraints", {}) or {}).get("prototype_constraints", []) or []
        items: List[Dict[str, Any]] = []
        for pc in pcs:
            conf = float(pc.get("confidence", 0.0))
            if conf < float(conf_floor):
                continue
            anchor = str(pc.get("anchor", "")).strip()
            if not anchor:
                continue
            items.append(
                {
                    "id": anchor,
                    "text": anchor,
                    "conf": conf,
                    "type": "anchor",
                }
            )

        out[str(cluster_name)] = items

    return out


def _build_global_vocab(cluster_tokens: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    vocab: List[str] = []
    meta: List[Dict[str, Any]] = []
    seen = set()

    for _, items in cluster_tokens.items():
        for it in items:
            tid = str(it["id"])
            if tid in seen:
                continue
            seen.add(tid)
            vocab.append(tid)
            meta.append(
                {
                    "id": tid,
                    "text": str(it.get("text", tid)),
                    "type": str(it.get("type", "token")),
                }
            )
    return vocab, meta


def _topm_and_softmax_weights(
    items: List[Dict[str, Any]],
    top_m: int,
    temp: float,
) -> List[Tuple[str, float]]:
    if not items:
        return []

    items = sorted(items, key=lambda x: float(x.get("conf", 0.0)), reverse=True)
    items = items[: max(1, int(top_m))]

    conf = torch.tensor([float(x.get("conf", 0.0)) for x in items], dtype=torch.float32)
    t = max(1e-6, float(temp))
    w = torch.softmax(conf / t, dim=0)

    out: List[Tuple[str, float]] = []
    for it, wi in zip(items, w.tolist()):
        out.append((str(it["id"]), float(wi)))
    return out


def build_ser_signals_from_teacher_json(
    json_path: Path,
    h5ad_path: Path,
    cluster_key: str,
    conf_floor: float = 0.55,
    normalize_cluster_weights: bool = True,
    top_m_per_cluster: int = 12,
    softmax_temp: float = 0.25,
) -> Dict[str, Any]:
    teacher = load_teacher_tokens(json_path)

    adata = sc.read_h5ad(h5ad_path)
    if cluster_key not in adata.obs:
        raise KeyError(f"cluster_key '{cluster_key}' not found in adata.obs")

    cluster_tokens = _extract_cluster_tokens_from_teacher(teacher, conf_floor=conf_floor)

    token_vocab, token_meta = _build_global_vocab(cluster_tokens)
    token_to_idx = {t: i for i, t in enumerate(token_vocab)}
    token_texts = [m.get("text", m.get("id", "")) for m in token_meta]

    cluster_weight_map: Dict[str, List[Tuple[str, float]]] = {}
    for cname, items in cluster_tokens.items():
        cluster_weight_map[cname] = _topm_and_softmax_weights(
            items,
            top_m=top_m_per_cluster,
            temp=softmax_temp,
        )

    N = adata.n_obs
    K = len(token_vocab)
    c = torch.zeros((N, K), dtype=torch.float32)

    cluster_vals = adata.obs[cluster_key].astype(str).tolist()
    for i, cv in enumerate(cluster_vals):
        items = cluster_weight_map.get(cv, None)
        if items is None:
            items = cluster_weight_map.get(f"{cluster_key}:{cv}", [])

        if not items:
            continue

        if normalize_cluster_weights:
            s = sum(w for _, w in items)
            if s > 0:
                items = [(tid, w / s) for tid, w in items]

        for tid, w in items:
            j = token_to_idx.get(tid, None)
            if j is not None:
                c[i, j] = float(w)

    return {
        "token_vocab": token_vocab,
        "token_texts": token_texts,
        "token_meta": token_meta,
        "c": c,
        "cluster_key": cluster_key,
        "conf_floor": float(conf_floor),
        "top_m_per_cluster": int(top_m_per_cluster),
        "softmax_temp": float(softmax_temp),
        "source_json": str(json_path),
        "source_h5ad": str(h5ad_path),
    }


def run_bridge(
    cfg: BridgeConfig,
    h5ad_path: Path | None = None,
    cluster_key: str | None = None,
) -> List[Path]:
    h5ad_path = h5ad_path or cfg.default_h5ad
    cluster_key = cluster_key or cfg.default_cluster_key

    if not cfg.token_dir.exists():
        raise RuntimeError(f"Missing token_dir: {cfg.token_dir}")

    json_files = sorted(cfg.token_dir.rglob("*.json"))
    if not json_files:
        raise RuntimeError(f"No *.json found in {cfg.token_dir}")

    out_paths: List[Path] = []
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    for jf in json_files:
        payload = build_ser_signals_from_teacher_json(
            json_path=jf,
            h5ad_path=h5ad_path,
            cluster_key=cluster_key,
            conf_floor=cfg.conf_floor,
            normalize_cluster_weights=cfg.normalize_cluster_weights,
            top_m_per_cluster=cfg.top_m_per_cluster,
            softmax_temp=cfg.softmax_temp,
        )
        out_path = cfg.out_dir / f"{jf.stem}_ser_signals.pt"
        save_ser_pt(out_path, payload)
        out_paths.append(out_path)

    return out_paths
