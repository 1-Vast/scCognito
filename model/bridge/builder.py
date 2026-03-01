from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import scanpy as sc

from .config import BridgeConfig
from .io import load_teacher_tokens, save_ser_pt


def _collect_token_vocab(teacher: Dict[str, Any]) -> List[str]:
    vocab, seen = [], set()
    for _, info in teacher.items():
        pcs = info.get("constraints", {}).get("prototype_constraints", [])
        for pc in pcs:
            anchor = str(pc.get("anchor", "")).strip()
            if not anchor:
                continue
            if anchor not in seen:
                seen.add(anchor)
                vocab.append(anchor)
    return vocab


def _build_cluster_anchor_map(
    teacher: Dict[str, Any],
    token_to_idx: Dict[str, int],
    conf_floor: float,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    cluster_name -> [(token_idx, confidence), ...]
    """
    out: Dict[str, List[Tuple[int, float]]] = {}
    for cluster_name, info in teacher.items():
        pcs = info.get("constraints", {}).get("prototype_constraints", [])
        anchors: List[Tuple[int, float]] = []
        for pc in pcs:
            conf = float(pc.get("confidence", 0.0))
            if conf < conf_floor:
                continue
            anchor = str(pc.get("anchor", "")).strip()
            if anchor in token_to_idx:
                anchors.append((token_to_idx[anchor], conf))
        out[str(cluster_name)] = anchors
    return out


def build_ser_signals_from_teacher_json(
    json_path: Path,
    h5ad_path: Path,
    cluster_key: str,
    conf_floor: float = 0.6,
    normalize_cluster_weights: bool = True,
) -> Dict[str, Any]:
    teacher = load_teacher_tokens(json_path)

    adata = sc.read_h5ad(h5ad_path)
    if cluster_key not in adata.obs:
        raise KeyError(f"cluster_key '{cluster_key}' not found in adata.obs")

    token_vocab = _collect_token_vocab(teacher)
    token_to_idx = {t: i for i, t in enumerate(token_vocab)}
    cluster_anchor_map = _build_cluster_anchor_map(teacher, token_to_idx, conf_floor)

    N = adata.n_obs
    K = len(token_vocab)
    c = torch.zeros((N, K), dtype=torch.float32)

    cluster_vals = adata.obs[cluster_key].astype(str).tolist()

    for i, cv in enumerate(cluster_vals):
        anchors = None

        if cv in cluster_anchor_map:
            anchors = cluster_anchor_map[cv]
        else:
            alt = f"{cluster_key}:{cv}"
            anchors = cluster_anchor_map.get(alt, [])

        if not anchors:
            continue

        weights = torch.tensor([w for _, w in anchors], dtype=torch.float32)
        if normalize_cluster_weights:
            weights = weights / (weights.sum() + 1e-12)

        for (tid, _), w in zip(anchors, weights):
            c[i, tid] = w

    return {
        "token_vocab": token_vocab,
        "c": c,
        "cluster_key": cluster_key,
        "conf_floor": conf_floor,
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
        )
        out_path = cfg.out_dir / f"{jf.stem}_ser_signals.pt"
        save_ser_pt(out_path, payload)
        out_paths.append(out_path)

    return out_paths
