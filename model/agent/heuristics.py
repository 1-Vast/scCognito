from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scanpy as sc


def analyze_embedded_h5ad(
    embedded_h5ad: str,
    out_dir: str,
    topk: int = 200,
    thresholds: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Read embedded h5ad and produce:
      - summary quantiles
      - top uncertain indices
      - heuristic suggestions for next run

    IMPORTANT:
      - NEVER raise hard exceptions for missing file/missing columns.
      - Return structured error JSON for LLM to diagnose (e.g., pipeline crashed / OOM).
    """
    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    emb_h5ad = Path(embedded_h5ad).resolve()
    if not emb_h5ad.exists():
        payload = {
            "ok": False,
            "error": "FileNotExists",
            "message": (
                f"Embedded h5ad not found at {str(emb_h5ad)}. "
                "The training pipeline might have crashed (e.g., OOM/NaN). "
                "Please inspect pipeline stderr_tail and adjust parameters safely."
            ),
            "suggestions": {
                "main.py": {},
                "bridge": {},
                "notes": ["No embedded output. Use pipeline stderr_tail as primary evidence."],
            },
        }
        out_json = out_dir_p / "agent_suggestions.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"out_json": str(out_json), "payload": payload}

    # Try reading h5ad safely
    try:
        adata = sc.read_h5ad(str(emb_h5ad))
    except Exception as e:
        payload = {
            "ok": False,
            "error": "ReadH5ADFailed",
            "message": f"Failed to read embedded h5ad: {e}",
            "suggestions": {
                "main.py": {},
                "bridge": {},
                "notes": ["Embedded file exists but cannot be read. Check file integrity / write interruption."],
            },
        }
        out_json = out_dir_p / "agent_suggestions.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"out_json": str(out_json), "payload": payload}

    required = ["plm_recon_err", "plm_ser_energy", "ser_coverage", "ser_strength"]
    missing = [k for k in required if k not in adata.obs.columns]
    if missing:
        payload = {
            "ok": False,
            "error": "MissingObsMetrics",
            "message": f"Missing obs metrics: {missing}. Ensure PLM export writes these into adata.obs.",
            "suggestions": {
                "main.py": {},
                "bridge": {},
                "notes": ["Cannot compute uncertainty without required metrics."],
            },
        }
        out_json = out_dir_p / "agent_suggestions.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"out_json": str(out_json), "payload": payload}

    recon = adata.obs["plm_recon_err"].to_numpy(dtype=np.float32)
    ser_e = adata.obs["plm_ser_energy"].to_numpy(dtype=np.float32)
    cov = adata.obs["ser_coverage"].to_numpy(dtype=np.int32)
    strength = adata.obs["ser_strength"].to_numpy(dtype=np.float32)

    N = int(adata.n_obs)
    topk = int(min(int(topk), N))

    def zscore(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return (x - x.mean()) / (x.std() + 1e-8)

    uncertainty = (
        1.0 * zscore(recon) +
        1.0 * zscore(ser_e) +
        0.5 * (-zscore(cov.astype(np.float32))) +
        0.5 * (-zscore(strength))
    )
    idx_uncertain = np.argsort(-uncertainty)[:topk]

    summary = {
        "N": N,
        "uncertainty_quantiles": {
            "q50": float(np.quantile(uncertainty, 0.50)),
            "q80": float(np.quantile(uncertainty, 0.80)),
            "q90": float(np.quantile(uncertainty, 0.90)),
            "q95": float(np.quantile(uncertainty, 0.95)),
            "q99": float(np.quantile(uncertainty, 0.99)),
        },
        "recon_err": {
            "mean": float(recon.mean()),
            "p95": float(np.quantile(recon, 0.95)),
            "p99": float(np.quantile(recon, 0.99)),
        },
        "ser_energy": {
            "mean": float(ser_e.mean()),
            "p95": float(np.quantile(ser_e, 0.95)),
            "p99": float(np.quantile(ser_e, 0.99)),
        },
        "coverage": {
            "mean": float(cov.mean()),
            "p10": float(np.quantile(cov, 0.10)),
        },
        "strength": {
            "mean": float(strength.mean()),
            "p10": float(np.quantile(strength, 0.10)),
        },
    }

    th = thresholds or {}

    def _get(key: str, default: Any) -> Any:
        return th.get(key, default)

    recon_p99_factor = float(_get("recon_p99_factor", 5.0))
    ser_p99_factor = float(_get("ser_p99_factor", 4.0))
    cov_p10_min = int(_get("cov_p10_min", 1))
    strength_p10_min = float(_get("strength_p10_min", 0.30))

    default_lam_ser = float(_get("default_lam_ser", 0.50))
    default_mask_ratio = float(_get("default_mask_ratio", 0.30))
    default_spatial_k = int(_get("default_spatial_k", 12))
    default_attr_k = int(_get("default_attr_k", 12))
    default_conf_floor = float(_get("default_conf_floor", 0.60))
    low_conf_floor = float(_get("low_conf_floor", 0.50))

    recon_mask_ratio = float(_get("recon_mask_ratio", 0.25))
    recon_lam_ser = float(_get("recon_lam_ser", 0.40))
    ser_lam_ser_cap = float(_get("ser_lam_ser_cap", 0.35))

    recon_p99 = float(summary["recon_err"]["p99"])
    recon_mean = float(summary["recon_err"]["mean"])
    ser_p99 = float(summary["ser_energy"]["p99"])
    ser_mean = float(summary["ser_energy"]["mean"])
    cov_p10 = float(summary["coverage"]["p10"])
    str_p10 = float(summary["strength"]["p10"])

    suggestions = {
        "main.py": {
            "lam_ser": default_lam_ser,
            "mask_ratio": default_mask_ratio,
            "spatial_k": default_spatial_k,
            "attr_k": default_attr_k,
        },
        "bridge": {
            "conf_floor": default_conf_floor,
            "rerun_teacher": False,
        },
        "notes": [],
    }

    if cov_p10 <= cov_p10_min or str_p10 < strength_p10_min:
        suggestions["bridge"]["conf_floor"] = low_conf_floor
        suggestions["bridge"]["rerun_teacher"] = True
        suggestions["notes"].append(
            f"Low token coverage/strength detected (cov_p10={cov_p10:.2f}, strength_p10={str_p10:.3f}). "
            "Lower conf_floor and rerun teacher to improve semantic signals."
        )

    if recon_mean > 0 and recon_p99 > recon_mean * recon_p99_factor:
        suggestions["main.py"]["mask_ratio"] = recon_mask_ratio
        suggestions["main.py"]["lam_ser"] = min(suggestions["main.py"]["lam_ser"], recon_lam_ser)
        suggestions["notes"].append(
            f"Recon anomaly: recon_p99={recon_p99:.4g} > mean*{recon_p99_factor} (mean={recon_mean:.4g}). "
            "Reduce mask_ratio and lam_ser."
        )

    if ser_mean > 0 and ser_p99 > ser_mean * ser_p99_factor:
        suggestions["main.py"]["lam_ser"] = min(suggestions["main.py"]["lam_ser"], ser_lam_ser_cap)
        suggestions["notes"].append(
            f"SER anomaly: ser_p99={ser_p99:.4g} > mean*{ser_p99_factor} (mean={ser_mean:.4g}). "
            "Cap lam_ser."
        )

    payload = {
        "ok": True,
        "summary": summary,
        "top_uncertain_indices": idx_uncertain[:50].tolist(),
        "suggestions": suggestions,
        "thresholds": {
            "recon_p99_factor": recon_p99_factor,
            "ser_p99_factor": ser_p99_factor,
            "cov_p10_min": cov_p10_min,
            "strength_p10_min": strength_p10_min,
        },
    }

    out_json = out_dir_p / "agent_suggestions.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"out_json": str(out_json), "payload": payload}