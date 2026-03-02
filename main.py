from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from model.bridge.builder import build_ser_signals_from_teacher_json
from model.bridge.io import save_ser_pt
from model.plm.config import PLMConfig
from model.plm.train import run_train
from model.plm.infer import export_embeddings


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _with_model_pythonpath(env: dict[str, str], root: Path) -> dict[str, str]:
    out = dict(env)
    py_model = str((root / "model").resolve())
    if out.get("PYTHONPATH"):
        out["PYTHONPATH"] = py_model + os.pathsep + out["PYTHONPATH"]
    else:
        out["PYTHONPATH"] = py_model
    return out


def _find_latest_tokens_json(out_root: Path) -> Optional[Path]:
    patterns = ("*llm_teacher_tokens.json", "*teacher_tokens*.json", "*tokens*.json")
    hits: list[Path] = []
    for ptn in patterns:
        hits.extend(out_root.rglob(ptn))
    hits = [p for p in hits if p.is_file()]
    if not hits:
        return None
    return sorted(hits, key=lambda p: p.stat().st_mtime)[-1]


def _infer_cluster_key_from_teacher_json(token_json: Path, fallback: str) -> str:
    try:
        obj = json.loads(token_json.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for _, info in obj.items():
                if not isinstance(info, dict):
                    continue
                cluster = str(info.get("cluster", ""))
                if ":" in cluster:
                    key = cluster.split(":", 1)[0].strip()
                    if key:
                        return key
    except Exception:
        pass
    return fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _copy_as_separate_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _run_teacher(
    h5ad: Path,
    out_dir: Path,
    groupby: str,
    teacher_cmd: str = "",
    model_id: str = "",
    knowledge_root: str = "",
    max_llm_calls: int = 24,
    privacy_guard: bool = True,
    debug_checks: bool = False,
) -> Path:
    root = _project_root()
    env = _with_model_pythonpath(os.environ, root)

    if teacher_cmd.strip():
        completed = subprocess.run(
            teacher_cmd,
            shell=True,
            cwd=str(root),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
    else:
        cmd = [
            sys.executable,
            "-m",
            "teacher.cli",
            "run",
            "--h5ad",
            str(h5ad.resolve()),
            "--out-dir",
            str(out_dir.resolve()),
            "--groupby",
            groupby,
            "--max-llm-calls",
            str(max(1, int(max_llm_calls))),
        ]
        if model_id.strip():
            cmd += ["--model-id", model_id.strip()]
        if knowledge_root.strip():
            cmd += ["--knowledge-root", knowledge_root.strip()]
        if not privacy_guard:
            cmd.append("--disable-privacy-guard")
        if debug_checks:
            cmd.append("--debug-checks")
        subprocess.run(cmd, cwd=str(root), env=env, check=True)

    found = _find_latest_tokens_json(out_dir)
    if found is None:
        raise RuntimeError(f"No teacher tokens json found under: {out_dir}")
    return found


def _build_ser(
    token_json: Path,
    h5ad: Path,
    cluster_key: str,
    out_dir: Path,
    conf_floor: float,
) -> Path:
    payload = build_ser_signals_from_teacher_json(
        json_path=token_json,
        h5ad_path=h5ad,
        cluster_key=cluster_key,
        conf_floor=conf_floor,
        normalize_cluster_weights=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / f"{token_json.stem}_ser_signals.pt"
    save_ser_pt(out_pt, payload)
    return out_pt


def _run_debug_checks(h5ad_path: Path, ser_pt: Path, embedded_h5ad: Path, run_tag: str, artifacts_dir: Path) -> Path:
    import torch
    import scanpy as sc

    report: dict[str, Any] = {
        "run_tag": run_tag,
        "checks": {},
        "errors": [],
    }

    try:
        adata = sc.read_h5ad(str(h5ad_path))
        report["checks"]["h5ad_n_obs"] = int(adata.n_obs)
        report["checks"]["h5ad_n_vars"] = int(adata.n_vars)
    except Exception as e:
        report["errors"].append(f"read_input_h5ad_failed: {e}")
        adata = None

    try:
        ser = torch.load(str(ser_pt), map_location="cpu")
        c = ser["c"]
        report["checks"]["ser_shape"] = [int(c.shape[0]), int(c.shape[1])]
        if adata is not None and int(c.shape[0]) != int(adata.n_obs):
            report["errors"].append("ser_row_count_mismatch_with_input_h5ad")
    except Exception as e:
        report["errors"].append(f"read_ser_failed: {e}")

    try:
        emb = sc.read_h5ad(str(embedded_h5ad))
        report["checks"]["embedded_n_obs"] = int(emb.n_obs)
        report["checks"]["embedded_obsm_keys"] = list(emb.obsm.keys())
        required_obs = ["plm_recon_err", "plm_ser_energy", "ser_coverage", "ser_strength"]
        missing = [k for k in required_obs if k not in emb.obs.columns]
        report["checks"]["embedded_missing_obs_metrics"] = missing
        if missing:
            report["errors"].append("embedded_missing_required_obs_metrics")
    except Exception as e:
        report["errors"].append(f"read_embedded_failed: {e}")

    report["ok"] = len(report["errors"]) == 0
    out = artifacts_dir / f"{run_tag}_debug_checks.json"
    _write_json(out, report)
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="main", description="Teacher -> Bridge -> PLM orchestrator")
    ap.add_argument("--h5ad", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--groupby", type=str, default="leiden")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--skip_teacher", action="store_true")
    ap.add_argument("--teacher_cmd", type=str, default="")
    ap.add_argument("--prefer_ser", type=str, default="")
    ap.add_argument("--teacher_model_id", type=str, default="")
    ap.add_argument("--knowledge_root", type=str, default="")
    ap.add_argument("--max_llm_calls", type=int, default=24)
    ap.add_argument("--disable_privacy_guard", action="store_true")

    ap.add_argument("--lam_ser", type=float, default=1.0)
    ap.add_argument("--lam_ser_warmup_ratio", type=float, default=0.15)
    ap.add_argument("--mode", type=str, default="finetune")
    ap.add_argument("--mask_ratio", type=float, default=0.25)
    ap.add_argument("--spatial_k", type=int, default=12)
    ap.add_argument("--attr_k", type=int, default=12)
    ap.add_argument("--conf_floor", type=float, default=0.6)

    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--d_hid", type=int, default=512)
    ap.add_argument("--d_out", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)

    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--w_recon", type=float, default=1.0)
    ap.add_argument("--w_spatial_pred", type=float, default=1.0)
    ap.add_argument("--w_spatial_smooth", type=float, default=0.5)
    ap.add_argument("--w_contrast", type=float, default=0.0)
    ap.add_argument("--ser_w_proto", type=float, default=1.0)

    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--debug_checks", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    h5ad = Path(args.h5ad).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    artifacts_dir = out_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    teacher_out = out_root / "teacher_outputs"
    ser_out = out_root / "ser_outputs"
    plm_out = out_root / "plm_outputs"

    manifest: dict[str, Any] = {
        "run_tag": run_tag,
        "input": {
            "h5ad": str(h5ad),
            "out_root": str(out_root),
            "groupby": args.groupby,
            "device": args.device,
            "skip_teacher": bool(args.skip_teacher),
            "self_supervised": True,
            "privacy_guard": (not args.disable_privacy_guard),
            "max_llm_calls": int(args.max_llm_calls),
            "train_cfg": {
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "d_hid": int(args.d_hid),
                "d_out": int(args.d_out),
                "n_layers": int(args.n_layers),
                "dropout": float(args.dropout),
                "grad_clip": float(args.grad_clip),
                "log_every": int(args.log_every),
                "w_recon": float(args.w_recon),
                "w_spatial_pred": float(args.w_spatial_pred),
                "w_spatial_smooth": float(args.w_spatial_smooth),
                "lam_ser": float(args.lam_ser),
                "lam_ser_warmup_ratio": float(args.lam_ser_warmup_ratio),
                "w_contrast": float(args.w_contrast),
                "mode": str(args.mode),
                "ser_w_proto": float(args.ser_w_proto),
            },
        },
        "stages": {},
    }
    print("[PROGRESS][PIPELINE] stage=init pct=0.0", flush=True)

    token_json: Optional[Path] = None
    if not args.skip_teacher:
        print("[PROGRESS][PIPELINE] stage=teacher pct=5.0", flush=True)
        teacher_out.mkdir(parents=True, exist_ok=True)
        token_json = _run_teacher(
            h5ad=h5ad,
            out_dir=teacher_out,
            groupby=args.groupby,
            teacher_cmd=args.teacher_cmd,
            model_id=args.teacher_model_id,
            knowledge_root=args.knowledge_root,
            max_llm_calls=int(args.max_llm_calls),
            privacy_guard=(not args.disable_privacy_guard),
            debug_checks=bool(args.debug_checks),
        )
        print(f"[OK] teacher tokens: {token_json}")
    else:
        found = _find_latest_tokens_json(out_root)
        if found is None:
            raise RuntimeError("skip_teacher=True but no token json found under out_root")
        token_json = found
        print(f"[SKIP] teacher; using existing tokens: {token_json}")
    print("[PROGRESS][PIPELINE] stage=teacher_done pct=30.0", flush=True)

    effective_cluster_key = _infer_cluster_key_from_teacher_json(token_json, args.groupby)

    stage1 = {
        "stage": "teacher",
        "token_json": str(token_json),
        "max_llm_calls": int(args.max_llm_calls),
        "privacy_guard": (not args.disable_privacy_guard),
        "effective_cluster_key": effective_cluster_key,
    }
    stage1_path = artifacts_dir / f"{run_tag}_stage1_teacher.json"
    _write_json(stage1_path, stage1)
    manifest["stages"]["teacher"] = str(stage1_path)

    ser_pt: Optional[Path] = None
    if args.prefer_ser.strip():
        p = Path(args.prefer_ser).resolve()
        if p.exists() and p.suffix.lower() == ".pt":
            ser_pt = p
            print(f"[OK] using prefer_ser: {ser_pt}")
    if ser_pt is None:
        print("[PROGRESS][PIPELINE] stage=bridge pct=35.0", flush=True)
        assert token_json is not None
        ser_pt = _build_ser(
            token_json=token_json,
            h5ad=h5ad,
            cluster_key=effective_cluster_key,
            out_dir=ser_out,
            conf_floor=float(args.conf_floor),
        )
        print(f"[OK] ser signals: {ser_pt}")
    print("[PROGRESS][PIPELINE] stage=bridge_done pct=45.0", flush=True)

    stage2 = {
        "stage": "bridge",
        "ser_pt": str(ser_pt),
        "conf_floor": float(args.conf_floor),
        "cluster_key": effective_cluster_key,
    }
    stage2_path = artifacts_dir / f"{run_tag}_stage2_bridge.json"
    _write_json(stage2_path, stage2)
    manifest["stages"]["bridge"] = str(stage2_path)

    cfg = PLMConfig(
        h5ad_path=h5ad,
        ser_pt_path=ser_pt,
        out_dir=plm_out,
        device=args.device,
        mode=str(args.mode).strip().lower(),
        lam_ser=float(args.lam_ser),
        lam_ser_warmup_ratio=max(0.0, min(1.0, float(args.lam_ser_warmup_ratio))),
        mask_ratio=float(args.mask_ratio),
        spatial_k=int(args.spatial_k),
        attr_k=int(args.attr_k),
        epochs=max(1, int(args.epochs)),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        d_hid=max(16, int(args.d_hid)),
        d_out=max(8, int(args.d_out)),
        n_layers=max(1, int(args.n_layers)),
        dropout=max(0.0, min(0.9, float(args.dropout))),
        grad_clip=max(0.1, float(args.grad_clip)),
        log_every=max(1, int(args.log_every)),
        w_recon=max(0.0, float(args.w_recon)),
        w_spatial_pred=max(0.0, float(args.w_spatial_pred)),
        w_spatial_smooth=max(0.0, float(args.w_spatial_smooth)),
        w_contrast=max(0.0, float(args.w_contrast)),
        ser_w_proto=max(0.0, float(args.ser_w_proto)),
    )
    print("[PROGRESS][PIPELINE] stage=plm_train pct=50.0", flush=True)

    ckpt = run_train(cfg)
    ckpt_unique = plm_out / f"plm_ckpt__{run_tag}.pt"
    if ckpt.resolve() != ckpt_unique.resolve():
        _copy_as_separate_file(ckpt, ckpt_unique)

    stage3 = {
        "stage": "plm_train",
        "ckpt": str(ckpt),
        "ckpt_separate": str(ckpt_unique),
        "config": cfg.__dict__,
    }
    stage3_path = artifacts_dir / f"{run_tag}_stage3_plm_train.json"
    _write_json(stage3_path, stage3)
    manifest["stages"]["plm_train"] = str(stage3_path)
    print("[PROGRESS][PIPELINE] stage=plm_train_done pct=90.0", flush=True)

    print("[PROGRESS][PIPELINE] stage=plm_export pct=92.0", flush=True)
    out_h5ad = export_embeddings(cfg, ckpt)
    emb_unique = plm_out / f"plm_embedded__{run_tag}.h5ad"
    if out_h5ad.resolve() != emb_unique.resolve():
        _copy_as_separate_file(out_h5ad, emb_unique)

    stage4 = {
        "stage": "plm_export",
        "embedded_h5ad": str(out_h5ad),
        "embedded_h5ad_separate": str(emb_unique),
    }
    stage4_path = artifacts_dir / f"{run_tag}_stage4_plm_export.json"
    _write_json(stage4_path, stage4)
    manifest["stages"]["plm_export"] = str(stage4_path)
    print("[PROGRESS][PIPELINE] stage=plm_export_done pct=98.0", flush=True)

    if args.debug_checks:
        debug_path = _run_debug_checks(
            h5ad_path=h5ad,
            ser_pt=ser_pt,
            embedded_h5ad=out_h5ad,
            run_tag=run_tag,
            artifacts_dir=artifacts_dir,
        )
        manifest["debug_checks"] = str(debug_path)

    manifest_path = artifacts_dir / f"{run_tag}_manifest.json"
    latest_manifest = artifacts_dir / "latest_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(latest_manifest, manifest)

    summary = {
        "teacher_tokens_json": str(token_json) if token_json else "",
        "ser_pt": str(ser_pt),
        "ckpt": str(ckpt),
        "embedded_h5ad": str(out_h5ad),
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False))
    print("[PROGRESS][PIPELINE] stage=done pct=100.0", flush=True)


if __name__ == "__main__":
    main()
