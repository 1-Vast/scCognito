from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .autoloop_guard import (
        RoundRecord,
        StateTracker,
        fingerprint,
        loop_breaker,
        regression_score,
        should_escalate,
    )
except ImportError:
    from autoloop_guard import (
        RoundRecord,
        StateTracker,
        fingerprint,
        loop_breaker,
        regression_score,
        should_escalate,
    )

app = FastAPI(title="scAgent Web Portal (Jobs + SSE + AutoLoop Guardrails)")

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
STATIC_DIR = ROOT / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

JOBS: Dict[str, "Job"] = {}
OUTROOT_LOCKS: Dict[str, str] = {}  # out_root -> job_id
CHAT_SESSIONS: Dict[str, "ChatSession"] = {}
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


MATHJAX_SNIPPET = """\
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def _inject_mathjax(html: str) -> str:
    """
    Ensure MathJax exists in the HTML output.
    """
    if "MathJax-script" in html or "mathjax@3" in html.lower():
        return html

    lower = html.lower()
    if "<head" in lower:
        i = lower.find("<head")
        j = lower.find(">", i)
        if j != -1:
            return html[: j + 1] + "\n" + MATHJAX_SNIPPET + "\n" + html[j + 1 :]

    if "<html" in lower:
        # Insert a minimal head if missing
        if "<head" not in lower:
            k = lower.find("<html")
            k2 = lower.find(">", k)
            if k2 != -1:
                return (
                    html[: k2 + 1]
                    + "\n<head>\n"
                    + MATHJAX_SNIPPET
                    + "\n</head>\n"
                    + html[k2 + 1 :]
                )

    # Fallback full wrapper
    return (
        "<!doctype html><html><head>"
        + MATHJAX_SNIPPET
        + "</head><body>"
        + html
        + "</body></html>"
    )


class RunReq(BaseModel):
    h5ad: str
    out_root: str
    groupby: str = "leiden"
    teacher_model_id: str = ""
    knowledge_root: str = "model\\teacher\\knowledge"
    teacher_base_url: str = ""
    teacher_api_key: str = ""

    agent_model_id: str = ""
    agent_base_url: str = ""
    agent_api_key: str = ""
    agent_max_turns: int = 6

    max_llm_calls: int = 24
    privacy_guard: bool = True
    debug_checks: bool = False

    epochs: int = 1500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    d_hid: int = 512
    d_out: int = 256
    n_layers: int = 3

    dropout: float = 0.1
    grad_clip: float = 5.0
    log_every: int = 10
    w_recon: float = 1.0
    w_spatial_pred: float = 1.0
    lam_ser_warmup_ratio: float = 0.15
    ser_w_proto: float = 1.0
    n_domains: int = 8
    label_col: str = ""
    batch_col: str = ""
    time_col: str = ""
    emb_key: str = "X_plm"
    perturb_genes: str = ""

    max_iterations: int = 3
    min_delta: float = 0.02
    stop_on_regress: bool = True
    skip_agent: bool = False

    regress_ratio: float = 0.10
    coverage_drop_ratio: float = 0.20



class StartJobReq(RunReq):
    kind: str = Field(..., description="teacher|main|agent|downstream|auto")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatReq(RunReq):
    session_id: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)


@dataclass
class Job:
    id: str
    kind: str
    req: RunReq
    created_at: float = field(default_factory=time.time)
    status: str = "queued"  # queued|running|done|error|canceled
    returncode: Optional[int] = None
    cmd: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    tokens_json: str = ""
    embedded_h5ad: str = ""
    report_path: str = ""
    error_message: str = ""
    env_overrides: Dict[str, str] = field(default_factory=dict)
    queue: "asyncio.Queue[dict]" = field(default_factory=asyncio.Queue)
    proc: Optional[asyncio.subprocess.Process] = None


@dataclass
class ChatSession:
    id: str
    req: RunReq
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[dict[str, str]] = field(default_factory=list)
    active_job_id: str = ""
    latest_job_id: str = ""


def _require_non_empty(req: RunReq) -> None:
    if not req.h5ad or not req.out_root:
        raise HTTPException(400, "h5ad/out_root required")


def _normalize_out_root(out_root: str) -> Path:
    return Path(out_root).expanduser()


def _safe_resolve_under(base: Path, target: Path) -> Path:
    base = base.resolve()
    target = target.resolve()
    try:
        target.relative_to(base)
    except Exception:
        raise HTTPException(403, "invalid path (outside allowed root)")
    return target


def _lock_file_path(out_root: Path) -> Path:
    return out_root / ".scagent_outroot.lock"


def _acquire_outroot_lock(out_root: Path, job_id: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    lf = _lock_file_path(out_root)
    try:
        fd = os.open(str(lf), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(job_id)
    except FileExistsError:
        raise HTTPException(409, f"out_root is busy: {out_root}")
    OUTROOT_LOCKS[str(out_root.resolve())] = job_id


def _release_outroot_lock(out_root: Path, job_id: str) -> None:
    key = str(out_root.resolve())
    if OUTROOT_LOCKS.get(key) != job_id:
        return
    lf = _lock_file_path(out_root)
    try:
        if lf.exists():
            content = lf.read_text(encoding="utf-8", errors="ignore").strip()
            if content == job_id:
                lf.unlink(missing_ok=True)
    except Exception:
        pass
    OUTROOT_LOCKS.pop(key, None)


def _jobs_db_path(out_root: Path) -> Path:
    return out_root / "jobs_db.json"


def _write_jobs_db(out_root: Path, job: Job) -> None:
    """
    Persist job metadata to out_root/jobs_db.json.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    p = _jobs_db_path(out_root)

    data: Dict[str, Any] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    data.setdefault("version", 1)
    data.setdefault("jobs", {})

    data["jobs"][job.id] = {
        "job_id": job.id,
        "kind": job.kind,
        "created_at": job.created_at,
        "status": job.status,
        "returncode": job.returncode,
        "cmd": job.cmd,
        "tokens_json": job.tokens_json,
        "embedded_h5ad": job.embedded_h5ad,
        "report_path": job.report_path,
        "error_message": job.error_message,
        "h5ad": job.req.h5ad,
        "groupby": job.req.groupby,
    }

    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def _build_env_overrides(req: RunReq) -> Dict[str, str]:
    env_overrides: Dict[str, str] = {}
    teacher_api = req.teacher_api_key.strip()
    teacher_base = req.teacher_base_url.strip()
    teacher_model = req.teacher_model_id.strip()
    if teacher_api:
        env_overrides["TEACHER_API_KEY"] = teacher_api
    if teacher_base:
        env_overrides["TEACHER_BASE_URL"] = teacher_base
    if teacher_model:
        env_overrides["TEACHER_MODEL_ID"] = teacher_model

    # Prefer Teacher credentials for Agent, so both LLM calls can share Teacher API.
    agent_api = teacher_api or req.agent_api_key.strip()
    agent_base = teacher_base or req.agent_base_url.strip()
    agent_model = teacher_model or req.agent_model_id.strip()
    if agent_api:
        env_overrides["AGENT_API_KEY"] = agent_api
    if agent_base:
        env_overrides["AGENT_BASE_URL"] = agent_base
    if agent_model:
        env_overrides["AGENT_MODEL_ID"] = agent_model
    env_overrides["AGENT_MAX_TURNS"] = str(max(1, int(req.agent_max_turns)))
    return env_overrides


def _spawn_job(kind: str, req: RunReq, env_overrides: Dict[str, str]) -> str:
    if kind not in ("teacher", "main", "agent", "downstream", "auto"):
        raise HTTPException(400, "kind must be one of: teacher|main|agent|downstream|auto")
    _require_non_empty(req)

    out_root = _normalize_out_root(req.out_root).resolve()
    lf = _lock_file_path(out_root)
    if lf.exists():
        raise HTTPException(409, f"out_root is busy: {out_root}")

    job_id = uuid.uuid4().hex
    job = Job(
        id=job_id,
        kind=kind,
        req=req,
        env_overrides=env_overrides,
    )
    JOBS[job_id] = job
    asyncio.create_task(_run_job(job))
    return job_id


async def _enqueue(job: Job, typ: str, line: str) -> None:
    if typ == "stdout":
        job.stdout_tail = (job.stdout_tail + line)[-20000:]
    elif typ == "stderr":
        job.stderr_tail = (job.stderr_tail + line)[-20000:]
    await job.queue.put({"type": typ, "line": line})


def _decode_stream_line(b: bytes) -> str:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        s = b.decode("utf-8", errors="replace")
    except Exception:
        s = b.decode(errors="replace")
    return _ANSI_ESCAPE_RE.sub("", s)


async def _read_stream(job: Job, stream: asyncio.StreamReader, typ: str) -> None:
    # Chunk-based reader for true streaming (no newline required).
    while True:
        b = await stream.read(512)
        if not b:
            break
        s = _decode_stream_line(b)  # keep your ANSI stripping + utf-8 decode
        await _enqueue(job, typ, s)


async def _run_cmd_streaming(job: Job, cmd: list[str], cwd: Optional[Path] = None) -> int:
    job.cmd = " ".join(cmd)
    job.status = "running"
    await _enqueue(job, "meta", f"[job] start: {job.cmd}\n")

    out_root = _normalize_out_root(job.req.out_root)
    _write_jobs_db(out_root, job)

    run_cwd = (cwd or PROJECT_ROOT).resolve()
    env = os.environ.copy()
    py_model = str((PROJECT_ROOT / "model").resolve())
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = py_model + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = py_model
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if job.env_overrides:
        env.update(job.env_overrides)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(run_cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    job.proc = proc

    assert proc.stdout is not None
    assert proc.stderr is not None
    t1 = asyncio.create_task(_read_stream(job, proc.stdout, "stdout"))
    t2 = asyncio.create_task(_read_stream(job, proc.stderr, "stderr"))

    rc = await proc.wait()
    await t1
    await t2

    job.returncode = rc
    await _enqueue(job, "meta", f"[job] exit code: {rc}\n")
    _write_jobs_db(out_root, job)
    return rc


def _find_tokens_json(out_root: Path) -> str:
    if not out_root.exists():
        return ""
    cands = sorted(out_root.rglob("*teacher_tokens*.json"))
    if cands:
        return str(cands[-1].resolve())
    cands2 = sorted(out_root.rglob("*tokens*.json"))
    return str(cands2[-1].resolve()) if cands2 else ""


def _embedded_h5ad_path(out_root: Path) -> Path:
    return (out_root / "plm_outputs" / "plm_embedded.h5ad").resolve()


def _report_path(out_root: Path) -> Path:
    return (out_root / "agent_latest_report.html").resolve()


def _downstream_report_path(out_root: Path) -> Path:
    return (out_root / "downstream_outputs" / "downstream_report.html").resolve()


def _build_next_config_fallback(req: RunReq, current_action: Dict[str, Any]) -> Dict[str, Any]:
    def _as_float(x: Any, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    out: Dict[str, Any] = {
        "lam_ser": _as_float(current_action.get("lam_ser"), 0.5),
        "mask_ratio": _as_float(current_action.get("mask_ratio"), 0.3),
        "groupby": str(current_action.get("groupby") or req.groupby or "leiden"),
        "notes": "AUTO-FALLBACK: next_config.json missing after agent run; keep loop alive with safe defaults/current action.",
    }

    if current_action.get("spatial_k") is not None:
        out["spatial_k"] = current_action.get("spatial_k")
    if current_action.get("attr_k") is not None:
        out["attr_k"] = current_action.get("attr_k")
    if current_action.get("conf_floor") is not None:
        out["conf_floor"] = current_action.get("conf_floor")
    if current_action.get("lam_ser_warmup_ratio") is not None:
        out["lam_ser_warmup_ratio"] = current_action.get("lam_ser_warmup_ratio")
    if current_action.get("lr") is not None:
        out["lr"] = current_action.get("lr")
    if current_action.get("grad_clip") is not None:
        out["grad_clip"] = current_action.get("grad_clip")
    if current_action.get("d_hid") is not None:
        out["d_hid"] = current_action.get("d_hid")
    if current_action.get("prefer_ser"):
        out["prefer_ser"] = current_action.get("prefer_ser")
    return out


def summarize_embedded_h5ad(h5ad_path: Path) -> dict:
    metrics: Dict[str, Any] = {}
    if not h5ad_path.exists():
        return {"error": "embedded_h5ad_missing", "path": str(h5ad_path)}

    try:
        import anndata as ad
        import numpy as np

        adata = ad.read_h5ad(str(h5ad_path))
        obs = adata.obs

        def _finite_stats(x: Any) -> Optional[Dict[str, float]]:
            arr = np.asarray(x, dtype=float)
            m = np.isfinite(arr)
            if not np.any(m):
                return None
            a = arr[m]
            return {
                "mean": float(np.mean(a)),
                "p99": float(np.quantile(a, 0.99)),
                "p10": float(np.quantile(a, 0.10)),
            }

        if "plm_recon_err" in obs:
            st = _finite_stats(obs["plm_recon_err"].values)
            if st is None:
                metrics["error"] = "recon_err_all_nonfinite"
            else:
                metrics["recon_mean"] = st["mean"]
                metrics["recon_p99"] = st["p99"]

        if "plm_ser_energy" in obs:
            st = _finite_stats(obs["plm_ser_energy"].values)
            if st is not None:
                metrics["energy_mean"] = st["mean"]
                metrics["energy_p99"] = st["p99"]

        if "ser_coverage" in obs:
            st = _finite_stats(obs["ser_coverage"].values)
            if st is not None:
                metrics["coverage_p10"] = st["p10"]

        if "ser_strength" in obs:
            st = _finite_stats(obs["ser_strength"].values)
            if st is not None:
                metrics["strength_p10"] = st["p10"]

        if not metrics:
            metrics["error"] = "metrics_missing_in_obs"

        return metrics
    except Exception as e:
        return {"error": f"read_h5ad_failed: {e!r}", "path": str(h5ad_path)}


async def _job_teacher(job: Job) -> None:
    req = job.req
    out = _normalize_out_root(req.out_root)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "teacher.cli",
        "run",
        "--h5ad",
        req.h5ad,
        "--out-dir",
        str(out),
        "--groupby",
        req.groupby,
        "--max-llm-calls",
        str(max(1, int(req.max_llm_calls))),
    ]
    if req.teacher_model_id.strip():
        cmd += ["--model-id", req.teacher_model_id.strip()]
    if req.knowledge_root.strip():
        cmd += ["--knowledge-root", req.knowledge_root.strip()]
    if not bool(req.privacy_guard):
        cmd += ["--disable-privacy-guard"]
    if bool(req.debug_checks):
        cmd += ["--debug-checks"]

    rc = await _run_cmd_streaming(job, cmd)
    if job.status == "canceled":
        return
    if rc != 0:
        job.status = "error"
        job.error_message = "teacher failed"
        _write_jobs_db(out, job)
        return

    job.tokens_json = _find_tokens_json(out)
    await _enqueue(job, "meta", f"[teacher] tokens_json={job.tokens_json}\n")
    job.status = "done"
    _write_jobs_db(out, job)


async def _job_main(job: Job, force_skip_teacher: bool = False) -> None:
    req = job.req
    out = _normalize_out_root(req.out_root)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "main",
        "--h5ad",
        req.h5ad,
        "--out_root",
        str(out),
        "--groupby",
        req.groupby,
        "--max_llm_calls",
        str(max(1, int(req.max_llm_calls))),
        "--epochs",
        str(max(1, int(req.epochs))),
        "--lr",
        str(float(req.lr)),
        "--weight_decay",
        str(float(req.weight_decay)),
        "--d_hid",
        str(max(16, int(req.d_hid))),
        "--d_out",
        str(max(8, int(req.d_out))),
        "--n_layers",
        str(max(1, int(req.n_layers))),
        "--dropout",
        str(max(0.0, min(0.9, float(req.dropout)))),
        "--grad_clip",
        str(max(0.1, float(req.grad_clip))),
        "--log_every",
        str(max(1, int(req.log_every))),
        "--w_recon",
        str(max(0.0, float(req.w_recon))),
        "--w_spatial_pred",
        str(max(0.0, float(req.w_spatial_pred))),
        "--lam_ser_warmup_ratio",
        str(max(0.0, min(1.0, float(req.lam_ser_warmup_ratio)))),
        "--ser_w_proto",
        str(max(0.0, float(req.ser_w_proto))),
    ]
    if req.teacher_model_id.strip():
        cmd += ["--teacher_model_id", req.teacher_model_id.strip()]
    if req.knowledge_root.strip():
        cmd += ["--knowledge_root", req.knowledge_root.strip()]
    if not bool(req.privacy_guard):
        cmd.append("--disable_privacy_guard")
    if bool(req.debug_checks):
        cmd.append("--debug_checks")

    skip_teacher_flag = bool(force_skip_teacher)
    next_cfg = out / "next_config.json"
    if next_cfg.exists():
        try:
            cfg = json.loads(next_cfg.read_text(encoding="utf-8"))
            if "lam_ser" in cfg:
                cmd += ["--lam_ser", str(cfg["lam_ser"])]
            if "mask_ratio" in cfg:
                cmd += ["--mask_ratio", str(cfg["mask_ratio"])]
            if "spatial_k" in cfg:
                cmd += ["--spatial_k", str(cfg["spatial_k"])]
            if "attr_k" in cfg:
                cmd += ["--attr_k", str(cfg["attr_k"])]
            if "conf_floor" in cfg:
                cmd += ["--conf_floor", str(cfg["conf_floor"])]
            if "epochs" in cfg:
                cmd += ["--epochs", str(cfg["epochs"])]
            if "lr" in cfg:
                cmd += ["--lr", str(cfg["lr"])]
            if "weight_decay" in cfg:
                cmd += ["--weight_decay", str(cfg["weight_decay"])]
            if "d_hid" in cfg:
                cmd += ["--d_hid", str(cfg["d_hid"])]
            if "d_out" in cfg:
                cmd += ["--d_out", str(cfg["d_out"])]
            if "n_layers" in cfg:
                cmd += ["--n_layers", str(cfg["n_layers"])]
            if "dropout" in cfg:
                cmd += ["--dropout", str(cfg["dropout"])]
            if "grad_clip" in cfg:
                cmd += ["--grad_clip", str(cfg["grad_clip"])]
            if "log_every" in cfg:
                cmd += ["--log_every", str(cfg["log_every"])]
            if "w_recon" in cfg:
                cmd += ["--w_recon", str(cfg["w_recon"])]
            if "w_spatial_pred" in cfg:
                cmd += ["--w_spatial_pred", str(cfg["w_spatial_pred"])]
            if "lam_ser_warmup_ratio" in cfg:
                cmd += ["--lam_ser_warmup_ratio", str(cfg["lam_ser_warmup_ratio"])]
            if "ser_w_proto" in cfg:
                cmd += ["--ser_w_proto", str(cfg["ser_w_proto"])]
            if "groupby" in cfg and str(cfg["groupby"]).strip():
                cmd += ["--groupby", str(cfg["groupby"]).strip()]
            if bool(cfg.get("skip_teacher", False)):
                skip_teacher_flag = True
            if "prefer_ser" in cfg and str(cfg["prefer_ser"]).strip():
                cmd += ["--prefer_ser", str(cfg["prefer_ser"]).strip()]
        except Exception as e:
            await _enqueue(job, "stderr", f"[main] failed to parse next_config.json: {e!r}\n")

    if skip_teacher_flag:
        cmd.append("--skip_teacher")

    rc = await _run_cmd_streaming(job, cmd)
    if job.status == "canceled":
        return
    if rc != 0:
        job.status = "error"
        job.error_message = "main failed"
        _write_jobs_db(out, job)
        return

    job.embedded_h5ad = str(_embedded_h5ad_path(out))
    await _enqueue(job, "meta", f"[main] embedded_h5ad={job.embedded_h5ad}\n")
    job.status = "done"
    _write_jobs_db(out, job)


async def _job_agent(job: Job) -> None:
    req = job.req
    out = _normalize_out_root(req.out_root)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "agent.cli",
        "--goal",
        "WebRun: generate an HTML report",
        "--h5ad",
        req.h5ad,
        "--out_root",
        str(out),
        "--groupby",
        req.groupby,
    ]

    rc = await _run_cmd_streaming(job, cmd)
    if job.status == "canceled":
        return
    if rc != 0:
        job.status = "error"
        job.error_message = "agent failed"
        _write_jobs_db(out, job)
        return

    report_path = _report_path(out)
    html = job.stdout_tail or ""
    if "<html" not in html.lower():
        html = "<html><body><pre>" + (job.stdout_tail or "No report") + "</pre></body></html>"

    html = _inject_mathjax(html)
    report_path.write_text(html, encoding="utf-8")

    job.report_path = str(report_path)
    await _enqueue(job, "meta", f"[agent] report_path={job.report_path}\n")
    job.status = "done"
    _write_jobs_db(out, job)


async def _job_downstream(job: Job) -> None:
    req = job.req
    out = _normalize_out_root(req.out_root)
    out.mkdir(parents=True, exist_ok=True)

    embedded = _embedded_h5ad_path(out)
    if not embedded.exists():
        job.status = "error"
        job.error_message = f"embedded_h5ad_missing: {embedded}"
        await _enqueue(job, "stderr", f"[downstream] missing embedding: {embedded}\n")
        _write_jobs_db(out, job)
        return

    ds_out = (out / "downstream_outputs").resolve()
    cmd = [
        sys.executable,
        "-m",
        "demo.run_downstream",
        "--embedded_h5ad",
        str(embedded),
        "--out_dir",
        str(ds_out),
        "--emb_key",
        (req.emb_key.strip() or "X_plm"),
        "--n_domains",
        str(max(2, int(req.n_domains))),
    ]
    if req.label_col.strip():
        cmd += ["--label_col", req.label_col.strip()]
    if req.batch_col.strip():
        cmd += ["--batch_col", req.batch_col.strip()]
    if req.time_col.strip():
        cmd += ["--time_col", req.time_col.strip()]
    if req.perturb_genes.strip():
        cmd += ["--perturb_genes", req.perturb_genes.strip()]

    rc = await _run_cmd_streaming(job, cmd)
    if job.status == "canceled":
        return
    if rc != 0:
        job.status = "error"
        job.error_message = "downstream failed"
        _write_jobs_db(out, job)
        return

    report_path = _downstream_report_path(out)
    if report_path.exists():
        job.report_path = str(report_path)
        await _enqueue(job, "meta", f"[downstream] report_path={job.report_path}\n")
    else:
        html = job.stdout_tail or ""
        if "<html" not in html.lower():
            html = "<html><body><pre>" + (job.stdout_tail or "No downstream report") + "</pre></body></html>"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_inject_mathjax(html), encoding="utf-8")
        job.report_path = str(report_path)
        await _enqueue(job, "meta", f"[downstream] report_path={job.report_path}\n")

    job.status = "done"
    _write_jobs_db(out, job)


async def _job_auto_loop(job: Job) -> None:
    req = job.req
    out = _normalize_out_root(req.out_root)
    out.mkdir(parents=True, exist_ok=True)

    tracker = StateTracker(out)
    fp_keys = [
        "h5ad",
        "groupby",
        "lam_ser",
        "mask_ratio",
        "spatial_k",
        "attr_k",
        "conf_floor",
        "lam_ser_warmup_ratio",
        "lr",
        "grad_clip",
        "d_hid",
    ]

    await _enqueue(
        job,
        "meta",
        f"[auto] max_iterations={req.max_iterations} min_delta={req.min_delta} "
        f"regress_ratio={req.regress_ratio} coverage_drop_ratio={req.coverage_drop_ratio}\n",
    )
    job.status = "running"
    _write_jobs_db(out, job)

    current_action: Dict[str, Any] = {
        "h5ad": req.h5ad,
        "groupby": req.groupby,
        "lam_ser": 1.0,
        "mask_ratio": 0.25,
        "spatial_k": 12,
        "attr_k": 12,
        "conf_floor": None,
        "lam_ser_warmup_ratio": req.lam_ser_warmup_ratio,
        "lr": req.lr,
        "grad_clip": req.grad_clip,
        "d_hid": req.d_hid,
    }

    best_metrics: Optional[Dict[str, Any]] = None

    def _as_float(x: Any, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _as_int(x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def _step_k(cur: int) -> int:
        if cur > 8:
            return 8
        if cur > 6:
            return 6
        if cur > 4:
            return 4
        return max(2, cur - 1)

    def _step_hid(cur: int) -> int:
        if cur > 128:
            return 128
        if cur > 64:
            return 64
        return max(32, cur // 2)

    for round_idx in range(1, req.max_iterations + 1):
        if job.status == "canceled":
            await _enqueue(job, "meta", "[auto] canceled\n")
            _write_jobs_db(out, job)
            return

        await _enqueue(job, "meta", f"\n[auto] ===== Round {round_idx} =====\n")

        t = Job(id=job.id, kind="teacher", req=req, queue=job.queue, env_overrides=job.env_overrides)
        await _job_teacher(t)
        if t.status != "done":
            job.status = t.status
            job.error_message = t.error_message
            _write_jobs_db(out, job)
            break
        job.tokens_json = t.tokens_json

        m = Job(id=job.id, kind="main", req=req, queue=job.queue, env_overrides=job.env_overrides)
        # Auto-loop already ran teacher in this round; avoid duplicate LLM/token generation.
        await _job_main(m, force_skip_teacher=True)
        if m.status != "done":
            low = ((m.stderr_tail or "") + "\n" + (m.error_message or "")).lower()
            recover_action: Optional[Dict[str, Any]] = None
            recover_reason = ""

            if "out of memory" in low or "cuda out of memory" in low:
                recover_action = dict(current_action)
                cur_sk = _as_int(current_action.get("spatial_k"), 12)
                cur_ak = _as_int(current_action.get("attr_k"), 12)
                cur_hid = _as_int(current_action.get("d_hid"), req.d_hid)
                cur_mask = _as_float(current_action.get("mask_ratio"), 0.25)

                recover_action["spatial_k"] = _step_k(cur_sk)
                recover_action["attr_k"] = _step_k(cur_ak)
                recover_action["d_hid"] = _step_hid(cur_hid)
                recover_action["mask_ratio"] = max(0.10, round(cur_mask - 0.05, 4))
                recover_action["notes"] = "AUTO_RECOVER: OOM detected. Reduce k -> d_hid -> mask_ratio."
                recover_reason = (
                    f"OOM recovery: spatial_k {cur_sk}->{recover_action['spatial_k']}, "
                    f"attr_k {cur_ak}->{recover_action['attr_k']}, "
                    f"d_hid {cur_hid}->{recover_action['d_hid']}, "
                    f"mask_ratio {cur_mask:.3f}->{recover_action['mask_ratio']:.3f}"
                )
            elif "nan" in low or "inf" in low:
                recover_action = dict(current_action)
                cur_lr = _as_float(current_action.get("lr"), req.lr)
                cur_gc = _as_float(current_action.get("grad_clip"), req.grad_clip)
                cur_lam = _as_float(current_action.get("lam_ser"), 1.0)
                cur_warm = _as_float(current_action.get("lam_ser_warmup_ratio"), req.lam_ser_warmup_ratio)

                recover_action["lr"] = max(1e-6, cur_lr * 0.5)
                recover_action["grad_clip"] = min(cur_gc, 2.0)
                recover_action["lam_ser"] = max(0.05, round(cur_lam * 0.8, 4))
                recover_action["lam_ser_warmup_ratio"] = max(cur_warm, 0.15)
                recover_action["notes"] = "AUTO_RECOVER: NaN/Inf detected. Lower lr, grad_clip, lam_ser and enforce warmup."
                recover_reason = (
                    f"NaN/Inf recovery: lr {cur_lr:.6g}->{recover_action['lr']:.6g}, "
                    f"grad_clip {cur_gc:.3f}->{recover_action['grad_clip']:.3f}, "
                    f"lam_ser {cur_lam:.3f}->{recover_action['lam_ser']:.3f}, "
                    f"warmup {cur_warm:.3f}->{recover_action['lam_ser_warmup_ratio']:.3f}"
                )

            if recover_action is not None:
                next_cfg = out / "next_config.json"
                try:
                    next_cfg.write_text(json.dumps(recover_action, ensure_ascii=False, indent=2), encoding="utf-8")
                    current_action.update(recover_action)
                    await _enqueue(job, "stderr", f"[auto][recover] {recover_reason}\n")
                    await _enqueue(job, "meta", "[auto] continue with recovered config in next round\n")
                    continue
                except Exception as e:
                    await _enqueue(job, "stderr", f"[auto][recover] failed to write next_config: {e!r}\n")

            job.status = m.status
            job.error_message = m.error_message
            _write_jobs_db(out, job)
            break
        job.embedded_h5ad = m.embedded_h5ad

        metrics = summarize_embedded_h5ad(Path(m.embedded_h5ad))
        await _enqueue(job, "meta", f"[verify] metrics={json.dumps(metrics, ensure_ascii=False)}\n")

        fp = fingerprint(current_action, keys=fp_keys)
        rec = RoundRecord(
            round_idx=round_idx,
            ts=time.time(),
            kind="auto",
            action=dict(current_action),
            fp=fp,
            metrics=metrics,
            status="done" if "error" not in metrics else "error",
            note="",
        )

        hist = tracker.last(3)

        esc, esc_reason = should_escalate(
            history=hist,
            cur_metrics=metrics,
            regress_ratio=req.regress_ratio,
            coverage_drop_ratio=req.coverage_drop_ratio,
            consecutive_errors=2,
        )
        if esc:
            rec.note = esc_reason
            tracker.append(rec)
            await _enqueue(job, "stderr", f"[ESCALATE] {esc_reason}\n")
            await _enqueue(job, "stderr", "[ESCALATE] stopping autoloop and waiting for human intervention\n")
            job.status = "done"
            _write_jobs_db(out, job)
            return

        brk, brk_reason = loop_breaker(
            history=hist,
            cur=rec,
            min_delta=req.min_delta,
            repeat_window=3,
            max_same_fp=2,
        )
        if brk:
            rec.note = brk_reason
            tracker.append(rec)
            await _enqueue(job, "stderr", f"[LOOP_BREAKER] {brk_reason}\n")
            await _enqueue(job, "meta", "[auto] stopping due to loop breaker\n")
            job.status = "done"
            _write_jobs_db(out, job)
            return

        if req.stop_on_regress and best_metrics is not None:
            reg, reg_reason = regression_score(
                best_metrics=best_metrics,
                cur_metrics=metrics,
                recon_regress_ratio=req.regress_ratio,
                coverage_drop_ratio=req.coverage_drop_ratio,
            )
            if reg:
                rec.note = "stop_on_regress: " + reg_reason
                tracker.append(rec)
                await _enqueue(job, "stderr", f"[STOP_ON_REGRESS] {rec.note}\n")
                await _enqueue(job, "meta", "[auto] stopping due to regression\n")
                job.status = "done"
                _write_jobs_db(out, job)
                return

        tracker.append(rec)

        if rec.status == "done":
            if best_metrics is None:
                best_metrics = metrics
            else:
                # Best selection: prefer lower recon_mean; tie-break by higher coverage_p10.
                def _f(x: Any) -> Optional[float]:
                    try:
                        import math
                        v = float(x)
                        return v if math.isfinite(v) else None
                    except Exception:
                        return None

                br = _f(best_metrics.get("recon_mean"))
                rr = _f(metrics.get("recon_mean"))
                bc = _f(best_metrics.get("coverage_p10"))
                rc = _f(metrics.get("coverage_p10"))

                if br is not None and rr is not None and rr < br:
                    best_metrics = metrics
                elif br is not None and rr is not None and rr == br:
                    if bc is not None and rc is not None and rc > bc:
                        best_metrics = metrics

                # ================== Agent / heuristics selection ==================
        if req.skip_agent:
            await _enqueue(
                job,
                "meta",
                "[auto] skip_agent=True. Bypassing LLM agent and using pure heuristics for next config...\n",
            )
            try:
                from model.agent.heuristics import analyze_embedded_h5ad

                heur_dir = (out / "heuristics_outputs" / f"round_{round_idx}").resolve()
                ana = analyze_embedded_h5ad(
                    embedded_h5ad=str(Path(m.embedded_h5ad).resolve()),
                    out_dir=str(heur_dir),
                    thresholds={},
                )

                payload = ana.get("payload", {}) if isinstance(ana, dict) else {}
                suggestions = payload.get("suggestions", {}) if isinstance(payload, dict) else {}

                main_sugg = suggestions.get("main.py", {}) if isinstance(suggestions, dict) else {}
                bridge_sugg = suggestions.get("bridge", {}) if isinstance(suggestions, dict) else {}
                notes_sugg = suggestions.get("notes", []) if isinstance(suggestions, dict) else []

                if not isinstance(main_sugg, dict):
                    main_sugg = {}
                if not isinstance(bridge_sugg, dict):
                    bridge_sugg = {}
                if not isinstance(notes_sugg, list):
                    notes_sugg = []

                notes_text = "Generated by fast heuristics (skip_agent=True)."
                if notes_sugg:
                    notes_text = notes_text + " " + "; ".join([str(x) for x in notes_sugg])

                fallback_cfg = {
                    "lam_ser": _as_float(main_sugg.get("lam_ser"), _as_float(current_action.get("lam_ser"), 1.0)),
                    "mask_ratio": _as_float(main_sugg.get("mask_ratio"), _as_float(current_action.get("mask_ratio"), 0.25)),
                    "spatial_k": _as_int(main_sugg.get("spatial_k"), _as_int(current_action.get("spatial_k"), 12)),
                    "attr_k": _as_int(main_sugg.get("attr_k"), _as_int(current_action.get("attr_k"), 12)),
                    "conf_floor": _as_float(bridge_sugg.get("conf_floor"), _as_float(current_action.get("conf_floor"), 0.6)),
                    "groupby": req.groupby,
                    "notes": notes_text,
                }

                next_cfg = out / "next_config.json"
                next_cfg.write_text(json.dumps(fallback_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
                job.report_path = ""
            except Exception as e:
                await _enqueue(job, "stderr", f"[auto] heuristics fallback failed: {e!r}\n")
                break
        else:
            a = Job(id=job.id, kind="agent", req=req, queue=job.queue, env_overrides=job.env_overrides)
            await _job_agent(a)
            if a.status != "done":
                job.status = a.status
                job.error_message = a.error_message
                _write_jobs_db(out, job)
                break
            job.report_path = a.report_path

        next_cfg = out / "next_config.json"
        if next_cfg.exists():
            try:
                new_action = json.loads(next_cfg.read_text(encoding="utf-8"))
            except Exception as e:
                await _enqueue(job, "stderr", f"[auto] failed to read next_config.json: {e!r}\n")
                job.status = "done"
                _write_jobs_db(out, job)
                return
        else:
            new_action = _build_next_config_fallback(req, current_action)
            try:
                next_cfg.write_text(json.dumps(new_action, ensure_ascii=False, indent=2), encoding="utf-8")
                await _enqueue(
                    job,
                    "meta",
                    "[auto] next_config.json missing; wrote fallback for continuity\n",
                )
            except Exception as e:
                await _enqueue(job, "stderr", f"[auto] failed to write fallback next_config.json: {e!r}\n")
                job.status = "done"
                _write_jobs_db(out, job)
                return

        allowed = {
            "lam_ser",
            "mask_ratio",
            "groupby",
            "spatial_k",
            "attr_k",
            "conf_floor",
            "skip_teacher",
            "prefer_ser",
            "lam_ser_warmup_ratio",
            "lr",
            "grad_clip",
            "d_hid",
        }
        cleaned = {k: new_action.get(k) for k in allowed if k in new_action}
        current_action.update(cleaned)

        await _enqueue(job, "meta", f"[auto] next_action={json.dumps(current_action, ensure_ascii=False)}\n")

    if job.status not in ("error", "canceled"):
        job.status = "done"
        _write_jobs_db(out, job)


async def _run_job(job: Job) -> None:
    out_root = _normalize_out_root(job.req.out_root)
    try:
        if job.kind in ("teacher", "main", "agent", "downstream", "auto"):
            _acquire_outroot_lock(out_root, job.id)

        _write_jobs_db(out_root, job)

        if job.kind == "teacher":
            await _job_teacher(job)
        elif job.kind == "main":
            await _job_main(job)
        elif job.kind == "agent":
            await _job_agent(job)
        elif job.kind == "downstream":
            await _job_downstream(job)
        elif job.kind == "auto":
            await _job_auto_loop(job)
        else:
            job.status = "error"
            job.error_message = f"unknown job kind: {job.kind}"
            _write_jobs_db(out_root, job)
    except asyncio.CancelledError:
        job.status = "canceled"
        job.error_message = "canceled"
        _write_jobs_db(out_root, job)
    except HTTPException:
        raise
    except Exception as e:
        job.status = "error"
        job.error_message = f"exception: {e!r}"
        await _enqueue(job, "stderr", f"[server] {job.error_message}\n")
        _write_jobs_db(out_root, job)
    finally:
        _release_outroot_lock(out_root, job.id)
        await job.queue.put({"type": "done", "line": ""})


def _job_public(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "returncode": job.returncode,
        "cmd": job.cmd,
        "tokens_json": job.tokens_json,
        "embedded_h5ad": job.embedded_h5ad,
        "report_path": job.report_path,
        "error_message": job.error_message,
        "stdout_tail": job.stdout_tail[-2000:],
        "stderr_tail": job.stderr_tail[-2000:],
    }


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/api/jobs/start")
async def start_job(req: StartJobReq) -> dict:
    kind = req.kind.strip().lower()
    run_req = RunReq(**req.model_dump(exclude={"kind"}))
    env_overrides = _build_env_overrides(run_req)
    job_id = _spawn_job(kind, run_req, env_overrides)
    return {"job_id": job_id}


def _infer_run_kind_from_text(text: str) -> str:
    low = (text or "").lower()
    if any(k in low for k in ("downstream", "evaluate", "evaluation", "eval", "integration")):
        return "downstream"
    if any(k in low for k in ("teacher", "label", "annotation")):
        return "teacher"
    if any(k in low for k in ("agent", "report", "html")):
        return "agent"
    if any(k in low for k in ("main", "train", "training", "plm")):
        return "main"
    return "auto"


def _wants_run(text: str) -> bool:
    low = (text or "").lower()
    keys = (
        "run",
        "start",
        "execute",
        "launch",
        "train",
        "auto",
        "teacher",
        "main",
        "agent",
        "downstream",
    )
    return any(k in low for k in keys)


def _latest_report_url_for_session(sess: ChatSession) -> str:
    if not sess.latest_job_id:
        return ""
    j = JOBS.get(sess.latest_job_id)
    if not j or not j.report_path:
        return ""
    return f"/api/report/{sess.latest_job_id}"


@app.post("/api/chat")
async def chat(req: ChatReq) -> dict:
    sid = req.session_id.strip() or uuid.uuid4().hex

    try:
        run_req = RunReq(**req.model_dump(exclude={"session_id", "messages"}))
    except Exception as e:
        return {
            "session_id": sid,
            "assistant_message": f"配置不完整，无法启动任务: {e}",
            "tool_calls": [],
            "run_status": "error",
            "latest_report_url": "",
        }

    sess = CHAT_SESSIONS.get(sid)
    if sess is None:
        sess = ChatSession(id=sid, req=run_req)
        CHAT_SESSIONS[sid] = sess
    else:
        sess.req = run_req

    for m in req.messages:
        role = (m.role or "").strip() or "user"
        content = (m.content or "").strip()
        if not content:
            continue
        rec = {"role": role, "content": content}
        if not sess.messages or sess.messages[-1] != rec:
            sess.messages.append(rec)
    sess.updated_at = time.time()

    user_text = ""
    for m in reversed(sess.messages):
        if m.get("role") == "user" and m.get("content"):
            user_text = str(m["content"])
            break

    if not user_text:
        msg = "请先输入一条消息。"
        sess.messages.append({"role": "assistant", "content": msg})
        return {
            "session_id": sid,
            "assistant_message": msg,
            "tool_calls": [],
            "run_status": "idle",
            "latest_report_url": _latest_report_url_for_session(sess),
        }

    active_job = JOBS.get(sess.active_job_id) if sess.active_job_id else None
    if active_job and active_job.status in ("queued", "running"):
        msg = f"任务仍在运行中（kind={active_job.kind}, status={active_job.status}, job_id={active_job.id}）。"
        sess.messages.append({"role": "assistant", "content": msg})
        return {
            "session_id": sid,
            "assistant_message": msg,
            "tool_calls": [],
            "run_status": active_job.status,
            "job_id": active_job.id,
            "latest_report_url": _latest_report_url_for_session(sess),
        }

    if active_job and active_job.status in ("done", "error", "canceled"):
        if active_job.report_path:
            sess.latest_job_id = active_job.id
        sess.active_job_id = ""

    if _wants_run(user_text):
        kind = _infer_run_kind_from_text(user_text)
        try:
            job_id = _spawn_job(kind, run_req, _build_env_overrides(run_req))
        except HTTPException as he:
            msg = f"任务启动失败: {he.detail}"
            sess.messages.append({"role": "assistant", "content": msg})
            return {
                "session_id": sid,
                "assistant_message": msg,
                "tool_calls": [],
                "run_status": "error",
                "latest_report_url": _latest_report_url_for_session(sess),
            }

        sess.active_job_id = job_id
        msg = f"已启动任务：kind={kind}，job_id={job_id}。你可以在 Live Logs 查看实时输出。"
        sess.messages.append({"role": "assistant", "content": msg})
        return {
            "session_id": sid,
            "assistant_message": msg,
            "tool_calls": [],
            "run_status": "queued",
            "job_id": job_id,
            "latest_report_url": _latest_report_url_for_session(sess),
        }

    if sess.latest_job_id:
        latest_url = _latest_report_url_for_session(sess)
        msg = "已记录你的消息。发送“开始运行”可触发任务；当前可直接打开上一份报告。"
    else:
        latest_url = ""
        msg = "已记录你的消息。发送“开始运行/执行训练/run auto”可触发任务。"
    sess.messages.append({"role": "assistant", "content": msg})
    return {
        "session_id": sid,
        "assistant_message": msg,
        "tool_calls": [],
        "run_status": "idle",
        "latest_report_url": latest_url,
    }


@app.get("/api/chat/session/{session_id}")
def chat_session(session_id: str) -> dict:
    sess = CHAT_SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")

    active_job = JOBS.get(sess.active_job_id) if sess.active_job_id else None
    status = "idle"
    if active_job:
        status = active_job.status
        if active_job.status in ("done", "error", "canceled"):
            if active_job.report_path:
                sess.latest_job_id = active_job.id
            sess.active_job_id = ""
            status = "idle"

    return {
        "session_id": sess.id,
        "messages": sess.messages[-200:],
        "run_status": status,
        "active_job_id": sess.active_job_id,
        "latest_report_url": _latest_report_url_for_session(sess),
    }


@app.get("/api/jobs/status/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return _job_public(job)


@app.post("/api/jobs/cancel/{job_id}")
async def cancel_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status not in ("queued", "running"):
        return {"ok": True, "status": job.status}

    job.status = "canceled"
    if job.proc and job.proc.returncode is None:
        try:
            job.proc.terminate()
        except Exception:
            pass

    out_root = _normalize_out_root(job.req.out_root)
    _write_jobs_db(out_root, job)

    await _enqueue(job, "meta", "[job] cancel requested\n")
    return {"ok": True, "status": job.status}


@app.get("/api/jobs/stream/{job_id}")
async def stream_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    async def event_gen() -> AsyncGenerator[bytes, None]:
        yield b"event: hello\ndata: {}\n\n"
        while True:
            msg = await job.queue.get()
            payload = json.dumps(msg, ensure_ascii=False)
            yield f"data: {payload}\n\n".encode("utf-8")
            if msg.get("type") == "done":
                break

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


@app.get("/api/report/{job_id}")
def report(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if not job.report_path:
        raise HTTPException(404, "report not ready")

    out_root = _normalize_out_root(job.req.out_root)
    rp = Path(job.report_path)
    safe_rp = _safe_resolve_under(out_root, rp)
    if not safe_rp.exists():
        raise HTTPException(404, "report not found on disk")
    return FileResponse(str(safe_rp), media_type="text/html")
