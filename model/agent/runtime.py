from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .settings import AgentSettings
from .providers.ark_client import ArkChatClient
from .heuristics import analyze_embedded_h5ad


# ============================================================
# Tool registry (MCP-like) with Pydantic validation
# ============================================================
class ToolArgsBase(BaseModel):
    model_config = ConfigDict(extra="forbid")  # block hallucinated params


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    args_model: Type[ToolArgsBase]
    description: str = ""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def json_schema(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name in self.list_names():
            spec = self._tools[name]
            schema = spec.args_model.model_json_schema()
            out.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": schema,
                },
            })
        return out

    def call(self, name: str, raw_args: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        spec = self._tools[name]

        try:
            args_obj = spec.args_model.model_validate(raw_args)
        except ValidationError as ve:
            return {
                "ok": False,
                "error_type": "ToolArgsValidationError",
                "tool": name,
                "details": json.loads(ve.json()),
                "raw_args": raw_args,
            }

        try:
            return spec.fn(**args_obj.model_dump())
        except Exception as e:
            return {
                "ok": False,
                "error_type": type(e).__name__,
                "tool": name,
                "error": str(e),
                "traceback": traceback.format_exc(limit=12),
            }


# ============================================================
# RAG (simple TF-IDF)
# ============================================================
_WORD = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(s) if len(t) >= 2]


@dataclass
class DocChunk:
    doc_id: str
    source: str
    text: str


class SimpleRAG:
    def __init__(self, kb_dir: Path, chunk_chars: int = 1200, overlap: int = 150):
        self.kb_dir = kb_dir
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.chunks: list[DocChunk] = []
        self.df: dict[str, int] = {}
        self._built = False

    def build(self) -> None:
        self.chunks = []
        self.df = {}
        if not self.kb_dir.exists():
            self._built = True
            return

        paths: list[Path] = []
        for ext in ("*.md", "*.txt", "*.py", "*.json", "*.yaml", "*.yml"):
            paths.extend(self.kb_dir.rglob(ext))

        for p in sorted(set(paths)):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not text.strip():
                continue

            start = 0
            idx = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_chars)
                chunk = text[start:end]
                doc_id = f"kb:{p.name}#{idx}"
                self.chunks.append(DocChunk(doc_id=doc_id, source=str(p), text=chunk))
                idx += 1
                if end == len(text):
                    break
                start = max(0, end - self.overlap)

        for ch in self.chunks:
            for t in set(_tokenize(ch.text)):
                self.df[t] = self.df.get(t, 0) + 1

        self._built = True

    def search(self, query: str, topk: int = 6) -> list[dict[str, Any]]:
        if not self._built:
            self.build()

        qtok = _tokenize(query)
        if not qtok or not self.chunks:
            return []

        N = len(self.chunks)
        q_tf: dict[str, int] = {}
        for t in qtok:
            q_tf[t] = q_tf.get(t, 0) + 1

        def idf(t: str) -> float:
            df = self.df.get(t, 0)
            return math.log((N + 1) / (df + 1)) + 1.0

        q_vec = {t: (q_tf[t] * idf(t)) for t in q_tf}

        def score_chunk(ch: DocChunk) -> float:
            ct = _tokenize(ch.text)
            if not ct:
                return 0.0
            c_tf: dict[str, int] = {}
            for t in ct:
                c_tf[t] = c_tf.get(t, 0) + 1
            s = 0.0
            for t, qw in q_vec.items():
                if t in c_tf:
                    s += qw * (c_tf[t] * idf(t))
            denom = math.sqrt(sum((c_tf[t] * idf(t)) ** 2 for t in c_tf.keys())) + 1e-9
            return s / denom

        scored = [(score_chunk(ch), ch) for ch in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[dict[str, Any]] = []
        for s, ch in scored[:topk]:
            if s <= 0:
                continue
            out.append({"doc_id": ch.doc_id, "source": ch.source, "score": float(s), "text": ch.text})
        return out


# ============================================================
# Tools
# ============================================================
def _tail_lines(txt: str, n: int) -> str:
    if not txt:
        return ""
    lines = txt.splitlines()
    if n <= 0 or len(lines) <= n:
        return txt
    return "\n".join(lines[-n:])


def _truncate_text(x: Any, max_chars: int) -> str:
    s = str(x or "")
    n = max(64, int(max_chars))
    if len(s) <= n:
        return s
    return s[:n] + f"... [truncated {len(s) - n} chars]"


def _is_numpy_scalar(x: Any) -> bool:
    mod = type(x).__module__
    return mod.startswith("numpy") and hasattr(x, "item")


def _is_numpy_array(x: Any) -> bool:
    mod = type(x).__module__
    return mod.startswith("numpy") and hasattr(x, "shape") and hasattr(x, "reshape") and hasattr(x, "dtype")


def _is_torch_tensor(x: Any) -> bool:
    mod = type(x).__module__
    return mod.startswith("torch") and hasattr(x, "detach") and hasattr(x, "shape")


def _to_jsonable(
    obj: Any,
    *,
    depth: int = 0,
    max_depth: int = 5,
    max_items: int = 64,
    max_preview: int = 24,
    seen: Optional[set[int]] = None,
) -> Any:
    if seen is None:
        seen = set()

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return _truncate_text(obj.decode("utf-8", errors="replace"), 512)
    if isinstance(obj, BaseModel):
        return _to_jsonable(
            obj.model_dump(),
            depth=depth + 1,
            max_depth=max_depth,
            max_items=max_items,
            max_preview=max_preview,
            seen=seen,
        )
    if _is_numpy_scalar(obj):
        try:
            return obj.item()
        except Exception:
            return _truncate_text(repr(obj), 256)
    if _is_numpy_array(obj):
        try:
            flat = obj.reshape(-1)
            n = int(flat.shape[0]) if hasattr(flat, "shape") else len(flat)
            keep = min(max_preview, n)
            preview = [_to_jsonable(flat[i], depth=depth + 1, max_depth=max_depth, max_items=max_items, max_preview=max_preview, seen=seen) for i in range(keep)]
            return {
                "__type__": "ndarray",
                "shape": list(getattr(obj, "shape", [])),
                "dtype": str(getattr(obj, "dtype", "")),
                "preview": preview,
                "truncated": max(0, n - keep),
            }
        except Exception:
            return _truncate_text(repr(obj), 512)
    if _is_torch_tensor(obj):
        try:
            cpu = obj.detach().cpu()
            flat = cpu.reshape(-1)
            n = int(flat.numel())
            keep = min(max_preview, n)
            preview = flat[:keep].tolist()
            return {
                "__type__": "tensor",
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "preview": preview,
                "truncated": max(0, n - keep),
            }
        except Exception:
            return _truncate_text(repr(obj), 512)

    oid = id(obj)
    if oid in seen:
        return "<recursive>"
    seen.add(oid)

    if depth >= max_depth:
        return _truncate_text(repr(obj), 512)

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        items = list(obj.items())
        for k, v in items[:max_items]:
            out[str(k)] = _to_jsonable(
                v,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_preview=max_preview,
                seen=seen,
            )
        if len(items) > max_items:
            out["__truncated_items__"] = len(items) - max_items
        return out

    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        out = [
            _to_jsonable(
                v,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_preview=max_preview,
                seen=seen,
            )
            for v in seq[:max_items]
        ]
        if len(seq) > max_items:
            out.append(f"... [{len(seq) - max_items} more items]")
        return out

    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _to_jsonable(
                obj.model_dump(),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_preview=max_preview,
                seen=seen,
            )
        except Exception:
            pass

    return _truncate_text(repr(obj), 512)


def safe_json_dumps(obj: Any, *, indent: Optional[int] = None, sort_keys: bool = False) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    except Exception:
        normalized = _to_jsonable(obj)
        return json.dumps(normalized, ensure_ascii=False, indent=indent, sort_keys=sort_keys)


def _compact_tool_result_for_llm(name: str, result: Any, settings: AgentSettings) -> dict[str, Any]:
    ctx_chars = max(256, int(settings.agent_tool_context_chars))
    rag_chars = max(80, int(settings.agent_rag_text_chars))
    rag_topk = max(1, int(settings.agent_rag_topk))

    if name == "rag.search" and isinstance(result, list):
        hits: list[dict[str, Any]] = []
        for h in result[:rag_topk]:
            if not isinstance(h, dict):
                continue
            hits.append({
                "doc_id": h.get("doc_id", ""),
                "source": h.get("source", ""),
                "score": h.get("score", 0.0),
                "text": _truncate_text(h.get("text", ""), rag_chars),
            })
        return {"hits": hits, "total_hits": len(result)}

    if name == "pipeline.run_main" and isinstance(result, dict):
        return {
            "ok": bool(result.get("ok", False)),
            "returncode": result.get("returncode"),
            "embedded_h5ad": result.get("embedded_h5ad", ""),
            "out_root": result.get("out_root", ""),
            "notes": result.get("notes", []),
            "stdout_tail": _truncate_text(result.get("stdout_tail", ""), ctx_chars),
            "stderr_tail": _truncate_text(result.get("stderr_tail", ""), ctx_chars),
        }

    if name == "agent.analyze_embedded" and isinstance(result, dict):
        payload = result.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        return {
            "ok": bool(result.get("ok", True)),
            "path": result.get("path", ""),
            "summary": payload.get("summary", {}),
            "suggestions": payload.get("suggestions", {}),
        }

    if name == "agent.save_next_config" and isinstance(result, dict):
        return {
            "ok": bool(result.get("ok", False)),
            "saved_path": result.get("saved_path", ""),
            "config": result.get("config", {}),
        }

    return {"preview": _truncate_text(safe_json_dumps(result), ctx_chars)}


class RagSearchArgs(ToolArgsBase):
    query: str
    topk: Optional[int] = None


class PipelineRunMainArgs(ToolArgsBase):
    project_root: str
    h5ad: str
    out_root: str
    groupby: str = "leiden"
    device: str = "cuda"
    skip_teacher: bool = False
    teacher_cmd: str = ""
    prefer_ser: str = ""
    timeout_sec: int = 0  # 0 = no timeout
    tail_lines: int = 50  


def pipeline_run_main(
    project_root: str,
    h5ad: str,
    out_root: str,
    groupby: str = "leiden",
    device: str = "cuda",
    skip_teacher: bool = False,
    teacher_cmd: str = "",
    prefer_ser: str = "",
    timeout_sec: int = 0,
    tail_lines: int = 50,
) -> dict[str, Any]:
    project_root_p = Path(project_root).resolve()

    cmd = [
        sys.executable, "-m", "main",
        "--h5ad", str(Path(h5ad).resolve()),
        "--out_root", str(Path(out_root).resolve()),
        "--groupby", groupby,
        "--device", device,
    ]
    if skip_teacher:
        cmd.append("--skip_teacher")
    if teacher_cmd.strip():
        cmd += ["--teacher_cmd", teacher_cmd.strip()]
    if prefer_ser.strip():
        cmd += ["--prefer_ser", prefer_ser.strip()]

    embedded = Path(out_root).resolve() / "plm_outputs" / "plm_embedded.h5ad"
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TQDM_DISABLE", "1")

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(project_root_p),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=(timeout_sec if timeout_sec and timeout_sec > 0 else None),
        )
        return {
            "ok": True,
            "cmd": " ".join(cmd),
            "returncode": completed.returncode,
            "stdout_tail": _tail_lines(completed.stdout or "", tail_lines),
            "stderr_tail": _tail_lines(completed.stderr or "", tail_lines),
            "embedded_h5ad": str(embedded),
            "out_root": str(Path(out_root).resolve()),
            "notes": [],
        }

    except subprocess.TimeoutExpired as te:
        return {
            "ok": False,
            "cmd": " ".join(cmd),
            "error_type": "TimeoutExpired",
            "timeout_sec": timeout_sec,
            "stdout_tail": _tail_lines(getattr(te, "stdout", "") or "", tail_lines),
            "stderr_tail": _tail_lines(getattr(te, "stderr", "") or "", tail_lines),
            "embedded_h5ad": str(embedded),
            "out_root": str(Path(out_root).resolve()),
            "notes": ["Pipeline timed out. Consider reducing epochs/batch or enabling resumable training mode."],
        }

    except subprocess.CalledProcessError as cpe:
        stderr = (cpe.stderr or "")
        notes: list[str] = []
        low = stderr.lower()
        if "out of memory" in low or "cuda out of memory" in low:
            notes.append(
                "Detected CUDA OOM. Priority actions: (1) reduce spatial_k/attr_k, "
                "(2) reduce d_hid, (3) reduce mask_ratio, then smaller batch/grad accumulation."
            )
        if "nan" in low or "inf" in low:
            notes.append(
                "Detected NaN/Inf. Priority actions: (1) lower lr, (2) tighten grad_clip to 1-2, "
                "(3) check input normalization/NaN rows, (4) lower or warm up lam_ser, (5) reduce k/d_hid."
            )
        return {
            "ok": False,
            "cmd": " ".join(cmd),
            "error_type": "CalledProcessError",
            "returncode": cpe.returncode,
            "stdout_tail": _tail_lines(cpe.stdout or "", tail_lines),
            "stderr_tail": _tail_lines(cpe.stderr or "", tail_lines),
            "embedded_h5ad": str(embedded),
            "out_root": str(Path(out_root).resolve()),
            "notes": notes,
        }


class AnalyzeEmbeddedArgs(ToolArgsBase):
    embedded_h5ad: str
    out_dir: str
    topk: int = 200
    thresholds: dict[str, Any] = Field(default_factory=dict)

class SaveNextConfigArgs(ToolArgsBase):
    out_root: str
    lam_ser: float
    mask_ratio: float
    groupby: str = "leiden"
    notes: str = Field(..., description="Explain briefly why these parameters were chosen.")


def save_next_config(out_root: str, lam_ser: float, mask_ratio: float, groupby: str, notes: str) -> dict[str, Any]:
    cfg = {
        "lam_ser": float(lam_ser),
        "mask_ratio": float(mask_ratio),
        "groupby": groupby,
        "notes": notes,
    }
    path = Path(out_root).resolve() / "next_config.json"
    path.write_text(safe_json_dumps(cfg, indent=2), encoding="utf-8")
    return {"ok": True, "saved_path": str(path), "config": cfg}


def build_system_prompt() -> str:
    return (
        "You are scAgent, an autonomous experiment optimization agent.\n"
        "Hard rules:\n"
        "1) Never invent file paths, metrics, or results.\n"
        "2) You MUST call rag.search before making recommendations.\n"
        "3) Any recommendation MUST cite evidence ids like [kb:xxx#i].\n"
        "4) If training is completed, you MUST call 'agent.save_next_config' to specify parameters for the next round BEFORE returning final_html.\n"
        "5) Prefer tools over free text.\n"
        "Return either tool calls or a final single <html>...</html>.\n"
    )


def render_html(
    run_id: str,
    goal: str,
    rag_hits: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    results: dict[str, Any],
) -> str:
    def esc(x: Any) -> str:
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rag_meta = [{"doc_id": h.get("doc_id"), "source": h.get("source"), "score": h.get("score")} for h in rag_hits]
    return "\n".join([
        "<html><head><meta charset='utf-8'/>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:18px;}pre{background:#f6f6f6;padding:10px;overflow:auto;}</style>",
        "</head><body>",
        "<h2>scAgent Report</h2>",
        f"<p><b>run_id:</b> {esc(run_id)}</p>",
        f"<p><b>goal:</b> {esc(goal)}</p>",
        "<h3>RAG Evidence</h3>",
        "<pre>" + esc(safe_json_dumps(rag_meta, indent=2)) + "</pre>",
        "<h3>Steps</h3>",
        "<pre>" + esc(safe_json_dumps(steps, indent=2)) + "</pre>",
        "<h3>Results</h3>",
        "<pre>" + esc(safe_json_dumps(results, indent=2)) + "</pre>",
        "</body></html>",
    ])


# ============================================================
# Runtime
# ============================================================
class AgentRuntime:
    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings

        self.llm = ArkChatClient(
            base_url=settings.agent_base_url,
            api_key=settings.agent_api_key,
            model_id=settings.agent_model_id,
        )
        self.rag = SimpleRAG(settings.kb_dir_abs())

        self.reg = ToolRegistry()
        self.reg.register(ToolSpec(
            name="rag.search",
            fn=lambda query, topk=None: self.rag.search(query, topk=topk or settings.agent_rag_topk),
            args_model=RagSearchArgs,
            description="Search local knowledge base for evidence (anti-hallucination).",
        ))
        self.reg.register(ToolSpec(
            name="pipeline.run_main",
            fn=lambda **kw: pipeline_run_main(**kw),
            args_model=PipelineRunMainArgs,
            description="Run Teacher->Bridge->PLM orchestrator (python -m main). Returns embedded h5ad path and logs.",
        ))
        self.reg.register(ToolSpec(
            name="agent.analyze_embedded",
            fn=lambda embedded_h5ad, out_dir, topk=200, thresholds=None: analyze_embedded_h5ad(
                embedded_h5ad=embedded_h5ad,
                out_dir=out_dir,
                topk=topk,
                thresholds=(thresholds or {}),
            ),
            args_model=AnalyzeEmbeddedArgs,
            description="Analyze embedded h5ad metrics and output heuristic suggestions JSON.",
        ))
        self.reg.register(ToolSpec(
            name="agent.save_next_config",
            fn=lambda **kw: save_next_config(**kw),
            args_model=SaveNextConfigArgs,
            description="MANDATORY TOOL: Save the final chosen parameters for the next auto-loop round based on your analysis. Must be called before generating the final HTML report.",
        ))

    def _now_tag(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _default_thresholds(self) -> dict[str, Any]:
        s = self.settings
        return {
            "recon_p99_factor": s.recon_p99_factor,
            "ser_p99_factor": s.ser_p99_factor,
            "cov_p10_min": s.cov_p10_min,
            "strength_p10_min": s.strength_p10_min,
            "default_lam_ser": s.default_lam_ser,
            "default_mask_ratio": s.default_mask_ratio,
            "default_spatial_k": s.default_spatial_k,
            "default_attr_k": s.default_attr_k,
            "default_conf_floor": s.default_conf_floor,
            "low_conf_floor": s.low_conf_floor,
            "recon_mask_ratio": s.recon_mask_ratio,
            "recon_lam_ser": s.recon_lam_ser,
            "ser_lam_ser_cap": s.ser_lam_ser_cap,
        }

    def _write_session_state(
        self,
        path: Path,
        run_id: str,
        goal: str,
        messages: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        results: dict[str, Any],
        rag_hits: list[dict[str, Any]],
    ) -> None:
        if not self.settings.session_state_enabled:
            return
        try:
            path.write_text(safe_json_dumps({
                "run_id": run_id,
                "goal": goal,
                "messages": messages,
                "steps": steps,
                "results": results,
                "rag_hits_meta": [{"doc_id": h.get("doc_id"), "source": h.get("source"), "score": h.get("score")} for h in rag_hits],
                "saved_at": datetime.now().isoformat(),
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _next_config_path(self, out_root_abs: Path) -> Path:
        return out_root_abs / "next_config.json"

    def _ensure_next_config_fallback(
        self,
        out_root_abs: Path,
        groupby: str,
        results: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Defensive fallback:
        - If training succeeded BUT LLM didn't call agent.save_next_config
        - And heuristics produced suggestions
        => write next_config.json to prevent web auto-loop stopping.
        """
        p = self._next_config_path(out_root_abs)
        if p.exists():
            return None

        pr = results.get("pipeline.run_main", {})
        if not isinstance(pr, dict) or not pr.get("ok", False):
            return None

        ana = results.get("agent.analyze_embedded", {})
        if not isinstance(ana, dict):
            return None

        payload = (ana.get("payload") or {})
        if not isinstance(payload, dict) or not payload.get("ok", False):
            return None

        sugg = payload.get("suggestions", {}).get("main.py", {})
        lam_ser = float(sugg.get("lam_ser", self.settings.default_lam_ser))
        mask_ratio = float(sugg.get("mask_ratio", self.settings.default_mask_ratio))

        cfg = {
            "lam_ser": lam_ser,
            "mask_ratio": mask_ratio,
            "groupby": groupby,
            "notes": "AUTO-FALLBACK: LLM did not call agent.save_next_config. Using heuristics suggestions to keep auto-loop alive.",
        }
        p.write_text(safe_json_dumps(cfg, indent=2), encoding="utf-8")
        return {"ok": True, "saved_path": str(p), "config": cfg, "fallback": True}

    def run_once(
        self,
        goal: str,
        h5ad: str,
        groupby: str = "leiden",
        device: str = "cuda",
        out_root: Optional[str] = None,
        skip_teacher: bool = False,
        teacher_cmd: str = "",
        prefer_ser: str = "",
    ) -> str:
        project_root = str(self.settings.project_root())
        out_root_abs = self.settings.out_root_abs() if out_root is None else Path(out_root).resolve()
        out_root_abs.mkdir(parents=True, exist_ok=True)

        run_id = f"agent_{self._now_tag()}"
        agent_out_dir = out_root_abs / "agent_outputs" / run_id
        agent_out_dir.mkdir(parents=True, exist_ok=True)

        trace_path = agent_out_dir / "trace.jsonl"
        state_path = agent_out_dir / "session_state.json"

        system = build_system_prompt()
        tools = self.reg.json_schema()

        steps: list[dict[str, Any]] = []
        results: dict[str, Any] = {}
        rag_hits: list[dict[str, Any]] = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"Goal: {goal}\n"
                f"h5ad={h5ad}\n"
                f"groupby={groupby}\n"
                f"device={device}\n"
                "First call rag.search to retrieve evidence, then decide tool calls.\n"
            )},
        ]

        def run_deterministic_fallback(err: Exception) -> str:
            fallback_steps: list[dict[str, Any]] = []
            fallback_results: dict[str, Any] = {
                "fallback_reason": f"tool_call_unavailable: {type(err).__name__}: {err}"
            }

            rag = self.reg.call("rag.search", {"query": goal, "topk": self.settings.agent_rag_topk})
            fallback_steps.append({"tool": "rag.search", "args": {"query": goal, "topk": self.settings.agent_rag_topk}})
            fallback_results["rag.search"] = rag

            run_args = {
                "project_root": project_root,
                "h5ad": h5ad,
                "out_root": str(out_root_abs),
                "groupby": groupby,
                "device": device,
                "skip_teacher": skip_teacher,
                "teacher_cmd": teacher_cmd,
                "prefer_ser": prefer_ser,
                "timeout_sec": int(self.settings.pipeline_timeout_sec),
                "tail_lines": int(self.settings.pipeline_tail_lines),
            }
            pr = self.reg.call("pipeline.run_main", run_args)
            fallback_steps.append({"tool": "pipeline.run_main", "args": run_args})
            fallback_results["pipeline.run_main"] = pr

            ana_args = {
                "embedded_h5ad": (pr.get("embedded_h5ad", "") if isinstance(pr, dict) else ""),
                "out_dir": str(agent_out_dir),
                "topk": 200,
                "thresholds": self._default_thresholds(),
            }
            ana = self.reg.call("agent.analyze_embedded", ana_args)
            fallback_steps.append({"tool": "agent.analyze_embedded", "args": ana_args})
            fallback_results["agent.analyze_embedded"] = ana

            lam_ser = self.settings.default_lam_ser
            mask_ratio = self.settings.default_mask_ratio
            notes = "fallback default suggestions"
            try:
                payload = ana.get("payload", {}) if isinstance(ana, dict) else {}
                sugg = payload.get("suggestions", {}).get("main.py", {})
                lam_ser = float(sugg.get("lam_ser", lam_ser))
                mask_ratio = float(sugg.get("mask_ratio", mask_ratio))
                notes = "fallback from analyze_embedded suggestions"
            except Exception:
                pass

            save_args = {
                "out_root": str(out_root_abs),
                "lam_ser": float(lam_ser),
                "mask_ratio": float(mask_ratio),
                "groupby": groupby,
                "notes": notes,
            }
            sv = self.reg.call("agent.save_next_config", save_args)
            fallback_steps.append({"tool": "agent.save_next_config", "args": save_args})
            fallback_results["agent.save_next_config"] = sv

            if isinstance(rag, list):
                fallback_rag_hits = rag
            else:
                fallback_rag_hits = []
            return render_html(run_id, goal, fallback_rag_hits, fallback_steps, fallback_results)

        def log(tool: str, args: dict[str, Any], result: Any) -> None:
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(safe_json_dumps({"tool": tool, "args": args, "result": result}) + "\n")

        for _ in range(max(1, int(self.settings.agent_max_turns))):
            try:
                resp = self.llm.chat_with_tools(
                    messages,
                    tools,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=max(256, int(self.settings.agent_model_max_tokens)),
                )
            except Exception as e:
                return run_deterministic_fallback(e)
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                fallback = self._ensure_next_config_fallback(out_root_abs, groupby, results)
                if fallback:
                    results["agent.save_next_config"] = fallback

                content = (msg.content or "").strip()
                if "<html" in content.lower():
                    return content
                return render_html(run_id, goal, rag_hits, steps, results)

            for tc in tool_calls:
                name = tc.function.name
                raw_args = json.loads(tc.function.arguments or "{}")

                if name == "pipeline.run_main":
                    raw_args.setdefault("project_root", project_root)
                    raw_args.setdefault("h5ad", h5ad)
                    raw_args.setdefault("out_root", str(out_root_abs))
                    raw_args.setdefault("groupby", groupby)
                    raw_args.setdefault("device", device)
                    raw_args.setdefault("skip_teacher", skip_teacher)
                    raw_args.setdefault("teacher_cmd", teacher_cmd)
                    raw_args.setdefault("prefer_ser", prefer_ser)
                    raw_args.setdefault("timeout_sec", int(self.settings.pipeline_timeout_sec))
                    raw_args.setdefault("tail_lines", int(self.settings.pipeline_tail_lines))

                if name == "agent.analyze_embedded":
                    if "embedded_h5ad" not in raw_args:
                        embedded = results.get("pipeline.run_main", {}).get("embedded_h5ad", "")
                        raw_args["embedded_h5ad"] = embedded
                    raw_args.setdefault("out_dir", str(agent_out_dir))
                    raw_args.setdefault("topk", 200)
                    raw_args.setdefault("thresholds", self._default_thresholds())

                if name == "agent.save_next_config":
                    raw_args.setdefault("out_root", str(out_root_abs))
                    raw_args.setdefault("groupby", groupby)

                r = self.reg.call(name, raw_args)

                if name == "rag.search" and isinstance(r, list):
                    rag_hits = r

                steps.append({"tool": name, "args": raw_args})
                results[name] = r
                log(name, raw_args, r)

                self._write_session_state(state_path, run_id, goal, messages, steps, results, rag_hits)

                compact = _compact_tool_result_for_llm(name, r, self.settings)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": safe_json_dumps(compact),
                })

            if msg.content:
                messages.append({"role": "assistant", "content": msg.content})

        fallback = self._ensure_next_config_fallback(out_root_abs, groupby, results)
        if fallback:
            results["agent.save_next_config"] = fallback

        return render_html(run_id, goal, rag_hits, steps, results)
