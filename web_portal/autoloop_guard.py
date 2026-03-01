from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def fingerprint(action: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    """
    Create a stable fingerprint for an action dict.
    If keys provided, only those keys are included.
    """
    payload = action if keys is None else {k: action.get(k) for k in keys}
    norm = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return _sha1(norm)


@dataclass
class RoundRecord:
    round_idx: int
    ts: float
    kind: str
    action: Dict[str, Any]
    fp: str
    metrics: Dict[str, Any]
    status: str
    note: str = ""


class StateTracker:
    """
    Long-term memory on disk:
      - autoloop_history.jsonl: full per-round records
      - autoloop_summary.md: compact summaries (context pruning)
    """

    def __init__(self, out_root: Path):
        self.out_root = out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.hist_path = self.out_root / "autoloop_history.jsonl"
        self.sum_path = self.out_root / "autoloop_summary.md"

    def append(self, rec: RoundRecord) -> None:
        with self.hist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        line = (
            f"- Round {rec.round_idx}: fp={rec.fp} "
            f"lam_ser={rec.action.get('lam_ser','?')} mask_ratio={rec.action.get('mask_ratio','?')} "
            f"recon_mean={rec.metrics.get('recon_mean','?')} recon_p99={rec.metrics.get('recon_p99','?')} "
            f"energy_mean={rec.metrics.get('energy_mean','?')} energy_p99={rec.metrics.get('energy_p99','?')} "
            f"coverage_p10={rec.metrics.get('coverage_p10','?')} strength_p10={rec.metrics.get('strength_p10','?')} "
            f"status={rec.status}"
        )
        if rec.note:
            line += f" | note={rec.note}"
        with self.sum_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def last(self, n: int = 3) -> List[RoundRecord]:
        if not self.hist_path.exists():
            return []
        text = self.hist_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        lines = text.splitlines()
        out: List[RoundRecord] = []
        for s in lines[-n:]:
            d = json.loads(s)
            out.append(RoundRecord(**d))
        return out


def _to_float(x: Any) -> Optional[float]:
    """
    Convert to float and reject NaN/Inf.
    """
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def no_improvement(prev: Dict[str, Any], cur: Dict[str, Any], min_delta: float) -> bool:
    """
    Return True if recon_mean did not improve at least min_delta ratio.
    Improvement defined as: prev - cur >= min_delta * abs(prev)
    """
    p = _to_float(prev.get("recon_mean"))
    c = _to_float(cur.get("recon_mean"))
    if p is None or c is None:
        return False
    return (p - c) < (min_delta * max(1e-9, abs(p)))


def loop_breaker(
    history: List[RoundRecord],
    cur: RoundRecord,
    min_delta: float = 0.02,
    repeat_window: int = 3,
    max_same_fp: int = 2,
) -> Tuple[bool, str]:
    """
    Break if the same fingerprint repeats too often in recent window
    and metrics do not improve.
    """
    recent = history[-repeat_window:]
    same = sum(1 for r in recent if r.fp == cur.fp)
    if same >= max_same_fp:
        prev_metrics = recent[-1].metrics if recent else {}
        if no_improvement(prev_metrics, cur.metrics, min_delta=min_delta):
            return True, "loop_breaker: repeated similar actions without improvement"
    return False, ""


def regression_score(
    best_metrics: Dict[str, Any],
    cur_metrics: Dict[str, Any],
    recon_regress_ratio: float = 0.10,
    coverage_drop_ratio: float = 0.20,
) -> Tuple[bool, str]:
    """
    Decide whether current metrics are a harmful regression.

    Rules:
      - recon_mean higher than best by recon_regress_ratio triggers regression
      - coverage_p10 lower than best by coverage_drop_ratio triggers regression

    Missing/NaN/Inf metrics are ignored (no trigger).
    """
    best_recon = _to_float(best_metrics.get("recon_mean"))
    cur_recon = _to_float(cur_metrics.get("recon_mean"))

    best_cov = _to_float(best_metrics.get("coverage_p10"))
    cur_cov = _to_float(cur_metrics.get("coverage_p10"))

    if best_recon is not None and cur_recon is not None:
        if cur_recon > best_recon * (1.0 + recon_regress_ratio):
            return True, f"regression: recon_mean worsened > {int(recon_regress_ratio*100)}% vs best"

    if best_cov is not None and cur_cov is not None:
        if cur_cov < best_cov * (1.0 - coverage_drop_ratio):
            return True, f"regression: coverage_p10 dropped > {int(coverage_drop_ratio*100)}% vs best"

    return False, ""


def should_escalate(
    history: List[RoundRecord],
    cur_metrics: Dict[str, Any],
    regress_ratio: float = 0.10,
    coverage_drop_ratio: float = 0.20,
    consecutive_errors: int = 2,
) -> Tuple[bool, str]:
    """
    Escalate on:
      - consecutive error records, or
      - recon_mean regresses vs best, or
      - coverage_p10 collapses vs best

    Best selection prefers lower recon_mean; tie-break by higher coverage_p10.
    """
    if history:
        tail = history[-consecutive_errors:]
        if len(tail) == consecutive_errors and all(r.status in ("error", "canceled") for r in tail):
            return True, "escalate: consecutive failures"

    best: Optional[Dict[str, Any]] = None
    for r in history:
        if r.status != "done":
            continue
        if best is None:
            best = r.metrics
            continue

        br = _to_float(best.get("recon_mean"))
        rr = _to_float(r.metrics.get("recon_mean"))
        bc = _to_float(best.get("coverage_p10"))
        rc = _to_float(r.metrics.get("coverage_p10"))

        if rr is not None and br is not None and rr < br:
            best = r.metrics
        elif rr is not None and br is not None and rr == br:
            if rc is not None and bc is not None and rc > bc:
                best = r.metrics

    if best is None:
        return False, ""

    reg, reason = regression_score(
        best_metrics=best,
        cur_metrics=cur_metrics,
        recon_regress_ratio=regress_ratio,
        coverage_drop_ratio=coverage_drop_ratio,
    )
    if reg:
        return True, "escalate: " + reason

    return False, ""