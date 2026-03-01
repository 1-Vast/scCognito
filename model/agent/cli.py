from __future__ import annotations

import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load project env first, then teacher env as fallback (for shared API mode).
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / "model" / "teacher" / ".env.teacher", override=False)

from .settings import AgentSettings
from .runtime import AgentRuntime


def main():
    ap = argparse.ArgumentParser(prog="agent.cli", description="scAgent Runtime (Ark OpenAI-compatible)")
    ap.add_argument("--goal", type=str, required=True)
    ap.add_argument("--h5ad", type=str, required=True)
    ap.add_argument("--groupby", type=str, default="leiden")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_root", type=str, default="")   # optional override
    ap.add_argument("--skip_teacher", action="store_true")
    ap.add_argument("--teacher_cmd", type=str, default="")
    ap.add_argument("--prefer_ser", type=str, default="")
    args = ap.parse_args()

    s = AgentSettings()
    rt = AgentRuntime(s)

    html = rt.run_once(
        goal=args.goal,
        h5ad=args.h5ad,
        groupby=args.groupby,
        device=args.device,
        out_root=(args.out_root.strip() or None),
        skip_teacher=bool(args.skip_teacher),
        teacher_cmd=args.teacher_cmd,
        prefer_ser=args.prefer_ser,
    )
    print(html)


if __name__ == "__main__":
    main()
