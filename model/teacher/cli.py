# teacher/cli.py
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .providers.ark_client import ArkChatClient
from .llm_teacher import LLMSemanticTeacher, run_teacher


# -----------------------------
# Load Teacher-only env
#   - Avoid mixing with Agent env in project_root/.env
# -----------------------------
_THIS_DIR = Path(__file__).resolve().parent
load_dotenv(_THIS_DIR / ".env.teacher", override=True)


# -----------------------------
# Knowledge paths
# -----------------------------
@dataclass
class KnowledgePaths:
    go_obo: Path
    goa_gaf: Path
    hallmark_gmt: Path
    reactome_gmt: Path


def resolve_default_knowledge_paths(knowledge_root: Path) -> KnowledgePaths:
    """
    Expected structure:
      knowledge/go/go-basic.obo
      knowledge/go/goa_human.gaf
      knowledge/msigdb/hallmark.gmt
      knowledge/reactome/ReactomePathways.gmt
    """
    return KnowledgePaths(
        go_obo=knowledge_root / "go" / "go-basic.obo",
        goa_gaf=knowledge_root / "go" / "goa_human.gaf",
        hallmark_gmt=knowledge_root / "msigdb" / "hallmark.gmt",
        reactome_gmt=knowledge_root / "reactome" / "ReactomePathways.gmt",
    )


def ensure_file_exists(p: Path, hint: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}\nHint: {hint}")


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teacher.cli",
        description="Run LLM Teacher (markers -> enrichment -> semantic tokens) on an .h5ad dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run teacher on a given .h5ad")
    run.add_argument("--h5ad", type=str, required=True, help="Path to input .h5ad")
    run.add_argument("--out-dir", type=str, required=True, help="Output directory for teacher results")

    # Ark config (allow env fallback)
    run.add_argument("--base-url", type=str, default=None, help="Ark base_url (fallback: TEACHER_BASE_URL)")
    run.add_argument("--model-id", type=str, default=None, help="Ark endpoint model id (fallback: TEACHER_MODEL_ID)")
    run.add_argument("--api-key-env", type=str, default="TEACHER_API_KEY", help="Env var name for API key")
    run.add_argument("--no-web-search", action="store_true", help="Disable web_search tool (if supported)")
    run.add_argument("--web-search-max-keyword", type=int, default=2, help="web_search max_keyword")

    # Knowledge
    run.add_argument(
        "--knowledge-root",
        type=str,
        default=None,
        help="knowledge folder (contains go/msigdb/reactome). Default: teacher/knowledge next to this file.",
    )
    run.add_argument("--go-obo", type=str, default=None, help="Override go-basic.obo path")
    run.add_argument("--goa-gaf", type=str, default=None, help="Override goa_human.gaf path")
    run.add_argument("--hallmark-gmt", type=str, default=None, help="Override hallmark.gmt path")
    run.add_argument("--reactome-gmt", type=str, default=None, help="Override ReactomePathways.gmt path")
    run.add_argument("--output-name", type=str, default=None, help="Custom output JSON filename")
    run.add_argument("--max-llm-calls", type=int, default=24, help="Maximum LLM calls for teacher generation")
    run.add_argument("--disable-privacy-guard", action="store_true", help="Disable payload privacy guard")
    run.add_argument("--debug-checks", action="store_true", help="Enable extra debug checks/logs")

    # Analysis
    run.add_argument("--groupby", type=str, default="cell_type", help="obs column for labels (or leiden if missing)")

    # Demo preset
    demo = sub.add_parser("demo", help="Shortcut preset for a demo dataset test")
    demo.add_argument("--h5ad", type=str, default="../data/demo_dataset.h5ad", help="Demo h5ad path")
    demo.add_argument("--out-dir", type=str, default="../outputs/demo_teacher", help="Output directory")
    demo.add_argument("--base-url", type=str, default=None, help="Ark base_url (fallback: TEACHER_BASE_URL)")
    demo.add_argument("--model-id", type=str, default=None, help="Ark endpoint model id (fallback: TEACHER_MODEL_ID)")
    demo.add_argument("--api-key-env", type=str, default="TEACHER_API_KEY", help="Env var name for API key")
    demo.add_argument("--no-web-search", action="store_true", help="Disable web_search tool (if supported)")
    demo.add_argument("--web-search-max-keyword", type=int, default=2, help="web_search max_keyword")
    demo.add_argument("--knowledge-root", type=str, default=None, help="knowledge folder override")
    demo.add_argument("--groupby", type=str, default="cell_type", help="obs column for labels")
    demo.add_argument("--max-llm-calls", type=int, default=24, help="Maximum LLM calls for teacher generation")
    demo.add_argument("--disable-privacy-guard", action="store_true", help="Disable payload privacy guard")
    demo.add_argument("--debug-checks", action="store_true", help="Enable extra debug checks/logs")

    return parser


def run_pipeline(
    h5ad_path: Path,
    out_dir: Path,
    base_url: Optional[str],
    model_id: Optional[str],
    api_key_env: str,
    enable_web_search: bool,
    web_search_max_keyword: int,
    knowledge_root: Optional[Path],
    go_obo: Optional[Path],
    goa_gaf: Optional[Path],
    hallmark_gmt: Optional[Path],
    reactome_gmt: Optional[Path],
    groupby: str,
    output_name: Optional[str] = None,
    max_llm_calls: int = 24,
    privacy_guard: bool = True,
    debug_checks: bool = False,
) -> None:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {api_key_env}")

    # Env fallback
    base_url = base_url or os.getenv("TEACHER_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    model_id = model_id or os.getenv("TEACHER_MODEL_ID", "")
    if not model_id:
        raise RuntimeError("Missing model id. Provide --model-id or set TEACHER_MODEL_ID in teacher/.env.teacher")

    # Knowledge root default
    if knowledge_root is None:
        knowledge_root = Path(__file__).resolve().parent / "knowledge"
    knowledge_root = knowledge_root.resolve()

    kp = resolve_default_knowledge_paths(knowledge_root)
    if go_obo is not None:
        kp.go_obo = go_obo
    if goa_gaf is not None:
        kp.goa_gaf = goa_gaf
    if hallmark_gmt is not None:
        kp.hallmark_gmt = hallmark_gmt
    if reactome_gmt is not None:
        kp.reactome_gmt = reactome_gmt

    ensure_file_exists(h5ad_path, "Check your --h5ad path.")
    ensure_file_exists(kp.go_obo, "Put go-basic.obo under knowledge/go/ or pass --go-obo.")
    ensure_file_exists(kp.goa_gaf, "Put goa_human.gaf under knowledge/go/ or pass --goa-gaf.")
    ensure_file_exists(kp.hallmark_gmt, "Put hallmark.gmt under knowledge/msigdb/ or pass --hallmark-gmt.")
    ensure_file_exists(kp.reactome_gmt, "Put ReactomePathways.gmt under knowledge/reactome/ or pass --reactome-gmt.")

    out_dir.mkdir(parents=True, exist_ok=True)

    llm = ArkChatClient(
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        enable_web_search=enable_web_search,
        web_search_max_keyword=web_search_max_keyword,
    )

    teacher = LLMSemanticTeacher(
        llm_client=llm,
        go_obo_path=str(kp.go_obo),
        goa_gaf_path=str(kp.goa_gaf),
        hallmark_gmt_path=str(kp.hallmark_gmt),
        reactome_gmt_path=str(kp.reactome_gmt),
        max_llm_calls=int(max_llm_calls),
        privacy_guard=bool(privacy_guard),
        debug_checks=bool(debug_checks),
    )

    out_json = run_teacher(
        teacher=teacher,
        h5ad_path=str(h5ad_path),
        out_dir=str(out_dir),
        groupby=str(groupby),
        output_name=output_name,
    )
    print(f"[OK] Done. JSON saved to: {out_json}")


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(
            h5ad_path=Path(args.h5ad),
            out_dir=Path(args.out_dir),
            base_url=args.base_url,
            model_id=args.model_id,
            api_key_env=args.api_key_env,
            enable_web_search=(not args.no_web_search),
            web_search_max_keyword=int(args.web_search_max_keyword),
            knowledge_root=Path(args.knowledge_root) if args.knowledge_root else None,
            go_obo=Path(args.go_obo) if args.go_obo else None,
            goa_gaf=Path(args.goa_gaf) if args.goa_gaf else None,
            hallmark_gmt=Path(args.hallmark_gmt) if args.hallmark_gmt else None,
            reactome_gmt=Path(args.reactome_gmt) if args.reactome_gmt else None,
            groupby=str(args.groupby),
            output_name=args.output_name,
            max_llm_calls=int(args.max_llm_calls),
            privacy_guard=(not args.disable_privacy_guard),
            debug_checks=bool(args.debug_checks),
        )
        return

    if args.command == "demo":
        run_pipeline(
            h5ad_path=Path(args.h5ad),
            out_dir=Path(args.out_dir),
            base_url=args.base_url,
            model_id=args.model_id,
            api_key_env=args.api_key_env,
            enable_web_search=(not args.no_web_search),
            web_search_max_keyword=int(args.web_search_max_keyword),
            knowledge_root=Path(args.knowledge_root) if args.knowledge_root else None,
            go_obo=None,
            goa_gaf=None,
            hallmark_gmt=None,
            reactome_gmt=None,
            groupby=str(args.groupby),
            output_name=None,
            max_llm_calls=int(args.max_llm_calls),
            privacy_guard=(not args.disable_privacy_guard),
            debug_checks=bool(args.debug_checks),
        )
        return


if __name__ == "__main__":
    main()
