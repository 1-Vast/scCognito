from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import torch

def list_token_jsons(token_dir: Path) -> List[Path]:
    if not token_dir.exists():
        return []
    return sorted(token_dir.glob("*.json"))

def load_teacher_tokens(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_ser_pt(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)