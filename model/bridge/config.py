from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BridgeConfig:
    base_dir: Path
    token_dir: Path
    out_dir: Path
    default_h5ad: Path
    default_cluster_key: str = "leiden"
    conf_floor: float = 0.6
    normalize_cluster_weights: bool = True

def default_config(base_dir: Path) -> BridgeConfig:
    return BridgeConfig(
        base_dir=base_dir,
        token_dir=base_dir / "outputs" / "teacher_outputs",
        out_dir=base_dir / "outputs" / "ser_outputs",
        default_h5ad=base_dir / "pbmc.h5ad",
        default_cluster_key="leiden",
        conf_floor=0.6,
        normalize_cluster_weights=True,
    )
