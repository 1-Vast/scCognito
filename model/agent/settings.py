from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:
    from dotenv import dotenv_values
    from pydantic import BaseModel, ConfigDict

    def SettingsConfigDict(**kwargs):
        return kwargs

    class BaseSettings(BaseModel):
        """
        Fallback settings loader used when pydantic-settings is unavailable.
        Reads .env (if present) and os.environ, then validates with Pydantic.
        """

        model_config = ConfigDict(extra="ignore", populate_by_name=True)

        def __init__(self, **data):
            cfg = getattr(self.__class__, "model_config", {}) or {}
            env_file = cfg.get("env_file", ".env") if isinstance(cfg, dict) else ".env"
            env_enc = cfg.get("env_file_encoding", "utf-8") if isinstance(cfg, dict) else "utf-8"

            merged: dict[str, str] = {}
            env_path = Path(env_file)
            if not env_path.is_absolute():
                env_path = Path.cwd() / env_path
            try:
                if env_path.exists():
                    merged.update({k: v for k, v in dotenv_values(env_path, encoding=env_enc).items() if v is not None})
            except Exception:
                pass

            merged.update(os.environ)
            merged.update(data)
            super().__init__(**merged)


class AgentSettings(BaseSettings):
    """
    Agent settings are loaded from environment/.env.
    Keep extra="ignore" for forward-compatible env expansion.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Provider (Agent). Fallback to TEACHER_* when AGENT_* is absent.
    agent_base_url: str = Field(..., validation_alias=AliasChoices("AGENT_BASE_URL", "TEACHER_BASE_URL"))
    agent_api_key: str = Field(..., validation_alias=AliasChoices("AGENT_API_KEY", "TEACHER_API_KEY"))
    agent_model_id: str = Field(..., validation_alias=AliasChoices("AGENT_MODEL_ID", "TEACHER_MODEL_ID"))

    # IO
    out_root: str = Field(default="outputs", alias="OUT_ROOT")
    agent_kb_dir: str = Field(default="agent/knowledge", alias="AGENT_KB_DIR")

    # RAG
    agent_rag_topk: int = Field(default=3, alias="AGENT_RAG_TOPK")

    # Runtime controls
    pipeline_timeout_sec: int = Field(default=0, alias="PIPELINE_TIMEOUT_SEC")  # 0 = no timeout
    pipeline_tail_lines: int = Field(default=20, alias="PIPELINE_TAIL_LINES")
    session_state_enabled: bool = Field(default=True, alias="AGENT_SESSION_STATE")
    agent_max_turns: int = Field(default=6, alias="AGENT_MAX_TURNS")
    agent_model_max_tokens: int = Field(default=1536, alias="AGENT_MODEL_MAX_TOKENS")
    agent_tool_context_chars: int = Field(default=1800, alias="AGENT_TOOL_CONTEXT_CHARS")
    agent_rag_text_chars: int = Field(default=280, alias="AGENT_RAG_TEXT_CHARS")

    # Heuristics thresholds
    recon_p99_factor: float = Field(default=5.0, alias="HEU_RECON_P99_FACTOR")
    ser_p99_factor: float = Field(default=4.0, alias="HEU_SER_P99_FACTOR")
    cov_p10_min: int = Field(default=1, alias="HEU_COV_P10_MIN")
    strength_p10_min: float = Field(default=0.30, alias="HEU_STRENGTH_P10_MIN")

    # Suggested knobs defaults
    default_lam_ser: float = Field(default=0.50, alias="HEU_DEFAULT_LAM_SER")
    default_mask_ratio: float = Field(default=0.30, alias="HEU_DEFAULT_MASK_RATIO")
    default_spatial_k: int = Field(default=12, alias="HEU_DEFAULT_SPATIAL_K")
    default_attr_k: int = Field(default=12, alias="HEU_DEFAULT_ATTR_K")
    default_conf_floor: float = Field(default=0.60, alias="HEU_DEFAULT_CONF_FLOOR")
    low_conf_floor: float = Field(default=0.50, alias="HEU_LOW_CONF_FLOOR")

    # Adjustments
    recon_mask_ratio: float = Field(default=0.25, alias="HEU_RECON_MASK_RATIO")
    recon_lam_ser: float = Field(default=0.40, alias="HEU_RECON_LAM_SER")
    ser_lam_ser_cap: float = Field(default=0.35, alias="HEU_SER_LAM_SER_CAP")

    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def out_root_abs(self) -> Path:
        return (self.project_root() / self.out_root).resolve()

    def kb_dir_abs(self) -> Path:
        return (self.project_root() / self.agent_kb_dir).resolve()
