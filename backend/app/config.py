"""Central settings + repo-relative paths. All other backend modules import paths from here rather
than re-deriving them, so the repo can be moved/renamed without touching every file."""

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EVIDENCE_DIR = DATA_DIR / "evidence"

GROUNDING_CHECKPOINT = MODELS_DIR / "grounding" / "checkpoints" / "dior_rsvg_finetuned.pth"
GROUNDING_CONFIG = MODELS_DIR / "grounding" / "GroundingDINO_SwinT_OGC.py"
WATER_SEG_CHECKPOINT = MODELS_DIR / "water_segmentation" / "checkpoints" / "water_body_unet_final.pt"
CHANGE_SEG_CHECKPOINT = MODELS_DIR / "change_detection" / "checkpoints" / "semantic_change_unet.pt"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # A typo here (e.g. "Gemini", "grok") used to pass silently as a plain `str` and only surface
    # deep inside a request as an uncaught ValueError -> raw 500, instead of the clean 503 every
    # other provider-failure path gets. Literal makes pydantic reject a bad value at startup.
    llm_provider: Literal["gemini", "groq"] = "gemini"
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    # Alias rather than a pinned version on purpose: a hardcoded "gemini-2.0-flash" here silently
    # rotted when Google retired it, and a dead model is a harder failure than a drifting one.
    # Pin to a specific id (e.g. "gemini-3.6-flash") via GEMINI_MODEL in .env if a run needs to be
    # exactly reproducible.
    gemini_model: str = "gemini-flash-latest"
    # Groq rotates its hosted lineup often -- "llama-3.3-70b-versatile" was retired out from under
    # this default too. Verified tool-calling against the live /models list before picking this one.
    groq_model: str = "openai/gpt-oss-120b"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]

    gee_project_id: Optional[str] = None
    gee_service_account_email: Optional[str] = None
    gee_service_account_key_file: Optional[str] = None

    # PaliGemma fine-tuned by Google on RSVQA-LR. Gated on the Hub, so it needs an HF token whose
    # account has accepted the Gemma license -- see .env.example.
    hf_token: Optional[str] = None
    vqa_model_id: str = "google/paligemma-3b-ft-rsvqa-lr-224"


settings = Settings()

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
