"""Environment-based configuration using Pydantic BaseSettings.

All settings are loaded from environment variables or a .env file,
enabling zero-code-change deployments across environments.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

VALID_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large"}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Whisper STT ---
    whisper_model_size: str = Field(
        default="base",
        description="Whisper model size: tiny, base, small, medium, large",
    )

    @field_validator("whisper_model_size")
    @classmethod
    def validate_model_size(cls, v: str) -> str:
        """Ensure only valid Whisper model sizes are accepted."""
        if v not in VALID_WHISPER_MODELS:
            raise ValueError(f"Invalid model size '{v}'. Must be one of: {VALID_WHISPER_MODELS}")
        return v

    # --- Upload limits ---
    max_file_size_bytes: int = Field(
        default=26_214_400,
        description="Maximum upload file size in bytes (default 25 MB)",
    )

    # --- Rate limiting ---
    rate_limit_per_minute: int = Field(
        default=10,
        description="Max requests per minute per IP address",
    )

    # --- Concurrency ---
    max_concurrent_transcriptions: int = Field(
        default=2,
        description="Max concurrent Whisper inference jobs (prevents OOM)",
    )

    # --- Authentication ---
    api_key: str = Field(
        default="",
        description="API key for authentication (leave empty to disable)",
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")

    # --- Database ---
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/transcripts.db",
        description="Async SQLAlchemy database URL",
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: str = Field(default="logs", description="Directory for log files")

    # --- CORS ---
    cors_origins: str = Field(
        default="*",
        description="Comma-separated allowed CORS origins (use '*' for dev only)",
    )

    # --- Post-processing toggles ---
    enable_filler_removal: bool = Field(
        default=True,
        description="Remove filler words from transcript",
    )
    enable_punctuation_correction: bool = Field(
        default=True,
        description="Apply rule-based punctuation correction",
    )
    enable_speaker_segmentation: bool = Field(
        default=True,
        description="Enable mock speaker segmentation",
    )
    speaker_segment_interval: int = Field(
        default=30,
        description="Speaker segmentation interval in seconds",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Create and return a cached Settings instance."""
    return Settings()
