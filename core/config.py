"""Application configuration via Pydantic V2 BaseSettings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Helmet Detection API", description="Human-readable service name.")
    app_version: str = Field(default="1.0.0", description="Semantic version string.")
    debug: bool = Field(default=False, description="Enable debug mode (never True in production).")

    # Model
    model_path: str = Field(default="best.pt", description="Path to the exported pt model file.")
    model_confidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to include a detection.",
    )
    model_iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold used for NMS.",
    )
    model_device: str = Field(
        default="cpu",
        description="Inference device: 'cpu', 'cuda:0', etc.",
    )

    # Rate Limiting
    rate_limit_predict: str = Field(
        default="60/minute",
        description="slowapi rate-limit string for POST /predict.",
    )

    # Google Cloud Logging
    gcp_project_id: str = Field(
        default="",
        description="GCP project ID for Cloud Logging. Empty → stdlib fallback.",
    )
    log_name: str = Field(default="helmet-detection-api", description="Cloud Logging log name.")

    # Local Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR.",
    )
    log_file: str = Field(
        default="app.log",
        description="Path to the .log file written alongside the app (relative to CWD).",
    )

    # Inference Retry
    retry_max_attempts: int = Field(default=3, ge=1, description="Maximum tenacity retry attempts.")
    retry_wait_min_seconds: float = Field(default=0.5, ge=0.0, description="Minimum back-off seconds.")
    retry_wait_max_seconds: float = Field(default=4.0, ge=0.0, description="Maximum back-off seconds.")

    # File Upload
    max_upload_size_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum allowed upload size in bytes.",
    )


# Module-level singleton — import this everywhere instead of re-instantiating.
settings: Settings = Settings()
