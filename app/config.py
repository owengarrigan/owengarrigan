from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    app_name: str = "cctv-ai-edge"
    camera_source: str = Field("./data/test-video.mp4", alias="CAMERA_SOURCE")
    camera_name: str = Field("default-camera", alias="CAMERA_NAME")
    detection_interval: int = Field(10, alias="DETECTION_INTERVAL", ge=1)
    yolo_model_path: str = Field("yolov8n.pt", alias="YOLO_MODEL_PATH")
    person_confidence_threshold: float = Field(
        0.5,
        alias="PERSON_CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    database_path: Path = Field(Path("./data/events.db"), alias="DATABASE_PATH")
    snapshots_dir: Path = Field(Path("./data/events"), alias="SNAPSHOTS_DIR")
    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8000, alias="APP_PORT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_local_storage(self) -> None:
        """Create local data folders used by the edge app."""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so every module sees the same config."""

    return Settings()
