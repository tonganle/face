from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    data_dir: Path = Field(default=Path("data"), alias="FACE_DATA_DIR")
    insightface_model: str = Field(default="buffalo_l", alias="INSIGHTFACE_MODEL")
    det_size: str = Field(default="640,640", alias="FACE_DET_SIZE")
    threshold: float = Field(default=0.38, alias="FACE_THRESHOLD")
    ctx_id: int = Field(default=-1, alias="FACE_CTX_ID")
    engine: str = Field(default="auto", alias="FACE_ENGINE")

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def det_size_tuple(self) -> tuple[int, int]:
        left, right = self.det_size.split(",", maxsplit=1)
        return int(left.strip()), int(right.strip())


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
