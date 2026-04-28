from pydantic import BaseModel, Field


class TrainPathRequest(BaseModel):
    dataset_path: str = Field(..., examples=["datasets/faces"])
    rebuild: bool = True


class PersonSummary(BaseModel):
    id: int
    identity_key: str
    display_name: str
    name: str
    image_count: int
    embedding_count: int


class TrainReport(BaseModel):
    persons: int
    images_seen: int
    faces_indexed: int
    skipped_images: int
    store_version: str
    message: str


class MatchResult(BaseModel):
    person_id: int | None
    identity_key: str | None = None
    display_name: str | None = None
    name: str | None
    score: float
    threshold: float
    accepted: bool


class RecognizeResponse(BaseModel):
    faces_detected: int
    matches: list[MatchResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    persons: int
    embeddings: int
