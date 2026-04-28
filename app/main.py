from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.schemas import HealthResponse, RecognizeResponse, TrainPathRequest, TrainReport
from app.services.face_engine import FaceEngine
from app.services.face_store import FaceStore
from app.services.onnx_face_engine import OnnxFaceEngine
from app.services.trainer import FaceTrainer


settings = get_settings()
engine = OnnxFaceEngine(settings) if settings.engine.lower() == "onnx" else FaceEngine(settings)
store = FaceStore(settings)
trainer = FaceTrainer(settings, engine, store)

app = FastAPI(title="InsightFace 人脸识别训练服务", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "web"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    snapshot = store.load()
    return HealthResponse(
        status="ok",
        model_loaded=engine.loaded,
        persons=len(snapshot.persons),
        embeddings=int(snapshot.embeddings.shape[0]),
    )


@app.get("/api/people")
def people():
    return store.list_people()


@app.post("/api/train/path", response_model=TrainReport)
def train_path(payload: TrainPathRequest) -> TrainReport:
    try:
        return trainer.train_from_path(Path(payload.dataset_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/train/upload", response_model=TrainReport)
async def train_upload(file: UploadFile = File(...)) -> TrainReport:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="请上传 zip 格式训练集")
    zip_path = settings.upload_dir / f"{uuid4().hex}.zip"
    zip_path.write_bytes(await file.read())
    try:
        return trainer.train_from_zip(zip_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile = File(...)) -> RecognizeResponse:
    try:
        image = engine.decode_image_bytes(await file.read())
        embeddings = engine.get_all_embeddings(image)
        matches = [store.search(embedding) for embedding in embeddings]
        return RecognizeResponse(faces_detected=len(embeddings), matches=matches)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
