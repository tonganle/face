from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.config import Settings


class OnnxFaceEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_path = settings.data_dir / "onnx_models" / "w600k_r50.onnx"
        self.detector_model_path = settings.data_dir / "onnx_models" / "face_detection_yunet_2023mar.onnx"
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(str(cascade_path))
        self._yunet = None
        if self.detector_model_path.exists() and hasattr(cv2, "FaceDetectorYN_create"):
            self._yunet = cv2.FaceDetectorYN_create(str(self.detector_model_path), "", (320, 320), 0.6, 0.3, 5000)

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        if self._session is not None:
            return
        if not self.model_path.exists():
            raise RuntimeError("ONNX recognition model is missing. Run scripts/download_onnx_models.py first.")
        self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def read_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return image

    def decode_image_bytes(self, payload: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode uploaded image")
        return image

    def get_largest_embedding(self, image: np.ndarray) -> np.ndarray | None:
        faces = self._detect_faces(image)
        if not faces:
            return None
        face = max(faces, key=lambda item: item[2] * item[3])
        return self._embed_crop(image, face)

    def get_all_embeddings(self, image: np.ndarray) -> list[np.ndarray]:
        return [self._embed_crop(image, face) for face in self._detect_faces(image)]

    def _detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._yunet is not None:
            height, width = image.shape[:2]
            self._yunet.setInputSize((width, height))
            _, faces = self._yunet.detect(image)
            if faces is not None and len(faces) > 0:
                results = []
                for face in faces:
                    x, y, w, h = face[:4]
                    results.append((max(0, int(x)), max(0, int(y)), int(w), int(h)))
                return results

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48))
        return [tuple(map(int, face)) for face in faces]

    def _embed_crop(self, image: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray:
        self.load()
        assert self._session is not None
        assert self._input_name is not None

        x, y, width, height = face
        pad = int(max(width, height) * 0.22)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + width + pad)
        y2 = min(image.shape[0], y + height + pad)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Detected face crop is invalid")

        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32")
        blob = (crop - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]
        embedding = self._session.run(None, {self._input_name: blob})[0][0].astype("float32")
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
