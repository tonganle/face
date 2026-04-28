from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.services.onnx_face_engine import OnnxFaceEngine


class FaceEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._app = None

    @property
    def loaded(self) -> bool:
        return self._app is not None

    def load(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            fallback = OnnxFaceEngine(self.settings)
            fallback.load()
            self.__class__ = OnnxFaceEngine
            self.__dict__ = fallback.__dict__
            return
        app = FaceAnalysis(name=self.settings.insightface_model)
        app.prepare(ctx_id=self.settings.ctx_id, det_size=self.settings.det_size_tuple)
        self._app = app

    def read_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        return image

    def decode_image_bytes(self, payload: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解析上传的图片")
        return image

    def get_faces(self, image: np.ndarray):
        self.load()
        assert self._app is not None
        faces = self._app.get(image)
        return sorted(faces, key=lambda face: self._area(face.bbox), reverse=True)

    def get_largest_embedding(self, image: np.ndarray) -> np.ndarray | None:
        faces = self.get_faces(image)
        if not faces:
            return None
        return self._normalize(faces[0].embedding)

    def get_all_embeddings(self, image: np.ndarray) -> list[np.ndarray]:
        return [self._normalize(face.embedding) for face in self.get_faces(image)]

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = embedding.astype("float32")
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    @staticmethod
    def _area(bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))
