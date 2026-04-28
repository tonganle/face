import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from app.config import Settings
from app.schemas import TrainReport
from app.services.face_engine import FaceEngine
from app.services.face_store import STORE_VERSION, FaceStore


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METADATA_FILENAMES = {"person.json", "metadata.json", "identity.json"}


class FaceTrainer:
    def __init__(self, settings: Settings, engine: FaceEngine, store: FaceStore):
        self.settings = settings
        self.engine = engine
        self.store = store

    def train_from_zip(self, zip_path: Path) -> TrainReport:
        target = self.settings.upload_dir / zip_path.stem
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target)
        dataset_root = self._find_dataset_root(target)
        return self.train_from_path(dataset_root)

    def train_from_path(self, dataset_path: Path) -> TrainReport:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Training directory does not exist: {dataset_path}")
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"Training path must be a directory: {dataset_path}")

        persons: list[dict] = []
        embeddings: list[np.ndarray] = []
        labels: list[int] = []
        identity_keys: set[str] = set()
        images_seen = 0
        skipped_images = 0

        person_dirs = [path for path in sorted(dataset_path.iterdir()) if path.is_dir()]
        for person_dir in person_dirs:
            metadata = self._read_metadata(person_dir)
            identity_key = str(metadata.get("identity_key") or metadata.get("external_id") or person_dir.name).strip()
            display_name = str(metadata.get("display_name") or metadata.get("name") or person_dir.name).strip()
            if not identity_key:
                raise ValueError(f"Identity key is empty for directory: {person_dir}")
            if not display_name:
                raise ValueError(f"Display name is empty for directory: {person_dir}")
            if identity_key in identity_keys:
                raise ValueError(f"Duplicate identity_key found: {identity_key}")
            identity_keys.add(identity_key)

            person_id = len(persons)
            image_paths = self._iter_images(person_dir)
            indexed_for_person = 0
            for image_path in image_paths:
                images_seen += 1
                try:
                    image = self.engine.read_image(image_path)
                    embedding = self.engine.get_largest_embedding(image)
                except RuntimeError:
                    raise
                except Exception:
                    embedding = None
                if embedding is None:
                    skipped_images += 1
                    continue
                embeddings.append(embedding)
                labels.append(person_id)
                indexed_for_person += 1

            if indexed_for_person > 0:
                persons.append(
                    {
                        "id": person_id,
                        "identity_key": identity_key,
                        "display_name": display_name,
                        "name": display_name,
                        "directory": person_dir.name,
                        "image_count": len(image_paths),
                    }
                )

        if not embeddings:
            raise ValueError("No usable faces were detected in the training dataset")

        embedding_array = np.vstack(embeddings).astype("float32")
        label_array = np.array(labels, dtype="int32")
        self.store.save(persons, embedding_array, label_array)
        return TrainReport(
            persons=len(persons),
            images_seen=images_seen,
            faces_indexed=len(embeddings),
            skipped_images=skipped_images,
            store_version=STORE_VERSION,
            message="Training complete. Face embedding store has been updated.",
        )

    @staticmethod
    def _iter_images(person_dir: Path) -> list[Path]:
        return [
            path
            for path in sorted(person_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    @staticmethod
    def _read_metadata(person_dir: Path) -> dict:
        for filename in METADATA_FILENAMES:
            metadata_path = person_dir / filename
            if metadata_path.exists():
                return json.loads(metadata_path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _find_dataset_root(path: Path) -> Path:
        children = [child for child in path.iterdir() if child.is_dir()]
        if len(children) == 1 and any(grandchild.is_dir() for grandchild in children[0].iterdir()):
            return children[0]
        return path
