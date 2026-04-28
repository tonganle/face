import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.schemas import MatchResult, PersonSummary


STORE_VERSION = "1.0"


@dataclass
class FaceStoreSnapshot:
    persons: list[dict]
    embeddings: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    centroid_labels: np.ndarray


class FaceStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.persons_file = settings.model_dir / "persons.json"
        self.embeddings_file = settings.model_dir / "embeddings.npz"
        self._snapshot: FaceStoreSnapshot | None = None

    def load(self) -> FaceStoreSnapshot:
        if self._snapshot is not None:
            return self._snapshot
        if not self.persons_file.exists() or not self.embeddings_file.exists():
            self._snapshot = FaceStoreSnapshot([], np.empty((0, 512), dtype="float32"), np.array([], dtype="int32"), np.empty((0, 512), dtype="float32"), np.array([], dtype="int32"))
            return self._snapshot

        persons = json.loads(self.persons_file.read_text(encoding="utf-8"))
        payload = np.load(self.embeddings_file)
        self._snapshot = FaceStoreSnapshot(
            persons=persons,
            embeddings=payload["embeddings"].astype("float32"),
            labels=payload["labels"].astype("int32"),
            centroids=payload["centroids"].astype("float32"),
            centroid_labels=payload["centroid_labels"].astype("int32"),
        )
        return self._snapshot

    def save(self, persons: list[dict], embeddings: np.ndarray, labels: np.ndarray) -> None:
        centroids, centroid_labels = self._build_centroids(embeddings, labels)
        self.persons_file.write_text(json.dumps(persons, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(
            self.embeddings_file,
            embeddings=embeddings.astype("float32"),
            labels=labels.astype("int32"),
            centroids=centroids.astype("float32"),
            centroid_labels=centroid_labels.astype("int32"),
            store_version=np.array([STORE_VERSION]),
        )
        self._snapshot = FaceStoreSnapshot(persons, embeddings.astype("float32"), labels.astype("int32"), centroids.astype("float32"), centroid_labels.astype("int32"))

    def clear_cache(self) -> None:
        self._snapshot = None

    def list_people(self) -> list[PersonSummary]:
        snapshot = self.load()
        counts = {int(label): int((snapshot.labels == label).sum()) for label in set(snapshot.labels.tolist())}
        return [
            PersonSummary(
                id=int(person["id"]),
                identity_key=str(person.get("identity_key") or person.get("name") or person["id"]),
                display_name=str(person.get("display_name") or person.get("name") or person.get("identity_key") or person["id"]),
                name=str(person.get("display_name") or person.get("name") or person.get("identity_key") or person["id"]),
                image_count=int(person.get("image_count", 0)),
                embedding_count=counts.get(int(person["id"]), 0),
            )
            for person in snapshot.persons
        ]

    def search(self, embedding: np.ndarray) -> MatchResult:
        snapshot = self.load()
        if snapshot.centroids.size == 0:
            return MatchResult(person_id=None, name=None, score=0.0, threshold=self.settings.threshold, accepted=False)

        scores = snapshot.centroids @ embedding.astype("float32")
        best_index = int(np.argmax(scores))
        score = float(scores[best_index])
        person_id = int(snapshot.centroid_labels[best_index])
        person = next((item for item in snapshot.persons if int(item["id"]) == person_id), None)
        accepted = score >= self.settings.threshold
        identity_key = str(person.get("identity_key") or person.get("name") or person_id) if person else None
        display_name = str(person.get("display_name") or person.get("name") or identity_key) if person else None
        return MatchResult(
            person_id=person_id if accepted else None,
            identity_key=identity_key if accepted else None,
            display_name=display_name if accepted else None,
            name=display_name if accepted else None,
            score=score,
            threshold=self.settings.threshold,
            accepted=accepted,
        )

    @staticmethod
    def _build_centroids(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centroids: list[np.ndarray] = []
        centroid_labels: list[int] = []
        for label in sorted(set(labels.tolist())):
            rows = embeddings[labels == label]
            centroid = rows.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid)
            centroid_labels.append(int(label))

        if not centroids:
            return np.empty((0, 512), dtype="float32"), np.array([], dtype="int32")
        return np.vstack(centroids).astype("float32"), np.array(centroid_labels, dtype="int32")
