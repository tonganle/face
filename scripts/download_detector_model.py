from __future__ import annotations

import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "onnx_models"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"model exists: {MODEL_PATH}")
        return 0
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "InsightFaceOnnxDownloader/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        MODEL_PATH.write_bytes(response.read())
    print(f"saved: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
