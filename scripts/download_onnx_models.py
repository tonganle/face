from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "onnx_models"
ZIP_PATH = MODEL_DIR / "buffalo_l.zip"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "InsightFaceOnnxDownloader/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rec_model = MODEL_DIR / "w600k_r50.onnx"
    if rec_model.exists():
        print(f"model exists: {rec_model}")
        return 0

    print(f"downloading {BUFFALO_URL}")
    download(BUFFALO_URL, ZIP_PATH)

    extract_dir = MODEL_DIR / "buffalo_l"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(extract_dir)

    matches = list(extract_dir.rglob("w600k_r50.onnx"))
    if not matches:
        raise FileNotFoundError("w600k_r50.onnx not found in buffalo_l.zip")
    shutil.copy2(matches[0], rec_model)
    print(f"saved: {rec_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
