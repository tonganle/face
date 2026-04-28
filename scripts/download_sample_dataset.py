from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "faces"
MANIFEST_FILE = ROOT / "datasets" / "sample_celebrities_manifest.json"

CELEBRITIES = [
    {"label": "jackie_chan", "title": "Jackie_Chan"},
    {"label": "jet_li", "title": "Jet_Li"},
    {"label": "tony_leung", "title": "Tony_Leung_Chiu-wai"},
    {"label": "gong_li", "title": "Gong_Li"},
    {"label": "zhang_ziyi", "title": "Zhang_Ziyi"},
    {"label": "fan_bingbing", "title": "Fan_Bingbing"},
    {"label": "andy_lau", "title": "Andy_Lau"},
    {"label": "jay_chou", "title": "Jay_Chou"},
    {"label": "leonardo_dicaprio", "title": "Leonardo_DiCaprio"},
    {"label": "taylor_swift", "title": "Taylor_Swift"},
]

FALLBACK_IMAGES = {
    "zhang_ziyi": {
        "source_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Zhang%20Ziyi%20Cannes.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Zhang_Ziyi_Cannes.jpg",
        "description": "Zhang Ziyi at the Cannes Film Festival",
    },
    "fan_bingbing": {
        "source_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Fan%20Bingbing.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Fan_Bingbing.jpg",
        "description": "Fan Bingbing at the Cannes Film Festival",
    },
}


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "InsightFaceSampleDataset/1.0 (local training sample)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "InsightFaceSampleDataset/1.0 (local training sample)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())


def extension_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def sanitize_license(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    failures: list[str] = []

    for celebrity in CELEBRITIES:
        label = celebrity["label"]
        title = celebrity["title"]
        person_dir = DATASET_DIR / label
        person_dir.mkdir(parents=True, exist_ok=True)

        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        try:
            summary = fetch_json(summary_url)
            image_url = summary.get("originalimage", {}).get("source") or summary.get("thumbnail", {}).get("source")
            if not image_url:
                raise RuntimeError("no page image found")

            image_path = person_dir / f"001{extension_from_url(image_url)}"
            download(image_url, image_path)
            manifest.append(
                {
                    "label": label,
                    "wikipedia_title": title,
                    "image_file": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                    "source_url": image_url,
                    "page_url": summary.get("content_urls", {}).get("desktop", {}).get("page"),
                    "description": summary.get("description"),
                    "license_note": sanitize_license(summary.get("extract_html")),
                }
            )
            print(f"downloaded {label}: {image_path}")
        except Exception as exc:
            fallback = FALLBACK_IMAGES.get(label)
            if not fallback:
                failures.append(f"{label}: {exc}")
                print(f"failed {label}: {exc}", file=sys.stderr)
                continue

            try:
                image_url = fallback["source_url"]
                image_path = person_dir / f"001{extension_from_url(image_url)}"
                download(image_url, image_path)
                manifest.append(
                    {
                        "label": label,
                        "wikipedia_title": title,
                        "image_file": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                        "source_url": image_url,
                        "page_url": fallback["page_url"],
                        "description": fallback["description"],
                        "license_note": "See Wikimedia Commons file page for license and attribution details.",
                    }
                )
                print(f"downloaded {label} from fallback: {image_path}")
            except Exception as fallback_exc:
                failures.append(f"{label}: {fallback_exc}")
                print(f"failed {label}: {fallback_exc}", file=sys.stderr)

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
