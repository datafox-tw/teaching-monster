from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from gtts import gTTS


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "newton_tts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_duration_seconds(audio_file: Path) -> float:
    proc = subprocess.run(
        ["afinfo", str(audio_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if "estimated duration" in line:
            raw = line.split(":", 1)[1].strip().split(" ")[0]
            return float(raw)
    return 0.0


def main() -> int:
    # Slide-by-slide script for the demo topic.
    slides = [
        {
            "page": 1,
            "title": "What Is Newton's Second Law?",
            "text": (
                "Newton's Second Law explains how force changes motion. "
                "It says the net force on an object equals mass times acceleration."
            ),
        },
        {
            "page": 2,
            "title": "Formula and Meaning",
            "text": (
                "The formula is F equals m a. "
                "If force increases, acceleration increases. "
                "If mass increases, acceleration decreases for the same force."
            ),
        },
        {
            "page": 3,
            "title": "Simple Example",
            "text": (
                "If a 2 kilogram cart is pushed with 10 newtons of net force, "
                "its acceleration is 10 divided by 2, which is 5 meters per second squared."
            ),
        },
    ]

    manifest = {"generated_at": iso_now(), "slides": [], "merged_audio": {}}

    # Per-page MP3
    for slide in slides:
        audio_path = OUT_DIR / f"page_{slide['page']}.mp3"
        gTTS(slide["text"], lang="en").save(str(audio_path))
        duration = read_duration_seconds(audio_path)
        manifest["slides"].append(
            {
                "page": slide["page"],
                "title": slide["title"],
                "audio": str(audio_path),
                "duration_sec": duration,
            }
        )

    # Merge MP3 by stream concatenation (MVP/simple path).
    merged_path = OUT_DIR / "newton_merged.mp3"
    with merged_path.open("wb") as merged:
        for slide in manifest["slides"]:
            chunk = Path(slide["audio"]).read_bytes()
            merged.write(chunk)

    merged_duration = read_duration_seconds(merged_path)
    total_pages_duration = sum(item["duration_sec"] for item in manifest["slides"])
    manifest["merged_audio"] = {
        "audio": str(merged_path),
        "duration_sec": merged_duration,
        "sum_of_pages_sec": total_pages_duration,
        "duration_diff_sec": round(merged_duration - total_pages_duration, 3),
    }

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nManifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

