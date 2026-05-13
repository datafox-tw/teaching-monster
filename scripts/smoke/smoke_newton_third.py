from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app

JOBS_DIR = ROOT / "artifacts" / "jobs"


def _latest_job(prefix: str) -> Path:
    matches = sorted(JOBS_DIR.glob(f"{prefix}_*"))
    if not matches:
        raise RuntimeError(f"No job directory found for prefix: {prefix}")
    return matches[-1]


def main() -> int:
    request_id = "req_smoke_newton_third"
    payload = {
        "request_id": request_id,
        "course_requirement": "Teach Newton's Third Law with one daily-life example.",
        "student_persona": "I am a grade 10 student with basic algebra and beginner physics.",
    }

    client = TestClient(app)
    response = client.post("/generate", json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"/generate failed: status={response.status_code}, body={response.text}")

    body = response.json()
    video_url = body.get("video_url", "")
    subtitle_url = body.get("subtitle_url", "")
    if not video_url.endswith(".mp4"):
        raise RuntimeError(f"Expected mp4 video_url, got: {video_url}")
    if not subtitle_url.endswith(".srt"):
        raise RuntimeError(f"Expected srt subtitle_url, got: {subtitle_url}")

    job_dir = _latest_job(request_id)
    manifest_path = job_dir / "manifest.json"
    video_path = job_dir / "lesson.mp4"
    srt_path = job_dir / "subtitle.srt"
    if not manifest_path.exists():
        raise RuntimeError("manifest.json not found")
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError("lesson.mp4 missing or empty")
    if not srt_path.exists() or srt_path.stat().st_size == 0:
        raise RuntimeError("subtitle.srt missing or empty")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timeline = manifest.get("timeline", [])
    timeline_text = " ".join(str(item.get("text", "")) for item in timeline).lower()
    if "third law" not in timeline_text and "equal and opposite" not in timeline_text:
        raise RuntimeError("Generated content does not appear to cover Newton's Third Law")

    print("Smoke test passed")
    print(f"job_dir={job_dir}")
    print(f"video={video_path} ({video_path.stat().st_size} bytes)")
    print(f"subtitle={srt_path} ({srt_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
