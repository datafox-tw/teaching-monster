from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import pypandoc
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from gtts import gTTS
from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    course_requirement: str = Field(..., description="Learning objective and scope")
    student_persona: str = Field(..., description="Student background statement")
    enable_subtitles: bool = Field(
        default=True,
        description="Whether to generate subtitle file and burn subtitles into the video",
    )


class GenerateResponse(BaseModel):
    video_url: str
    subtitle_url: str | None = None
    supplementary_url: list[str] | str | None = None


app = FastAPI(title="Teaching Monster Backend (MVP)")
ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = ROOT_DIR / "artifacts"
JOBS_ROOT = ARTIFACTS_ROOT / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_ROOT)), name="artifacts")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()  # local | s3
PRESIGNED_EXPIRES_SECONDS = int(os.getenv("PRESIGNED_EXPIRES_SECONDS", "172800"))  # 48 hours
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("_") or "request"


def _read_duration_seconds(audio_file: Path) -> float:
    # Cross-platform duration probe (Linux/macOS/Windows) without OS-specific tools.
    try:
        clip = AudioFileClip(str(audio_file))
        duration = float(clip.duration or 0.0)
        clip.close()
        if duration > 0:
            return duration
    except Exception:
        pass

    # Optional fallback when ffprobe is available in runtime image.
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = (proc.stdout or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            return 0.0
    return 0.0


def _srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _build_slide_plan(course_requirement: str, student_persona: str) -> list[dict[str, str]]:
    return [
        {
            "page": "1",
            "title": "Learning Goal",
            "text": (
                f"Today we will learn: {course_requirement}. "
                f"This lesson is adapted for this learner profile: {student_persona}. "
                "We will first define the key idea, then connect it to one concrete example."
            ),
        },
        {
            "page": "2",
            "title": "Core Concept",
            "text": (
                "Core explanation of the requested topic: identify the main principle, "
                "the important terms, and the condition where the idea applies correctly."
            ),
        },
        {
            "page": "3",
            "title": "Worked Example",
            "text": (
                "Worked example for the requested topic: walk through one concrete case step by step, "
                "then summarize why the result follows from the core concept."
            ),
        },
    ]


def _subject_from_requirement(course_requirement: str) -> str:
    req = course_requirement.lower()
    if any(k in req for k in ["physics", "newton", "force", "motion", "energy", "momentum"]):
        return "physics"
    if any(k in req for k in ["biology", "cell", "dna", "gene", "evolution", "ecosystem"]):
        return "biology"
    if any(k in req for k in ["computer science", "algorithm", "data structure", "python", "programming"]):
        return "computer_science"
    if any(
        k in req
        for k in [
            "math",
            "mathematics",
            "calculus",
            "algebra",
            "geometry",
            "probability",
            "quadratic",
            "polynomial",
            "function",
            "equation",
            "trigonometry",
            "statistics",
        ]
    ):
        return "mathematics"
    return "physics"


def _build_slide_plan_with_llm(course_requirement: str, student_persona: str) -> list[dict[str, str]] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    prompt = f"""
You are creating an AP-level high-school teaching video script outline.
Constraints:
- Output language must be English.
- Subject must be one of: physics, biology, computer_science, mathematics.
- Return strict JSON only (no markdown), as:
{{
  "subject": "...",
  "slides": [
    {{"page":"1","title":"...","text":"..."}},
    {{"page":"2","title":"...","text":"..."}},
    {{"page":"3","title":"...","text":"..."}}
  ]
}}
- Exactly 3 slides.
- Keep each text around 2-4 concise teaching sentences.
- Include one worked example in slide 3.
- Keep each slide body under 150 words.

Course requirement: {course_requirement}
Student persona: {student_persona}
"""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0.3,
        )
        content = getattr(resp, "output_text", "").strip()
        if not content:
            return None
        data = json.loads(content)
        slides = data.get("slides", [])
        if not isinstance(slides, list) or len(slides) != 3:
            return None
        normalized: list[dict[str, str]] = []
        for i, slide in enumerate(slides, start=1):
            title = str(slide.get("title", f"Slide {i}")).strip()
            text = str(slide.get("text", "")).strip()
            if not text:
                return None
            normalized.append({"page": str(i), "title": title, "text": text})
        return normalized
    except Exception:
        return None


def _truncate_to_word_limit(text: str, max_words: int = 150) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _infer_level_from_persona(student_persona: str) -> str:
    p = student_persona.lower()
    if any(k in p for k in ["beginner", "new to", "no background", "first time"]):
        return "introductory"
    if any(k in p for k in ["ap", "advanced", "strong", "olympiad"]):
        return "advanced"
    return "intermediate"


def _build_blueprint_fallback(
    course_requirement: str,
    student_persona: str,
    subject: str,
    slides: list[dict[str, str]],
) -> dict[str, Any]:
    level = _infer_level_from_persona(student_persona)
    if level == "introductory":
        objectives = [
            "Understand the core definition and vocabulary",
            "Connect concept to one concrete daily-life example",
            "Avoid common beginner misconceptions",
        ]
        misconceptions = [
            "Confusing related terms without context",
            "Assuming one example covers all cases",
        ]
    elif level == "advanced":
        objectives = [
            "Explain the concept with precise terminology",
            "Apply concept across multiple contexts",
            "Compare edge cases and limitations",
        ]
        misconceptions = [
            "Overgeneralizing formula/definition beyond assumptions",
            "Ignoring system boundaries in reasoning",
        ]
    else:
        objectives = [
            "Build a correct conceptual model",
            "Apply concept to one worked example",
            "Identify and correct likely misunderstandings",
        ]
        misconceptions = [
            "Mixing cause and effect in explanations",
            "Skipping conditions required for correct application",
        ]

    covered = [slide["title"] for slide in slides]
    return {
        "subject": subject,
        "language": "English",
        "curriculum_reference": "AP (Advanced Placement)",
        "student_persona": student_persona,
        "difficulty_level": level,
        "teaching_intent": {
            "approach": "scaffolded_progression",
            "objectives": objectives,
        },
        "covered_concepts": covered,
        "misconceptions": misconceptions,
    }


def _build_blueprint_with_llm(
    course_requirement: str,
    student_persona: str,
    subject: str,
    slides: list[dict[str, str]],
) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    prompt = f"""
Create a strict JSON course blueprint for adaptive teaching.
Output JSON only, schema:
{{
  "subject": "physics|biology|computer_science|mathematics",
  "language": "English",
  "curriculum_reference": "AP (Advanced Placement)",
  "student_persona": "...",
  "difficulty_level": "introductory|intermediate|advanced",
  "teaching_intent": {{
    "approach": "...",
    "objectives": ["...", "...", "..."]
  }},
  "covered_concepts": ["...", "..."],
  "misconceptions": ["...", "..."]
}}

Course requirement: {course_requirement}
Student persona: {student_persona}
Detected subject: {subject}
Slides:
{json.dumps(slides, ensure_ascii=False)}
"""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0.2,
        )
        content = getattr(resp, "output_text", "").strip()
        if not content:
            return None
        bp = json.loads(content)
        if not isinstance(bp, dict):
            return None
        required = [
            "subject",
            "language",
            "curriculum_reference",
            "student_persona",
            "difficulty_level",
            "teaching_intent",
            "covered_concepts",
            "misconceptions",
        ]
        if not all(k in bp for k in required):
            return None
        return bp
    except Exception:
        return None


def _slides_markdown(slides: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for slide in slides:
        lines.append(f"# Slide {slide['page']}: {slide['title']}")
        lines.append("")
        lines.append(slide["text"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _url_for_artifact(request: Request, abs_path: Path) -> str:
    rel = abs_path.relative_to(ARTIFACTS_ROOT).as_posix()
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}/artifacts/{rel}"
    return str(request.url_for("artifacts", path=rel))


def _get_s3_client():
    region = os.getenv("S3_REGION", "").strip() or None
    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip() or None
    key = os.getenv("S3_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
    if not key or not secret:
        raise RuntimeError("Missing S3_ACCESS_KEY_ID or S3_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def _upload_and_presign_s3(local_path: Path, object_key: str) -> str:
    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("Missing S3_BUCKET")
    s3 = _get_s3_client()
    s3.upload_file(str(local_path), bucket, object_key)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=PRESIGNED_EXPIRES_SECONDS,
    )


def _publish_artifact(request: Request, local_path: Path, request_id: str) -> str:
    if STORAGE_BACKEND != "s3":
        return _url_for_artifact(request, local_path)
    object_prefix = os.getenv("S3_PREFIX", "teaching-monster").strip().strip("/")
    rel = local_path.relative_to(ARTIFACTS_ROOT).as_posix()
    key = f"{object_prefix}/{request_id}/{rel}"
    return _upload_and_presign_s3(local_path, key)


def _check_s3_ready() -> tuple[bool, str]:
    try:
        bucket = os.getenv("S3_BUCKET", "").strip()
        if not bucket:
            return False, "Missing S3_BUCKET"
        s3 = _get_s3_client()
        test_key = f"{os.getenv('S3_PREFIX', 'teaching-monster').strip().strip('/')}/healthcheck.txt"
        s3.put_object(Bucket=bucket, Key=test_key, Body=b"ok")
        s3.delete_object(Bucket=bucket, Key=test_key)
        return True, "S3 write/delete check passed"
    except Exception as exc:  # pragma: no cover - runtime guard
        return False, f"S3 check failed: {type(exc).__name__}: {exc}"


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _render_slide_image(
    page: int,
    title: str,
    body: str,
    output_path: Path,
    subtitle_text: str | None = None,
) -> None:
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), (243, 247, 252))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=44)
    body_font = ImageFont.load_default(size=28)
    small_font = ImageFont.load_default(size=20)

    # Header block
    draw.rectangle([(0, 0), (width, 96)], fill=(19, 62, 135))
    draw.text((40, 28), f"Slide {page}", fill=(236, 244, 255), font=small_font)

    # Title
    draw.text((64, 132), title, fill=(17, 32, 60), font=title_font)

    # Body with wrapping
    max_body_width = width - 128
    lines = _wrap_text(draw, body, body_font, max_body_width)
    y = 220
    for line in lines:
        draw.text((64, y), line, fill=(26, 44, 76), font=body_font)
        y += 42

    # Optional burned-in subtitle area (bottom center)
    if subtitle_text:
        subtitle_box_top = height - 148
        subtitle_box_bottom = height - 28
        draw.rectangle(
            [(48, subtitle_box_top), (width - 48, subtitle_box_bottom)],
            fill=(0, 0, 0),
        )
        subtitle_lines = _wrap_text(draw, subtitle_text, small_font, width - 160)
        max_lines = 3
        subtitle_lines = subtitle_lines[:max_lines]
        sy = subtitle_box_top + 18
        for line in subtitle_lines:
            bbox = draw.textbbox((0, 0), line, font=small_font)
            line_w = bbox[2] - bbox[0]
            sx = (width - line_w) // 2
            draw.text((sx, sy), line, fill=(255, 255, 255), font=small_font)
            sy += 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def _compose_mp4(timeline: list[dict[str, Any]], output_path: Path) -> float:
    clips = []
    for item in timeline:
        image_path = Path(item["image_file"])
        audio_path = Path(item["audio_file"])
        audio_clip = AudioFileClip(str(audio_path))
        image_clip = ImageClip(str(image_path)).with_duration(audio_clip.duration).with_audio(audio_clip)
        clips.append(image_clip)

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=24,
        logger=None,
    )
    duration = float(final_clip.duration)
    final_clip.close()
    for clip in clips:
        clip.close()
    return duration


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "timestamp": _utc_now().isoformat(),
    }


@app.get("/health/ready")
def health_ready() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    # Storage backend check
    if STORAGE_BACKEND == "s3":
        ok, detail = _check_s3_ready()
        checks["storage"] = {
            "ok": ok,
            "backend": STORAGE_BACKEND,
            "detail": detail,
            "expires_seconds": PRESIGNED_EXPIRES_SECONDS,
        }
    else:
        checks["storage"] = {
            "ok": True,
            "backend": STORAGE_BACKEND,
            "detail": "local static storage enabled",
            "expires_seconds": None,
        }

    # OpenAI key presence check (optional path because fallback exists)
    has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    checks["openai"] = {
        "ok": True,
        "detail": "OPENAI_API_KEY present" if has_openai else "OPENAI_API_KEY missing (fallback mode)",
    }

    # Core dependency/runtime checks
    checks["runtime"] = {
        "ok": True,
        "detail": "moviepy/pillow/gtts modules loaded at startup",
    }

    all_ok = all(v.get("ok", False) for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "timestamp": _utc_now().isoformat(),
        "checks": checks,
    }


def _run_generation(payload: GenerateRequest, request: Request) -> GenerateResponse:
    job_id = _safe_id(payload.request_id)
    run_tag = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    job_dir = JOBS_ROOT / f"{job_id}_{run_tag}"
    audio_dir = job_dir / "audio"
    image_dir = job_dir / "images"
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    slides = _build_slide_plan_with_llm(payload.course_requirement, payload.student_persona)
    if slides is None:
        slides = _build_slide_plan(payload.course_requirement, payload.student_persona)
    # Hard cap: no more than 150 words per slide body.
    for slide in slides:
        slide["text"] = _truncate_to_word_limit(str(slide.get("text", "")), max_words=150)
    detected_subject = _subject_from_requirement(payload.course_requirement)
    blueprint = _build_blueprint_with_llm(
        payload.course_requirement,
        payload.student_persona,
        detected_subject,
        slides,
    )
    if blueprint is None:
        blueprint = _build_blueprint_fallback(
            payload.course_requirement,
            payload.student_persona,
            detected_subject,
            slides,
        )

    # 1) Build markdown + script artifacts.
    slides_md = _slides_markdown(slides)
    markdown_path = job_dir / "slides.md"
    script_path = job_dir / "script.json"
    blueprint_path = job_dir / "course_blueprint.json"
    markdown_path.write_text(slides_md, encoding="utf-8")
    script_path.write_text(json.dumps(slides, indent=2), encoding="utf-8")
    blueprint_path.write_text(json.dumps(blueprint, indent=2), encoding="utf-8")

    # 2) Markdown -> PPTX (best effort).
    pptx_path = job_dir / "slides.pptx"
    pptx_error: str | None = None
    try:
        pypandoc.convert_text(slides_md, to="pptx", format="md", outputfile=str(pptx_path))
    except Exception as exc:  # pragma: no cover - runtime guard
        pptx_error = f"{type(exc).__name__}: {exc}"

    # 3) Per-page TTS MP3.
    timeline: list[dict[str, Any]] = []
    current_start = 0.0
    for idx, slide in enumerate(slides, start=1):
        page_audio = audio_dir / f"page_{idx}.mp3"
        page_image = image_dir / f"page_{idx}.png"
        try:
            gTTS(slide["text"], lang="en").save(str(page_audio))
        except Exception as exc:  # pragma: no cover - runtime guard
            raise HTTPException(
                status_code=502,
                detail=f"TTS generation failed on page {idx}: {type(exc).__name__}: {exc}",
            ) from exc
        _render_slide_image(
            page=idx,
            title=slide["title"],
            body=slide["text"],
            output_path=page_image,
            subtitle_text=slide["text"] if payload.enable_subtitles else None,
        )
        duration = _read_duration_seconds(page_audio)
        if duration <= 0:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Generated audio duration is zero on page {idx}. "
                    "Likely runtime codec/probe issue or empty TTS output."
                ),
            )
        start = current_start
        end = start + duration
        timeline.append(
            {
                "page": idx,
                "title": slide["title"],
                "text": slide["text"],
                "audio_file": str(page_audio),
                "image_file": str(page_image),
                "duration_sec": duration,
                "start_sec": start,
                "end_sec": end,
            }
        )
        current_start = end

    # 4) Merge MP3.
    merged_mp3_path = job_dir / "lesson.mp3"
    with merged_mp3_path.open("wb") as out:
        for item in timeline:
            out.write(Path(item["audio_file"]).read_bytes())
    merged_duration = _read_duration_seconds(merged_mp3_path)

    # 5) Build MP4 from slide images + per-page audio timing.
    video_path = job_dir / "lesson.mp4"
    try:
        video_duration = _compose_mp4(timeline=timeline, output_path=video_path)
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(
            status_code=500,
            detail=f"MP4 composition failed: {type(exc).__name__}: {exc}",
        ) from exc

    # 6) Build SRT (optional).
    srt_path: Path | None = None
    if payload.enable_subtitles:
        srt_path = job_dir / "subtitle.srt"
        srt_lines: list[str] = []
        for i, item in enumerate(timeline, start=1):
            srt_lines.append(str(i))
            srt_lines.append(f"{_srt_time(item['start_sec'])} --> {_srt_time(item['end_sec'])}")
            srt_lines.append(item["text"])
            srt_lines.append("")
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    # 7) Manifest.
    manifest = {
        "request_id": payload.request_id,
        "subject": detected_subject,
        "curriculum_reference": "AP (Advanced Placement)",
        "language": "English",
        "difficulty_level": blueprint.get("difficulty_level", "intermediate"),
        "job_dir": str(job_dir),
        "generated_at": _utc_now().isoformat(),
        "timeline": timeline,
        "merged_audio": {
            "file": str(merged_mp3_path),
            "duration_sec": merged_duration,
            "sum_page_duration_sec": round(sum(item["duration_sec"] for item in timeline), 3),
        },
        "video": {
            "file": str(video_path),
            "duration_sec": round(video_duration, 3),
        },
        "pptx_generated": pptx_path.exists(),
        "pptx_error": pptx_error,
    }
    manifest_path = job_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Competition constraint: supplementary files should not exceed 5 files.
    supplementary_paths: list[Path] = [
        blueprint_path,
        manifest_path,
        markdown_path,
        script_path,
    ]
    if pptx_path.exists():
        supplementary_paths.append(pptx_path)
    supplementary_paths = supplementary_paths[:5]
    supplementary: list[str] = [_publish_artifact(request, p, payload.request_id) for p in supplementary_paths]

    # Ensure downloadable artifacts exist before returning URLs.
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise HTTPException(status_code=500, detail="Generated video file is missing or empty.")
    if payload.enable_subtitles:
        if srt_path is None or (not srt_path.exists()) or srt_path.stat().st_size <= 0:
            raise HTTPException(status_code=500, detail="Generated subtitle file is missing or empty.")

    video_url = _publish_artifact(request, video_path, payload.request_id)
    subtitle_url = _publish_artifact(request, srt_path, payload.request_id) if srt_path else None

    return GenerateResponse(
        video_url=video_url,
        subtitle_url=subtitle_url,
        supplementary_url=supplementary,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
    if request.headers.get("X-Dry-Run", "").lower() == "true":
        # Fast connection test path for platforms with short timeout.
        return GenerateResponse(
            video_url="https://example.com/test.mp4",
            subtitle_url="https://example.com/test.srt",
            supplementary_url=[],
        )
    return _run_generation(payload, request)


@app.post("/v1/video/generate", response_model=GenerateResponse)
def generate_v1(payload: GenerateRequest, request: Request) -> GenerateResponse:
    if request.headers.get("X-Dry-Run", "").lower() == "true":
        # Fast connection test path for platforms with short timeout.
        return GenerateResponse(
            video_url="https://example.com/test.mp4",
            subtitle_url="https://example.com/test.srt",
            supplementary_url=[],
        )
    return _run_generation(payload, request)
