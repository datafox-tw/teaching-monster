from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
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
        default=False,
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
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()  # auto | gtts | openai
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")


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
    # Fallback when no LLM is available — produces bare-minimum teacher-mode content.
    topic = course_requirement.strip() or "this topic"
    return [
        {
            "page": "1",
            "title": f"What Is {topic}?",
            "text": (
                f"{topic} is a fundamental concept you need to understand clearly. "
                "The core idea is that every phenomenon has a precise definition that "
                "distinguishes it from related but different concepts. "
                "Pay attention to the exact wording — vague understanding leads to wrong answers."
            ),
            "visual": None,
        },
        {
            "page": "2",
            "title": "The Key Principle",
            "text": (
                f"Here is the central principle of {topic}: the relationship between "
                "cause and effect is governed by a specific rule that only applies under "
                "defined conditions. "
                "Think of it like a recipe — every ingredient and step must be correct, "
                "or the outcome changes. "
                "Identify what stays constant and what varies."
            ),
            "visual": None,
        },
        {
            "page": "3",
            "title": "Worked Example — Step by Step",
            "text": (
                f"Let us apply {topic} to a concrete case. "
                "Step 1: identify the known quantities. "
                "Step 2: choose the correct formula or rule. "
                "Step 3: substitute values carefully, tracking units. "
                "Step 4: interpret the result — does it make physical or logical sense? "
                "This four-step habit prevents most errors."
            ),
            "visual": None,
        },
        {
            "page": "4",
            "title": "Common Mistake",
            "text": (
                f"A frequent mistake with {topic} is applying the rule without checking conditions. "
                "Always verify assumptions first. "
                "If assumptions fail, the conclusion may be invalid."
            ),
            "visual": None,
        },
        {
            "page": "5",
            "title": "Quick Recap",
            "text": (
                f"Recap {topic} in three checks: define it, apply it, and verify assumptions. "
                "Use this checklist whenever you solve related questions."
            ),
            "visual": None,
        },
    ]


def _subject_from_requirement(course_requirement: str) -> str:
    req = course_requirement.lower()
    # Chinese keywords for the known competition topics
    _ZH_BIOLOGY = {"植物", "進化", "保護", "生物", "細胞", "基因", "演化", "生態", "光合", "呼吸作用", "蛋白質"}
    _ZH_PHYSICS = {"物理", "牛頓", "力學", "運動", "能量", "動量", "熱力學", "電磁", "量子", "相對論"}
    _ZH_CS = {"人工智慧", "機器學習", "演算法", "程式", "神經網路", "深度學習", "資料結構", "電腦科學", "迴歸"}
    _ZH_MATH = {"幾何", "函數", "估算", "數學", "微積分", "代數", "機率", "統計", "方程", "三角", "矩陣", "向量", "建模"}
    # English keywords
    _BIOLOGY = {"biology", "cell", "dna", "gene", "evolution", "ecosystem",
                "photosynthesis", "respiration", "protein", "enzyme", "organism",
                "anatomy", "mitosis", "meiosis", "chlorophyll", "atp",
                "metabolism", "osmosis", "diffusion", "plant", "conservation",
                "morphology", "species", "natural selection", "biodiversity"}
    _PHYSICS = {"physics", "newton", "force", "motion", "energy", "momentum",
                "thermodynamics", "wave", "optics", "electricity", "magnetism",
                "quantum", "relativity", "velocity", "acceleration", "torque",
                "circuit", "gravitational"}
    _CS = {"algorithm", "data structure", "python", "programming", "software",
           "machine learning", "artificial intelligence", "neural network",
           "deep learning", "gradient descent", "backpropagation", "llm",
           "natural language processing", "computer vision", "computer science",
           "approaches to ai", "regression", "classification", "supervised",
           "unsupervised", "inference"}
    _MATH = {"math", "mathematics", "calculus", "algebra", "geometry", "probability",
             "quadratic", "polynomial", "function", "equation", "trigonometry",
             "statistics", "integral", "derivative", "matrix", "vector",
             "estimation", "modeling", "geometric", "properties", "relations",
             "regression"}
    for kw in _ZH_BIOLOGY:
        if kw in req:
            return "biology"
    for kw in _ZH_PHYSICS:
        if kw in req:
            return "physics"
    for kw in _ZH_CS:
        if kw in req:
            return "computer_science"
    for kw in _ZH_MATH:
        if kw in req:
            return "mathematics"
    # English — biology before CS so "neural network" in a bio context stays bio
    for kw in _BIOLOGY:
        if kw in req:
            return "biology"
    for kw in _PHYSICS:
        if kw in req:
            return "physics"
    for kw in _CS:
        if kw in req:
            return "computer_science"
    for kw in _MATH:
        if kw in req:
            return "mathematics"
    return "computer_science"


def _build_slide_plan_with_llm(course_requirement: str, student_persona: str) -> list[dict[str, str]] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    prompt = f"""
You are a master AP-level teacher with 20 years of experience. You are speaking directly to
a specific student who needs to understand "{course_requirement}".

Student profile: {student_persona}

=== TEACHING FRAMEWORK — apply to every slide ===
1. HOOK: Open with a Socratic question or a surprising fact that creates genuine curiosity.
   Example opener for a slide on velocity: "Here's a question most students get wrong:
   if a car's speedometer reads 60 mph, does that mean it travelled 60 miles? Not quite..."
2. EXPLAIN: Give the precise definition in plain language, then connect it to an analogy
   drawn from concepts the STUDENT ALREADY KNOWS (read their persona carefully).
3. APPLY: Walk through a concrete worked example with real numbers and every step shown.
   Double-check your arithmetic — a wrong calculation is an immediate scoring failure.
4. REINFORCE: Close each slide with one crisp sentence that states the irreducible core idea.

=== SCAFFOLDING RULE ===
Start from what THIS SPECIFIC STUDENT already knows. Build a bridge from their existing
knowledge to the new concept. If the student knows basic algebra but not calculus, explain
a rate of change as "the slope of the graph" — never as "the derivative" without first
building that bridge. Mis-calibrating to the student is a scoring failure.

=== ABSOLUTE PROHIBITIONS ===
1. NEVER write "In this slide...", "This lesson covers...", "Learning objectives...",
   "Key terms include...", or any sentence that describes the lesson rather than teaching it.
2. NEVER produce outlines, bullet lists of topics, or "first we will... then we will..." framing.
3. NEVER hallucinate formulas, data, citations, statistics, or case studies — only use
   well-established, verifiable knowledge.
4. NEVER write a sentence that could appear unchanged in a lesson on a completely different topic.
5. NEVER leave a worked example with unchecked arithmetic — verify every numerical result.

=== TITLE RULES ===
Titles must name the specific concept being taught on that slide — not the lesson structure.
Maximum 10 words. No generic labels.

BANNED titles (immediate failure — these prove the video is meta, not teaching):
"Learning Goal", "Learning Goals", "Key Learning Points", "Core Concept", "Introduction",
"Overview", "Summary", "Worked Example", "Key Takeaway", "Lesson Objectives",
or any title that could appear unchanged on a slide about a completely different topic.

GOOD titles name the actual fact or mechanism:
  ✓ "Glucose Is the Fuel Cells Actually Burn"
  ✓ "Why Light and Dark Reactions Run in Parallel"
  ✓ "Chlorophyll Absorbs Red and Blue, Reflects Green"
  ✗ "Core Concept" — banned, says nothing specific
  ✗ "Key Learning Points" — banned, describes the lesson not the content

=== SLIDE COUNT ===
Choose 5–8 slides based on how many distinct sub-concepts the topic genuinely requires.
Do not pad. Do not compress unrelated ideas into one slide.
Topic-specific guidance:
- A topic with one core formula/concept (e.g. "conservation of momentum"): 5 slides.
- A topic with multiple interacting mechanisms (e.g. "photosynthesis", "regression"): 7–8 slides.
- A conceptual topic needing comparison of approaches (e.g. "approaches to AI"): 6–7 slides.

=== OUTPUT FORMAT (strict JSON, no markdown fences) ===
All slide titles and text MUST be written in English regardless of the language used in
the course requirement or student persona.

For each slide, decide independently whether a visual will help comprehension:
- Use "manim" when the slide contains: a mathematical equation, geometric shape, graph,
  coordinate system, algorithm step, or proof that benefits from animated illustration.
- Use "image_gen" when the slide explains: a biological structure, physical apparatus,
  chemical process, or system diagram that benefits from a labeled schematic.
- Use null when the slide is: a Socratic hook, pure definition, analogy, or takeaway
  that is already clear as prose — adding a visual would just be decoration.

{{
  "subject": "physics|biology|computer_science|mathematics",
  "slides": [
    {{
      "page": "1",
      "title": "<specific, topic-anchored title — never generic>",
      "text": "<direct teaching prose in English, max 150 words>",
      "visual": null
    }},
    {{
      "page": "2",
      "title": "...",
      "text": "...",
      "visual": {{
        "type": "manim",
        "description": "<precise Manim animation description: what shapes, equations, motions>"
      }}
    }},
    {{
      "page": "3",
      "title": "...",
      "text": "...",
      "visual": {{
        "type": "image_gen",
        "description": "<diagram description: what to label, what process to show>"
      }}
    }}
  ]
}}

Course requirement: {course_requirement}
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
        if not isinstance(slides, list) or not (4 <= len(slides) <= 10):
            return None
        normalized: list[dict] = []
        for i, slide in enumerate(slides, start=1):
            title  = _truncate_title(str(slide.get("title", f"Slide {i}")).strip())
            text   = str(slide.get("text", "")).strip()
            visual = slide.get("visual")  # dict or None
            if not text:
                return None
            # Validate visual spec if present
            if isinstance(visual, dict):
                if visual.get("type") not in ("manim", "image_gen"):
                    visual = None
                elif not visual.get("description", "").strip():
                    visual = None
            else:
                visual = None
            normalized.append({"page": str(i), "title": title, "text": text, "visual": visual})
        return normalized
    except Exception:
        return None


def _truncate_to_word_limit(text: str, max_words: int = 100) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _truncate_title(title: str, max_words: int = 10) -> str:
    words = title.split()
    if len(words) <= max_words:
        return title
    return " ".join(words[:max_words])


def _to_bullets(text: str, max_items: int = 4) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    bullets = [p.strip().rstrip(".") for p in parts if p.strip()]
    if not bullets:
        cleaned = text.strip()
        return [cleaned] if cleaned else []
    return bullets[:max_items]


def _ensure_bullet_count(bullets: list[str], min_items: int = 3, max_items: int = 4) -> list[str]:
    cleaned = [b.strip() for b in bullets if b and b.strip()]
    if not cleaned:
        return ["Key idea", "How it works", "Worked example"]
    if len(cleaned) >= min_items:
        return cleaned[:max_items]

    # If too few bullets, split long text into evenly sized chunks.
    text = " ".join(cleaned)
    words = text.split()
    if len(words) < min_items:
        base = cleaned + [""] * (min_items - len(cleaned))
        return [b if b else "Key point" for b in base][:max_items]

    chunk_size = max(1, len(words) // min_items)
    new_bullets: list[str] = []
    start = 0
    for i in range(min_items):
        end = len(words) if i == min_items - 1 else min(len(words), start + chunk_size)
        segment = " ".join(words[start:end]).strip()
        if segment:
            new_bullets.append(segment)
        start = end
    return new_bullets[:max_items]


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
    topic = course_requirement.strip()
    if level == "introductory":
        objectives = [
            f"Student can state the definition of {topic} in their own words",
            f"Student can connect {topic} to one concrete real-world example",
            f"Student can identify the most common beginner mistake about {topic}",
        ]
        misconceptions = [
            f"Treating {topic} as interchangeable with a related but distinct concept",
            "Assuming the introductory example covers all cases",
        ]
    elif level == "advanced":
        objectives = [
            f"Student can apply {topic} to multi-step problems with correct notation",
            f"Student can explain when and why {topic} fails or has exceptions",
            f"Student can compare {topic} to related advanced concepts",
        ]
        misconceptions = [
            f"Overgeneralizing the formula or rule for {topic} beyond its stated assumptions",
            "Ignoring boundary conditions or system constraints when applying the concept",
        ]
    else:
        objectives = [
            f"Student can describe the core mechanism of {topic}",
            f"Student can work through a concrete example of {topic} step by step",
            f"Student can articulate why a common misconception about {topic} is wrong",
        ]
        misconceptions = [
            f"Confusing the direction of causality in {topic}",
            "Skipping the conditions required for the concept to apply correctly",
        ]

    covered = [slide["title"] for slide in slides]
    return {
        "subject": subject,
        "language": "English",
        "curriculum_reference": "AP (Advanced Placement)",
        "student_persona": student_persona,
        "difficulty_level": level,
        "teaching_intent": {
            "approach": "scaffolded direct instruction: hook → definition → analogy → worked example → reinforcement",
            "zone_of_proximal_development": f"Student brings prior knowledge from their stated background; lesson bridges to {topic}.",
            "objectives": objectives,
        },
        "covered_concepts": covered,
        "misconceptions": misconceptions,
        "accuracy_notes": f"Verify all formulas and worked examples for {topic} against established AP curriculum.",
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
Create a strict JSON teaching blueprint aligned with AP-level educational standards.
This blueprint will be evaluated on four dimensions: content accuracy, teaching logic,
learner adaptability, and cognitive engagement.

Output JSON only, schema:
{{
  "subject": "physics|biology|computer_science|mathematics",
  "language": "English",
  "curriculum_reference": "AP (Advanced Placement)",
  "student_persona": "...",
  "difficulty_level": "introductory|intermediate|advanced",
  "teaching_intent": {{
    "approach": "scaffolded direct instruction: hook → definition → analogy → worked example → reinforcement",
    "zone_of_proximal_development": "<one sentence: what the student knows vs. what they will learn>",
    "objectives": [
      "<student outcome: 'Student can [verb] [specific thing]' — not 'cover' or 'introduce'>",
      ...
    ]
  }},
  "covered_concepts": ["<exact concept as taught in the slides, topic-specific>", ...],
  "misconceptions": [
    "<specific wrong belief students bring to THIS topic — not a generic mixing-of-terms error>",
    ...
  ],
  "accuracy_notes": "<any domain-specific accuracy constraints or common hallucination risks for this topic>"
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
        bullets = slide.get("bullet_points")
        if isinstance(bullets, list) and bullets:
            for bullet in bullets:
                lines.append(f"- {str(bullet).strip()}")
        else:
            for bullet in _to_bullets(str(slide.get("text", ""))):
                lines.append(f"- {bullet}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _ensure_slide_defaults(slides: list[dict[str, str]]) -> None:
    for idx, slide in enumerate(slides, start=1):
        if "bullet_points" not in slide:
            slide["bullet_points"] = _to_bullets(str(slide.get("text", "")))
        if "visual_type" not in slide:
            slide["visual_type"] = "none" if idx == 1 else ("image" if idx == 2 else "code_diagram")
        if "visual_prompt" not in slide:
            if slide["visual_type"] == "image":
                slide["visual_prompt"] = "A clear educational diagram relevant to the concept."
            elif slide["visual_type"] == "code_diagram":
                slide["visual_prompt"] = "A simple visual flow of steps for the worked example."
            elif slide["visual_type"] == "math_animation":
                slide["visual_prompt"] = "A visual sequence showing transformation of equations."
            else:
                slide["visual_prompt"] = ""


def _build_narration_text(slide: dict[str, str]) -> str:
    bullets = slide.get("bullet_points")
    if isinstance(bullets, list) and bullets:
        base = " ".join(str(b).strip() for b in bullets if str(b).strip())
    else:
        base = str(slide.get("text", "")).strip()
    visual_type = str(slide.get("visual_type", "none")).strip().lower()
    if visual_type == "none":
        return base
    if visual_type == "image":
        return (
            f"Let's use this visual to understand the idea. {base} "
            "Focus on how each labeled part connects to the main principle."
        )
    if visual_type == "code_diagram":
        return (
            f"We will follow this step flow together. {base} "
            "Watch how each step leads to the next and why the final result is valid."
        )
    if visual_type == "math_animation":
        return (
            f"Track the transformation carefully as we move through each stage. {base} "
            "Each change follows a specific mathematical rule."
        )
    return base


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


def _synthesize_tts_openai(text: str, out_path: Path) -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for OpenAI TTS")
    client = OpenAI(api_key=api_key)
    speech = client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
        response_format="mp3",
    )
    speech.stream_to_file(str(out_path))


def _synthesize_tts(text: str, out_path: Path) -> str:
    errors: list[str] = []
    providers: list[str]
    if TTS_PROVIDER == "gtts":
        providers = ["gtts"]
    elif TTS_PROVIDER == "openai":
        providers = ["openai"]
    else:
        providers = ["gtts", "openai"]  # auto fallback

    for provider in providers:
        try:
            if provider == "gtts":
                gTTS(text, lang="en").save(str(out_path))
                return "gtts"
            if provider == "openai":
                _synthesize_tts_openai(text, out_path)
                return "openai"
        except Exception as exc:  # pragma: no cover - runtime guard
            errors.append(f"{provider}:{type(exc).__name__}:{exc}")
    raise RuntimeError("TTS failed for all providers: " + " | ".join(errors))


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


def _generate_manim_script_with_llm(description: str, subject: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    prompt = f"""
Write a self-contained Manim (Community Edition v0.18) Python script for this educational visual:
"{description}"

Rules (violations produce unusable output):
- First line must be: from manim import *
- Class name must be exactly: VisualScene (extends Scene)
- Inside construct(), first line: self.camera.background_color = "#1e1e2e"
- Use 3Blue1Brown palette — BLUE_D (#58C4DD), YELLOW (#FFFF00), WHITE, GREEN_B (#83C167)
- Total duration 4–7 seconds (self.play + self.wait only)
- For equations use MathTex; for labels use Text(font_size=28)
- Maximum 30 lines inside construct()
- Do NOT use external image files or custom fonts
- Output ONLY valid Python code — no markdown fences, no explanation

Subject context: {subject}
"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.responses.create(model=OPENAI_MODEL, input=prompt, temperature=0.1)
        code = getattr(resp, "output_text", "").strip()
        # Strip accidental markdown fences
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        return code.strip() if code else None
    except Exception:
        return None


def _render_manim_script(script_code: str, job_dir: Path, idx: int) -> Path | None:
    script_path = job_dir / f"manim_scene_{idx}.py"
    media_dir   = job_dir / f"manim_media_{idx}"
    media_dir.mkdir(exist_ok=True)
    script_path.write_text(script_code, encoding="utf-8")
    try:
        subprocess.run(
            ["manim", "-ql", "-s",
             "--media_dir", str(media_dir),
             str(script_path), "VisualScene"],
            capture_output=True, text=True, timeout=90, check=False,
        )
        png_files = sorted(media_dir.glob("**/*.png"))
        return png_files[0] if png_files else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _generate_dalle_image(description: str, job_dir: Path, idx: int) -> Path | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        client = OpenAI(api_key=api_key)
        dalle_prompt = (
            f"AP-level educational diagram for a student: {description}. "
            "Clean scientific illustration, light background (#F0F4FA), "
            "clear labels on all key components, professional textbook style, "
            "no watermarks, no decorative borders."
        )
        response = client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1792x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        img_path = job_dir / f"visual_{idx}.png"
        urllib.request.urlretrieve(image_url, str(img_path))
        return img_path
    except Exception:
        return None


def _generate_slide_visual(
    visual_spec: dict | None,
    job_dir: Path,
    idx: int,
    subject: str,
) -> Path | None:
    if visual_spec is None:
        return None
    vtype       = visual_spec.get("type", "")
    description = visual_spec.get("description", "").strip()
    if not description:
        return None

    if vtype == "manim":
        script = _generate_manim_script_with_llm(description, subject)
        if script:
            img = _render_manim_script(script, job_dir, idx)
            if img:
                return img
        # Manim unavailable or failed — fall through to image_gen
        return _generate_dalle_image(description, job_dir, idx)

    if vtype == "image_gen":
        return _generate_dalle_image(description, job_dir, idx)

    return None


def _resize_to_zone(src: Path, zone_w: int, zone_h: int) -> Image.Image:
    img = Image.open(src).convert("RGBA")
    img.thumbnail((zone_w, zone_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (zone_w, zone_h), (0, 0, 0, 0))
    x_off = (zone_w - img.width)  // 2
    y_off = (zone_h - img.height) // 2
    canvas.paste(img, (x_off, y_off))
    return canvas.convert("RGB")


def _render_slide_image(
    page: int,
    title: str,
    body: str,
    output_path: Path,
    subtitle_text: str | None = None,
    visual_path: Path | None = None,
) -> None:
    # ── Zone map (1280 × 720) ─────────────────────────────────────────────────
    #
    #  WITHOUT visual             WITH visual
    #  ─────────────────          ─────────────────
    #  0   – 72   header          0   – 72   header
    #  80  – 148  title           80  – 148  title
    #  158 – 490  body text       158 – 295  body text  (shorter)
    #  490 – 518  gap             305 – 510  image zone (centred)
    #  518        divider         510 – 518  gap
    #  530 – 700  subtitle (opt)  518        divider
    #                             530 – 700  subtitle (opt)
    # ─────────────────────────────────────────────────────────────────────────
    width, height  = 1280, 720
    HEADER_H       = 72
    TITLE_Y        = 80
    MARGIN         = 64
    DIVIDER_Y      = 518
    SUB_Y_START    = 530
    SUB_Y_END      = 700

    has_visual    = visual_path is not None and visual_path.exists()
    TEXT_Y_START  = 158
    TEXT_Y_MAX    = 295 if has_visual else 490

    IMAGE_Y_START = 305
    IMAGE_Y_END   = 510
    IMAGE_ZONE_W  = width - MARGIN * 2   # 1152 px
    IMAGE_ZONE_H  = IMAGE_Y_END - IMAGE_Y_START  # 205 px

    canvas = Image.new("RGB", (width, height), (243, 247, 252))
    draw   = ImageDraw.Draw(canvas)

    title_font = ImageFont.load_default(size=40)
    body_font  = ImageFont.load_default(size=26 if has_visual else 27)
    small_font = ImageFont.load_default(size=19)
    badge_font = ImageFont.load_default(size=17)

    # ── Header band ──────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (width, HEADER_H)], fill=(19, 62, 135))
    draw.text((MARGIN, 22), f"  Slide {page}  ", fill=(200, 220, 255), font=badge_font)

    # ── Title (centred, underlined) ───────────────────────────────────────────
    tb    = draw.textbbox((0, 0), title, font=title_font)
    tx    = max(MARGIN, (width - (tb[2] - tb[0])) // 2)
    draw.text((tx, TITLE_Y), title, fill=(17, 32, 60), font=title_font)
    draw.rectangle([(MARGIN, 152), (width - MARGIN, 154)], fill=(19, 62, 135))

    # ── Body text (upper zone) ────────────────────────────────────────────────
    body_line_h    = 34
    max_body_lines = (TEXT_Y_MAX - TEXT_Y_START) // body_line_h
    body_lines: list[str] = []
    raw_lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if raw_lines:
        for raw in raw_lines:
            line = raw if raw.startswith("•") else f"• {raw}"
            body_lines.extend(_wrap_text(draw, line, body_font, width - MARGIN * 2))
    else:
        body_lines = _wrap_text(draw, body, body_font, width - MARGIN * 2)
    lines = body_lines[:max_body_lines]
    y = TEXT_Y_START
    for line in lines:
        draw.text((MARGIN, y), line, fill=(26, 44, 76), font=body_font)
        y += body_line_h

    # ── Visual (lower zone, only when present) ────────────────────────────────
    if has_visual:
        vis_img = _resize_to_zone(visual_path, IMAGE_ZONE_W, IMAGE_ZONE_H)
        # Subtle rounded-rect border behind image
        draw.rectangle(
            [(MARGIN - 4, IMAGE_Y_START - 4), (width - MARGIN + 4, IMAGE_Y_END + 4)],
            fill=(220, 228, 240),
        )
        canvas.paste(vis_img, (MARGIN, IMAGE_Y_START))

    # ── Divider + subtitle (optional) ────────────────────────────────────────
    if subtitle_text:
        draw.rectangle(
            [(MARGIN, DIVIDER_Y), (width - MARGIN, DIVIDER_Y + 1)], fill=(180, 195, 220)
        )
        draw.rectangle([(0, SUB_Y_START - 4), (width, SUB_Y_END)], fill=(18, 24, 40))
        sub_lines = _wrap_text(draw, subtitle_text, small_font, width - MARGIN * 2)[:3]
        total_h   = len(sub_lines) * 28
        sy        = SUB_Y_START + (SUB_Y_END - SUB_Y_START - total_h) // 2
        for line in sub_lines:
            lw = draw.textbbox((0, 0), line, font=small_font)[2]
            draw.text(((width - lw) // 2, sy), line, fill=(230, 240, 255), font=small_font)
            sy += 28

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


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
    _ensure_slide_defaults(slides)
    # Hard cap: no more than 100 words per slide body; one text block with real line breaks.
    for slide in slides:
        text = _truncate_to_word_limit(str(slide.get("text", "")), max_words=100)
        bullets = slide.get("bullet_points")
        if not isinstance(bullets, list) or not bullets:
            bullets = _to_bullets(text, max_items=4)
        bullets = [str(b).strip() for b in bullets if str(b).strip()]
        combined = _truncate_to_word_limit(" ".join(bullets), max_words=100)
        bullets = _to_bullets(combined, max_items=4)
        bullets = _ensure_bullet_count(bullets, min_items=3, max_items=4)
        if not bullets:
            bullets = [combined] if combined else ["Key idea"]
        slide["bullet_points"] = bullets
        slide["text"] = "\n".join(bullets)
        slide["narration_text"] = _truncate_to_word_limit(_build_narration_text(slide), max_words=120)
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
            tts_used = _synthesize_tts(str(slide.get("narration_text", slide["text"])), page_audio)
        except Exception as exc:  # pragma: no cover - runtime guard
            raise HTTPException(
                status_code=502,
                detail=f"TTS generation failed on page {idx}: {type(exc).__name__}: {exc}",
            ) from exc
        visual_path = _generate_slide_visual(
            slide.get("visual"),
            job_dir=job_dir,
            idx=idx,
            subject=detected_subject,
        )
        _render_slide_image(
            page=idx,
            title=slide["title"],
            body=slide["text"],
            output_path=page_image,
            subtitle_text=None,
            visual_path=visual_path,
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
                "narration_text": str(slide.get("narration_text", slide["text"])),
                "visual_type": str(slide.get("visual_type", "none")),
                "visual_prompt": str(slide.get("visual_prompt", "")),
                "visual_file": str(visual_path) if visual_path is not None else None,
                "tts_provider_used": tts_used,
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
