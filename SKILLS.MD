# Module I/O JSON Schema

本文件定義 Teaching Monster 各模組的輸入/輸出契約（contract），供 API 編排與模組解耦使用。

## 0) Competition API Contract

### Input (from organizer)
```json
{
  "request_id": "req_20260310_0001",
  "course_requirement": "Teach Grade 10 students Newton's Second Law with 2 worked examples and one misconception check.",
  "student_persona": "I am a 10th-grade student. I know basic algebra but no calculus."
}
```

### Output (to organizer)
```json
{
  "video_url": "https://cdn.example.com/videos/req_20260310_0001.mp4",
  "subtitle_url": "https://cdn.example.com/subtitles/req_20260310_0001.srt",
  "supplementary_url": [
    "https://cdn.example.com/slides/req_20260310_0001.pdf",
    "https://cdn.example.com/notes/req_20260310_0001.md"
  ]
}
```

---

## 1) Request Ingestor

### Input
```json
{
  "request_id": "string",
  "course_requirement": "string",
  "student_persona": "string"
}
```

### Output (`Job`)
```json
{
  "job_id": "job_9f3b2",
  "request_id": "req_20260310_0001",
  "received_at": "2026-03-10T06:30:00Z",
  "status": "RECEIVED",
  "raw_request": {
    "course_requirement": "string",
    "student_persona": "string"
  }
}
```

## 2) Learner Model Builder

### Input
```json
{
  "job_id": "job_9f3b2",
  "student_persona": "string",
  "course_requirement": "string"
}
```

### Output (`LearnerProfile`)
```json
{
  "job_id": "job_9f3b2",
  "grade_band": "high_school",
  "estimated_level": "introductory",
  "language_level": "B2",
  "known_prerequisites": ["basic algebra"],
  "missing_prerequisites": ["vector decomposition"],
  "likely_misconceptions": ["force is needed to keep constant velocity"],
  "teaching_preferences": {
    "pace": "moderate",
    "example_density": "high"
  }
}
```

## 3) Curriculum & Knowledge Retriever

### Input
```json
{
  "job_id": "job_9f3b2",
  "course_requirement": "string",
  "learner_profile": {
    "grade_band": "high_school",
    "estimated_level": "introductory"
  }
}
```

### Output (`KnowledgePack`)
```json
{
  "job_id": "job_9f3b2",
  "topics": ["Newton's Second Law", "Free-body diagram"],
  "retrieved_chunks": [
    {
      "source_id": "phy_ib_mech_001",
      "title": "Forces and Motion",
      "content": "F = ma connects net force to acceleration.",
      "confidence": 0.92
    }
  ],
  "glossary": [
    {"term": "net force", "definition": "vector sum of all forces"}
  ]
}
```

## 4) Pedagogical Planner

### Input
```json
{
  "job_id": "job_9f3b2",
  "course_requirement": "string",
  "learner_profile": {},
  "knowledge_pack": {}
}
```

### Output (`LessonPlan`)
```json
{
  "job_id": "job_9f3b2",
  "target_duration_sec": 720,
  "slides": [
    {
      "page": 1,
      "intent": "hook",
      "title": "Why Do Objects Speed Up?",
      "learning_objective": "Connect force with acceleration intuitively",
      "key_points": ["Every acceleration has a net force cause"],
      "estimated_sec": 45
    },
    {
      "page": 2,
      "intent": "concept_build",
      "title": "Newton's Second Law",
      "learning_objective": "Use F = ma correctly",
      "key_points": ["Net force", "Mass", "Acceleration"],
      "estimated_sec": 80
    }
  ]
}
```

## 5) Quality Gate (Plan)

### Input
```json
{
  "job_id": "job_9f3b2",
  "lesson_plan": {},
  "learner_profile": {}
}
```

### Output
```json
{
  "job_id": "job_9f3b2",
  "is_passed": true,
  "score": {
    "difficulty_alignment": 0.9,
    "coverage": 0.87,
    "scaffolding": 0.91
  },
  "issues": []
}
```

## 6) Script Generator

### Input
```json
{
  "job_id": "job_9f3b2",
  "validated_lesson_plan": {},
  "knowledge_pack": {},
  "learner_profile": {}
}
```

### Output (`DraftSlideScripts`)
```json
{
  "job_id": "job_9f3b2",
  "slides": [
    {
      "page": 1,
      "narration_text": "Have you noticed that a shopping cart speeds up only when you push it?",
      "speaker_notes": "Pause after question for 1 second."
    }
  ]
}
```

## 7) Slide Authoring Engine

### Input
```json
{
  "job_id": "job_9f3b2",
  "lesson_plan": {},
  "draft_slide_scripts": {}
}
```

### Output (`SlideDeckMarkdown`)
```json
{
  "job_id": "job_9f3b2",
  "template_policy": ["title_subtitle", "title_body", "text_only"],
  "slides_markdown": "# Slide 1\n## Why Do Objects Speed Up?\n- Force changes motion\n",
  "slide_assets": [
    {"page": 1, "markdown_ref": "slide_1.md"}
  ]
}
```

## 8) Visual Asset Generator

### Input
```json
{
  "job_id": "job_9f3b2",
  "slides_markdown": "string",
  "lesson_plan": {}
}
```

### Output (`VisualAssets`)
```json
{
  "job_id": "job_9f3b2",
  "assets": [
    {
      "page": 2,
      "asset_type": "diagram",
      "asset_url": "s3://bucket/job_9f3b2/page2_force_diagram.png",
      "caption": "Free-body diagram for a box on a surface"
    }
  ]
}
```

## 9) Script Refiner

### Input
```json
{
  "job_id": "job_9f3b2",
  "draft_slide_scripts": {},
  "visual_assets": {}
}
```

### Output (`FinalSlideScripts`)
```json
{
  "job_id": "job_9f3b2",
  "slides": [
    {
      "page": 2,
      "final_narration_text": "This diagram shows the net force as the vector sum...",
      "pronunciation_hints": ["Newton", "inertia"]
    }
  ]
}
```

## 10) TTS & Subtitle Builder (Per-Slide)

### Input
```json
{
  "job_id": "job_9f3b2",
  "final_slide_scripts": {}
}
```

### Output (`SlideAudios + SlideSubtitles`)
```json
{
  "job_id": "job_9f3b2",
  "slides": [
    {
      "page": 1,
      "audio_url": "s3://bucket/job_9f3b2/audio/page1.mp3",
      "duration_sec": 45.5,
      "subtitle_segments": [
        {
          "start_sec": 0.0,
          "end_sec": 4.2,
          "text": "Have you noticed that a shopping cart speeds up..."
        }
      ]
    }
  ]
}
```

## 11) Timing Mapper & Video Composer

### Input
```json
{
  "job_id": "job_9f3b2",
  "slide_images": [
    {"page": 1, "image_url": "s3://bucket/job_9f3b2/slides/page1.png"}
  ],
  "slide_audios": [
    {"page": 1, "audio_url": "s3://bucket/job_9f3b2/audio/page1.mp3", "duration_sec": 45.5}
  ]
}
```

### Output (`FinalVideoArtifact`)
```json
{
  "job_id": "job_9f3b2",
  "timeline": [
    {"page": 1, "start_sec": 0.0, "end_sec": 45.5}
  ],
  "video_url": "s3://bucket/job_9f3b2/final/video.mp4",
  "subtitle_url": "s3://bucket/job_9f3b2/final/subtitles.srt"
}
```

## 12) Artifact Publisher

### Input
```json
{
  "job_id": "job_9f3b2",
  "video_artifact": {},
  "supplementary_files": [
    "s3://bucket/job_9f3b2/final/slides.pdf"
  ]
}
```

### Output (final competition response)
```json
{
  "video_url": "https://cdn.example.com/videos/req_20260310_0001.mp4",
  "subtitle_url": "https://cdn.example.com/subtitles/req_20260310_0001.srt",
  "supplementary_url": [
    "https://cdn.example.com/slides/req_20260310_0001.pdf"
  ],
  "ready_to_download": true
}
```

---

## Shared Status Schema (Optional but Recommended)
```json
{
  "job_id": "job_9f3b2",
  "status": "PLANNING|SCRIPTING|SLIDE_RENDERING|TTS|COMPOSING|PUBLISHING|DONE|FAILED",
  "progress": 0.0,
  "updated_at": "2026-03-10T06:55:00Z",
  "error": {
    "code": "string",
    "message": "string"
  }
}
```
