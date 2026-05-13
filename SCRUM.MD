# Scrum Board (MVP-first)

## Product Goal
在競賽規格下，先交付一條可穩定產出教學影片的後端自動化管線（MVP），再逐步提升教學品質與動畫能力。

## Sprint 0 - Environment & Feasibility

### Sprint Goal
完成可執行開發環境，驗證核心套件可行性，並建立後端最小 API 骨架。

### Done
- [x] 建立 `.venv`
- [x] 建立 `requirements.txt` / `requirements-optional.txt`
- [x] 建立 `.env.example` / `.gitignore`
- [x] 建立 FastAPI MVP 入口（`/health`, `/generate`）
- [x] 完成 smoke test 腳本（Markdown→PPT、TTS、OpenAI SDK、Manim import）
- [x] 輸出驗證報告：`artifacts/smoke/report.json`

### Risks
- `manim` 需額外系統依賴（`pkg-config`, `cairo`, `pango`, `ffmpeg`），暫不納入 MVP 必選路徑。

## Sprint 1 - End-to-End MVP Pipeline

### Sprint Goal
接通 `generate` 任務：輸入 JSON -> 產生簡報 -> 逐頁語音 -> 合成影片 -> 回傳 URL（本機暫存可先）。

### Backlog
- [x] Request parser：驗證 `request_id/course_requirement/student_persona`
- [x] Lesson planner：輸出 slide-by-slide 大綱 JSON
- [x] Script generator：每頁旁白稿
- [x] Markdown slide builder：轉 `pptx`
- [x] Per-slide TTS：輸出音檔與時長
- [x] Subtitle builder：輸出 `srt`
- [x] Video composer：靜態頁面 + 音訊合成 `mp4`
- [x] Artifact publisher：本機 static URL
- [x] Integration test：單一請求走完整管線
- [x] Smoke test：牛頓第三定律（`scripts/smoke/smoke_newton_third.py`）

## Sprint 2 - Prompt Engineering & Teaching Quality

### Sprint Goal
參考 `TeachingMonster-released` 的兩階段流程，提升內容生成可控性、適性化、與教學品質評估。

### Backlog
- [x] 階段一藍圖化：Requirement/Persona -> 結構化 `Course Blueprint`
- [ ] 階段一擴充：Blueprint -> Markdown Outline（可審核）
- [ ] 階段二包裝器：PPT wrapper（slide specs + script）
- [ ] Quality gate：難度對齊、概念涵蓋、迷思檢查（失敗自動重寫）
- [ ] Prompt templates 版本化：系統提示詞、學科提示詞、persona提示詞
- [ ] 評估指標：內容正確率、時長控制、字幕對齊率

### 已落地能力
- `supplementary_url` 已包含 `course_blueprint.json`
- `course_blueprint.json` 含：教學意圖、難度分級、涵蓋概念、misconceptions
- 可依 `student_persona` 自動調整難度（introductory/intermediate/advanced）
