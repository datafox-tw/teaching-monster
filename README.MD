# Teaching Monster: AI 教學影片生成系統藍圖

## 目標
建立一套可全自動接收學習需求並生成教學影片的系統，對齊「教學怪物挑戰賽」規格：
- 輸入：`request_id`、`course_requirement`、`student_persona`
- 輸出：`video_url`、`subtitle_url`（optional）、`supplementary_url`（optional）

## 系統模組切分

### 1) Request Ingestor
- 功能：接收競賽 API JSON，建立任務與追蹤狀態。
- 產出：標準化任務物件 `Job`.

### 2) Learner Model Builder
- 功能：將 `student_persona` 轉成結構化學習者模型（年級、先備知識、易混淆點、語言深度）。
- 產出：`LearnerProfile`.

### 3) Curriculum & Knowledge Retriever
- 功能：依 `course_requirement` + `LearnerProfile` 查詢課程資料庫/知識庫（可先 RAG，後續升級 GraphRAG）。
- 產出：`KnowledgePack`.

### 4) Pedagogical Planner
- 功能：生成分頁教學大綱（每頁目標、重點、教學意圖、預估時長）。
- 產出：`LessonPlan`（slide-by-slide）。

### 5) Quality Gate (Plan)
- 功能：檢查大綱是否符合難度、完整性、鷹架邏輯；不過關就回寫修正。
- 產出：`ValidatedLessonPlan`.

### 6) Script Generator
- 功能：依每頁大綱產生旁白稿與講者備註（先不含圖說）。
- 產出：`DraftSlideScripts`.

### 7) Slide Authoring Engine
- 功能：依固定母片規則生成投影片 Markdown（標題/標題+內文/純文字）。
- 產出：`SlideDeckMarkdown`.

### 8) Visual Asset Generator
- 功能：按需生成每頁圖像或動畫（Manim/繪圖模型），並回填圖說需求。
- 產出：`VisualAssets`.

### 9) Script Refiner
- 功能：將圖說、公式解釋、頁面節奏回寫到旁白稿。
- 產出：`FinalSlideScripts`.

### 10) TTS & Subtitle Builder (Per-Slide)
- 功能：逐頁生成 MP3、取得時長、逐頁字幕時間戳。
- 產出：`SlideAudios`、`SlideSubtitles`.

### 11) Timing Mapper & Video Composer
- 功能：依逐頁時長建立換頁時間軸，將投影片影格+音訊合成 MP4。
- 產出：`FinalVideoArtifact`.

### 12) Artifact Publisher
- 功能：上傳影片/字幕/補充教材，確認可下載後回傳最終 API response。
- 產出：競賽要求的回應 JSON。

## 端到端流程
1. 接收請求並建立 `Job`
2. 建立 `LearnerProfile`
3. 拉取 `KnowledgePack`
4. 生成 `LessonPlan` 並通過 `Quality Gate`
5. 生成每頁腳本、投影片、視覺素材
6. 腳本精修後逐頁 TTS + 字幕
7. 依頁面時長進行合成
8. 上傳產物，確認可下載，回傳 URL

## MVP 建議（先求穩）
- 先用靜態投影片 + 逐頁 TTS + SRT
- 視覺素材採「必要頁才生成」
- RAG 優先於完整 GraphRAG
- 先保證「生成完成再回應」與「成功率」

## 檔案說明
- `README.MD`：系統模組切分與整體架構
- `SKILLS.MD`：各模組 I/O JSON schema（可作為實作契約）

## Sprint 0 (MVP) 開發基線

### 目標
- 先完成後端可執行環境與核心套件可行性驗證，不做前端。

### 已建立內容
- 後端入口：`backend/app/main.py`（`/health`, `/generate`）
- 依賴清單：`requirements.txt`
- 可選依賴：`requirements-optional.txt`（Manim）
- 環境範本：`.env.example`
- Git 忽略規則：`.gitignore`
- 套件驗證腳本：`scripts/smoke/smoke_check.py`

### 本機啟動
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

### 執行 smoke test
```bash
source .venv/bin/activate
python scripts/smoke/smoke_check.py
```

### Manim 備註
- `manim` 不納入 MVP 必選依賴，先列為 optional。
- 啟用前通常需要先裝系統套件：`pkg-config`、`cairo`、`pango`、`ffmpeg`。

### TTS 備註（MP3）
- MVP 預設用 `gTTS` 直接輸出 `mp3`。
- `gTTS` 需要外網連線；若在無網路環境執行，請改為 OpenAI TTS 或本機引擎 + 轉檔方案。

## `/generate` 目前行為（已非 stub）
- 會建立 `artifacts/jobs/<request_id>_<timestamp>/` 任務目錄。
- 產生內容：
  - `course_blueprint.json`（教學意圖、難度、涵蓋概念、misconception）
  - `slides.md`
  - `slides.pptx`
  - `audio/page_*.mp3`（逐頁）
  - `images/page_*.png`（每頁靜態畫面）
  - `lesson.mp3`（合併）
  - `lesson.mp4`（逐頁音訊對應換頁，含 burn-in 英文字幕）
  - `subtitle.srt`
  - `manifest.json`
- API 回傳：
  - `video_url`：指向 `lesson.mp4`
  - `subtitle_url`：指向 `subtitle.srt`
  - `supplementary_url`：包含 `course_blueprint.json` 與其他中間產物/教材
