from __future__ import annotations

import importlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "smoke"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_markdown_to_ppt() -> dict:
    result = {"name": "markdown_to_ppt", "ok": False, "detail": ""}
    try:
        import pypandoc

        md = "# Newton's Second Law\n\n- Formula: F = ma\n- Unit: N\n"
        out_file = ARTIFACTS / "sample.pptx"
        pypandoc.convert_text(md, to="pptx", format="md", outputfile=str(out_file))
        result["ok"] = out_file.exists() and out_file.stat().st_size > 0
        result["detail"] = f"created={out_file}"
    except Exception as exc:  # pragma: no cover - smoke script
        result["detail"] = f"{type(exc).__name__}: {exc}"
    return result


def read_duration_seconds(audio_file: Path) -> float:
    cmd = ["afinfo", str(audio_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    for line in proc.stdout.splitlines():
        if "estimated duration" in line:
            value = line.split(":", 1)[1].strip().split(" ")[0]
            return float(value)
    return 0.0


def test_tts_mp3() -> dict:
    result = {"name": "tts_mp3_gtts", "ok": False, "detail": ""}
    try:
        from gtts import gTTS

        out_file = ARTIFACTS / "sample.mp3"
        gTTS("This is a smoke test for Teaching Monster.", lang="en").save(str(out_file))
        duration = read_duration_seconds(out_file)
        result["ok"] = out_file.exists() and out_file.stat().st_size > 0 and duration > 0
        result["detail"] = f"created={out_file}, duration_sec={duration:.3f}"
    except Exception as exc:  # pragma: no cover - smoke script
        result["detail"] = f"{type(exc).__name__}: {exc}"
    return result


def test_openai_sdk() -> dict:
    result = {"name": "openai_sdk", "ok": False, "detail": ""}
    try:
        from openai import OpenAI

        key = os.getenv("OPENAI_API_KEY", "")
        _client = OpenAI(api_key=key or "sk-placeholder")
        result["ok"] = True
        result["detail"] = "OpenAI client instantiated (network call not executed)"
    except Exception as exc:  # pragma: no cover - smoke script
        result["detail"] = f"{type(exc).__name__}: {exc}"
    return result


def test_manim_import() -> dict:
    result = {"name": "manim_import", "ok": False, "detail": ""}
    try:
        importlib.import_module("manim")
        result["ok"] = True
        result["detail"] = "manim importable"
    except Exception as exc:  # pragma: no cover - smoke script
        result["detail"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    checks = [
        test_markdown_to_ppt(),
        test_tts_mp3(),
        test_openai_sdk(),
        test_manim_import(),
    ]
    ok = all(item["ok"] for item in checks if item["name"] != "manim_import")
    report = {
        "generated_at": now_iso(),
        "root": str(ROOT),
        "artifacts_dir": str(ARTIFACTS),
        "checks": checks,
        "overall_ok_excluding_optional_manim": ok,
    }
    report_path = ARTIFACTS / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport written to: {report_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
