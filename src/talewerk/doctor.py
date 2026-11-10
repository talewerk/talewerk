"""Environment checks: is this machine ready to run the Talewerk pipeline?

Standard library only, so `talewerk doctor` works before any heavy
dependency (LangGraph, MLX, FLUX...) is installed.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_LLM_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible API


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def check_python() -> Check:
    ok = sys.version_info >= (3, 11)
    return Check("Python >= 3.11", ok, platform.python_version())


def check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    detail = path or "not found (macOS: brew install ffmpeg)"
    return Check("FFmpeg", path is not None, detail)


def check_platform() -> Check:
    machine = platform.machine()
    apple_silicon = sys.platform == "darwin" and machine == "arm64"
    detail = f"{platform.system()} {machine}"
    if not apple_silicon:
        detail += " (MLX backends need Apple Silicon; cloud backends still work)"
    return Check("Apple Silicon (for MLX)", apple_silicon, detail, required=False)


def check_llm(base_url: str = DEFAULT_LLM_URL, timeout: float = 3.0) -> Check:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id", "?") for m in data.get("data", [])]
        detail = f"{base_url} -> {len(models)} model(s)"
        if models:
            detail += ": " + ", ".join(models[:3])
        return Check("Local LLM server", True, detail, required=False)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return Check(
            "Local LLM server",
            False,
            f"{base_url} unreachable ({exc.__class__.__name__}); start Ollama or vllm-metal",
            required=False,
        )


def run_checks(llm_url: str = DEFAULT_LLM_URL) -> list[Check]:
    return [check_python(), check_ffmpeg(), check_platform(), check_llm(llm_url)]
