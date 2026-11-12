"""LLM backend for Ollama's native API (/api/chat).

Why not the OpenAI-compatible endpoint: the native API can
  1. turn "thinking" off (think: false), which saves minutes per run on Qwen3, and
  2. force the reply to match a JSON Schema (format: <schema>), so the model
     cannot return broken or wrongly shaped JSON.
Standard library only: no extra dependency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("TALEWERK_OLLAMA_URL", "http://localhost:11434")


class OllamaError(RuntimeError):
    pass


class OllamaLLM:
    supports_schema = True

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = DEFAULT_URL,
        temperature: float = 0.7,
        timeout: float = 600.0,
        think: bool = False,
    ) -> None:
        # Accept ".../v1" too, so the same --llm-url works for both backends.
        base_url = base_url.rstrip("/")
        base_url = base_url.removesuffix("/v1")
        self.base_url = base_url
        self.model = model
        self.name = f"ollama:{model}"
        self.temperature = temperature
        self.timeout = timeout
        self.think = think

    def complete(
        self, system: str, user: str, *, json_mode: bool = False,
        temperature: float | None = None, schema: dict | None = None,
    ) -> str:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature if temperature is None else temperature},
        }
        if schema is not None:
            body["format"] = schema
        elif json_mode:
            body["format"] = "json"
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ConnectionError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        return data.get("message", {}).get("content", "")
