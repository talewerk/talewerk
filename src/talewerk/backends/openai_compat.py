"""LLM backend for any OpenAI-compatible server.

Works with Ollama (http://localhost:11434/v1), vllm-metal (http://localhost:8000/v1)
and hosted providers that expose the same API.
"""

from __future__ import annotations

import os

DEFAULT_URL = os.environ.get("TALEWERK_LLM_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("TALEWERK_MODEL", "qwen3:8b")


class OpenAICompatLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        api_key: str | None = None,
        temperature: float = 0.7,
        timeout: float = 300.0,
    ) -> None:
        from openai import OpenAI  # imported lazily so `talewerk doctor` stays light

        self.model = model
        self.name = f"openai-compat:{model}"
        self.temperature = temperature
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("TALEWERK_API_KEY", "local"),
            timeout=timeout,
        )
        # Qwen3 "thinks" before answering by default; /no_think skips that, which is
        # much faster for structured planning tasks.
        self._no_think = model.lower().startswith("qwen3")

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        if self._no_think:
            user = user + "\n/no_think"
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""
