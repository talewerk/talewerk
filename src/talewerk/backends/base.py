"""Backend interfaces. The agent depends only on these, never on a vendor SDK.

Swapping Ollama for vllm-metal, Claude or any other provider means writing one
new class that satisfies the protocol; no agent code changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def complete(
        self, system: str, user: str, *, json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        """Return the model's text reply. json_mode asks for a JSON object;
        temperature overrides the default (low = consistent, high = creative).

        Backends that set `supports_schema = True` also accept `schema=<JSON Schema>`
        and force the reply to match it."""
        ...
