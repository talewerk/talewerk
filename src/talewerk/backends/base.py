"""Backend interfaces. The agent depends only on these, never on a vendor SDK.

Swapping Ollama for vllm-metal, Claude or any other provider means writing one
new class that satisfies the protocol; no agent code changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Return the model's text reply. json_mode asks the model for a JSON object."""
        ...
