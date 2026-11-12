"""Get a validated Pydantic object out of an LLM, retrying with error feedback.

Small local models sometimes return broken or incomplete JSON. Instead of
failing, we show the model its own validation errors and ask it to fix them.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .base import LLMBackend

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(RuntimeError):
    pass


def extract_json(text: str) -> str:
    """Strip reasoning blocks and code fences; return the outermost {...} object."""
    text = _THINK.sub("", text).strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in reply")
    return text[start : end + 1]


def inline_refs(schema: dict) -> dict:
    """Replace every {"$ref": "#/$defs/X"} with the definition itself.
    Nested Pydantic models produce $refs; inlined schemas work with more backends."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def _field_guide(schema: dict) -> str:
    """Short 'name: description' list of top-level fields, easier for small models
    than a full JSON Schema."""
    lines = []
    for name, prop in schema.get("properties", {}).items():
        desc = prop.get("description", "")
        lines.append(f"- {name}" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for e in exc.errors()[:3]:
            loc = ".".join(str(x) for x in e["loc"]) or "(root)"
            parts.append(f"{loc}: {e['msg']}")
        return "; ".join(parts)
    return str(exc).splitlines()[0]


def generate_structured(
    llm: LLMBackend,
    system: str,
    user: str,
    model_cls: type[T],
    *,
    retries: int = 2,
    temperature: float | None = None,
) -> T:
    schema = model_cls.model_json_schema()
    native = getattr(llm, "supports_schema", False)
    if native:
        # The backend forces the JSON shape, so the prompt only explains the fields.
        sys_prompt = f"{system}\n\nReply in JSON with these fields:\n{_field_guide(schema)}"
    else:
        sys_prompt = (
            f"{system}\n\nReply with ONE JSON object only, no prose, matching this JSON Schema:\n"
            f"{json.dumps(schema)}"
        )
    prompt = user
    last_error = ""
    for attempt in range(retries + 1):
        kwargs = {"json_mode": True, "temperature": temperature}
        if native:
            kwargs["schema"] = inline_refs(schema)
        raw = llm.complete(sys_prompt, prompt, **kwargs)
        try:
            return model_cls.model_validate_json(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            last_error = _short_error(exc)
            log.warning("attempt %d/%d invalid %s: %s", attempt + 1, retries + 1,
                        model_cls.__name__, last_error)
            prompt = (
                f"{user}\n\nYour previous reply was invalid: {last_error}\n"
                f"Previous reply:\n{raw[:2000]}\n\nReturn a corrected JSON object."
            )
    raise StructuredOutputError(
        f"{model_cls.__name__}: no valid output after {retries + 1} attempts: {last_error}"
    )
