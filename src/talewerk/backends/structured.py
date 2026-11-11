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


def generate_structured(
    llm: LLMBackend,
    system: str,
    user: str,
    model_cls: type[T],
    *,
    retries: int = 2,
) -> T:
    schema = json.dumps(model_cls.model_json_schema(), indent=None)
    sys_prompt = (
        f"{system}\n\nReply with ONE JSON object only, no prose, matching this JSON Schema:\n"
        f"{schema}"
    )
    prompt = user
    last_error = ""
    for attempt in range(retries + 1):
        raw = llm.complete(sys_prompt, prompt, json_mode=True)
        try:
            return model_cls.model_validate_json(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
            log.warning("attempt %d/%d invalid %s: %s", attempt + 1, retries + 1,
                        model_cls.__name__, last_error.splitlines()[0])
            prompt = (
                f"{user}\n\nYour previous reply was invalid:\n{last_error}\n"
                f"Previous reply:\n{raw[:2000]}\n\nReturn a corrected JSON object."
            )
    raise StructuredOutputError(
        f"{model_cls.__name__}: no valid output after {retries + 1} attempts: {last_error}"
    )
