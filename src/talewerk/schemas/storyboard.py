"""The contract between planning (LLM) and production (media models).

Every LLM output is parsed into one of these models. If it does not validate,
the error is fed back to the LLM and it tries again (see backends/structured.py).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Platform = Literal["instagram", "youtube_shorts", "tiktok", "youtube"]
AspectRatio = Literal["9:16", "16:9", "1:1"]


MAX_HOOK_WORDS = 12


class Brief(BaseModel):
    """Phase 1: the idea expanded into constraints every later phase obeys."""

    idea: str = Field(description="The user's original one-line idea")
    platform: Platform = "instagram"
    aspect_ratio: AspectRatio = "9:16"
    duration_s: int = Field(30, ge=5, le=600, description="Target video length in seconds")
    audience: str = Field(description="Who the video is for")
    tone: str = Field(description="Emotional tone, e.g. 'warm and nostalgic'")
    call_to_action: str = Field(description="What the viewer should do at the end")


class BriefDraft(BaseModel):
    """What the LLM fills in. The user's own choices (idea, platform, length) are
    added in code, so the model cannot get them wrong."""

    audience: str = Field(description="Who the video is for, specific")
    tone: str = Field(description="Emotional tone, e.g. 'warm and nostalgic'")
    call_to_action: str = Field(description="What the viewer should do at the end")


class Hook(BaseModel):
    """Phase 2: one candidate opening (first 1-3 seconds)."""

    hook: str = Field(description="The opening line, at most 12 words (1-3 seconds spoken)")
    angle: str = Field(description="The creative angle behind it")

    @field_validator("hook")
    @classmethod
    def _short(cls, v: str) -> str:
        n = len(v.split())
        if n > MAX_HOOK_WORDS:
            raise ValueError(f"hook has {n} words; at most {MAX_HOOK_WORDS} (1-3 seconds spoken)")
        return v


class HookList(BaseModel):
    hooks: list[Hook] = Field(min_length=3, max_length=10)


class HookVerdict(BaseModel):
    """The judge's decision: a separate LLM call, so the writer never grades itself."""

    best: int = Field(ge=1, description="Number of the best hook (1-based)")
    reason: str = Field(description="One sentence: why it beats the others")


class Beat(BaseModel):
    """Phase 3: one timed moment of the script."""

    vo_line: str = Field(description="Voiceover line spoken in this beat ('' for none)")
    overlay_text: str = Field("", description="Short on-screen text ('' for none)")
    duration_s: float = Field(gt=0, le=60)


class Script(BaseModel):
    title: str
    beats: list[Beat] = Field(min_length=2)

    @property
    def total_s(self) -> float:
        return sum(b.duration_s for b in self.beats)


class ShotType(str, Enum):
    """Routes each shot to the right (and cheapest adequate) generator."""

    IMAGE_MOTION = "image_motion"  # still image + Ken Burns pan/zoom (cheap, fast)
    GEN_VIDEO = "gen_video"  # image-to-video model, hero shots only (slow)
    KINETIC_TEXT = "kinetic_text"  # animated typography, no generation
    STOCK = "stock"  # licensed stock footage


class Shot(BaseModel):
    """Phase 4: one row of the storyboard."""

    id: int = Field(ge=1)
    duration_s: float = Field(gt=0, le=20)
    shot_type: ShotType = ShotType.IMAGE_MOTION
    visual_prompt: str = Field(description="Prompt for the image/video model")
    camera: str = Field("static", description="e.g. 'slow push-in', 'pan left'")
    overlay_text: str = ""
    vo_line: str = ""


# Words that make an image model fail: many scenes in one picture, or text in the
# picture (image models garble letters; text belongs in overlay_text).
_BANNED_VISUAL = re.compile(
    r"\b(montage|collage|split[- ]screen|overlay|caption|hashtag|logo|slogan|lettering"
    r"|label+ed|written|that reads|that says|reading)\b"
    r"|#\w"
    # quoted words, e.g. a sign saying 'Coffee Lab' (apostrophes inside words are fine)
    r"|(?<!\w)['\"\u2018\u201c]\w[^'\"\u2019\u201d]*['\"\u2019\u201d](?!\w)",
    re.IGNORECASE,
)


class Storyboard(BaseModel):
    title: str
    hook: str
    shots: list[Shot] = Field(min_length=2)

    @property
    def total_s(self) -> float:
        return sum(s.duration_s for s in self.shots)

    @model_validator(mode="after")
    def _check(self) -> Storyboard:
        ids = [s.id for s in self.shots]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(f"shot ids must be 1..N in order, got {ids}")
        gen = sum(1 for s in self.shots if s.shot_type == ShotType.GEN_VIDEO)
        if gen > 3:
            raise ValueError(f"at most 3 gen_video shots allowed (got {gen}); use image_motion")
        for s in self.shots:
            bad = _BANNED_VISUAL.search(s.visual_prompt)
            if bad:
                raise ValueError(
                    f"shot {s.id}: visual_prompt contains '{bad.group(0)}'. Describe ONE "
                    "subject in ONE moment, with no text; put any words in overlay_text"
                )
        for s in (self.shots[0], self.shots[-1]):
            if not s.overlay_text.strip():
                raise ValueError(
                    f"shot {s.id}: first and last shots need overlay_text "
                    "(many viewers watch muted)"
                )
        return self

    def fits(self, brief: Brief, tolerance: float = 0.2) -> bool:
        """True if total length is within ±tolerance of the brief's target."""
        return abs(self.total_s - brief.duration_s) <= brief.duration_s * tolerance

    def with_voice_from(self, script: Script) -> Storyboard:
        """If the model left every vo_line empty, spread the script's lines over
        the shots in order. A video without its voice lines is not usable."""
        if any(s.vo_line.strip() for s in self.shots):
            return self
        lines = [b.vo_line.strip() for b in script.beats if b.vo_line.strip()]
        if not lines:
            return self
        slots: dict[int, list[str]] = {}
        for j, line in enumerate(lines):
            slots.setdefault(j * len(self.shots) // len(lines), []).append(line)
        shots = [s.model_copy(update={"vo_line": " ".join(slots.get(i, []))})
                 for i, s in enumerate(self.shots)]
        return self.model_copy(update={"shots": shots})

    def scaled_to(self, target_s: float) -> Storyboard:
        """Stretch or shrink every shot by the same factor so the total hits the
        target. Timing is arithmetic, so it is done in code, not by the LLM."""
        factor = target_s / self.total_s
        shots = [s.model_copy(update={"duration_s": round(min(20.0, s.duration_s * factor), 1)})
                 for s in self.shots]
        return self.model_copy(update={"shots": shots})
