"""The contract between planning (LLM) and production (media models).

Every LLM output is parsed into one of these models. If it does not validate,
the error is fed back to the LLM and it tries again (see backends/structured.py).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Platform = Literal["instagram", "youtube_shorts", "tiktok", "youtube"]
AspectRatio = Literal["9:16", "16:9", "1:1"]


class Brief(BaseModel):
    """Phase 1: the idea expanded into constraints every later phase obeys."""

    idea: str = Field(description="The user's original one-line idea")
    platform: Platform = "instagram"
    aspect_ratio: AspectRatio = "9:16"
    duration_s: int = Field(30, ge=5, le=600, description="Target video length in seconds")
    audience: str = Field(description="Who the video is for")
    tone: str = Field(description="Emotional tone, e.g. 'warm and nostalgic'")
    call_to_action: str = Field(description="What the viewer should do at the end")


class Hook(BaseModel):
    """Phase 2: one candidate opening (first 1-3 seconds)."""

    hook: str = Field(description="The opening line or visual moment")
    angle: str = Field(description="The creative angle behind it")
    score: int = Field(ge=1, le=10, description="Self-rated scroll-stopping power")


class HookList(BaseModel):
    hooks: list[Hook] = Field(min_length=3, max_length=10)

    def best(self) -> Hook:
        return max(self.hooks, key=lambda h: h.score)


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
        return self

    def fits(self, brief: Brief, tolerance: float = 0.2) -> bool:
        """True if total length is within ±tolerance of the brief's target."""
        return abs(self.total_s - brief.duration_s) <= brief.duration_s * tolerance
