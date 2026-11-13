"""Week 2, step 1: storyboard.json -> one keyframe image per shot.

Images first, video later: a still takes seconds and is easy to judge, so we
only spend slow video generation on shots that already look right.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from .schemas import ShotType, Storyboard

DEFAULT_STYLE = "cinematic photograph, natural light, shallow depth of field, rich colour"


def load_storyboard(path: Path) -> Storyboard:
    data = json.loads(path.read_text())
    return Storyboard.model_validate(data.get("storyboard", data))


def render_images(
    storyboard: Storyboard,
    backend,
    out_dir: Path,
    *,
    width: int = 720,
    height: int = 1280,
    seed: int = 7,
    style: str = DEFAULT_STYLE,
    force: bool = False,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    say = progress or (lambda _m: None)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    shots = storyboard.shots[:limit] if limit else storyboard.shots
    for shot in shots:
        if shot.shot_type == ShotType.KINETIC_TEXT:
            say(f"shot {shot.id}: text-only, no image needed")
            continue
        out = img_dir / f"shot_{shot.id:02d}.png"
        prompt = f"{shot.visual_prompt.rstrip('. ')}. {style}" if style else shot.visual_prompt
        entry = {"shot": shot.id, "file": str(out.relative_to(out_dir)), "prompt": prompt,
                 "seed": seed + shot.id, "width": width, "height": height,
                 "model": backend.name}
        if out.exists() and not force:
            say(f"shot {shot.id}: already done, skipping ({out.name})")
            entry["status"] = "skipped"
        else:
            t0 = time.monotonic()
            say(f"shot {shot.id}: generating...")
            backend.generate(prompt, out, seed=seed + shot.id, width=width, height=height)
            entry["status"] = "generated"
            entry["seconds"] = round(time.monotonic() - t0, 1)
            say(f"shot {shot.id}: done in {entry['seconds']}s -> {out.name}")
        manifest.append(entry)
        # Written after every shot, so a crash never loses the record of finished work.
        (out_dir / "render.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
