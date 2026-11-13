"""Week 2, step 2: images + voice + on-screen text -> one MP4.

For each shot we build a short clip (slow zoom on the image, text on top, the
voice line underneath), then join the clips. All video work is done by FFmpeg.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schemas import Shot, Storyboard

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",   # macOS
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux (CI)
]
VOICE_PAD_S = 0.4   # breathing room after each voice line
SAMPLE_RATE = 44100


class AssembleError(RuntimeError):
    pass


Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def _draw_block(draw, text: str, font, width: int, y: int, *, from_bottom: bool = False) -> None:
    lines = _wrap(draw, text, font, int(width * 0.86))
    line_h = int(font.size * 1.25)
    if from_bottom:
        y -= line_h * len(lines)
    for i, line in enumerate(lines):
        w = draw.textlength(line, font=font)
        # White text with a black outline stays readable on any picture.
        draw.text(((width - w) / 2, y + i * line_h), line, font=font, fill="white",
                  stroke_width=max(2, font.size // 12), stroke_fill="black")


def make_text_layer(shot: Shot, out: Path, width: int, height: int, *, subtitles: bool) -> Path:
    """A see-through PNG the size of the video, holding this shot's words."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    title = shot.overlay_text.strip()
    sub = shot.vo_line.strip() if subtitles else ""
    if sub and sub.lower().rstrip(".!? ") == title.lower().rstrip(".!? "):
        sub = ""  # do not show the same words twice
    if title:
        _draw_block(draw, title, _font(width // 13), width, int(height * 0.12))
    if sub:
        # Kept above the bottom 20%, where apps place their own buttons.
        _draw_block(draw, sub, _font(width // 20), width, int(height * 0.78), from_bottom=True)
    img.save(out)
    return out


def probe_duration(path: Path, runner: Runner = _run) -> float:
    proc = runner(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "csv=p=0", str(path)])
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError) as exc:
        raise AssembleError(f"could not read the length of {path}") from exc


def segment_command(image: Path, text_png: Path, voice: Path | None, out: Path, *,
                    duration: float, width: int, height: int, fps: int,
                    zoom_in: bool) -> list[str]:
    frames = max(1, round(duration * fps))
    # Slow zoom ("Ken Burns"): gives a still picture a feeling of motion.
    z = f"1+0.10*on/{frames}" if zoom_in else f"1.10-0.10*on/{frames}"
    video = (
        f"[0:v]scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
        f"crop={width * 2}:{height * 2},"
        f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={width}x{height}:fps={fps}[bg];"
        f"[bg][1:v]overlay=0:0,format=yuv420p[v]"
    )
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(image), "-i", str(text_png)]
    if voice is not None:
        cmd += ["-i", str(voice)]
    else:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono"]
    audio = f"[2:a]aresample={SAMPLE_RATE},apad,atrim=0:{duration:.3f}[a]"
    cmd += ["-filter_complex", f"{video};{audio}", "-map", "[v]", "-map", "[a]",
            "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-ar", str(SAMPLE_RATE), "-ac", "1", str(out)]
    return cmd


def assemble_video(
    storyboard: Storyboard,
    out_dir: Path,
    *,
    tts=None,
    width: int = 720,
    height: int = 1280,
    fps: int = 30,
    subtitles: bool = True,
    runner: Runner = _run,
    progress: Callable[[str], None] | None = None,
) -> Path:
    say = progress or (lambda _m: None)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []
    timeline: list[dict] = []
    for n, shot in enumerate(storyboard.shots):
        image = out_dir / "images" / f"shot_{shot.id:02d}.png"
        if not image.exists():
            if shot.shot_type.value != "kinetic_text":
                raise AssembleError(f"{image} is missing. Run `talewerk render` first.")
            image = work / f"plain_{shot.id:02d}.png"
            Image.new("RGB", (width, height), (18, 18, 22)).save(image)

        voice = None
        duration = shot.duration_s
        if tts is not None and shot.vo_line.strip():
            voice = tts.synth(shot.vo_line, work / f"voice_{shot.id:02d}{tts.suffix}")
            spoken = probe_duration(voice, runner)
            # The voice decides the timing: never cut a sentence short.
            duration = max(duration, spoken + VOICE_PAD_S)

        text_png = make_text_layer(shot, work / f"text_{shot.id:02d}.png", width, height,
                                   subtitles=subtitles)
        seg = work / f"seg_{shot.id:02d}.mp4"
        say(f"shot {shot.id}: {duration:.1f}s" + (" + voice" if voice else ""))
        proc = runner(segment_command(image, text_png, voice, seg, duration=duration,
                                      width=width, height=height, fps=fps, zoom_in=n % 2 == 0))
        if proc.returncode != 0 or not seg.exists():
            raise AssembleError(f"ffmpeg failed on shot {shot.id}: "
                                + (proc.stderr or "").strip()[-400:])
        segments.append(seg)
        timeline.append({"shot": shot.id, "seconds": round(duration, 2), "voice": bool(voice)})

    listing = work / "segments.txt"
    listing.write_text("".join(f"file '{s.resolve()}'\n" for s in segments))
    final = out_dir / "video.mp4"
    say("joining clips")
    proc = runner(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                   "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(final)])
    if proc.returncode != 0 or not final.exists():
        raise AssembleError("ffmpeg failed while joining clips: "
                            + (proc.stderr or "").strip()[-400:])
    (out_dir / "video.json").write_text(json.dumps(
        {"file": final.name, "seconds": round(sum(t["seconds"] for t in timeline), 2),
         "voice": getattr(tts, "name", "none"), "shots": timeline}, indent=2))
    return final
