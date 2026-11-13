"""Command-line entry point: `talewerk`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .doctor import DEFAULT_LLM_URL, run_checks


def _doctor(args: argparse.Namespace) -> int:
    checks = run_checks(args.llm_url)
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.required else "warn")
        print(f"[{mark}] {c.name}: {c.detail}")
    failed = [c for c in checks if c.required and not c.ok]
    print("\nReady." if not failed else f"\n{len(failed)} required check(s) failed.")
    return 1 if failed else 0


def _plan(args: argparse.Namespace) -> int:
    # Heavy imports only when planning, so `talewerk doctor` stays instant.
    import logging

    import openai

    from .agent import run_plan
    from .backends import OllamaError, OllamaLLM, OpenAICompatLLM, StructuredOutputError

    logging.basicConfig(level=logging.WARNING, format="  ! %(message)s")
    if args.backend == "ollama":
        llm = OllamaLLM(model=args.model, base_url=args.llm_url)
    else:
        llm = OpenAICompatLLM(model=args.model, base_url=args.llm_url)
    start = time.monotonic()

    def say(msg: str) -> None:
        print(f"[{time.monotonic() - start:5.1f}s] {msg}", file=sys.stderr)

    say(f"planning with {llm.name}")
    try:
        state = run_plan(llm, args.idea, platform=args.platform,
                         duration_s=args.duration, progress=say)
    except StructuredOutputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (openai.APIConnectionError, ConnectionError):
        print(f"error: cannot reach the LLM server at {args.llm_url}\n"
              "  Is Ollama running? Try: brew services start ollama, then talewerk doctor",
              file=sys.stderr)
        return 3
    except OllamaError as exc:
        hint = f"\n  Try: ollama pull {args.model}" if "not found" in str(exc) else ""
        print(f"error: {exc}{hint}", file=sys.stderr)
        return 3
    except openai.NotFoundError:
        print(f"error: model '{args.model}' not found on the server.\n"
              f"  Try: ollama pull {args.model}", file=sys.stderr)
        return 3

    sb, brief = state["storyboard"], state["brief"]
    result = {
        "brief": brief.model_dump(),
        "hooks": [h.model_dump() for h in state["hooks"]],
        "chosen_hook": {"hook": state["hook"].hook, "reason": state.get("hook_reason", "")},
        "script": state["script"].model_dump(),
        "storyboard": sb.model_dump(mode="json"),
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{sb.title}  ({sb.total_s:.1f}s, target {brief.duration_s}s, "
          f"{len(sb.shots)} shots)")
    print(f"Hook: {sb.hook}\n")
    for s in sb.shots:
        print(f"  {s.id:>2}. [{s.duration_s:4.1f}s {s.shot_type.value:<12}] {s.visual_prompt}")
        if s.vo_line:
            print(f"      VO: {s.vo_line}")
    if not sb.fits(brief):
        print(f"\nwarning: length is outside ±20% of the {brief.duration_s}s target")
    say(f"saved {out}")
    return 0


def _render(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from .backends.image_mflux import ImageBackendError, MfluxImage
    from .render import load_storyboard, render_images

    start = time.monotonic()

    def say(msg: str) -> None:
        print(f"[{time.monotonic() - start:6.1f}s] {msg}", file=sys.stderr)

    src = Path(args.storyboard)
    if not src.exists():
        print(f"error: {src} not found. Run `talewerk plan \"your idea\"` first.", file=sys.stderr)
        return 2
    try:
        sb = load_storyboard(src)
        backend = MfluxImage(preset=args.image_model, quantize=args.quantize, steps=args.steps)
        say(f"rendering {len(sb.shots)} shots with {backend.name} at {args.width}x{args.height}")
        manifest = render_images(sb, backend, Path(args.out), width=args.width,
                                 height=args.height, seed=args.seed, style=args.style,
                                 force=args.force, limit=args.limit, progress=say)
    except ValidationError as exc:
        print(f"error: {src} is not a valid storyboard: {exc.errors()[0]['msg']}", file=sys.stderr)
        return 2
    except ImageBackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    made = sum(1 for m in manifest if m["status"] == "generated")
    print(f"\n{made} image(s) generated, {len(manifest) - made} skipped. "
          f"Open: {Path(args.out) / 'images'}")
    return 0


def _video(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from .assemble import AssembleError, assemble_video, probe_duration
    from .backends.tts_say import SayTTS, TTSError
    from .render import load_storyboard

    start = time.monotonic()

    def say(msg: str) -> None:
        print(f"[{time.monotonic() - start:6.1f}s] {msg}", file=sys.stderr)

    src = Path(args.storyboard)
    if not src.exists():
        print(f"error: {src} not found. Run `talewerk plan \"your idea\"` first.", file=sys.stderr)
        return 2
    tts = None if args.voice == "none" else SayTTS(voice=args.voice_name, rate=args.rate)
    try:
        sb = load_storyboard(src)
        say(f"building video from {len(sb.shots)} shots, voice: {tts.name if tts else 'none'}")
        final = assemble_video(sb, Path(args.out), tts=tts, width=args.width, height=args.height,
                               fps=args.fps, subtitles=not args.no_subtitles, progress=say)
        seconds = probe_duration(final)
    except ValidationError as exc:
        print(f"error: {src} is not a valid storyboard: {exc.errors()[0]['msg']}", file=sys.stderr)
        return 2
    except (AssembleError, TTSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(f"\nVideo ready: {final}  ({seconds:.1f}s)\nOpen it with: open {final}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="talewerk",
        description="Turn a simple idea into a finished video.",
    )
    parser.add_argument("--version", action="version", version=f"talewerk {__version__}")
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="check that this machine is ready")
    doctor.add_argument(
        "--llm-url",
        default=DEFAULT_LLM_URL,
        help=f"OpenAI-compatible base URL to probe (default: {DEFAULT_LLM_URL})",
    )
    doctor.set_defaults(func=_doctor)

    plan = sub.add_parser("plan", help="idea -> brief, hooks, script, storyboard (JSON)")
    plan.add_argument("idea", help='the video idea, e.g. "why Bangalore loves filter coffee"')
    plan.add_argument("--duration", type=int, default=30, help="target seconds (default 30)")
    plan.add_argument("--platform", default="instagram",
                      choices=["instagram", "youtube_shorts", "tiktok", "youtube"])
    plan.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                      help="ollama = native API, fast + schema-forced JSON (default); "
                           "openai = any OpenAI-compatible server")
    plan.add_argument("--model", default=None, help="LLM model (default: $TALEWERK_MODEL or qwen3:8b)")
    plan.add_argument("--llm-url", default=None,
                      help="server URL (default: Ollama on localhost)")
    plan.add_argument("-o", "--out", default="storyboard.json", help="output file")
    plan.set_defaults(func=_plan)

    render = sub.add_parser("render", help="storyboard.json -> one image per shot (FLUX via mflux)")
    render.add_argument("storyboard", nargs="?", default="storyboard.json")
    render.add_argument("-o", "--out", default="out", help="output folder (default: out)")
    render.add_argument("--image-model", default="klein", choices=["klein", "schnell"],
                        help="klein = FLUX.2 klein 4B (default, smaller); schnell = FLUX.1 schnell")
    render.add_argument("--quantize", type=int, default=8, choices=[4, 8],
                        help="8 = better quality (default), 4 = less memory")
    render.add_argument("--steps", type=int, default=None, help="denoising steps (default: preset)")
    render.add_argument("--width", type=int, default=720)
    render.add_argument("--height", type=int, default=1280)
    render.add_argument("--seed", type=int, default=7, help="same seed = same images")
    render.add_argument("--style", default=None, help="style words added to every shot")
    render.add_argument("--limit", type=int, default=None, help="only the first N shots (for testing)")
    render.add_argument("--force", action="store_true", help="regenerate images that already exist")
    render.set_defaults(func=_render)

    video = sub.add_parser("video", help="images + voice + text -> out/video.mp4")
    video.add_argument("storyboard", nargs="?", default="storyboard.json")
    video.add_argument("-o", "--out", default="out", help="folder that holds images/ (default: out)")
    video.add_argument("--voice", default="say", choices=["say", "none"],
                       help="say = the Mac's built-in voice (default); none = silent video")
    video.add_argument("--voice-name", default=None,
                       help="a voice from `say -v '?'`, e.g. Rishi (Indian English)")
    video.add_argument("--rate", type=int, default=None, help="speaking speed in words per minute")
    video.add_argument("--no-subtitles", action="store_true", help="do not show the voice lines")
    video.add_argument("--width", type=int, default=720)
    video.add_argument("--height", type=int, default=1280)
    video.add_argument("--fps", type=int, default=30)
    video.set_defaults(func=_video)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    if args.command == "render" and args.style is None:
        from .render import DEFAULT_STYLE

        args.style = DEFAULT_STYLE
    if args.command == "plan":
        from .backends.ollama import DEFAULT_URL as OLLAMA_URL
        from .backends.openai_compat import DEFAULT_MODEL, DEFAULT_URL

        args.model = args.model or DEFAULT_MODEL
        args.llm_url = args.llm_url or (OLLAMA_URL if args.backend == "ollama" else DEFAULT_URL)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
