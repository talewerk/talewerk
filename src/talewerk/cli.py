"""Command-line entry point: `talewerk`."""

from __future__ import annotations

import argparse

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
