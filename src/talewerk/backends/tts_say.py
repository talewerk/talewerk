"""Voice backend: the `say` command built into every Mac.

Why start here: zero install, zero download, works offline. It sits behind the
same small interface a better voice model (Kokoro) will use later.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path


class TTSError(RuntimeError):
    pass


Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


class SayTTS:
    suffix = ".aiff"

    def __init__(self, voice: str | None = None, rate: int | None = None,
                 runner: Runner | None = None) -> None:
        self.voice = voice
        self.rate = rate
        self.name = f"say:{voice or 'default'}"
        self._runner = runner or _run

    def command(self, text: str, out: Path) -> list[str]:
        cmd = ["say", "-o", str(out)]
        if self.voice:
            cmd += ["-v", self.voice]
        if self.rate:
            cmd += ["-r", str(self.rate)]
        # A leading "-" would be read as an option, so guard it with a space.
        cmd.append(" " + text if text.startswith("-") else text)
        return cmd

    def synth(self, text: str, out: Path) -> Path:
        if self._runner is _run and shutil.which("say") is None:
            raise TTSError("the `say` command was not found (it is built into macOS). "
                           "Use --voice none to make a silent video.")
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = self._runner(self.command(text, out))
        if proc.returncode != 0 or not out.exists():
            msg = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
            hint = (f"\n  Is the voice '{self.voice}' installed? List voices with: say -v '?'"
                    if self.voice else "")
            raise TTSError(f"say failed: {msg[0]}{hint}")
        return out
