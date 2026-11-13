"""Image backend: FLUX on Apple Silicon through the `mflux` command-line tools.

Why call the CLI instead of importing mflux: it keeps Talewerk light (no MLX in
our own dependencies, CI still runs on Linux) and each image runs in its own
process, so the model's memory is fully released when it finishes.

Install once:  uv tool install --upgrade mflux
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class ImageBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class Preset:
    command: str
    model: str
    steps: int
    note: str


PRESETS = {
    # FLUX.2 klein 4B: Apache 2.0, small and fast. Good default for a 24 GB Mac.
    "klein": Preset("mflux-generate-flux2", "flux2-klein-4b", 4, "FLUX.2 klein 4B, Apache 2.0"),
    # FLUX.1 schnell: Apache 2.0, bigger download (~34 GB), 2 steps.
    "schnell": Preset("mflux-generate", "schnell", 2, "FLUX.1 schnell, Apache 2.0"),
}

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # Output is not captured: mflux prints its own progress bar (and the one-time
    # model download), so the user can see that it is working.
    return subprocess.run(cmd, check=False)


class MfluxImage:
    def __init__(self, preset: str = "klein", quantize: int = 8,
                 steps: int | None = None, runner: Runner | None = None) -> None:
        if preset not in PRESETS:
            raise ImageBackendError(f"unknown image model '{preset}'; choose {sorted(PRESETS)}")
        self.preset = PRESETS[preset]
        self.quantize = quantize
        self.steps = steps or self.preset.steps
        self.name = f"mflux:{self.preset.model}:q{quantize}"
        self._runner = runner or _run

    def available(self) -> bool:
        return shutil.which(self.preset.command) is not None

    def command(self, prompt: str, out: Path, *, seed: int, width: int, height: int) -> list[str]:
        return [
            self.preset.command,
            "--model", self.preset.model,
            "--prompt", prompt,
            "--steps", str(self.steps),
            "--seed", str(seed),
            "--width", str(width),
            "--height", str(height),
            "--quantize", str(self.quantize),
            "--output", str(out),
        ]

    def generate(self, prompt: str, out: Path, *, seed: int, width: int, height: int) -> Path:
        if self._runner is _run and not self.available():
            raise ImageBackendError(
                f"'{self.preset.command}' not found. Install it with:\n"
                "  uv tool install --upgrade mflux"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = self._runner(self.command(prompt, out, seed=seed, width=width, height=height))
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-6:])
            raise ImageBackendError(
                f"{self.preset.command} failed (exit {proc.returncode})"
                + (f":\n{tail}" if tail else "; see its messages above")
            )
        if not out.exists():
            raise ImageBackendError(f"{self.preset.command} finished but {out} was not created")
        return out
