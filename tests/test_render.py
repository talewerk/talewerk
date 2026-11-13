"""Render tests with a fake mflux runner: no model, no Mac needed."""

import json
import subprocess
from pathlib import Path

import pytest

from talewerk.backends.image_mflux import ImageBackendError, MfluxImage
from talewerk.cli import main
from talewerk.render import load_storyboard, render_images

SB = {"storyboard": {"title": "Chai", "hook": "4am chai", "shots": [
    {"id": 1, "duration_s": 5, "visual_prompt": "steel tumbler of chai.", "overlay_text": "4am"},
    {"id": 2, "duration_s": 5, "visual_prompt": "big words", "shot_type": "kinetic_text"},
    {"id": 3, "duration_s": 5, "visual_prompt": "ginger in a mortar", "overlay_text": "Follow"},
]}}


def fake_runner(calls, code=0):
    def run(cmd):
        calls.append(cmd)
        if code == 0:
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, code, "", "boom: out of memory")
    return run


@pytest.fixture
def sb(tmp_path):
    p = tmp_path / "storyboard.json"
    p.write_text(json.dumps(SB))
    return load_storyboard(p)


def test_command_is_built_as_a_safe_argument_list():
    cmd = MfluxImage("schnell", quantize=4).command("a; rm -rf /", Path("o.png"),
                                                     seed=3, width=720, height=1280)
    assert cmd[0] == "mflux-generate" and cmd[cmd.index("--model") + 1] == "schnell"
    assert cmd[cmd.index("--prompt") + 1] == "a; rm -rf /"  # one argument, never a shell string
    assert cmd[cmd.index("--steps") + 1] == "2" and cmd[cmd.index("--quantize") + 1] == "4"


def test_render_skips_text_shots_adds_style_and_resumes(sb, tmp_path):
    calls = []
    backend = MfluxImage("klein", runner=fake_runner(calls))
    m = render_images(sb, backend, tmp_path / "out", style="film look", seed=10)
    assert [e["shot"] for e in m] == [1, 3]  # shot 2 is kinetic text
    assert calls[0][0] == "mflux-generate-flux2"
    assert calls[0][calls[0].index("--prompt") + 1] == "steel tumbler of chai. film look"
    assert calls[1][calls[1].index("--seed") + 1] == "13"
    assert json.loads((tmp_path / "out" / "render.json").read_text())[0]["status"] == "generated"
    # second run: nothing regenerated
    m2 = render_images(sb, backend, tmp_path / "out", style="film look", seed=10)
    assert len(calls) == 2 and all(e["status"] == "skipped" for e in m2)


def test_failed_generation_gives_clear_error(sb, tmp_path):
    backend = MfluxImage("klein", runner=fake_runner([], code=1))
    with pytest.raises(ImageBackendError, match="out of memory"):
        render_images(sb, backend, tmp_path / "out")


def test_cli_missing_storyboard_and_missing_mflux(tmp_path, capsys, monkeypatch):
    assert main(["render", str(tmp_path / "nope.json")]) == 2
    p = tmp_path / "storyboard.json"
    p.write_text(json.dumps(SB))
    monkeypatch.setattr("shutil.which", lambda _c: None)
    assert main(["render", str(p), "-o", str(tmp_path / "out")]) == 3
    assert "uv tool install --upgrade mflux" in capsys.readouterr().err
