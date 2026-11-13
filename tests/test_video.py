"""Video assembly tests. The end-to-end one runs real FFmpeg when it is installed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from talewerk.assemble import AssembleError, assemble_video, make_text_layer, probe_duration
from talewerk.backends.tts_say import SayTTS, TTSError
from talewerk.schemas import Script, Storyboard

SB = Storyboard.model_validate({"title": "Chai", "hook": "4am chai", "shots": [
    {"id": 1, "duration_s": 1.0, "visual_prompt": "tumbler", "overlay_text": "4am chai",
     "vo_line": "4am chai."},
    {"id": 2, "duration_s": 1.0, "visual_prompt": "words", "shot_type": "kinetic_text",
     "vo_line": "Ginger, cardamom, patience."},
    {"id": 3, "duration_s": 1.0, "visual_prompt": "mortar", "overlay_text": "Follow"},
]})

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class ToneTTS:
    """Stands in for a real voice: makes a beep of a known length with FFmpeg."""

    name, suffix = "tone", ".wav"

    def __init__(self, seconds):
        self.seconds = seconds

    def synth(self, text, out):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"sine=frequency=440:duration={self.seconds}", str(out)], check=True)
        return out


def images(out, ids=(1, 3)):
    (out / "images").mkdir(parents=True)
    for i in ids:
        Image.new("RGB", (360, 640), (120, 80, 40)).save(out / "images" / f"shot_{i:02d}.png")


def test_say_command_and_missing_voice_hint(tmp_path):
    tts = SayTTS(voice="Rishi", rate=170,
                 runner=lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "Voice not found"))
    cmd = tts.command("-hello", tmp_path / "a.aiff")
    assert cmd[:3] == ["say", "-o", str(tmp_path / "a.aiff")]
    assert cmd[3:7] == ["-v", "Rishi", "-r", "170"] and cmd[-1] == " -hello"
    with pytest.raises(TTSError, match="say -v"):
        tts.synth("hi", tmp_path / "a.aiff")


def test_text_layer_is_transparent_with_words_and_skips_duplicate(tmp_path):
    p = make_text_layer(SB.shots[0], tmp_path / "t.png", 360, 640, subtitles=True)
    img = Image.open(p)
    assert img.mode == "RGBA" and img.size == (360, 640)
    alpha = img.getchannel("A")
    assert alpha.getextrema() == (0, 255)            # see-through, with some solid text
    assert alpha.crop((0, 320, 360, 640)).getextrema()[1] == 0  # no repeated subtitle below


def test_missing_image_gives_clear_error(tmp_path):
    with pytest.raises(AssembleError, match="talewerk render"):
        assemble_video(SB, tmp_path / "out")


def test_empty_voice_lines_are_filled_from_script():
    sb = SB.model_copy(update={"shots": [s.model_copy(update={"vo_line": ""}) for s in SB.shots]})
    script = Script.model_validate({"title": "t", "beats": [
        {"vo_line": "One.", "duration_s": 1}, {"vo_line": "Two.", "duration_s": 1},
        {"vo_line": "Three.", "duration_s": 1}, {"vo_line": "Four.", "duration_s": 1}]})
    filled = sb.with_voice_from(script)
    assert [s.vo_line for s in filled.shots] == ["One. Two.", "Three.", "Four."]
    assert SB.with_voice_from(script) is SB  # already has voice lines: untouched


@needs_ffmpeg
def test_end_to_end_silent_video(tmp_path):
    out = tmp_path / "out"
    images(out)
    final = assemble_video(SB, out, width=360, height=640, fps=24)
    assert abs(probe_duration(final) - 3.0) < 0.3
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "stream=codec_type,width,height", "-of", "json", str(final)],
                             capture_output=True, text=True, check=True)
    kinds = {s["codec_type"]: s for s in json.loads(streams.stdout)["streams"]}
    assert kinds["video"]["width"] == 360 and kinds["video"]["height"] == 640
    assert "audio" in kinds


@needs_ffmpeg
def test_voice_longer_than_shot_stretches_the_shot(tmp_path):
    out = tmp_path / "out"
    images(out)
    final = assemble_video(SB, out, tts=ToneTTS(2.0), width=360, height=640, fps=24)
    info = json.loads((out / "video.json").read_text())
    assert [s["voice"] for s in info["shots"]] == [True, True, False]
    assert info["shots"][0]["seconds"] == pytest.approx(2.4, abs=0.1)  # 2.0 voice + 0.4 pad
    assert abs(probe_duration(final) - 5.8) < 0.4
    assert Path(out / "video.mp4").stat().st_size > 5000
