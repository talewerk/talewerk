"""Planning tests with a scripted fake LLM: deterministic, no Ollama needed in CI."""

import json

import pytest

from talewerk.agent import run_plan
from talewerk.backends import StructuredOutputError, generate_structured
from talewerk.backends.structured import extract_json
from talewerk.schemas import Brief, HookList, Storyboard


class FakeLLM:
    """Returns queued replies in order and records every prompt it saw."""

    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, *, json_mode=False):
        self.calls.append(user)
        return self.replies.pop(0)


BRIEF = {"idea": "x", "audience": "chai lovers in Bangalore", "tone": "warm",
         "call_to_action": "Follow for more", "platform": "youtube", "duration_s": 99}
HOOKS = {"hooks": [
    {"hook": "Nobody talks about this", "angle": "secret", "score": 6},
    {"hook": "The 4am chai ritual", "angle": "relatable", "score": 9},
    {"hook": "Is chai better than coffee?", "angle": "debate", "score": 7},
]}
SCRIPT = {"title": "Morning Chai", "beats": [
    {"vo_line": "The 4am chai ritual.", "duration_s": 4},
    {"vo_line": "Ginger, cardamom, patience.", "duration_s": 20},
    {"vo_line": "Follow for more.", "overlay_text": "Follow", "duration_s": 6},
]}


def shots(durations):
    return {"title": "Morning Chai", "hook": "The 4am chai ritual", "shots": [
        {"id": i + 1, "duration_s": d, "visual_prompt": f"shot {i + 1}"}
        for i, d in enumerate(durations)
    ]}


def test_extract_json_strips_think_and_fences():
    raw = '<think>hmm</think>\nSure!\n```json\n{"a": 1}\n```'
    assert json.loads(extract_json(raw)) == {"a": 1}


def test_structured_retries_with_error_feedback():
    llm = FakeLLM(["not json at all", json.dumps(HOOKS)])
    hl = generate_structured(llm, "sys", "make hooks", HookList)
    assert hl.best().score == 9
    assert "previous reply was invalid" in llm.calls[1]


def test_structured_gives_up_after_retries():
    llm = FakeLLM(["{}"] * 3)
    with pytest.raises(StructuredOutputError):
        generate_structured(llm, "sys", "u", HookList, retries=2)


def test_storyboard_rejects_bad_ids_and_too_many_gen_video():
    with pytest.raises(ValueError):
        Storyboard.model_validate(
            {**shots([3, 3]), "shots": [{"id": 2, "duration_s": 3, "visual_prompt": "a"},
                                        {"id": 1, "duration_s": 3, "visual_prompt": "b"}]})
    many = shots([2, 2, 2, 2])
    for s in many["shots"]:
        s["shot_type"] = "gen_video"
    with pytest.raises(ValueError):
        Storyboard.model_validate(many)


def test_plan_end_to_end_user_choices_win():
    llm = FakeLLM([json.dumps(BRIEF), json.dumps(HOOKS), json.dumps(SCRIPT),
                   json.dumps(shots([4, 10, 10, 6]))])
    state = run_plan(llm, "morning chai", platform="instagram", duration_s=30)
    assert state["brief"].platform == "instagram"  # model said youtube; user wins
    assert state["brief"].duration_s == 30
    assert state["hook"].hook == "The 4am chai ritual"
    assert state["storyboard"].total_s == 30
    assert state["revisions"] == 0


def test_plan_revises_storyboard_when_too_long():
    llm = FakeLLM([json.dumps(BRIEF), json.dumps(HOOKS), json.dumps(SCRIPT),
                   json.dumps(shots([20, 20, 20])),  # 60s: too long
                   json.dumps(shots([10, 10, 10]))])  # fixed
    state = run_plan(llm, "morning chai", duration_s=30)
    assert state["revisions"] == 1
    assert state["storyboard"].fits(state["brief"])
    assert "totalled 60.0s" in llm.calls[-1]


def test_plan_cli_unreachable_server_is_a_clean_error(capsys, tmp_path):
    from talewerk.cli import main

    code = main(["plan", "idea", "--llm-url", "http://127.0.0.1:9/v1",
                 "-o", str(tmp_path / "sb.json")])
    assert code == 3
    assert "cannot reach the LLM server" in capsys.readouterr().err


def test_brief_schema_bounds():
    with pytest.raises(ValueError):
        Brief.model_validate({**BRIEF, "duration_s": 2})
