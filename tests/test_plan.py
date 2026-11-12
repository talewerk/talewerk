"""Planning tests with a scripted fake LLM: deterministic, no Ollama needed in CI."""

import json

import pytest

from talewerk.agent import run_plan
from talewerk.backends import StructuredOutputError, generate_structured
from talewerk.backends.structured import extract_json
from talewerk.schemas import Brief, HookList, Storyboard


class FakeLLM:
    """Returns queued replies in order and records every call it saw."""

    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, *, json_mode=False, temperature=None, schema=None):
        self.calls.append({"system": system, "user": user, "temperature": temperature,
                           "schema": schema})
        return self.replies.pop(0)


DRAFT = {"audience": "chai lovers in Bangalore", "tone": "warm", "call_to_action": "Follow"}
HOOKS = {"hooks": [
    {"hook": "Nobody talks about this", "angle": "secret"},
    {"hook": "The 4am chai ritual", "angle": "relatable"},
    {"hook": "Is chai better than coffee?", "angle": "debate"},
]}
VERDICT = {"best": 2, "reason": "specific and relatable"}
SCRIPT = {"title": "Morning Chai", "beats": [
    {"vo_line": "The 4am chai ritual.", "duration_s": 4},
    {"vo_line": "Ginger, cardamom, patience.", "duration_s": 20},
    {"vo_line": "Follow for more.", "overlay_text": "Follow", "duration_s": 6},
]}


def shots(durations, prompts=None):
    n = len(durations)
    return {"title": "Morning Chai", "hook": "The 4am chai ritual", "shots": [
        {"id": i + 1, "duration_s": d,
         "visual_prompt": (prompts or {}).get(i, f"steel tumbler of chai, shot {i + 1}"),
         "overlay_text": "text" if i in (0, n - 1) else ""}
        for i, d in enumerate(durations)
    ]}


def replies(*storyboards):
    return [json.dumps(DRAFT), json.dumps(HOOKS), json.dumps(VERDICT), json.dumps(SCRIPT),
            *[json.dumps(sb) for sb in storyboards]]


def test_extract_json_strips_think_and_fences():
    raw = '<think>hmm</think>\nSure!\n```json\n{"a": 1}\n```'
    assert json.loads(extract_json(raw)) == {"a": 1}


def test_structured_retries_with_error_feedback():
    llm = FakeLLM(["not json at all", json.dumps(HOOKS)])
    hl = generate_structured(llm, "sys", "make hooks", HookList)
    assert len(hl.hooks) == 3
    assert "previous reply was invalid" in llm.calls[1]["user"]


def test_structured_gives_up_after_retries():
    llm = FakeLLM(["{}"] * 3)
    with pytest.raises(StructuredOutputError):
        generate_structured(llm, "sys", "u", HookList, retries=2)


def test_storyboard_rejects_bad_ids_and_too_many_gen_video():
    bad_ids = shots([3, 3])
    bad_ids["shots"][0]["id"], bad_ids["shots"][1]["id"] = 2, 1
    with pytest.raises(ValueError):
        Storyboard.model_validate(bad_ids)
    many = shots([2, 2, 2, 2])
    for s in many["shots"]:
        s["shot_type"] = "gen_video"
    with pytest.raises(ValueError):
        Storyboard.model_validate(many)


@pytest.mark.parametrize("prompt", [
    "montage of people drinking coffee",
    "a cup with #BangaloreCoffee written on it",
    "split-screen of morning and night",
    "mug with caption below",
    "steel pot labeled 'Kaveri Coffee' in faded paint",
    "paper bag with a coffee shop logo",
    "shop with a sign that reads Coffee Lab",
    "a mug with a \u2018Coffee for Thought\u2019 slogan",
])
def test_storyboard_rejects_montage_and_text_in_picture(prompt):
    with pytest.raises(ValueError, match="ONE subject"):
        Storyboard.model_validate(shots([5, 5, 5], prompts={1: prompt}))


def test_storyboard_requires_overlay_on_first_and_last():
    sb = shots([5, 5, 5])
    sb["shots"][-1]["overlay_text"] = ""
    with pytest.raises(ValueError, match="overlay_text"):
        Storyboard.model_validate(sb)


def test_plan_end_to_end_judge_picks_and_user_choices_win():
    llm = FakeLLM(replies(shots([4, 10, 10, 6])))
    state = run_plan(llm, "morning chai", platform="instagram", duration_s=30)
    assert state["brief"].platform == "instagram"
    assert state["brief"].duration_s == 30
    assert state["brief"].audience == "chai lovers in Bangalore"
    assert state["hook"].hook == "The 4am chai ritual"  # judge chose #2
    assert state["hook_reason"] == "specific and relatable"
    judge_call = llm.calls[2]
    assert "strict social media editor" in judge_call["system"]
    assert judge_call["temperature"] == 0.2
    assert state["storyboard"].total_s == 30
    assert state["retimed"] is False


def test_judge_out_of_range_falls_back_to_first_hook():
    r = replies(shots([10, 10, 10]))
    r[2] = json.dumps({"best": 9, "reason": "?"})
    state = run_plan(FakeLLM(r), "chai", duration_s=30)
    assert state["hook"].hook == "Nobody talks about this"


def test_too_long_storyboard_is_rescaled_in_code_without_llm_call():
    llm = FakeLLM(replies(shots([20, 20, 20])))  # 60s for a 30s target
    state = run_plan(llm, "morning chai", duration_s=30)
    assert state["retimed"] is True
    assert state["storyboard"].total_s == 30
    assert len(llm.calls) == 5  # brief, hooks, judge, script, storyboard: no rewrite


class SchemaLLM(FakeLLM):
    supports_schema = True


def test_native_schema_backend_gets_schema_and_short_prompt():
    llm = SchemaLLM([json.dumps(HOOKS)])
    generate_structured(llm, "sys", "make hooks", HookList)
    call = llm.calls[0]
    assert call["schema"]["title"] == "HookList"
    assert "JSON Schema" not in call["system"]  # field guide instead of full schema
    assert "- hooks" in call["system"]


def test_plan_cli_unreachable_server_is_a_clean_error(capsys, tmp_path):
    from talewerk.cli import main

    for backend in ("ollama", "openai"):
        code = main(["plan", "idea", "--backend", backend, "--llm-url", "http://127.0.0.1:9/v1",
                     "-o", str(tmp_path / "sb.json")])
        assert code == 3
        assert "cannot reach the LLM server" in capsys.readouterr().err


def test_brief_schema_bounds():
    with pytest.raises(ValueError):
        Brief.model_validate({**DRAFT, "idea": "x", "duration_s": 2})


def test_inline_refs_removes_all_refs():
    from talewerk.backends.structured import inline_refs

    flat = json.dumps(inline_refs(Storyboard.model_json_schema()))
    assert "$ref" not in flat and "$defs" not in flat
    assert "visual_prompt" in flat


def test_apostrophes_are_not_treated_as_quoted_text():
    ok = "the grinder's handle turning in a woman's hands, Bengaluru's skyline behind"
    Storyboard.model_validate(shots([5, 5, 5], prompts={1: ok}))


def test_hook_longer_than_12_words_is_rejected():
    from talewerk.schemas import Hook

    with pytest.raises(ValueError, match="at most 12"):
        Hook(hook=" ".join(["word"] * 13), angle="a")
