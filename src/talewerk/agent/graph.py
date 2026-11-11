"""Week 1 planning graph:  brief -> concept -> script -> storyboard (-> revise).

Each node is a small function: read state, call the LLM through the backend
interface, validate into a schema, write back to state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..backends import LLMBackend, generate_structured
from ..schemas import Brief, Hook, HookList, Script, Storyboard
from . import prompts

MAX_REVISIONS = 2


class PlanState(TypedDict, total=False):
    idea: str
    platform: str
    duration_s: int
    brief: Brief
    hooks: list[Hook]
    hook: Hook
    script: Script
    storyboard: Storyboard
    revisions: int


Progress = Callable[[str], None]


def build_plan_graph(llm: LLMBackend, progress: Progress | None = None):
    say = progress or (lambda _msg: None)

    def brief_node(s: PlanState) -> PlanState:
        say("brief: expanding the idea")
        b = generate_structured(
            llm, prompts.SYSTEM,
            prompts.BRIEF.format(idea=s["idea"], platform=s["platform"], duration_s=s["duration_s"]),
            Brief,
        )
        # The user's explicit choices always win over the model's guesses.
        b = b.model_copy(update={"idea": s["idea"], "platform": s["platform"],
                                 "duration_s": s["duration_s"]})
        return {"brief": b}

    def concept_node(s: PlanState) -> PlanState:
        say("concept: writing hooks")
        hl = generate_structured(
            llm, prompts.SYSTEM,
            prompts.CONCEPT.format(brief=s["brief"].model_dump_json(indent=2), n=5),
            HookList,
        )
        best = hl.best()
        say(f"concept: picked hook ({best.score}/10): {best.hook}")
        return {"hooks": hl.hooks, "hook": best}

    def script_node(s: PlanState) -> PlanState:
        say("script: writing timed beats")
        sc = generate_structured(
            llm, prompts.SYSTEM,
            prompts.SCRIPT.format(brief=s["brief"].model_dump_json(indent=2),
                                  hook=s["hook"].hook, duration_s=s["duration_s"]),
            Script,
        )
        return {"script": sc}

    def storyboard_node(s: PlanState) -> PlanState:
        revisions = s.get("revisions", -1) + 1
        say("storyboard: planning shots" + (f" (revision {revisions})" if revisions else ""))
        user = prompts.STORYBOARD.format(
            brief=s["brief"].model_dump_json(indent=2),
            script=s["script"].model_dump_json(indent=2),
            duration_s=s["duration_s"],
        )
        if revisions and "storyboard" in s:
            user += prompts.STORYBOARD_FIX.format(total_s=s["storyboard"].total_s,
                                                  duration_s=s["duration_s"])
        sb = generate_structured(llm, prompts.SYSTEM, user, Storyboard)
        return {"storyboard": sb, "revisions": revisions}

    def needs_revision(s: PlanState) -> str:
        if s["storyboard"].fits(s["brief"]) or s["revisions"] >= MAX_REVISIONS:
            return "done"
        return "revise"

    g = StateGraph(PlanState)
    g.add_node("brief", brief_node)
    g.add_node("concept", concept_node)
    g.add_node("script", script_node)
    g.add_node("storyboard", storyboard_node)
    g.add_edge(START, "brief")
    g.add_edge("brief", "concept")
    g.add_edge("concept", "script")
    g.add_edge("script", "storyboard")
    g.add_conditional_edges("storyboard", needs_revision, {"revise": "storyboard", "done": END})
    return g.compile()


def run_plan(
    llm: LLMBackend,
    idea: str,
    *,
    platform: str = "instagram",
    duration_s: int = 30,
    progress: Progress | None = None,
) -> PlanState:
    graph = build_plan_graph(llm, progress)
    return graph.invoke({"idea": idea, "platform": platform, "duration_s": duration_s})
