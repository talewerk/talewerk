"""Planning graph:  brief -> concept -> judge -> script -> storyboard -> fit.

Each node is a small function: read state, call the LLM through the backend
interface, validate into a schema, write back to state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..backends import LLMBackend, generate_structured
from ..schemas import Brief, BriefDraft, Hook, HookList, HookVerdict, Script, Storyboard
from . import prompts


class PlanState(TypedDict, total=False):
    idea: str
    platform: str
    duration_s: int
    brief: Brief
    hooks: list[Hook]
    hook: Hook
    hook_reason: str
    script: Script
    storyboard: Storyboard
    retimed: bool


Progress = Callable[[str], None]


def build_plan_graph(llm: LLMBackend, progress: Progress | None = None):
    say = progress or (lambda _msg: None)

    def brief_node(s: PlanState) -> PlanState:
        say("brief: expanding the idea")
        # The LLM only fills what it must invent; the user's choices are set in code.
        draft = generate_structured(
            llm, prompts.SYSTEM,
            prompts.BRIEF.format(idea=s["idea"], platform=s["platform"], duration_s=s["duration_s"]),
            BriefDraft,
        )
        b = Brief(idea=s["idea"], platform=s["platform"], duration_s=s["duration_s"],
                  **draft.model_dump())
        return {"brief": b}

    def concept_node(s: PlanState) -> PlanState:
        say("concept: writing hooks")
        hl = generate_structured(
            llm, prompts.SYSTEM,
            prompts.CONCEPT.format(brief=s["brief"].model_dump_json(indent=2), n=5),
            HookList,
        )
        return {"hooks": hl.hooks}

    def judge_node(s: PlanState) -> PlanState:
        say("judge: picking the best hook")
        hooks = s["hooks"]
        listing = "\n".join(f"{i}. {h.hook}  (angle: {h.angle})" for i, h in enumerate(hooks, 1))
        verdict = generate_structured(
            llm, prompts.JUDGE_SYSTEM,
            prompts.JUDGE.format(brief=s["brief"].model_dump_json(indent=2), hooks=listing),
            HookVerdict, temperature=0.2,
        )
        idx = verdict.best if 1 <= verdict.best <= len(hooks) else 1
        best = hooks[idx - 1]
        say(f"judge: picked #{idx}: {best.hook}  ({verdict.reason})")
        return {"hook": best, "hook_reason": verdict.reason}

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
        say("storyboard: planning shots")
        user = prompts.STORYBOARD.format(
            brief=s["brief"].model_dump_json(indent=2),
            script=s["script"].model_dump_json(indent=2),
            duration_s=s["duration_s"],
        )
        sb = generate_structured(llm, prompts.SYSTEM, user, Storyboard)
        # The hook was already chosen by the judge; do not let this step reword it.
        sb = sb.model_copy(update={"hook": s["hook"].hook})
        return {"storyboard": sb.with_voice_from(s["script"])}

    def fit_node(s: PlanState) -> PlanState:
        """Fix the total length in code: no extra LLM call needed."""
        sb = s["storyboard"]
        if sb.fits(s["brief"], tolerance=0.05):
            return {"retimed": False}
        say(f"fit: rescaling {sb.total_s:.1f}s -> {s['duration_s']}s")
        return {"storyboard": sb.scaled_to(s["duration_s"]), "retimed": True}

    g = StateGraph(PlanState)
    g.add_node("brief", brief_node)
    g.add_node("concept", concept_node)
    g.add_node("judge", judge_node)
    g.add_node("script", script_node)
    g.add_node("storyboard", storyboard_node)
    g.add_node("fit", fit_node)
    g.add_edge(START, "brief")
    g.add_edge("brief", "concept")
    g.add_edge("concept", "judge")
    g.add_edge("judge", "script")
    g.add_edge("script", "storyboard")
    g.add_edge("storyboard", "fit")
    g.add_edge("fit", END)
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
