"""Prompt templates. Kept in code and in git: prompts are versioned like code,
so a change in output quality can be traced to the prompt change behind it."""

SYSTEM = (
    "You are the creative director of a short-form video studio. You write for "
    "social media: strong hooks, clear pacing, concrete visuals, no filler. "
    "Use specific, local, real-world details (objects, places, rituals, sounds) "
    "instead of generic ones. Avoid cliches such as 'more than just', 'heartbeat', "
    "'journey', 'game-changer', 'unlock'."
)

BRIEF = """Idea: {idea}
Platform: {platform}
Target length: {duration_s} seconds

Who exactly is this video for, what tone should it have, and what one-line
call to action should end it?"""

CONCEPT = """Brief:
{brief}

Write {n} different opening hooks for this video. Each hook is what the viewer sees
and hears in the first 1-3 seconds and must stop them scrolling. Use different
angles (surprise, question, bold claim, relatable moment, visual reveal...).
Each hook must name something concrete and specific to this idea.
Each hook is at most 12 words."""

JUDGE_SYSTEM = (
    "You are a strict social media editor. You did not write these hooks. "
    "You pick the one most likely to stop a real person scrolling. You penalise "
    "cliches, vague words and hooks that could fit any other video."
)

JUDGE = """Brief:
{brief}

Hooks:
{hooks}

Which hook is best? Reply with its number and one sentence of reasoning."""

SCRIPT = """Brief:
{brief}

Chosen hook: {hook}

Write the script as timed beats. The first beat delivers the hook. The last beat
delivers the call to action. Beat durations must add up to about {duration_s} seconds.
Keep voiceover lines short and spoken-sounding (about 2.5 words per second)."""

STORYBOARD = """Brief:
{brief}

Script:
{script}

Turn the script into a storyboard: one or more shots per beat, numbered from 1.
Shot durations must add up to about {duration_s} seconds; each shot 1.5-6 seconds.

Rules for visual_prompt (it goes to an image model that makes ONE still picture):
- ONE subject in ONE moment. Never a montage, collage or split screen.
  If a beat needs several scenes, make several shots.
- Concrete and specific: real objects, place, light, style, framing.
  Prefer local, authentic details over generic ones.
- NO text in the picture: no signs with words, labels, logos, slogans, hashtags
  or quoted words. Put any words in overlay_text.

Rules for overlay_text (short on-screen words, many viewers watch muted):
- Required on the first shot (the hook) and the last shot (the call to action).

shot_type: "image_motion" by default; "kinetic_text" for text-only moments;
"gen_video" only for at most 2 hero shots where real motion matters."""
