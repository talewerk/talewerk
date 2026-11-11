"""Prompt templates. Kept in code and in git: prompts are versioned like code,
so a change in output quality can be traced to the prompt change behind it."""

SYSTEM = (
    "You are the creative director of a short-form video studio. You write for "
    "social media: strong hooks, clear pacing, concrete visuals, no filler."
)

BRIEF = """Turn this idea into a production brief.
Idea: {idea}
Platform: {platform}
Target length: {duration_s} seconds
Infer a specific audience, a tone, and a one-line call to action."""

CONCEPT = """Brief:
{brief}

Write {n} different opening hooks for this video. Each hook is what the viewer sees
and hears in the first 1-3 seconds and must stop them scrolling. Use different
angles (surprise, question, bold claim, relatable moment, visual reveal...).
Score each hook 1-10 honestly for scroll-stopping power."""

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
For each shot write a concrete visual_prompt for an image model (subject, setting,
lighting, style, composition), a camera move, overlay_text and vo_line.
shot_type rules: use "image_motion" by default; "kinetic_text" for text-only moments;
"gen_video" only for at most 2 hero shots where real motion matters.
Shot durations must add up to about {duration_s} seconds; each shot 1.5-6 seconds."""

STORYBOARD_FIX = """
Your previous storyboard totalled {total_s:.1f}s but the target is {duration_s}s.
Adjust shot durations (or add/remove shots) so the total is within 20% of the target."""
