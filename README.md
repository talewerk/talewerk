# Talewerk

An AI agent that turns a simple idea into a finished video: script, storyboard, visuals, voice and edit.

> Status: early development (v0.0.x). The CLI currently ships `talewerk doctor`; the agent pipeline lands next.

## Install

```bash
pip install talewerk
# or
uv tool install talewerk
```

## Usage

```bash
talewerk --version
talewerk doctor                                   # checks Python, FFmpeg, Apple Silicon, local LLM
talewerk doctor --llm-url http://localhost:8000/v1  # probe vllm-metal instead of Ollama
```

## How it works (planned)

1. **Plan** with an LLM: brief, concept (hooks), script, storyboard (JSON).
2. **Produce** one model at a time: images (FLUX), video (LTX), voice (Kokoro), captions (Whisper).
3. **Critic**: a vision model checks each shot and retries bad ones.
4. **Assemble**: MoviePy + FFmpeg, captions, music.
5. **Publish** with human approval.

Every model sits behind a backend interface, so local and cloud models are swappable by config.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

## License

MIT
