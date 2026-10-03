# Hepilot

Hepilot is a small local coding agent that runs from the terminal. It reads the current workspace, keeps session state on disk, and uses a limited set of tools to inspect files, edit files, and run commands.

The project is meant for lightweight repository work: code review, test failure inspection, small implementation tasks, and repeated follow-up work inside the same project.

## Features

- Command line entry point: `hepilot`
- Module entry point: `python -m hepilot`
- Interactive REPL and one-shot prompt modes
- Session resume support with `--resume`
- Local run artifacts under `.hepilot/`
- Provider support for OpenAI-compatible, Anthropic-compatible, and Ollama models
- Secret environment variable redaction for traces and reports
- Approval modes for risky actions: `ask`, `auto`, and `never`

## Installation

Hepilot requires Python 3.10 or newer.

Install in editable mode:

```bash
pip install -e .
```

If you use `uv`, install dependencies with:

```bash
uv sync
```

## Usage

Start the interactive agent in the current directory:

```bash
hepilot
```

Run it through Python:

```bash
python -m hepilot
```

Run a one-shot task:

```bash
hepilot "inspect the tests and summarize the failures"
```

Use another workspace:

```bash
hepilot --cwd /path/to/repo
```

Resume the latest session:

```bash
hepilot --resume latest
```

## Model Providers

OpenAI-compatible provider:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://your-api.example/v1"
export OPENAI_MODEL="gpt-5.4"
hepilot --provider openai
```

Anthropic-compatible provider:

```bash
export ANTHROPIC_API_KEY="your-api-key"
export ANTHROPIC_API_BASE="https://your-api.example/v1"
export ANTHROPIC_MODEL="claude-sonnet-4-6"
hepilot --provider anthropic
```

Ollama provider:

```bash
ollama serve
ollama pull qwen3.5:4b
hepilot --provider ollama --model qwen3.5:4b
```

You can also pass `--model` or `--base-url` directly on the command line.

## REPL Commands

- `/help` shows built-in commands
- `/memory` prints the current distilled memory
- `/session` prints the current session file path
- `/reset` clears the current session
- `/exit` or `/quit` exits the REPL

## Local State

Hepilot writes local state inside the active repository:

- `.hepilot/sessions/` stores resumable session history
- `.hepilot/runs/<run_id>/` stores task artifacts such as `task_state.json`, `trace.jsonl`, and `report.json`

These files are local runtime data and normally should not be committed.

## Development

Run tests:

```bash
uv run pytest
```

Run lint checks:

```bash
uv run ruff check .
```
