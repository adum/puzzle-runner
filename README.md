# Puzzle Runner

Iterative benchmark orchestrator for AI coding agents.

Backends: OpenAI Codex CLI, Claude Code, and OpenRouter API.

Requires Python 3.10 or newer.

## Quick Start

```sh
cp config.example.toml runner.toml
PYTHONPATH=src python3 -m puzzle_runner --config runner.toml
```

On Windows, run this inside WSL.

The runner clones `https://github.com/adum/coilbench.git` into a new run workspace, runs `download_full_levels.sh` there, prompts Codex, runs `evaluate_full.py`, logs each round, and stops after the configured number of no-progress rounds. The benchmark's `evaluate_full.py` always evaluates `./run_solver`.

Fresh git workspaces are checked out with `core.autocrlf=false` and LF endings before downloads, builds, or guard baselines. This keeps agents from needing to touch protected evaluator scripts just to make shebangs runnable on WSL/macOS.

Fresh cloned runs generate an ephemeral password for `download_full_levels.sh` and `evaluate_full.py`. Set `COIL_FULL_PASSWORD` only when using an existing encrypted level archive.

Optional install:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
puzzle-runner --config runner.toml
```

## Output

Default output goes under `.puzzle-runs/`.

Per-run logs include:

- prompts
- agent stdout/stderr
- evaluation stdout/stderr
- parsed scores
- git diffs
- final result

Top-level final summaries append to `final_results.md`.

For Claude stream JSON runs, `Agent Chars` counts assistant text content only,
not the raw JSON event envelope.

## Live Status

The runner updates these files throughout the run:

```text
.puzzle-runs/current/status.md
.puzzle-runs/current/status.json
```

Use them to track phase, round, best score, last score, stale count, remaining no-progress tries, and latest log paths.

```sh
watch -n 5 cat .puzzle-runs/current/status.md
```

For a terminal dashboard with colors:

```sh
PYTHONPATH=src python3 -m puzzle_runner watch
```

or after installing:

```sh
puzzle-runner watch
```

## Claude Code

Use the Claude config:

```sh
cp config.claude.example.toml runner.claude.toml
PYTHONPATH=src python3 -m puzzle_runner --config runner.claude.toml
```

Watch that run with:

```sh
PYTHONPATH=src python3 -m puzzle_runner watch --config runner.claude.toml
```

The Claude config pipes Puzzle Runner's prompt to:

```sh
claude --print --no-session-persistence --verbose --output-format stream-json --include-partial-messages --dangerously-skip-permissions --model claude-sonnet-4-6 --effort xhigh
```

The command goes through `scripts/claude-code`, which sources `nvm` first when available. This keeps unattended WSL runs from accidentally using an older system Node. Claude agent output is raw stream JSON in the logs; use the watcher for a readable live view. Edit the model value or `effort` in `[agent]` when desired.

## OpenRouter

Use the OpenRouter config:

```sh
cp config.openrouter.example.toml runner.openrouter.toml
export OPENROUTER_API_KEY=...
PYTHONPATH=src python3 -m puzzle_runner --config runner.openrouter.toml
```

The OpenRouter backend calls the chat completions API directly and runs a small JSON-action loop around the model so it can read files, run shell commands, write files, and return control. Configure the model with `[agent].model`; the example starts with `poolside/laguna-xs.2:free`.

Each OpenRouter response is logged as `openrouter-response-*.json`. When the response has a generation id, Puzzle Runner also asks OpenRouter for `/generation` metadata and logs `openrouter-generation-*.json`. The watcher and final results use that metadata for API calls, token counts, cost, provider, latency, and metadata fetch failures.

## Benchmark Source

Default config clones the benchmark per run.

For local development, set:

```toml
benchmark_path = "../cb_base"
download_full_levels = false
```

When `benchmark_path` is set, the runner creates an isolated worktree or copy from that local checkout.

If `download_full_levels = false`, the runner cannot generate a useful password for an existing encrypted archive. Set `COIL_FULL_PASSWORD` in that case.

## Agent Feedback

The agent receives score-only full-evaluation feedback:

- last highest passed level
- whether that improved
- best score so far
- stale count

The agent does not receive private level contents, private checker traces, or private failure details.
