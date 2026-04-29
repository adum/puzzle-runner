# Puzzle Runner

Iterative benchmark orchestrator for AI coding agents.

First backend: OpenAI Codex CLI using `gpt-5.3-codex-spark`.

Requires Python 3.10 or newer.

## Quick Start

```sh
cp config.example.toml runner.toml
PYTHONPATH=src python3 -m puzzle_runner --config runner.toml
```

On Windows, run this inside WSL.

The runner clones `https://github.com/adum/coilbench.git` into a new run workspace, runs `download_full_levels.sh` there, prompts Codex, runs `evaluate_full.py`, logs each round, and stops after the configured number of no-progress rounds. The benchmark's `evaluate_full.py` always evaluates `./run_solver`.

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
