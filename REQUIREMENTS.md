# Puzzle Runner Requirements

## Purpose

Automate iterative benchmark runs for AI coding agents.

The runner supervises the agent, runs evaluation, detects progress stalls, logs all activity, and writes a final result.

## Scope

Initial target:

- Benchmark repo: Coil puzzle benchmark.
- Benchmark source: cloned per run from the configured repository.
- Agent backend: OpenAI Codex CLI.
- Evaluation command: `evaluate_full.py`.
- Solver entrypoint: one fixed wrapper file.
- Platforms: macOS and Windows via WSL.

Future target:

- Multiple agent backends.
- Multiple benchmark repos.
- Parallel or tournament-style runs.

## Core Loop

1. Prepare a clean run workspace.
2. Clone or copy the benchmark into that workspace.
3. Run benchmark setup, including full-level download when configured.
4. Start or resume the agent.
5. Send the current prompt.
6. Wait until the agent returns control.
7. Run full evaluation.
8. Record score, failure, logs, and workspace diff.
9. If score improved, reset stale counter.
10. If score did not improve, increment stale counter.
11. Stop when stale counter reaches configured limit.
12. Write final result.

Default stale limit: `3`.

## Evaluation

The runner always calls the same solver wrapper.

Example:

```sh
./evaluate_full.py ./run_solver --timeout 600
```

The wrapper is the stable contract between runner and benchmark.

Agent-generated code may change solver internals, but the evaluator command does not change.

The runner owns evaluation. The agent does not choose evaluation arguments.

## Private-Level Rule

Full evaluation may run after each agent return.

The full-evaluation score may be given back to the agent after each round.

Allowed feedback:

- Highest level passed.
- Whether the score improved.
- Best score so far.
- Stale count.

Recommended default:

- Agent sees score-only full-evaluation feedback.
- Agent does not see private level contents.
- Agent does not see detailed private failure traces.
- Agent does not see checker debug output from private failures.

Reason: private levels should not become iterative feedback.

## Prompts

Prompts are configurable text templates.

Required templates:

- Initial prompt.
- Continue prompt after improvement.
- Continue prompt after no improvement.
- Final stop prompt, if needed.

Prompts should include explicit control language.

Example phrases:

- "Read the benchmark instructions and continue improving the solver."
- "Do not ask what to do next."
- "Keep working until you return control."
- "Do not modify benchmark assets or evaluation scripts."
- "The solver entrypoint is fixed: `./run_solver`."

## Agent Return

The runner needs a reliable return condition.

Supported return modes:

- CLI process exits.
- CLI emits configured sentinel text.
- CLI session reaches idle timeout.

Preferred sentinel:

```text
PUZZLE_RUNNER_DONE
```

The agent prompt should tell the agent to print this only when yielding control.

## Logging

Log everything needed to reproduce a run.

Per-run logs:

- Runner config.
- Agent command.
- Prompt text.
- Agent stdout.
- Agent stderr.
- Evaluation command.
- Evaluation stdout.
- Evaluation stderr.
- Parsed score.
- First failing level.
- Failure reason.
- Wall time.
- Git diff.
- Stop reason.

Final result file:

- Model/backend.
- Benchmark repo.
- Solver wrapper.
- Best full score.
- Best round.
- Timeout.
- Stale limit.
- Total rounds.
- Total wall time.
- Final commit/diff reference.

Logs should be append-only.

Final result should be written separately from raw logs.

## Workspace Isolation

Each run uses a separate workspace.

Preferred mechanism:

- Git worktree on macOS/Linux/WSL.

Alternative:

- Directory copy.

The runner must reject or flag forbidden edits.

Forbidden by default:

- Public levels.
- Private level archives.
- Evaluation scripts.
- Checker source.
- Checker binary, unless rebuilt from trusted source.
- Existing benchmark result logs, unless explicitly allowed.

## Configuration

Config file controls:

- Benchmark repo URL.
- Benchmark ref.
- Optional local benchmark path override.
- Whether to run `download_full_levels.sh`.
- Whether to generate an ephemeral full-evaluation password.
- Worktree root.
- Agent backend.
- Agent command.
- Solver wrapper path.
- Evaluation command.
- Evaluation timeout.
- Stale limit.
- Max rounds.
- Prompt templates.
- Log directory.
- Final result path.
- Forbidden paths.
- Whether full output may be shown to the agent.

Config should be portable across macOS and WSL.

Avoid hard-coded absolute paths.

## Platform Requirements

Must run on:

- macOS.
- Windows with WSL.

Path handling requirements:

- Use POSIX paths inside WSL.
- Avoid Windows-only shell syntax.
- Avoid macOS-only shell syntax.
- Use Python standard library path APIs.
- Treat subprocess commands as argv arrays, not shell strings.

## Initial Backend: Codex

The first backend calls OpenAI Codex CLI.

Backend contract:

- Start session.
- Send prompt.
- Stream output to log.
- Detect return.
- Stop session if timeout is reached.

Backend implementation should be replaceable.

Do not bake Codex assumptions into runner core.

## Score Parsing

The runner parses evaluation output or benchmark log rows.

Required fields:

- Highest passed level.
- Evaluation mode.
- Timeout.
- First failing level, if any.
- Failure reason, if any.

If parsing fails, the round is invalid and logged.

## Stop Conditions

Stop when any condition is met:

- No improvement for `stale_limit` rounds.
- `max_rounds` reached.
- Agent command repeatedly fails.
- Evaluation command repeatedly fails.
- Forbidden edit detected.
- Manual stop requested.

## Results

Write raw run data continuously.

Write final result once.

The final result should use the best score observed, not necessarily the last round.

## Non-Goals

Initial version does not need:

- Parallel agents.
- Web UI.
- API-based model calls.
- Cross-benchmark abstraction.
- Automatic private archive setup.
- Automatic cost accounting.
