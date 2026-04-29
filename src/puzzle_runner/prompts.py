from __future__ import annotations

from dataclasses import dataclass

SENTINEL = "PUZZLE_RUNNER_DONE"


@dataclass(frozen=True)
class ScoreFeedback:
    last_score: int | None
    best_score: int
    improved: bool | None
    stale_count: int
    stale_limit: int
    round_number: int


def compose_prompt(feedback: ScoreFeedback) -> str:
    remaining_tries = max(feedback.stale_limit - feedback.stale_count, 0)
    if feedback.last_score is None:
        score_text = (
            "No full-evaluation score has been recorded yet.\n"
            f"Remaining no-progress tries before stop: {remaining_tries}."
        )
        turn_text = "This is the first agent call for this run."
    else:
        status = "improved" if feedback.improved else "did not improve"
        score_text = (
            f"Last full-evaluation highest passed level: {feedback.last_score}.\n"
            f"That {status} the best score.\n"
            f"Best full-evaluation score so far: {feedback.best_score}.\n"
            f"No-progress count: {feedback.stale_count}/{feedback.stale_limit}.\n"
            f"Remaining no-progress tries before stop: {remaining_tries}."
        )
        turn_text = "Continue from the current workspace state."

    return f"""You are running inside Puzzle Runner, an automated benchmark harness.

{turn_text}

Read `prompts.txt` and execute it to the best of your ability.

Full-evaluation score feedback:
{score_text}

Rules:
- Do not ask the user what to do next.
- Keep working until you have made the best progress you can in this call, then return control.
- Do not modify benchmark assets, level files, evaluator scripts, or checker source.
- Do not inspect, decrypt, print, or expose private level contents.
- Do not run `evaluate_full.py`; Puzzle Runner owns full evaluation.
- You may run public development evaluation with `./evaluate.py --start 1`.
- The fixed solver entrypoint is `./run_solver`.
- You may edit `./run_solver` or files it points to, but Puzzle Runner will always evaluate that same entrypoint.
- When you return control, print `{SENTINEL}` on its own line.

Round: {feedback.round_number}
"""
