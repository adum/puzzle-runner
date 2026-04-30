from __future__ import annotations

import dataclasses
import json
import os
import secrets
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import RunnerConfig
from .evaluation import EvaluationParse, parse_evaluation_output
from .guard import ForbiddenGuard
from .process import CommandResult, run_streamed
from .prompts import ScoreFeedback, compose_prompt


class RunnerError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class FinalResult:
    run_id: str
    best_score: int
    best_round: int | None
    total_rounds: int
    stop_reason: str
    log_dir: Path
    workspace: Path


class Runner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.log_dir = config.log_root / config.run_id
        self.workspace = config.worktree_root / config.run_id
        self.events_path = self.log_dir / "events.jsonl"
        self.status_json_path = config.status_dir / "status.json"
        self.status_md_path = config.status_dir / "status.md"
        self._generated_full_eval_password: str | None = None
        self._run_started_monotonic: float | None = None
        self._status: dict = {}

    def run(self) -> FinalResult:
        started = time.monotonic()
        self._run_started_monotonic = started
        self._prepare_paths()
        self._write_json("config.json", _jsonable(dataclasses.asdict(self.config)))
        self._update_status(
            active=True,
            phase="preparing_workspace",
            current_round=0,
            best_score=0,
            best_round=None,
            last_score=None,
            last_improved=None,
            stale_count=0,
            stop_reason=None,
        )
        self._prepare_workspace()
        self._update_status(phase="ensuring_solver_wrapper")
        self._ensure_solver_wrapper()

        if self.config.build_checker:
            self._update_status(phase="building_checker")
            self._build_checker()

        self._update_status(phase="starting_rounds")
        guard = ForbiddenGuard(self.workspace, self.config.forbidden_paths)

        best_score = -1
        best_round: int | None = None
        stale_count = 0
        last_score: int | None = None
        last_improved: bool | None = None
        stop_reason = "max_rounds"
        completed_rounds = 0

        self._event("run_started", workspace=str(self.workspace), log_dir=str(self.log_dir))

        for round_number in range(1, self.config.max_rounds + 1):
            completed_rounds = round_number
            round_dir = self.log_dir / f"round-{round_number:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            self._update_status(
                phase="composing_prompt",
                current_round=round_number,
                latest={
                    "round_dir": round_dir,
                    "prompt": round_dir / "prompt.md",
                    "agent_stdout": round_dir / "agent.stdout.log",
                    "agent_stderr": round_dir / "agent.stderr.log",
                    "evaluation_stdout": round_dir / "evaluation.stdout.log",
                    "evaluation_stderr": round_dir / "evaluation.stderr.log",
                    "evaluation_parse": round_dir / "evaluation_parse.json",
                    "workspace_diff": round_dir / "workspace.diff",
                },
            )

            feedback = ScoreFeedback(
                last_score=last_score,
                best_score=max(best_score, 0),
                improved=last_improved,
                stale_count=stale_count,
                stale_limit=self.config.stale_limit,
                round_number=round_number,
            )
            prompt = compose_prompt(feedback)
            (round_dir / "prompt.md").write_text(prompt, encoding="utf-8")
            self._update_status(phase="agent_running")
            self._event("agent_started", round=round_number)

            agent_result = self._run_agent(round_dir, prompt)
            self._write_round_command(round_dir, "agent_result.json", agent_result)
            self._event(
                "agent_finished",
                round=round_number,
                returncode=agent_result.returncode,
                timed_out=agent_result.timed_out,
                elapsed_seconds=agent_result.elapsed_seconds,
            )
            self._update_status(
                phase="agent_finished",
                last_agent_returncode=agent_result.returncode,
                last_agent_timed_out=agent_result.timed_out,
                last_agent_timeout_reason=agent_result.timeout_reason,
                last_agent_elapsed_seconds=round(agent_result.elapsed_seconds, 2),
            )

            if agent_result.timed_out:
                stop_reason = (
                    "agent_idle_timeout"
                    if agent_result.timeout_reason == "idle"
                    else "agent_timeout"
                )
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break
            if agent_result.returncode != 0:
                stop_reason = "agent_failed"
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break

            self._update_status(phase="pre_evaluation_guard")
            pre_eval_findings = guard.check()
            self._write_json_at(
                round_dir / "pre_evaluation_guard_findings.json",
                [_jsonable(dataclasses.asdict(finding)) for finding in pre_eval_findings],
            )
            if pre_eval_findings:
                stop_reason = "forbidden_edit_detected"
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break

            self._update_status(phase="evaluation_running")
            eval_result = self._run_evaluation(round_dir)
            self._write_round_command(round_dir, "evaluation_result.json", eval_result)
            self._update_status(
                phase="evaluation_parsing",
                last_evaluation_returncode=eval_result.returncode,
                last_evaluation_timed_out=eval_result.timed_out,
            )
            parsed = parse_evaluation_output(eval_result.stdout_path, eval_result.stderr_path)
            self._write_json_at(round_dir / "evaluation_parse.json", _jsonable(dataclasses.asdict(parsed)))

            improved = parsed.highest_passed > best_score
            if improved:
                best_score = parsed.highest_passed
                best_round = round_number
                stale_count = 0
            else:
                stale_count += 1

            last_score = parsed.highest_passed
            last_improved = improved
            self._update_status(
                phase="evaluation_finished",
                best_score=best_score,
                best_round=best_round,
                last_score=last_score,
                last_improved=last_improved,
                stale_count=stale_count,
                first_failing_level=parsed.first_failing_level,
                stop_status=parsed.stop_status,
            )

            diff_path = round_dir / "workspace.diff"
            self._update_status(phase="writing_diff")
            self._write_git_diff(diff_path)

            self._update_status(phase="post_evaluation_guard")
            findings = guard.check()
            self._write_json_at(
                round_dir / "guard_findings.json",
                [_jsonable(dataclasses.asdict(finding)) for finding in findings],
            )

            self._event(
                "evaluation_finished",
                round=round_number,
                score=parsed.highest_passed,
                improved=improved,
                best_score=best_score,
                stale_count=stale_count,
                first_failing_level=parsed.first_failing_level,
                stop_status=parsed.stop_status,
            )

            if findings:
                stop_reason = "forbidden_edit_detected"
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break

            if eval_result.timed_out:
                stop_reason = "evaluation_timeout"
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break
            if eval_result.returncode != 0:
                stop_reason = "evaluation_failed"
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break

            if stale_count >= self.config.stale_limit:
                stop_reason = "stale_limit"
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break
        else:
            completed_rounds = self.config.max_rounds

        if best_score < 0:
            best_score = 0

        final = FinalResult(
            run_id=self.config.run_id,
            best_score=best_score,
            best_round=best_round,
            total_rounds=completed_rounds,
            stop_reason=stop_reason,
            log_dir=self.log_dir,
            workspace=self.workspace,
        )
        elapsed = time.monotonic() - started
        self._update_status(
            active=False,
            phase="finished",
            best_score=final.best_score,
            best_round=final.best_round,
            current_round=final.total_rounds,
            stop_reason=final.stop_reason,
            final_result=self.log_dir / "final_result.md",
        )
        self._write_final_result(final, elapsed)
        self._append_results_summary(final, elapsed)
        self._event("run_finished", **_jsonable(dataclasses.asdict(final)), elapsed_seconds=elapsed)
        return final

    def _prepare_paths(self) -> None:
        if self.log_dir.exists():
            raise RunnerError(f"log directory already exists: {self.log_dir}")
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.config.worktree_root.mkdir(parents=True, exist_ok=True)
        self.config.status_dir.mkdir(parents=True, exist_ok=True)
        self.config.results_path.parent.mkdir(parents=True, exist_ok=True)

    def _prepare_workspace(self) -> None:
        if self.workspace.exists():
            raise RunnerError(f"workspace already exists: {self.workspace}")

        if self.config.benchmark_path is None:
            self._clone_benchmark()
            if self.config.download_full_levels:
                self._download_full_levels()
            return

        if not self.config.benchmark_path.exists():
            raise RunnerError(f"benchmark path does not exist: {self.config.benchmark_path}")

        if self.config.workspace_mode == "worktree":
            self._run_setup_command(
                ["git", "-C", str(self.config.benchmark_path), "worktree", "add", "--detach", str(self.workspace), "HEAD"],
                "git-worktree",
                cwd=self.config.benchmark_path,
            )
            if self.config.download_full_levels:
                self._download_full_levels()
            return

        shutil.copytree(
            self.config.benchmark_path,
            self.workspace,
            ignore=shutil.ignore_patterns(".git"),
        )
        if self.config.download_full_levels:
            self._download_full_levels()

    def _clone_benchmark(self) -> None:
        self._run_setup_command(
            ["git", "clone", self.config.benchmark_repo_url, str(self.workspace)],
            "git-clone",
            cwd=self.config.worktree_root,
            timeout_seconds=1800,
        )
        if self.config.benchmark_ref is not None:
            self._run_setup_command(
                ["git", "checkout", self.config.benchmark_ref],
                "git-checkout",
                cwd=self.workspace,
            )

    def _download_full_levels(self) -> None:
        script = self.workspace / "download_full_levels.sh"
        if not script.exists():
            raise RunnerError("download_full_levels is true but download_full_levels.sh is missing")

        password = self._get_full_eval_password()

        self._run_setup_command(
            ["bash", "./download_full_levels.sh"],
            "download-full-levels",
            cwd=self.workspace,
            stdin_text=f"{password}\n{password}\n",
            timeout_seconds=1800,
        )

    def _ensure_solver_wrapper(self) -> None:
        wrapper = self.workspace / self.config.solver_wrapper
        if wrapper.exists():
            return
        wrapper.write_text(
            """#!/usr/bin/env sh
set -eu
if [ -x ./solver ]; then
  exec ./solver
fi
if [ -f ./solver.py ]; then
  exec python3 ./solver.py
fi
exec python3 ./coil_solver.py
""",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)

    def _build_checker(self) -> None:
        makefile = self.workspace / "coil_check" / "Makefile"
        if not makefile.exists():
            raise RunnerError("build_checker is true but coil_check/Makefile is missing")
        self._run_setup_command(["make", "-C", "coil_check"], "build-checker")

    def _run_setup_command(
        self,
        argv: list[str],
        name: str,
        *,
        cwd: Path | None = None,
        stdin_text: str | None = None,
        timeout_seconds: int = 600,
    ) -> None:
        setup_dir = self.log_dir / "setup"
        stdout_path = setup_dir / f"{name}.stdout.log"
        stderr_path = setup_dir / f"{name}.stderr.log"
        self._update_status(
            phase=f"setup:{name}",
            current_command=argv,
            latest={
                "setup_stdout": stdout_path,
                "setup_stderr": stderr_path,
            },
        )
        result = run_streamed(
            argv,
            cwd=cwd or self.workspace,
            stdin_text=stdin_text,
            timeout_seconds=timeout_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            echo=True,
        )
        self._write_round_command(setup_dir, f"{name}.json", result)
        self._update_status(
            phase=f"setup:{name}:finished",
            last_setup_returncode=result.returncode,
            last_setup_timed_out=result.timed_out,
        )
        if result.returncode != 0 or result.timed_out:
            raise RunnerError(f"setup command failed: {' '.join(argv)}")

    def _run_agent(self, round_dir: Path, prompt: str) -> CommandResult:
        command = self._render_command(self.config.agent.command, round_dir)
        if self.config.agent.prompt_mode == "arg":
            command = [*command, prompt]
            stdin_text = None
        else:
            stdin_text = prompt
        self._update_status(current_command=command)

        return run_streamed(
            command,
            cwd=self.workspace,
            stdin_text=stdin_text,
            timeout_seconds=self.config.agent_timeout_seconds,
            stdout_path=round_dir / "agent.stdout.log",
            stderr_path=round_dir / "agent.stderr.log",
            echo=self.config.echo_agent_output,
            idle_timeout_seconds=self.config.agent_idle_timeout_seconds,
        )

    def _run_evaluation(self, round_dir: Path) -> CommandResult:
        password = self._get_full_eval_password()

        argv = [
            "python3",
            self.config.evaluation_script,
            "--timeout",
            str(self.config.evaluation_timeout_seconds),
        ]
        self._update_status(current_command=argv)
        self._event("evaluation_started", argv=argv)
        return run_streamed(
            argv,
            cwd=self.workspace,
            stdin_text=f"{password}\n",
            env={
                self.config.full_eval_password_env: password,
                "COIL_FULL_PASSWORD": password,
            },
            timeout_seconds=(
                self.config.evaluation_process_timeout_seconds
                if self.config.evaluation_process_timeout_seconds > 0
                else None
            ),
            stdout_path=round_dir / "evaluation.stdout.log",
            stderr_path=round_dir / "evaluation.stderr.log",
            echo=self.config.echo_evaluation_output,
        )

    def _get_full_eval_password(self) -> str:
        env_password = os.environ.get(self.config.full_eval_password_env)
        if env_password:
            return env_password

        if self._generated_full_eval_password is not None:
            return self._generated_full_eval_password

        if not self.config.generate_full_eval_password:
            raise RunnerError(
                f"missing full eval password env var: {self.config.full_eval_password_env}"
            )

        if not self.config.download_full_levels:
            raise RunnerError(
                "cannot generate a password for an existing encrypted archive; "
                f"set {self.config.full_eval_password_env} or enable download_full_levels"
            )

        self._generated_full_eval_password = secrets.token_urlsafe(48)
        self._event("full_eval_password_generated", source="ephemeral")
        return self._generated_full_eval_password

    def _render_command(self, command: list[str], round_dir: Path) -> list[str]:
        replacements = {
            "workspace": str(self.workspace),
            "log_dir": str(self.log_dir),
            "round_dir": str(round_dir),
            "run_id": self.config.run_id,
        }
        return [part.format(**replacements) for part in command]

    def _write_git_diff(self, path: Path) -> None:
        result = subprocess.run(
            ["git", "-C", str(self.workspace), "diff", "--no-ext-diff", "--"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        path.write_text(result.stdout, encoding="utf-8", errors="replace")

    def _write_final_result(self, final: FinalResult, elapsed_seconds: float) -> None:
        body = f"""# Puzzle Runner Final Result

Run id: {final.run_id}
Agent: {self.config.agent.name}
Backend: {self.config.agent.backend}
Benchmark repo: {self.config.benchmark_repo_url}
Benchmark ref: {self.config.benchmark_ref}
Benchmark local path: {self.config.benchmark_path}
Workspace: {final.workspace}
Solver wrapper: ./{self.config.solver_wrapper}
Evaluation: ./{self.config.evaluation_script}
Evaluation timeout: {self.config.evaluation_timeout_seconds}s
Stale limit: {self.config.stale_limit}
Agent timeout: {self.config.agent_timeout_seconds}s
Agent idle timeout: {self.config.agent_idle_timeout_seconds}s
Total rounds: {final.total_rounds}
Best score: {final.best_score}
Best round: {final.best_round}
Stop reason: {final.stop_reason}
Elapsed seconds: {elapsed_seconds:.2f}
"""
        (self.log_dir / "final_result.md").write_text(body, encoding="utf-8")

    def _append_results_summary(self, final: FinalResult, elapsed_seconds: float) -> None:
        if not self.config.results_path.exists():
            self.config.results_path.write_text(
                "| Run ID | Agent | Best Score | Best Round | Rounds | Stop Reason | Timeout | Logs |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- |\n",
                encoding="utf-8",
            )
        with self.config.results_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"| {final.run_id} | {self.config.agent.name} | {final.best_score} | "
                f"{final.best_round} | {final.total_rounds} | {final.stop_reason} | "
                f"{self.config.evaluation_timeout_seconds}s | {self.log_dir} |\n"
            )

    def _write_json(self, name: str, data) -> None:
        self._write_json_at(self.log_dir / name, data)

    def _write_json_at(self, path: Path, data) -> None:
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_round_command(self, directory: Path, name: str, result: CommandResult) -> None:
        self._write_json_at(directory / name, _jsonable(dataclasses.asdict(result)))

    def _event(self, event: str, **data) -> None:
        payload = {"event": event, "time": time.time(), **data}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")

    def _update_status(self, **updates) -> None:
        now = _utc_now()
        if not self._status:
            self._status = {
                "run_id": self.config.run_id,
                "active": True,
                "phase": "initializing",
                "agent": self.config.agent.name,
                "backend": self.config.agent.backend,
                "benchmark_repo_url": self.config.benchmark_repo_url,
                "benchmark_ref": self.config.benchmark_ref,
                "benchmark_path": self.config.benchmark_path,
                "workspace": self.workspace,
                "log_dir": self.log_dir,
                "events_log": self.events_path,
                "results_path": self.config.results_path,
                "status_json": self.status_json_path,
                "status_md": self.status_md_path,
                "started_at": now,
                "phase_started_at": now,
                "current_round": 0,
                "max_rounds": self.config.max_rounds,
                "best_score": 0,
                "best_round": None,
                "last_score": None,
                "last_improved": None,
                "stale_count": 0,
                "stale_limit": self.config.stale_limit,
                "stop_reason": None,
                "latest": {},
            }

        if "phase" in updates and updates["phase"] != self._status.get("phase"):
            updates.setdefault("phase_started_at", now)
            if updates["phase"] == "agent_running":
                updates.setdefault("agent_started_at", now)
            elif updates["phase"] == "agent_finished":
                updates.setdefault("last_agent_finished_at", now)

        for key, value in updates.items():
            if key == "latest":
                latest = dict(self._status.get("latest") or {})
                latest.update(value)
                self._status["latest"] = latest
            else:
                self._status[key] = value

        stale_count = int(self._status.get("stale_count") or 0)
        remaining = max(self.config.stale_limit - stale_count, 0)
        self._status["remaining_no_progress_tries"] = remaining
        self._status["updated_at"] = now
        if self._run_started_monotonic is not None:
            self._status["elapsed_seconds"] = round(time.monotonic() - self._run_started_monotonic, 2)

        payload = _jsonable(self._status)
        self._write_json_atomic(self.status_json_path, payload)
        self._write_text_atomic(self.status_md_path, _status_markdown(payload))

    def _write_json_atomic(self, path: Path, data) -> None:
        self._write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")

    def _write_text_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _status_markdown(status: dict) -> str:
    latest = status.get("latest") or {}
    lines = [
        "# Puzzle Runner Status",
        "",
        f"- Run ID: `{status.get('run_id')}`",
        f"- Active: `{status.get('active')}`",
        f"- Phase: `{status.get('phase')}`",
        f"- Agent: `{status.get('agent')}`",
        f"- Round: `{status.get('current_round')}/{status.get('max_rounds')}`",
        f"- Best Score: `{status.get('best_score')}`",
        f"- Best Round: `{status.get('best_round')}`",
        f"- Last Score: `{status.get('last_score')}`",
        f"- Last Improved: `{status.get('last_improved')}`",
        f"- No-Progress Count: `{status.get('stale_count')}/{status.get('stale_limit')}`",
        f"- Remaining No-Progress Tries: `{status.get('remaining_no_progress_tries')}`",
        f"- Stop Reason: `{status.get('stop_reason')}`",
        f"- Elapsed Seconds: `{status.get('elapsed_seconds')}`",
        f"- Updated At: `{status.get('updated_at')}`",
        "",
        "## Paths",
        "",
        f"- Workspace: `{status.get('workspace')}`",
        f"- Log Dir: `{status.get('log_dir')}`",
        f"- Events: `{status.get('events_log')}`",
        f"- Results: `{status.get('results_path')}`",
    ]

    if latest:
        lines.extend(["", "## Latest Files", ""])
        for key, value in sorted(latest.items()):
            lines.append(f"- {key}: `{value}`")

    command = status.get("current_command")
    if command:
        lines.extend(["", "## Current Command", "", "```sh", " ".join(command), "```"])

    lines.append("")
    return "\n".join(lines)
