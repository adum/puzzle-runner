from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import RunnerConfig
from .evaluation import EvaluationParse, parse_evaluation_output
from .guard import ForbiddenGuard, GuardFinding
from .openrouter_agent import (
    AGENT_CONFIG_ERROR_RETURN_CODE,
    AGENT_MAX_STEPS_RETURN_CODE,
    run_openrouter_agent,
)
from .openrouter_usage import (
    openrouter_usage_to_dict,
    summarize_opencode_openrouter_usage,
    summarize_openrouter_usage,
)
from .process import CommandResult, run_streamed
from .prompts import ScoreFeedback, compose_prompt


AGENT_RETRY_INITIAL_DELAY_SECONDS = 5.0
CODEX_REASONING_EFFORT_RE = re.compile(
    r"(?:^|\s)model_reasoning_effort\s*=\s*[\"']?([^\"'\s]+)"
)
MODEL_NOT_FOUND_ERROR_RE = re.compile(
    r"\bmodel\s+not\s+found\b|(?:Provider)?ModelNotFoundError|\bmodel_not_found\b",
    re.IGNORECASE,
)
MODEL_NOT_FOUND_MESSAGE_RE = re.compile(
    r"\bmodel\s+not\s+found\s*:\s*([^\s\"']+)",
    re.IGNORECASE,
)
PROVIDER_MODEL_ID_RE = re.compile(r'\bmodelID:\s*"([^"]+)"')
PROVIDER_ID_RE = re.compile(r'\bproviderID:\s*"([^"]+)"')
RESULTS_SUMMARY_HEADER = (
    "| Run ID | Agent | Effort | Best Score | Best Round | Rounds | Stop Reason | "
    "Timeout | Wall Time | Agent Chars | Code Lines Added | OpenRouter Calls | "
    "OpenRouter Cost | OpenRouter Tokens |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)
RESULTS_SUMMARY_NO_OPENROUTER_HEADER = (
    "| Run ID | Agent | Effort | Best Score | Best Round | Rounds | Stop Reason | "
    "Timeout | Wall Time | Agent Chars | Code Lines Added |"
)
RESULTS_SUMMARY_NO_OPENROUTER_SEPARATOR = (
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)
RESULTS_SUMMARY_NO_EFFORT_HEADER = (
    "| Run ID | Agent | Best Score | Best Round | Rounds | Stop Reason | "
    "Timeout | Wall Time | Agent Chars | Code Lines Added |"
)
RESULTS_SUMMARY_NO_EFFORT_SEPARATOR = (
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)
RESULTS_SUMMARY_OLD_LOGS_HEADER = (
    "| Run ID | Agent | Best Score | Best Round | Rounds | Stop Reason | Timeout | Logs |"
)
RESULTS_SUMMARY_OLD_LOGS_SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- |"
AGENT_OUTPUT_LOG_PATTERNS = (
    "round-*/agent*.stdout.log",
    "round-*/agent*.stderr.log",
)
NOISY_CODE_EXACT_PATHS = {
    "coil_check/check",
    "levels_secret_even.tar.enc",
}
NOISY_CODE_PREFIXES = (
    ".git/",
    "__pycache__/",
    "levels_public/",
    "levels_secret_even/",
)
NOISY_CODE_SUFFIXES = (
    ".pyc",
    ".pyo",
)
CODE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".cxx",
    ".fish",
    ".go",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".lua",
    ".m",
    ".mm",
    ".php",
    ".pl",
    ".pm",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
    ".zsh",
}
CODE_FILENAMES = {
    "Makefile",
    "makefile",
}
MAX_UNTRACKED_CODE_LINE_COUNT_BYTES = 2_000_000
MAX_AGENT_ERROR_SCAN_BYTES = 2_000_000
DEFAULT_SOLVER_FULL_EVAL_SCORE = 47
DEFAULT_SOLVER_FULL_EVAL_FIRST_FAIL = 48
SCRIPT_LINE_ENDING_SUFFIXES = {
    ".bash",
    ".fish",
    ".pl",
    ".py",
    ".rb",
    ".sh",
    ".zsh",
}
SCRIPT_LINE_ENDING_FILENAMES = {
    "Makefile",
    "makefile",
}
MAX_LINE_ENDING_NORMALIZE_BYTES = 5_000_000


class RunnerError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class FinalResult:
    run_id: str
    best_score: int
    best_round: int | None
    total_rounds: int
    stop_reason: str
    stop_detail: str
    total_wall_seconds: float
    agent_output_chars: int
    code_lines_added: int
    log_dir: Path
    workspace: Path
    openrouter_usage: dict | None = None


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
        self._default_solver_baseline: dict[str, str | None] | None = None
        self._last_agent_output_status_monotonic = 0.0
        self._status_lock = threading.Lock()
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
        self._event("run_started", workspace=str(self.workspace), log_dir=str(self.log_dir))

        model_problem = self._opencode_model_preflight_problem()
        if model_problem is not None:
            stop_reason = "agent_model_not_found"
            self._update_status(
                phase="stopping",
                stop_reason=stop_reason,
                agent_model_preflight_problem=model_problem,
                last_agent_model_not_found=True,
                last_agent_model_not_found_model=model_problem.get("model"),
            )
            self._event("agent_model_not_found", **model_problem)
            return self._finish_run(
                started=started,
                best_score=0,
                best_round=None,
                completed_rounds=0,
                stop_reason=stop_reason,
            )

        self._prepare_workspace()
        self._normalize_workspace_line_endings()
        if self.config.download_full_levels:
            self._download_full_levels()
        self._update_status(phase="ensuring_solver_wrapper")
        self._ensure_solver_wrapper()
        self._default_solver_baseline = _default_solver_signature(
            self.workspace,
            self.config.solver_wrapper,
        )

        if self.config.build_checker:
            self._update_status(phase="building_checker")
            self._build_checker()

        best_score = -1
        best_round: int | None = None
        stale_count = 0
        last_score: int | None = None
        last_improved: bool | None = None
        score_history: list[int] = []
        stop_reason = "max_rounds"
        completed_rounds = 0

        self._update_status(phase="starting_rounds")
        auth_problem = self._agent_auth_preflight_problem()
        if auth_problem is not None:
            stop_reason = "agent_auth_missing"
            round_range = range(1, 1)
            self._update_status(
                phase="stopping",
                stop_reason=stop_reason,
                agent_auth_problem=auth_problem,
            )
            self._event("agent_auth_missing", **auth_problem)
        else:
            round_range = range(1, self.config.max_rounds + 1)

        guard = ForbiddenGuard(self.workspace, self.config.forbidden_paths)

        for round_number in round_range:
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
                    "openrouter_usage_summary": round_dir / "openrouter-usage-summary.json",
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

            agent_result = self._run_agent(round_number, round_dir, prompt)
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
                agent_retry_delay_seconds=None,
            )

            if agent_result.timed_out:
                stop_reason = (
                    "agent_idle_timeout"
                    if agent_result.timeout_reason == "idle"
                    else "agent_timeout"
                )
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break
            if self._status.get("last_agent_model_not_found"):
                stop_reason = "agent_model_not_found"
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break
            if agent_result.returncode != 0:
                stop_reason = (
                    "agent_max_steps"
                    if agent_result.returncode == AGENT_MAX_STEPS_RETURN_CODE
                    else "agent_failed"
                )
                self._update_status(phase="stopping", stop_reason=stop_reason)
                break

            self._update_status(phase="pre_evaluation_guard")
            pre_eval_findings = guard.check()
            pre_eval_findings_payload = _guard_findings_payload(pre_eval_findings)
            self._write_json_at(round_dir / "pre_evaluation_guard_findings.json", pre_eval_findings_payload)
            if pre_eval_findings:
                stop_reason = "forbidden_edit_detected"
                self._update_status(
                    phase="stopping",
                    stop_reason=stop_reason,
                    guard_findings=pre_eval_findings_payload,
                    guard_phase="pre_evaluation",
                )
                break

            if self._can_shortcut_default_solver_evaluation():
                self._update_status(
                    phase="evaluation_shortcut",
                    default_solver_evaluation_shortcut=True,
                )
                eval_result = self._write_default_solver_evaluation_result(round_dir)
            else:
                self._update_status(
                    phase="evaluation_running",
                    default_solver_evaluation_shortcut=False,
                )
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
            score_history.append(last_score)
            self._update_status(
                phase="evaluation_finished",
                best_score=best_score,
                best_round=best_round,
                last_score=last_score,
                score_history=list(score_history),
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
            findings_payload = _guard_findings_payload(findings)
            self._write_json_at(round_dir / "guard_findings.json", findings_payload)

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
                self._update_status(
                    phase="stopping",
                    stop_reason=stop_reason,
                    guard_findings=findings_payload,
                    guard_phase="post_evaluation",
                )
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
            if stop_reason != "max_rounds":
                completed_rounds = 0
            else:
                completed_rounds = self.config.max_rounds

        return self._finish_run(
            started=started,
            best_score=best_score,
            best_round=best_round,
            completed_rounds=completed_rounds,
            stop_reason=stop_reason,
        )

    def _finish_run(
        self,
        *,
        started: float,
        best_score: int,
        best_round: int | None,
        completed_rounds: int,
        stop_reason: str,
    ) -> FinalResult:
        if best_score < 0:
            best_score = 0

        elapsed = time.monotonic() - started
        agent_output_chars = count_agent_output_chars(
            self.log_dir,
            agent_stream_format=_agent_stream_format(self.config),
        )
        code_lines_added = count_code_lines_added(self.workspace)
        openrouter_usage = _openrouter_usage_for_result(self.config, self.log_dir)
        stop_detail = explain_stop_reason(stop_reason, self.config, self._status)
        final = FinalResult(
            run_id=self.config.run_id,
            best_score=best_score,
            best_round=best_round,
            total_rounds=completed_rounds,
            stop_reason=stop_reason,
            stop_detail=stop_detail,
            total_wall_seconds=elapsed,
            agent_output_chars=agent_output_chars,
            code_lines_added=code_lines_added,
            log_dir=self.log_dir,
            workspace=self.workspace,
            openrouter_usage=openrouter_usage,
        )
        self._update_status(
            active=False,
            phase="finished",
            best_score=final.best_score,
            best_round=final.best_round,
            current_round=final.total_rounds,
            stop_reason=final.stop_reason,
            stop_detail=final.stop_detail,
            agent_output_chars=final.agent_output_chars,
            code_lines_added=final.code_lines_added,
            openrouter_usage=final.openrouter_usage,
            final_result=self.log_dir / "final_result.md",
        )
        self._write_final_result(final)
        self._append_results_summary(final)
        self._event("run_finished", **_jsonable(dataclasses.asdict(final)))
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
            return

        if not self.config.benchmark_path.exists():
            raise RunnerError(f"benchmark path does not exist: {self.config.benchmark_path}")

        if self.config.workspace_mode == "worktree":
            self._run_setup_command(
                ["git", "-C", str(self.config.benchmark_path), "worktree", "add", "--detach", str(self.workspace), "HEAD"],
                "git-worktree",
                cwd=self.config.benchmark_path,
            )
            return

        shutil.copytree(
            self.config.benchmark_path,
            self.workspace,
            ignore=shutil.ignore_patterns(".git"),
        )

    def _clone_benchmark(self) -> None:
        self._run_setup_command(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.eol=lf",
                "clone",
                self.config.benchmark_repo_url,
                str(self.workspace),
            ],
            "git-clone",
            cwd=self.config.worktree_root,
            timeout_seconds=1800,
        )
        if self.config.benchmark_ref is not None:
            self._run_setup_command(
                [
                    "git",
                    "-c",
                    "core.autocrlf=false",
                    "-c",
                    "core.eol=lf",
                    "checkout",
                    self.config.benchmark_ref,
                ],
                "git-checkout",
                cwd=self.workspace,
            )

    def _normalize_workspace_line_endings(self) -> None:
        self._update_status(phase="normalizing_line_endings")
        if _is_git_workspace(self.workspace):
            self._run_setup_command(
                [
                    "git",
                    "-c",
                    "core.autocrlf=false",
                    "-c",
                    "core.eol=lf",
                    "checkout",
                    "-f",
                    "HEAD",
                    "--",
                    ".",
                ],
                "git-checkout-lf",
                cwd=self.workspace,
                timeout_seconds=1800,
            )
            self._event("line_endings_normalized", mode="git_checkout_lf")
            return

        changed_paths = normalize_script_line_endings(self.workspace)
        self._write_json_at(
            self.log_dir / "setup" / "line-ending-normalization.json",
            {"changed_paths": changed_paths},
        )
        self._event("line_endings_normalized", mode="script_scan", changed_paths=changed_paths)

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

    def _opencode_model_preflight_problem(self) -> dict | None:
        if not _is_opencode_backend(self.config):
            return None

        model = self.config.agent.model
        if not model:
            return None

        provider = model.split("/", 1)[0] if "/" in model else None
        setup_dir = self.log_dir / "setup"
        suffix = _safe_log_name(provider or "all")
        stdout_path = setup_dir / f"opencode-models-{suffix}.stdout.log"
        stderr_path = setup_dir / f"opencode-models-{suffix}.stderr.log"
        command = self._opencode_models_command(provider)
        self._update_status(
            phase=f"model_check:{provider or 'all'}",
            current_command=command,
            latest={
                "model_check_stdout": stdout_path,
                "model_check_stderr": stderr_path,
            },
        )
        result = run_streamed(
            command,
            cwd=self.config.config_path.parent,
            stdin_text=None,
            timeout_seconds=self.config.agent.command_timeout_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            echo=False,
        )
        self._write_round_command(setup_dir, f"opencode-models-{suffix}.json", result)

        listing = ""
        for path in (stdout_path, stderr_path):
            try:
                listing += path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

        problem = _opencode_model_problem_from_listing(
            self.config,
            provider=provider,
            listing=listing,
            returncode=result.returncode,
            timed_out=result.timed_out,
        )
        if problem is not None:
            problem["model_check_returncode"] = result.returncode
            problem["model_check_timed_out"] = result.timed_out
            problem["model_check_stdout"] = stdout_path
            problem["model_check_stderr"] = stderr_path
            problem["model_check_command"] = command
        return problem

    def _opencode_models_command(self, provider: str | None) -> list[str]:
        command = self._render_command(self.config.agent.command, self.log_dir)
        executable = command[0] if command else "opencode"
        if provider:
            return [executable, "models", provider]
        return [executable, "models"]

    def _agent_auth_preflight_problem(self) -> dict | None:
        provider = _opencode_required_auth_provider(self.config)
        if provider is None:
            return None

        setup_dir = self.log_dir / "setup"
        stdout_path = setup_dir / f"{provider}-auth-list.stdout.log"
        stderr_path = setup_dir / f"{provider}-auth-list.stderr.log"
        command = self._opencode_auth_list_command()
        self._update_status(
            phase=f"auth_check:{provider}",
            current_command=command,
            latest={
                "auth_stdout": stdout_path,
                "auth_stderr": stderr_path,
            },
        )
        result = run_streamed(
            command,
            cwd=self.workspace,
            stdin_text=None,
            timeout_seconds=30,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            echo=False,
        )
        self._write_round_command(setup_dir, f"{provider}-auth-list.json", result)
        listing = ""
        for path in (stdout_path, stderr_path):
            try:
                listing += path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

        problem = _opencode_auth_problem_from_listing(self.config, provider, listing, os.environ)
        if problem is not None:
            problem["auth_check_returncode"] = result.returncode
            problem["auth_check_stdout"] = stdout_path
            problem["auth_check_stderr"] = stderr_path
        return problem

    def _opencode_auth_list_command(self) -> list[str]:
        command = self._render_command(self.config.agent.command, self.log_dir)
        executable = command[0] if command else "opencode"
        return [executable, "auth", "list"]

    def _run_agent(self, round_number: int, round_dir: Path, prompt: str) -> CommandResult:
        command = self._agent_command(round_dir)
        if self.config.agent.backend != "openrouter" and self.config.agent.prompt_mode == "arg":
            command = [*command, prompt]
            stdin_text = None
        else:
            stdin_text = prompt

        attempt = 1
        retry_deadline: float | None = None
        retry_delay = AGENT_RETRY_INITIAL_DELAY_SECONDS

        while True:
            stdout_path, stderr_path = _agent_attempt_log_paths(round_dir, attempt)
            self._update_status(
                phase="agent_running",
                current_command=command,
                agent_attempt=attempt,
                agent_retry_count=attempt - 1,
                agent_retry_delay_seconds=None,
                latest={
                    "agent_stdout": stdout_path,
                    "agent_stderr": stderr_path,
                },
            )
            self._event(
                "agent_attempt_started",
                round=round_number,
                attempt=attempt,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )

            if self.config.agent.backend == "openrouter":
                result = run_openrouter_agent(
                    self.config,
                    cwd=self.workspace,
                    prompt=prompt,
                    round_dir=round_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    timeout_seconds=self.config.agent_timeout_seconds,
                    echo=self.config.echo_agent_output,
                    status_callback=self._openrouter_status_callback(round_number, attempt),
                )
            else:
                result = run_streamed(
                    command,
                    cwd=self.workspace,
                    stdin_text=stdin_text,
                    timeout_seconds=self.config.agent_timeout_seconds,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    echo=self.config.echo_agent_output,
                    idle_timeout_seconds=self.config.agent_idle_timeout_seconds,
                    stdout_completion_predicate=_agent_stdout_completion_predicate(self.config),
                    stdout_line_callback=self._agent_stdout_line_callback(stdout_path, stderr_path),
                    stderr_line_callback=self._agent_output_status_callback(stdout_path, stderr_path),
                )
            self._write_round_command(round_dir, f"agent_attempt-{attempt:03d}.json", result)
            agent_returned_error = _agent_returned_error(self.config, stdout_path)
            agent_model_not_found_model = _agent_model_not_found_detail(stdout_path, stderr_path)
            agent_model_not_found = agent_model_not_found_model is not None
            if agent_returned_error:
                agent_error_count = int(self._status.get("agent_error_count") or 0) + 1
                self._event(
                    "agent_returned_error",
                    round=round_number,
                    attempt=attempt,
                    error_count=agent_error_count,
                )
            else:
                agent_error_count = int(self._status.get("agent_error_count") or 0)
            self._update_status(
                agent_error_count=agent_error_count,
                last_agent_returned_error=agent_returned_error,
                last_agent_model_not_found=agent_model_not_found,
                last_agent_model_not_found_model=agent_model_not_found_model,
            )
            if agent_model_not_found:
                self._event(
                    "agent_model_not_found",
                    round=round_number,
                    attempt=attempt,
                    model=agent_model_not_found_model,
                )
            self._event(
                "agent_attempt_finished",
                round=round_number,
                attempt=attempt,
                returncode=result.returncode,
                timed_out=result.timed_out,
                timeout_reason=result.timeout_reason,
                elapsed_seconds=result.elapsed_seconds,
            )

            if agent_model_not_found:
                return result

            if not _agent_result_is_retryable(result):
                return result

            retry_limit = self.config.agent_failure_retry_limit_seconds
            if retry_limit <= 0:
                return result

            if retry_deadline is None:
                retry_deadline = time.monotonic() + retry_limit

            remaining = retry_deadline - time.monotonic()
            if remaining <= 0:
                return result

            sleep_seconds = min(retry_delay, remaining)
            self._event(
                "agent_retry_scheduled",
                round=round_number,
                attempt=attempt,
                delay_seconds=sleep_seconds,
                remaining_retry_seconds=max(remaining - sleep_seconds, 0),
            )
            self._update_status(
                phase="agent_retry_wait",
                last_agent_returncode=result.returncode,
                last_agent_timed_out=result.timed_out,
                last_agent_timeout_reason=result.timeout_reason,
                agent_retry_delay_seconds=round(sleep_seconds, 2),
                agent_retry_remaining_seconds=round(max(remaining - sleep_seconds, 0), 2),
            )
            time.sleep(sleep_seconds)
            attempt += 1
            retry_delay *= 2

    def _agent_stdout_line_callback(
        self,
        stdout_path: Path,
        stderr_path: Path,
    ) -> Callable[[str], None] | None:
        progress_callback = _agent_stdout_line_callback(self.config)
        output_callback = self._agent_output_status_callback(stdout_path, stderr_path)

        def callback(line: str) -> None:
            output_callback(line)
            if progress_callback is not None:
                progress_callback(line)

        return callback

    def _agent_output_status_callback(
        self,
        stdout_path: Path,
        stderr_path: Path,
    ) -> Callable[[str], None]:
        def callback(_line: str) -> None:
            now = time.monotonic()
            if now - self._last_agent_output_status_monotonic < 1.0:
                return
            self._last_agent_output_status_monotonic = now
            self._update_status(
                agent_output_chars_live=_agent_output_file_size(stdout_path, stderr_path),
                agent_last_output_at=_utc_now(),
            )

        return callback

    def _openrouter_status_callback(
        self,
        round_number: int,
        attempt: int,
    ) -> Callable[[dict], None]:
        def update(event: dict) -> None:
            if event.get("event") != "openrouter_completion_limit":
                return
            count = int(self._status.get("openrouter_max_tokens_count") or 0) + 1
            self._update_status(
                openrouter_max_tokens_count=count,
                last_openrouter_max_tokens_step=event.get("step"),
                last_openrouter_max_tokens_max_tokens=event.get("configured_max_tokens"),
                last_openrouter_max_tokens_completion_tokens=event.get("completion_tokens"),
                last_openrouter_max_tokens_reasoning_tokens=event.get("reasoning_tokens"),
            )
            self._event(
                "openrouter_max_tokens_hit",
                round=round_number,
                attempt=attempt,
                count=count,
                step=event.get("step"),
                configured_max_tokens=event.get("configured_max_tokens"),
                completion_tokens=event.get("completion_tokens"),
                reasoning_tokens=event.get("reasoning_tokens"),
            )

        return update

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

    def _can_shortcut_default_solver_evaluation(self) -> bool:
        if self._default_solver_baseline is None:
            return False
        return self._default_solver_baseline == _default_solver_signature(
            self.workspace,
            self.config.solver_wrapper,
        )

    def _write_default_solver_evaluation_result(self, round_dir: Path) -> CommandResult:
        stdout_path = round_dir / "evaluation.stdout.log"
        stderr_path = round_dir / "evaluation.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            "\n".join(
                [
                    "Puzzle Runner skipped evaluate_full.py because the configured "
                    "solver wrapper still resolves to the unchanged default Coilbench solver.",
                    (
                        f"Level {DEFAULT_SOLVER_FULL_EVAL_SCORE} (baseline): "
                        "PASS (0.00s)"
                    ),
                    (
                        f"Level {DEFAULT_SOLVER_FULL_EVAL_FIRST_FAIL} (baseline): "
                        "FAIL - unchanged default solver baseline (0.00s)"
                    ),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        argv = [
            "puzzle-runner",
            "evaluation-shortcut",
            "--score",
            str(DEFAULT_SOLVER_FULL_EVAL_SCORE),
        ]
        self._update_status(current_command=argv)
        self._event(
            "evaluation_shortcut_used",
            score=DEFAULT_SOLVER_FULL_EVAL_SCORE,
            first_failing_level=DEFAULT_SOLVER_FULL_EVAL_FIRST_FAIL,
            reason="unchanged_default_solver",
        )
        return CommandResult(
            argv=argv,
            cwd=self.workspace,
            returncode=0,
            elapsed_seconds=0.0,
            timed_out=False,
            timeout_reason=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
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
            "config_dir": str(self.config.config_path.parent),
            "workspace": str(self.workspace),
            "log_dir": str(self.log_dir),
            "round_dir": str(round_dir),
            "run_id": self.config.run_id,
        }
        return [part.format(**replacements) for part in command]

    def _agent_command(self, round_dir: Path) -> list[str]:
        if self.config.agent.backend == "openrouter":
            return [
                "openrouter-api",
                "--model",
                str(self.config.agent.model or ""),
                "--api-key-env",
                self.config.agent.api_key_env,
            ]
        command = self._render_command(self.config.agent.command, round_dir)
        command = _apply_agent_effort(self.config, command)
        return _apply_agent_model(self.config, command)

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

    def _write_final_result(self, final: FinalResult) -> None:
        body = f"""# Puzzle Runner Final Result

Run id: {final.run_id}
Agent: {self.config.agent.name}
Backend: {self.config.agent.backend}
Effort: {_agent_effort_text(self.config)}
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
Agent failure retry limit: {self.config.agent_failure_retry_limit_seconds}s
Total rounds: {final.total_rounds}
Best score: {final.best_score}
Best round: {final.best_round}
Stop reason: {final.stop_reason}
Stop detail: {final.stop_detail}
Wall time: {_format_duration(final.total_wall_seconds)}
Wall time seconds: {final.total_wall_seconds:.2f}
Agent output chars: {final.agent_output_chars}
Code lines added: {final.code_lines_added}
"""
        if self.config.agent.backend == "openrouter":
            body += (
                "OpenRouter max token hits: "
                f"{self._status.get('openrouter_max_tokens_count') or 0}\n"
            )
        if final.openrouter_usage is not None:
            body += _openrouter_final_result_text(final.openrouter_usage)
        (self.log_dir / "final_result.md").write_text(body, encoding="utf-8")

    def _append_results_summary(self, final: FinalResult) -> None:
        _ensure_results_summary_header(self.config.results_path)
        with self.config.results_path.open("a", encoding="utf-8") as handle:
            handle.write(_results_summary_row(final, self.config))

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
        with self._status_lock:
            now = _utc_now()
            if not self._status:
                self._status = {
                    "run_id": self.config.run_id,
                    "active": True,
                    "phase": "initializing",
                    "agent": self.config.agent.name,
                    "backend": self.config.agent.backend,
                    "agent_effort": self.config.agent.effort,
                    "agent_stream_format": _agent_stream_format(self.config),
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
                    "score_history": [],
                    "last_improved": None,
                    "agent_attempt": None,
                    "agent_error_count": 0,
                    "last_agent_model_not_found": False,
                    "last_agent_model_not_found_model": None,
                    "agent_output_chars_live": 0,
                    "agent_last_output_at": None,
                    "openrouter_max_tokens_count": 0,
                    "last_openrouter_max_tokens_step": None,
                    "last_openrouter_max_tokens_max_tokens": None,
                    "last_openrouter_max_tokens_completion_tokens": None,
                    "last_openrouter_max_tokens_reasoning_tokens": None,
                    "agent_retry_count": 0,
                    "agent_retry_delay_seconds": None,
                    "agent_retry_remaining_seconds": None,
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


def _guard_findings_payload(findings: list[GuardFinding]) -> list[dict]:
    return [_jsonable(dataclasses.asdict(finding)) for finding in findings]


def _agent_output_file_size(stdout_path: Path, stderr_path: Path) -> int:
    total = 0
    for path in (stdout_path, stderr_path):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _is_git_workspace(workspace: Path) -> bool:
    return (workspace / ".git").exists()


def _default_solver_signature(
    workspace: Path,
    solver_wrapper: str,
) -> dict[str, str | None] | None:
    wrapper_rel = _normalize_git_path(solver_wrapper)
    wrapper_path = workspace / wrapper_rel
    default_solver_path = workspace / "coil_solver.py"

    if not wrapper_path.is_file() or not default_solver_path.is_file():
        return None
    if (workspace / "solver").exists() or (workspace / "solver.py").exists():
        return None

    try:
        wrapper_text = wrapper_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "coil_solver.py" not in wrapper_text:
        return None

    try:
        return {
            wrapper_rel: _file_signature(wrapper_path),
            "coil_solver.py": _file_signature(default_solver_path),
            "solver": None,
            "solver.py": None,
        }
    except OSError:
        return None


def _file_signature(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    executable_bits = path.stat().st_mode & 0o111
    return f"{digest.hexdigest()}:{executable_bits:o}"


def normalize_script_line_endings(workspace: Path) -> list[str]:
    changed_paths = []
    if not workspace.exists():
        return changed_paths

    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or _skip_line_ending_path(path, workspace):
            continue

        try:
            if path.stat().st_size > MAX_LINE_ENDING_NORMALIZE_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue

        if b"\r\n" not in data or not _is_script_like_path(path, data):
            continue

        path.write_bytes(data.replace(b"\r\n", b"\n"))
        changed_paths.append(path.relative_to(workspace).as_posix())

    return changed_paths


def _skip_line_ending_path(path: Path, workspace: Path) -> bool:
    rel_path = path.relative_to(workspace).as_posix()
    return rel_path == ".git" or rel_path.startswith(".git/")


def _is_script_like_path(path: Path, data: bytes) -> bool:
    if data.startswith(b"#!"):
        return True
    if path.name in SCRIPT_LINE_ENDING_FILENAMES:
        return True
    return path.suffix.lower() in SCRIPT_LINE_ENDING_SUFFIXES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _agent_result_is_retryable(result: CommandResult) -> bool:
    return (
        result.returncode != 0
        and result.returncode != AGENT_CONFIG_ERROR_RETURN_CODE
        and result.returncode != AGENT_MAX_STEPS_RETURN_CODE
        and not result.timed_out
    )


def _agent_stream_format(config: RunnerConfig) -> str | None:
    command = config.agent.command
    if config.agent.backend == "claude-code":
        if not _command_uses_stream_json(command):
            return None
        return "claude-stream-json"
    if _is_gemini_backend(config):
        if not _command_uses_stream_json(command):
            return None
        return "gemini-stream-json"
    if _is_opencode_backend(config):
        if not _command_uses_opencode_json(command):
            return None
        return "opencode-json"
    return None


def _agent_stdout_completion_predicate(config: RunnerConfig) -> Callable[[str], bool] | None:
    if _agent_stream_format(config) in {"claude-stream-json", "gemini-stream-json"}:
        return _is_terminal_stream_result_line
    return None


def _agent_stdout_line_callback(config: RunnerConfig) -> Callable[[str], None] | None:
    if not config.echo_agent_progress or config.echo_agent_output:
        return None
    if _agent_stream_format(config) != "opencode-json":
        return None

    state = {"step": 0}

    def print_progress(line: str) -> None:
        progress = _opencode_progress_line(line, state)
        if progress:
            print(progress, flush=True)

    return print_progress


def _is_terminal_claude_result_line(line: str) -> bool:
    return _is_terminal_stream_result_line(line)


def _is_terminal_stream_result_line(line: str) -> bool:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(event, dict) and event.get("type") == "result"


def _agent_returned_error(config: RunnerConfig, stdout_path: Path) -> bool:
    agent_stream_format = _agent_stream_format(config)
    if agent_stream_format == "claude-stream-json":
        return _claude_stdout_has_error_result(stdout_path)
    if agent_stream_format == "gemini-stream-json":
        return _gemini_stdout_has_error_result(stdout_path)
    if agent_stream_format == "opencode-json":
        return _opencode_stdout_has_error_result(stdout_path)
    return False


def _agent_model_not_found_error(stdout_path: Path, stderr_path: Path) -> bool:
    return _agent_model_not_found_detail(stdout_path, stderr_path) is not None


def _agent_model_not_found_detail(stdout_path: Path, stderr_path: Path) -> str | None:
    saw_model_not_found = False
    for text in (_tail_file_text(stdout_path), _tail_file_text(stderr_path)):
        if not text:
            continue
        detail = _model_not_found_detail_from_text(text)
        if detail:
            return detail
        if MODEL_NOT_FOUND_ERROR_RE.search(text):
            saw_model_not_found = True
    return "unknown" if saw_model_not_found else None


def _model_not_found_detail_from_text(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        detail = _model_not_found_detail_from_json_line(stripped)
        if detail:
            return detail

    message_match = MODEL_NOT_FOUND_MESSAGE_RE.search(text)
    if message_match:
        return _clean_model_not_found_model(message_match.group(1))

    model_match = PROVIDER_MODEL_ID_RE.search(text)
    if not model_match:
        return None

    model = model_match.group(1)
    provider_match = PROVIDER_ID_RE.search(text)
    if provider_match:
        provider = provider_match.group(1)
        if provider and not model.startswith(f"{provider}/"):
            return f"{provider}/{model}"
    return model


def _model_not_found_detail_from_json_line(line: str) -> str | None:
    if not line.startswith("{"):
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    message = _opencode_error_message(payload)
    if not message:
        return None
    message_match = MODEL_NOT_FOUND_MESSAGE_RE.search(message)
    if message_match:
        return _clean_model_not_found_model(message_match.group(1))
    if MODEL_NOT_FOUND_ERROR_RE.search(message):
        return "unknown"
    return None


def _clean_model_not_found_model(value: str) -> str:
    return value.strip().strip("`'\"").rstrip(".,;:")


def _tail_file_text(path: Path, max_bytes: int = MAX_AGENT_ERROR_SCAN_BYTES) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _claude_stdout_has_error_result(path: Path) -> bool:
    for event in _claude_stream_events(path):
        if event.get("type") == "result" and event.get("is_error") is True:
            return True
    return False


def _gemini_stdout_has_error_result(path: Path) -> bool:
    for event in _claude_stream_events(path):
        if event.get("type") == "error":
            return True
        if event.get("type") != "result":
            continue
        status = event.get("status")
        if isinstance(status, str) and status.lower() != "success":
            return True
    return False


def _opencode_stdout_has_error_result(path: Path) -> bool:
    return any(event.get("type") == "error" for event in _claude_stream_events(path))


def _opencode_progress_line(line: str, state: dict) -> str | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None

    event_type = event.get("type")
    if event_type == "step_start":
        state["step"] = int(state.get("step") or 0) + 1
        return f"--- OpenCode step {state['step']} ---"
    if event_type == "text":
        text = _compact_progress_text(_opencode_event_text(event), 260)
        return f"OpenCode: {text}" if text else None
    if event_type == "tool_use":
        summary = _opencode_tool_progress(event)
        return f"OpenCode tool: {summary}" if summary else None
    if event_type == "step_finish":
        return _opencode_step_finish_progress(event)
    if event_type == "error":
        message = _compact_progress_text(_opencode_error_message(event), 260)
        return f"OpenCode error: {message}" if message else "OpenCode error"
    return None


def _opencode_tool_progress(event: dict) -> str | None:
    part = event.get("part")
    if not isinstance(part, dict):
        return None

    tool = str(part.get("tool") or "tool")
    state = part.get("state")
    status = None
    detail = None
    exit_code = None
    if isinstance(state, dict):
        status_value = state.get("status")
        status = str(status_value) if isinstance(status_value, str) and status_value else None
        detail = _opencode_tool_detail(state)
        if tool.lower() == "todowrite" and detail and detail.endswith(" todos"):
            detail = None
        metadata = state.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("exit"), int):
            exit_code = metadata["exit"]

    parts = [tool]
    if status:
        parts.append(status)
    facts = []
    duration = _opencode_tool_duration(state) if isinstance(state, dict) else None
    if duration:
        facts.append(duration)
    if exit_code is not None:
        facts.append(f"exit {exit_code}")
    if facts:
        parts.append(f"({', '.join(facts)})")
    if detail:
        parts.append(f"- {detail}")
    extra = _opencode_tool_extra(tool, state) if isinstance(state, dict) else None
    summary = " ".join(parts)
    if extra:
        summary += f"; {extra}"
    return _compact_progress_text(summary, 420)


def _opencode_tool_detail(state: dict) -> str | None:
    title = state.get("title")
    if isinstance(title, str) and title.strip():
        return _compact_progress_text(title, 160)

    tool_input = state.get("input")
    if not isinstance(tool_input, dict):
        return None

    for key in ("description", "filePath", "path", "command"):
        value = tool_input.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if key in {"filePath", "path"}:
            value = _opencode_display_path(value)
        return _compact_progress_text(value, 160)
    return None


def _opencode_tool_duration(state: dict) -> str | None:
    timing = state.get("time")
    if not isinstance(timing, dict):
        return None
    start = timing.get("start")
    end = timing.get("end")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    elapsed = max((end - start) / 1000.0, 0)
    if elapsed < 1:
        return f"{elapsed * 1000:.0f}ms"
    if elapsed < 10:
        return f"{elapsed:.2f}s"
    return f"{elapsed:.1f}s"


def _opencode_tool_extra(tool: str, state: dict) -> str | None:
    tool_input = state.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    tool_name = tool.lower()
    extras: list[str] = []
    error = _opencode_tool_error(state)
    if error:
        extras.append(f"error: {error}")
    if tool_name == "bash":
        command = tool_input.get("command")
        description = tool_input.get("description")
        if isinstance(command, str) and command.strip() and command.strip() != str(description or "").strip():
            extras.append(f"cmd: {_compact_progress_text(command, 160)}")
        output = metadata.get("output")
        if not isinstance(output, str):
            output = state.get("output")
        if isinstance(output, str) and output.strip():
            extras.append(_opencode_output_preview(output))
    elif tool_name == "read":
        preview = metadata.get("preview")
        if isinstance(preview, str) and preview:
            extras.append(f"preview {_line_char_summary(preview)}")
        if metadata.get("truncated") is True:
            extras.append("truncated")
    elif tool_name in {"write", "edit", "patch"}:
        changed = _opencode_changed_text_summary(tool_input)
        if changed:
            extras.append(changed)
    elif tool_name == "todowrite":
        todo_summary = _opencode_todo_summary(tool_input, metadata)
        if todo_summary:
            extras.append(todo_summary)
    else:
        output = metadata.get("output")
        if not isinstance(output, str):
            output = state.get("output")
        if isinstance(output, str) and output.strip():
            extras.append(_opencode_output_preview(output))

    return "; ".join(extras) if extras else None


def _opencode_tool_error(state: dict) -> str | None:
    error = state.get("error")
    if isinstance(error, str) and error.strip():
        return _compact_progress_text(error, 180)
    if isinstance(error, dict):
        for key in ("message", "name", "type"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return _compact_progress_text(value, 180)
    return None


def _opencode_changed_text_summary(tool_input: dict) -> str | None:
    candidates = []
    for key in ("content", "newString", "new_string", "patch"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)
    if not candidates:
        return None
    text = max(candidates, key=len)
    return _line_char_summary(text)


def _opencode_todo_summary(tool_input: dict, metadata: dict) -> str | None:
    todos = metadata.get("todos")
    if not isinstance(todos, list):
        todos = tool_input.get("todos")
    if not isinstance(todos, list) or not todos:
        return None

    counts: dict[str, int] = {}
    active = None
    for item in todos:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        if active is None and status == "in_progress":
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                active = _compact_progress_text(content, 80)
    parts = [f"{len(todos)} todos"]
    if counts:
        parts.append(", ".join(f"{key} {value}" for key, value in sorted(counts.items())))
    if active:
        parts.append(f"active: {active}")
    return "; ".join(parts)


def _opencode_output_preview(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    preview = " | ".join(lines[:2]) if lines else text.strip()
    return f"output {_line_char_summary(text)}: {_compact_progress_text(preview, 180)}"


def _line_char_summary(text: str) -> str:
    line_count = len(text.splitlines()) or 1
    line_word = "line" if line_count == 1 else "lines"
    char_word = "char" if len(text) == 1 else "chars"
    return f"{line_count:,} {line_word}, {len(text):,} {char_word}"


def _opencode_display_path(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return value
    for marker in ("levels_public", "levels_secret_even", "coil_check", "src", "tests", "scripts"):
        if marker in parts:
            return "/".join(parts[parts.index(marker) :])
    return parts[-1]


def _opencode_step_finish_progress(event: dict) -> str:
    part = event.get("part")
    if not isinstance(part, dict):
        return "OpenCode step finished"

    details = []
    reason = part.get("reason")
    if isinstance(reason, str) and reason:
        details.append(reason)
    tokens = part.get("tokens")
    if isinstance(tokens, dict):
        total = tokens.get("total")
        if isinstance(total, int):
            details.append(f"tokens {total:,}")
        reasoning = tokens.get("reasoning")
        if isinstance(reasoning, int) and reasoning > 0:
            details.append(f"reasoning {reasoning:,}")
    cost = part.get("cost")
    if isinstance(cost, (int, float)):
        details.append(f"cost ${float(cost):.4f}")
    if not details:
        return "OpenCode step finished"
    return "OpenCode step finished: " + ", ".join(details)


def _opencode_error_message(event: dict) -> str:
    error = event.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        data = error.get("data")
        if isinstance(data, dict):
            value = data.get("message")
            if isinstance(value, str) and value:
                return value
        for key in ("message", "name", "type"):
            value = error.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _compact_progress_text(text: str, limit: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: max(limit - 3, 0)].rstrip() + "..."


def _apply_agent_effort(config: RunnerConfig, command: list[str]) -> list[str]:
    effort = config.agent.effort
    if not effort:
        return command
    if config.agent.backend == "claude-code":
        if _command_has_option(command, "--effort"):
            return command
        return [*command, "--effort", effort]
    if _is_opencode_backend(config):
        if _command_has_option(command, "--variant"):
            return command
        return [*command, "--variant", effort]
    return command


def _apply_agent_model(config: RunnerConfig, command: list[str]) -> list[str]:
    model = config.agent.model
    if not model:
        return command
    if _is_gemini_backend(config):
        if _command_has_option(command, "--model"):
            return command
        return [*command, "--model", model]
    if _is_opencode_backend(config):
        if _command_has_any_option(command, {"--model", "-m"}):
            return command
        return [*command, "--model", model]
    return command


def _is_gemini_backend(config: RunnerConfig) -> bool:
    return config.agent.backend == "gemini-cli" or config.agent.backend.startswith("gemini-")


def _is_opencode_backend(config: RunnerConfig) -> bool:
    return config.agent.backend == "opencode" or config.agent.backend.startswith("opencode-")


def _opencode_required_auth_provider(config: RunnerConfig) -> str | None:
    if not _is_opencode_backend(config):
        return None
    model = config.agent.model
    if not model or "/" not in model:
        return None
    provider, _model = model.split("/", 1)
    if provider == "openrouter":
        return provider
    return None


def _opencode_model_problem_from_listing(
    config: RunnerConfig,
    *,
    provider: str | None,
    listing: str,
    returncode: int,
    timed_out: bool,
) -> dict | None:
    model = config.agent.model
    if not model:
        return None

    command_text = f"opencode models {provider}" if provider else "opencode models"
    if timed_out:
        detail = (
            f"OpenCode model preflight timed out while running `{command_text}`. "
            "Puzzle Runner stopped before benchmark download/setup."
        )
    elif returncode != 0:
        detail = (
            f"OpenCode model preflight failed while running `{command_text}` "
            f"(exit {returncode}). Puzzle Runner stopped before benchmark download/setup."
        )
    else:
        normalized_lines = {
            line.strip()
            for line in _strip_ansi(listing).splitlines()
            if line.strip()
        }
        if model in normalized_lines:
            return None
        detail = (
            f"OpenCode models did not list configured model `{model}` from `{command_text}`. "
            "Puzzle Runner stopped before benchmark download/setup."
        )

    return {
        "provider": provider,
        "model": model,
        "detail": detail,
    }


def _opencode_auth_problem_from_listing(
    config: RunnerConfig,
    provider: str,
    listing: str,
    environ,
) -> dict | None:
    if provider != "openrouter":
        return None

    env_var = config.agent.api_key_env or "OPENROUTER_API_KEY"
    if environ.get(env_var) or environ.get("OPENROUTER_API_KEY"):
        return None

    normalized = _strip_ansi(listing).lower()
    if "openrouter" in normalized or "openrouter_api_key" in normalized:
        return None

    return {
        "provider": "openrouter",
        "model": config.agent.model,
        "env_var": "OPENROUTER_API_KEY",
        "detail": (
            "OpenCode has no OpenRouter credential and OPENROUTER_API_KEY is not set. "
            "Run `opencode auth login --provider openrouter` or set OPENROUTER_API_KEY."
        ),
    }


def _safe_log_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-_.") or "value"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)


def _command_has_option(command: list[str], option: str) -> bool:
    return any(part == option or part.startswith(f"{option}=") for part in command)


def _command_has_any_option(command: list[str], options: set[str]) -> bool:
    return any(_command_has_option(command, option) for option in options)


def _command_uses_stream_json(command: list[str]) -> bool:
    for index, part in enumerate(command):
        if part == "--output-format" and index + 1 < len(command) and command[index + 1] == "stream-json":
            return True
        if part == "--output-format=stream-json":
            return True
    return False


def _command_uses_opencode_json(command: list[str]) -> bool:
    for index, part in enumerate(command):
        if part == "--format" and index + 1 < len(command) and command[index + 1] == "json":
            return True
        if part == "--format=json":
            return True
    return False


def _agent_attempt_log_paths(round_dir: Path, attempt: int) -> tuple[Path, Path]:
    if attempt == 1:
        return round_dir / "agent.stdout.log", round_dir / "agent.stderr.log"
    prefix = f"agent.attempt-{attempt:03d}"
    return round_dir / f"{prefix}.stdout.log", round_dir / f"{prefix}.stderr.log"


def count_agent_output_chars(log_dir: Path, *, agent_stream_format: str | None = None) -> int:
    if agent_stream_format == "claude-stream-json":
        return _count_claude_agent_text_chars(log_dir)
    if agent_stream_format == "gemini-stream-json":
        return _count_gemini_agent_text_chars(log_dir)
    if agent_stream_format == "opencode-json":
        return _count_opencode_agent_text_chars(log_dir)

    total = 0
    seen: set[Path] = set()
    for pattern in AGENT_OUTPUT_LOG_PATTERNS:
        for path in log_dir.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            try:
                total += len(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return total


def _count_claude_agent_text_chars(log_dir: Path) -> int:
    total = 0
    seen: set[Path] = set()
    for path in log_dir.glob("round-*/agent*.stdout.log"):
        if path in seen:
            continue
        seen.add(path)
        total += _claude_stream_text_char_count(path)
    return total


def _count_gemini_agent_text_chars(log_dir: Path) -> int:
    total = 0
    seen: set[Path] = set()
    for path in log_dir.glob("round-*/agent*.stdout.log"):
        if path in seen:
            continue
        seen.add(path)
        total += _gemini_stream_text_char_count(path)
    return total


def _count_opencode_agent_text_chars(log_dir: Path) -> int:
    total = 0
    seen: set[Path] = set()
    for path in log_dir.glob("round-*/agent*.stdout.log"):
        if path in seen:
            continue
        seen.add(path)
        total += _opencode_stream_text_char_count(path)
    return total


def _claude_stream_text_char_count(path: Path) -> int:
    delta_chars = 0
    assistant_chars = 0
    result_chars = 0

    for event in _claude_stream_events(path):
        event_type = event.get("type")
        if event_type == "stream_event":
            stream_event = event.get("event")
            if not isinstance(stream_event, dict) or stream_event.get("type") != "content_block_delta":
                continue
            delta = stream_event.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                delta_chars += len(delta["text"])
        elif event_type == "assistant":
            message = event.get("message")
            if isinstance(message, dict):
                assistant_chars += len(_assistant_message_text(message))
        elif event_type == "result":
            result = event.get("result")
            if isinstance(result, str):
                result_chars += len(result)

    return delta_chars or assistant_chars or result_chars


def _gemini_stream_text_char_count(path: Path) -> int:
    delta_chars = 0
    assistant_chars = 0

    for event in _claude_stream_events(path):
        if event.get("type") != "message" or event.get("role") != "assistant":
            continue
        text = _gemini_message_text(event)
        if event.get("delta") is True:
            delta_chars += len(text)
        else:
            assistant_chars += len(text)

    return delta_chars or assistant_chars


def _opencode_stream_text_char_count(path: Path) -> int:
    total = 0
    for event in _claude_stream_events(path):
        if event.get("type") != "text":
            continue
        total += len(_opencode_event_text(event))
    return total


def _opencode_event_text(event: dict) -> str:
    part = event.get("part")
    if isinstance(part, dict) and isinstance(part.get("text"), str):
        return part["text"]
    text = event.get("text")
    if isinstance(text, str):
        return text
    return ""


def _claude_stream_events(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith("{"):
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event
    except OSError:
        return


def _gemini_message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _assistant_message_text(message: dict) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def count_code_lines_added(workspace: Path) -> int:
    if not workspace.exists():
        return 0

    total = 0
    for path, additions in _tracked_code_additions(workspace):
        if _is_counted_code_path(path, workspace / path):
            total += additions

    for path in _untracked_workspace_paths(workspace):
        if not _is_counted_code_path(path, workspace / path):
            continue
        total += _count_text_lines(workspace / path) or 0

    return total


def _tracked_code_additions(workspace: Path) -> list[tuple[str, int]]:
    result = _git_numstat(workspace, ["diff", "HEAD", "--numstat", "--"])
    if result is None:
        result = _git_numstat(workspace, ["diff", "--numstat", "--"])
    if result is None:
        return []

    changes: list[tuple[str, int]] = []
    for line in result.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        additions_text = parts[0]
        path = _normalize_git_path(parts[2])
        additions = int(additions_text) if additions_text.isdigit() else 0
        changes.append((path, additions))
    return changes


def _git_numstat(workspace: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _untracked_workspace_paths(workspace: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [_normalize_git_path(line) for line in result.stdout.splitlines() if line.strip()]


def _normalize_git_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def _is_counted_code_path(path: str, full_path: Path) -> bool:
    normalized = _normalize_git_path(path)
    if _ignore_code_path(normalized):
        return False
    path_obj = Path(normalized)
    if path_obj.name in CODE_FILENAMES:
        return True
    if path_obj.suffix.lower() in CODE_SUFFIXES:
        return True
    return _shebang_mentions(full_path)


def _ignore_code_path(path: str) -> bool:
    if path in NOISY_CODE_EXACT_PATHS:
        return True
    if path.endswith(NOISY_CODE_SUFFIXES):
        return True
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in NOISY_CODE_PREFIXES)


def _shebang_mentions(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            first_line = handle.readline(200).decode("utf-8", errors="ignore")
    except OSError:
        return False
    return first_line.startswith("#!")


def _count_text_lines(path: Path) -> int | None:
    try:
        if path.stat().st_size > MAX_UNTRACKED_CODE_LINE_COUNT_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _ensure_results_summary_header(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(RESULTS_SUMMARY_HEADER, encoding="utf-8")
        return

    text = path.read_text(encoding="utf-8", errors="replace")
    if RESULTS_SUMMARY_HEADER.splitlines()[0] in text.splitlines():
        return

    migrated = _migrate_results_summary_schema(text)
    if migrated != text:
        path.write_text(migrated, encoding="utf-8")
        return

    with path.open("a", encoding="utf-8") as handle:
        if text and not text.endswith("\n"):
            handle.write("\n")
        if text.strip():
            handle.write("\n")
        handle.write(RESULTS_SUMMARY_HEADER)


def _migrate_results_summary_effort_column(text: str) -> str:
    return _migrate_results_summary_schema(text)


def _migrate_results_summary_schema(text: str) -> str:
    lines = text.splitlines()
    output = []
    in_migrated_table = False

    for line in lines:
        if line in {
            RESULTS_SUMMARY_NO_EFFORT_HEADER,
            RESULTS_SUMMARY_NO_OPENROUTER_HEADER,
            RESULTS_SUMMARY_OLD_LOGS_HEADER,
        }:
            output.append(RESULTS_SUMMARY_HEADER.splitlines()[0])
            in_migrated_table = True
            continue

        if in_migrated_table and line in {
            RESULTS_SUMMARY_NO_EFFORT_SEPARATOR,
            RESULTS_SUMMARY_NO_OPENROUTER_SEPARATOR,
            RESULTS_SUMMARY_OLD_LOGS_SEPARATOR,
        }:
            output.append(RESULTS_SUMMARY_HEADER.splitlines()[1])
            continue

        if in_migrated_table and line.startswith("|"):
            cells = _markdown_table_cells(line)
            migrated_cells = _migrate_results_summary_row_cells(cells)
            if migrated_cells is not None:
                output.append(_markdown_table_row(migrated_cells))
                continue

        if in_migrated_table and not line.startswith("|"):
            in_migrated_table = False
        output.append(line)

    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + trailing_newline


def _migrate_results_summary_row_cells(cells: list[str]) -> list[str] | None:
    if len(cells) == 14:
        return cells
    if len(cells) == 11:
        return [*cells, "", "", ""]
    if len(cells) == 10:
        migrated = list(cells)
        migrated.insert(2, "")
        return [*migrated, "", "", ""]
    if len(cells) == 8:
        return [
            cells[0],
            cells[1],
            "",
            cells[2],
            cells[3],
            cells[4],
            cells[5],
            cells[6],
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    return None


def _markdown_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _markdown_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _results_summary_row(final: FinalResult, config: RunnerConfig) -> str:
    usage = final.openrouter_usage or {}
    row = [
        final.run_id,
        config.agent.name,
        _agent_effort_text(config),
        str(final.best_score),
        "" if final.best_round is None else str(final.best_round),
        str(final.total_rounds),
        final.stop_reason,
        f"{config.evaluation_timeout_seconds}s",
        _format_duration(final.total_wall_seconds),
        str(final.agent_output_chars),
        str(final.code_lines_added),
        _openrouter_int_cell(usage, "calls"),
        _openrouter_cost_cell(usage),
        _openrouter_int_cell(usage, "total_tokens"),
    ]
    return "| " + " | ".join(_escape_table_cell(value) for value in row) + " |\n"


def _openrouter_usage_for_result(config: RunnerConfig, log_dir: Path) -> dict | None:
    if config.agent.backend == "openrouter":
        summary = summarize_openrouter_usage(log_dir)
    elif _agent_uses_opencode_openrouter(config):
        summary = summarize_opencode_openrouter_usage(log_dir, model=config.agent.model)
    else:
        return None

    if summary.calls == 0:
        return None
    return openrouter_usage_to_dict(summary)


def _agent_uses_opencode_openrouter(config: RunnerConfig) -> bool:
    backend = config.agent.backend
    if backend != "opencode" and not backend.startswith("opencode-"):
        return False

    candidates = []
    if config.agent.model:
        candidates.append(config.agent.model)
    candidates.extend(config.agent.command)
    return any(
        candidate.startswith("openrouter/")
        or candidate.startswith("--model=openrouter/")
        or candidate.startswith("-m=openrouter/")
        for candidate in candidates
    )


def _openrouter_final_result_text(usage: dict) -> str:
    lines = [
        "",
        "OpenRouter calls: " + _openrouter_int_cell(usage, "calls"),
        "OpenRouter metadata calls: " + _openrouter_int_cell(usage, "metadata_calls"),
        "OpenRouter metadata failures: " + _openrouter_int_cell(usage, "metadata_failures"),
        "OpenRouter cost USD: " + _openrouter_cost_cell(usage),
        "OpenRouter prompt tokens: " + _openrouter_int_cell(usage, "prompt_tokens"),
        "OpenRouter completion tokens: " + _openrouter_int_cell(usage, "completion_tokens"),
        "OpenRouter total tokens: " + _openrouter_int_cell(usage, "total_tokens"),
        "OpenRouter native reasoning tokens: " + _openrouter_int_cell(usage, "native_reasoning_tokens"),
        "OpenRouter native cached tokens: " + _openrouter_int_cell(usage, "native_cached_tokens"),
        "OpenRouter last provider: " + str(usage.get("last_provider") or ""),
        "OpenRouter last finish reason: " + str(usage.get("last_finish_reason") or ""),
    ]
    return "\n".join(lines) + "\n"


def _openrouter_int_cell(usage: dict, key: str) -> str:
    value = usage.get(key)
    return str(value) if isinstance(value, int) else ""


def _openrouter_cost_cell(usage: dict) -> str:
    value = usage.get("cost_usd")
    if not isinstance(value, (int, float)):
        return ""
    return f"${float(value):.6f}"


def _agent_effort_text(config: RunnerConfig) -> str:
    if config.agent.effort:
        return config.agent.effort
    return _codex_reasoning_effort_from_command(config.agent.command) or ""


def _codex_reasoning_effort_from_command(command: list[str]) -> str | None:
    for part in command:
        match = CODEX_REASONING_EFFORT_RE.search(part)
        if match:
            return match.group(1)
    return None


def _escape_table_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def explain_stop_reason(stop_reason: str, config: RunnerConfig, status: dict) -> str:
    if stop_reason == "agent_timeout":
        elapsed = _duration_text(status.get("last_agent_elapsed_seconds"))
        return (
            "Agent call exceeded "
            f"agent_timeout_seconds={config.agent_timeout_seconds}s"
            f"{elapsed} and was killed."
        )
    if stop_reason == "agent_idle_timeout":
        return (
            "Agent call produced no stdout or stderr for "
            f"agent_idle_timeout_seconds={config.agent_idle_timeout_seconds}s "
            "and was killed."
        )
    if stop_reason == "agent_failed":
        return (
            "Agent process exited nonzero. Puzzle Runner retried eligible failures "
            f"for up to agent_failure_retry_limit_seconds="
            f"{config.agent_failure_retry_limit_seconds}s."
        )
    if stop_reason == "agent_model_not_found":
        problem = status.get("agent_model_preflight_problem")
        if isinstance(problem, dict):
            detail = problem.get("detail")
            if isinstance(detail, str) and detail:
                return detail
        model = status.get("last_agent_model_not_found_model") or config.agent.model
        model_text = f" for {model}" if isinstance(model, str) and model else ""
        return (
            f"Agent reported a model-not-found error{model_text}. "
            "Puzzle Runner stopped immediately without running evaluation."
        )
    if stop_reason == "agent_auth_missing":
        problem = status.get("agent_auth_problem")
        if isinstance(problem, dict):
            detail = problem.get("detail")
            if isinstance(detail, str) and detail:
                return detail
        return "Agent provider credentials are missing."
    if stop_reason == "agent_max_steps":
        return (
            "OpenRouter agent reached "
            f"agent.max_steps={config.agent.max_steps} tool-call steps without ending "
            "the turn with a normal assistant response. Puzzle Runner stopped before "
            "running evaluation on an arbitrary intermediate workspace state."
        )
    if stop_reason == "evaluation_timeout":
        return (
            "Evaluation process exceeded "
            f"evaluation_process_timeout_seconds={config.evaluation_process_timeout_seconds}s "
            "and was killed."
        )
    if stop_reason == "evaluation_failed":
        return "Evaluation process exited nonzero after running evaluate_full.py."
    if stop_reason == "stale_limit":
        return (
            "No score improvement for "
            f"stale_limit={config.stale_limit} consecutive completed evaluations."
        )
    if stop_reason == "max_rounds":
        return f"Reached max_rounds={config.max_rounds}."
    if stop_reason == "forbidden_edit_detected":
        summary = _guard_findings_summary(status.get("guard_findings"))
        if summary:
            phase = status.get("guard_phase")
            phase_text = f" during {phase}" if isinstance(phase, str) and phase else ""
            return f"Forbidden path guard detected{phase_text}: {summary}."
        return "Forbidden path guard detected edits under configured forbidden_paths."
    return "Run stopped."


def _guard_findings_summary(findings) -> str:
    if not isinstance(findings, list) or not findings:
        return ""

    parts = []
    for finding in findings[:5]:
        if not isinstance(finding, dict):
            continue
        path = str(finding.get("path") or "?")
        reason = str(finding.get("reason") or "changed forbidden file")
        pattern = finding.get("pattern")
        if isinstance(pattern, str) and pattern:
            parts.append(f"{reason}: {path} (matched {pattern})")
        else:
            parts.append(f"{reason}: {path}")

    remaining = len(findings) - len(parts)
    if remaining > 0:
        parts.append(f"{remaining} more forbidden {_plural('change', remaining)}")
    return "; ".join(parts)


def _plural(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"


def _duration_text(value) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    return f" after {seconds:.2f}s"


def _status_markdown(status: dict) -> str:
    latest = status.get("latest") or {}
    lines = [
        "# Puzzle Runner Status",
        "",
        f"- Run ID: `{status.get('run_id')}`",
        f"- Active: `{status.get('active')}`",
        f"- Phase: `{status.get('phase')}`",
        f"- Agent: `{status.get('agent')}`",
        f"- Agent Effort: `{status.get('agent_effort')}`",
        f"- Agent Stream Format: `{status.get('agent_stream_format')}`",
        f"- Round: `{status.get('current_round')}/{status.get('max_rounds')}`",
        f"- Best Score: `{status.get('best_score')}`",
        f"- Best Round: `{status.get('best_round')}`",
        f"- Scores: `{_score_history_text(status)}`",
        f"- Last Improved: `{status.get('last_improved')}`",
        f"- Agent Errors: `{status.get('agent_error_count')}`",
        f"- Agent Model Not Found: `{status.get('last_agent_model_not_found')}`",
        f"- Agent Model Not Found Model: `{status.get('last_agent_model_not_found_model')}`",
    ]
    if status.get("backend") == "openrouter":
        lines.append(
            f"- OpenRouter Max Token Hits: `{status.get('openrouter_max_tokens_count')}`"
        )
    lines.extend(
        [
            f"- No-Progress Count: `{status.get('stale_count')}/{status.get('stale_limit')}`",
            f"- Remaining No-Progress Tries: `{status.get('remaining_no_progress_tries')}`",
            f"- Stop Reason: `{status.get('stop_reason')}`",
        ]
    )
    if status.get("stop_detail"):
        lines.append(f"- Stop Detail: `{status.get('stop_detail')}`")
    auth_problem = status.get("agent_auth_problem")
    if isinstance(auth_problem, dict) and auth_problem.get("detail"):
        lines.append(f"- Agent Auth Problem: `{auth_problem.get('detail')}`")
    guard_summary = _guard_findings_summary(status.get("guard_findings"))
    if guard_summary:
        lines.append(f"- Forbidden Changes: `{guard_summary}`")
    if status.get("agent_output_chars") is not None:
        lines.append(f"- Agent Output Chars: `{status.get('agent_output_chars')}`")
    if status.get("code_lines_added") is not None:
        lines.append(f"- Code Lines Added: `{status.get('code_lines_added')}`")
    if status.get("openrouter_usage") is not None:
        usage = status.get("openrouter_usage") or {}
        lines.append(f"- OpenRouter Calls: `{usage.get('calls')}`")
        lines.append(f"- OpenRouter Cost: `{_openrouter_cost_cell(usage)}`")
        lines.append(f"- OpenRouter Tokens: `{usage.get('total_tokens')}`")
    lines.extend(
        [
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
    )

    if latest:
        lines.extend(["", "## Latest Files", ""])
        for key, value in sorted(latest.items()):
            lines.append(f"- {key}: `{value}`")

    command = status.get("current_command")
    if command:
        lines.extend(["", "## Current Command", "", "```sh", " ".join(command), "```"])

    lines.append("")
    return "\n".join(lines)


def _score_history_text(status: dict) -> str:
    history = status.get("score_history")
    if isinstance(history, list) and history:
        return ", ".join(str(item) for item in history)
    if status.get("last_score") is not None:
        return str(status.get("last_score"))
    return "-"
