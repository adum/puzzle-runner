from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from json import JSONDecoder
from pathlib import Path
from typing import Any

from .config import RunnerConfig
from .process import CommandResult
from .prompts import SENTINEL


MAX_OBSERVATION_CHARS = 60_000
DEFAULT_READ_CHARS = 20_000
HTTP_REFERER = "https://github.com/adum/puzzle-runner"
APP_TITLE = "Puzzle Runner"


SYSTEM_PROMPT = f"""You are an autonomous coding agent controlled by Puzzle Runner.

You are working inside a benchmark workspace. You cannot directly see or edit files
unless you use the JSON actions below.

Return exactly one JSON object each turn, with no markdown wrapper:

{{"action":"shell","command":"python3 evaluate.py --start 1 --timeout 10","timeout_seconds":120}}
{{"action":"read_file","path":"prompts.txt","max_chars":20000}}
{{"action":"write_file","path":"solver.py","content":"..."}}
{{"action":"finish","message":"summary of what changed\\n{SENTINEL}"}}

Rules:
- Use shell/read_file/write_file actions to inspect, test, and edit the workspace.
- Do not inspect private level contents, decrypt private levels, or expose benchmark secrets.
- Do not modify benchmark assets, level files, evaluator scripts, or checker source.
- Do not run evaluate_full.py; Puzzle Runner owns full evaluation.
- When you are done with this agent call, use the finish action.
"""


class OpenRouterAgentError(RuntimeError):
    pass


def run_openrouter_agent(
    config: RunnerConfig,
    *,
    cwd: Path,
    prompt: str,
    round_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int | None,
    echo: bool,
) -> CommandResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with stdout_path.open("w", encoding="utf-8", errors="replace") as out_handle:
        with stderr_path.open("w", encoding="utf-8", errors="replace") as err_handle:
            api_key = os.environ.get(config.agent.api_key_env)
            if not api_key:
                message = f"missing OpenRouter API key env var: {config.agent.api_key_env}\n"
                _write(err_handle, message, echo=echo, echo_handle=sys.stderr)
                return _result(config, cwd, started, 1, False, None, stdout_path, stderr_path)

            messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            for step in range(1, config.agent.max_steps + 1):
                remaining = _remaining_seconds(started, timeout_seconds)
                if remaining is not None and remaining <= 0:
                    return _result(config, cwd, started, -9, True, "wall", stdout_path, stderr_path)

                request_payload = _request_payload(config, messages)
                _write_json(round_dir / f"openrouter-request-{step:03d}.json", request_payload)

                try:
                    response = _send_chat_completion(
                        config,
                        api_key=api_key,
                        payload=request_payload,
                        timeout_seconds=remaining,
                    )
                except OpenRouterAgentError as exc:
                    _write(err_handle, f"{exc}\n", echo=echo, echo_handle=sys.stderr)
                    return _result(config, cwd, started, 1, False, None, stdout_path, stderr_path)

                _write_json(round_dir / f"openrouter-response-{step:03d}.json", response)
                assistant_text = _assistant_text(response)
                messages.append({"role": "assistant", "content": assistant_text})
                _write(
                    out_handle,
                    f"\n--- openrouter assistant step {step} ---\n{assistant_text}\n",
                    echo=echo,
                    echo_handle=sys.stdout,
                )

                action = parse_action(assistant_text)
                if action is None:
                    observation = (
                        "Observation: response was not valid action JSON. "
                        "Return exactly one JSON object with action shell, read_file, write_file, or finish."
                    )
                    messages.append({"role": "user", "content": observation})
                    _write(out_handle, observation + "\n", echo=echo, echo_handle=sys.stdout)
                    continue

                action_name = str(action.get("action", "")).strip().lower()
                if action_name == "finish":
                    message = str(action.get("message") or "")
                    if SENTINEL not in message:
                        message = f"{message.rstrip()}\n{SENTINEL}".strip()
                    _write(out_handle, message + "\n", echo=echo, echo_handle=sys.stdout)
                    return _result(config, cwd, started, 0, False, None, stdout_path, stderr_path)

                observation = _execute_action(config, cwd, action, remaining)
                messages.append({"role": "user", "content": observation})
                _write(out_handle, observation + "\n", echo=echo, echo_handle=sys.stdout)

            _write(
                out_handle,
                f"Observation: max_steps={config.agent.max_steps} reached.\n{SENTINEL}\n",
                echo=echo,
                echo_handle=sys.stdout,
            )
            return _result(config, cwd, started, 0, False, None, stdout_path, stderr_path)


def parse_action(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    if "```" in text:
        parts = text.split("```")
        candidates.extend(part.removeprefix("json").strip() for part in parts[1::2])

    decoder = JSONDecoder()
    for candidate in candidates:
        parsed = _parse_json_object(candidate, decoder)
        if parsed is not None:
            return parsed
    for index, char in enumerate(text):
        if char != "{":
            continue
        parsed = _parse_json_object(text[index:], decoder)
        if parsed is not None:
            return parsed
    return None


def _parse_json_object(text: str, decoder: JSONDecoder) -> dict[str, Any] | None:
    try:
        parsed, _ = decoder.raw_decode(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) and isinstance(parsed.get("action"), str) else None


def _request_payload(config: RunnerConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.agent.model,
        "messages": messages,
    }
    if config.agent.max_tokens is not None:
        payload["max_tokens"] = config.agent.max_tokens
    return payload


def _send_chat_completion(
    config: RunnerConfig,
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    url = config.agent.api_base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": APP_TITLE,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OpenRouterAgentError(f"OpenRouter HTTP {exc.code}: {_truncate(body, 4000)}") from exc
    except urllib.error.URLError as exc:
        raise OpenRouterAgentError(f"OpenRouter request failed: {exc}") from exc
    except TimeoutError as exc:
        raise OpenRouterAgentError("OpenRouter request timed out") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenRouterAgentError(f"OpenRouter returned non-JSON response: {_truncate(body, 4000)}") from exc
    if not isinstance(parsed, dict):
        raise OpenRouterAgentError("OpenRouter returned an unexpected response shape")
    return parsed


def _assistant_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text") for item in content if isinstance(item, dict)]
        return "".join(part for part in parts if isinstance(part, str))
    return ""


def _execute_action(
    config: RunnerConfig,
    workspace: Path,
    action: dict[str, Any],
    remaining_seconds: float | None,
) -> str:
    action_name = str(action.get("action", "")).strip().lower()
    try:
        if action_name == "shell":
            return _run_shell_action(config, workspace, action, remaining_seconds)
        if action_name == "read_file":
            return _read_file_action(workspace, action)
        if action_name == "write_file":
            return _write_file_action(workspace, action)
    except OpenRouterAgentError as exc:
        return f"Observation: action error: {exc}"
    return "Observation: unknown action. Use shell, read_file, write_file, or finish."


def _run_shell_action(
    config: RunnerConfig,
    workspace: Path,
    action: dict[str, Any],
    remaining_seconds: float | None,
) -> str:
    command = action.get("command")
    if not isinstance(command, str) or not command.strip():
        raise OpenRouterAgentError("shell action requires a non-empty command")
    requested_timeout = action.get("timeout_seconds")
    timeout_seconds = config.agent.command_timeout_seconds
    if isinstance(requested_timeout, int) and requested_timeout > 0:
        timeout_seconds = min(timeout_seconds, requested_timeout)
    if remaining_seconds is not None:
        timeout_seconds = max(min(timeout_seconds, int(remaining_seconds)), 1)

    argv = ["powershell", "-NoProfile", "-Command", command] if os.name == "nt" else ["bash", "-lc", command]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.monotonic() - started
        text = (
            f"Observation: shell returncode={completed.returncode} elapsed={elapsed:.2f}s\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        return _truncate(text, MAX_OBSERVATION_CHARS)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        text = (
            f"Observation: shell timed out after {timeout_seconds}s\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
        return _truncate(text, MAX_OBSERVATION_CHARS)


def _read_file_action(workspace: Path, action: dict[str, Any]) -> str:
    path = _workspace_path(workspace, action.get("path"))
    max_chars = action.get("max_chars")
    if not isinstance(max_chars, int) or max_chars <= 0:
        max_chars = DEFAULT_READ_CHARS
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise OpenRouterAgentError(f"could not read {path.relative_to(workspace)}: {exc}") from exc
    truncated = _truncate(text, max_chars)
    return f"Observation: read_file {path.relative_to(workspace)} ({len(text)} chars)\n{truncated}"


def _write_file_action(workspace: Path, action: dict[str, Any]) -> str:
    path = _workspace_path(workspace, action.get("path"))
    content = action.get("content")
    if not isinstance(content, str):
        raise OpenRouterAgentError("write_file action requires string content")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Observation: wrote {path.relative_to(workspace)} ({len(content)} chars)"


def _workspace_path(workspace: Path, value) -> Path:
    if not isinstance(value, str) or not value:
        raise OpenRouterAgentError("path must be a non-empty string")
    raw_path = Path(value).expanduser()
    path = raw_path if raw_path.is_absolute() else workspace / raw_path
    resolved = path.resolve()
    if not _is_relative_to(resolved, workspace.resolve()):
        raise OpenRouterAgentError("path must stay inside the workspace")
    return resolved


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _remaining_seconds(started: float, timeout_seconds: int | None) -> float | None:
    if timeout_seconds is None:
        return None
    return timeout_seconds - (time.monotonic() - started)


def _result(
    config: RunnerConfig,
    cwd: Path,
    started: float,
    returncode: int,
    timed_out: bool,
    timeout_reason: str | None,
    stdout_path: Path,
    stderr_path: Path,
) -> CommandResult:
    return CommandResult(
        argv=_display_command(config),
        cwd=cwd,
        returncode=returncode,
        elapsed_seconds=time.monotonic() - started,
        timed_out=timed_out,
        timeout_reason=timeout_reason,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _display_command(config: RunnerConfig) -> list[str]:
    return [
        "openrouter-api",
        "--model",
        str(config.agent.model or ""),
        "--api-key-env",
        config.agent.api_key_env,
    ]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write(handle, text: str, *, echo: bool, echo_handle) -> None:
    handle.write(text)
    handle.flush()
    if echo:
        echo_handle.write(text)
        echo_handle.flush()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + f"\n... truncated {len(text) - max_chars + 80} chars ..."
