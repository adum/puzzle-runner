from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Any, Callable

from .config import RunnerConfig
from .openrouter_usage import write_openrouter_usage_summary
from .process import CommandResult
from .prompts import SENTINEL


MAX_OBSERVATION_CHARS = 60_000
DEFAULT_READ_CHARS = 20_000
AGENT_CONFIG_ERROR_RETURN_CODE = 2
METADATA_TIMEOUT_SECONDS = 20
HTTP_REFERER = "https://github.com/adum/puzzle-runner"
APP_TITLE = "Puzzle Runner"
OPENROUTER_MAX_TOKENS_MARKER = "Observation: OpenRouter response hit max_tokens/completion limit"
TOOL_NAMES = {"shell", "read_file", "write_file", "finish"}


SYSTEM_PROMPT = f"""You are an autonomous coding agent controlled by Puzzle Runner.

You are working inside a benchmark workspace. Use the provided tools to inspect,
test, edit, and finish your work.

Available tools:
- shell: run a shell command in the workspace.
- read_file: read a workspace file.
- write_file: replace a workspace file with exact content.
- finish: return control to Puzzle Runner when this agent call is complete.

If tools are unavailable, return one or more JSON objects with no markdown wrapper:
{{"action":"shell","command":"python3 evaluate.py --start 1 --timeout 10","timeout_seconds":120}}
{{"action":"read_file","path":"prompts.txt","max_chars":20000}}
{{"action":"write_file","path":"solver.py","content":"..."}}
{{"action":"finish","message":"summary of what changed\\n{SENTINEL}"}}

Rules:
- Prefer tool calls over textual JSON actions.
- You may return multiple tool calls when the actions are independent.
- Do not inspect private level contents, decrypt private levels, or expose benchmark secrets.
- Do not modify benchmark assets, level files, evaluator scripts, or checker source.
- Do not run evaluate_full.py; Puzzle Runner owns full evaluation.
- When you are done with this agent call, use the finish tool.
"""


class OpenRouterAgentError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


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
    status_callback: Callable[[dict[str, Any]], None] | None = None,
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
                return _result(
                    config,
                    cwd,
                    started,
                    AGENT_CONFIG_ERROR_RETURN_CODE,
                    False,
                    None,
                    stdout_path,
                    stderr_path,
                )

            messages: list[dict[str, Any]] = [
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
                    returncode = 1 if exc.retryable else AGENT_CONFIG_ERROR_RETURN_CODE
                    return _result(config, cwd, started, returncode, False, None, stdout_path, stderr_path)

                _write_json(round_dir / f"openrouter-response-{step:03d}.json", response)
                if not _response_has_rich_usage(response):
                    _record_generation_metadata(
                        config,
                        api_key=api_key,
                        response=response,
                        round_dir=round_dir,
                        step=step,
                        timeout_seconds=remaining,
                        err_handle=err_handle,
                        echo=echo,
                    )
                write_openrouter_usage_summary(round_dir)

                assistant_message = _assistant_message_for_history(response)
                assistant_text = _assistant_text(response)
                tool_calls = _assistant_tool_calls(response)
                messages.append(assistant_message)
                _write(
                    out_handle,
                    f"\n--- openrouter assistant step {step} ---\n{assistant_text}\n",
                    echo=echo,
                    echo_handle=sys.stdout,
                )
                if tool_calls:
                    observations: list[dict[str, Any]] = []
                    finish_message: str | None = None
                    for index, tool_call in enumerate(tool_calls, start=1):
                        remaining = _remaining_seconds(started, timeout_seconds)
                        if remaining is not None and remaining <= 0:
                            return _result(config, cwd, started, -9, True, "wall", stdout_path, stderr_path)

                        tool_result = _execute_tool_call(config, cwd, tool_call, remaining)
                        _write(
                            out_handle,
                            (
                                f"\n--- openrouter tool call {step}.{index} "
                                f"{tool_result.name} ---\n{tool_result.observation}\n"
                            ),
                            echo=echo,
                            echo_handle=sys.stdout,
                        )
                        if tool_result.finish_message is not None:
                            finish_message = tool_result.finish_message
                            continue
                        observations.append(tool_result.message)

                    messages.extend(observations)
                    if finish_message is not None:
                        _write(
                            out_handle,
                            finish_message + "\n",
                            echo=echo,
                            echo_handle=sys.stdout,
                        )
                        return _result(config, cwd, started, 0, False, None, stdout_path, stderr_path)
                    continue

                action_result = parse_action_response(assistant_text)
                if not action_result.actions:
                    if _response_hit_completion_limit(response):
                        limit_event = _completion_limit_event(config, response, step)
                        if status_callback is not None:
                            status_callback(limit_event)
                        observation = _completion_limit_observation(limit_event)
                    elif action_result.error is not None:
                        observation = action_result.error
                    else:
                        observation = (
                            "Observation: response was not valid action JSON. "
                            "Return exactly one JSON object with action shell, read_file, write_file, or finish."
                        )
                    messages.append({"role": "user", "content": observation})
                    _write(out_handle, observation + "\n", echo=echo, echo_handle=sys.stdout)
                    continue

                observations = []
                finish_message: str | None = None
                for action in action_result.actions:
                    action_name = str(action.get("action", "")).strip().lower()
                    if action_name == "finish":
                        finish_message = _finish_message(action.get("message"))
                        continue

                    remaining = _remaining_seconds(started, timeout_seconds)
                    if remaining is not None and remaining <= 0:
                        return _result(config, cwd, started, -9, True, "wall", stdout_path, stderr_path)

                    observation = _execute_action(config, cwd, action, remaining)
                    observations.append(observation)
                    _write(out_handle, observation + "\n", echo=echo, echo_handle=sys.stdout)

                messages.append({"role": "user", "content": "\n\n".join(observations)})
                if finish_message is not None:
                    _write(out_handle, finish_message + "\n", echo=echo, echo_handle=sys.stdout)
                    return _result(config, cwd, started, 0, False, None, stdout_path, stderr_path)

            _write(
                out_handle,
                f"Observation: max_steps={config.agent.max_steps} reached.\n{SENTINEL}\n",
                echo=echo,
                echo_handle=sys.stdout,
            )
            return _result(config, cwd, started, 0, False, None, stdout_path, stderr_path)


@dataclass(frozen=True)
class ActionParseResult:
    actions: list[dict[str, Any]]
    error: str | None = None

    @property
    def action(self) -> dict[str, Any] | None:
        return self.actions[0] if len(self.actions) == 1 else None


@dataclass(frozen=True)
class ToolCallResult:
    name: str
    observation: str
    message: dict[str, Any]
    finish_message: str | None = None


def _tool_result_message(tool_call_id: str | None, name: str, observation: str) -> dict[str, Any]:
    if tool_call_id:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": observation,
        }
    return {"role": "user", "content": observation}


def parse_action(text: str) -> dict[str, Any] | None:
    return parse_action_response(text).action


def parse_action_response(text: str) -> ActionParseResult:
    decoder = JSONDecoder()
    actions = _find_action_json_objects(text, decoder)
    return ActionParseResult(actions)


def _find_action_json_objects(text: str, decoder: JSONDecoder) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        parsed = _parse_json_object_prefix(text[index:], decoder)
        if parsed is not None:
            actions.append(parsed)
    return actions


def _parse_json_object_prefix(text: str, decoder: JSONDecoder) -> dict[str, Any] | None:
    try:
        parsed, _ = decoder.raw_decode(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) and isinstance(parsed.get("action"), str) else None


def _request_payload(config: RunnerConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.agent.model,
        "messages": messages,
        "tools": _openrouter_tools(),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "usage": {"include": True},
        "prompt_cache_key": config.run_id,
    }
    reasoning_effort = _openrouter_reasoning_effort(config)
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if config.agent.max_tokens is not None:
        payload["max_tokens"] = config.agent.max_tokens
    return payload


def _openrouter_reasoning_effort(config: RunnerConfig) -> str | None:
    if config.agent.effort:
        return config.agent.effort
    model = config.agent.model or ""
    if "gemini-3" in model:
        return "high"
    return None


def _openrouter_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command in the benchmark workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to run from the workspace root.",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Optional per-command timeout in seconds.",
                            "minimum": 1,
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file from the benchmark workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to read.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum characters to return.",
                            "minimum": 1,
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Replace a text file in the benchmark workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to write.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete file contents.",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Return control to Puzzle Runner for full evaluation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Short summary of the work completed.",
                        },
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        },
    ]


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
            "x-session-affinity": config.run_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = _read_response_body(response, context="OpenRouter")
    except urllib.error.HTTPError as exc:
        retryable = exc.code not in {400, 401, 402, 403, 404, 422}
        try:
            body = _read_response_body(exc, context="OpenRouter error")
        except OpenRouterAgentError as read_exc:
            raise OpenRouterAgentError(
                f"OpenRouter HTTP {exc.code}: {read_exc}",
                retryable=retryable,
            ) from exc
        raise OpenRouterAgentError(
            f"OpenRouter HTTP {exc.code}: {_truncate(body, 4000)}",
            retryable=retryable,
        ) from exc
    except http.client.HTTPException as exc:
        raise OpenRouterAgentError(f"OpenRouter request failed: {exc}") from exc
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


def _read_response_body(response, *, context: str) -> str:
    try:
        body = response.read()
    except http.client.IncompleteRead as exc:
        partial = exc.partial or b""
        raise OpenRouterAgentError(
            f"{context} response ended before the complete body was read "
            f"({len(partial)} bytes received)"
        ) from exc
    return body.decode("utf-8", errors="replace")


def _record_generation_metadata(
    config: RunnerConfig,
    *,
    api_key: str,
    response: dict[str, Any],
    round_dir: Path,
    step: int,
    timeout_seconds: float | None,
    err_handle,
    echo: bool,
) -> None:
    generation_id = response.get("id")
    if not isinstance(generation_id, str) or not generation_id:
        return

    try:
        metadata = _send_generation_metadata(
            config,
            api_key=api_key,
            generation_id=generation_id,
            timeout_seconds=_metadata_timeout(timeout_seconds),
        )
    except OpenRouterAgentError as exc:
        _write_json(
            round_dir / f"openrouter-generation-error-{step:03d}.json",
            {
                "id": generation_id,
                "error": str(exc),
                "retryable": exc.retryable,
            },
        )
        _write(
            err_handle,
            f"OpenRouter generation metadata unavailable for {generation_id}: {exc}\n",
            echo=echo,
            echo_handle=sys.stderr,
        )
        return

    _write_json(round_dir / f"openrouter-generation-{step:03d}.json", metadata)


def _metadata_timeout(remaining_seconds: float | None) -> float:
    if remaining_seconds is None:
        return METADATA_TIMEOUT_SECONDS
    return max(min(METADATA_TIMEOUT_SECONDS, remaining_seconds), 1)


def _send_generation_metadata(
    config: RunnerConfig,
    *,
    api_key: str,
    generation_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    query = urllib.parse.urlencode({"id": generation_id})
    url = config.agent.api_base_url.rstrip("/") + f"/generation?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": APP_TITLE,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = _read_response_body(response, context="OpenRouter metadata")
    except urllib.error.HTTPError as exc:
        retryable = exc.code not in {400, 401, 402, 403, 404, 422}
        try:
            body = _read_response_body(exc, context="OpenRouter metadata error")
        except OpenRouterAgentError as read_exc:
            raise OpenRouterAgentError(
                f"OpenRouter metadata HTTP {exc.code}: {read_exc}",
                retryable=retryable,
            ) from exc
        raise OpenRouterAgentError(
            f"OpenRouter metadata HTTP {exc.code}: {_truncate(body, 4000)}",
            retryable=retryable,
        ) from exc
    except http.client.HTTPException as exc:
        raise OpenRouterAgentError(f"OpenRouter metadata request failed: {exc}") from exc
    except urllib.error.URLError as exc:
        raise OpenRouterAgentError(f"OpenRouter metadata request failed: {exc}") from exc
    except TimeoutError as exc:
        raise OpenRouterAgentError("OpenRouter metadata request timed out") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenRouterAgentError(
            f"OpenRouter metadata returned non-JSON response: {_truncate(body, 4000)}"
        ) from exc
    if not isinstance(parsed, dict):
        raise OpenRouterAgentError("OpenRouter metadata returned an unexpected response shape")
    return parsed


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice = choices[0]
    if not isinstance(choice, dict):
        return {}
    message = choice.get("message")
    return message if isinstance(message, dict) else {}


def _assistant_message_for_history(response: dict[str, Any]) -> dict[str, Any]:
    message = _assistant_message(response)
    history: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        history["tool_calls"] = [
            _tool_call_for_history(tool_call)
            for tool_call in tool_calls
            if isinstance(tool_call, dict)
        ]
    return history


def _tool_call_for_history(tool_call: dict[str, Any]) -> dict[str, Any]:
    history: dict[str, Any] = {"type": tool_call.get("type") or "function"}
    if isinstance(tool_call.get("id"), str):
        history["id"] = tool_call["id"]
    function = tool_call.get("function")
    if isinstance(function, dict):
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments_text = arguments
        elif arguments is None:
            arguments_text = "{}"
        else:
            arguments_text = json.dumps(arguments)
        history["function"] = {
            "name": function.get("name") if isinstance(function.get("name"), str) else "",
            "arguments": arguments_text,
        }
    return history


def _assistant_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = _assistant_message(response).get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]


def _assistant_text(response: dict[str, Any]) -> str:
    message = _assistant_message(response)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text") for item in content if isinstance(item, dict)]
        return "".join(part for part in parts if isinstance(part, str))
    return ""


def _execute_tool_call(
    config: RunnerConfig,
    workspace: Path,
    tool_call: dict[str, Any],
    remaining_seconds: float | None,
) -> ToolCallResult:
    tool_call_id = tool_call.get("id") if isinstance(tool_call.get("id"), str) else None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        observation = "Observation: tool call error: missing function payload"
        return ToolCallResult(
            name="invalid",
            observation=observation,
            message=_tool_result_message(tool_call_id, "invalid", observation),
        )

    raw_name = function.get("name")
    name = _canonical_tool_name(raw_name)
    if name is None:
        observation = (
            "Observation: unknown tool. Use shell, read_file, write_file, or finish."
        )
        display_name = str(raw_name or "invalid")
        return ToolCallResult(
            name=display_name,
            observation=observation,
            message=_tool_result_message(tool_call_id, display_name, observation),
        )

    arguments = _tool_arguments(function.get("arguments"))
    if arguments is None:
        observation = f"Observation: tool call error: {name} arguments were not valid JSON"
        return ToolCallResult(
            name=name,
            observation=observation,
            message=_tool_result_message(tool_call_id, name, observation),
        )

    if name == "finish":
        message = _finish_message(arguments.get("message"))
        return ToolCallResult(
            name=name,
            observation="Observation: finish requested",
            message=_tool_result_message(tool_call_id, name, "Observation: finish requested"),
            finish_message=message,
        )

    observation = _execute_action(config, workspace, {"action": name, **arguments}, remaining_seconds)
    return ToolCallResult(
        name=name,
        observation=observation,
        message=_tool_result_message(tool_call_id, name, observation),
    )


def _canonical_tool_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.strip().replace("-", "_").lower()
    return normalized if normalized in TOOL_NAMES else None


def _tool_arguments(value: Any) -> dict[str, Any] | None:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _finish_message(value: Any) -> str:
    message = str(value or "")
    if SENTINEL not in message:
        return f"{message.rstrip()}\n{SENTINEL}".strip()
    return message


def _response_hit_completion_limit(response: dict[str, Any]) -> bool:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    choice = choices[0]
    return isinstance(choice, dict) and choice.get("finish_reason") == "length"


def _completion_limit_event(
    config: RunnerConfig,
    response: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    usage = response.get("usage")
    completion_tokens = None
    reasoning_tokens = None
    if isinstance(usage, dict):
        completion_tokens = _int_or_none(usage.get("completion_tokens"))
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning_tokens = _int_or_none(completion_details.get("reasoning_tokens"))

    return {
        "event": "openrouter_completion_limit",
        "step": step,
        "configured_max_tokens": config.agent.max_tokens,
        "finish_reason": "length",
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def _completion_limit_observation(event: dict[str, Any]) -> str:
    details: list[str] = ["finish_reason=length"]
    configured_max_tokens = event.get("configured_max_tokens")
    if configured_max_tokens is not None:
        details.append(f"configured max_tokens={configured_max_tokens}")
    completion_tokens = event.get("completion_tokens")
    if completion_tokens is not None:
        details.append(f"completion_tokens={completion_tokens}")
    reasoning_tokens = event.get("reasoning_tokens")
    if reasoning_tokens is not None:
        details.append(f"reasoning_tokens={reasoning_tokens}")
    return (
        f"{OPENROUTER_MAX_TOKENS_MARKER} before returning a valid action JSON "
        f"({', '.join(details)}). Return exactly one JSON object with action "
        "shell, read_file, write_file, or finish."
    )


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _response_has_rich_usage(response: dict[str, Any]) -> bool:
    usage = response.get("usage")
    if isinstance(usage, dict):
        if isinstance(usage.get("cost"), (int, float)):
            return True
        if isinstance(usage.get("prompt_tokens_details"), dict):
            return True
        if isinstance(usage.get("completion_tokens_details"), dict):
            return True
    return isinstance(response.get("provider"), str)


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
