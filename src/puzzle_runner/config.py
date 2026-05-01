from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback for this project's small config subset.
    tomllib = None


class ConfigError(ValueError):
    pass


PromptMode = Literal["stdin", "arg"]
WorkspaceMode = Literal["worktree", "copy"]


@dataclasses.dataclass(frozen=True)
class AgentConfig:
    name: str
    backend: str
    command: list[str]
    prompt_mode: PromptMode = "stdin"
    effort: str | None = None


@dataclasses.dataclass(frozen=True)
class RunnerConfig:
    config_path: Path
    run_id: str
    benchmark_path: Path | None
    benchmark_repo_url: str
    benchmark_ref: str | None
    download_full_levels: bool
    workspace_mode: WorkspaceMode
    worktree_root: Path
    log_root: Path
    status_dir: Path
    results_path: Path
    solver_wrapper: str
    evaluation_script: str
    evaluation_timeout_seconds: int
    evaluation_process_timeout_seconds: int
    full_eval_password_env: str
    generate_full_eval_password: bool
    stale_limit: int
    max_rounds: int
    agent_timeout_seconds: int
    agent_idle_timeout_seconds: int
    agent_failure_retry_limit_seconds: int
    build_checker: bool
    echo_agent_output: bool
    echo_evaluation_output: bool
    forbidden_paths: list[str]
    agent: AgentConfig


def load_config(path: str, *, run_id: str | None = None) -> RunnerConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"missing config file: {config_path}")

    raw = _load_toml(config_path)

    base_dir = config_path.parent
    agent_raw = _table(raw, "agent")
    resolved_run_id = run_id or raw.get("run_id") or _default_run_id(agent_raw.get("name", "run"))
    log_root = _path(base_dir, raw, "log_root")
    status_dir = _optional_path(base_dir, raw, "status_dir") or (log_root.parent / "current").resolve()

    return RunnerConfig(
        config_path=config_path,
        run_id=str(resolved_run_id),
        benchmark_path=_optional_path(base_dir, raw, "benchmark_path"),
        benchmark_repo_url=_str(raw, "benchmark_repo_url", "https://github.com/adum/coilbench.git"),
        benchmark_ref=_optional_str(raw, "benchmark_ref", "main"),
        download_full_levels=bool(raw.get("download_full_levels", True)),
        workspace_mode=_literal(raw.get("workspace_mode", "worktree"), {"worktree", "copy"}, "workspace_mode"),
        worktree_root=_path(base_dir, raw, "worktree_root"),
        log_root=log_root,
        status_dir=status_dir,
        results_path=_path(base_dir, raw, "results_path"),
        solver_wrapper=_str(raw, "solver_wrapper", "run_solver"),
        evaluation_script=_str(raw, "evaluation_script", "evaluate_full.py"),
        evaluation_timeout_seconds=_positive_int(raw, "evaluation_timeout_seconds", 600),
        evaluation_process_timeout_seconds=_non_negative_int(raw, "evaluation_process_timeout_seconds", 0),
        full_eval_password_env=_str(raw, "full_eval_password_env", "COIL_FULL_PASSWORD"),
        generate_full_eval_password=bool(raw.get("generate_full_eval_password", True)),
        stale_limit=_positive_int(raw, "stale_limit", 3),
        max_rounds=_positive_int(raw, "max_rounds", 20),
        agent_timeout_seconds=_positive_int(raw, "agent_timeout_seconds", 86400),
        agent_idle_timeout_seconds=_non_negative_int(raw, "agent_idle_timeout_seconds", 1800),
        agent_failure_retry_limit_seconds=_non_negative_int(raw, "agent_failure_retry_limit_seconds", 900),
        build_checker=bool(raw.get("build_checker", True)),
        echo_agent_output=bool(raw.get("echo_agent_output", True)),
        echo_evaluation_output=bool(raw.get("echo_evaluation_output", True)),
        forbidden_paths=_str_list(raw.get("forbidden_paths", []), "forbidden_paths"),
        agent=AgentConfig(
            name=_str(agent_raw, "name", "codex-5.3-spark"),
            backend=_str(agent_raw, "backend", "codex"),
            command=_str_list(agent_raw.get("command"), "agent.command"),
            prompt_mode=_literal(agent_raw.get("prompt_mode", "stdin"), {"stdin", "arg"}, "agent.prompt_mode"),
            effort=_optional_str(agent_raw, "effort"),
        ),
    )


def _default_run_id(agent_name: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in agent_name)
    return f"{stamp}-{safe_name}"


def _load_toml(path: Path) -> dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    return _load_minimal_toml(path.read_text(encoding="utf-8"))


def _load_minimal_toml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current = data
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        index += 1

        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            table_name = line[1:-1].strip()
            if not table_name:
                raise ConfigError("empty TOML table name")
            current = data.setdefault(table_name, {})
            if not isinstance(current, dict):
                raise ConfigError(f"TOML table conflicts with value: {table_name}")
            continue

        if "=" not in line:
            raise ConfigError(f"unsupported TOML line: {raw_line}")

        key, value_text = [part.strip() for part in line.split("=", 1)]
        if value_text == "[":
            items: list[str] = []
            while index < len(lines):
                item_line = lines[index].strip()
                index += 1
                if not item_line or item_line.startswith("#"):
                    continue
                if item_line == "]":
                    break
                if item_line.endswith(","):
                    item_line = item_line[:-1].strip()
                items.append(_parse_string(item_line))
            else:
                raise ConfigError(f"unterminated TOML array for {key}")
            current[key] = items
            continue

        current[key] = _parse_scalar(value_text)

    return data


def _parse_scalar(value_text: str) -> Any:
    value_text = value_text.strip()
    if value_text.startswith("[") and value_text.endswith("]"):
        inner = value_text[1:-1].strip()
        if not inner:
            return []
        return [_parse_string(item.strip()) for item in inner.split(",") if item.strip()]
    if value_text.startswith('"'):
        return _parse_string(value_text)
    if value_text == "true":
        return True
    if value_text == "false":
        return False
    try:
        return int(value_text)
    except ValueError as exc:
        raise ConfigError(f"unsupported TOML value: {value_text}") from exc


def _parse_string(value_text: str) -> str:
    if not (value_text.startswith('"') and value_text.endswith('"')):
        raise ConfigError(f"expected quoted TOML string: {value_text}")
    return bytes(value_text[1:-1], "utf-8").decode("unicode_escape")


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{key}] table")
    return value


def _path(base_dir: Path, raw: dict[str, Any], key: str) -> Path:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"missing string config value: {key}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _optional_path(base_dir: Path, raw: dict[str, Any], key: str) -> Path | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string when set")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _str(raw: dict[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _optional_str(raw: dict[str, Any], key: str, default: str | None = None) -> str | None:
    value = raw.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    value = value.strip()
    return value or None


def _positive_int(raw: dict[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer")
    return value


def _non_negative_int(raw: dict[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or value < 0:
        raise ConfigError(f"{key} must be a non-negative integer")
    return value


def _str_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{key} must be a non-empty list of strings")
    if not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{key} must be a non-empty list of strings")
    return list(value)


def _literal(value: Any, allowed: set[str], key: str) -> Any:
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ConfigError(f"{key} must be one of: {allowed_text}")
    return value
