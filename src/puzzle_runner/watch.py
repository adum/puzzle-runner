from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
CLEAR_SCREEN = "\033[2J\033[H"
CLEAR_LINE = "\033[2K"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
EVALUATION_LEVEL_RE = re.compile(
    r"^Level\s+(\d+)\s+\(([^)]*)\):[ \t]*(PASS|FAIL|TIMEOUT|ERROR)?",
    re.MULTILINE,
)
TESTED_LEVEL_LINE_RE = re.compile(
    r"^Level\s+\d+\s+\([^)]*\):\s+(?:PASS|FAIL|TIMEOUT|ERROR)\b.*$",
    re.MULTILINE,
)
NOISY_WORKSPACE_EXACT_PATHS = {
    "coil_check/check",
    "levels_secret_even.tar.enc",
}
NOISY_WORKSPACE_PREFIXES = (
    ".git/",
    "__pycache__/",
    "levels_public/",
    "levels_secret_even/",
)
NOISY_WORKSPACE_SUFFIXES = (
    ".pyc",
    ".pyo",
)
MAX_UNTRACKED_LINE_COUNT_BYTES = 2_000_000
AGENT_TESTED_LINE_TAIL_BYTES = 256_000


@dataclasses.dataclass(frozen=True)
class AgentOutputStats:
    chars: int
    last_output_at: datetime | None
    chars_per_minute: float | None


@dataclasses.dataclass
class FileTypeChange:
    files: int = 0
    additions: int = 0
    deletions: int = 0


class WorkspaceChangeCache:
    def __init__(self, refresh_interval: float) -> None:
        self.refresh_interval = refresh_interval
        self._workspace: Path | None = None
        self._summary: str | None = None
        self._refreshed_at = 0.0

    def get(self, status: dict[str, Any]) -> str | None:
        if self.refresh_interval <= 0:
            return None

        workspace_value = status.get("workspace")
        if not workspace_value:
            return None

        workspace = Path(str(workspace_value))
        now = time.monotonic()
        if (
            self._workspace == workspace
            and self._summary is not None
            and now - self._refreshed_at < self.refresh_interval
        ):
            return self._summary

        self._workspace = workspace
        self._summary = _workspace_change_summary(workspace)
        self._refreshed_at = now
        return self._summary


def add_watch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="runner.toml",
        help="Config path used to locate status_dir. Defaults to runner.toml.",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="Explicit status.json path. Overrides --config.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Refresh interval in seconds. Defaults to 1.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render once and exit.",
    )
    parser.add_argument(
        "--changes-interval",
        type=float,
        default=15.0,
        help="Seconds between workspace change summary refreshes. Use 0 to disable. Defaults to 15.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch Puzzle Runner live status.")
    add_watch_arguments(parser)
    return run_watch(parser.parse_args(argv))


def run_watch(args: argparse.Namespace) -> int:
    status_path = resolve_status_path(args.status, args.config)
    use_ansi = sys.stdout.isatty()
    color = (not args.no_color) and use_ansi and os.environ.get("NO_COLOR") is None
    change_cache = WorkspaceChangeCache(args.changes_interval)

    try:
        if args.once:
            status = load_status(status_path)
            print(
                render_status(
                    status,
                    status_path=status_path,
                    color=color,
                    workspace_changes=change_cache.get(status),
                )
            )
            return 0

        previous_lines: list[str] = []
        first_frame = True
        sys.stdout.write(HIDE_CURSOR if use_ansi else "")
        sys.stdout.flush()
        while True:
            status = load_status(status_path)
            rendered = render_status(
                status,
                status_path=status_path,
                color=color,
                workspace_changes=change_cache.get(status),
            )
            if use_ansi:
                previous_lines = _draw_frame(rendered, previous_lines, first_frame=first_frame)
                first_frame = False
            else:
                sys.stdout.write(rendered + "\n")
            sys.stdout.flush()
            time.sleep(max(args.interval, 0.1))
    except KeyboardInterrupt:
        return 130
    finally:
        if use_ansi:
            sys.stdout.write(SHOW_CURSOR + RESET + "\n")
            sys.stdout.flush()


def resolve_status_path(status_arg: str | None, config_arg: str) -> Path:
    if status_arg:
        return Path(status_arg).expanduser().resolve()

    config_path = Path(config_arg).expanduser()
    if config_path.exists():
        try:
            config = load_config(str(config_path), run_id="watch")
            return (config.status_dir / "status.json").resolve()
        except ConfigError:
            pass

    return Path(".puzzle-runs/current/status.json").resolve()


def load_status(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "active": False,
            "phase": "waiting_for_status",
            "status_json": path,
            "message": f"Waiting for {path}",
        }
    except json.JSONDecodeError as exc:
        return {
            "active": False,
            "phase": "status_read_error",
            "status_json": path,
            "message": f"Could not parse {path}: {exc}",
        }


def render_status(
    status: dict[str, Any],
    *,
    status_path: Path,
    color: bool = True,
    workspace_changes: str | None = None,
) -> str:
    width = shutil.get_terminal_size((100, 30)).columns
    latest = status.get("latest") or {}
    phase = str(status.get("phase") or "unknown")
    active = bool(status.get("active"))
    stop_reason = status.get("stop_reason")
    current_round = status.get("current_round")
    max_rounds = status.get("max_rounds")
    stale_count = _int(status.get("stale_count"))
    stale_limit = _int(status.get("stale_limit"))
    remaining = status.get("remaining_no_progress_tries")
    phase_elapsed = _duration_since(status.get("phase_started_at"))
    agent_elapsed = _agent_elapsed(status, phase)

    title = _c("Puzzle Runner", BOLD + CYAN, color)
    state = _state_label(active, phase, stop_reason, color)
    lines = [
        f"{title}  {state}",
        _rule(width, color),
        _kv("Run", status.get("run_id"), color, width),
        _kv("Agent", status.get("agent"), color, width),
        _kv("Phase", phase, color, width),
        _kv("Round", f"{current_round}/{max_rounds}", color, width),
        _kv("Updated", status.get("updated_at"), color, width),
        _kv("Elapsed", _duration(status.get("elapsed_seconds")), color, width),
        _kv("Phase time", phase_elapsed, color, width),
    ]
    if phase == "agent_running":
        lines.append(_kv("Agent running", agent_elapsed, color, width))
    elif status.get("last_agent_elapsed_seconds") is not None:
        lines.append(_kv("Last agent run", _duration(status.get("last_agent_elapsed_seconds")), color, width))
    if phase in {"agent_running", "agent_retry_wait"} and status.get("agent_attempt") is not None:
        lines.append(_kv("Agent attempt", status.get("agent_attempt"), color, width))
    if phase == "agent_retry_wait":
        lines.append(_kv("Retrying in", _retry_countdown(status), color, width))
    if phase == "evaluation_running":
        evaluation_progress = _evaluation_progress(latest)
        if evaluation_progress is not None:
            lines.append(_kv("Evaluating", evaluation_progress, color, width))
    agent_output = _agent_output_stats(latest, status, phase)
    if agent_output is not None:
        lines.append(_kv("Agent output", _agent_output_summary(agent_output), color, width))
        lines.append(_kv("Last output", _last_output_age(agent_output), color, width))
    last_tested = _last_tested_puzzle(latest)
    if last_tested is not None:
        lines.append(_kv("Last tested", last_tested, color, width))

    lines.extend(
        [
            "",
            _section("Score", color),
            _kv("Best", status.get("best_score"), color, width),
            _kv("Best round", status.get("best_round"), color, width),
            _kv("Scores", _score_history(status), color, width),
            _kv("Last eval", _last_eval_summary(status), color, width),
            _kv("Improved", _yes_no(status.get("last_improved")), color, width),
            _kv("No-progress", f"{stale_count}/{stale_limit} {_bar(stale_count, stale_limit, color)}", color, width),
            _kv("Remaining tries", remaining, color, width),
            _kv("Stop reason", stop_reason, color, width),
        ]
    )
    if status.get("stop_detail"):
        lines.append(_kv("Stop detail", status.get("stop_detail"), color, width))

    if workspace_changes is not None:
        lines.extend(["", _section("Workspace", color), _kv("Changes", workspace_changes, color, width)])

    command = status.get("current_command")
    if command:
        lines.extend(["", _section("Current Command", color), _wrap_command(command, width)])

    message = status.get("message")
    if message:
        lines.extend(["", _c(str(message), YELLOW, color)])

    lines.append("")
    lines.append(_c("Ctrl-C to exit watcher. The run keeps going.", DIM, color))
    return "\n".join(_trim(line, width) for line in lines) + "\n"


def _draw_frame(rendered: str, previous_lines: list[str], *, first_frame: bool) -> list[str]:
    lines = rendered.rstrip("\n").split("\n")
    if first_frame:
        sys.stdout.write(CLEAR_SCREEN + rendered)
        return lines

    max_lines = max(len(lines), len(previous_lines))
    for index in range(max_lines):
        old = previous_lines[index] if index < len(previous_lines) else None
        new = lines[index] if index < len(lines) else ""
        if new == old:
            continue
        sys.stdout.write(f"\033[{index + 1};1H{CLEAR_LINE}{new}")

    sys.stdout.write(f"\033[{len(lines) + 1};1H{CLEAR_LINE}")
    return lines


def _state_label(active: bool, phase: str, stop_reason, color: bool) -> str:
    if active:
        return _c("[ACTIVE]", GREEN + BOLD, color)
    if phase in {"waiting_for_status", "status_read_error"}:
        return _c("[WAITING]", YELLOW + BOLD, color)
    if stop_reason in {None, "max_rounds", "stale_limit"}:
        return _c("[FINISHED]", BLUE + BOLD, color)
    return _c("[STOPPED]", RED + BOLD, color)


def _section(label: str, color: bool) -> str:
    return _c(label, MAGENTA + BOLD, color)


def _kv(label: str, value, color: bool, width: int) -> str:
    value_text = "-" if value is None else str(value)
    label_text = (label + ":").ljust(18)
    plain_prefix_len = len(label_text) + 1
    value_text = _shorten_middle(value_text, max(width - plain_prefix_len, 16))
    return f"{_c(label_text, WHITE + BOLD, color)} {value_text}"


def _rule(width: int, color: bool) -> str:
    return _c("-" * min(max(width, 20), 120), DIM, color)


def _bar(value: int, total: int, color: bool) -> str:
    if total <= 0:
        return ""
    size = 12
    filled = max(0, min(size, round(size * value / total)))
    bar = "#" * filled + "-" * (size - filled)
    style = GREEN if value == 0 else YELLOW if value < total else RED
    return _c(f"[{bar}]", style, color)


def _duration(value) -> str:
    seconds = _float(value)
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _duration_since(value) -> str:
    instant = _parse_timestamp(value)
    if instant is None:
        return "-"
    elapsed = max((datetime.now(timezone.utc) - instant).total_seconds(), 0)
    return _duration(elapsed)


def _agent_elapsed(status: dict[str, Any], phase: str) -> str:
    timestamp = status.get("agent_started_at")
    if timestamp is None and phase == "agent_running":
        timestamp = status.get("phase_started_at") or status.get("updated_at")
    return _duration_since(timestamp)


def _agent_output_stats(
    latest: dict[str, Any],
    status: dict[str, Any],
    phase: str,
) -> AgentOutputStats | None:
    paths = [latest.get("agent_stdout"), latest.get("agent_stderr")]
    if not any(paths):
        return None

    total = 0
    last_output_at: datetime | None = None
    saw_file = False
    for path_value in paths:
        if not path_value:
            continue
        try:
            stat_result = Path(str(path_value)).stat()
            total += stat_result.st_size
            saw_file = True
            if stat_result.st_size > 0:
                modified_at = datetime.fromtimestamp(stat_result.st_mtime, timezone.utc)
                if last_output_at is None or modified_at > last_output_at:
                    last_output_at = modified_at
        except OSError:
            continue
    if not saw_file:
        return None

    elapsed = _agent_output_elapsed_seconds(status, phase)
    chars_per_minute = None
    if elapsed is not None and elapsed > 0:
        chars_per_minute = total / elapsed * 60
    return AgentOutputStats(total, last_output_at, chars_per_minute)


def _agent_output_elapsed_seconds(status: dict[str, Any], phase: str) -> float | None:
    if phase in {"agent_running", "agent_retry_wait"}:
        started = _parse_timestamp(status.get("agent_started_at"))
        if started is None:
            return None
        return max((datetime.now(timezone.utc) - started).total_seconds(), 0)
    return _float(status.get("last_agent_elapsed_seconds"))


def _agent_output_summary(stats: AgentOutputStats) -> str:
    rate = _format_rate(stats.chars_per_minute)
    if rate is None:
        return f"{stats.chars:,} chars"
    return f"{stats.chars:,} chars ({rate})"


def _format_rate(chars_per_minute: float | None) -> str | None:
    if chars_per_minute is None:
        return None
    if chars_per_minute < 10:
        return f"{chars_per_minute:.1f}/min"
    return f"{chars_per_minute:,.0f}/min"


def _last_output_age(stats: AgentOutputStats) -> str:
    if stats.last_output_at is None:
        return "none yet"
    elapsed = max((datetime.now(timezone.utc) - stats.last_output_at).total_seconds(), 0)
    return f"{_duration(elapsed)} ago"


def _last_tested_puzzle(latest: dict[str, Any]) -> str | None:
    candidates: list[tuple[float, str]] = []
    for key in ("agent_stdout", "agent_stderr"):
        path_value = latest.get(key)
        if not path_value:
            continue
        path = Path(str(path_value))
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue

        line = _last_tested_line_in_file(path)
        if line is not None:
            candidates.append((modified, line))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _last_tested_line_in_file(path: Path) -> str | None:
    text = _tail_text(path, AGENT_TESTED_LINE_TAIL_BYTES)
    if text is None:
        return None

    matches = list(TESTED_LEVEL_LINE_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(0).strip()


def _tail_text(path: Path, max_bytes: int) -> str | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0), os.SEEK_SET)
            data = handle.read()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def _workspace_change_summary(workspace: Path) -> str | None:
    if not workspace.exists():
        return None

    changed_paths: set[str] = set()
    totals: dict[str, FileTypeChange] = {}

    for path, additions, deletions in _tracked_workspace_changes(workspace):
        if _ignore_workspace_path(path):
            continue
        changed_paths.add(path)
        _add_file_type_change(totals, _file_type(path, workspace / path), additions, deletions)

    for path in _untracked_workspace_paths(workspace):
        if _ignore_workspace_path(path):
            continue
        changed_paths.add(path)
        additions = _count_text_lines(workspace / path)
        _add_file_type_change(totals, _file_type(path, workspace / path), additions or 0, 0)

    if not changed_paths:
        return "none"

    total_additions = sum(item.additions for item in totals.values())
    total_deletions = sum(item.deletions for item in totals.values())
    summary = f"{len(changed_paths)} {_plural('file', len(changed_paths))}, {_change_counts(total_additions, total_deletions)}"

    breakdown = _file_type_breakdown(totals)
    if breakdown:
        summary = f"{summary} | {breakdown}"
    return summary


def _tracked_workspace_changes(workspace: Path) -> list[tuple[str, int, int]]:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--numstat", "--"],
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

    changes: list[tuple[str, int, int]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        additions_text, deletions_text, path = parts[0], parts[1], parts[2]
        additions = int(additions_text) if additions_text.isdigit() else 0
        deletions = int(deletions_text) if deletions_text.isdigit() else 0
        changes.append((_normalize_git_path(path), additions, deletions))
    return changes


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


def _ignore_workspace_path(path: str) -> bool:
    normalized = _normalize_git_path(path)
    if normalized in NOISY_WORKSPACE_EXACT_PATHS:
        return True
    if normalized.endswith(NOISY_WORKSPACE_SUFFIXES):
        return True
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in NOISY_WORKSPACE_PREFIXES
    )


def _add_file_type_change(
    totals: dict[str, FileTypeChange],
    file_type: str,
    additions: int,
    deletions: int,
) -> None:
    item = totals.setdefault(file_type, FileTypeChange())
    item.files += 1
    item.additions += additions
    item.deletions += deletions


def _file_type(path: str, full_path: Path) -> str:
    name = Path(path).name
    suffix = Path(path).suffix.lower()

    if name in {"Makefile", "makefile"} or suffix == ".mk":
        return "Make"
    if suffix == ".py" or _shebang_mentions(full_path, "python"):
        return "Python"
    if suffix in {".c", ".h"}:
        return "C"
    if suffix in {".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"}:
        return "C++"
    if suffix in {".sh", ".bash", ".zsh", ".fish"} or _shebang_mentions(full_path, "sh"):
        return "Shell"
    if suffix == ".md":
        return "Markdown"
    if suffix == ".toml":
        return "TOML"
    if suffix == ".json":
        return "JSON"
    if suffix in {".txt", ".text"}:
        return "Text"
    if suffix in {".js", ".jsx"}:
        return "JavaScript"
    if suffix in {".ts", ".tsx"}:
        return "TypeScript"
    if suffix == ".rs":
        return "Rust"
    if suffix == ".go":
        return "Go"
    return "Other"


def _shebang_mentions(path: Path, needle: str) -> bool:
    try:
        with path.open("rb") as handle:
            first_line = handle.readline(200).decode("utf-8", errors="ignore").lower()
    except OSError:
        return False
    return first_line.startswith("#!") and needle in first_line


def _count_text_lines(path: Path) -> int | None:
    try:
        if path.stat().st_size > MAX_UNTRACKED_LINE_COUNT_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _file_type_breakdown(totals: dict[str, FileTypeChange]) -> str:
    parts = []
    for file_type, item in sorted(
        totals.items(),
        key=lambda pair: (pair[1].additions + pair[1].deletions, pair[1].files, pair[0]),
        reverse=True,
    ):
        parts.append(f"{file_type} {_change_counts(item.additions, item.deletions)}")
    return "; ".join(parts[:6])


def _change_counts(additions: int, deletions: int) -> str:
    if additions and deletions:
        return f"+{additions}/-{deletions}"
    if additions:
        return f"+{additions}"
    if deletions:
        return f"-{deletions}"
    return "+0/-0"


def _plural(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"


def _evaluation_progress(latest: dict[str, Any]) -> str | None:
    path_value = latest.get("evaluation_stdout")
    if not path_value:
        return None
    try:
        text = Path(str(path_value)).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    matches = list(EVALUATION_LEVEL_RE.finditer(text))
    if not matches:
        return None

    match = matches[-1]
    level = match.group(1)
    dimensions = match.group(2)
    status = match.group(3)
    progress = f"level {level} ({dimensions})"
    if status:
        progress = f"{progress} {status.lower()}"
    pass_levels = [
        int(item.group(1))
        for item in matches
        if item.group(3) == "PASS"
    ]
    if pass_levels:
        progress = f"{progress}, latest pass {max(pass_levels)}, passed {len(pass_levels)}"
    return progress


def _last_eval_summary(status: dict[str, Any]) -> str:
    score = status.get("last_score")
    failing_level = status.get("first_failing_level")
    stop_status = status.get("stop_status")
    if score is None and failing_level is None:
        return "-"

    parts = []
    if score is not None:
        parts.append(f"passed {score}")
    if failing_level is not None:
        label = str(stop_status).lower() if stop_status else "stopped"
        parts.append(f"{label} at {failing_level}")
    return ", ".join(parts)


def _retry_countdown(status: dict[str, Any]) -> str:
    delay = _float(status.get("agent_retry_delay_seconds"))
    if delay is None:
        return "-"

    phase_started = _parse_timestamp(status.get("phase_started_at"))
    elapsed = 0.0
    if phase_started is not None:
        elapsed = max((datetime.now(timezone.utc) - phase_started).total_seconds(), 0)
    remaining = max(delay - elapsed, 0)

    attempt = _int(status.get("agent_attempt"))
    next_attempt = attempt + 1 if attempt > 0 else None
    budget_after_wait = _float(status.get("agent_retry_remaining_seconds")) or 0.0
    budget_left = remaining + budget_after_wait

    detail = f"{_duration(remaining)}"
    extras = []
    if next_attempt is not None:
        extras.append(f"next attempt {next_attempt}")
    if budget_left > 0:
        extras.append(f"budget {_duration(budget_left)}")
    if extras:
        detail = f"{detail} ({', '.join(extras)})"
    return detail


def _score_history(status: dict[str, Any]) -> str:
    history = status.get("score_history")
    if isinstance(history, list) and history:
        return ", ".join(str(item) for item in history)
    if status.get("last_score") is not None:
        return str(status.get("last_score"))
    return "-"


def _yes_no(value) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def _parse_timestamp(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _wrap_command(command, width: int) -> str:
    if isinstance(command, list):
        text = " ".join(str(part) for part in command)
    else:
        text = str(command)
    max_width = max(width - 2, 40)
    if len(text) <= max_width:
        return f"  {text}"
    return f"  {text[: max_width - 3]}..."


def _trim(line: str, width: int) -> str:
    if "\033[" in line:
        return line
    if width <= 0 or len(line) <= width:
        return line
    return _shorten_middle(line, width)


def _shorten_middle(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return "." * max_len
    keep_left = max((max_len - 3) // 2, 1)
    keep_right = max_len - 3 - keep_left
    return f"{text[:keep_left]}...{text[-keep_right:]}"


def _c(text: str, style: str, color: bool) -> str:
    return f"{style}{text}{RESET}" if color else text


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
