from __future__ import annotations

import argparse
import json
import os
import shutil
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

    try:
        if args.once:
            print(render_status(load_status(status_path), status_path=status_path, color=color))
            return 0

        previous_lines: list[str] = []
        first_frame = True
        sys.stdout.write(HIDE_CURSOR if use_ansi else "")
        sys.stdout.flush()
        while True:
            rendered = render_status(load_status(status_path), status_path=status_path, color=color)
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


def render_status(status: dict[str, Any], *, status_path: Path, color: bool = True) -> str:
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
    agent_chars = _agent_output_chars(latest)
    if agent_chars is not None:
        lines.append(_kv("Agent output", f"{agent_chars:,} chars", color, width))

    lines.extend(
        [
            "",
            _section("Score", color),
            _kv("Best", status.get("best_score"), color, width),
            _kv("Best round", status.get("best_round"), color, width),
            _kv("Last", status.get("last_score"), color, width),
            _kv("Improved", _yes_no(status.get("last_improved")), color, width),
            _kv("No-progress", f"{stale_count}/{stale_limit} {_bar(stale_count, stale_limit, color)}", color, width),
            _kv("Remaining tries", remaining, color, width),
            _kv("Stop reason", stop_reason, color, width),
        ]
    )

    command = status.get("current_command")
    if command:
        lines.extend(["", _section("Current Command", color), _wrap_command(command, width)])

    key_paths = [
        ("Workspace", status.get("workspace")),
        ("Log dir", status.get("log_dir")),
        ("Events", status.get("events_log")),
        ("Status", status_path),
    ]
    lines.extend(["", _section("Paths", color)])
    for label, value in key_paths:
        lines.append(_kv(label, value, color, width))

    if latest:
        lines.extend(["", _section("Latest Round Files", color)])
        for key in [
            "prompt",
            "agent_stdout",
            "agent_stderr",
            "evaluation_stdout",
            "evaluation_stderr",
            "evaluation_parse",
            "workspace_diff",
        ]:
            if key in latest:
                lines.append(_kv(key, latest[key], color, width))

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


def _agent_output_chars(latest: dict[str, Any]) -> int | None:
    paths = [latest.get("agent_stdout"), latest.get("agent_stderr")]
    if not any(paths):
        return None

    total = 0
    saw_file = False
    for path_value in paths:
        if not path_value:
            continue
        try:
            total += Path(str(path_value)).stat().st_size
            saw_file = True
        except OSError:
            continue
    return total if saw_file else None


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
