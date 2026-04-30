from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    cwd: Path
    returncode: int
    elapsed_seconds: float
    timed_out: bool
    timeout_reason: str | None
    stdout_path: Path
    stderr_path: Path


@dataclasses.dataclass
class _OutputActivity:
    last_output_monotonic: float
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def mark(self) -> None:
        with self.lock:
            self.last_output_monotonic = time.monotonic()

    def idle_seconds(self) -> float:
        with self.lock:
            return time.monotonic() - self.last_output_monotonic


def run_streamed(
    argv: list[str],
    *,
    cwd: Path,
    stdin_text: str | None,
    timeout_seconds: int | None,
    stdout_path: Path,
    stderr_path: Path,
    echo: bool,
    idle_timeout_seconds: int | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with stdout_path.open("w", encoding="utf-8", errors="replace") as out_handle:
        with stderr_path.open("w", encoding="utf-8", errors="replace") as err_handle:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=_merged_env(env),
                    stdin=subprocess.PIPE if stdin_text is not None else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                err_handle.write(f"failed to start command: {exc}\n")
                err_handle.flush()
                if echo:
                    sys.stderr.write(f"failed to start command: {exc}\n")
                    sys.stderr.flush()
                return CommandResult(
                    argv=argv,
                    cwd=cwd,
                    returncode=127,
                    elapsed_seconds=time.monotonic() - started,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )

            activity = _OutputActivity(started)
            stdout_thread = threading.Thread(
                target=_copy_stream,
                args=(process.stdout, out_handle, sys.stdout, echo, activity),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_copy_stream,
                args=(process.stderr, err_handle, sys.stderr, echo, activity),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            if stdin_text is not None and process.stdin is not None:
                try:
                    try:
                        process.stdin.write(stdin_text)
                        process.stdin.flush()
                    except BrokenPipeError:
                        pass
                finally:
                    process.stdin.close()

            returncode, timed_out, timeout_reason = _wait_for_process(
                process,
                started=started,
                timeout_seconds=timeout_seconds,
                idle_timeout_seconds=idle_timeout_seconds,
                activity=activity,
            )

            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

    return CommandResult(
        argv=argv,
        cwd=cwd,
        returncode=returncode,
        elapsed_seconds=time.monotonic() - started,
        timed_out=timed_out,
        timeout_reason=timeout_reason,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _copy_stream(source, dest_handle, echo_handle, echo: bool, activity: _OutputActivity) -> None:
    if source is None:
        return
    try:
        for chunk in iter(lambda: source.read(1), ""):
            activity.mark()
            dest_handle.write(chunk)
            dest_handle.flush()
            if echo:
                echo_handle.write(chunk)
                echo_handle.flush()
    finally:
        source.close()


def _wait_for_process(
    process: subprocess.Popen,
    *,
    started: float,
    timeout_seconds: int | None,
    idle_timeout_seconds: int | None,
    activity: _OutputActivity,
) -> tuple[int, bool, str | None]:
    deadline = started + timeout_seconds if timeout_seconds is not None else None
    idle_timeout = (
        idle_timeout_seconds
        if idle_timeout_seconds is not None and idle_timeout_seconds > 0
        else None
    )

    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode, False, None

        now = time.monotonic()
        if deadline is not None and now >= deadline:
            return _kill_timed_out_process(process, "wall")
        if idle_timeout is not None and activity.idle_seconds() >= idle_timeout:
            return _kill_timed_out_process(process, "idle")

        sleep_seconds = 0.1
        if deadline is not None:
            sleep_seconds = min(sleep_seconds, max(deadline - now, 0.0))
        if idle_timeout is not None:
            sleep_seconds = min(sleep_seconds, max(idle_timeout - activity.idle_seconds(), 0.0))
        time.sleep(max(sleep_seconds, 0.01))


def _kill_timed_out_process(process: subprocess.Popen, reason: str) -> tuple[int, bool, str]:
    try:
        process.kill()
    except OSError:
        pass
    return process.wait(), True, reason


def _merged_env(extra: dict[str, str] | None) -> dict[str, str] | None:
    if extra is None:
        return None
    merged = os.environ.copy()
    merged.update(extra)
    return merged
