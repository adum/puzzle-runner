from __future__ import annotations

import dataclasses
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
    stdout_path: Path
    stderr_path: Path


def run_streamed(
    argv: list[str],
    *,
    cwd: Path,
    stdin_text: str | None,
    timeout_seconds: int | None,
    stdout_path: Path,
    stderr_path: Path,
    echo: bool,
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
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )

            stdout_thread = threading.Thread(
                target=_copy_stream,
                args=(process.stdout, out_handle, sys.stdout, echo),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_copy_stream,
                args=(process.stderr, err_handle, sys.stderr, echo),
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

            timed_out = False
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                returncode = process.wait()

            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

    return CommandResult(
        argv=argv,
        cwd=cwd,
        returncode=returncode,
        elapsed_seconds=time.monotonic() - started,
        timed_out=timed_out,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _copy_stream(source, dest_handle, echo_handle, echo: bool) -> None:
    if source is None:
        return
    for line in iter(source.readline, ""):
        dest_handle.write(line)
        dest_handle.flush()
        if echo:
            echo_handle.write(line)
            echo_handle.flush()
