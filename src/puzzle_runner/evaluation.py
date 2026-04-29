from __future__ import annotations

import dataclasses
import re
from pathlib import Path


PASS_RE = re.compile(r"^Level\s+(\d+)\s+\([^)]*\):\s+PASS\b", re.MULTILINE)
STOP_RE = re.compile(r"^Level\s+(\d+)\s+\([^)]*\):\s+(FAIL|TIMEOUT|ERROR)\b(?P<rest>.*)$", re.MULTILINE)


@dataclasses.dataclass(frozen=True)
class EvaluationParse:
    highest_passed: int
    first_failing_level: int | None
    stop_status: str | None
    failure_reason: str | None


def parse_evaluation_output(stdout_path: Path, stderr_path: Path) -> EvaluationParse:
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")

    passed = [int(match.group(1)) for match in PASS_RE.finditer(stdout)]
    highest = max(passed) if passed else 0

    stop_match = STOP_RE.search(stdout)
    first_failing_level = int(stop_match.group(1)) if stop_match else None
    stop_status = stop_match.group(2) if stop_match else None
    failure_reason = _extract_failure_reason(stdout, stderr, stop_match)

    return EvaluationParse(
        highest_passed=highest,
        first_failing_level=first_failing_level,
        stop_status=stop_status,
        failure_reason=failure_reason,
    )


def _extract_failure_reason(stdout: str, stderr: str, stop_match: re.Match[str] | None) -> str | None:
    if stop_match:
        rest = stop_match.group("rest").strip()
        if rest and not re.fullmatch(r"\([^)]+\)", rest):
            return rest.removeprefix("-").strip()

    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Error:"):
            return stripped
        if stripped.startswith("Solver stderr:"):
            return stripped

    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None
