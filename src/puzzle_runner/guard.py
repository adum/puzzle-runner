from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class GuardFinding:
    path: str
    reason: str
    pattern: str


class ForbiddenGuard:
    def __init__(self, workspace: Path, patterns: list[str]) -> None:
        self.workspace = workspace
        self.patterns = patterns
        self._baseline = self._snapshot()

    def check(self) -> list[GuardFinding]:
        current = self._snapshot()
        findings: list[GuardFinding] = []

        for rel_path, digest in self._baseline.items():
            if rel_path not in current:
                findings.append(
                    GuardFinding(rel_path, "deleted forbidden file", self._matching_pattern(rel_path))
                )
            elif current[rel_path] != digest:
                findings.append(
                    GuardFinding(rel_path, "modified forbidden file", self._matching_pattern(rel_path))
                )

        for rel_path in current:
            if rel_path not in self._baseline and self._is_forbidden(rel_path):
                findings.append(
                    GuardFinding(rel_path, "created forbidden file", self._matching_pattern(rel_path))
                )

        return findings

    def _snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in self.workspace.rglob("*"):
            if not path.is_file():
                continue
            rel_path = path.relative_to(self.workspace).as_posix()
            if self._is_forbidden(rel_path):
                snapshot[rel_path] = _sha256(path)
        return snapshot

    def _is_forbidden(self, rel_path: str) -> bool:
        return self._matching_pattern(rel_path) != ""

    def _matching_pattern(self, rel_path: str) -> str:
        for pattern in self.patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return pattern
        return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
