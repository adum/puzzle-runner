import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from puzzle_runner.watch import (
    CLEAR_SCREEN,
    WorkspaceChangeCache,
    _draw_frame,
    _workspace_change_summary,
    render_status,
    resolve_status_path,
)


class WatchTests(unittest.TestCase):
    def test_render_status_without_color(self) -> None:
        agent_started_at = (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        status = {
            "active": True,
            "phase": "agent_running",
            "phase_started_at": agent_started_at,
            "agent_started_at": agent_started_at,
            "run_id": "run-1",
            "agent": "codex",
            "current_round": 2,
            "max_rounds": 5,
            "best_score": 47,
            "best_round": 1,
            "last_score": 47,
            "score_history": [35, 47],
            "last_improved": False,
            "stale_count": 1,
            "stale_limit": 3,
            "remaining_no_progress_tries": 2,
            "elapsed_seconds": 65.2,
            "workspace": "/tmp/workspace",
            "log_dir": "/tmp/logs",
            "latest": {"agent_stdout": "/tmp/agent.stdout.log"},
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Puzzle Runner", rendered)
        self.assertIn("[ACTIVE]", rendered)
        self.assertIn("agent_running", rendered)
        self.assertIn("Agent running", rendered)
        self.assertIn("47", rendered)
        self.assertIn("Scores", rendered)
        self.assertIn("35, 47", rendered)
        self.assertIn("Remaining tries", rendered)
        self.assertNotIn("Paths", rendered)
        self.assertNotIn("Latest Round Files", rendered)
        self.assertNotIn("/tmp/workspace", rendered)
        self.assertNotIn("/tmp/logs", rendered)
        self.assertNotIn("agent.stdout.log", rendered)
        self.assertNotIn("\033[", rendered)

    def test_render_status_includes_agent_output_chars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "agent.stdout.log"
            stderr = Path(temp_dir) / "agent.stderr.log"
            stdout.write_text("abc", encoding="utf-8")
            stderr.write_text("defgh", encoding="utf-8")
            agent_started_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            status = {
                "active": True,
                "phase": "agent_running",
                "agent_started_at": agent_started_at,
                "latest": {
                    "agent_stdout": str(stdout),
                    "agent_stderr": str(stderr),
                },
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Agent output", rendered)
        self.assertIn("8 chars", rendered)
        self.assertIn("/min", rendered)
        self.assertIn("Last output", rendered)

    def test_render_status_includes_evaluation_level_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "evaluation.stdout.log"
            stdout.write_text(
                "Level 1 (5x5): PASS (0.02s)\nLevel 3 (10x10): ",
                encoding="utf-8",
            )
            status = {
                "active": True,
                "phase": "evaluation_running",
                "latest": {"evaluation_stdout": str(stdout)},
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Evaluating", rendered)
        self.assertIn("level 3 (10x10)", rendered)
        self.assertIn("latest pass 1", rendered)
        self.assertIn("passed 1", rendered)

    def test_render_status_includes_last_evaluation_summary(self) -> None:
        status = {
            "active": True,
            "phase": "agent_running",
            "last_score": 119,
            "first_failing_level": 120,
            "stop_status": "TIMEOUT",
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Last eval", rendered)
        self.assertIn("passed 119, timeout at 120", rendered)

    def test_render_status_includes_retry_countdown(self) -> None:
        phase_started_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        status = {
            "active": True,
            "phase": "agent_retry_wait",
            "phase_started_at": phase_started_at,
            "agent_attempt": 2,
            "agent_retry_delay_seconds": 40,
            "agent_retry_remaining_seconds": 120,
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Retrying in", rendered)
        self.assertIn("next attempt 3", rendered)
        self.assertIn("budget", rendered)

    def test_render_status_includes_workspace_change_summary(self) -> None:
        status = {
            "active": True,
            "phase": "agent_running",
        }

        rendered = render_status(
            status,
            status_path=Path("/tmp/status.json"),
            color=False,
            workspace_changes="2 files, +5/-1 | Python +3/-1; Shell +2",
        )

        self.assertIn("Workspace", rendered)
        self.assertIn("Changes", rendered)
        self.assertIn("Python +3/-1", rendered)

    def test_workspace_change_summary_groups_file_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)

            solver = root / "solver.py"
            solver.write_text("print('old')\n", encoding="utf-8")
            subprocess.run(["git", "add", "solver.py"], cwd=root, check=True, capture_output=True)
            solver.write_text("print('old')\nprint('new')\n", encoding="utf-8")

            script = root / "tool.sh"
            script.write_text("#!/usr/bin/env sh\necho ok\n", encoding="utf-8")

            public_level = root / "levels_public" / "101"
            public_level.parent.mkdir()
            public_level.write_text("ignored\n", encoding="utf-8")

            summary = _workspace_change_summary(root)

        self.assertIsNotNone(summary)
        self.assertIn("2 files", summary)
        self.assertIn("Python +1", summary)
        self.assertIn("Shell +2", summary)
        self.assertNotIn("ignored", summary)

    def test_workspace_change_cache_uses_refresh_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            (root / "one.py").write_text("print(1)\n", encoding="utf-8")
            cache = WorkspaceChangeCache(refresh_interval=999)
            status = {"workspace": str(root)}

            first = cache.get(status)
            (root / "two.py").write_text("print(2)\n", encoding="utf-8")
            second = cache.get(status)

        self.assertEqual(first, second)

    def test_resolve_status_path_default_without_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(temp_dir)
                path = resolve_status_path(None, "missing.toml")
            finally:
                os.chdir(cwd)

        self.assertEqual(path.name, "status.json")
        self.assertIn(".puzzle-runs", path.as_posix())

    def test_draw_frame_only_clears_screen_once(self) -> None:
        out = StringIO()
        with redirect_stdout(out):
            previous = _draw_frame("one\ntwo\n", [], first_frame=True)
            previous = _draw_frame("one\nthree\n", previous, first_frame=False)

        written = out.getvalue()
        self.assertEqual(written.count(CLEAR_SCREEN), 1)
        self.assertIn("three", written)


if __name__ == "__main__":
    unittest.main()
