import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from puzzle_runner.watch import CLEAR_SCREEN, _draw_frame, render_status, resolve_status_path


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
            "last_improved": False,
            "stale_count": 1,
            "stale_limit": 3,
            "remaining_no_progress_tries": 2,
            "elapsed_seconds": 65.2,
            "latest": {"agent_stdout": "/tmp/agent.stdout.log"},
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Puzzle Runner", rendered)
        self.assertIn("[ACTIVE]", rendered)
        self.assertIn("agent_running", rendered)
        self.assertIn("Agent running", rendered)
        self.assertIn("47", rendered)
        self.assertIn("Remaining tries", rendered)
        self.assertIn("agent.stdout.log", rendered)
        self.assertNotIn("\033[", rendered)

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
