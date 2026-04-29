import tempfile
import unittest
from pathlib import Path

from puzzle_runner.watch import render_status, resolve_status_path


class WatchTests(unittest.TestCase):
    def test_render_status_without_color(self) -> None:
        status = {
            "active": True,
            "phase": "agent_running",
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


if __name__ == "__main__":
    unittest.main()
