import tempfile
import unittest
from pathlib import Path

from puzzle_runner.process import run_streamed
from puzzle_runner.runner import _agent_attempt_log_paths


class ProcessTests(unittest.TestCase):
    def test_run_streamed_merges_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stdout = temp_path / "stdout.log"
            stderr = temp_path / "stderr.log"

            result = run_streamed(
                ["python3", "-c", "import os; print(os.environ['PUZZLE_TEST_ENV'])"],
                cwd=temp_path,
                stdin_text=None,
                timeout_seconds=10,
                stdout_path=stdout,
                stderr_path=stderr,
                echo=False,
                env={"PUZZLE_TEST_ENV": "ok"},
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(stdout.read_text(encoding="utf-8").strip(), "ok")

    def test_run_streamed_times_out_when_agent_goes_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stdout = temp_path / "stdout.log"
            stderr = temp_path / "stderr.log"

            result = run_streamed(
                ["python3", "-c", "import time; time.sleep(5)"],
                cwd=temp_path,
                stdin_text=None,
                timeout_seconds=10,
                stdout_path=stdout,
                stderr_path=stderr,
                echo=False,
                idle_timeout_seconds=1,
            )

            self.assertTrue(result.timed_out)
            self.assertEqual(result.timeout_reason, "idle")
            self.assertLess(result.elapsed_seconds, 3)

    def test_agent_retry_attempt_paths_keep_first_attempt_compatible(self) -> None:
        root = Path("/tmp/round")

        first_stdout, first_stderr = _agent_attempt_log_paths(root, 1)
        second_stdout, second_stderr = _agent_attempt_log_paths(root, 2)

        self.assertEqual(first_stdout.name, "agent.stdout.log")
        self.assertEqual(first_stderr.name, "agent.stderr.log")
        self.assertEqual(second_stdout.name, "agent.attempt-002.stdout.log")
        self.assertEqual(second_stderr.name, "agent.attempt-002.stderr.log")


if __name__ == "__main__":
    unittest.main()
