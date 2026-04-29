import tempfile
import unittest
from pathlib import Path

from puzzle_runner.process import run_streamed


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


if __name__ == "__main__":
    unittest.main()
