import tempfile
import unittest
from pathlib import Path

from puzzle_runner.evaluation import parse_evaluation_output


class EvaluationParseTests(unittest.TestCase):
    def test_parse_pass_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            stdout = tmp_path / "stdout.log"
            stderr = tmp_path / "stderr.log"
            stdout.write_text(
                "\n".join(
                    [
                        "Level 1 (3x3): PASS (0.01s)",
                        "Level 2 (4x4): PASS (0.02s)",
                        "Level 3 (5x5): TIMEOUT - Exceeded 600s limit (600.01s)",
                    ]
                ),
                encoding="utf-8",
            )
            stderr.write_text("", encoding="utf-8")

            parsed = parse_evaluation_output(stdout, stderr)

            self.assertEqual(parsed.highest_passed, 2)
            self.assertEqual(parsed.first_failing_level, 3)
            self.assertEqual(parsed.stop_status, "TIMEOUT")
            self.assertIn("Exceeded 600s", parsed.failure_reason or "")

    def test_parse_failure_error_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            stdout = tmp_path / "stdout.log"
            stderr = tmp_path / "stderr.log"
            stdout.write_text(
                "\n".join(
                    [
                        "Level 1 (3x3): PASS (0.01s)",
                        "Level 2 (4x4): FAIL (0.02s)",
                        "  Error: path misses 5 fields",
                    ]
                ),
                encoding="utf-8",
            )
            stderr.write_text("", encoding="utf-8")

            parsed = parse_evaluation_output(stdout, stderr)

            self.assertEqual(parsed.highest_passed, 1)
            self.assertEqual(parsed.first_failing_level, 2)
            self.assertEqual(parsed.stop_status, "FAIL")
            self.assertEqual(parsed.failure_reason, "Error: path misses 5 fields")


if __name__ == "__main__":
    unittest.main()
