import tempfile
import unittest
from pathlib import Path

from puzzle_runner.evaluation import parse_evaluation_output


class EvaluationParseTests(unittest.TestCase):
    def test_resumed_evaluation_scores(self) -> None:
        cases = [
            ("Level 430 (3x3): PASS (0.01s)\nLevel 451 (4x4): PASS (0.02s)\n", 451),
            ("Level 430 (3x3): FAIL (0.01s)\n", 429),
            ("Level 430 (3x3): TIMEOUT - Exceeded 600s limit (600.01s)\n", 429),
            ("Level 430 (3x3): ERROR (0.01s): solver unavailable\n", 429),
            ("Missing encrypted even-level archive\n", 0),
            ("", 0),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "stdout.log"
            stderr = Path(temp_dir) / "stderr.log"
            stderr.write_text("", encoding="utf-8")
            for output, expected_score in cases:
                with self.subTest(output=output):
                    stdout.write_text(output, encoding="utf-8")
                    parsed = parse_evaluation_output(stdout, stderr, start_level=430)
                    self.assertEqual(parsed.highest_passed, expected_score)

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
