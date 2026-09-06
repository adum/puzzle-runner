import contextlib
import dataclasses
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from puzzle_runner.cli import build_parser, main
from puzzle_runner.config import load_config


class CliTests(unittest.TestCase):
    def test_evaluation_overrides_and_config_precedence(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        final = SimpleNamespace(
            best_score=0, stop_reason="max_rounds", stop_detail="test",
            total_wall_seconds=0.0, agent_output_chars=0, code_lines_added=0,
            log_dir=Path("logs"),
        )
        cases = [
            ([], True, 20),
            (["--no-evaluation-resume-from-best"], False, 20),
            (["--evaluation-backtrack-levels", "0"], True, 0),
            (["--evaluation-resume-from-best", "--evaluation-backtrack-levels", "7"], True, 7),
        ]
        for configured_enabled in (True, False):
            configured = dataclasses.replace(config, evaluation_resume_from_best=configured_enabled)
            for args, expected_enabled, expected_backtrack in cases:
                with self.subTest(configured_enabled=configured_enabled, args=args):
                    with mock.patch("puzzle_runner.cli.load_config", return_value=configured):
                        with mock.patch("puzzle_runner.cli.Runner") as runner:
                            runner.return_value.run.return_value = final
                            with contextlib.redirect_stdout(io.StringIO()):
                                self.assertEqual(main(["run", *args]), 0)
                            actual = runner.call_args.args[0]
                    if not any("resume-from-best" in arg for arg in args):
                        expected_enabled = configured_enabled
                    self.assertEqual(actual.evaluation_resume_from_best, expected_enabled)
                    self.assertEqual(actual.evaluation_backtrack_levels, expected_backtrack)

    def test_invalid_backtrack_is_rejected_before_running(self) -> None:
        for value in ("-1", "1.5", "abc"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    build_parser().parse_args(["--evaluation-backtrack-levels", value])
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
