from __future__ import annotations

import unittest

from puzzle_runner.cli import build_parser


class CliTests(unittest.TestCase):
    def test_run_parser_accepts_effort_override(self) -> None:
        args = build_parser().parse_args(["--effort", "high"])

        self.assertEqual(args.effort, "high")


if __name__ == "__main__":
    unittest.main()
