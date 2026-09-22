import dataclasses
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from puzzle_runner.config import load_config
from puzzle_runner.process import CommandResult
from puzzle_runner.runner import Runner


class EvaluationResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name).resolve()
        config_path = Path(__file__).resolve().parents[1] / "config.example.toml"
        self.config = dataclasses.replace(
            load_config(str(config_path), run_id="test-run"),
            download_full_levels=False,
            build_checker=False,
            echo_evaluation_output=False,
            worktree_root=root / "worktrees",
            log_root=root / "logs",
            status_dir=root / "current",
            results_path=root / "final_results.md",
        )

    def run_scenario(
        self, run_id, scores, *, shortcut_first=False, clean_evaluation=False,
        evaluation_returncode=0, evaluation_timed_out=False, **overrides,
    ):
        class ScenarioRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)
                (self.workspace / "run_solver").write_text("#!/bin/sh\necho solver\n")
                if clean_evaluation:
                    (self.workspace / "clean-evaluation").touch()
                (self.workspace / self.config.evaluation_script).write_text(
                    textwrap.dedent("""\
                        import argparse
                        from pathlib import Path

                        parser = argparse.ArgumentParser()
                        parser.add_argument('--start', type=int, default=1)
                        parser.add_argument('--timeout', type=int)
                        args = parser.parse_args()
                        score = int(Path('target-score').read_text())
                        for level in range(args.start, score + 1):
                            print(f'Level {level} (3x3): PASS (0.01s)')
                        if not Path('clean-evaluation').exists():
                            print(f'Level {max(args.start, score + 1)} (3x3): FAIL (0.01s)')
                        """),
                    encoding="utf-8",
                )

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _can_shortcut_default_solver_evaluation(self) -> bool:
                return shortcut_first and self._status["current_round"] == 1

            def _get_full_eval_password(self) -> str:
                return "test-password"

            def _run_evaluation(self, round_dir, *, start_level=1):
                result = super()._run_evaluation(round_dir, start_level=start_level)
                return dataclasses.replace(
                    result, returncode=evaluation_returncode, timed_out=evaluation_timed_out,
                )

            def _run_agent(self, round_number, round_dir, prompt):
                (self.workspace / "target-score").write_text(str(scores[round_number - 1]))
                stdout = round_dir / "agent.stdout.log"
                stderr = round_dir / "agent.stderr.log"
                stdout.write_text("PUZZLE_RUNNER_DONE\n", encoding="utf-8")
                stderr.write_text("", encoding="utf-8")
                return CommandResult(
                    argv=["agent"], cwd=self.workspace, returncode=0,
                    elapsed_seconds=0.0, timed_out=False, timeout_reason=None,
                    stdout_path=stdout, stderr_path=stderr,
                )

        config = dataclasses.replace(
            self.config, run_id=run_id, max_rounds=len(scores), **overrides
        )
        runner = ScenarioRunner(config)
        final = runner.run()
        events = [json.loads(line) for line in runner.events_path.read_text().splitlines()]
        starts = []
        for round_number in range(1, final.total_rounds + 1):
            round_dir = final.log_dir / f"round-{round_number:03d}"
            result = json.loads((round_dir / "evaluation_result.json").read_text())
            argv = result["argv"]
            start = int(argv[argv.index("--start") + 1]) if "--start" in argv else 1
            starts.append(start)
            evaluation_events = [
                event for event in events
                if event["event"] == "evaluation_finished" and event["round"] == round_number
            ]
            self.assertEqual(evaluation_events[0]["start_level"], start)
        self.assertEqual(runner._status["evaluation_start_level"], starts[-1])
        return runner, final, starts

    def test_resume_uses_best_score_across_improvements_and_regressions(self) -> None:
        runner, final, starts = self.run_scenario("progress", [450, 435, 451, 430, 460])
        self.assertEqual(starts, [1, 430, 430, 431, 431])
        self.assertEqual(runner._status["score_history"], [450, 435, 451, 430, 460])
        self.assertEqual(final.best_score, 460)
        self.assertEqual(final.best_round, 5)
        self.assertEqual(final.stop_reason, "max_rounds")
        prompt = (final.log_dir / "round-003" / "prompt.md").read_text()
        self.assertIn("Last full-evaluation highest passed level: 435.", prompt)
        self.assertIn("Best full-evaluation score so far: 450.", prompt)

        # A new run sharing the same status/results paths still starts from 1.
        _, _, starts = self.run_scenario("fresh", [5, 6])
        self.assertEqual(starts, [1, 1])

    def test_all_levels_solved_stops_immediately_and_records_result(self) -> None:
        for scores, expected_starts in [([1208, 1208], [1]), ([1190, 1208, 1208], [1, 1170])]:
            with self.subTest(scores=scores):
                runner, final, starts = self.run_scenario(
                    f"solved-{len(scores)}", scores, clean_evaluation=True,
                )
                self.assertEqual(starts, expected_starts)
                self.assertEqual(final.best_score, 1208)
                self.assertEqual(final.stop_reason, "all_levels_solved")
                self.assertEqual(runner._status["stop_reason"], "all_levels_solved")
                self.assertIn("all_levels_solved", runner.config.results_path.read_text())
                self.assertFalse((final.log_dir / f"round-{final.total_rounds + 1:03d}").exists())

    def test_completion_requires_clean_successful_evaluation(self) -> None:
        cases = [
            ("failure", {}, "max_rounds"),
            ("nonzero", {"clean_evaluation": True, "evaluation_returncode": 1}, "evaluation_failed"),
            ("timeout", {"clean_evaluation": True, "evaluation_timed_out": True}, "evaluation_timeout"),
            ("disabled", {"clean_evaluation": True, "evaluation_final_level": 0}, "max_rounds"),
        ]
        for run_id, options, reason in cases:
            with self.subTest(run_id=run_id):
                _, final, _ = self.run_scenario(run_id, [1208], **options)
                self.assertEqual(final.stop_reason, reason)

    def test_custom_final_level(self) -> None:
        _, final, starts = self.run_scenario(
            "custom-end", [10, 10], clean_evaluation=True, evaluation_final_level=10,
        )
        self.assertEqual(starts, [1])
        self.assertEqual(final.stop_reason, "all_levels_solved")

    def test_disabled_resume_starts_every_round_at_one(self) -> None:
        _, final, starts = self.run_scenario(
            "disabled", [450, 451], evaluation_resume_from_best=False
        )
        self.assertEqual(starts, [1, 1])
        self.assertEqual(final.best_score, 451)

    def test_custom_and_zero_backtrack(self) -> None:
        for backtrack, expected in [(7, [1, 1, 443]), (0, [1, 5, 450])]:
            with self.subTest(backtrack=backtrack):
                _, _, starts = self.run_scenario(
                    f"backtrack-{backtrack}", [5, 450, 451],
                    evaluation_backtrack_levels=backtrack,
                )
                self.assertEqual(starts, expected)

    def test_resumed_failures_still_reach_stale_limit(self) -> None:
        runner, final, starts = self.run_scenario("stale", [450, 435, 429, 449, 460])
        self.assertEqual(starts, [1, 430, 430, 430])
        self.assertEqual(runner._status["score_history"], [450, 435, 429, 449])
        self.assertEqual(final.best_score, 450)
        self.assertEqual(final.best_round, 1)
        self.assertEqual(final.stop_reason, "stale_limit")

    def test_default_solver_shortcut_seeds_next_evaluation(self) -> None:
        _, final, starts = self.run_scenario("shortcut", [47, 50], shortcut_first=True)
        self.assertEqual(starts, [1, 27])
        self.assertEqual(final.best_score, 50)


if __name__ == "__main__":
    unittest.main()
