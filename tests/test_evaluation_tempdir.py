import dataclasses
import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from puzzle_runner.config import load_config
from puzzle_runner.runner import Runner


class EvaluationTempdirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name).resolve()
        config_path = Path(__file__).resolve().parents[1] / "config.example.toml"
        config = dataclasses.replace(
            load_config(str(config_path), run_id="test-run"),
            worktree_root=root / "workspaces",
            log_root=root / "run logs",
            status_dir=root / "status",
            results_path=root / "results.md",
            echo_evaluation_output=False,
            evaluation_process_timeout_seconds=2,
        )
        self.runner = Runner(config)
        self.runner._prepare_paths()
        self.runner.workspace.mkdir()
        self.password_patch = mock.patch.object(
            self.runner, "_get_full_eval_password", return_value="test-password"
        )
        self.password_patch.start()
        self.addCleanup(self.password_patch.stop)

    def test_subprocess_tempfiles_are_isolated_and_cleaned_for_each_outcome(self) -> None:
        inherited_temp = Path(self.temp_dir.name) / "system-temp"
        inherited_temp.mkdir()
        sentinel = inherited_temp / "keep.txt"
        sentinel.write_text("unrelated file", encoding="utf-8")
        inherited_env = {key: str(inherited_temp) for key in ("TMPDIR", "TMP", "TEMP")}
        evaluator = self.runner.workspace / self.runner.config.evaluation_script
        scratch_dirs = set()

        for outcome in ("success", "failure", "timeout"):
            with self.subTest(outcome=outcome):
                # Leave scratch files behind deliberately: the runner owns cleanup,
                # including when the evaluator is killed before its own cleanup.
                evaluator.write_text(
                    textwrap.dedent("""\
                        import json
                        import os
                        from pathlib import Path
                        import sys
                        import tempfile
                        import time

                        assert os.environ['COIL_FULL_PASSWORD'] == 'test-password'
                        assert sys.stdin.readline().strip() == 'test-password'
                        scratch = Path(tempfile.gettempdir())
                        levels = Path(tempfile.mkdtemp(prefix='coil_even_levels_'))
                        level = levels / '732'
                        level.write_text('dummy level', encoding='utf-8')
                        os.utime(level, (0, 0))
                        print(json.dumps({
                            'scratch': str(scratch),
                            'level': str(level),
                            'env': {key: os.environ[key] for key in ('TMPDIR', 'TMP', 'TEMP')},
                            'mode': scratch.stat().st_mode & 0o777,
                        }), flush=True)
                        """)
                    + ("time.sleep(60)\n" if outcome == "timeout" else "")
                    + f"sys.exit({7 if outcome == 'failure' else 0})\n",
                    encoding="utf-8",
                )
                round_dir = self.runner.log_dir / outcome

                with mock.patch.dict(os.environ, inherited_env):
                    result = self.runner._run_evaluation(round_dir)
                    self.assertEqual({key: os.environ[key] for key in inherited_env}, inherited_env)

                payload = json.loads(result.stdout_path.read_text(encoding="utf-8"))
                scratch = Path(payload["scratch"])
                self.assertTrue(scratch.is_absolute())
                self.assertEqual(scratch.parent, round_dir.resolve())
                self.assertIn(scratch, Path(payload["level"]).parents)
                self.assertEqual(payload["env"], dict.fromkeys(inherited_env, str(scratch)))
                if os.name == "posix":
                    self.assertEqual(payload["mode"], 0o700)
                self.assertNotIn(scratch, scratch_dirs)
                scratch_dirs.add(scratch)
                self.assertFalse(scratch.exists())
                self.assertFalse(Path(payload["level"]).exists())
                self.assertEqual(list(round_dir.glob("evaluation-tmp-*")), [])
                self.assertTrue(result.stderr_path.exists())
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "unrelated file")
                self.assertEqual(result.timed_out, outcome == "timeout")
                if outcome == "timeout":
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.timeout_reason, "wall")
                else:
                    self.assertEqual(result.returncode, 7 if outcome == "failure" else 0)

    def test_scratch_directory_is_cleaned_if_process_runner_raises(self) -> None:
        round_dir = self.runner.log_dir / "round-001"
        scratch_dirs = []

        def fail_to_run(*args, **kwargs):
            scratch = Path(kwargs["env"]["TMPDIR"])
            scratch_dirs.append(scratch)
            (scratch / "private-level").write_text("dummy level", encoding="utf-8")
            raise RuntimeError("process runner failed")

        with mock.patch("puzzle_runner.runner.run_streamed", side_effect=fail_to_run):
            with self.assertRaisesRegex(RuntimeError, "process runner failed"):
                self.runner._run_evaluation(round_dir)

        self.assertEqual(len(scratch_dirs), 1)
        self.assertFalse(scratch_dirs[0].exists())
        self.assertEqual(list(round_dir.glob("evaluation-tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
