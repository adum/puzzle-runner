import subprocess
import dataclasses
import tempfile
import unittest
from pathlib import Path

from puzzle_runner.config import load_config
from puzzle_runner.openrouter_agent import AGENT_CONFIG_ERROR_RETURN_CODE, AGENT_MAX_STEPS_RETURN_CODE
from puzzle_runner.process import CommandResult
from puzzle_runner.runner import (
    FinalResult,
    Runner,
    _agent_effort_text,
    _agent_result_is_retryable,
    _apply_agent_effort,
    _apply_agent_model,
    _claude_stdout_has_error_result,
    _ensure_results_summary_header,
    _is_terminal_claude_result_line,
    _migrate_results_summary_effort_column,
    _results_summary_row,
    count_agent_output_chars,
    count_code_lines_added,
    explain_stop_reason,
    normalize_script_line_endings,
)


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.example.toml"
        self.config = load_config(str(config_path), run_id="test-run")

    def test_explain_agent_wall_timeout_names_parameter(self) -> None:
        detail = explain_stop_reason(
            "agent_timeout",
            self.config,
            {"last_agent_elapsed_seconds": 7210.03},
        )

        self.assertIn("agent_timeout_seconds=86400s", detail)
        self.assertIn("7210.03s", detail)

    def test_explain_agent_idle_timeout_names_parameter(self) -> None:
        detail = explain_stop_reason("agent_idle_timeout", self.config, {})

        self.assertIn("agent_idle_timeout_seconds=1800s", detail)

    def test_explain_stale_limit_names_parameter(self) -> None:
        detail = explain_stop_reason("stale_limit", self.config, {})

        self.assertIn("stale_limit=3", detail)

    def test_explain_agent_max_steps_names_parameter(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.openrouter.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        detail = explain_stop_reason("agent_max_steps", config, {})

        self.assertIn("agent.max_steps=200", detail)
        self.assertIn("before running evaluation", detail)

    def test_explain_forbidden_edit_names_path_reason_and_pattern(self) -> None:
        detail = explain_stop_reason(
            "forbidden_edit_detected",
            self.config,
            {
                "guard_phase": "pre_evaluation",
                "guard_findings": [
                    {
                        "path": "levels_public/101",
                        "reason": "modified forbidden file",
                        "pattern": "levels_public/**",
                    }
                ],
            },
        )

        self.assertIn("pre_evaluation", detail)
        self.assertIn("modified forbidden file", detail)
        self.assertIn("levels_public/101", detail)
        self.assertIn("levels_public/**", detail)

    def test_count_agent_output_chars_sums_attempt_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            round_dir = log_dir / "round-001"
            round_dir.mkdir()
            (round_dir / "agent.stdout.log").write_text("abc", encoding="utf-8")
            (round_dir / "agent.stderr.log").write_text("defg", encoding="utf-8")
            (round_dir / "agent.attempt-002.stderr.log").write_text("hi", encoding="utf-8")
            (round_dir / "evaluation.stdout.log").write_text("ignored", encoding="utf-8")

            total = count_agent_output_chars(log_dir)

        self.assertEqual(total, 9)

    def test_normalize_script_line_endings_only_touches_script_like_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "evaluate.py"
            script.write_bytes(b"#!/usr/bin/env python3\r\nprint('ok')\r\n")
            shell = root / "run_solver"
            shell.write_bytes(b"#!/usr/bin/env sh\r\necho ok\r\n")
            notes = root / "notes.txt"
            notes.write_bytes(b"hello\r\n")

            changed = normalize_script_line_endings(root)
            script_bytes = script.read_bytes()
            shell_bytes = shell.read_bytes()
            notes_bytes = notes.read_bytes()

        self.assertEqual(changed, ["evaluate.py", "run_solver"])
        self.assertEqual(script_bytes, b"#!/usr/bin/env python3\nprint('ok')\n")
        self.assertEqual(shell_bytes, b"#!/usr/bin/env sh\necho ok\n")
        self.assertEqual(notes_bytes, b"hello\r\n")

    def test_count_agent_output_chars_uses_claude_text_not_json_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            round_dir = log_dir / "round-001"
            round_dir.mkdir()
            (round_dir / "agent.stdout.log").write_text(
                '{"type":"system","subtype":"init"}\n'
                '{"type":"stream_event","event":{"type":"content_block_delta",'
                '"delta":{"type":"text_delta","text":"Hello"}}}\n'
                '{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}\n'
                '{"type":"result","result":"Hello"}\n',
                encoding="utf-8",
            )
            (round_dir / "agent.stderr.log").write_text("raw stderr ignored", encoding="utf-8")
            (round_dir / "agent.attempt-002.stdout.log").write_text(
                '{"type":"assistant","message":{"content":[{"type":"text","text":"Bye"}]}}\n',
                encoding="utf-8",
            )

            total = count_agent_output_chars(log_dir, agent_stream_format="claude-stream-json")

        self.assertEqual(total, len("HelloBye"))

    def test_count_agent_output_chars_uses_gemini_text_not_json_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            round_dir = log_dir / "round-001"
            round_dir.mkdir()
            (round_dir / "agent.stdout.log").write_text(
                '{"type":"init","model":"gemini-3.1-pro-preview"}\n'
                '{"type":"message","role":"assistant","content":"Hello","delta":true}\n'
                '{"type":"result","status":"success","stats":{"total_tokens":12}}\n',
                encoding="utf-8",
            )
            (round_dir / "agent.attempt-002.stdout.log").write_text(
                '{"type":"message","role":"assistant","content":"Bye","delta":true}\n',
                encoding="utf-8",
            )
            (round_dir / "agent.stderr.log").write_text("raw stderr ignored", encoding="utf-8")

            total = count_agent_output_chars(log_dir, agent_stream_format="gemini-stream-json")

        self.assertEqual(total, len("HelloBye"))

    def test_count_code_lines_added_counts_code_and_ignores_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)

            solver = root / "solver.py"
            solver.write_text("print('old')\n", encoding="utf-8")
            subprocess.run(["git", "add", "solver.py"], cwd=root, check=True, capture_output=True)
            solver.write_text("print('old')\nprint('new')\n", encoding="utf-8")

            script = root / "run_solver"
            script.write_text("#!/usr/bin/env sh\necho ok\n", encoding="utf-8")

            notes = root / "notes.md"
            notes.write_text("# ignored\n", encoding="utf-8")

            public_level = root / "levels_public" / "101"
            public_level.parent.mkdir()
            public_level.write_text("ignored\n", encoding="utf-8")

            total = count_code_lines_added(root)

        self.assertEqual(total, 3)

    def test_results_summary_row_has_new_metrics_and_no_logs_column(self) -> None:
        final = FinalResult(
            run_id="run-1",
            best_score=119,
            best_round=1,
            total_rounds=2,
            stop_reason="stale_limit",
            stop_detail="done",
            total_wall_seconds=3661,
            agent_output_chars=12345,
            code_lines_added=67,
            log_dir=Path("/tmp/logs"),
            workspace=Path("/tmp/workspace"),
        )

        row = _results_summary_row(final, self.config)

        self.assertIn("1h 1m", row)
        self.assertIn("12345", row)
        self.assertIn("67", row)
        self.assertNotIn("/tmp/logs", row)

    def test_results_summary_header_appends_new_schema_after_old_header(self) -> None:
        old_header = (
            "| Run ID | Agent | Best Score | Best Round | Rounds | Stop Reason | Timeout | Logs |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "final_results.md"
            path.write_text(old_header, encoding="utf-8")

            _ensure_results_summary_header(path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("Agent Chars", text)
        self.assertIn("Code Lines Added", text)
        self.assertIn("OpenRouter Cost", text)
        self.assertIn("| Run ID | Agent | Effort |", text)
        self.assertNotIn("Timeout | Logs |\n\n| Run ID | Agent | Best Score | Best Round | Rounds | Stop Reason | Timeout | Logs", text)

    def test_results_summary_header_migrates_no_effort_schema(self) -> None:
        old_table = (
            "| Run ID | Agent | Best Score | Best Round | Rounds | Stop Reason | Timeout | Wall Time | Agent Chars | Code Lines Added |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| run-1 | claude | 0 |  | 1 | agent_idle_timeout | 600s | 30m 13s | 0 | 311 |\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "final_results.md"
            path.write_text(old_table, encoding="utf-8")

            _ensure_results_summary_header(path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("| Run ID | Agent | Effort | Best Score |", text)
        self.assertIn("| run-1 | claude |  | 0 |", text)
        self.assertIn("| run-1 | claude |  | 0 |  | 1 | agent_idle_timeout | 600s | 30m 13s | 0 | 311 |  |  |  |", text)
        self.assertEqual(text.count("| Run ID | Agent |"), 1)

    def test_results_summary_row_includes_openrouter_usage_when_present(self) -> None:
        final = FinalResult(
            run_id="run-1",
            best_score=3,
            best_round=1,
            total_rounds=1,
            stop_reason="stale_limit",
            stop_detail="done",
            total_wall_seconds=10,
            agent_output_chars=50,
            code_lines_added=2,
            log_dir=Path("/tmp/logs"),
            workspace=Path("/tmp/workspace"),
            openrouter_usage={
                "calls": 4,
                "cost_usd": 0.0123456,
                "total_tokens": 1234,
            },
        )
        config_path = Path(__file__).resolve().parents[1] / "config.openrouter.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        row = _results_summary_row(final, config)

        self.assertIn("| 4 | $0.012346 | 1234 |", row)

    def test_results_summary_migration_preserves_trailing_newline(self) -> None:
        old_table = (
            "| Run ID | Agent | Best Score | Best Round | Rounds | Stop Reason | Timeout | Wall Time | Agent Chars | Code Lines Added |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )

        migrated = _migrate_results_summary_effort_column(old_table)

        self.assertTrue(migrated.endswith("\n"))

    def test_render_command_supports_config_dir_placeholder(self) -> None:
        runner = Runner(self.config)

        command = runner._render_command(["{config_dir}/scripts/tool", "{run_id}"], Path("/tmp/round"))

        self.assertEqual(command[0], str(self.config.config_path.parent / "scripts/tool"))
        self.assertEqual(command[1], "test-run")

    def test_claude_effort_is_added_to_agent_command(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.claude.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        runner = Runner(config)

        command = runner._agent_command(Path("/tmp/round"))

        self.assertIn("--effort", command)
        self.assertIn("xhigh", command)

    def test_agent_effort_is_not_duplicated(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.claude.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        command = _apply_agent_effort(config, ["claude", "--effort", "high"])

        self.assertEqual(command, ["claude", "--effort", "high"])

    def test_gemini_model_is_added_to_agent_command(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.gemini.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        runner = Runner(config)

        command = runner._agent_command(Path("/tmp/round"))

        self.assertIn("--model", command)
        self.assertIn("gemini-3.1-pro-preview", command)

    def test_gemini_model_is_not_duplicated(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.gemini.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        command = _apply_agent_model(config, ["gemini", "--model", "flash"])

        self.assertEqual(command, ["gemini", "--model", "flash"])

    def test_gemini_model_is_added_for_named_gemini_backend(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.gemini.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        config = dataclasses.replace(
            config,
            agent=dataclasses.replace(config.agent, backend="gemini-3.1-pro-preview"),
        )

        command = _apply_agent_model(config, ["gemini"])

        self.assertEqual(command, ["gemini", "--model", "gemini-3.1-pro-preview"])

    def test_claude_result_line_is_terminal_even_when_error(self) -> None:
        self.assertTrue(
            _is_terminal_claude_result_line(
                '{"type":"result","subtype":"success","is_error":false}\n'
            )
        )
        self.assertTrue(
            _is_terminal_claude_result_line(
                '{"type":"result","subtype":"success","is_error":true}\n'
            )
        )
        self.assertFalse(_is_terminal_claude_result_line('{"type":"assistant"}\n'))
        self.assertFalse(_is_terminal_claude_result_line("not json\n"))

    def test_claude_error_result_is_detected_from_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "agent.stdout.log"
            stdout.write_text(
                '{"type":"assistant","message":{"content":[]}}\n'
                '{"type":"result","subtype":"success","is_error":true}\n',
                encoding="utf-8",
            )

            self.assertTrue(_claude_stdout_has_error_result(stdout))

    def test_codex_effort_is_read_from_command_config(self) -> None:
        self.assertEqual(_agent_effort_text(self.config), "xhigh")

    def test_agent_config_error_is_not_retryable(self) -> None:
        result = CommandResult(
            argv=["openrouter-api"],
            cwd=Path("/tmp"),
            returncode=AGENT_CONFIG_ERROR_RETURN_CODE,
            elapsed_seconds=0.1,
            timed_out=False,
            timeout_reason=None,
            stdout_path=Path("/tmp/stdout.log"),
            stderr_path=Path("/tmp/stderr.log"),
        )

        self.assertFalse(_agent_result_is_retryable(result))

    def test_agent_max_steps_is_not_retryable(self) -> None:
        result = CommandResult(
            argv=["openrouter-api"],
            cwd=Path("/tmp"),
            returncode=AGENT_MAX_STEPS_RETURN_CODE,
            elapsed_seconds=0.1,
            timed_out=False,
            timeout_reason="max_steps",
            stdout_path=Path("/tmp/stdout.log"),
            stderr_path=Path("/tmp/stderr.log"),
        )

        self.assertFalse(_agent_result_is_retryable(result))


if __name__ == "__main__":
    unittest.main()
