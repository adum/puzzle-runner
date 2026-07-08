import contextlib
import dataclasses
import io
import json
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from puzzle_runner.config import load_config
from puzzle_runner.openrouter_agent import AGENT_CONFIG_ERROR_RETURN_CODE, AGENT_MAX_STEPS_RETURN_CODE
from puzzle_runner.process import CommandResult
from puzzle_runner.runner import (
    FinalResult,
    Runner,
    _agent_effort_text,
    _agent_error_detail,
    _agent_result_is_retryable,
    _agent_stdout_completion_predicate,
    _apply_agent_effort,
    _apply_agent_model,
    _claude_stdout_has_error_result,
    _ensure_results_summary_header,
    _agent_model_not_found_error,
    _agent_model_not_found_detail,
    _is_terminal_claude_result_line,
    _migrate_results_summary_effort_column,
    _opencode_stdout_has_error_result,
    _opencode_progress_line,
    _opencode_auth_problem_from_listing,
    _opencode_model_problem_from_listing,
    _openrouter_usage_for_result,
    _results_summary_row,
    _should_append_results_summary,
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

    def test_explain_agent_model_not_found_says_evaluation_was_skipped(self) -> None:
        detail = explain_stop_reason(
            "agent_model_not_found",
            self.config,
            {"last_agent_model_not_found_model": "openrouter/moonshotai/kimi-k2.6"},
        )

        self.assertIn("model-not-found", detail)
        self.assertIn("openrouter/moonshotai/kimi-k2.6", detail)
        self.assertIn("without running evaluation", detail)

    def test_explain_agent_auth_missing_uses_problem_detail(self) -> None:
        detail = explain_stop_reason(
            "agent_auth_missing",
            self.config,
            {"agent_auth_problem": {"detail": "OpenCode has no OpenRouter credential."}},
        )

        self.assertEqual(detail, "OpenCode has no OpenRouter credential.")

    def test_explain_retryable_agent_error_mentions_retry_window(self) -> None:
        detail = explain_stop_reason(
            "agent_error",
            self.config,
            {
                "agent_retry_count": 2,
                "agent_total_retry_count": 5,
                "last_agent_error": {
                    "detail": "OpenCode returned an error API status 502: provider unavailable",
                    "retryable": True,
                },
            },
        )

        self.assertIn("retried eligible failures", detail)
        self.assertIn("2 retries", detail)
        self.assertIn("5 retries total this run", detail)
        self.assertIn("agent_failure_retry_limit_seconds=900s", detail)
        self.assertIn("without running evaluation", detail)

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

    def test_count_agent_output_chars_uses_opencode_text_not_json_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            round_dir = log_dir / "round-001"
            round_dir.mkdir()
            (round_dir / "agent.stdout.log").write_text(
                '{"type":"step_start","part":{"type":"step-start"}}\n'
                '{"type":"text","part":{"type":"text","text":"Hello"}}\n'
                '{"type":"tool_use","part":{"tool":"bash","state":{"status":"completed"}}}\n',
                encoding="utf-8",
            )
            (round_dir / "agent.attempt-002.stdout.log").write_text(
                '{"type":"text","part":{"type":"text","text":"Bye"}}\n',
                encoding="utf-8",
            )
            (round_dir / "agent.stderr.log").write_text("raw stderr ignored", encoding="utf-8")

            total = count_agent_output_chars(log_dir, agent_stream_format="opencode-json")

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
        self.assertIn("| run-1 | gpt-5.3-codex | codex | xhigh |", row)
        self.assertNotIn("/tmp/logs", row)

    def test_results_summary_row_records_grok_build_harness(self) -> None:
        final = FinalResult(
            run_id="run-1",
            best_score=1,
            best_round=1,
            total_rounds=1,
            stop_reason="stale_limit",
            stop_detail="done",
            total_wall_seconds=10,
            agent_output_chars=10,
            code_lines_added=1,
            log_dir=Path("/tmp/logs"),
            workspace=Path("/tmp/workspace"),
        )
        config_path = Path(__file__).resolve().parents[1] / "config.grok-build.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        row = _results_summary_row(final, config)

        self.assertIn("| run-1 | grok-composer-2.5-fast | grokbuild |", row)

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
        self.assertIn("| Run ID | Agent | Harness | Effort |", text)
        self.assertNotIn("Timeout | Logs |\n\n| Run ID | Agent | Best Score | Best Round | Rounds | Stop Reason | Timeout | Logs", text)

    def test_results_summary_header_migrates_no_effort_schema(self) -> None:
        old_table = (
            "| Run ID | Agent | Best Score | Best Round | Rounds | Stop Reason | Timeout | Wall Time | Agent Chars | Code Lines Added |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| run-1 | claude-fable-5 | 0 |  | 1 | agent_idle_timeout | 600s | 30m 13s | 0 | 311 |\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "final_results.md"
            path.write_text(old_table, encoding="utf-8")

            _ensure_results_summary_header(path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("| Run ID | Agent | Harness | Effort | Best Score |", text)
        self.assertIn("| run-1 | claude-fable-5 | claudecode |  | 0 |", text)
        self.assertIn("| run-1 | claude-fable-5 | claudecode |  | 0 |  | 1 | agent_idle_timeout | 600s | 30m 13s | 0 | 311 |  |  |  |", text)
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

    def test_results_summary_is_not_appended_without_evaluation(self) -> None:
        final = FinalResult(
            run_id="run-1",
            best_score=0,
            best_round=None,
            total_rounds=1,
            stop_reason="agent_failed",
            stop_detail="agent failed before evaluation",
            total_wall_seconds=10,
            agent_output_chars=100,
            code_lines_added=10,
            log_dir=Path("/tmp/logs"),
            workspace=Path("/tmp/workspace"),
        )

        self.assertFalse(_should_append_results_summary(final))

    def test_results_summary_is_appended_for_agent_error_after_evaluation(self) -> None:
        final = FinalResult(
            run_id="run-1",
            best_score=12,
            best_round=1,
            total_rounds=1,
            stop_reason="agent_error",
            stop_detail="agent failed after final evaluation",
            total_wall_seconds=10,
            agent_output_chars=100,
            code_lines_added=10,
            log_dir=Path("/tmp/logs"),
            workspace=Path("/tmp/workspace"),
        )

        self.assertTrue(_should_append_results_summary(final))

    def test_opencode_openrouter_usage_is_included_for_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            round_dir = log_dir / "round-001"
            round_dir.mkdir()
            (round_dir / "agent.stdout.log").write_text(
                '{"type":"step_finish","part":{"reason":"stop","cost":0.25,'
                '"tokens":{"total":123,"input":100,"output":7,"reasoning":3,'
                '"cache":{"read":13,"write":0}}}}\n',
                encoding="utf-8",
            )
            config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
            config = load_config(str(config_path), run_id="test-run")

            usage = _openrouter_usage_for_result(config, log_dir)

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage["calls"], 1)
        self.assertEqual(usage["total_tokens"], 123)
        self.assertEqual(usage["native_reasoning_tokens"], 3)
        self.assertAlmostEqual(usage["cost_usd"], 0.25)

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
        self.assertIn("Gemini 3.5 Flash (High)", command)

    def test_gemini_model_is_not_duplicated(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.gemini.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        command = _apply_agent_model(config, ["agy", "--model", "Gemini 3 Flash"])

        self.assertEqual(command, ["agy", "--model", "Gemini 3 Flash"])

    def test_gemini_model_is_added_for_named_gemini_backend(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.gemini-cli.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        config = dataclasses.replace(
            config,
            agent=dataclasses.replace(config.agent, backend="gemini-3.1-pro-preview"),
        )

        command = _apply_agent_model(config, ["gemini"])

        self.assertEqual(command, ["gemini", "--model", "gemini-3.1-pro-preview"])

    def test_opencode_model_and_effort_are_added_to_agent_command(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        runner = Runner(config)

        command = runner._agent_command(Path("/tmp/round"))

        self.assertIn("--model", command)
        self.assertIn("openrouter/google/gemini-3-flash-preview", command)
        self.assertIn("--variant", command)
        self.assertIn("high", command)

    def test_opencode_model_alias_is_not_duplicated(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        command = _apply_agent_model(config, ["opencode", "run", "-m", "openrouter/test"])

        self.assertEqual(command, ["opencode", "run", "-m", "openrouter/test"])

    def test_opencode_variant_is_not_duplicated(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        command = _apply_agent_effort(config, ["opencode", "run", "--variant", "max"])

        self.assertEqual(command, ["opencode", "run", "--variant", "max"])

    def test_grok_build_command_uses_prompt_file_and_model(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.grok-build.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        runner = Runner(config)

        command = runner._agent_command(Path("/tmp/round"))

        self.assertIn("--prompt-file", command)
        self.assertIn("/tmp/round/prompt.md", command)
        self.assertIn("--model", command)
        self.assertIn("composer-2.5-fast", command)

    def test_grok_build_effort_is_added_to_agent_command(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.grok-build.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        config = dataclasses.replace(config, agent=dataclasses.replace(config.agent, effort="high"))

        command = _apply_agent_effort(config, ["grok", "--prompt-file", "prompt.md"])

        self.assertEqual(command, ["grok", "--prompt-file", "prompt.md", "--effort", "high"])

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

    def test_opencode_error_event_is_detected_from_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "agent.stdout.log"
            stdout.write_text(
                '{"type":"text","part":{"text":"working"}}\n'
                '{"type":"error","error":{"message":"rate limit"}}\n',
                encoding="utf-8",
            )

            self.assertTrue(_opencode_stdout_has_error_result(stdout))

    def test_opencode_json_completion_predicate_detects_done_output(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        predicate = _agent_stdout_completion_predicate(config)

        self.assertIsNotNone(predicate)
        assert predicate is not None

        self.assertTrue(
            predicate(
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "type": "tool",
                            "tool": "bash",
                            "state": {
                                "status": "completed",
                                "input": {"command": 'echo "PUZZLE_RUNNER_DONE"'},
                                "output": "PUZZLE_RUNNER_DONE\n",
                                "metadata": {"output": "PUZZLE_RUNNER_DONE\n"},
                            },
                        },
                    }
                )
                + "\n"
            )
        )
        self.assertTrue(
            predicate(
                json.dumps(
                    {
                        "type": "text",
                        "part": {
                            "type": "text",
                            "text": "Final result:\nPUZZLE_RUNNER_DONE\n",
                        },
                    }
                )
                + "\n"
            )
        )
        self.assertFalse(
            predicate(
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "type": "tool",
                            "tool": "bash",
                            "state": {
                                "status": "completed",
                                "input": {"command": 'echo "PUZZLE_RUNNER_DONE"'},
                            },
                        },
                    }
                )
                + "\n"
            )
        )
        self.assertFalse(
            predicate(
                json.dumps(
                    {
                        "type": "text",
                        "part": {
                            "type": "text",
                            "text": "I will print PUZZLE_RUNNER_DONE next.",
                        },
                    }
                )
                + "\n"
            )
        )
        self.assertFalse(
            predicate(
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "type": "tool",
                            "tool": "read",
                            "state": {
                                "status": "completed",
                                "metadata": {"preview": "PUZZLE_RUNNER_DONE\n"},
                            },
                        },
                    }
                )
                + "\n"
            )
        )

    def test_opencode_progress_line_summarizes_human_readable_events(self) -> None:
        state: dict = {}

        self.assertEqual(
            _opencode_progress_line('{"type":"step_start"}\n', state),
            "--- OpenCode step 1 ---",
        )
        self.assertEqual(
            _opencode_progress_line(
                '{"type":"text","part":{"type":"text","text":"Working on it"}}\n',
                state,
            ),
            "OpenCode: Working on it",
        )
        self.assertEqual(
            _opencode_progress_line(
                '{"type":"tool_use","part":{"tool":"bash","state":{"status":"completed",'
                '"input":{"description":"Run tests","command":"pytest -q"},'
                '"metadata":{"exit":0,"output":"2 passed\\n"},'
                '"time":{"start":1000,"end":1250}}}}\n',
                state,
            ),
            "OpenCode tool: bash completed (250ms, exit 0) - Run tests; "
            "cmd: pytest -q; output 1 line, 9 chars: 2 passed",
        )
        self.assertEqual(
            _opencode_progress_line(
                '{"type":"tool_use","part":{"tool":"read","state":{"status":"completed",'
                '"input":{"filePath":"/tmp/run/levels_public/79"},'
                '"metadata":{"preview":"x=30&y=28&board=...","truncated":false},'
                '"time":{"start":1000,"end":1012}}}}\n',
                state,
            ),
            "OpenCode tool: read completed (12ms) - levels_public/79; preview 1 line, 19 chars",
        )
        self.assertEqual(
            _opencode_progress_line(
                '{"type":"tool_use","part":{"tool":"read","state":{"status":"error",'
                '"input":{"filePath":"/tmp/run/AGENTS.md"},'
                '"error":"File not found: /tmp/run/AGENTS.md",'
                '"time":{"start":1000,"end":1008}}}}\n',
                state,
            ),
            "OpenCode tool: read error (8ms) - AGENTS.md; error: File not found: /tmp/run/AGENTS.md",
        )
        self.assertEqual(
            _opencode_progress_line(
                '{"type":"tool_use","part":{"tool":"write","state":{"status":"completed",'
                '"input":{"filePath":"/tmp/run/solver.py","content":"print(1)\\nprint(2)\\n"},'
                '"time":{"start":1000,"end":2100}}}}\n',
                state,
            ),
            "OpenCode tool: write completed (1.10s) - solver.py; 2 lines, 18 chars",
        )
        self.assertEqual(
            _opencode_progress_line(
                '{"type":"tool_use","part":{"tool":"todowrite","state":{"status":"completed",'
                '"input":{"todos":[{"content":"Check levels","status":"in_progress"},'
                '{"content":"Write solver","status":"pending"}]},'
                '"title":"2 todos","time":{"start":1000,"end":1004}}}}\n',
                state,
            ),
            "OpenCode tool: todowrite completed (4ms); 2 todos; in_progress 1, pending 1; active: Check levels",
        )
        self.assertEqual(
            _opencode_progress_line(
                '{"type":"step_finish","part":{"reason":"tool-calls",'
                '"tokens":{"total":1234,"reasoning":56},"cost":0.0123}}\n',
                state,
            ),
            "OpenCode step finished: tool-calls, tokens 1,234, reasoning 56, cost $0.0123",
        )
        self.assertEqual(
            _opencode_progress_line(
                '{"type":"error","error":{"name":"UnknownError","data":{'
                '"message":"Model not found: openrouter/moonshotai/kimi-k2.6."}}}\n',
                state,
            ),
            "OpenCode error: Model not found: openrouter/moonshotai/kimi-k2.6.",
        )

    def test_opencode_auth_problem_detects_missing_openrouter_credentials(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        problem = _opencode_auth_problem_from_listing(
            config,
            "openrouter",
            "Credentials ~/.local/share/opencode/auth.json\n0 credentials\n",
            {},
        )

        self.assertIsNotNone(problem)
        assert problem is not None
        self.assertEqual(problem["provider"], "openrouter")
        self.assertIn("OPENROUTER_API_KEY", problem["detail"])

    def test_opencode_auth_problem_accepts_env_or_auth_list_credentials(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        self.assertIsNone(
            _opencode_auth_problem_from_listing(config, "openrouter", "0 credentials\n", {"OPENROUTER_API_KEY": "x"})
        )
        self.assertIsNone(
            _opencode_auth_problem_from_listing(config, "openrouter", "OpenRouter api\n1 credential\n", {})
        )

    def test_opencode_model_preflight_accepts_listed_model(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        problem = _opencode_model_problem_from_listing(
            config,
            provider="openrouter",
            listing="openrouter/google/gemini-3-flash-preview\nopenrouter/x-ai/grok-4.3\n",
            returncode=0,
            timed_out=False,
        )

        self.assertIsNone(problem)

    def test_opencode_model_preflight_reports_missing_model(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
        config = load_config(str(config_path), run_id="test-run")

        problem = _opencode_model_problem_from_listing(
            config,
            provider="openrouter",
            listing="openrouter/x-ai/grok-4.3\n",
            returncode=0,
            timed_out=False,
        )

        self.assertIsNotNone(problem)
        assert problem is not None
        self.assertEqual(problem["model"], "openrouter/google/gemini-3-flash-preview")
        self.assertIn("did not list configured model", problem["detail"])
        self.assertIn("before benchmark download/setup", problem["detail"])

    def test_agent_model_not_found_is_detected_from_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = root / "agent.stdout.log"
            stderr = root / "agent.stderr.log"
            stdout.write_text(
                '{"type":"error","error":{"data":{"message":"Model not found: x-ai/grok-4.3."}}}\n',
                encoding="utf-8",
            )
            stderr.write_text("ProviderModelNotFoundError\n", encoding="utf-8")

            self.assertTrue(_agent_model_not_found_error(stdout, stderr))
            self.assertEqual(_agent_model_not_found_detail(stdout, stderr), "x-ai/grok-4.3")

    def test_agent_model_not_found_model_is_detected_from_provider_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = root / "agent.stdout.log"
            stderr = root / "agent.stderr.log"
            stdout.write_text("", encoding="utf-8")
            stderr.write_text(
                'ProviderModelNotFoundError\n'
                ' data: {\n'
                '  providerID: "openrouter",\n'
                '  modelID: "moonshotai/kimi-k2.6",\n'
                '  suggestions: [],\n'
                '}\n',
                encoding="utf-8",
            )

            self.assertEqual(
                _agent_model_not_found_detail(stdout, stderr),
                "openrouter/moonshotai/kimi-k2.6",
            )

    def test_claude_auth_error_detail_is_explicit(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.claude.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = root / "agent.stdout.log"
            stderr = root / "agent.stderr.log"
            stdout.write_text(
                '{"type":"system","subtype":"api_retry","error_status":401,'
                '"error":"authentication_failed"}\n'
                '{"type":"result","is_error":true,"api_error_status":401,'
                '"result":"Failed to authenticate. API Error: 401 Invalid authentication credentials"}\n',
                encoding="utf-8",
            )
            stderr.write_text("", encoding="utf-8")

            detail = _agent_error_detail(config, stdout, stderr)

        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["kind"], "auth")
        self.assertIn("Claude Code authentication failed with API status 401", detail["detail"])
        self.assertIn("Invalid authentication credentials", detail["detail"])

    def test_model_not_found_stops_before_evaluation(self) -> None:
        class ModelNotFoundRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _ensure_solver_wrapper(self) -> None:
                pass

            def _run_agent(self, round_number: int, round_dir: Path, prompt: str) -> CommandResult:
                stdout = round_dir / "agent.stdout.log"
                stderr = round_dir / "agent.stderr.log"
                stdout.write_text(
                    '{"type":"error","error":{"data":{"message":"Model not found: x-ai/grok-4.3."}}}\n',
                    encoding="utf-8",
                )
                stderr.write_text("", encoding="utf-8")
                self._update_status(
                    latest={"agent_stdout": stdout, "agent_stderr": stderr},
                    last_agent_model_not_found=True,
                )
                return CommandResult(
                    argv=["opencode", "run"],
                    cwd=self.workspace,
                    returncode=0,
                    elapsed_seconds=0.1,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

            def _run_evaluation(self, round_dir: Path) -> CommandResult:
                raise AssertionError("evaluation should not run after model-not-found")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = dataclasses.replace(
                self.config,
                download_full_levels=False,
                build_checker=False,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
            )
            final = ModelNotFoundRunner(config).run()
            final_result_exists = (final.log_dir / "final_result.md").exists()

            self.assertEqual(final.stop_reason, "agent_model_not_found")
            self.assertEqual(final.best_score, 0)
            self.assertFalse(config.results_path.exists())
            self.assertTrue(final_result_exists)

    def test_claude_auth_error_stops_before_evaluation_and_prints_detail(self) -> None:
        class ClaudeAuthErrorRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)
                (self.workspace / "run_solver").write_text(
                    "#!/usr/bin/env sh\nexec python3 ./coil_solver.py\n",
                    encoding="utf-8",
                )
                (self.workspace / "coil_solver.py").write_text(
                    "print('default')\n",
                    encoding="utf-8",
                )

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _run_evaluation(self, round_dir: Path) -> CommandResult:
                raise AssertionError("evaluation should not run after agent auth error")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_claude = root / "fake_claude.py"
            fake_claude.write_text(
                "import json\n"
                "import sys\n"
                "print(json.dumps({'type':'system','subtype':'api_retry',"
                "'error_status':401,'error':'authentication_failed'}), flush=True)\n"
                "print(json.dumps({'type':'result','is_error':True,'api_error_status':401,"
                "'result':'Failed to authenticate. API Error: 401 Invalid authentication credentials'}), flush=True)\n"
                "sys.exit(1)\n",
                encoding="utf-8",
            )
            config = dataclasses.replace(
                self.config,
                download_full_levels=False,
                build_checker=False,
                max_rounds=3,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
                agent=dataclasses.replace(
                    self.config.agent,
                    name="claude-sonnet-5",
                    backend="claude-code",
                    command=[
                        "python3",
                        str(fake_claude),
                        "--output-format",
                        "stream-json",
                    ],
                    model="claude-sonnet-5",
                ),
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                final = ClaudeAuthErrorRunner(config).run()
            status_text = (config.status_dir / "status.md").read_text(encoding="utf-8")

        self.assertEqual(final.stop_reason, "agent_auth_error")
        self.assertEqual(final.best_score, 0)
        self.assertEqual(final.total_rounds, 1)
        self.assertIn("API status 401", final.stop_detail)
        self.assertIn("Invalid authentication credentials", final.stop_detail)
        self.assertIn("Puzzle Runner agent error:", output.getvalue())
        self.assertIn("Agent Error:", status_text)
        self.assertFalse((final.log_dir / "round-001" / "agent_attempt-002.json").exists())
        self.assertFalse(config.results_path.exists())

    def test_opencode_provider_unavailable_error_is_retryable(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = root / "agent.stdout.log"
            stderr = root / "agent.stderr.log"
            stdout.write_text(
                '{"type":"error","error":{"name":"UnknownError","data":{"message":'
                '"{\\"code\\":502,\\"message\\":\\"Exception: Response payload is not completed\\",'
                '\\"metadata\\":{\\"error_type\\":\\"provider_unavailable\\"}}"}}}\n',
                encoding="utf-8",
            )
            stderr.write_text("", encoding="utf-8")

            detail = _agent_error_detail(config, stdout, stderr)

        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["api_error_status"], 502)
        self.assertEqual(detail["error"], "provider_unavailable")
        self.assertTrue(detail["retryable"])
        self.assertIn("API status 502", detail["detail"])

    def test_retryable_opencode_error_uses_exponential_retry_path(self) -> None:
        class RetryableErrorRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)
                (self.workspace / "run_solver").write_text(
                    "#!/usr/bin/env sh\necho solved\n",
                    encoding="utf-8",
                )

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _ensure_solver_wrapper(self) -> None:
                pass

            def _agent_auth_preflight_problem(self) -> dict | None:
                return None

            def _opencode_model_preflight_problem(self) -> dict | None:
                return None

            def _can_shortcut_default_solver_evaluation(self) -> bool:
                return False

            def _run_evaluation(self, round_dir: Path) -> CommandResult:
                stdout = round_dir / "evaluation.stdout.log"
                stderr = round_dir / "evaluation.stderr.log"
                stdout.write_text(
                    "Level 77 (baseline): PASS (0.00s)\n"
                    "Level 78 (baseline): FAIL - test stop (0.00s)\n",
                    encoding="utf-8",
                )
                stderr.write_text("", encoding="utf-8")
                return CommandResult(
                    argv=["python3", "evaluate_full.py"],
                    cwd=self.workspace,
                    returncode=0,
                    elapsed_seconds=0.1,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            attempt_file = root / "attempts.txt"
            fake_opencode = root / "fake_opencode.py"
            fake_opencode.write_text(
                "import json\n"
                "import pathlib\n"
                "import sys\n"
                f"attempt_path = pathlib.Path({str(attempt_file)!r})\n"
                "attempt = int(attempt_path.read_text() or '0') + 1 if attempt_path.exists() else 1\n"
                "attempt_path.write_text(str(attempt))\n"
                "if attempt == 1:\n"
                "    message = json.dumps({'code': 502, 'message': 'Exception: Response payload is not completed', 'metadata': {'error_type': 'provider_unavailable'}})\n"
                "    print(json.dumps({'type': 'error', 'error': {'name': 'UnknownError', 'data': {'message': message}}}), flush=True)\n"
                "    sys.exit(1)\n"
                "print(json.dumps({'type': 'text', 'part': {'type': 'text', 'text': 'done'}}), flush=True)\n",
                encoding="utf-8",
            )
            config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
            opencode_config = load_config(str(config_path), run_id="test-run")
            config = dataclasses.replace(
                opencode_config,
                download_full_levels=False,
                build_checker=False,
                max_rounds=1,
                agent_failure_retry_limit_seconds=60,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
                agent=dataclasses.replace(
                    opencode_config.agent,
                    command=["python3", str(fake_opencode), "--format", "json"],
                ),
            )

            output = io.StringIO()
            with mock.patch("puzzle_runner.runner.time.sleep") as sleep:
                with contextlib.redirect_stdout(output):
                    final = RetryableErrorRunner(config).run()

            attempts_text = attempt_file.read_text()
            events_text = (final.log_dir / "events.jsonl").read_text(encoding="utf-8")
            second_attempt_exists = (final.log_dir / "round-001" / "agent_attempt-002.json").exists()

        self.assertEqual(final.best_score, 77)
        self.assertEqual(attempts_text, "2")
        self.assertIn(mock.call(5.0), sleep.call_args_list)
        self.assertTrue(second_attempt_exists)
        self.assertIn("agent_retry_scheduled", events_text)

    def test_auth_preflight_problem_stops_before_agent(self) -> None:
        class AuthMissingRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _ensure_solver_wrapper(self) -> None:
                pass

            def _agent_auth_preflight_problem(self) -> dict | None:
                return {
                    "provider": "openrouter",
                    "model": "openrouter/test-model",
                    "env_var": "OPENROUTER_API_KEY",
                    "detail": "OpenCode has no OpenRouter credential.",
                }

            def _run_agent(self, round_number: int, round_dir: Path, prompt: str) -> CommandResult:
                raise AssertionError("agent should not run when auth preflight fails")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = dataclasses.replace(
                self.config,
                download_full_levels=False,
                build_checker=False,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
            )
            final = AuthMissingRunner(config).run()

        self.assertEqual(final.stop_reason, "agent_auth_missing")
        self.assertEqual(final.total_rounds, 0)
        self.assertIn("OpenRouter credential", final.stop_detail)

    def test_opencode_model_preflight_stops_before_workspace_setup(self) -> None:
        class MissingModelRunner(Runner):
            def _opencode_model_preflight_problem(self) -> dict | None:
                return {
                    "provider": "openrouter",
                    "model": "openrouter/test-missing",
                    "detail": "OpenCode models did not list configured model `openrouter/test-missing`.",
                }

            def _prepare_workspace(self) -> None:
                raise AssertionError("workspace should not be prepared when model preflight fails")

            def _run_agent(self, round_number: int, round_dir: Path, prompt: str) -> CommandResult:
                raise AssertionError("agent should not run when model preflight fails")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"
            opencode_config = load_config(str(config_path), run_id="test-run")
            config = dataclasses.replace(
                opencode_config,
                download_full_levels=False,
                build_checker=False,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
            )
            final = MissingModelRunner(config).run()

            self.assertEqual(final.stop_reason, "agent_model_not_found")
            self.assertEqual(final.total_rounds, 0)
            self.assertEqual(final.best_score, 0)
            self.assertIn("did not list configured model", final.stop_detail)
            self.assertFalse(final.workspace.exists())
            self.assertFalse(config.results_path.exists())
            self.assertTrue((final.log_dir / "final_result.md").exists())

    def test_agent_idle_timeout_runs_final_evaluation(self) -> None:
        class IdleTimeoutRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)
                (self.workspace / "run_solver").write_text(
                    "#!/usr/bin/env sh\necho solved\n",
                    encoding="utf-8",
                )

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _ensure_solver_wrapper(self) -> None:
                pass

            def _can_shortcut_default_solver_evaluation(self) -> bool:
                return False

            def _run_agent(self, round_number: int, round_dir: Path, prompt: str) -> CommandResult:
                stdout = round_dir / "agent.stdout.log"
                stderr = round_dir / "agent.stderr.log"
                stdout.write_text("partial work before idle\n", encoding="utf-8")
                stderr.write_text("", encoding="utf-8")
                return CommandResult(
                    argv=["agent"],
                    cwd=self.workspace,
                    returncode=-9,
                    elapsed_seconds=1800.0,
                    timed_out=True,
                    timeout_reason="idle",
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

            def _run_evaluation(self, round_dir: Path) -> CommandResult:
                stdout = round_dir / "evaluation.stdout.log"
                stderr = round_dir / "evaluation.stderr.log"
                stdout.write_text(
                    "Level 123 (baseline): PASS (0.00s)\n"
                    "Level 124 (baseline): FAIL - test stop (0.00s)\n",
                    encoding="utf-8",
                )
                stderr.write_text("", encoding="utf-8")
                return CommandResult(
                    argv=["python3", "evaluate_full.py"],
                    cwd=self.workspace,
                    returncode=0,
                    elapsed_seconds=0.1,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = dataclasses.replace(
                self.config,
                download_full_levels=False,
                build_checker=False,
                max_rounds=1,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
            )
            final = IdleTimeoutRunner(config).run()
            evaluation_result_exists = (
                final.log_dir / "round-001" / "evaluation_result.json"
            ).exists()

        self.assertEqual(final.stop_reason, "agent_idle_timeout")
        self.assertEqual(final.best_score, 123)
        self.assertEqual(final.best_round, 1)
        self.assertEqual(final.total_rounds, 1)
        self.assertTrue(evaluation_result_exists)

    def test_agent_failure_runs_final_evaluation(self) -> None:
        class AgentFailedRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)
                (self.workspace / "run_solver").write_text(
                    "#!/usr/bin/env sh\necho solved\n",
                    encoding="utf-8",
                )

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _ensure_solver_wrapper(self) -> None:
                pass

            def _can_shortcut_default_solver_evaluation(self) -> bool:
                return False

            def _run_agent(self, round_number: int, round_dir: Path, prompt: str) -> CommandResult:
                stdout = round_dir / "agent.stdout.log"
                stderr = round_dir / "agent.stderr.log"
                stdout.write_text("work completed before failure\n", encoding="utf-8")
                stderr.write_text("Error: max turns reached\n", encoding="utf-8")
                (self.workspace / "solver.py").write_text("print('changed')\n", encoding="utf-8")
                return CommandResult(
                    argv=["agent"],
                    cwd=self.workspace,
                    returncode=1,
                    elapsed_seconds=10.0,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

            def _run_evaluation(self, round_dir: Path) -> CommandResult:
                stdout = round_dir / "evaluation.stdout.log"
                stderr = round_dir / "evaluation.stderr.log"
                stdout.write_text(
                    "Level 88 (baseline): PASS (0.00s)\n"
                    "Level 89 (baseline): FAIL - test stop (0.00s)\n",
                    encoding="utf-8",
                )
                stderr.write_text("", encoding="utf-8")
                return CommandResult(
                    argv=["python3", "evaluate_full.py"],
                    cwd=self.workspace,
                    returncode=0,
                    elapsed_seconds=0.1,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = dataclasses.replace(
                self.config,
                download_full_levels=False,
                build_checker=False,
                max_rounds=1,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
            )
            final = AgentFailedRunner(config).run()
            evaluation_result_exists = (
                final.log_dir / "round-001" / "evaluation_result.json"
            ).exists()

        self.assertEqual(final.stop_reason, "agent_failed")
        self.assertEqual(final.best_score, 88)
        self.assertEqual(final.best_round, 1)
        self.assertEqual(final.total_rounds, 1)
        self.assertTrue(evaluation_result_exists)

    def test_retryable_agent_error_runs_final_evaluation(self) -> None:
        class RetryableAgentErrorRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)
                (self.workspace / "run_solver").write_text(
                    "#!/usr/bin/env sh\necho solved\n",
                    encoding="utf-8",
                )

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _ensure_solver_wrapper(self) -> None:
                pass

            def _can_shortcut_default_solver_evaluation(self) -> bool:
                return False

            def _run_agent(self, round_number: int, round_dir: Path, prompt: str) -> CommandResult:
                stdout = round_dir / "agent.stdout.log"
                stderr = round_dir / "agent.stderr.log"
                stdout.write_text("partial work before rate limit\n", encoding="utf-8")
                stderr.write_text("API status 429: rate limit exceeded\n", encoding="utf-8")
                (self.workspace / "solver.py").write_text("print('changed')\n", encoding="utf-8")
                self._update_status(
                    agent_retry_count=4,
                    agent_total_retry_count=6,
                    agent_error_count=7,
                    last_agent_returned_error=True,
                    last_agent_error={
                        "detail": "OpenCode returned an error API status 429: rate limit exceeded",
                        "retryable": True,
                        "kind": "error",
                    },
                    last_agent_error_stop_reason="agent_error",
                )
                return CommandResult(
                    argv=["agent"],
                    cwd=self.workspace,
                    returncode=1,
                    elapsed_seconds=10.0,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

            def _run_evaluation(self, round_dir: Path) -> CommandResult:
                stdout = round_dir / "evaluation.stdout.log"
                stderr = round_dir / "evaluation.stderr.log"
                stdout.write_text(
                    "Level 77 (baseline): PASS (0.00s)\n"
                    "Level 78 (baseline): FAIL - test stop (0.00s)\n",
                    encoding="utf-8",
                )
                stderr.write_text("", encoding="utf-8")
                return CommandResult(
                    argv=["python3", "evaluate_full.py"],
                    cwd=self.workspace,
                    returncode=0,
                    elapsed_seconds=0.1,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = dataclasses.replace(
                self.config,
                download_full_levels=False,
                build_checker=False,
                max_rounds=1,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
            )
            final = RetryableAgentErrorRunner(config).run()
            evaluation_result_exists = (
                final.log_dir / "round-001" / "evaluation_result.json"
            ).exists()
            final_result_text = (final.log_dir / "final_result.md").read_text(
                encoding="utf-8"
            )
            results_summary_text = config.results_path.read_text(encoding="utf-8")

        self.assertEqual(final.stop_reason, "agent_error")
        self.assertEqual(final.best_score, 77)
        self.assertEqual(final.best_round, 1)
        self.assertEqual(final.total_rounds, 1)
        self.assertIn("4 retries", final.stop_detail)
        self.assertIn("6 retries total this run", final.stop_detail)
        self.assertIn("ran final evaluation", final.stop_detail)
        self.assertIn("Agent retry count: 4", final_result_text)
        self.assertIn("Agent total retry count: 6", final_result_text)
        self.assertIn("Agent error count: 7", final_result_text)
        self.assertIn("| test-run |", results_summary_text)
        self.assertIn("| agent_error |", results_summary_text)
        self.assertTrue(evaluation_result_exists)

    def test_unchanged_default_solver_shortcuts_full_evaluation(self) -> None:
        class NoChangeRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)
                (self.workspace / "run_solver").write_text(
                    "#!/usr/bin/env sh\nexec python3 ./coil_solver.py\n",
                    encoding="utf-8",
                )
                (self.workspace / "coil_solver.py").write_text(
                    "print('default')\n",
                    encoding="utf-8",
                )

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _run_agent(self, round_number: int, round_dir: Path, prompt: str) -> CommandResult:
                stdout = round_dir / "agent.stdout.log"
                stderr = round_dir / "agent.stderr.log"
                stdout.write_text("PUZZLE_RUNNER_DONE\n", encoding="utf-8")
                stderr.write_text("", encoding="utf-8")
                return CommandResult(
                    argv=["agent"],
                    cwd=self.workspace,
                    returncode=0,
                    elapsed_seconds=0.1,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

            def _run_evaluation(self, round_dir: Path) -> CommandResult:
                raise AssertionError("unchanged default solver should use shortcut")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = dataclasses.replace(
                self.config,
                download_full_levels=False,
                build_checker=False,
                max_rounds=1,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
            )
            final = NoChangeRunner(config).run()
            stdout = (
                final.log_dir / "round-001" / "evaluation.stdout.log"
            ).read_text(encoding="utf-8")
            result_json = (
                final.log_dir / "round-001" / "evaluation_result.json"
            ).read_text(encoding="utf-8")

        self.assertEqual(final.best_score, 47)
        self.assertEqual(final.best_round, 1)
        self.assertIn("skipped evaluate_full.py", stdout)
        self.assertIn("evaluation-shortcut", result_json)

    def test_default_solver_shortcut_is_disabled_after_solver_override(self) -> None:
        class SolverOverrideRunner(Runner):
            def _prepare_workspace(self) -> None:
                self.workspace.mkdir(parents=True)
                (self.workspace / "run_solver").write_text(
                    "#!/usr/bin/env sh\n"
                    "if [ -f ./solver.py ]; then exec python3 ./solver.py; fi\n"
                    "exec python3 ./coil_solver.py\n",
                    encoding="utf-8",
                )
                (self.workspace / "coil_solver.py").write_text(
                    "print('default')\n",
                    encoding="utf-8",
                )

            def _normalize_workspace_line_endings(self) -> None:
                pass

            def _run_agent(self, round_number: int, round_dir: Path, prompt: str) -> CommandResult:
                stdout = round_dir / "agent.stdout.log"
                stderr = round_dir / "agent.stderr.log"
                stdout.write_text("PUZZLE_RUNNER_DONE\n", encoding="utf-8")
                stderr.write_text("", encoding="utf-8")
                (self.workspace / "solver.py").write_text("print('override')\n", encoding="utf-8")
                return CommandResult(
                    argv=["agent"],
                    cwd=self.workspace,
                    returncode=0,
                    elapsed_seconds=0.1,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

            def _run_evaluation(self, round_dir: Path) -> CommandResult:
                stdout = round_dir / "evaluation.stdout.log"
                stderr = round_dir / "evaluation.stderr.log"
                stdout.write_text(
                    "Level 99 (baseline): PASS (0.00s)\n"
                    "Level 100 (baseline): FAIL - test stop (0.00s)\n",
                    encoding="utf-8",
                )
                stderr.write_text("", encoding="utf-8")
                return CommandResult(
                    argv=["python3", "evaluate_full.py"],
                    cwd=self.workspace,
                    returncode=0,
                    elapsed_seconds=0.1,
                    timed_out=False,
                    timeout_reason=None,
                    stdout_path=stdout,
                    stderr_path=stderr,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = dataclasses.replace(
                self.config,
                download_full_levels=False,
                build_checker=False,
                max_rounds=1,
                worktree_root=root / "worktrees",
                log_root=root / "logs",
                status_dir=root / "current",
                results_path=root / "final_results.md",
            )
            final = SolverOverrideRunner(config).run()

        self.assertEqual(final.best_score, 99)

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
