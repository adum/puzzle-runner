import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from puzzle_runner.watch import (
    CLEAR_SCREEN,
    WorkspaceChangeCache,
    _draw_frame,
    _workspace_change_summary,
    render_status,
    resolve_status_path,
)


class WatchTests(unittest.TestCase):
    def test_render_status_without_color(self) -> None:
        agent_started_at = (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        status = {
            "active": True,
            "phase": "agent_running",
            "phase_started_at": agent_started_at,
            "agent_started_at": agent_started_at,
            "run_id": "run-1",
            "agent": "codex",
            "agent_effort": "xhigh",
            "current_round": 2,
            "max_rounds": 5,
            "best_score": 47,
            "best_round": 1,
            "last_score": 47,
            "score_history": [35, 47],
            "last_improved": False,
            "agent_error_count": 2,
            "last_agent_returned_error": True,
            "stale_count": 1,
            "stale_limit": 3,
            "remaining_no_progress_tries": 2,
            "elapsed_seconds": 65.2,
            "workspace": "/tmp/workspace",
            "log_dir": "/tmp/logs",
            "latest": {"agent_stdout": "/tmp/agent.stdout.log"},
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Puzzle Runner", rendered)
        self.assertIn("[ACTIVE]", rendered)
        self.assertIn("agent_running", rendered)
        self.assertIn("Agent running", rendered)
        self.assertIn("Effort", rendered)
        self.assertIn("xhigh", rendered)
        self.assertIn("47", rendered)
        self.assertIn("Scores", rendered)
        self.assertIn("35, 47", rendered)
        self.assertIn("Agent errors", rendered)
        self.assertIn("2 (last turn error)", rendered)
        self.assertIn("Remaining tries", rendered)
        self.assertNotIn("Paths", rendered)
        self.assertNotIn("Latest Round Files", rendered)
        self.assertNotIn("/tmp/workspace", rendered)
        self.assertNotIn("/tmp/logs", rendered)
        self.assertNotIn("agent.stdout.log", rendered)
        self.assertNotIn("\033[", rendered)

    def test_render_status_includes_openrouter_max_token_hits(self) -> None:
        status = {
            "active": True,
            "phase": "agent_running",
            "backend": "openrouter",
            "openrouter_max_tokens_count": 2,
            "last_openrouter_max_tokens_step": 14,
            "last_openrouter_max_tokens_max_tokens": 16384,
            "last_openrouter_max_tokens_completion_tokens": 16384,
            "last_openrouter_max_tokens_reasoning_tokens": 14000,
            "latest": {},
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("OR max tokens", rendered)
        self.assertIn("2 (last step 14, limit 16,384, out 16,384, reason 14,000)", rendered)

    def test_render_status_includes_agent_output_chars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "agent.stdout.log"
            stderr = Path(temp_dir) / "agent.stderr.log"
            stdout.write_text("abc", encoding="utf-8")
            stderr.write_text("defgh", encoding="utf-8")
            agent_started_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            status = {
                "active": True,
                "phase": "agent_running",
                "agent_started_at": agent_started_at,
                "latest": {
                    "agent_stdout": str(stdout),
                    "agent_stderr": str(stderr),
                },
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Agent output", rendered)
        self.assertIn("8 chars", rendered)
        self.assertIn("/min", rendered)
        self.assertIn("Last output", rendered)

    def test_render_status_uses_live_agent_output_status_when_newer(self) -> None:
        agent_started_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        agent_last_output_at = (datetime.now(timezone.utc) - timedelta(seconds=3)).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        status = {
            "active": True,
            "phase": "agent_running",
            "agent_started_at": agent_started_at,
            "agent_output_chars_live": 22749,
            "agent_last_output_at": agent_last_output_at,
            "latest": {},
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Agent output", rendered)
        self.assertIn("22,749 chars", rendered)
        self.assertIn("Last output", rendered)

    def test_render_status_active_elapsed_uses_started_at(self) -> None:
        started_at = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        status = {
            "active": True,
            "phase": "agent_running",
            "started_at": started_at,
            "elapsed_seconds": 1,
            "latest": {},
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Elapsed", rendered)
        self.assertIn("2m", rendered)

    def test_render_status_includes_last_tested_puzzle_from_agent_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stderr = Path(temp_dir) / "agent.stderr.log"
            stderr.write_text(
                "Level 73 (27x27): PASS (7.51s)\n"
                "Level 75 (29x27): PASS (37.22s)\n"
                "Level 77 (28x28): PASS (2.09s)\n"
                "Level 79 (30x28): FAIL (No solution found) (58.03s)\n",
                encoding="utf-8",
            )
            status = {
                "active": True,
                "phase": "agent_running",
                "latest": {"agent_stderr": str(stderr)},
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Last tested", rendered)
        self.assertIn("Level 79 (30x28): FAIL (No solution found) (58.03s)", rendered)

    def test_render_status_summarizes_claude_stream_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "agent.stdout.log"
            stdout.write_text(
                '{"type":"system","subtype":"init"}\n'
                '{"type":"stream_event","event":{"type":"content_block_delta",'
                '"delta":{"type":"text_delta","text":"Level 79 (30x28): FAIL (No solution found) (58.03s)"}}}\n'
                '{"type":"stream_event","event":{"type":"message_delta",'
                '"usage":{"input_tokens":2,"output_tokens":11,'
                '"cache_read_input_tokens":11631,"cache_creation_input_tokens":5813}}}\n'
                '{"type":"result","subtype":"success","is_error":false,'
                '"result":"Level 79 (30x28): FAIL (No solution found) (58.03s)",'
                '"total_cost_usd":0.02587405,'
                '"usage":{"input_tokens":2,"output_tokens":11,'
                '"cache_read_input_tokens":11631,"cache_creation_input_tokens":5813}}\n',
                encoding="utf-8",
            )
            status = {
                "active": True,
                "phase": "agent_running",
                "backend": "claude-code",
                "agent_stream_format": "claude-stream-json",
                "latest": {"agent_stdout": str(stdout)},
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Claude stream", rendered)
        self.assertIn("4 recent events", rendered)
        self.assertIn("latest result success", rendered)
        self.assertIn("Claude text", rendered)
        self.assertIn("Claude usage", rendered)
        self.assertIn("in 2, out 11, cache read 11,631, cache write 5,813, cost $0.0259", rendered)
        self.assertIn("Last tested", rendered)
        self.assertIn("Level 79 (30x28): FAIL (No solution found) (58.03s)", rendered)

    def test_render_status_summarizes_gemini_stream_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "agent.stdout.log"
            stdout.write_text(
                '{"type":"init","model":"gemini-3.1-pro-preview"}\n'
                '{"type":"message","role":"assistant","content":"Level 79 (30x28): FAIL (No solution found) (58.03s)","delta":true}\n'
                '{"type":"result","status":"success","stats":{"input_tokens":9513,'
                '"output_tokens":1,"total_tokens":9607,"cached":0,"tool_calls":0,'
                '"duration_ms":3283}}\n',
                encoding="utf-8",
            )
            status = {
                "active": True,
                "phase": "agent_running",
                "backend": "gemini-cli",
                "agent_stream_format": "gemini-stream-json",
                "latest": {"agent_stdout": str(stdout)},
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Gemini stream", rendered)
        self.assertIn("3 recent events", rendered)
        self.assertIn("latest result success", rendered)
        self.assertIn("Gemini text", rendered)
        self.assertIn("Gemini usage", rendered)
        self.assertIn("in 9,513, out 1, total 9,607, cached 0, tools 0, time 3.3s", rendered)
        self.assertIn("Last tested", rendered)
        self.assertIn("Level 79 (30x28): FAIL (No solution found) (58.03s)", rendered)

    def test_render_status_summarizes_opencode_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "agent.stdout.log"
            stdout.write_text(
                '{"type":"step_start","timestamp":1,"sessionID":"ses_1",'
                '"part":{"type":"step-start"}}\n'
                '{"type":"text","timestamp":2,"sessionID":"ses_1",'
                '"part":{"type":"text","text":"Level 79 (30x28): FAIL (No solution found) (58.03s)"}}\n'
                '{"type":"tool_use","timestamp":3,"sessionID":"ses_1",'
                '"part":{"tool":"bash","state":{"status":"completed"}}}\n'
                '{"type":"step_finish","timestamp":4,"sessionID":"ses_1",'
                '"part":{"type":"step-finish","reason":"stop","cost":0.0123,'
                '"tokens":{"input":100,"output":25,"reasoning":5,"total":130,'
                '"cache":{"read":7,"write":3}}}}\n',
                encoding="utf-8",
            )
            status = {
                "active": True,
                "phase": "agent_running",
                "backend": "opencode",
                "agent_stream_format": "opencode-json",
                "latest": {"agent_stdout": str(stdout)},
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("OpenCode stream", rendered)
        self.assertIn("4 recent events", rendered)
        self.assertIn("latest step finish stop", rendered)
        self.assertIn("OpenCode text", rendered)
        self.assertIn("OpenCode usage", rendered)
        self.assertIn("in 100, out 25, reason 5, total 130, cache read 7, cache write 3, cost $0.0123", rendered)
        self.assertIn("Last tested", rendered)
        self.assertIn("Level 79 (30x28): FAIL (No solution found) (58.03s)", rendered)

    def test_render_status_summarizes_openrouter_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            round_dir = log_dir / "round-001"
            round_dir.mkdir()
            (round_dir / "openrouter-response-001.json").write_text(
                '{"id":"gen-1","usage":{"prompt_tokens":100,"completion_tokens":25,"total_tokens":125},'
                '"choices":[{"finish_reason":"stop"}]}\n',
                encoding="utf-8",
            )
            (round_dir / "openrouter-generation-001.json").write_text(
                '{"data":{"id":"gen-1","total_cost":0.012345,"provider_name":"Infermatic",'
                '"native_tokens_reasoning":7,"native_tokens_cached":11,"latency":1500,'
                '"finish_reason":"stop"}}\n',
                encoding="utf-8",
            )
            status = {
                "active": True,
                "phase": "agent_running",
                "backend": "openrouter",
                "log_dir": str(log_dir),
                "latest": {},
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("OpenRouter usage", rendered)
        self.assertIn("1 call", rendered)
        self.assertIn("$0.012345", rendered)
        self.assertIn("in 100", rendered)
        self.assertIn("out 25", rendered)
        self.assertIn("reason 7", rendered)
        self.assertIn("Infermatic", rendered)

    def test_render_status_summarizes_final_opencode_openrouter_usage(self) -> None:
        status = {
            "backend": "opencode",
            "openrouter_usage": {
                "calls": 2,
                "cost_usd": 0.25,
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 18,
            },
            "latest": {},
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("OpenRouter usage", rendered)
        self.assertIn("2 calls", rendered)
        self.assertIn("$0.250000", rendered)
        self.assertIn("in 10", rendered)
        self.assertIn("out 3", rendered)
        self.assertIn("total 18", rendered)

    def test_render_status_includes_evaluation_level_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "evaluation.stdout.log"
            stdout.write_text(
                "Level 1 (5x5): PASS (0.02s)\nLevel 3 (10x10): ",
                encoding="utf-8",
            )
            status = {
                "active": True,
                "phase": "evaluation_running",
                "latest": {"evaluation_stdout": str(stdout)},
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Evaluating", rendered)
        self.assertIn("level 3 (10x10)", rendered)
        self.assertIn("latest pass 1", rendered)
        self.assertIn("passed 1", rendered)

    def test_render_status_includes_last_evaluation_summary(self) -> None:
        status = {
            "active": True,
            "phase": "agent_running",
            "last_score": 119,
            "first_failing_level": 120,
            "stop_status": "TIMEOUT",
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Last eval", rendered)
        self.assertIn("passed 119, timeout at 120", rendered)

    def test_render_status_includes_last_eval_times_and_failure_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "evaluation.stdout.log"
            stdout.write_text(
                "Level 149 (31x31): PASS (1.00s)\n"
                "Level 151 (32x32): PASS (12.34s)\n"
                "Level 152 (32x33): FAIL (No solution found) (58.03s)\n",
                encoding="utf-8",
            )
            status = {
                "active": True,
                "phase": "agent_running",
                "last_score": 151,
                "first_failing_level": 152,
                "stop_status": "FAIL",
                "latest": {"evaluation_stdout": str(stdout)},
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Last eval", rendered)
        self.assertIn("passed 151 (12.34s), fail at 152 (No solution found, 58.03s)", rendered)

    def test_render_status_includes_last_eval_timeout_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = Path(temp_dir) / "evaluation.stdout.log"
            stdout.write_text(
                "Level 119 (30x30): PASS (3.21s)\n"
                "Level 120 (31x30): TIMEOUT - Exceeded 600.0s limit (600.11s)\n",
                encoding="utf-8",
            )
            status = {
                "active": True,
                "phase": "agent_running",
                "latest": {"evaluation_stdout": str(stdout)},
            }

            rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("passed 119 (3.21s), timeout at 120 (Exceeded 600.0s limit, 600.11s)", rendered)

    def test_render_status_includes_retry_countdown(self) -> None:
        phase_started_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        status = {
            "active": True,
            "phase": "agent_retry_wait",
            "phase_started_at": phase_started_at,
            "agent_attempt": 2,
            "agent_retry_delay_seconds": 40,
            "agent_retry_remaining_seconds": 120,
        }

        rendered = render_status(status, status_path=Path("/tmp/status.json"), color=False)

        self.assertIn("Retrying in", rendered)
        self.assertIn("next attempt 3", rendered)
        self.assertIn("budget", rendered)

    def test_render_status_includes_workspace_change_summary(self) -> None:
        status = {
            "active": True,
            "phase": "agent_running",
        }

        rendered = render_status(
            status,
            status_path=Path("/tmp/status.json"),
            color=False,
            workspace_changes="2 files, +5/-1 | Python +3/-1; Shell +2",
        )

        self.assertIn("Workspace", rendered)
        self.assertIn("Changes", rendered)
        self.assertIn("Python +3/-1", rendered)

    def test_workspace_change_summary_groups_file_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)

            solver = root / "solver.py"
            solver.write_text("print('old')\n", encoding="utf-8")
            subprocess.run(["git", "add", "solver.py"], cwd=root, check=True, capture_output=True)
            solver.write_text("print('old')\nprint('new')\n", encoding="utf-8")

            script = root / "tool.sh"
            script.write_text("#!/usr/bin/env sh\necho ok\n", encoding="utf-8")

            public_level = root / "levels_public" / "101"
            public_level.parent.mkdir()
            public_level.write_text("ignored\n", encoding="utf-8")

            summary = _workspace_change_summary(root)

        self.assertIsNotNone(summary)
        self.assertIn("2 files", summary)
        self.assertIn("Python +1", summary)
        self.assertIn("Shell +2", summary)
        self.assertNotIn("ignored", summary)

    def test_workspace_change_cache_uses_refresh_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            (root / "one.py").write_text("print(1)\n", encoding="utf-8")
            cache = WorkspaceChangeCache(refresh_interval=999)
            status = {"workspace": str(root)}

            first = cache.get(status)
            (root / "two.py").write_text("print(2)\n", encoding="utf-8")
            second = cache.get(status)

        self.assertEqual(first, second)

    def test_resolve_status_path_default_without_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(temp_dir)
                path = resolve_status_path(None, "missing.toml")
            finally:
                os.chdir(cwd)

        self.assertEqual(path.name, "status.json")
        self.assertIn(".puzzle-runs", path.as_posix())

    def test_draw_frame_only_clears_screen_once(self) -> None:
        out = StringIO()
        with redirect_stdout(out):
            previous = _draw_frame("one\ntwo\n", [], first_frame=True)
            previous = _draw_frame("one\nthree\n", previous, first_frame=False)

        written = out.getvalue()
        self.assertEqual(written.count(CLEAR_SCREEN), 1)
        self.assertIn("three", written)


if __name__ == "__main__":
    unittest.main()
