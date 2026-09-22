import contextlib
import dataclasses
import io
import json
import unittest
from pathlib import Path

from puzzle_runner.config import load_config
from puzzle_runner.runner import _agent_stdout_line_callback


class ClaudeProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(Path(__file__).resolve().parents[1] / "config.claude.example.toml")

    def test_live_text_tools_and_results_without_duplicate_messages(self) -> None:
        callback = _agent_stdout_line_callback(self.config)
        self.assertIsNotNone(callback)
        output = io.StringIO()

        def emit(event):
            callback(json.dumps(event) + "\n")

        with contextlib.redirect_stdout(output):
            emit({"type": "system", "subtype": "init"})
            emit({"type": "stream_event", "event": {"type": "message_start"}})
            emit({"type": "stream_event", "event": {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "Checking the solver."},
            }})
            # The partial text is visible before the completed assistant message.
            self.assertIn("Checking the solver.", output.getvalue())
            emit({"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}})
            emit({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Checking the solver."},
                {"type": "tool_use", "name": "Bash", "input": {"command": "cat solver.py"}},
            ]}})
            emit({"type": "user", "message": {"content": [
                {"type": "tool_result", "content": [{"type": "text", "text": "solver contents"}]},
            ]}})
            emit({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Done."},
            ]}})
            emit({"type": "result", "subtype": "success", "result": "Done."})
        text = output.getvalue()
        self.assertEqual(text.count("Checking the solver."), 1)
        self.assertEqual(text.count("Done."), 1)
        self.assertIn("Claude tool: Bash", text)
        self.assertIn("cat solver.py", text)
        self.assertIn("solver contents", text)
        self.assertTrue(text.endswith("Claude: success\n"))

    def test_thinking_errors_and_result_only_output(self) -> None:
        callback = _agent_stdout_line_callback(self.config)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            callback("not json\n")
            callback("[]\n")
            callback(json.dumps({"type": "stream_event", "event": {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Inspecting constraints."},
            }}))
            callback(json.dumps({"type": "user", "message": {"content": [
                {"type": "tool_result", "is_error": True, "content": "Command failed"},
            ]}}))
            fresh_callback = _agent_stdout_line_callback(self.config)
            fresh_callback(json.dumps({"type": "result", "subtype": "success", "result": "Finished"}))
        self.assertIn("Inspecting constraints.\nClaude tool error:\nCommand failed\n", output.getvalue())
        self.assertIn("Finished\nClaude: success", output.getvalue())

    def test_respects_quiet_and_raw_output_settings(self) -> None:
        for changes in ({"echo_agent_progress": False}, {"echo_agent_output": True}):
            with self.subTest(changes=changes):
                self.assertIsNone(_agent_stdout_line_callback(dataclasses.replace(self.config, **changes)))
