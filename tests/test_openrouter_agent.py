import dataclasses
import http.client
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from puzzle_runner.config import load_config
from puzzle_runner.openrouter_agent import (
    AGENT_CONFIG_ERROR_RETURN_CODE,
    OPENROUTER_MAX_TOKENS_MARKER,
    parse_action,
    run_openrouter_agent,
)


class OpenRouterAgentTests(unittest.TestCase):
    def test_parse_action_accepts_plain_and_fenced_json(self) -> None:
        self.assertEqual(parse_action('{"action":"finish","message":"done"}')["action"], "finish")
        self.assertEqual(
            parse_action('```json\n{"action":"read_file","path":"prompts.txt"}\n```')["path"],
            "prompts.txt",
        )

    def test_missing_api_key_returns_agent_failure_without_calling_network(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.openrouter.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        missing_env = "PUZZLE_RUNNER_TEST_MISSING_OPENROUTER_API_KEY"
        os.environ.pop(missing_env, None)
        config = dataclasses.replace(
            config,
            agent=dataclasses.replace(config.agent, api_key_env=missing_env),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"

            result = run_openrouter_agent(
                config,
                cwd=root,
                prompt="hello",
                round_dir=root,
                stdout_path=stdout,
                stderr_path=stderr,
                timeout_seconds=10,
                echo=False,
            )
            stderr_text = stderr.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, AGENT_CONFIG_ERROR_RETURN_CODE)
        self.assertFalse(result.timed_out)
        self.assertIn(missing_env, stderr_text)

    def test_incomplete_openrouter_response_returns_retryable_failure(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.openrouter.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        api_key_env = "PUZZLE_RUNNER_TEST_OPENROUTER_API_KEY"
        config = dataclasses.replace(
            config,
            agent=dataclasses.replace(config.agent, api_key_env=api_key_env),
        )

        class BrokenResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                return None

            def read(self) -> bytes:
                raise http.client.IncompleteRead(b'{"partial":', 100)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"

            with mock.patch.dict(os.environ, {api_key_env: "test-key"}), mock.patch(
                "puzzle_runner.openrouter_agent.urllib.request.urlopen",
                return_value=BrokenResponse(),
            ):
                result = run_openrouter_agent(
                    config,
                    cwd=root,
                    prompt="hello",
                    round_dir=root,
                    stdout_path=stdout,
                    stderr_path=stderr,
                    timeout_seconds=10,
                    echo=False,
                )
            stderr_text = stderr.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.timed_out)
        self.assertIn("response ended before the complete body was read", stderr_text)
        self.assertIn("11 bytes received", stderr_text)

    def test_length_limited_response_reports_max_tokens_observation(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.openrouter.example.toml"
        config = load_config(str(config_path), run_id="test-run")
        api_key_env = "PUZZLE_RUNNER_TEST_OPENROUTER_API_KEY"
        config = dataclasses.replace(
            config,
            agent=dataclasses.replace(config.agent, api_key_env=api_key_env),
        )

        class JsonResponse:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                return None

            def read(self) -> bytes:
                import json

                return json.dumps(self.payload).encode("utf-8")

        length_response = {
            "id": "gen-length",
            "provider": "TestProvider",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning": "thinking until the limit"},
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 16384,
                "completion_tokens_details": {"reasoning_tokens": 14000},
                "total_tokens": 16396,
            },
        }
        finish_response = {
            "id": "gen-finish",
            "provider": "TestProvider",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"action":"finish","message":"done"}'},
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        }
        events: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"

            with mock.patch.dict(os.environ, {api_key_env: "test-key"}), mock.patch(
                "puzzle_runner.openrouter_agent.urllib.request.urlopen",
                side_effect=[JsonResponse(length_response), JsonResponse(finish_response)],
            ):
                result = run_openrouter_agent(
                    config,
                    cwd=root,
                    prompt="hello",
                    round_dir=root,
                    stdout_path=stdout,
                    stderr_path=stderr,
                    timeout_seconds=10,
                    echo=False,
                    status_callback=events.append,
                )
            stdout_text = stdout.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0)
        self.assertIn(OPENROUTER_MAX_TOKENS_MARKER, stdout_text)
        self.assertIn("configured max_tokens=16384", stdout_text)
        self.assertIn("completion_tokens=16384", stdout_text)
        self.assertIn("reasoning_tokens=14000", stdout_text)
        self.assertNotIn("response was not valid action JSON", stdout_text)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "openrouter_completion_limit")
        self.assertEqual(events[0]["step"], 1)
        self.assertEqual(events[0]["completion_tokens"], 16384)


if __name__ == "__main__":
    unittest.main()
