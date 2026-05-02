import dataclasses
import os
import tempfile
import unittest
from pathlib import Path

from puzzle_runner.config import load_config
from puzzle_runner.openrouter_agent import (
    AGENT_CONFIG_ERROR_RETURN_CODE,
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


if __name__ == "__main__":
    unittest.main()
