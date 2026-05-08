import unittest
from pathlib import Path

from puzzle_runner.config import load_config


class ConfigTests(unittest.TestCase):
    def test_example_config_loads_agent_retry_defaults(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.example.toml"

        config = load_config(str(config_path), run_id="test-run")

        self.assertEqual(config.agent.name, "codex-5.3")
        self.assertIn("model_reasoning_effort=\"xhigh\"", config.agent.command)
        self.assertIn("gpt-5.3-codex", config.agent.command)
        self.assertEqual(config.agent_failure_retry_limit_seconds, 900)

    def test_claude_example_config_loads(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.claude.example.toml"

        config = load_config(str(config_path), run_id="test-run")

        self.assertEqual(config.agent.name, "claude-code-sonnet")
        self.assertEqual(config.agent.backend, "claude-code")
        self.assertEqual(config.agent.prompt_mode, "stdin")
        self.assertEqual(config.agent.effort, "xhigh")
        self.assertEqual(config.agent_idle_timeout_seconds, 1800)
        self.assertFalse(config.echo_agent_output)
        self.assertIn("{config_dir}/scripts/claude-code", config.agent.command)
        self.assertIn("--print", config.agent.command)
        self.assertIn("--no-session-persistence", config.agent.command)
        self.assertIn("--verbose", config.agent.command)
        self.assertIn("--output-format", config.agent.command)
        self.assertIn("stream-json", config.agent.command)
        self.assertIn("--include-partial-messages", config.agent.command)
        self.assertIn("--dangerously-skip-permissions", config.agent.command)
        self.assertIn("claude-sonnet-4-6", config.agent.command)

    def test_gemini_example_config_loads(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.gemini.example.toml"

        config = load_config(str(config_path), run_id="test-run")

        self.assertEqual(config.agent.name, "gemini-3.1-pro-preview")
        self.assertEqual(config.agent.backend, "gemini-cli")
        self.assertEqual(config.agent.prompt_mode, "stdin")
        self.assertEqual(config.agent.model, "gemini-3.1-pro-preview")
        self.assertIn("{config_dir}/scripts/gemini-cli", config.agent.command)
        self.assertIn("--approval-mode", config.agent.command)
        self.assertIn("yolo", config.agent.command)
        self.assertIn("--skip-trust", config.agent.command)
        self.assertIn("--output-format", config.agent.command)
        self.assertIn("stream-json", config.agent.command)

    def test_openrouter_example_config_loads(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.openrouter.example.toml"

        config = load_config(str(config_path), run_id="test-run")

        self.assertEqual(config.agent.name, "openrouter-laguna-xs-2-free")
        self.assertEqual(config.agent.backend, "openrouter")
        self.assertEqual(config.agent.command, [])
        self.assertEqual(config.agent.model, "poolside/laguna-xs.2:free")
        self.assertEqual(config.agent.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(config.agent.max_tokens, 16384)
        self.assertEqual(config.agent.max_steps, 200)
        self.assertEqual(config.agent.command_timeout_seconds, 120)


if __name__ == "__main__":
    unittest.main()
