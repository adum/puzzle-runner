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


if __name__ == "__main__":
    unittest.main()
