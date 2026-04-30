import unittest
from pathlib import Path

from puzzle_runner.config import load_config
from puzzle_runner.runner import explain_stop_reason


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


if __name__ == "__main__":
    unittest.main()
