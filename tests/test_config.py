import tempfile
import unittest
from pathlib import Path

from puzzle_runner.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_evaluation_final_level(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "config.example.toml").read_text()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runner.toml"
            for setting, expected in [("", 1208), ("evaluation_final_level = 10\n", 10), ("evaluation_final_level = 0\n", 0)]:
                with self.subTest(setting=setting):
                    path.write_text(setting + source)
                    self.assertEqual(load_config(str(path)).evaluation_final_level, expected)
            for value in ["-1", "true", '"1208"']:
                with self.subTest(value=value):
                    path.write_text(f"evaluation_final_level = {value}\n" + source)
                    with self.assertRaises(ConfigError):
                        load_config(str(path))

    def test_evaluation_resume_settings(self) -> None:
        source_path = Path(__file__).resolve().parents[1] / "config.example.toml"
        source = source_path.read_text(encoding="utf-8")
        source = source.replace("evaluation_resume_from_best = true\n", "")
        source = source.replace("evaluation_backtrack_levels = 20\n", "")
        cases = [
            ("", True, 20),
            ("evaluation_resume_from_best = false\n", False, 20),
            ("evaluation_backtrack_levels = 7\n", True, 7),
            ("evaluation_backtrack_levels = 0\n", True, 0),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "runner.toml"
            for settings, enabled, backtrack in cases:
                with self.subTest(settings=settings):
                    config_path.write_text(settings + source, encoding="utf-8")
                    config = load_config(str(config_path), run_id="test-run")
                    self.assertEqual(config.evaluation_resume_from_best, enabled)
                    self.assertEqual(config.evaluation_backtrack_levels, backtrack)

    def test_invalid_evaluation_resume_settings_are_rejected(self) -> None:
        source_path = Path(__file__).resolve().parents[1] / "config.example.toml"
        source = source_path.read_text(encoding="utf-8")
        source = source.replace("evaluation_resume_from_best = true\n", "")
        source = source.replace("evaluation_backtrack_levels = 20\n", "")
        cases = [
            ("evaluation_resume_from_best", '"false"'),
            ("evaluation_resume_from_best", "1"),
            ("evaluation_backtrack_levels", "-1"),
            ("evaluation_backtrack_levels", "true"),
            ("evaluation_backtrack_levels", '"20"'),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "runner.toml"
            for key, value in cases:
                with self.subTest(key=key, value=value):
                    config_path.write_text(f"{key} = {value}\n" + source, encoding="utf-8")
                    with self.assertRaisesRegex(ConfigError, key):
                        load_config(str(config_path), run_id="test-run")

    def test_example_config_loads_agent_retry_defaults(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.example.toml"

        config = load_config(str(config_path), run_id="test-run")

        self.assertEqual(config.agent.name, "gpt-5.3-codex")
        self.assertIn("model_reasoning_effort=\"xhigh\"", config.agent.command)
        self.assertIn("gpt-5.3-codex", config.agent.command)
        self.assertEqual(config.agent_failure_retry_limit_seconds, 900)

    def test_claude_example_config_loads(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.claude.example.toml"

        config = load_config(str(config_path), run_id="test-run")

        self.assertEqual(config.agent.name, "claude-sonnet-4-6")
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

        self.assertEqual(config.agent.name, "gemini-3.5-flash-high")
        self.assertEqual(config.agent.backend, "antigravity-cli")
        self.assertEqual(config.agent.prompt_mode, "stdin")
        self.assertEqual(config.agent.model, "Gemini 3.5 Flash (High)")
        self.assertIn("{config_dir}/scripts/antigravity-cli", config.agent.command)
        self.assertIn("--dangerously-skip-permissions", config.agent.command)
        self.assertIn("--print-timeout", config.agent.command)
        self.assertIn("30m", config.agent.command)

    def test_gemini_cli_example_config_loads(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.gemini-cli.example.toml"

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

        self.assertEqual(config.agent.name, "openrouter-poolside-laguna-xs.2-free")
        self.assertEqual(config.agent.backend, "openrouter")
        self.assertEqual(config.agent.command, [])
        self.assertEqual(config.agent.model, "poolside/laguna-xs.2:free")
        self.assertEqual(config.agent.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(config.agent.max_tokens, 16384)
        self.assertEqual(config.agent.max_steps, 200)
        self.assertEqual(config.agent.command_timeout_seconds, 120)

    def test_opencode_example_config_loads(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"

        config = load_config(str(config_path), run_id="test-run")

        self.assertEqual(config.agent.name, "opencode-openrouter-google-gemini-3-flash-preview")
        self.assertEqual(config.agent.backend, "opencode")
        self.assertEqual(config.agent.prompt_mode, "stdin")
        self.assertEqual(config.agent.model, "openrouter/google/gemini-3-flash-preview")
        self.assertEqual(config.agent.effort, "high")
        self.assertFalse(config.echo_agent_output)
        self.assertTrue(config.echo_agent_progress)
        self.assertIn("{config_dir}/scripts/opencode", config.agent.command)
        self.assertIn("run", config.agent.command)
        self.assertIn("--title", config.agent.command)
        self.assertIn("{run_id}", config.agent.command)
        self.assertIn("--format", config.agent.command)
        self.assertIn("json", config.agent.command)
        self.assertIn("--dangerously-skip-permissions", config.agent.command)

    def test_grok_build_example_config_loads(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.grok-build.example.toml"

        config = load_config(str(config_path), run_id="test-run")

        self.assertEqual(config.agent.name, "grok-composer-2.5-fast")
        self.assertEqual(config.agent.backend, "grok-build")
        self.assertEqual(config.agent.prompt_mode, "file")
        self.assertEqual(config.agent.model, "composer-2.5-fast")
        self.assertIn("grok", config.agent.command)
        self.assertIn("--prompt-file", config.agent.command)
        self.assertIn("{prompt_path}", config.agent.command)
        self.assertIn("--permission-mode", config.agent.command)
        self.assertIn("bypassPermissions", config.agent.command)
        max_turns_index = config.agent.command.index("--max-turns")
        self.assertEqual(config.agent.command[max_turns_index + 1], "512")
        self.assertEqual(config.agent_idle_timeout_seconds, 1800)
        self.assertTrue(config.echo_agent_output)

    def test_explicit_agent_name_overrides_model_default(self) -> None:
        source_path = Path(__file__).resolve().parents[1] / "config.example.toml"
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "runner.toml"
            text = source_path.read_text(encoding="utf-8")
            config_path.write_text(
                text.replace("[agent]\n", '[agent]\nname = "custom-codex"\n', 1),
                encoding="utf-8",
            )

            config = load_config(str(config_path), run_id="test-run")

        self.assertEqual(config.agent.name, "custom-codex")

    def test_default_run_id_uses_derived_agent_name(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.opencode.example.toml"

        config = load_config(str(config_path))

        self.assertRegex(
            config.run_id,
            r"^\d{8}-\d{6}-opencode-openrouter-google-gemini-3-flash-preview$",
        )


if __name__ == "__main__":
    unittest.main()
