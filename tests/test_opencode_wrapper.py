import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from puzzle_runner.config import load_config
from puzzle_runner.runner import Runner


ROOT = Path(__file__).resolve().parents[1]


class OpenCodeWrapperTests(unittest.TestCase):
    def invoke(self, version, args):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "opencode"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print(os.environ['TEST_OPENCODE_VERSION'])\n"
                "else:\n"
                "    print(json.dumps(sys.argv[1:]))\n"
            )
            executable.chmod(0o755)
            result = subprocess.run(
                [str(ROOT / "scripts/opencode"), *args],
                env={**os.environ, "NVM_DIR": str(root / "no-nvm"),
                     "PATH": f"{root}:{os.environ['PATH']}",
                     "TEST_OPENCODE_VERSION": version},
                text=True, capture_output=True, check=True,
            )
            return json.loads(result.stdout)

    def test_v1_preserves_arguments(self):
        args = ["run", "--dangerously-skip-permissions", "--model", "provider/model",
                "--variant", "high"]
        self.assertEqual(self.invoke("1.9.0", args), args)

    def test_v2_translates_runner_command(self):
        config = load_config(str(ROOT / "config.opencode.example.toml"), run_id="test-run")
        runner = Runner(config)
        command = runner._agent_command(Path("/tmp/round"))
        actual = self.invoke("opencode v2.0.32", command[1:])
        self.assertIn("--auto", actual)
        self.assertIn("openrouter/google/gemini-3-flash-preview#high", actual)
        self.assertNotIn("--variant", actual)
        self.assertNotIn("--dangerously-skip-permissions", actual)
        self.assertEqual(runner._opencode_models_command("openrouter"), [command[0], "models"])

    def test_v2_keeps_explicit_variant_and_prompt(self):
        self.assertEqual(
            self.invoke("2.0.32", ["run", "--variant=high", "-m", "provider/model#low",
                                   "--", "a prompt with spaces"]),
            ["run", "-m", "provider/model#low", "--", "a prompt with spaces"],
        )
