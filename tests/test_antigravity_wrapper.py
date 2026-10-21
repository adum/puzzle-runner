import dataclasses
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from puzzle_runner.config import load_config
from puzzle_runner.runner import Runner, _agent_error_detail, _apply_agent_effort

ROOT = Path(__file__).resolve().parents[1]


class AntigravityTests(unittest.TestCase):
    def test_effort_forwarding_and_explicit_override(self):
        config = load_config(str(ROOT / 'config.gemini.example.toml'), run_id='test-run')
        config = dataclasses.replace(config, agent=dataclasses.replace(config.agent, effort='max'))
        command = Runner(config)._agent_command(Path('/tmp/round'))
        self.assertEqual(command[command.index('--effort') + 1], 'max')
        for explicit in (['agy', '--effort', 'low'], ['agy', '--effort=low']):
            self.assertEqual(_apply_agent_effort(config, explicit), explicit)

    def test_diagnostics_classify_auth_and_transient_errors(self):
        config = load_config(str(ROOT / 'config.gemini.example.toml'), run_id='test-run')
        with tempfile.TemporaryDirectory() as directory:
            stdout = Path(directory) / 'stdout'
            stderr = Path(directory) / 'stderr'
            stdout.write_text('AGY_ERROR: unauthorized\n')
            stderr.write_text('')
            self.assertIsNone(_agent_error_detail(config, stdout, stderr))
            for message, kind, retryable in [
                ('unauthorized', 'auth', False),
                ('rate limit exceeded', 'error', True),
                ('invalid request', 'error', False),
            ]:
                stderr.write_text('AGY_ERROR: ' + message + '\n')
                detail = _agent_error_detail(config, stdout, stderr)
                self.assertEqual(detail['kind'], kind)
                self.assertEqual(detail['retryable'], retryable)

    def test_wrapper_isolates_settings_forwards_prompt_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / 'home'
            config_dir = home / '.gemini/antigravity-cli'
            config_dir.mkdir(parents=True)
            original = '{"model": "original", "trustedWorkspaces": []}\n'
            (config_dir / 'settings.json').write_text(original)
            (config_dir / 'antigravity-oauth-token').write_text('test-token')
            executable = root / 'agy'
            executable.write_text(
                '#!/usr/bin/env python3\n'
                'import json, os, pathlib, sys\n'
                'home = pathlib.Path.home()\n'
                'config = home / ".gemini/antigravity-cli"\n'
                'print(json.dumps({"args": sys.argv[1:], "home": str(home), '
                '"settings": json.loads((config / "settings.json").read_text()), '
                '"token": (config / "antigravity-oauth-token").read_text()}))\n'
                'print("AGY_ERROR: unauthorized", file=sys.stderr)\n'
                'sys.exit(3)\n'
            )
            executable.chmod(0o755)
            result = subprocess.run(
                [str(ROOT / 'scripts/antigravity-cli'), '--model', 'test-model', '--effort', 'max'],
                cwd=root, input='prompt with spaces\nand newlines', text=True, capture_output=True,
                env={**os.environ, 'HOME': str(home), 'PATH': f'{root}:{os.environ["PATH"]}',
                     'AGY_PRINT_TIMEOUT': '0'},
            )
            self.assertEqual(result.returncode, 3)
            payload = json.loads(result.stdout)
            self.assertEqual(payload['args'], ['--print-timeout', '0', '--effort', 'max',
                                             '--print', 'prompt with spaces\nand newlines'])
            self.assertEqual(payload['settings']['model'], 'test-model')
            self.assertIn(str(root.resolve()), payload['settings']['trustedWorkspaces'])
            self.assertEqual(payload['token'], 'test-token')
            self.assertFalse(Path(payload['home']).exists())
            self.assertEqual((config_dir / 'settings.json').read_text(), original)
            self.assertIn('AGY_ERROR: unauthorized', result.stderr)
