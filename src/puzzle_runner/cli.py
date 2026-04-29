from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .runner import Runner, RunnerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an iterative puzzle benchmark agent.")
    parser.add_argument(
        "--config",
        default="runner.toml",
        help="Path to runner config TOML. Defaults to runner.toml.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. Defaults to a timestamped id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, run_id=args.run_id)
        result = Runner(config).run()
    except (ConfigError, RunnerError) as exc:
        print(f"puzzle-runner: {exc}", file=sys.stderr)
        return 1

    print(f"Final score: {result.best_score}")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Logs: {result.log_dir}")
    return 0

