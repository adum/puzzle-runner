from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .runner import Runner, RunnerError
from .watch import add_watch_arguments, run_watch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an iterative puzzle benchmark agent.",
        epilog="Commands: run (default), watch. Use `puzzle-runner watch --help` for the live dashboard.",
    )
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


def build_watch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch Puzzle Runner live status.")
    add_watch_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "watch":
        args = build_watch_parser().parse_args(argv[1:])
        return run_watch(args)
    if argv and argv[0] == "run":
        argv = argv[1:]

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
