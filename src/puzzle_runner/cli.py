from __future__ import annotations

import argparse
import dataclasses
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
    parser.add_argument(
        "--effort",
        default=None,
        help="Override agent effort for this run (for example: low, high, or max).",
    )
    parser.add_argument(
        "--evaluation-resume-from-best",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start evaluations near this run's best score. Enabled by default; overrides config.",
    )
    parser.add_argument(
        "--evaluation-backtrack-levels",
        type=_non_negative_int,
        default=None,
        help="Levels to backtrack from the best score. Defaults to 20; overrides config.",
    )
    return parser


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


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
        if args.effort is not None:
            config = dataclasses.replace(
                config,
                agent=dataclasses.replace(config.agent, effort=args.effort),
            )
        effort = config.agent.effort or "unspecified"
        print(
            f"Running agent: {config.agent.name} "
            f"(backend: {config.agent.backend}, effort: {effort})",
            flush=True,
        )
        overrides = {
            key: value
            for key in ("evaluation_resume_from_best", "evaluation_backtrack_levels")
            if (value := getattr(args, key)) is not None
        }
        config = dataclasses.replace(config, **overrides)
        result = Runner(config).run()
    except (ConfigError, RunnerError) as exc:
        print(f"puzzle-runner: {exc}", file=sys.stderr)
        return 1

    print(f"Final score: {result.best_score}")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Stop detail: {result.stop_detail}")
    print(f"Wall time: {result.total_wall_seconds:.2f}s")
    print(f"Agent output chars: {result.agent_output_chars}")
    print(f"Code lines added: {result.code_lines_added}")
    print(f"Logs: {result.log_dir}")
    return 0
