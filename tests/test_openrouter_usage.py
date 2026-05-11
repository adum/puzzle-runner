import json
import tempfile
import unittest
from pathlib import Path

from puzzle_runner.openrouter_usage import (
    SUMMARY_FILENAME,
    load_openrouter_usage_summary,
    summarize_opencode_openrouter_usage,
    summarize_openrouter_usage,
    write_openrouter_usage_summary,
)


class OpenRouterUsageTests(unittest.TestCase):
    def test_summarize_openrouter_usage_combines_response_and_generation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            round_dir = root / "round-001"
            round_dir.mkdir()
            (round_dir / "openrouter-response-001.json").write_text(
                json.dumps(
                    {
                        "id": "gen-1",
                        "model": "poolside/laguna-xs.2:free",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                        "choices": [{"finish_reason": "stop"}],
                    }
                ),
                encoding="utf-8",
            )
            (round_dir / "openrouter-generation-001.json").write_text(
                json.dumps(
                    {
                        "data": {
                            "id": "gen-1",
                            "total_cost": 0.00125,
                            "provider_name": "Infermatic",
                            "model": "poolside/laguna-xs.2:free",
                            "finish_reason": "stop",
                            "native_finish_reason": "stop",
                            "native_tokens_prompt": 11,
                            "native_tokens_completion": 6,
                            "native_tokens_reasoning": 2,
                            "native_tokens_cached": 3,
                            "latency": 1200,
                            "generation_time": 900,
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_openrouter_usage(root)

        self.assertEqual(summary.calls, 1)
        self.assertEqual(summary.metadata_calls, 1)
        self.assertEqual(summary.prompt_tokens, 10)
        self.assertEqual(summary.completion_tokens, 5)
        self.assertEqual(summary.total_tokens, 15)
        self.assertEqual(summary.native_reasoning_tokens, 2)
        self.assertEqual(summary.native_cached_tokens, 3)
        self.assertEqual(summary.last_provider, "Infermatic")
        self.assertEqual(summary.last_latency_ms, 1200)
        self.assertAlmostEqual(summary.cost_usd, 0.00125)

    def test_write_openrouter_usage_summary_writes_loadable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "openrouter-response-001.json").write_text(
                json.dumps(
                    {
                        "id": "gen-1",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    }
                ),
                encoding="utf-8",
            )

            written = write_openrouter_usage_summary(root)
            loaded = load_openrouter_usage_summary(root / SUMMARY_FILENAME)

        self.assertEqual(written.calls, 1)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.total_tokens, 3)

    def test_response_level_cost_provider_and_details_are_counted_without_metadata_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "openrouter-response-001.json").write_text(
                json.dumps(
                    {
                        "id": "gen-1",
                        "model": "deepseek/deepseek-v4-flash-20260423",
                        "provider": "Parasail",
                        "usage": {
                            "prompt_tokens": 459,
                            "completion_tokens": 38,
                            "total_tokens": 497,
                            "cost": 7.49e-05,
                            "prompt_tokens_details": {"cached_tokens": 12},
                            "completion_tokens_details": {"reasoning_tokens": 20},
                        },
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "native_finish_reason": "stop",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "openrouter-generation-error-001.json").write_text(
                '{"error":"OpenRouter metadata HTTP 404"}\n',
                encoding="utf-8",
            )

            summary = summarize_openrouter_usage(root)

        self.assertEqual(summary.calls, 1)
        self.assertEqual(summary.metadata_failures, 0)
        self.assertEqual(summary.prompt_tokens, 459)
        self.assertEqual(summary.completion_tokens, 38)
        self.assertEqual(summary.total_tokens, 497)
        self.assertEqual(summary.native_cached_tokens, 12)
        self.assertEqual(summary.native_reasoning_tokens, 20)
        self.assertEqual(summary.last_provider, "Parasail")
        self.assertAlmostEqual(summary.cost_usd, 7.49e-05)

    def test_summarize_opencode_openrouter_usage_from_step_finish_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            round_dir = root / "round-001"
            round_dir.mkdir()
            (round_dir / "agent.stdout.log").write_text(
                "\n".join(
                    [
                        json.dumps({"type": "step_start"}),
                        json.dumps(
                            {
                                "type": "step_finish",
                                "part": {
                                    "reason": "tool-calls",
                                    "cost": 0.0125,
                                    "tokens": {
                                        "total": 100,
                                        "input": 60,
                                        "output": 10,
                                        "reasoning": 5,
                                        "cache": {"read": 25, "write": 0},
                                    },
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "step_finish",
                                "part": {
                                    "reason": "stop",
                                    "cost": 0.0025,
                                    "tokens": {
                                        "total": 90,
                                        "input": 50,
                                        "output": -3,
                                        "reasoning": 7,
                                        "cache": {"read": 33, "write": 0},
                                    },
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_opencode_openrouter_usage(
                root,
                model="openrouter/example/model",
            )

        self.assertEqual(summary.calls, 2)
        self.assertEqual(summary.prompt_tokens, 110)
        self.assertEqual(summary.completion_tokens, 10)
        self.assertEqual(summary.total_tokens, 190)
        self.assertEqual(summary.native_reasoning_tokens, 12)
        self.assertEqual(summary.native_cached_tokens, 58)
        self.assertEqual(summary.last_provider, "OpenRouter via OpenCode")
        self.assertEqual(summary.last_model, "openrouter/example/model")
        self.assertEqual(summary.last_finish_reason, "stop")
        self.assertAlmostEqual(summary.cost_usd, 0.015)


if __name__ == "__main__":
    unittest.main()
