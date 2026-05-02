from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


SUMMARY_FILENAME = "openrouter-usage-summary.json"


@dataclasses.dataclass
class OpenRouterUsageSummary:
    calls: int = 0
    metadata_calls: int = 0
    metadata_failures: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    native_prompt_tokens: int = 0
    native_completion_tokens: int = 0
    native_reasoning_tokens: int = 0
    native_cached_tokens: int = 0
    latency_ms: int = 0
    generation_time_ms: int = 0
    providers: dict[str, int] = dataclasses.field(default_factory=dict)
    models: dict[str, int] = dataclasses.field(default_factory=dict)
    last_provider: str | None = None
    last_model: str | None = None
    last_finish_reason: str | None = None
    last_native_finish_reason: str | None = None
    last_latency_ms: int | None = None
    last_generation_time_ms: int | None = None


def summarize_openrouter_usage(root: Path) -> OpenRouterUsageSummary:
    summary = OpenRouterUsageSummary()
    response_paths = _matching_paths(root, "openrouter-response-*.json")

    for response_path in response_paths:
        response = _read_json_object(response_path)
        if response is None:
            continue

        summary.calls += 1
        response_had_usage = _add_response_usage(summary, response)

        step = _step_suffix(response_path, "openrouter-response-")
        if step is None:
            continue

        generation_path = response_path.with_name(f"openrouter-generation-{step}.json")
        error_path = response_path.with_name(f"openrouter-generation-error-{step}.json")
        generation = _generation_data(generation_path)
        if generation is not None:
            summary.metadata_calls += 1
            _add_generation_metadata(summary, generation, add_standard_tokens=not response_had_usage)
        elif error_path.exists():
            summary.metadata_failures += 1

    return summary


def load_openrouter_usage_summary(path: Path) -> OpenRouterUsageSummary | None:
    data = _read_json_object(path)
    if data is None:
        return None
    return openrouter_usage_from_dict(data)


def openrouter_usage_from_dict(data: dict[str, Any]) -> OpenRouterUsageSummary:
    summary = OpenRouterUsageSummary()
    int_fields = {
        "calls",
        "metadata_calls",
        "metadata_failures",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "native_prompt_tokens",
        "native_completion_tokens",
        "native_reasoning_tokens",
        "native_cached_tokens",
        "latency_ms",
        "generation_time_ms",
        "last_latency_ms",
        "last_generation_time_ms",
    }
    for field in dataclasses.fields(OpenRouterUsageSummary):
        value = data.get(field.name)
        if field.name in {"providers", "models"}:
            if isinstance(value, dict):
                setattr(
                    summary,
                    field.name,
                    {str(key): int(count) for key, count in value.items() if isinstance(count, int)},
                )
            continue
        if field.name in int_fields:
            parsed_int = _int(value)
            if parsed_int is not None:
                setattr(summary, field.name, parsed_int)
            continue
        if field.name == "cost_usd":
            parsed_float = _float(value)
            if parsed_float is not None:
                setattr(summary, field.name, parsed_float)
            continue
        if isinstance(value, str) or value is None:
            setattr(summary, field.name, value)
    return summary


def openrouter_usage_to_dict(summary: OpenRouterUsageSummary) -> dict[str, Any]:
    data = dataclasses.asdict(summary)
    data["cost_usd"] = round(summary.cost_usd, 12)
    return data


def write_openrouter_usage_summary(root: Path) -> OpenRouterUsageSummary:
    summary = summarize_openrouter_usage(root)
    path = root / SUMMARY_FILENAME
    path.write_text(
        json.dumps(openrouter_usage_to_dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _matching_paths(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    paths = list(root.glob(pattern))
    paths.extend(root.glob(f"round-*/{pattern}"))
    return sorted(set(paths))


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _generation_data(path: Path) -> dict[str, Any] | None:
    parsed = _read_json_object(path)
    if parsed is None:
        return None
    data = parsed.get("data")
    return data if isinstance(data, dict) else parsed


def _step_suffix(path: Path, prefix: str) -> str | None:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(".json"):
        return None
    return name[len(prefix) : -len(".json")]


def _add_response_usage(summary: OpenRouterUsageSummary, response: dict[str, Any]) -> bool:
    usage = response.get("usage")
    used_usage = False
    if isinstance(usage, dict):
        prompt = _int(usage.get("prompt_tokens"))
        completion = _int(usage.get("completion_tokens"))
        total = _int(usage.get("total_tokens"))
        if prompt is not None:
            summary.prompt_tokens += prompt
        if completion is not None:
            summary.completion_tokens += completion
        if total is not None:
            summary.total_tokens += total
            used_usage = True
        elif prompt is not None or completion is not None:
            summary.total_tokens += (prompt or 0) + (completion or 0)
            used_usage = True

    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
        if isinstance(finish_reason, str):
            summary.last_finish_reason = finish_reason

    model = response.get("model")
    if isinstance(model, str) and model:
        _increment(summary.models, model)
        summary.last_model = model
    return used_usage


def _add_generation_metadata(
    summary: OpenRouterUsageSummary,
    data: dict[str, Any],
    *,
    add_standard_tokens: bool,
) -> None:
    cost = _float(data.get("total_cost"))
    if cost is None:
        cost = _float(data.get("usage"))
    if cost is not None:
        summary.cost_usd += cost

    if add_standard_tokens:
        prompt = _int(data.get("tokens_prompt"))
        completion = _int(data.get("tokens_completion"))
        if prompt is not None:
            summary.prompt_tokens += prompt
        if completion is not None:
            summary.completion_tokens += completion
        if prompt is not None or completion is not None:
            summary.total_tokens += (prompt or 0) + (completion or 0)

    summary.native_prompt_tokens += _int(data.get("native_tokens_prompt")) or 0
    summary.native_completion_tokens += _int(data.get("native_tokens_completion")) or 0
    summary.native_reasoning_tokens += _int(data.get("native_tokens_reasoning")) or 0
    summary.native_cached_tokens += _int(data.get("native_tokens_cached")) or 0

    latency = _int(data.get("latency"))
    if latency is not None:
        summary.latency_ms += latency
        summary.last_latency_ms = latency

    generation_time = _int(data.get("generation_time"))
    if generation_time is not None:
        summary.generation_time_ms += generation_time
        summary.last_generation_time_ms = generation_time

    provider = data.get("provider_name")
    if isinstance(provider, str) and provider:
        _increment(summary.providers, provider)
        summary.last_provider = provider

    model = data.get("model")
    if isinstance(model, str) and model:
        _increment(summary.models, model)
        summary.last_model = model

    finish_reason = data.get("finish_reason")
    if isinstance(finish_reason, str):
        summary.last_finish_reason = finish_reason

    native_finish_reason = data.get("native_finish_reason")
    if isinstance(native_finish_reason, str):
        summary.last_native_finish_reason = native_finish_reason


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
