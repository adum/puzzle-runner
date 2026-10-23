#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import bisect
import csv
import dataclasses
import datetime as dt
import html
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path


MODEL_PALETTE = [
    "#1d4ed8",
    "#64748b",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#9f6b7d",
    "#4338ca",
    "#65a30d",
    "#ca8a04",
    "#0f766e",
    "#c026d3",
    "#b45309",
    "#2563eb",
    "#8b5cf6",
    "#16a34a",
    "#9333ea",
    "#f97316",
    "#0284c7",
    "#78716c",
]

MODEL_PATTERNS = ["solid", "diagonal", "cross", "dots", "vertical", "horizontal"]

FAMILY_ICON_FILES = {
    "ChatGPT / Codex": "openai.svg",
    "Claude": "claude.svg",
    "DeepSeek": "deepseek.svg",
    "Gemini": "gemini.svg",
    "GLM": "glm.svg",
    "Grok": "grok.svg",
    "Kimi": "kimi.svg",
    "Meta": "meta.svg",
    "MiniMax": "minimax.svg",
    "Mistral": "mistral.svg",
    "Nemotron": "nemotron.svg",
    "Qwen": "qwen.svg",
    "Sakana Fugu": "sakana-fugu.png",
    "Tencent Hunyuan": "tencent-hunyuan.svg",
}

SINGLE_SERIES_COLOR = "#64748b"

# Fixed per family and capped at eight colorblind-validated slots; any other family shares
# OTHER_COLOR and is identified by its icon, label, or tooltip.
FAMILY_COLORS = {
    "Gemini": "#2a78d6",
    "Claude": "#eb6834",
    "ChatGPT / Codex": "#1baf7a",
    "GLM": "#eda100",
    "Kimi": "#e87ba4",
    "DeepSeek": "#008300",
    "Qwen": "#6250d6",
    "Grok": "#e34948",
}

# In a scatter plot any two points can sit side by side; only the first three slots stay distinct then.
SCATTER_FAMILIES = list(FAMILY_COLORS)[:3]

OTHER_COLOR = "#a8b1bd"
OTHER_FAMILIES_LABEL = "Other families"

ORIGIN_COLORS = {
    "America": "#2a78d6",
    "China": "#eda100",
    "Europe": "#008300",
    "Japan": "#e87ba4",
    "Unknown": OTHER_COLOR,
}

OPEN_WEIGHTS_COLORS = {
    "Open weights": "#059669",
    "Closed weights": "#64748b",
}

HUMAN_BEST_SCORE = 1206
BENCHMARK_FINAL_LEVEL = 1208
DEFAULT_BRUTE_FORCE_SCORE = 47
HUMAN_COLOR = "#7c3aed"
BRUTE_FORCE_COLOR = "#ea580c"
RESULT_RULE = (
    "If a model has multiple runs, its reported result is the maximum Best Score "
    "observed for that normalized model version. Score charts use those max-only "
    "model results; raw runs remain in Run Details for audit."
)


@dataclasses.dataclass(frozen=True)
class ModelMeta:
    family: str
    version: str
    release_date: dt.date
    origin: str
    open_weights: bool


@dataclasses.dataclass(frozen=True)
class RunResult:
    run_id: str
    run_date: dt.date
    agent: str
    harness: str
    effort: str
    best_score: int
    best_round: int | None
    rounds: int | None
    stop_reason: str
    wall_seconds: int | None
    agent_chars: int | None
    code_lines_added: int | None
    openrouter_calls: int | None
    openrouter_cost: float | None
    openrouter_tokens: int | None
    model: ModelMeta | None

    @property
    def family(self) -> str:
        return self.model.family if self.model else "Unknown"

    @property
    def version(self) -> str:
        return self.model.version if self.model else self.agent

    @property
    def origin(self) -> str:
        return self.model.origin if self.model else "Unknown"

    @property
    def release_date(self) -> dt.date | None:
        return self.model.release_date if self.model else None

    @property
    def open_weights(self) -> bool:
        return self.model.open_weights if self.model else False

    @property
    def weights_status(self) -> str:
        return "Open weights" if self.open_weights else "Closed weights"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build a static HTML/SVG visualization report from benchmark markdown tables."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=root / "final_results.md",
        help="Markdown table of benchmark results. Defaults to final_results.md.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "model_metadata.md",
        help="Markdown table of model metadata. Defaults to model_metadata.md.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "visualizations.html",
        help="HTML report path. Defaults to visualizations.html.",
    )
    parser.add_argument(
        "--level-dimensions",
        type=Path,
        default=root / "level_dimensions.csv",
        help="CSV mapping public level numbers to board dimensions. Missing levels are interpolated.",
    )
    parser.add_argument(
        "--family-icons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use model-family icons for large chart markers. Defaults to enabled.",
    )
    parser.add_argument(
        "--family-icon-dir",
        type=Path,
        default=root / "assets" / "model-family-icons",
        help="Directory containing model-family icon assets.",
    )
    return parser.parse_args()


def load_family_icons(path: Path) -> dict[str, str]:
    icons: dict[str, str] = {}
    for family, filename in FAMILY_ICON_FILES.items():
        icon_path = path / filename
        if not icon_path.is_file():
            continue
        mime_type = "image/svg+xml" if icon_path.suffix.lower() == ".svg" else "image/png"
        payload = base64.b64encode(icon_path.read_bytes()).decode("ascii")
        icons[family] = f"data:{mime_type};base64,{payload}"
    return icons


def load_level_dimensions(path: Path) -> dict[int, tuple[int, int]]:
    dimensions: dict[int, tuple[int, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            level = int(row["level"])
            dimensions[level] = (int(row["width"]), int(row["height"]))
    if len(dimensions) < 2:
        raise ValueError(f"at least two level dimensions are required in {path}")
    return dimensions


def board_dimensions_for_level(
    level: int,
    dimensions: dict[int, tuple[int, int]],
) -> tuple[int, int]:
    if level <= 0:
        return 0, 0
    if level in dimensions:
        return dimensions[level]

    known_levels = sorted(dimensions)
    index = bisect.bisect_left(known_levels, level)
    if index == 0:
        lower, upper = known_levels[0], known_levels[1]
    elif index == len(known_levels):
        lower, upper = known_levels[-2], known_levels[-1]
    else:
        lower, upper = known_levels[index - 1], known_levels[index]

    fraction = (level - lower) / (upper - lower)
    lower_width, lower_height = dimensions[lower]
    upper_width, upper_height = dimensions[upper]
    width = round(lower_width + (upper_width - lower_width) * fraction)
    height = round(lower_height + (upper_height - lower_height) * fraction)
    return max(1, width), max(1, height)


def board_area_for_level(level: int, dimensions: dict[int, tuple[int, int]]) -> int:
    width, height = board_dimensions_for_level(level, dimensions)
    return width * height


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines: list[str] = []
    in_table = False
    for line in lines:
        if line.lstrip().startswith("|"):
            table_lines.append(line)
            in_table = True
        elif in_table:
            break

    if len(table_lines) < 2:
        raise ValueError(f"no markdown table found in {path}")

    rows = [split_table_row(line) for line in table_lines]
    header = rows[0]
    body = rows[1:]
    if body and is_separator_row(body[0]):
        body = body[1:]

    parsed: list[dict[str, str]] = []
    for row in body:
        if not any(row):
            continue
        padded = row + [""] * max(0, len(header) - len(row))
        parsed.append(dict(zip(header, padded[: len(header)])))
    return parsed


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(row: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row)


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value.strip(), "%B %d, %Y").date()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "yes", "1"}


def parse_run_date(run_id: str) -> dt.date:
    return dt.datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S").date()


def parse_int(value: str) -> int | None:
    value = value.strip().replace(",", "")
    if not value:
        return None
    return int(value)


def parse_cost(value: str) -> float | None:
    value = value.strip().replace("$", "").replace(",", "")
    if not value:
        return None
    return float(value)


def parse_duration_seconds(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([dhms])", value):
        number = float(amount)
        if unit == "d":
            total += number * 86400
        elif unit == "h":
            total += number * 3600
        elif unit == "m":
            total += number * 60
        elif unit == "s":
            total += number
    return int(total) if total else None


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def candidate_model_keys(agent: str) -> list[str]:
    base = normalize_key(agent)
    candidates = [base]

    stripped = base
    if stripped.startswith("opencode openrouter "):
        stripped = stripped.removeprefix("opencode openrouter ")
        candidates.append(stripped)
    if stripped.startswith("moonshotai "):
        candidates.append(stripped.removeprefix("moonshotai "))
    if stripped.startswith("meta llama llama "):
        candidates.append("llama " + stripped.removeprefix("meta llama llama "))
    if stripped.startswith("meta muse "):
        candidates.append(stripped.removeprefix("meta "))
    if stripped.startswith("x ai "):
        candidates.append(stripped.removeprefix("x ai "))
    if stripped.startswith("openai "):
        candidates.append(stripped.removeprefix("openai "))
    if stripped.startswith("anthropic "):
        candidates.append(stripped.removeprefix("anthropic "))
    if stripped.startswith("mistralai "):
        candidates.append(stripped.removeprefix("mistralai "))
    if stripped.startswith("minimax minimax "):
        candidates.append(stripped.removeprefix("minimax "))
    if stripped.startswith("z ai "):
        candidates.append(stripped.removeprefix("z ai "))
    if stripped.startswith("qwen qwen"):
        candidates.append("qwen" + stripped.removeprefix("qwen qwen"))
    if stripped.startswith("sakana "):
        candidates.append(stripped.removeprefix("sakana "))
    if stripped.startswith("deepseek deepseek "):
        candidates.append(stripped.removeprefix("deepseek "))
    if stripped.startswith("thinkingmachines "):
        candidates.append(stripped.removeprefix("thinkingmachines "))
    if stripped.startswith("stealth "):
        candidates.append(stripped.removeprefix("stealth "))
    if base in {"gemini 4 argon low", "gemini 4 argon medium", "gemini 4 argon high"}:
        candidates.append("gemini 4 argon")

    if base.startswith("codex "):
        candidates.append("gpt " + base.removeprefix("codex ") + " codex")
    if base.startswith("claude code "):
        candidates.append("claude " + base.removeprefix("claude code "))

    return dedupe(candidates)


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def load_metadata(path: Path) -> dict[str, ModelMeta]:
    metadata: dict[str, ModelMeta] = {}
    for row in parse_markdown_table(path):
        model = ModelMeta(
            family=row["Model family"],
            version=row["Specific model version"],
            release_date=parse_date(row["Release date"]),
            origin=row["Origin"],
            open_weights=parse_bool(row.get("Open weights", "false")),
        )
        metadata[normalize_key(model.version)] = model
    return metadata


def load_results(path: Path, metadata: dict[str, ModelMeta]) -> tuple[list[RunResult], list[str]]:
    runs: list[RunResult] = []
    unmatched: list[str] = []
    for row in parse_markdown_table(path):
        model = find_model(row["Agent"], metadata)
        if model is None:
            unmatched.append(row["Agent"])
        runs.append(
            RunResult(
                run_id=row["Run ID"],
                run_date=parse_run_date(row["Run ID"]),
                agent=row["Agent"],
                harness=row.get("Harness", "") or infer_harness(row["Agent"]),
                effort=row["Effort"] or "unspecified",
                best_score=parse_int(row["Best Score"]) or 0,
                best_round=parse_int(row["Best Round"]),
                rounds=parse_int(row["Rounds"]),
                stop_reason=row["Stop Reason"],
                wall_seconds=parse_duration_seconds(row["Wall Time"]),
                agent_chars=parse_int(row["Agent Chars"]),
                code_lines_added=parse_int(row["Code Lines Added"]),
                openrouter_calls=parse_int(row["OpenRouter Calls"]),
                openrouter_cost=parse_cost(row["OpenRouter Cost"]),
                openrouter_tokens=parse_int(row["OpenRouter Tokens"]),
                model=model,
            )
        )
    return sorted(runs, key=lambda run: (run.release_date or run.run_date, run.run_id)), unmatched


def find_model(agent: str, metadata: dict[str, ModelMeta]) -> ModelMeta | None:
    for candidate in candidate_model_keys(agent):
        if candidate in metadata:
            return metadata[candidate]
    return None


def infer_harness(agent: str) -> str:
    normalized = agent.strip().lower()
    if normalized.startswith("grok-"):
        return "grokbuild"
    if normalized.startswith("opencode-"):
        return "opencode"
    if normalized.startswith("claude-code-") or normalized.startswith("claude-"):
        return "claudecode"
    if normalized.startswith("gemini"):
        return "antigravity"
    return "codex"


def tint(color: str, amount: float) -> str:
    channels = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return "#" + "".join(f"{round(channel + (255 - channel) * amount):02x}" for channel in channels)


def family_color_map(families: list[str], named_families: list[str]) -> dict[str, str]:
    return {family: FAMILY_COLORS[family] if family in named_families else OTHER_COLOR for family in families}


def family_legend(family_rows: list[dict[str, object]], named_families: list[str]) -> dict[str, str]:
    legend = {
        str(row["label"]): FAMILY_COLORS[str(row["label"])]
        for row in family_rows
        if row["label"] in named_families
    }
    if any(row["label"] not in named_families for row in family_rows):
        legend[OTHER_FAMILIES_LABEL] = OTHER_COLOR
    return legend


def model_style_map(runs: list[RunResult]) -> dict[str, dict[str, str]]:
    ordered_versions = sorted(
        {run.version: run.release_date or run.run_date for run in runs}.items(),
        key=lambda item: (item[1], item[0]),
    )
    styles: dict[str, dict[str, str]] = {}
    for index, (version, _) in enumerate(ordered_versions):
        styles[version] = {
            "color": MODEL_PALETTE[index % len(MODEL_PALETTE)],
            "pattern": MODEL_PATTERNS[index % len(MODEL_PATTERNS)],
            "pattern_id": f"model-pattern-{index}",
        }
    return styles


def best_runs_by_version(runs: list[RunResult]) -> list[RunResult]:
    best_runs: dict[str, RunResult] = {}
    for run in runs:
        current = best_runs.get(run.version)
        if current is None or (run.best_score, run.run_date, run.run_id) > (
            current.best_score,
            current.run_date,
            current.run_id,
        ):
            best_runs[run.version] = run
    return sorted(best_runs.values(), key=lambda run: (run.release_date or run.run_date, run.version, run.run_id))


def aggregate_scores(runs: list[RunResult], key: str, count_label: str = "model") -> list[dict[str, object]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        buckets[getattr(run, key)].append(run.best_score)
    rows: list[dict[str, object]] = []
    for label, scores in buckets.items():
        rows.append(
            {
                "label": label,
                "best": max(scores),
                "count": len(scores),
                "count_label": count_label,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["best"]), str(row["label"])))


def setting_label(run: RunResult, include_harness: bool) -> str:
    if not include_harness:
        return run.effort
    return run.harness if run.effort == "unspecified" else f"{run.harness} · {run.effort}"


def leaderboard_rows(runs: list[RunResult]) -> list[dict[str, object]]:
    best_by_setting: dict[str, dict[tuple[str, str], RunResult]] = defaultdict(dict)
    for run in runs:
        setting = (run.harness, run.effort)
        current = best_by_setting[run.version].get(setting)
        if current is None or run.best_score > current.best_score:
            best_by_setting[run.version][setting] = run
    rows: list[dict[str, object]] = []
    for version, setting_runs in best_by_setting.items():
        ordered = sorted(setting_runs.values(), key=lambda run: (-run.best_score, run.effort))
        include_harness = len({run.harness for run in ordered}) > 1
        rows.append(
            {
                "label": version,
                "family": ordered[0].family,
                "best": ordered[0].best_score,
                "settings": [(setting_label(run, include_harness), run.best_score) for run in ordered],
            }
        )
    return sorted(rows, key=lambda row: (-int(row["best"]), str(row["label"])))


def html_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt_date(value: dt.date) -> str:
    return f"{value.strftime('%b')} {value.day}, {value.year}"


def fmt_short_date(value: dt.date) -> str:
    return f"{value.strftime('%b')} {value.day}"


def fmt_month_year(value: dt.date) -> str:
    return f"{value.strftime('%b')} {value.year}"


def fmt_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}"
    return f"{int(value):,}"


def fmt_compact_number(value: float | int) -> str:
    absolute = abs(float(value))
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}".rstrip("0").rstrip(".") + "K"
    return fmt_number(value)


def fmt_duration(seconds: int | None) -> str:
    if seconds is None:
        return ""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def fmt_count(count: object, singular: object) -> str:
    count_int = int(count)
    singular_text = str(singular)
    suffix = singular_text if count_int == 1 else f"{singular_text}s"
    return f"{count_int} {suffix}"


def linear_ticks(max_value: float, max_intervals: int = 8) -> list[float]:
    if max_value <= 0:
        return [0, 1]
    magnitude = 10 ** math.floor(math.log10(max_value / max_intervals))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if math.ceil(max_value / step) <= max_intervals:
            break
    return [step * index for index in range(math.ceil(max_value / step) + 1)]


def score_axis_max(scores: list[int]) -> int:
    return max(50, math.ceil(max(scores) / 50) * 50)


def scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def chart_shell(title: str, subtitle: str, svg: str) -> str:
    return f"""
    <section class="chart-panel">
      <div class="chart-heading">
        <h2>{html_escape(title)}</h2>
        <p>{html_escape(subtitle)}</p>
      </div>
      {svg}
    </section>
    """


def svg_model_pattern_defs(styles: dict[str, dict[str, str]]) -> str:
    if not styles:
        return ""
    elements = ["<defs>"]
    for style in styles.values():
        pattern_id = style["pattern_id"]
        color = style["color"]
        pattern = style["pattern"]
        elements.append(f'<pattern id="{pattern_id}" patternUnits="userSpaceOnUse" width="8" height="8">')
        elements.append(f'<rect width="8" height="8" fill="{color}" />')
        if pattern == "diagonal":
            elements.append('<path d="M-2 8 L8 -2 M0 10 L10 0" stroke="#ffffff" stroke-width="1.2" opacity="0.75" />')
        elif pattern == "cross":
            elements.append('<path d="M-2 8 L8 -2 M0 10 L10 0 M-2 0 L8 10 M0 -2 L10 8" stroke="#ffffff" stroke-width="1" opacity="0.7" />')
        elif pattern == "dots":
            elements.append('<circle cx="2" cy="2" r="1.1" fill="#ffffff" opacity="0.8" />')
            elements.append('<circle cx="6" cy="6" r="1.1" fill="#ffffff" opacity="0.8" />')
        elif pattern == "vertical":
            elements.append('<path d="M2 0 V8 M6 0 V8" stroke="#ffffff" stroke-width="1" opacity="0.7" />')
        elif pattern == "horizontal":
            elements.append('<path d="M0 2 H8 M0 6 H8" stroke="#ffffff" stroke-width="1" opacity="0.7" />')
        elements.append("</pattern>")
    elements.append("</defs>")
    return "\n".join(elements)


def family_icon_id(family: str, prefix: str = "family-icon") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")
    return f"{prefix}-{slug}"


def svg_family_icon_defs(
    family_icons: dict[str, str],
    prefix: str = "family-icon",
) -> str:
    if not family_icons:
        return ""
    elements = ["<defs>"]
    for family, data_uri in sorted(family_icons.items()):
        elements.append(
            f'<image id="{family_icon_id(family, prefix)}" href="{data_uri}" x="-7.5" y="-7.5" '
            'width="15" height="15" preserveAspectRatio="xMidYMid meet" />'
        )
    elements.append("</defs>")
    return "\n".join(elements)


def html_model_label(
    label: object,
    family: str,
    family_icons: dict[str, str],
) -> str:
    escaped_label = html_escape(label)
    if family not in family_icons:
        return escaped_label
    return (
        '<span class="model-name-with-icon">'
        f'<span class="model-family-icon {family_icon_id(family)}" aria-hidden="true"></span>'
        f"<span>{escaped_label}</span>"
        "</span>"
    )


def html_family_icon_css(family_icons: dict[str, str]) -> str:
    return "\n".join(
        f'.{family_icon_id(family)} {{ background-image: url("{data_uri}"); }}'
        for family, data_uri in sorted(family_icons.items())
    )


def svg_model_legend(
    styles: dict[str, dict[str, str]],
    x: int,
    y: int,
    family_by_version: dict[str, str],
    family_icons: dict[str, str],
    columns: int = 2,
) -> str:
    labels = list(styles)
    rows = math.ceil(len(labels) / columns)
    column_width = 190
    row_height = 24
    elements = ['<g class="legend model-legend">']
    for index, label in enumerate(labels):
        column = index // rows
        row = index % rows
        style = styles[label]
        marker_x = x + column * column_width
        marker_y = y + row * row_height
        family = family_by_version[label]
        if family in family_icons:
            elements.append(
                f'<g transform="translate({marker_x + 6} {marker_y + 6})">'
                '<circle r="7" fill="#ffffff" stroke="#94a3b8" stroke-width="1.25" />'
                f'<use href="#{family_icon_id(family)}" transform="scale(0.78)" pointer-events="none" />'
                "</g>"
            )
        else:
            elements.append(
                f'<circle cx="{marker_x + 6}" cy="{marker_y + 6}" r="6" fill="url(#{style["pattern_id"]})" '
                f'stroke="{style["color"]}" stroke-width="1.5" />'
            )
        elements.append(f'<text x="{marker_x + 18}" y="{marker_y + 10}">{html_escape(label)}</text>')
    elements.append("</g>")
    return "\n".join(elements)


def svg_bullseye_marker(
    x: float,
    y: float,
    main_color: str,
    accent_color: str,
    title: str | None = None,
) -> str:
    title_element = f"<title>{html_escape(title)}</title>" if title else ""
    return (
        f'<g class="origin-marker" transform="translate({x:.1f} {y:.1f})">'
        f"{title_element}"
        f'<circle r="6" fill="{main_color}" />'
        '<circle r="4" fill="#ffffff" />'
        f'<circle r="2.2" fill="{accent_color}" />'
        "</g>"
    )


def svg_two_color_legend(
    colors: dict[str, str],
    accents: dict[str, str],
    x: int,
    y: int,
) -> str:
    elements = ['<g class="legend origin-legend">']
    for index, (label, color) in enumerate(colors.items()):
        row_y = y + index * 24
        accent = accents.get(label, "#cbd5e1")
        elements.append(
            f'<line x1="{x}" x2="{x + 20}" y1="{row_y + 6}" y2="{row_y + 6}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round" />'
        )
        elements.append(svg_bullseye_marker(x + 10, row_y + 6, color, accent))
        elements.append(f'<text x="{x + 30}" y="{row_y + 10}">{html_escape(label)}</text>')
    elements.append("</g>")
    return "\n".join(elements)


def svg_model_release_date_scatter(
    runs: list[RunResult],
    family_icons: dict[str, str] | None = None,
) -> str:
    family_icons = family_icons or {}
    plotted = [run for run in runs if run.release_date]
    plotted.sort(key=lambda run: (run.release_date or run.run_date, run.version, run.run_id))
    styles = model_style_map(plotted)
    family_by_version = {run.version: run.family for run in plotted}
    fallback_styles = {
        version: style
        for version, style in styles.items()
        if family_by_version[version] not in family_icons
    }
    width = 1180
    height = 560
    left = 70
    right = 420
    top = 30
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_score = BENCHMARK_FINAL_LEVEL
    min_date = min(run.release_date for run in plotted if run.release_date)
    max_date = max(run.release_date for run in plotted if run.release_date)
    total_days = max(1, (max_date - min_date).days)

    elements = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Best score by model release date and model version">',
        svg_model_pattern_defs(fallback_styles),
        svg_family_icon_defs(family_icons),
        grid_lines_for_values(
            left, top, plot_w, plot_h, 0, max_score,
            [0, 200, 400, 600, 800, 1000, max_score], float, fmt_number,
        ),
        axis_labels(left, top, plot_w, plot_h, "Model release date", "Best score"),
        f'<line class="benchmark-ceiling" x1="{left}" x2="{left + plot_w}" '
        f'y1="{top}" y2="{top}" stroke="#527568" stroke-width="1.5" stroke-dasharray="5 4" />',
        f'<text x="{left + 8}" y="{top - 12}" fill="#527568" font-size="12">'
        f'All levels solved ({fmt_number(BENCHMARK_FINAL_LEVEL)})</text>',
    ]

    points: list[tuple[RunResult, float, float]] = []
    for run in plotted:
        release_date = run.release_date or run.run_date
        x = scale((release_date - min_date).days, 0, total_days, left, left + plot_w)
        y = scale(run.best_score, 0, max_score, top + plot_h, top)
        points.append((run, x, y))

    by_version: dict[str, list[tuple[RunResult, float, float]]] = defaultdict(list)
    for point in points:
        by_version[point[0].version].append(point)

    for version, version_points in by_version.items():
        if len(version_points) < 2:
            continue
        x = version_points[0][1]
        y_values = [point[2] for point in version_points]
        style = styles[version]
        line_color = SINGLE_SERIES_COLOR if family_by_version[version] in family_icons else style["color"]
        elements.append(
            f'<line class="same-model-line" x1="{x:.1f}" x2="{x:.1f}" '
            f'y1="{min(y_values):.1f}" y2="{max(y_values):.1f}" stroke="{line_color}">'
            f"<title>{html_escape(version)} repeated runs</title></line>"
        )

    for run, x, y in points:
        style = styles[run.version]
        tooltip = f"Released {fmt_date(run.release_date or run.run_date)} - {run.version}: {run.best_score} ({run.run_id})"
        if run.best_score == BENCHMARK_FINAL_LEVEL:
            tooltip += " - All levels solved; maximum possible score"
            elements.append(
                f'<circle class="completed-model-ring" cx="{x:.1f}" cy="{y:.1f}" '
                'r="14" fill="none" stroke="#527568" stroke-width="2">'
                f'<title>{html_escape(run.version)}: all levels solved; maximum possible score</title></circle>'
            )
        if run.family in family_icons:
            elements.append(
                f'<g class="data-point model-point model-icon-point" transform="translate({x:.1f} {y:.1f})">'
                f"<title>{html_escape(tooltip)}</title>"
                '<circle r="10.5" fill="#ffffff" stroke="#64748b" stroke-width="1.6" />'
                f'<use href="#{family_icon_id(run.family)}" pointer-events="none" />'
                "</g>"
            )
        else:
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.5" fill="url(#{style["pattern_id"]})" '
                f'class="data-point model-point" style="stroke: {style["color"]}; stroke-width: 2.2;">'
                f"<title>{html_escape(tooltip)}</title></circle>"
            )

    for tick_index in range(5):
        tick_date = min_date + dt.timedelta(days=round(total_days * tick_index / 4))
        x = scale((tick_date - min_date).days, 0, total_days, left, left + plot_w)
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 34}" text-anchor="middle">{fmt_month_year(tick_date)}</text>')

    elements.append(
        svg_model_legend(
            styles,
            left + plot_w + 30,
            top + 4,
            family_by_version,
            family_icons,
            columns=2,
        )
    )
    elements.append("</svg>")
    return "\n".join(elements)


def svg_release_date_scatter(
    runs: list[RunResult],
    colors: dict[str, str],
    color_key: str,
    *,
    accent_colors: dict[str, str] | None = None,
    width: int = 1180,
    height: int = 430,
    left: int = 70,
    right: int = 190,
    top: int = 30,
    bottom: int = 72,
    aria_label: str = "Best score by model release date",
) -> str:
    accent_colors = accent_colors or {}
    plotted = [run for run in runs if run.release_date]
    plotted.sort(key=lambda run: (run.release_date or run.run_date, run.version, run.run_id))
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_score = score_axis_max([run.best_score for run in plotted])
    min_date = min(run.release_date for run in plotted if run.release_date)
    max_date = max(run.release_date for run in plotted if run.release_date)
    total_days = max(1, (max_date - min_date).days)

    elements = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(aria_label)}">',
        grid_lines(left, top, plot_w, plot_h, max_score),
        axis_labels(left, top, plot_w, plot_h, "Model release date", "Best score"),
    ]

    points: list[tuple[float, float]] = []
    for run in plotted:
        release_date = run.release_date or run.run_date
        x = scale((release_date - min_date).days, 0, total_days, left, left + plot_w)
        y = scale(run.best_score, 0, max_score, top + plot_h, top)
        points.append((x, y))

    for run, (x, y) in zip(plotted, points):
        color_label = str(getattr(run, color_key))
        color = colors.get(color_label, "#64748b")
        tooltip = f"Released {fmt_date(run.release_date or run.run_date)} - {run.version}: {run.best_score} ({color_label})"
        if color_label in accent_colors:
            elements.append(
                svg_bullseye_marker(
                    x,
                    y,
                    color,
                    accent_colors[color_label],
                    tooltip,
                )
            )
        else:
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{color}" class="data-point">'
                f"<title>{html_escape(tooltip)}</title></circle>"
            )

    for tick_index in range(5):
        tick_date = min_date + dt.timedelta(days=round(total_days * tick_index / 4))
        x = scale((tick_date - min_date).days, 0, total_days, left, left + plot_w)
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 34}" text-anchor="middle">{fmt_month_year(tick_date)}</text>')

    if accent_colors:
        elements.append(svg_two_color_legend(colors, accent_colors, width - right + 28, top + 4))
    else:
        elements.append(svg_legend(colors, width - right + 28, top + 4))
    elements.append("</svg>")
    return "\n".join(elements)


def svg_score_over_time(
    runs: list[RunResult],
    family_icons: dict[str, str] | None = None,
) -> str:
    return svg_model_release_date_scatter(runs, family_icons)


def cumulative_ai_progress(runs: list[RunResult]) -> list[tuple[dt.date, int, RunResult]]:
    release_rows = [run for run in runs if run.release_date]
    by_date: dict[dt.date, list[RunResult]] = defaultdict(list)
    for run in release_rows:
        by_date[run.release_date or run.run_date].append(run)

    best_score = 0
    best_run: RunResult | None = None
    points: list[tuple[dt.date, int, RunResult]] = []
    for release_date in sorted(by_date):
        day_best = max(by_date[release_date], key=lambda run: run.best_score)
        if day_best.best_score > best_score or best_run is None:
            best_score = day_best.best_score
            best_run = day_best
        points.append((release_date, best_score, best_run))
    return points


def svg_ai_vs_human_metric_over_time(
    runs: list[RunResult],
    metric_for_level: Callable[[int], int],
    axis_max_for_values: Callable[[list[int]], int],
    tick_formatter: Callable[[float | int], str],
    metric_tooltip: Callable[[int], str],
    y_label: str,
    aria_label: str,
    progress_title: str,
    human_label: str,
    brute_force_label: str,
    axis_min_value: int = 0,
    value_transform: Callable[[float | int], float] = float,
    grid_values: list[int] | None = None,
) -> str:
    points = cumulative_ai_progress(runs)

    width = 650
    height = 360
    left = 72
    right = 190
    top = 34
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [metric_for_level(level) for _, level, _ in points]
    human_value = metric_for_level(HUMAN_BEST_SCORE)
    brute_force_value = metric_for_level(DEFAULT_BRUTE_FORCE_SCORE)
    max_value = axis_max_for_values([human_value, brute_force_value, *values])
    axis_min_plot = value_transform(axis_min_value)
    axis_max_plot = value_transform(max_value)

    def y_position(value: int) -> float:
        return scale(value_transform(value), axis_min_plot, axis_max_plot, top + plot_h, top)

    min_date = min(release_date for release_date, _, _ in points)
    max_date = max(release_date for release_date, _, _ in points)
    total_days = max(1, (max_date - min_date).days)
    ai_label = "AI best so far"
    colors = {
        ai_label: "#2563eb",
        human_label: HUMAN_COLOR,
        brute_force_label: BRUTE_FORCE_COLOR,
    }

    elements = [
        f'<svg class="chart-svg compact" viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(aria_label)}">',
        grid_lines(left, top, plot_w, plot_h, max_value, tick_formatter)
        if grid_values is None
        else grid_lines_for_values(
            left,
            top,
            plot_w,
            plot_h,
            axis_min_value,
            max_value,
            grid_values,
            value_transform,
            tick_formatter,
        ),
        axis_labels(left, top, plot_w, plot_h, "Model release date", y_label),
    ]

    for label, level, value in [
        (human_label, HUMAN_BEST_SCORE, human_value),
        (brute_force_label, DEFAULT_BRUTE_FORCE_SCORE, brute_force_value),
    ]:
        y = y_position(value)
        color = colors[label]
        elements.append(
            f'<line class="reference-line" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}">'
            f"<title>{html_escape(f'{label}: {metric_tooltip(level)}')}</title></line>"
        )

    scaled: list[tuple[dt.date, int, int, RunResult, float, float]] = []
    for release_date, level, run in points:
        value = metric_for_level(level)
        x = scale((release_date - min_date).days, 0, total_days, left, left + plot_w)
        y = y_position(value)
        scaled.append((release_date, level, value, run, x, y))

    path_parts: list[str] = []
    for index, (_, _, _, _, x, y) in enumerate(scaled):
        if index == 0:
            path_parts.append(f"M {x:.1f} {y:.1f}")
            continue
        previous_y = scaled[index - 1][5]
        path_parts.append(f"L {x:.1f} {previous_y:.1f}")
        path_parts.append(f"L {x:.1f} {y:.1f}")
    if path_parts:
        elements.append(
            f'<path class="progress-line" d="{" ".join(path_parts)}" stroke="{colors[ai_label]}">'
            f"<title>{html_escape(progress_title)}</title></path>"
        )

    records = [point for index, point in enumerate(scaled) if index == 0 or point[1] > scaled[index - 1][1]]
    for release_date, level, _, run, x, y in records:
        tooltip = (
            f"AI best so far {metric_tooltip(level)} as of {fmt_date(release_date)} "
            f"from {run.version} ({run.run_id}, run {fmt_date(run.run_date)})"
        )
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" fill="{colors[ai_label]}" class="progress-point">'
            f"<title>{html_escape(tooltip)}</title></circle>"
        )
    step_boxes: list[tuple[float, float, float, float]] = []
    for (*_, x0, y0), (*_, x1, y1) in zip(scaled, scaled[1:]):
        step_boxes.append((x0 - 1, y0 - 1, x1 + 1, y0 + 1))
        step_boxes.append((x1 - 1, min(y0, y1) - 1, x1 + 1, max(y0, y1) + 1))
    step_boxes.extend((x - 7, y - 7, x + 7, y + 7) for *_, x, y in records)
    elements.append(
        svg_point_labels(
            [(x, y, run.version) for _, _, _, run, x, y in sorted(records, key=lambda point: -point[1])],
            (left, top - 20, left + plot_w, top + plot_h),
            sides=("left",),
            obstacles=step_boxes,
        )
    )

    for tick_index in range(3):
        tick_date = min_date + dt.timedelta(days=round(total_days * tick_index / 2))
        x = scale((tick_date - min_date).days, 0, total_days, left, left + plot_w)
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 34}" text-anchor="middle">{fmt_date(tick_date)}</text>')

    elements.append(svg_legend(colors, width - right + 20, top + 4))
    elements.append("</svg>")
    return "\n".join(elements)


def svg_ai_vs_human_level_over_time(runs: list[RunResult]) -> str:
    return svg_ai_vs_human_metric_over_time(
        runs,
        lambda level: level,
        score_axis_max,
        fmt_number,
        lambda level: f"level {fmt_number(level)}",
        "Highest level",
        "AI benchmark progress by model release date versus human and default solver levels",
        "AI best level so far by model release date",
        f"Human best ({HUMAN_BEST_SCORE})",
        f"Bundled brute force ({DEFAULT_BRUTE_FORCE_SCORE})",
    )


def svg_ai_vs_human_board_area_over_time(
    runs: list[RunResult],
    dimensions: dict[int, tuple[int, int]],
) -> str:
    def metric_for_level(level: int) -> int:
        return board_area_for_level(level, dimensions)

    def axis_max_for_values(values: list[int]) -> int:
        return max(1_000_000, math.ceil(max(values) / 1_000_000) * 1_000_000)

    def metric_tooltip(level: int) -> str:
        width, height = board_dimensions_for_level(level, dimensions)
        area = width * height
        return f"level {fmt_number(level)}, {fmt_number(width)} x {fmt_number(height)} ({fmt_number(area)} cells)"

    human_area = metric_for_level(HUMAN_BEST_SCORE)
    brute_force_area = metric_for_level(DEFAULT_BRUTE_FORCE_SCORE)
    return svg_ai_vs_human_metric_over_time(
        runs,
        metric_for_level,
        axis_max_for_values,
        fmt_compact_number,
        metric_tooltip,
        "Largest board (cells, log scale)",
        "Largest Mortal Coil board area solved by AI on a logarithmic cell scale over model release date versus human and default solver results",
        "AI largest board area solved so far by model release date",
        f"Human ({fmt_compact_number(human_area)} cells)",
        f"Brute force ({fmt_compact_number(brute_force_area)} cells)",
        axis_min_value=100,
        value_transform=lambda value: math.log10(max(1, value)),
        grid_values=[100, 1_000, 10_000, 100_000, 1_000_000, 4_000_000],
    )


def svg_group_best_over_time(
    runs: list[RunResult],
    colors: dict[str, str],
    group_key: str,
    y_label: str,
    aria_label: str,
    accent_colors: dict[str, str] | None = None,
    legend_colors: dict[str, str] | None = None,
) -> str:
    accent_colors = accent_colors or {}
    legend_colors = legend_colors or colors
    release_rows = [run for run in runs if run.release_date]
    by_group_date: dict[str, dict[dt.date, int]] = defaultdict(dict)
    for run in release_rows:
        release_date = run.release_date or run.run_date
        group = str(getattr(run, group_key))
        current = by_group_date[group].get(release_date, 0)
        by_group_date[group][release_date] = max(current, run.best_score)

    group_points: dict[str, list[tuple[dt.date, int]]] = {}
    for group, dated_scores in by_group_date.items():
        best_so_far = 0
        points: list[tuple[dt.date, int]] = []
        for release_date, score in sorted(dated_scores.items()):
            best_so_far = max(best_so_far, score)
            points.append((release_date, best_so_far))
        group_points[group] = points

    width = 1180
    height = 430
    left = 70
    right = 190
    top = 30
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_score = score_axis_max([score for points in group_points.values() for _, score in points])
    min_date = min(run.release_date for run in release_rows if run.release_date)
    max_date = max(run.release_date for run in release_rows if run.release_date)
    total_days = max(1, (max_date - min_date).days)

    elements = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(aria_label)}">',
        grid_lines(left, top, plot_w, plot_h, max_score),
        axis_labels(left, top, plot_w, plot_h, "Model release date", y_label),
    ]

    for group in sorted(group_points, key=lambda group: (group in legend_colors, group)):
        points = group_points[group]
        color = colors.get(group, "#64748b")
        scaled: list[tuple[dt.date, int, float, float]] = []
        for release_date, score in points:
            x = scale((release_date - min_date).days, 0, total_days, left, left + plot_w)
            y = scale(score, 0, max_score, top + plot_h, top)
            scaled.append((release_date, score, x, y))

        path_parts: list[str] = []
        for index, (_, _, x, y) in enumerate(scaled):
            if index == 0:
                path_parts.append(f"M {x:.1f} {y:.1f}")
                continue
            previous_y = scaled[index - 1][3]
            path_parts.append(f"L {x:.1f} {previous_y:.1f}")
            path_parts.append(f"L {x:.1f} {y:.1f}")
        if path_parts:
            path_class = "progress-line origin-progress-line" if group in accent_colors else "progress-line"
            elements.append(
                f'<path class="{path_class}" d="{" ".join(path_parts)}" stroke="{color}">'
                f"<title>{html_escape(group)} best score over time</title></path>"
            )

        for index, (release_date, score, x, y) in enumerate(scaled):
            if index > 0 and score <= scaled[index - 1][1]:
                continue
            tooltip = f"{group}: best score so far {score} as of {fmt_date(release_date)}"
            if group in accent_colors:
                elements.append(svg_bullseye_marker(x, y, color, accent_colors[group], tooltip))
            else:
                elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" class="progress-point">'
                    f"<title>{html_escape(tooltip)}</title></circle>"
                )

    for tick_index in range(5):
        tick_date = min_date + dt.timedelta(days=round(total_days * tick_index / 4))
        x = scale((tick_date - min_date).days, 0, total_days, left, left + plot_w)
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 34}" text-anchor="middle">{fmt_month_year(tick_date)}</text>')

    if accent_colors:
        elements.append(svg_two_color_legend(colors, accent_colors, width - right + 28, top + 4))
    else:
        elements.append(svg_legend(legend_colors, width - right + 28, top + 4))
    elements.append("</svg>")
    return "\n".join(elements)


def svg_family_best_over_time(
    runs: list[RunResult],
    family_colors: dict[str, str],
    legend_colors: dict[str, str],
) -> str:
    return svg_group_best_over_time(
        runs,
        family_colors,
        "family",
        "Family best score so far",
        "Cumulative best score by model family over release date",
        legend_colors=legend_colors,
    )


def svg_origin_best_over_time(
    runs: list[RunResult],
    origin_colors: dict[str, str],
) -> str:
    return svg_group_best_over_time(
        runs,
        origin_colors,
        "origin",
        "Origin best score so far",
        "Cumulative best score by model origin over release date",
    )


def svg_open_weights_best_over_time(runs: list[RunResult]) -> str:
    return svg_group_best_over_time(
        runs,
        OPEN_WEIGHTS_COLORS,
        "weights_status",
        "Best score so far",
        "Cumulative best score by open weights status over release date",
    )


def grid_lines(
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    max_value: float,
    value_formatter: Callable[[float | int], str] = fmt_number,
) -> str:
    elements: list[str] = []
    for index in range(6):
        value = max_value * index / 5
        y = scale(value, 0, max_value, top + plot_h, top)
        elements.append(f'<line class="grid-line" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" />')
        elements.append(f'<text class="tick-label" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{value_formatter(value)}</text>')
    elements.append(f'<line class="axis-line" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" />')
    elements.append(f'<line class="axis-line" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" />')
    return "\n".join(elements)


def grid_lines_for_values(
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    min_value: float,
    max_value: float,
    tick_values: list[int],
    value_transform: Callable[[float | int], float],
    value_formatter: Callable[[float | int], str],
) -> str:
    transformed_min = value_transform(min_value)
    transformed_max = value_transform(max_value)
    elements: list[str] = []
    for value in sorted(set(tick_values)):
        if value < min_value or value > max_value:
            continue
        y = scale(value_transform(value), transformed_min, transformed_max, top + plot_h, top)
        elements.append(f'<line class="grid-line" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" />')
        elements.append(f'<text class="tick-label" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{value_formatter(value)}</text>')
    elements.append(f'<line class="axis-line" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" />')
    elements.append(f'<line class="axis-line" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" />')
    return "\n".join(elements)


def axis_labels(left: int, top: int, plot_w: int, plot_h: int, x_label: str, y_label: str) -> str:
    return "\n".join(
        [
            f'<text class="axis-title" x="{left + plot_w / 2:.1f}" y="{top + plot_h + 58}" text-anchor="middle">{html_escape(x_label)}</text>',
            f'<text class="axis-title" transform="translate(20 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">{html_escape(y_label)}</text>',
        ]
    )


def svg_legend(colors: dict[str, str], x: int, y: int) -> str:
    elements = ['<g class="legend">']
    for index, (label, color) in enumerate(colors.items()):
        row_y = y + index * 24
        elements.append(f'<rect x="{x}" y="{row_y}" width="12" height="12" rx="2" fill="{color}" />')
        elements.append(f'<text x="{x + 20}" y="{row_y + 10}">{html_escape(label)}</text>')
    elements.append("</g>")
    return "\n".join(elements)


def svg_point_labels(
    candidates: list[tuple[float, float, str]],
    bounds: tuple[float, float, float, float],
    sides: tuple[str, ...] = ("right", "left"),
    obstacles: list[tuple[float, float, float, float]] | None = None,
) -> str:
    char_width = 6.8
    min_x, min_y, max_x, max_y = bounds
    placed = list(obstacles or [])
    elements = ['<g class="point-labels">']
    for x, y, text in candidates:
        text_width = len(text) * char_width
        for side in sides:
            anchor_x = x + 10 if side == "right" else x - 10
            box_left = anchor_x if side == "right" else anchor_x - text_width
            box = (box_left, y - 9, box_left + text_width, y + 7)
            inside = min_x <= box[0] and box[2] <= max_x and min_y <= box[1] and box[3] <= max_y
            clear = all(
                box[2] < other[0] or other[2] < box[0] or box[3] < other[1] or other[3] < box[1]
                for other in placed
            )
            if inside and clear:
                placed.append(box)
                anchor = "start" if side == "right" else "end"
                elements.append(
                    f'<text class="point-label" x="{anchor_x:.1f}" y="{y + 4:.1f}" '
                    f'text-anchor="{anchor}">{html_escape(text)}</text>'
                )
                break
    elements.append("</g>")
    return "\n".join(elements)


def svg_best_bars(
    rows: list[dict[str, object]],
    title: str,
    bar_colors: dict[str, str] | None = None,
) -> str:
    bar_colors = bar_colors or {}
    width = 650
    height = 360
    left = 64
    right = 28
    top = 34
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = score_axis_max([int(row["best"]) for row in rows])
    group_w = plot_w / max(1, len(rows))
    bar_w = min(34, group_w * 0.45)

    elements = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(title)}">',
        grid_lines(left, top, plot_w, plot_h, max_value),
    ]
    for group_index, row in enumerate(rows):
        center = left + group_w * group_index + group_w / 2
        value = float(row["best"])
        label = str(row["label"])
        x = center - bar_w / 2
        y = scale(value, 0, max_value, top + plot_h, top)
        height_px = top + plot_h - y
        color = bar_colors.get(label, SINGLE_SERIES_COLOR)
        elements.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height_px:.1f}" fill="{color}">'
            f"<title>{html_escape(label)} best: {fmt_number(value)}</title></rect>"
        )
        elements.append(
            f'<text class="tick-label" x="{center:.1f}" y="{height - 42}" text-anchor="middle">{html_escape(label)}</text>'
        )
        elements.append(
            f'<text class="muted-label" x="{center:.1f}" y="{height - 24}" text-anchor="middle">{html_escape(fmt_count(row["count"], row["count_label"]))}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements)


def svg_best_horizontal_bars(rows: list[dict[str, object]], title: str) -> str:
    width = 520
    row_h = 42
    top = 18
    bottom = 18
    left = 164
    right = 58
    height = top + bottom + row_h * len(rows)
    plot_w = width - left - right
    max_value = score_axis_max([int(row["best"]) for row in rows])

    elements = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(title)}">',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_h
        value = int(row["best"])
        width_px = scale(value, 0, max_value, 0, plot_w)
        label = html_escape(row["label"])
        count = html_escape(fmt_count(row["count"], row["count_label"]))
        elements.append(f'<text class="bar-label" x="{left - 12}" y="{y + 17}" text-anchor="end">{label}</text>')
        elements.append(f'<text class="muted-label" x="{left - 12}" y="{y + 37}" text-anchor="end">{count}</text>')
        elements.append(f'<rect class="bar-track" x="{left}" y="{y + 10}" width="{plot_w}" height="20" rx="4" />')
        elements.append(
            f'<rect class="bar" x="{left}" y="{y + 10}" width="{width_px:.1f}" height="20" rx="4" fill="{SINGLE_SERIES_COLOR}">'
            f"<title>{label} best: {value}; {count}</title></rect>"
        )
        elements.append(f'<text class="value-label" x="{left + width_px + 8:.1f}" y="{y + 25}">{value}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def svg_leaderboard_part(
    rows: list[dict[str, object]],
    colors: dict[str, str],
    family_icons: dict[str, str],
    max_value: int,
    first: bool,
) -> str:
    width = 1180
    row_h = 34
    setting_h = 20
    top = 44 if first else 6
    bottom = 10
    left = 330
    right = 80
    plot_w = width - left - right
    height = top + bottom + sum(row_h + setting_h * (len(row["settings"]) - 1) for row in rows)
    tracks: list[str] = []
    bars: list[str] = []
    labels: list[str] = []
    y = top
    for row in rows:
        label = str(row["label"])
        family = str(row["family"])
        color = colors.get(family, OTHER_COLOR)
        (best_setting, best), *other_settings = row["settings"]
        if family in family_icons:
            icon = f'<use href="#{family_icon_id(family, "leaderboard-icon")}" pointer-events="none" />'
        else:
            icon = f'<text class="icon-initial" y="4" text-anchor="middle">{html_escape(family[:1])}</text>'
        labels.append(
            f'<g transform="translate(20 {y + 16})">'
            '<circle r="10" fill="#ffffff" stroke="#94a3b8" stroke-width="1.25" />'
            f"{icon}</g>"
        )
        labels.append(f'<text class="bar-label" x="38" y="{y + 22}">{html_escape(label)}</text>')
        labels.append(
            f'<text class="setting-label" x="{left - 10}" y="{y + 21}" text-anchor="end">{html_escape(best_setting)}</text>'
        )
        width_px = scale(best, 0, max_value, 0, plot_w)
        tracks.append(f'<rect class="bar-track" x="{left}" y="{y + 7}" width="{plot_w}" height="18" rx="4" />')
        bars.append(
            f'<rect class="bar" x="{left}" y="{y + 7}" width="{width_px:.1f}" height="18" rx="4" fill="{color}">'
            f"<title>{html_escape(label)}: best {best} ({html_escape(best_setting)})</title></rect>"
        )
        labels.append(f'<text class="value-label" x="{left + width_px + 8:.1f}" y="{y + 22}">{best}</text>')
        for index, (setting, value) in enumerate(other_settings):
            setting_y = y + row_h + index * setting_h
            width_px = scale(value, 0, max_value, 0, plot_w)
            bars.append(
                f'<rect class="bar" x="{left}" y="{setting_y + 2}" width="{width_px:.1f}" height="8" rx="3" '
                f'fill="{tint(color, 0.5)}"><title>{html_escape(label)} at {html_escape(setting)}: best {value}</title></rect>'
            )
            labels.append(
                f'<text class="setting-label" x="{left - 10}" y="{setting_y + 10}" text-anchor="end">{html_escape(setting)}</text>'
            )
            labels.append(f'<text class="setting-label" x="{left + width_px + 8:.1f}" y="{setting_y + 10}">{value}</text>')
        y += row_h + setting_h * len(other_settings)

    references: list[str] = []
    for name, value, color, anchor in [
        ("Brute force", DEFAULT_BRUTE_FORCE_SCORE, BRUTE_FORCE_COLOR, "start"),
        ("Human best", HUMAN_BEST_SCORE, HUMAN_COLOR, "end"),
    ]:
        x = left + scale(value, 0, max_value, 0, plot_w)
        references.append(
            f'<line class="reference-line" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{height - bottom}" '
            f'stroke="{color}"><title>{name}: {value}</title></line>'
        )
        if first:
            label_x = x + 6 if anchor == "start" else x - 6
            labels.append(
                f'<text class="reference-label" x="{label_x:.1f}" y="{top - 16}" text-anchor="{anchor}">{name} ({value})</text>'
            )
    if first:
        labels.append(f'<text class="reference-label" x="{left - 10}" y="{top - 16}" text-anchor="end">Setting</text>')

    aria_label = "Leaderboard: best score by model version" if first else "Leaderboard: remaining models"
    return "\n".join(
        [
            f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{aria_label}">',
            svg_family_icon_defs(family_icons, prefix="leaderboard-icon") if first else "",
            *tracks,
            *references,
            *bars,
            *labels,
            "</svg>",
        ]
    )


def html_leaderboard(
    rows: list[dict[str, object]],
    colors: dict[str, str],
    family_icons: dict[str, str],
    visible_rows: int = 20,
) -> str:
    max_value = score_axis_max([int(row["best"]) for row in rows])
    html_output = svg_leaderboard_part(rows[:visible_rows], colors, family_icons, max_value, first=True)
    if len(rows) > visible_rows:
        rest = svg_leaderboard_part(rows[visible_rows:], colors, family_icons, max_value, first=False)
        html_output += (
            f'<details class="leaderboard-more"><summary>{len(rows) - visible_rows} more models</summary>{rest}</details>'
        )
    return html_output


def svg_count_bars(
    counts: Counter[str],
    title: str,
    *,
    width: int = 520,
    left: int = 170,
    right: int = 48,
) -> str:
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    row_h = 36
    top = 22
    bottom = 24
    height = top + bottom + row_h * len(rows)
    plot_w = width - left - right
    max_value = max(counts.values()) if counts else 1
    elements = [f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(title)}">']
    for index, (label, count) in enumerate(rows):
        y = top + index * row_h
        width_px = scale(count, 0, max_value, 0, plot_w)
        elements.append(f'<text class="bar-label" x="{left - 10}" y="{y + 23}" text-anchor="end">{html_escape(label)}</text>')
        elements.append(f'<rect class="bar-track" x="{left}" y="{y + 8}" width="{plot_w}" height="18" rx="4" />')
        elements.append(f'<rect class="bar" x="{left}" y="{y + 8}" width="{width_px:.1f}" height="18" rx="4" fill="{SINGLE_SERIES_COLOR}" />')
        elements.append(f'<text class="value-label" x="{left + width_px + 8:.1f}" y="{y + 23}">{count}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def svg_score_scatter(
    points: list[tuple[RunResult, float]],
    x_fraction: Callable[[float], float],
    x_ticks: list[float],
    x_tick_formatter: Callable[[float | int], str],
    x_label: str,
    aria_label: str,
    point_detail: Callable[[RunResult], str],
    colors: dict[str, str],
    legend_colors: dict[str, str],
    label_count: int = 8,
) -> str:
    width = 1180
    height = 430
    left = 70
    right = 190
    top = 30
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_score = score_axis_max([run.best_score for run, _ in points])
    elements = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(aria_label)}">',
        grid_lines(left, top, plot_w, plot_h, max_score),
        axis_labels(left, top, plot_w, plot_h, x_label, "Best score"),
    ]
    for tick in x_ticks:
        x = left + x_fraction(tick) * plot_w
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 38}" text-anchor="middle">{x_tick_formatter(tick)}</text>')
    scaled = [
        (run, left + x_fraction(value) * plot_w, scale(run.best_score, 0, max_score, top + plot_h, top))
        for run, value in points
    ]
    for run, x, y in sorted(scaled, key=lambda point: point[0].family in legend_colors):
        tooltip = f"{run.version}: {run.best_score}, {point_detail(run)}"
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{colors.get(run.family, OTHER_COLOR)}" class="data-point">'
            f"<title>{html_escape(tooltip)}</title></circle>"
        )
    elements.append(
        svg_point_labels(
            [(x, y, run.version) for run, x, y in sorted(scaled, key=lambda point: -point[0].best_score)[:label_count]],
            (left, top - 20, left + plot_w, top + plot_h),
            obstacles=[(x - 8, y - 8, x + 8, y + 8) for _, x, y in scaled],
        )
    )
    elements.append(svg_legend(legend_colors, width - right + 28, top + 4))
    elements.append("</svg>")
    return "\n".join(elements)


def svg_score_vs_wall_time(
    runs: list[RunResult],
    colors: dict[str, str],
    legend_colors: dict[str, str],
) -> str:
    points = [(run, (run.wall_seconds or 0) / 3600) for run in runs if run.wall_seconds]
    ticks = linear_ticks(max(hours for _, hours in points))
    return svg_score_scatter(
        points,
        lambda hours: hours / ticks[-1],
        ticks,
        fmt_number,
        "Wall time (hours)",
        "Score versus wall time",
        lambda run: fmt_duration(run.wall_seconds),
        colors,
        legend_colors,
    )


def svg_score_vs_code_lines(
    runs: list[RunResult],
    colors: dict[str, str],
    legend_colors: dict[str, str],
) -> str:
    points = [(run, float(run.code_lines_added or 0)) for run in runs if run.code_lines_added is not None]
    decades = max(1, math.ceil(math.log10(max(lines for _, lines in points) or 1)))
    ticks = [0, *(10**power for power in range(1, decades + 1))]
    return svg_score_scatter(
        points,
        lambda lines: math.log10(1 + lines) / math.log10(1 + ticks[-1]),
        ticks,
        fmt_compact_number,
        "Code lines added (log scale)",
        "Score versus code lines added on a log scale",
        lambda run: f"{fmt_number(run.code_lines_added)} code lines added",
        colors,
        legend_colors,
    )


def summary_cards(runs: list[RunResult]) -> str:
    best_run = max(runs, key=lambda run: run.best_score)
    newest_release = max((run for run in runs if run.release_date), key=lambda run: run.release_date or dt.date.min)
    cost_runs = [run for run in runs if run.openrouter_cost is not None]
    total_cost = sum(run.openrouter_cost or 0 for run in cost_runs)
    cost_harnesses = sorted({run.harness for run in cost_runs})
    cost_note = f"{len(cost_runs)} of {len(runs)} runs"
    if len(cost_harnesses) == 1:
        cost_note += f", {cost_harnesses[0]} only"
    metrics = [
        ("Runs", len(runs), ""),
        ("Models", len({run.version for run in runs}), ""),
        ("Top score", best_run.best_score, ""),
        ("Newest release", fmt_date(newest_release.release_date or newest_release.run_date), ""),
        ("Origins", len({run.origin for run in runs}), ""),
        ("OpenRouter cost", f"${total_cost:,.2f}" if total_cost else "$0.00", cost_note),
    ]
    cards = []
    for label, value, note in metrics:
        note_html = f"<small>{html_escape(note)}</small>" if note else ""
        cards.append(
            f"""
            <div class="metric">
              <span>{html_escape(label)}</span>
              <strong>{html_escape(value)}</strong>
              {note_html}
            </div>
            """
        )
    return '<div class="metrics">' + "\n".join(cards) + "</div>"


def data_table(
    runs: list[RunResult],
    family_icons: dict[str, str] | None = None,
) -> str:
    family_icons = family_icons or {}
    rows = []
    for run in sorted(runs, key=lambda item: (item.release_date or item.run_date, item.run_date), reverse=True):
        rows.append(
            "<tr>"
            f"<td>{html_escape(fmt_date(run.release_date or run.run_date))}</td>"
            f"<td>{html_model_label(run.version, run.family, family_icons)}</td>"
            f"<td>{html_escape(run.family)}</td>"
            f"<td>{html_escape(run.harness)}</td>"
            f"<td>{html_escape(run.origin)}</td>"
            f"<td>{str(run.open_weights).lower()}</td>"
            f"<td>{run.best_score}</td>"
            f"<td>{html_escape(run.effort)}</td>"
            f"<td>{html_escape(fmt_date(run.run_date))}</td>"
            f"<td>{html_escape(fmt_duration(run.wall_seconds))}</td>"
            f"<td>{html_escape(run.stop_reason)}</td>"
            "</tr>"
        )
    return f"""
    <section class="chart-panel">
      <div class="chart-heading">
        <h2>Run Details</h2>
        <p>Raw run rows used as source data. Score charts collapse repeated model versions to their maximum score and use model release date.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Release date</th>
              <th>Model version</th>
              <th>Model family</th>
              <th>Harness</th>
              <th>Origin</th>
              <th>Open weights</th>
              <th>Best score</th>
              <th>Effort</th>
              <th>Run date</th>
              <th>Wall time</th>
              <th>Stop reason</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </section>
    """


def model_run_count_table(
    runs: list[RunResult],
    family_icons: dict[str, str] | None = None,
) -> str:
    family_icons = family_icons or {}
    buckets: dict[str, list[RunResult]] = defaultdict(list)
    for run in runs:
        buckets[run.version].append(run)

    rows: list[dict[str, object]] = []
    for version, version_runs in buckets.items():
        best_run = max(version_runs, key=lambda run: run.best_score)
        families = sorted({run.family for run in version_runs})
        origins = sorted({run.origin for run in version_runs})
        harnesses = sorted({run.harness for run in version_runs})
        release_dates = sorted({run.release_date for run in version_runs if run.release_date})
        rows.append(
            {
                "version": version,
                "family": ", ".join(families),
                "icon_family": best_run.family,
                "origin": ", ".join(origins),
                "release_date": fmt_date(release_dates[0]) if release_dates else "",
                "harnesses": ", ".join(harnesses),
                "runs": len(version_runs),
                "best": best_run.best_score,
            }
        )

    rows.sort(key=lambda row: (-int(row["runs"]), str(row["version"]).lower()))
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html_model_label(row['version'], str(row['icon_family']), family_icons)}</td>"
            f"<td>{html_escape(row['family'])}</td>"
            f"<td>{html_escape(row['origin'])}</td>"
            f"<td>{html_escape(row['release_date'])}</td>"
            f"<td>{html_escape(row['harnesses'])}</td>"
            f"<td>{int(row['runs'])}</td>"
            f"<td>{int(row['best'])}</td>"
            "</tr>"
        )

    return f"""
    <section class="chart-panel">
      <div class="chart-heading">
        <h2>Runs By Model</h2>
        <p>Raw run counts grouped by normalized model version.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model version</th>
              <th>Model family</th>
              <th>Origin</th>
              <th>Release date</th>
              <th>Harnesses</th>
              <th>Runs</th>
              <th>Best score</th>
            </tr>
          </thead>
          <tbody>
            {''.join(body)}
          </tbody>
        </table>
      </div>
    </section>
    """


def render_html(
    runs: list[RunResult],
    unmatched: list[str],
    level_dimensions: dict[int, tuple[int, int]],
    family_icons: dict[str, str] | None = None,
) -> str:
    if not runs:
        raise ValueError("no runs to visualize")

    family_icons = family_icons or {}
    family_icon_css = html_family_icon_css(family_icons)
    result_runs = best_runs_by_version(runs)
    families = sorted({run.family for run in result_runs})
    family_colors = family_color_map(families, list(FAMILY_COLORS))
    scatter_colors = family_color_map(families, SCATTER_FAMILIES)
    origins = {run.origin for run in result_runs}
    origin_colors = {origin: color for origin, color in ORIGIN_COLORS.items() if origin in origins}
    for origin in sorted(origins - set(origin_colors)):
        origin_colors[origin] = OTHER_COLOR
    origin_rows = aggregate_scores(result_runs, "origin")
    family_rows = aggregate_scores(result_runs, "family")
    family_legend_colors = family_legend(family_rows, list(FAMILY_COLORS))
    scatter_legend_colors = family_legend(family_rows, SCATTER_FAMILIES)
    harness_rows = aggregate_scores(result_runs, "harness")
    family_counts = Counter(run.family for run in result_runs)
    effort_counts = Counter(run.effort for run in result_runs)
    stop_reason_counts = Counter(run.stop_reason.strip() or "unspecified" for run in runs)
    min_date = min(run.release_date for run in result_runs if run.release_date)
    max_date = max(run.release_date for run in result_runs if run.release_date)

    unmatched_html = ""
    if unmatched:
        unmatched_list = ", ".join(sorted(set(unmatched)))
        unmatched_html = f'<p class="warning">Metadata missing for: {html_escape(unmatched_list)}</p>'

    charts = [
        chart_shell(
            "Leaderboard",
            "Each model's best score. Models run at more than one setting get a thinner row for each other setting's best.",
            html_leaderboard(leaderboard_rows(runs), family_colors, family_icons),
        ),
        chart_shell(
            "All LLMs By Release Date",
            "Each normalized model version plotted once by public release date, using its maximum score across all runs.",
            svg_score_over_time(result_runs, family_icons),
        ),
        '<div class="chart-grid">',
        chart_shell(
            "AI Versus Humans: Highest Level",
            "Cumulative best level reached by release date, compared with the human and bundled brute-force baselines.",
            svg_ai_vs_human_level_over_time(result_runs),
        ),
        chart_shell(
            "AI Versus Humans: Largest Board",
            "The same results translated into the largest board area solved.",
            svg_ai_vs_human_board_area_over_time(result_runs, level_dimensions),
        ),
        "</div>",
        chart_shell(
            "Family Best Score Over Time",
            "Cumulative best max-only model result each model family has achieved as newer models are released.",
            svg_family_best_over_time(result_runs, family_colors, family_legend_colors),
        ),
        chart_shell(
            "Origin Best Score Over Time",
            "Cumulative best max-only model result each origin has achieved as newer models are released.",
            svg_origin_best_over_time(result_runs, origin_colors),
        ),
        chart_shell(
            "Open Weights Versus Closed Weights",
            "Cumulative best max-only model result achieved by public-weight and closed-weight models as newer models are released.",
            svg_open_weights_best_over_time(result_runs),
        ),
        '<div class="chart-grid">',
        chart_shell(
            "Origin Comparison",
            "Each normalized model version plotted once by model release date, using its maximum score and styled by origin.",
            svg_release_date_scatter(
                result_runs,
                origin_colors,
                "origin",
                width=650,
                height=360,
                left=66,
                right=120,
                top=34,
                bottom=70,
                aria_label="Best score by model release date and origin",
            ),
        ),
        chart_shell(
            "Best Score By Origin",
            "Best max-only model result observed for each origin bucket.",
            svg_best_bars(origin_rows, "Origin best score comparison", origin_colors),
        ),
        "</div>",
        '<div class="chart-grid">',
        chart_shell(
            "Model Family Comparison",
            "Best max-only model result observed for each model family.",
            svg_best_horizontal_bars(family_rows, "Model family best score comparison"),
        ),
        chart_shell(
            "Models By Family",
            "Counts normalized model results by family after repeated runs are collapsed to each model's maximum score.",
            svg_count_bars(family_counts, "Models by family"),
        ),
        "</div>",
        '<div class="chart-grid">',
        chart_shell(
            "Harness Comparison",
            "Best max-only model result represented by each evaluation harness, with model counts shown below each bar.",
            svg_best_bars(harness_rows, "Harness best score comparison"),
        ),
        chart_shell(
            "Results By Effort",
            "Counts max-only model results by the effort setting on the run that produced that model's maximum score.",
            svg_count_bars(effort_counts, "Results by effort"),
        ),
        "</div>",
        chart_shell(
            "Stop Reason Counts",
            "Raw run counts grouped by final stop reason.",
            svg_count_bars(
                stop_reason_counts,
                "Stop reason counts",
                width=1180,
                left=260,
                right=80,
            ),
        ),
        chart_shell(
            "Score Versus Wall Time",
            "Each point uses the run that produced a normalized model version's maximum score.",
            svg_score_vs_wall_time(result_runs, scatter_colors, scatter_legend_colors),
        ),
        chart_shell(
            "Score Versus Code Lines Added",
            "Each point uses the run that produced a normalized model version's maximum score.",
            svg_score_vs_code_lines(result_runs, scatter_colors, scatter_legend_colors),
        ),
        data_table(runs, family_icons),
        model_run_count_table(runs, family_icons),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mortal Coil Benchmark Visualizations</title>
  <link rel="icon" href="favicon.svg" type="image/svg+xml">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #dbe3ef;
      --track: #e8eef6;
      --accent: #2563eb;
      --warn-bg: #fff7ed;
      --warn-text: #9a3412;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    main {{
      width: min(1240px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }}

    header {{
      margin-bottom: 24px;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.05;
      font-weight: 760;
    }}

    h2 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }}

    p {{
      margin: 0;
      color: var(--muted);
    }}

    .subhead {{
      max-width: 760px;
      font-size: 16px;
    }}

    .rule-note {{
      max-width: 860px;
      margin-top: 10px;
      color: #334155;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}

    .metric,
    .chart-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
    }}

    .metric {{
      padding: 14px 16px;
    }}

    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}

    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 23px;
      line-height: 1.1;
    }}

    .metric small {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}

    .chart-panel {{
      margin: 16px 0;
      padding: 18px;
      overflow: hidden;
    }}

    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: stretch;
    }}

    .chart-grid .chart-panel {{
      margin: 0;
    }}

    .chart-heading {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
      margin-bottom: 12px;
    }}

    .chart-heading p {{
      max-width: 520px;
      text-align: right;
    }}

    .chart-svg {{
      display: block;
      width: 100%;
      height: auto;
    }}

    .chart-svg.compact {{
      max-height: 360px;
    }}

    .model-name-with-icon {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      white-space: nowrap;
    }}

    .model-family-icon {{
      display: inline-block;
      width: 20px;
      height: 20px;
      flex: 0 0 20px;
      border: 1px solid #cbd5e1;
      border-radius: 50%;
      background-color: #ffffff;
      background-position: center;
      background-repeat: no-repeat;
      background-size: 14px 14px;
    }}

    {family_icon_css}

    .grid-line {{
      stroke: var(--line);
      stroke-width: 1;
    }}

    .axis-line {{
      stroke: #94a3b8;
      stroke-width: 1.2;
    }}

    .data-point {{
      stroke: #ffffff;
      stroke-width: 2;
    }}

    .model-point {{
      stroke-width: 2.2;
    }}

    .same-model-line {{
      stroke-width: 1.3;
      stroke-linecap: round;
      opacity: 0.42;
    }}

    .progress-line {{
      fill: none;
      stroke-width: 2.4;
      stroke-linejoin: round;
      stroke-linecap: round;
    }}

    .origin-progress-line {{
      stroke-width: 3;
    }}

    .progress-point {{
      stroke: #ffffff;
      stroke-width: 1.8;
    }}

    .reference-line {{
      stroke-width: 2;
      stroke-dasharray: 7 6;
      stroke-linecap: round;
    }}

    .bar-track {{
      fill: var(--track);
    }}

    .bar {{
      shape-rendering: geometricPrecision;
    }}

    text {{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      fill: var(--ink);
    }}

    .tick-label,
    .muted-label {{
      fill: var(--muted);
      font-size: 12px;
    }}

    .axis-title {{
      fill: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }}

    .legend text,
    .bar-label,
    .value-label {{
      font-size: 13px;
    }}

    .model-legend text {{
      font-size: 11px;
    }}

    .bar-label {{
      fill: #334155;
    }}

    .value-label {{
      fill: #0f172a;
      font-weight: 700;
    }}

    .point-label {{
      font-size: 12px;
      fill: #334155;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 3px;
      stroke-linejoin: round;
    }}

    .setting-label,
    .icon-initial {{
      fill: var(--muted);
      font-size: 11px;
    }}

    .icon-initial {{
      font-weight: 700;
    }}

    .reference-label {{
      fill: var(--muted);
      font-size: 12px;
    }}

    .leaderboard-more summary {{
      margin: 6px 0;
      color: var(--accent);
      font-weight: 650;
      cursor: pointer;
    }}

    .warning {{
      margin: 14px 0 0;
      padding: 10px 12px;
      border-radius: 8px;
      background: var(--warn-bg);
      color: var(--warn-text);
      border: 1px solid #fed7aa;
    }}

    .table-wrap {{
      overflow-x: auto;
    }}

    table {{
      width: 100%;
      min-width: 980px;
      border-collapse: collapse;
      font-size: 13px;
    }}

    th,
    td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      white-space: nowrap;
    }}

    th {{
      color: #334155;
      background: #f1f5f9;
      font-weight: 720;
    }}

    tr:last-child td {{
      border-bottom: 0;
    }}

    footer {{
      color: var(--muted);
      margin-top: 20px;
      font-size: 13px;
    }}

    @media (max-width: 900px) {{
      main {{
        width: min(100vw - 20px, 1240px);
        padding-top: 20px;
      }}

      .metrics,
      .chart-grid {{
        grid-template-columns: 1fr;
      }}

      .chart-heading {{
        display: block;
      }}

      .chart-heading p {{
        margin-top: 6px;
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Mortal Coil Benchmark Visualizations</h1>
      <p class="subhead">Static SVG report for Puzzle Runner runs testing Mortal Coil, generated from <code>final_results.md</code> and <code>model_metadata.md</code>. Model releases span {fmt_date(min_date)} through {fmt_date(max_date)}.</p>
      <p class="rule-note"><strong>Result rule:</strong> {html_escape(RESULT_RULE)}</p>
      {unmatched_html}
    </header>
    {summary_cards(runs)}
    {''.join(charts)}
    <footer>
      Generated by <code>scripts/build_visualizations.py</code>. Rerun the script after updating benchmark results or model metadata.
    </footer>
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    runs, unmatched = load_results(args.results, metadata)
    level_dimensions = load_level_dimensions(args.level_dimensions)
    family_icons = load_family_icons(args.family_icon_dir) if args.family_icons else {}
    html_output = render_html(runs, unmatched, level_dimensions, family_icons)
    html_output = "\n".join(line.rstrip() for line in html_output.splitlines()) + "\n"
    args.output.write_text(html_output, encoding="utf-8")

    print(f"Wrote {args.output}")
    if unmatched:
        print(
            "Warning: metadata missing for " + ", ".join(sorted(set(unmatched))),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
