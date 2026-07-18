#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


PALETTE = [
    "#2563eb",
    "#64748b",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#9f6b7d",
    "#4f46e5",
    "#65a30d",
    "#ca8a04",
]

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

SINGLE_SERIES_COLOR = "#64748b"

ORIGIN_COLORS = {
    "America": "#2563eb",
    "China": "#7c3aed",
    "Europe": "#059669",
    "Japan": "#b45309",
    "Unknown": "#64748b",
}

OPEN_WEIGHTS_COLORS = {
    "Open weights": "#059669",
    "Closed weights": "#64748b",
}

HUMAN_BEST_SCORE = 1206
DEFAULT_BRUTE_FORCE_SCORE = 47
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
    return parser.parse_args()


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


def color_map(labels: list[str]) -> dict[str, str]:
    return {label: PALETTE[index % len(PALETTE)] for index, label in enumerate(sorted(set(labels)))}


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


def best_by_version(runs: list[RunResult]) -> list[dict[str, object]]:
    buckets: dict[str, list[RunResult]] = defaultdict(list)
    for run in runs:
        buckets[run.version].append(run)
    rows: list[dict[str, object]] = []
    for version, version_runs in buckets.items():
        best_run = max(version_runs, key=lambda run: run.best_score)
        rows.append(
            {
                "label": version,
                "best": best_run.best_score,
                "count": len(version_runs),
                "count_label": "run",
                "family": best_run.family,
                "origin": best_run.origin,
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


def nice_max(value: float) -> float:
    if value <= 0:
        return 1
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


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


def svg_model_legend(styles: dict[str, dict[str, str]], x: int, y: int, columns: int = 2) -> str:
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
        elements.append(
            f'<circle cx="{marker_x + 6}" cy="{marker_y + 6}" r="6" fill="url(#{style["pattern_id"]})" '
            f'stroke="{style["color"]}" stroke-width="1.5" />'
        )
        elements.append(f'<text x="{marker_x + 18}" y="{marker_y + 10}">{html_escape(label)}</text>')
    elements.append("</g>")
    return "\n".join(elements)


def svg_model_release_date_scatter(runs: list[RunResult]) -> str:
    plotted = [run for run in runs if run.release_date]
    plotted.sort(key=lambda run: (run.release_date or run.run_date, run.version, run.run_id))
    styles = model_style_map(plotted)
    width = 1180
    height = 560
    left = 70
    right = 420
    top = 30
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_score = score_axis_max([run.best_score for run in plotted])
    min_date = min(run.release_date for run in plotted if run.release_date)
    max_date = max(run.release_date for run in plotted if run.release_date)
    total_days = max(1, (max_date - min_date).days)

    elements = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Best score by model release date and model version">',
        svg_model_pattern_defs(styles),
        grid_lines(left, top, plot_w, plot_h, max_score),
        axis_labels(left, top, plot_w, plot_h, "Model release date", "Best score"),
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
        elements.append(
            f'<line class="same-model-line" x1="{x:.1f}" x2="{x:.1f}" '
            f'y1="{min(y_values):.1f}" y2="{max(y_values):.1f}" stroke="{style["color"]}">'
            f"<title>{html_escape(version)} repeated runs</title></line>"
        )

    for run, x, y in points:
        style = styles[run.version]
        tooltip = f"Released {fmt_date(run.release_date or run.run_date)} - {run.version}: {run.best_score} ({run.run_id})"
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.5" fill="url(#{style["pattern_id"]})" '
            f'class="data-point model-point" style="stroke: {style["color"]}; stroke-width: 2.2;">'
            f"<title>{html_escape(tooltip)}</title></circle>"
        )

    for tick_index in range(5):
        tick_date = min_date + dt.timedelta(days=round(total_days * tick_index / 4))
        x = scale((tick_date - min_date).days, 0, total_days, left, left + plot_w)
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 34}" text-anchor="middle">{fmt_month_year(tick_date)}</text>')

    elements.append(svg_model_legend(styles, left + plot_w + 30, top + 4, columns=2))
    elements.append("</svg>")
    return "\n".join(elements)


def svg_release_date_scatter(
    runs: list[RunResult],
    colors: dict[str, str],
    color_key: str,
    *,
    width: int = 1180,
    height: int = 430,
    left: int = 70,
    right: int = 190,
    top: int = 30,
    bottom: int = 72,
    aria_label: str = "Best score by model release date",
) -> str:
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
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{color}" class="data-point">'
            f"<title>{html_escape(tooltip)}</title></circle>"
        )

    for tick_index in range(5):
        tick_date = min_date + dt.timedelta(days=round(total_days * tick_index / 4))
        x = scale((tick_date - min_date).days, 0, total_days, left, left + plot_w)
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 34}" text-anchor="middle">{fmt_month_year(tick_date)}</text>')

    elements.append(svg_legend(colors, width - right + 28, top + 4))
    elements.append("</svg>")
    return "\n".join(elements)


def svg_score_over_time(runs: list[RunResult]) -> str:
    return svg_model_release_date_scatter(runs)


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
        human_label: "#7c3aed",
        brute_force_label: "#ea580c",
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

    for release_date, level, _, run, x, y in scaled:
        tooltip = (
            f"AI best so far {metric_tooltip(level)} as of {fmt_date(release_date)} "
            f"from {run.version} ({run.run_id}, run {fmt_date(run.run_date)})"
        )
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" fill="{colors[ai_label]}" class="progress-point">'
            f"<title>{html_escape(tooltip)}</title></circle>"
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
) -> str:
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

    for group in sorted(group_points):
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
            elements.append(
                f'<path class="progress-line" d="{" ".join(path_parts)}" stroke="{color}">'
                f"<title>{html_escape(group)} best score over time</title></path>"
            )

        for release_date, score, x, y in scaled:
            tooltip = f"{group}: best score so far {score} as of {fmt_date(release_date)}"
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" class="progress-point">'
                f"<title>{html_escape(tooltip)}</title></circle>"
            )

    for tick_index in range(5):
        tick_date = min_date + dt.timedelta(days=round(total_days * tick_index / 4))
        x = scale((tick_date - min_date).days, 0, total_days, left, left + plot_w)
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 34}" text-anchor="middle">{fmt_month_year(tick_date)}</text>')

    elements.append(svg_legend(colors, width - right + 28, top + 4))
    elements.append("</svg>")
    return "\n".join(elements)


def svg_family_best_over_time(runs: list[RunResult], family_colors: dict[str, str]) -> str:
    return svg_group_best_over_time(
        runs,
        family_colors,
        "family",
        "Family best score so far",
        "Cumulative best score by model family over release date",
    )


def svg_origin_best_over_time(runs: list[RunResult], origin_colors: dict[str, str]) -> str:
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


def svg_best_bars(rows: list[dict[str, object]], title: str) -> str:
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
        x = center - bar_w / 2
        y = scale(value, 0, max_value, top + plot_h, top)
        height_px = top + plot_h - y
        elements.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height_px:.1f}" fill="{SINGLE_SERIES_COLOR}">'
            f"<title>{html_escape(row['label'])} best: {fmt_number(value)}</title></rect>"
        )
        elements.append(
            f'<text class="tick-label" x="{center:.1f}" y="{height - 42}" text-anchor="middle">{html_escape(row["label"])}</text>'
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


def svg_horizontal_bars(rows: list[dict[str, object]], colors: dict[str, str]) -> str:
    width = 1180
    row_h = 34
    top = 26
    bottom = 44
    left = 260
    right = 80
    height = top + bottom + row_h * len(rows)
    plot_w = width - left - right
    max_value = score_axis_max([int(row["best"]) for row in rows])
    elements = [
        f'<svg class="chart-svg tall" viewBox="0 0 {width} {height}" role="img" aria-label="Best score by model version">',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_h
        value = int(row["best"])
        width_px = scale(value, 0, max_value, 0, plot_w)
        color = colors.get(str(row["family"]), "#64748b")
        elements.append(f'<text class="bar-label" x="{left - 12}" y="{y + 22}" text-anchor="end">{html_escape(row["label"])}</text>')
        elements.append(f'<rect class="bar-track" x="{left}" y="{y + 7}" width="{plot_w}" height="18" rx="4" />')
        elements.append(
            f'<rect class="bar" x="{left}" y="{y + 7}" width="{width_px:.1f}" height="18" rx="4" fill="{color}">'
            f"<title>{html_escape(row['label'])}: best {value}, {html_escape(fmt_count(row['count'], row['count_label']))}</title></rect>"
        )
        elements.append(f'<text class="value-label" x="{left + width_px + 8:.1f}" y="{y + 22}">{value}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def svg_count_bars(
    counts: Counter[str],
    colors: dict[str, str],
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
    svg_class = "chart-svg compact" if width <= 650 else "chart-svg"
    elements = [f'<svg class="{svg_class}" viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(title)}">']
    for index, (label, count) in enumerate(rows):
        y = top + index * row_h
        width_px = scale(count, 0, max_value, 0, plot_w)
        color = colors.get(label, "#64748b")
        elements.append(f'<text class="bar-label" x="{left - 10}" y="{y + 23}" text-anchor="end">{html_escape(label)}</text>')
        elements.append(f'<rect class="bar-track" x="{left}" y="{y + 8}" width="{plot_w}" height="18" rx="4" />')
        elements.append(f'<rect class="bar" x="{left}" y="{y + 8}" width="{width_px:.1f}" height="18" rx="4" fill="{color}" />')
        elements.append(f'<text class="value-label" x="{left + width_px + 8:.1f}" y="{y + 23}">{count}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def svg_score_vs_wall_time(runs: list[RunResult], family_colors: dict[str, str]) -> str:
    plotted = [run for run in runs if run.wall_seconds]
    width = 1180
    height = 430
    left = 70
    right = 42
    top = 30
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_score = score_axis_max([run.best_score for run in plotted])
    max_hours = nice_max(max((run.wall_seconds or 0) / 3600 for run in plotted))
    elements = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Score versus wall time">',
        grid_lines(left, top, plot_w, plot_h, max_score),
        axis_labels(left, top, plot_w, plot_h, "Wall time (hours)", "Best score"),
    ]
    for tick_index in range(6):
        hours = max_hours * tick_index / 5
        x = scale(hours, 0, max_hours, left, left + plot_w)
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 38}" text-anchor="middle">{fmt_number(hours)}</text>')
    for run in plotted:
        hours = (run.wall_seconds or 0) / 3600
        x = scale(hours, 0, max_hours, left, left + plot_w)
        y = scale(run.best_score, 0, max_score, top + plot_h, top)
        color = family_colors.get(run.family, "#64748b")
        tooltip = f"{run.version}: {run.best_score}, {fmt_duration(run.wall_seconds)}"
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{color}" class="data-point">'
            f"<title>{html_escape(tooltip)}</title></circle>"
        )
    elements.append("</svg>")
    return "\n".join(elements)


def svg_score_vs_code_lines(runs: list[RunResult], family_colors: dict[str, str]) -> str:
    plotted = [run for run in runs if run.code_lines_added is not None]
    width = 1180
    height = 430
    left = 70
    right = 42
    top = 30
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_score = score_axis_max([run.best_score for run in plotted])
    max_lines = nice_max(max(run.code_lines_added or 0 for run in plotted))
    elements = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Score versus code lines added">',
        grid_lines(left, top, plot_w, plot_h, max_score),
        axis_labels(left, top, plot_w, plot_h, "Code lines added", "Best score"),
    ]
    for tick_index in range(6):
        lines = max_lines * tick_index / 5
        x = scale(lines, 0, max_lines, left, left + plot_w)
        elements.append(f'<text class="tick-label" x="{x:.1f}" y="{height - 38}" text-anchor="middle">{fmt_number(lines)}</text>')
    for run in plotted:
        lines = run.code_lines_added or 0
        x = scale(lines, 0, max_lines, left, left + plot_w)
        y = scale(run.best_score, 0, max_score, top + plot_h, top)
        color = family_colors.get(run.family, "#64748b")
        tooltip = f"{run.version}: {run.best_score}, {fmt_number(lines)} code lines added"
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{color}" class="data-point">'
            f"<title>{html_escape(tooltip)}</title></circle>"
        )
    elements.append("</svg>")
    return "\n".join(elements)


def summary_cards(runs: list[RunResult]) -> str:
    best_run = max(runs, key=lambda run: run.best_score)
    newest_release = max((run for run in runs if run.release_date), key=lambda run: run.release_date or dt.date.min)
    total_cost = sum(run.openrouter_cost or 0 for run in runs)
    metrics = [
        ("Runs", len(runs)),
        ("Models", len({run.version for run in runs})),
        ("Top score", best_run.best_score),
        ("Newest release", fmt_date(newest_release.release_date or newest_release.run_date)),
        ("Origins", len({run.origin for run in runs})),
        ("OpenRouter cost", f"${total_cost:,.2f}" if total_cost else "$0.00"),
    ]
    cards = []
    for label, value in metrics:
        cards.append(
            f"""
            <div class="metric">
              <span>{html_escape(label)}</span>
              <strong>{html_escape(value)}</strong>
            </div>
            """
        )
    return '<div class="metrics">' + "\n".join(cards) + "</div>"


def data_table(runs: list[RunResult]) -> str:
    rows = []
    for run in sorted(runs, key=lambda item: (item.release_date or item.run_date, item.run_date), reverse=True):
        rows.append(
            "<tr>"
            f"<td>{html_escape(fmt_date(run.release_date or run.run_date))}</td>"
            f"<td>{html_escape(run.version)}</td>"
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


def model_run_count_table(runs: list[RunResult]) -> str:
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
            f"<td>{html_escape(row['version'])}</td>"
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
) -> str:
    if not runs:
        raise ValueError("no runs to visualize")

    result_runs = best_runs_by_version(runs)
    family_colors = color_map([run.family for run in result_runs])
    origins = {run.origin for run in result_runs}
    origin_colors = {
        origin: ORIGIN_COLORS.get(origin, "#64748b")
        for origin in ORIGIN_COLORS
        if origin in origins
    }
    for origin in sorted(origins - set(origin_colors)):
        origin_colors[origin] = ORIGIN_COLORS.get(origin, "#64748b")
    origin_rows = aggregate_scores(result_runs, "origin")
    family_rows = aggregate_scores(result_runs, "family")
    harness_rows = aggregate_scores(result_runs, "harness")
    version_rows = best_by_version(runs)
    family_counts = Counter(run.family for run in result_runs)
    effort_counts = Counter(run.effort for run in result_runs)
    effort_colors = color_map(list(effort_counts))
    stop_reason_counts = Counter(run.stop_reason.strip() or "unspecified" for run in runs)
    stop_reason_colors = color_map(list(stop_reason_counts))
    min_date = min(run.release_date for run in result_runs if run.release_date)
    max_date = max(run.release_date for run in result_runs if run.release_date)

    unmatched_html = ""
    if unmatched:
        unmatched_list = ", ".join(sorted(set(unmatched)))
        unmatched_html = f'<p class="warning">Metadata missing for: {html_escape(unmatched_list)}</p>'

    charts = [
        chart_shell(
            "All LLMs By Release Date",
            "Each normalized model version plotted once by public release date, using its maximum score across all runs.",
            svg_score_over_time(result_runs),
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
            svg_family_best_over_time(result_runs, family_colors),
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
            "Each normalized model version plotted once by model release date, using its maximum score and colored by origin.",
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
            svg_best_bars(origin_rows, "Origin best score comparison"),
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
            svg_count_bars(family_counts, family_colors, "Models by family"),
        ),
        "</div>",
        chart_shell(
            "Best Score By Model Version",
            "Highest score observed for each normalized model version; repeated raw runs are never averaged.",
            svg_horizontal_bars(version_rows, family_colors),
        ),
        '<div class="chart-grid">',
        chart_shell(
            "Harness Comparison",
            "Best max-only model result represented by each evaluation harness, with model counts shown below each bar.",
            svg_best_bars(harness_rows, "Harness best score comparison"),
        ),
        chart_shell(
            "Results By Effort",
            "Counts max-only model results by the effort setting on the run that produced that model's maximum score.",
            svg_count_bars(effort_counts, effort_colors, "Results by effort"),
        ),
        "</div>",
        chart_shell(
            "Stop Reason Counts",
            "Raw run counts grouped by final stop reason.",
            svg_count_bars(
                stop_reason_counts,
                stop_reason_colors,
                "Stop reason counts",
                width=1180,
                left=260,
                right=80,
            ),
        ),
        chart_shell(
            "Score Versus Wall Time",
            "Each point uses the run that produced a normalized model version's maximum score.",
            svg_score_vs_wall_time(result_runs, family_colors),
        ),
        chart_shell(
            "Score Versus Code Lines Added",
            "Each point uses the run that produced a normalized model version's maximum score.",
            svg_score_vs_code_lines(result_runs, family_colors),
        ),
        data_table(runs),
        model_run_count_table(runs),
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
    html_output = render_html(runs, unmatched, level_dimensions)
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
