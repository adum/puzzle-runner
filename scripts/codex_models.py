#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys


def run_codex_models() -> dict:
    try:
        result = subprocess.run(
            ["codex", "debug", "models"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print("Error: codex CLI not found in PATH", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("Error running: codex debug models", file=sys.stderr)
        print(e.stderr.strip(), file=sys.stderr)
        sys.exit(e.returncode)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Error: codex returned invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)


def get_models(blob):
    models = blob.get("models", blob)

    if isinstance(models, dict):
        # Handle either {"slug": {...}} or other dict-ish shapes.
        out = []
        for key, value in models.items():
            if isinstance(value, dict):
                value = {"slug": key, **value}
                out.append(value)
        return out

    if isinstance(models, list):
        return [m for m in models if isinstance(m, dict)]

    return []


def model_id(model):
    return (
        model.get("slug")
        or model.get("model")
        or model.get("id")
        or model.get("name")
        or ""
    )


def model_display_name(model):
    return (
        model.get("display_name")
        or model.get("displayName")
        or model.get("name")
        or ""
    )


def get_efforts(model):
    efforts = (
        model.get("supported_reasoning_levels")
        or model.get("supported_reasoning_efforts")
        or model.get("reasoning_efforts")
        or []
    )

    result = []
    for effort in efforts:
        if isinstance(effort, dict):
            name = (
                effort.get("effort")
                or effort.get("id")
                or effort.get("name")
                or ""
            )
            desc = effort.get("description") or ""
            result.append((str(name), str(desc)))
        else:
            result.append((str(effort), ""))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Print Codex models matching a substring, with valid reasoning efforts."
    )
    parser.add_argument(
        "substring",
        help='Substring to search for in model id/display name, e.g. "5.6"',
    )
    args = parser.parse_args()

    needle = args.substring.lower()
    blob = run_codex_models()
    models = get_models(blob)

    matches = []
    for model in models:
        mid = model_id(model)
        display = model_display_name(model)
        haystack = f"{mid} {display}".lower()

        if needle in haystack:
            matches.append(model)

    if not matches:
        print(f"No models matched: {args.substring}")
        sys.exit(1)

    for model in matches:
        mid = model_id(model)
        display = model_display_name(model)
        default_effort = (
            model.get("default_reasoning_level")
            or model.get("default_reasoning_effort")
            or "n/a"
        )
        efforts = get_efforts(model)

        print(mid)
        if display and display != mid:
            print(f"  name: {display}")
        print(f"  default effort: {default_effort}")

        if efforts:
            print("  valid efforts:")
            for name, desc in efforts:
                if desc:
                    print(f"    - {name}: {desc}")
                else:
                    print(f"    - {name}")
        else:
            print("  valid efforts: n/a")

        print()


if __name__ == "__main__":
    main()
