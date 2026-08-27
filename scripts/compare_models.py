#!/usr/bin/env python3
"""Plot success rates stored in the repository's result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_rate(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("success_rate", "overall_success_rate", "aggregate_success_rate", "eval_success_rate"):
        if key in payload:
            return float(payload[key])
    for container in ("results", "summary"):
        value = payload.get(container, {})
        for key in ("success_rate", "overall_success_rate", "aggregate_success_rate", "eval_success_rate"):
            if key in value:
                return float(value[key])
    raise KeyError(f"No success-rate field found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("figures/model_comparison.png"))
    args = parser.parse_args()

    labels = args.labels or [path.stem for path in args.inputs]
    if len(labels) != len(args.inputs):
        parser.error("--labels must contain one label per input")

    rates = [100 * load_rate(path) for path in args.inputs]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, rates, color="#3366cc")
    axis.set_ylabel("Task success (%)")
    axis.set_ylim(0, 100)
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
