#!/usr/bin/env python3
"""Run a SmolVLA policy through LeRobot's maintained LIBERO evaluator."""

# Upstream evaluation entry point:
# https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py
# Upstream LIBERO environment adapter:
# https://github.com/huggingface/lerobot/blob/main/src/lerobot/envs/libero.py

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/libero"))
    return parser.parse_args()


def build_lerobot_args(args: argparse.Namespace) -> list[str]:
    return [
        "lerobot-eval",
        "--policy.type=smolvla",
        f"--policy.pretrained_path={args.policy_path}",
        "--env.type=libero",
        f"--env.task={args.task_suite}",
        f"--env.task_ids=[{args.task_id}]",
        f"--eval.n_episodes={args.episodes}",
        f"--eval.batch_size={args.batch_size}",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir}",
    ]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        from lerobot.scripts.lerobot_eval import eval_main
    except ImportError as exc:
        raise SystemExit(
            "Install LeRobot with LIBERO support before evaluation: "
            "pip install 'lerobot[libero]'"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = build_lerobot_args(args)
    eval_main()


if __name__ == "__main__":
    main()
