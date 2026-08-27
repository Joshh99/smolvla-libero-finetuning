#!/usr/bin/env python3
"""Launch the recorded SmolVLA training configuration through LeRobot."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def command(config: dict) -> list[str]:
    run = config["run"]
    model = config["model"]
    data = config["data"]
    optimization = config["optimization"]
    return [
        "lerobot-train",
        "--policy.type=smolvla",
        f"--policy.pretrained_path={model['pretrained_path']}",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={model['chunk_size']}",
        f"--policy.n_action_steps={model['n_action_steps']}",
        f"--policy.num_steps={model['num_steps']}",
        f"--policy.train_expert_only={str(model['train_expert_only']).lower()}",
        f"--policy.freeze_vision_encoder={str(model['freeze_vision_encoder']).lower()}",
        f"--policy.train_state_proj={str(model['train_state_proj']).lower()}",
        f"--policy.num_vlm_layers={model['num_vlm_layers']}",
        f"--policy.expert_width_multiplier={model['expert_width_multiplier']}",
        f"--policy.num_expert_layers={model['num_expert_layers']}",
        f"--policy.use_cache={str(model['use_cache']).lower()}",
        f"--policy.compile_model={str(model['compile_model']).lower()}",
        f"--policy.optimizer_lr={optimization['learning_rate']}",
        f"--policy.optimizer_betas={optimization['betas']}",
        f"--policy.optimizer_eps={optimization['epsilon']}",
        f"--policy.optimizer_weight_decay={optimization['weight_decay']}",
        f"--policy.optimizer_grad_clip_norm={optimization['grad_clip_norm']}",
        f"--policy.scheduler_warmup_steps={optimization['warmup_steps']}",
        f"--policy.scheduler_decay_steps={optimization['decay_steps']}",
        f"--policy.scheduler_decay_lr={optimization['final_learning_rate']}",
        f"--dataset.repo_id={data['repo_id']}",
        f"--batch_size={data['batch_size']}",
        f"--steps={optimization['steps']}",
        f"--eval_freq={optimization['eval_freq']}",
        f"--save_freq={optimization['save_freq']}",
        f"--policy.use_amp={str(optimization['use_amp']).lower()}",
        f"--seed={run['seed']}",
        f"--output_dir={run['output_dir']}",
        "--wandb.enable=true",
        "--wandb.project=smolvla-libero",
        f"--wandb.run_name={run['name']}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smolvla_libero_spatial.yaml"),
    )
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    invocation = command(load_config(args.config))
    if args.print_command:
        print(" ".join(invocation))
        return 0
    return subprocess.run(invocation, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
