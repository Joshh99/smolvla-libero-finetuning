#!/usr/bin/env python3
"""
Training script for SmolVLA on LIBERO-Spatial.

Launches lerobot-train with T4-safe defaults, logs timing metadata,
and supports ablation variants via command-line flags.

Usage:
    # Baseline training (20K steps, chunk_size=50)
    python train_smolvla.py

    # Ablation: chunk_size=25
    python train_smolvla.py --chunk_size 25 --run_name ablation_chunk25

    # Ablation: chunk_size=10
    python train_smolvla.py --chunk_size 10 --run_name ablation_chunk10

    # Resume interrupted training
    python train_smolvla.py --resume

Environment:
    Requires MUJOCO_GL=egl and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    These are set automatically by this script if not already present.
"""

import argparse
import json
import os
import subprocess
import time

import torch


def get_gpu_info() -> dict:
    """Detect GPU and return device info."""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Switch to a GPU runtime.")
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    return {"name": name, "memory_gb": round(mem_gb, 1)}


def get_batch_size(gpu_mem_gb: float, user_override: int = None) -> int:
    """Select batch size based on available VRAM."""
    if user_override is not None:
        return user_override
    if gpu_mem_gb >= 38:    # A100 40GB
        return 16
    elif gpu_mem_gb >= 22:  # RTX 3090/4090
        return 8
    elif gpu_mem_gb >= 14:  # T4 16GB
        return 2
    else:                   # 8GB cards
        return 1


def get_eval_freq(gpu_mem_gb: float) -> int:
    """Disable in-training eval on <=16GB GPUs (README recommendation)."""
    if gpu_mem_gb >= 22:
        return 5000
    return 0


def build_train_command(args, gpu_info: dict) -> list:
    """Build the lerobot-train CLI command."""
    batch_size = get_batch_size(gpu_info["memory_gb"], args.batch_size)
    eval_freq = get_eval_freq(gpu_info["memory_gb"])

    cmd = [
        "lerobot-train",
        "--policy.type=smolvla",
        f"--policy.pretrained_path={args.pretrained_path}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={args.dataset}",
        f"--batch_size={batch_size}",
        f"--steps={args.steps}",
        f"--eval_freq={eval_freq}",
        f"--save_freq={args.save_freq}",
        "--policy.use_amp=true",
        "--wandb.enable=true",
        "--wandb.project=smolvla-libero",
        f"--wandb.run_name={args.run_name}",
        f"--output_dir={args.output_dir}",
        f"--seed={args.seed}",
    ]

    # Ablation: action chunk size
    if args.chunk_size != 50:
        cmd.append(f"--policy.n_action_steps={args.chunk_size}")

    return cmd, batch_size, eval_freq


def build_resume_command(args) -> list:
    """Build resume command from last checkpoint."""
    config_path = os.path.join(
        args.output_dir, "checkpoints", "last", "pretrained_model", "train_config.json"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Resume config not found: {config_path}")

    return [
        "lerobot-train",
        "--resume=true",
        f"--config_path={config_path}",
        "--eval_freq=0",
    ]


def save_metadata(args, gpu_info: dict, batch_size: int, eval_freq: int,
                  elapsed_min: float, output_path: str) -> None:
    """Save training metadata for the report."""
    meta = {
        "run_name": args.run_name,
        "model": args.pretrained_path,
        "dataset": args.dataset,
        "task": "libero_spatial",
        "steps": args.steps,
        "batch_size": batch_size,
        "seed": args.seed,
        "eval_freq": eval_freq,
        "save_freq": args.save_freq,
        "use_amp": True,
        "chunk_size": args.chunk_size,
        "num_denoising_steps": 10,
        "gpu": gpu_info["name"],
        "gpu_memory_gb": gpu_info["memory_gb"],
        "training_time_minutes": round(elapsed_min, 1),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA on LIBERO-Spatial")
    parser.add_argument("--pretrained_path", default="lerobot/smolvla_base",
                        help="Pretrained model path (default: lerobot/smolvla_base)")
    parser.add_argument("--dataset", default="HuggingFaceVLA/libero",
                        help="Dataset repo ID")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Training steps (default: 20000)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (auto-detected from GPU if omitted)")
    parser.add_argument("--save_freq", type=int, default=5000,
                        help="Checkpoint save frequency (default: 5000)")
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Action chunk size (default: 50, ablation: 10 or 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--run_name", default="baseline_chunk50_20k",
                        help="Wandb run name")
    parser.add_argument("--output_dir", default="outputs/train/smolvla_libero_spatial",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    # Set required environment variables
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # GPU info
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']} GB)")

    if args.resume:
        cmd = build_resume_command(args)
        print(f"Resuming training from checkpoint...")
    else:
        cmd, batch_size, eval_freq = build_train_command(args, gpu_info)
        print(f"Config: batch_size={batch_size}, eval_freq={eval_freq}, "
              f"chunk_size={args.chunk_size}, steps={args.steps}, seed={args.seed}")

    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    start = time.time()
    result = subprocess.run(cmd)
    elapsed_min = (time.time() - start) / 60

    print("=" * 60)
    print(f"Training {'completed' if result.returncode == 0 else 'FAILED'} "
          f"in {elapsed_min:.1f} minutes")

    if result.returncode == 0 and not args.resume:
        meta_path = os.path.join("logs", f"training_meta_{args.run_name}.json")
        save_metadata(args, gpu_info, batch_size, eval_freq, elapsed_min, meta_path)

    return result.returncode


if __name__ == "__main__":
    exit(main())
