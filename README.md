# SmolVLA fine-tuning on LIBERO-Spatial

This repository records a 20,000-step SmolVLA fine-tuning run on the LIBERO-Spatial demonstrations and the follow-up ablations used to diagnose its rollout behavior.

The reproduced policy reached **14% task success**, compared with **75% for the reference checkpoint** under the recorded evaluation setup. The central finding is negative but useful: executing all 50 predicted actions compounds early spatial errors, and updating all 16 vision-language layers through the action expert increases the adaptation burden. Shorter action chunks improved the observed result to 20%, but did not close the gap.

## Recorded training setup

- Base policy: `lerobot/smolvla_base`
- Dataset: `HuggingFaceVLA/libero`, `libero_spatial`
- Hardware: Tesla T4 (14.6 GB recorded)
- Training: 20,000 steps, batch size 2, FP16 mixed precision, seed 42
- Optimizer: learning rate `1e-4`, betas `(0.9, 0.95)`, weight decay `1e-10`
- Policy: chunk size 50, 50 executed actions, 10 denoising steps

The complete recovered values are in `configs/smolvla_libero_spatial.yaml`. The data-loader worker count and the exact LeRobot source revision used for the historical run were not recorded; LeRobot `0.5.1` is the recoverable package version.

## Evaluation

LeRobot currently supports LIBERO on Linux. Install the environment and invoke the repository wrapper, which delegates rollout execution to LeRobot's maintained evaluator:

```bash
pip install -r requirements.txt
python scripts/evaluate_libero.py \
  --policy-path path/to/checkpoint/pretrained_model \
  --task-suite libero_spatial \
  --task-id 0 \
  --episodes 1
```

`MUJOCO_GL=egl` is set automatically if the variable is absent.

## Results and figures

Machine-readable aggregate results are retained in `results/`. Generated checkpoints, videos, logs, and bulk figures are ignored. Four small representative figures are tracked as ordinary Git objects.

Regenerate the comparison figure with:

```bash
python scripts/compare_models.py \
  results/smolvla_finetuned_results.json \
  results/smolvla_reference_results.json \
  --labels fine-tuned reference \
  --output figures/model_comparison.png
```

The notebooks document architecture inspection, training, data exploration, and ablations. Their outputs are cleared so the repository remains lightweight and reproducible.
