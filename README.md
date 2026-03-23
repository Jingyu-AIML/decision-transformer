# Decision Transformer

A clean implementation of [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) (Chen et al., 2021).

## Structure

```
src/
  model.py      # DecisionTransformer architecture
  trainer.py    # training loop
  utils.py      # dataset, data loading helpers
scripts/
  train.py      # training entry point
  evaluate.py   # rollout evaluation
configs/
  hopper_medium.yaml
tests/
  test_model.py
```

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/
```

## Training

```bash
# Single run
python scripts/train.py --config configs/hopper_medium.yaml --seed 0

# Override steps (default: 100K from config)
python scripts/train.py --config configs/hopper_medium.yaml --seed 0 --max-steps 100000

# Resume from checkpoint
python scripts/train.py --config configs/hopper_medium.yaml --seed 0 --resume checkpoints/hopper_medium_seed0/best_model.pt
```

Checkpoints are saved to `checkpoints/{config_stem}_seed{N}/` and include `best_model.pt`, `state_mean.npy`, and `state_std.npy`.

### Example training run (Hopper-Medium-v2, 10 epochs)

| Epoch | Loss   |
|-------|--------|
| 001   | 0.0898 |
| 002   | 0.0652 |
| 003   | 0.0585 |
| 004   | 0.0551 |
| 005   | 0.0531 |
| 006   | 0.0516 |
| 007   | 0.0506 |
| 008   | 0.0498 |
| 009   | 0.0491 |
| 010   | 0.0485 |

**Best loss: 0.0485**

Config: `batch_size=64`, `lr=1e-4`, `hidden_size=128`, `n_layer=3`, `context_len=20`

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/hopper_medium_seed0/best_model.pt \
    --config configs/hopper_medium.yaml \
    --target_return 3600 \
    --n_eval 10
```

`state_mean.npy` and `state_std.npy` are loaded automatically from the checkpoint directory.

### Example evaluation run (target_return=3600, n_eval=10)

| Episode | Return  |
|---------|---------|
| 1       | 1588.10 |
| 2       | 1715.47 |
| 3       | 1520.00 |
| 4       | 3634.45 |
| 5       | 1446.95 |
| 6       | 3592.62 |
| 7       | 1445.19 |
| 8       | 1886.41 |
| 9       | 1812.24 |
| 10      | 3616.57 |

**Mean return: 2225.80 ± 919.46**

Episodes 4, 6, and 10 hit ~3600 (the target), confirming the model learned to condition on high RTG. High variance is expected with 10 epochs — more training reduces it.

## Training on RunPod

### Pod config

- **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **GPU:** any RTX 3090 / A4000 or better
- **Volume disk:** 20 GB
- **Access:** SSH terminal

### Setup on pod

```bash
cd /workspace
git clone <your-repo-url>
cd decision-transformer
pip install "minari[hf]" gymnasium[mujoco] mujoco numpy pyyaml
python scripts/train.py --config configs/hopper_medium.yaml --seed 0
```

### Download checkpoints to local machine

```bash
scp -P <port> -r root@<ip>:/workspace/decision-transformer/checkpoints/ \
  /path/to/local/decision-transformer/checkpoints/
```

IP and port are shown in the RunPod pod's Connect tab.

## Status

- [x] Model architecture
- [x] Trainer
- [x] Dataset utilities
- [x] Minari dataset integration (Hopper-Medium-v2)
- [x] Full training run (Hopper-medium, best loss 0.0485)
- [x] Evaluation (mean return 2225.80 over 10 episodes)
- [x] RTG scaling (`rtg_scale: 1000`, paper §A.1)
- [x] Seed control (`--seed` arg, all RNGs seeded)
- [x] LR warmup scheduler (linear over `warmup_steps: 10000`)
- [x] Per-run checkpoint dirs (`checkpoints/{config}_seed{N}/`)
- [x] Step-based training (`max_steps: 100000` in config)
- [ ] Multi-env configs (HalfCheetah, Walker2d)
- [ ] Normalized score reporting in evaluation
