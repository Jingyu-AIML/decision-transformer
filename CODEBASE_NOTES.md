# Decision Transformer — Codebase Notes
> Reference: Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)

---

## Project Structure

```
decision-transformer/
├── src/
│   ├── model.py        # DecisionTransformer nn.Module
│   ├── trainer.py      # Training loop (Trainer class)
│   └── utils.py        # Data loading, TrajectoryDataset, normalization
├── scripts/
│   ├── train.py        # Training entry point
│   └── evaluate.py     # Evaluation / rollout entry point
├── configs/
│   └── hopper_medium.yaml   # Hyperparams for Hopper-Medium-v2
└── tests/
    └── test_model.py
```

---

## Core Idea

Instead of RL with reward maximization, the Decision Transformer frames the problem as **sequence modeling**:

> Given a desired future return, past states, and past actions — predict the next action.

At inference, you set a **target return** (e.g. 3600 for Hopper). The model generates actions it believes will achieve that target, no value functions or policy gradients needed.

---

## Data Pipeline

### Raw d4rl / Minari format
Flat arrays of N total transitions across all episodes:
```
dataset["observations"]  shape (N, state_dim)
dataset["actions"]       shape (N, act_dim)
dataset["rewards"]       shape (N,)
dataset["terminals"]     shape (N,)   # true terminal
dataset["timeouts"]      shape (N,)   # episode cut by time limit (not a true terminal)
```

### After parse_d4rl_dataset / parse_minari_dataset
A list of trajectory dicts, one per episode:
```python
trajectory = {
    "observations": np.ndarray (T, state_dim),
    "actions":      np.ndarray (T, act_dim),
    "rewards":      np.ndarray (T,),
    "terminals":    np.ndarray (T,)  bool
}
```
Note: Minari stores T+1 observations per episode (includes next-obs), so the last one is dropped.

### State normalization
States are normalized to zero mean / unit variance using stats computed over the entire dataset:
```python
state_mean, state_std = compute_state_stats(trajectories)
trajectories = normalize_states(trajectories, state_mean, state_std)
# saved to checkpoints/state_mean.npy, checkpoints/state_std.npy
# must be reloaded and applied identically at evaluation time
```

---

## Reward → Return-to-Go (RTG)

RTG at timestep t = sum of all **future** rewards from t to end of episode.

```
RTG[t] = r[t] + r[t+1] + ... + r[T-1]
       = r[t] + RTG[t+1]          (recursive definition)
```

### Concrete example
```
rewards = [1.0,  1.2,  0.8,  1.5,  0.5]

t=4: RTG = 0.5
t=3: RTG = 1.5 + 0.5  = 2.0
t=2: RTG = 0.8 + 2.0  = 2.8
t=1: RTG = 1.2 + 2.8  = 4.0
t=0: RTG = 1.0 + 4.0  = 5.0

rtg = [5.0, 4.0, 2.8, 2.0, 0.5]  shape (T, 1)
```

Implementation (`utils.py:108`):
```python
rtg = np.zeros_like(rewards)
rtg[-1] = rewards[-1]
for t in reversed(range(len(rewards) - 1)):
    rtg[t] = rewards[t] + rtg[t + 1]
```

---

## TrajectoryDataset (utils.py)

Wraps trajectories into fixed-length context windows of size K (context_len).

- Samples trajectories **proportional to length** (longer trajectories sampled more).
- Picks a **random starting position** within the trajectory.
- **Left-pads** short windows with zeros to always output shape `(K, dim)`.
- Returns an **attention mask**: 0 for padded positions, 1 for real.

```python
# One sample returned by __getitem__:
states         # (K, state_dim)  float32
actions        # (K, act_dim)    float32
returns_to_go  # (K, 1)         float32
timesteps      # (K,)            int64
mask           # (K,)            float32  (0=pad, 1=real)
```

---

## Model Architecture (model.py)

Input sequence interleaved as: `R_1, s_1, a_1, R_2, s_2, a_2, ...`  → length 3K

```
Each modality gets its own linear embedding + shared timestep embedding:
  r_emb = embed_return(RTG) + embed_timestep(t)    # (B, K, H)
  s_emb = embed_state(state) + embed_timestep(t)   # (B, K, H)
  a_emb = embed_action(action) + embed_timestep(t) # (B, K, H)

Stacked → (B, 3K, H) → LayerNorm → CausalTransformer → (B, 3K, H)

Action is predicted from the STATE token output (index 1 mod 3):
  action_pred = Linear(hidden) → Tanh → (B, K, act_dim)
```

Key config (Hopper):
| param | value |
|---|---|
| state_dim | 11 |
| act_dim | 3 |
| hidden_size | 128 |
| context_len K | 20 |
| n_layer | 3 |
| n_head | 1 |

---

## Training (trainer.py + scripts/train.py)

- **Loss**: MSE between predicted and ground-truth actions (only on non-padded positions)
- **Optimizer**: AdamW, lr=1e-4, weight_decay=1e-4
- **Grad clipping**: max_norm=0.25
- Checkpoint saved to `checkpoints/best_model.pt`

```bash
# Full training
python scripts/train.py --config configs/hopper_medium.yaml

# Quick smoke test (no MuJoCo/Minari needed — uses synthetic data)
python scripts/train.py --config configs/hopper_medium.yaml --smoke

# Train for fixed number of gradient steps
python scripts/train.py --config configs/hopper_medium.yaml --max-steps 10000
```

---

## Evaluation (scripts/evaluate.py)

Uses a **sliding context window** of size K. At each timestep:
1. Roll the context buffer left by 1.
2. Insert current `obs`, current remaining RTG (`target_return - episode_return`), and `t`.
3. Forward pass → take action from last position → step env.

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/hopper_medium.yaml \
    --target_return 3600 \
    --n_eval 10
```

**Known TODO in evaluate.py**: `state_mean` / `state_std` are currently hardcoded to 0/1.
They should be loaded from `checkpoints/state_mean.npy` and `checkpoints/state_std.npy`.

---

## Training vs Inference: How RTG is Used

### Training
The model never sees a hand-picked target like 3600. It sees **real RTG values computed from the dataset**.

For a trajectory window (K=3 example):
```
t=0: RTG=5.0,  s=[0.1, 0.2, ...],  a=[0.3, -0.1, 0.5]  ← ground truth
t=1: RTG=4.0,  s=[0.11, 0.21, ...], a=[0.2, -0.2, 0.4]
t=2: RTG=2.8,  s=[0.12, 0.22, ...], a=[0.1, -0.3, 0.3]
```

The model predicts actions autoregressively using causal attention:
```
RTG=5.0, s0              →  predict a0_hat
RTG=5.0, s0, a0, RTG=4.0, s1              →  predict a1_hat
RTG=5.0, s0, a0, RTG=4.0, s1, a1, RTG=2.8, s2  →  predict a2_hat
```

Loss is MSE against ground-truth actions (padded positions excluded):
```
loss = MSE(a0_hat, a0) + MSE(a1_hat, a1) + MSE(a2_hat, a2)
```

The model learns: **"when RTG is high, imitate the actions that historically achieved those returns."**
No reward maximization — pure supervised imitation conditioned on RTG.

### Inference
You provide a target return (e.g. 3600) and the current state. The remaining RTG decays naturally as rewards are collected:

```
t=0: feed RTG=3600,          take a0, get r0=10
t=1: feed RTG=3590,          take a1, get r1=12
t=2: feed RTG=3578,          take a2, get r2=11
...
remaining_RTG = target - sum(rewards so far)
```

The context window grows up to K=20, then slides — oldest timestep dropped each step.

**Only two things you need to provide at inference:**
1. Target return (your ambition level)
2. Current state from the environment

Actions and remaining RTG are filled in as the episode unfolds.

### RTG as a behavior dial
Because the model learned to associate RTG level with action quality, you can control behavior by changing the target:

- `target=3600` → model imitates the best trajectories in the dataset → high performance
- `target=500`  → model imitates mediocre trajectories → deliberately worse performance

The model is not maximizing — it's matching. So a low target produces low performance on purpose.

---

## Key Design Decisions

| Decision | Why |
|---|---|
| RTG as input (not reward) | Conditions the model on desired future performance — enables behavior steering at test time |
| Causal mask | Enforces autoregressive prediction; each token only sees the past |
| Predict action from state token | After `R_t, s_t`, the model has seen enough context to decide `a_t` |
| Left-pad (not right-pad) | The most recent timestep is always at position -1, which is what the model attends to most |
| Sample ∝ trajectory length | Prevents short trajectories from being over-represented |
