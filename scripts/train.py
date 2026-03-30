"""
Training entry point for Decision Transformer.

Usage:
    python scripts/train.py --config configs/hopper_medium.yaml
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DecisionTransformer
from src.trainer import Trainer
from src.utils import TrajectoryDataset, compute_state_stats, normalize_states, parse_minari_dataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hopper_medium.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run 2 batches to verify pipeline")
    parser.add_argument("--max-steps", type=int, default=None, help="Train for N gradient steps (overrides n_epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Seed everything before any data loading or model init
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Per-run checkpoint directory: checkpoints/<config_stem>_seed<N>/
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    run_dir = os.path.join("checkpoints", f"{config_stem}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Using device: {device} | seed: {args.seed} | run_dir: {run_dir}")

    # --- Load dataset ---
    if args.smoke:
        # Synthetic data — no MuJoCo/d4rl needed
        rng = np.random.default_rng(0)
        trajectories = [
            {
                "observations": rng.standard_normal((200, cfg["state_dim"])).astype(np.float32),
                "actions":      rng.standard_normal((200, cfg["act_dim"])).astype(np.float32),
                "rewards":      rng.random(200).astype(np.float32),
                "terminals":    np.zeros(200, dtype=bool),
            }
            for _ in range(50)
        ]
        print("Smoke test: using 50 synthetic trajectories (no d4rl needed)")
    else:
        import minari

        dataset = minari.load_dataset(cfg["dataset_id"], download=False)
        trajectories = parse_minari_dataset(dataset)
        print(f"Loaded {len(trajectories)} trajectories")

    # --- State normalization (paper §A.1) ---
    state_mean, state_std = compute_state_stats(trajectories)
    trajectories = normalize_states(trajectories, state_mean, state_std)

    # Save stats so evaluation can use the same normalization
    np.save(os.path.join(run_dir, "state_mean.npy"), state_mean)
    np.save(os.path.join(run_dir, "state_std.npy"), state_std)

    # --- Dataset / DataLoader ---
    train_dataset = TrajectoryDataset(
        trajectories,
        context_len=cfg["context_len"],
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
        rtg_scale=cfg.get("rtg_scale", 1.0),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    # --- Model ---
    model = DecisionTransformer(
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
        hidden_size=cfg["hidden_size"],
        max_length=cfg["context_len"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    warmup_steps = cfg.get("warmup_steps", 10000)

    def lr_lambda(step: int) -> float:
        # Linear warmup, then constant
        return min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    trainer = Trainer(model, optimizer, device=device, scheduler=scheduler)
    max_steps = args.max_steps or cfg.get("max_steps")

    # --- Smoke test: just verify the pipeline runs end-to-end ---
    if args.smoke:
        model.train()
        batch = next(iter(train_loader))
        loss = trainer.train_step(batch)
        batch = next(iter(train_loader))
        loss = trainer.train_step(batch)
        print(f"Smoke test passed. Loss on 2 batches: {loss:.4f}")
        return

    # --- Training loop ---
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    if max_steps:
        print(f"Training for {max_steps} gradient steps...")
        trainer.train_steps(train_loader, max_steps=max_steps)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Done. Saved to {ckpt_path}")
    else:
        best_loss = float("inf")
        for epoch in range(1, cfg["n_epochs"] + 1):
            loss = trainer.train_epoch(train_loader)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), ckpt_path)
        print(f"Training complete. Best loss: {best_loss:.4f} | Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
