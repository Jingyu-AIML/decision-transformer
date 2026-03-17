"""
Training entry point for Decision Transformer.

Usage:
    python scripts/train.py --config configs/hopper_medium.yaml
"""

import argparse
import os
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

        dataset = minari.load_dataset(cfg["dataset_id"], download=True)
        trajectories = parse_minari_dataset(dataset)
        print(f"Loaded {len(trajectories)} trajectories")

    # --- State normalization (paper §A.1) ---
    state_mean, state_std = compute_state_stats(trajectories)
    trajectories = normalize_states(trajectories, state_mean, state_std)

    # Save stats so evaluation can use the same normalization
    np.save("checkpoints/state_mean.npy", state_mean)
    np.save("checkpoints/state_std.npy", state_std)

    # --- Dataset / DataLoader ---
    train_dataset = TrajectoryDataset(
        trajectories,
        context_len=cfg["context_len"],
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
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

    trainer = Trainer(model, optimizer, device=device)
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
    os.makedirs("checkpoints", exist_ok=True)
    if max_steps:
        print(f"Training for {max_steps} gradient steps...")
        trainer.train_steps(train_loader, max_steps=max_steps)
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print("Done.")
    else:
        best_loss = float("inf")
        for epoch in range(1, cfg["n_epochs"] + 1):
            loss = trainer.train_epoch(train_loader)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
