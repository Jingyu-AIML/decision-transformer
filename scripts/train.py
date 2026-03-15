"""
Training entry point for Decision Transformer.

Usage:
    python scripts/train.py --config configs/hopper_medium.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DecisionTransformer
from src.trainer import Trainer
from src.utils import TrajectoryDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hopper_medium.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load dataset (placeholder: swap in d4rl later) ---
    # import d4rl, gymnasium as gym
    # env = gym.make(cfg["env_name"])
    # dataset = env.get_dataset()
    # trajectories = parse_d4rl_dataset(dataset)
    raise NotImplementedError(
        "Dataset loading not yet implemented. "
        "Wire up d4rl or minari here, then remove this line."
    )

    train_dataset = TrajectoryDataset(
        trajectories,
        context_len=cfg["context_len"],
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = DecisionTransformer(
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
        hidden_size=cfg["hidden_size"],
        max_length=cfg["context_len"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    for epoch in range(1, cfg["n_epochs"] + 1):
        loss = Trainer(model, optimizer, device=device).train_epoch(train_loader)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
