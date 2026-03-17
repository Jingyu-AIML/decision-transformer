"""
Training loop for Decision Transformer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.loss_fn = nn.MSELoss()

    def train_step(self, batch):
        """Single gradient update step."""
        states, actions, returns_to_go, timesteps, mask = [x.to(self.device) for x in batch]

        action_preds = self.model(states, actions, returns_to_go, timesteps)

        # Only compute loss on non-padded timesteps
        action_preds = action_preds[mask > 0]
        action_targets = actions[mask > 0]

        loss = self.loss_fn(action_preds, action_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.25)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            total_loss += self.train_step(batch)
        return total_loss / len(dataloader)

    def train_steps(self, dataloader: DataLoader, max_steps: int, log_interval: int = 100):
        """Train for a fixed number of gradient steps, cycling through the dataloader."""
        self.model.train()
        loader_iter = iter(dataloader)
        total_loss = 0.0

        for step in range(1, max_steps + 1):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                batch = next(loader_iter)

            loss = self.train_step(batch)
            total_loss += loss

            if step % log_interval == 0:
                avg = total_loss / log_interval
                print(f"  Step {step:>6d}/{max_steps} | Loss: {avg:.4f}")
                total_loss = 0.0
