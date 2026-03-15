"""
Utility functions: data loading, trajectory processing, evaluation helpers.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Wraps an offline dataset of trajectories into fixed-length context windows
    suitable for Decision Transformer training.
    """

    def __init__(self, trajectories, context_len: int, state_dim: int, act_dim: int):
        """
        Args:
            trajectories: list of dicts with keys:
                          'observations', 'actions', 'rewards', 'terminals'
            context_len:  K, the number of timesteps per sample
            state_dim:    dimensionality of state space
            act_dim:      dimensionality of action space
        """
        self.context_len = context_len
        self.state_dim = state_dim
        self.act_dim = act_dim

        # Compute returns-to-go for each trajectory
        self.states, self.actions, self.returns_to_go, self.timesteps, self.traj_lens = (
            [], [], [], [], []
        )

        for traj in trajectories:
            obs = np.array(traj["observations"], dtype=np.float32)
            acts = np.array(traj["actions"], dtype=np.float32)
            rews = np.array(traj["rewards"], dtype=np.float32)

            rtg = self._compute_rtg(rews)

            self.states.append(obs)
            self.actions.append(acts)
            self.returns_to_go.append(rtg)
            self.timesteps.append(np.arange(len(obs)))
            self.traj_lens.append(len(obs))

        # Sample probability proportional to trajectory length
        traj_lens = np.array(self.traj_lens)
        self.p_sample = traj_lens / traj_lens.sum()

    @staticmethod
    def _compute_rtg(rewards: np.ndarray) -> np.ndarray:
        rtg = np.zeros_like(rewards)
        rtg[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            rtg[t] = rewards[t] + rtg[t + 1]
        return rtg.reshape(-1, 1).astype(np.float32)

    def __len__(self):
        return sum(self.traj_lens)

    def __getitem__(self, idx):
        # Sample a trajectory proportional to length, then a random window
        traj_idx = np.random.choice(len(self.traj_lens), p=self.p_sample)
        traj_len = self.traj_lens[traj_idx]

        # Random start within trajectory
        start = np.random.randint(0, traj_len)
        end = min(start + self.context_len, traj_len)
        actual_len = end - start

        # Pad to context_len if necessary
        def pad(arr, pad_val=0.0):
            pad_len = self.context_len - actual_len
            if arr.ndim == 1:
                return np.concatenate([np.full(pad_len, pad_val), arr[start:end]])
            return np.concatenate([np.full((pad_len, arr.shape[1]), pad_val), arr[start:end]])

        states = pad(self.states[traj_idx])
        actions = pad(self.actions[traj_idx])
        returns_to_go = pad(self.returns_to_go[traj_idx])
        timesteps = pad(self.timesteps[traj_idx], pad_val=0).astype(np.int64)

        # Attention mask: 0 for padded, 1 for real
        mask = np.concatenate([
            np.zeros(self.context_len - actual_len),
            np.ones(actual_len),
        ])

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(returns_to_go, dtype=torch.float32),
            torch.tensor(timesteps, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )


def normalize_states(states: np.ndarray):
    mean = states.mean(axis=0)
    std = states.std(axis=0) + 1e-8
    return (states - mean) / std, mean, std
