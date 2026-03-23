"""
Utility functions: data loading, trajectory processing, evaluation helpers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_d4rl_dataset(dataset: dict) -> list[dict]:
    """
    Convert a raw d4rl dataset dict into a list of trajectory dicts.

    d4rl stores everything as flat arrays of length N (total transitions).
    Episode boundaries are indicated by ``terminals`` (true done) or
    ``timeouts`` (episode cut off due to time limit — the agent did not
    actually reach a terminal state).

    Args:
        dataset: dict returned by ``env.get_dataset()``, with keys
                 'observations', 'actions', 'rewards', 'terminals',
                 and optionally 'timeouts'.

    Returns:
        List of dicts, each with keys:
            'observations'  – np.ndarray (T, state_dim)
            'actions'       – np.ndarray (T, act_dim)
            'rewards'       – np.ndarray (T,)
            'terminals'     – np.ndarray (T,) bool
    """
    obs     = dataset["observations"]
    acts    = dataset["actions"]
    rews    = dataset["rewards"]
    terms   = dataset["terminals"].astype(bool)
    # timeouts mark end-of-episode but are NOT true terminal states
    timeouts = dataset.get("timeouts", np.zeros_like(terms, dtype=bool)).astype(bool)

    episode_ends = np.where(terms | timeouts)[0]

    trajectories = []
    start = 0
    for end in episode_ends:
        end = int(end) + 1          # slice is exclusive
        trajectories.append({
            "observations": obs[start:end],
            "actions":      acts[start:end],
            "rewards":      rews[start:end],
            "terminals":    terms[start:end],
        })
        start = end

    # Include any trailing transitions not closed by a terminal/timeout
    if start < len(obs):
        trajectories.append({
            "observations": obs[start:],
            "actions":      acts[start:],
            "rewards":      rews[start:],
            "terminals":    terms[start:],
        })

    return trajectories


class TrajectoryDataset(Dataset):
    """
    Wraps an offline dataset of trajectories into fixed-length context windows
    suitable for Decision Transformer training.
    """

    def __init__(self, trajectories, context_len: int, state_dim: int, act_dim: int, rtg_scale: float = 1.0):
        """
        Args:
            trajectories: list of dicts with keys:
                          'observations', 'actions', 'rewards', 'terminals'
            context_len:  K, the number of timesteps per sample
            state_dim:    dimensionality of state space
            act_dim:      dimensionality of action space
            rtg_scale:    divide RTG values by this before feeding to the model
                          (paper §A.1: scale=1000 for most MuJoCo envs)
        """
        self.context_len = context_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rtg_scale = rtg_scale

        # Compute returns-to-go for each trajectory
        self.states, self.actions, self.returns_to_go, self.timesteps, self.traj_lens = (
            [], [], [], [], []
        )

        for traj in trajectories:
            obs = np.array(traj["observations"], dtype=np.float32)
            acts = np.array(traj["actions"], dtype=np.float32)
            rews = np.array(traj["rewards"], dtype=np.float32)

            rtg = self._compute_rtg(rews) / self.rtg_scale

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


def parse_minari_dataset(dataset) -> list[dict]:
    """
    Convert a Minari dataset into a list of trajectory dicts.

    Minari stores T+1 observations per episode (includes the final next-obs),
    so we drop the last observation to align with actions/rewards.

    Args:
        dataset: MinariDataset returned by ``minari.load_dataset()``.

    Returns:
        List of dicts with keys 'observations', 'actions', 'rewards', 'terminals'.
    """
    trajectories = []
    for episode in dataset.iterate_episodes():
        obs = episode.observations
        # Some envs return obs as a dict — flatten to array
        if isinstance(obs, dict):
            obs = np.concatenate([np.atleast_2d(v) for v in obs.values()], axis=-1)
        obs = np.array(obs[:-1], dtype=np.float32)   # drop final next-obs

        trajectories.append({
            "observations": obs,
            "actions":      np.array(episode.actions, dtype=np.float32),
            "rewards":      np.array(episode.rewards, dtype=np.float32),
            "terminals":    np.array(episode.terminations, dtype=bool),
        })
    return trajectories


def compute_state_stats(trajectories: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std over all states in the dataset."""
    all_obs = np.concatenate([t["observations"] for t in trajectories], axis=0)
    mean = all_obs.mean(axis=0)
    std  = all_obs.std(axis=0) + 1e-8
    return mean, std


def normalize_states(trajectories: list[dict], mean: np.ndarray, std: np.ndarray) -> list[dict]:
    """Return new trajectory list with observations normalized in-place."""
    normalized = []
    for t in trajectories:
        normalized.append({**t, "observations": (t["observations"] - mean) / std})
    return normalized
