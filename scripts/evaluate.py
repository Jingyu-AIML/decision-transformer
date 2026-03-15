"""
Evaluation script: roll out a trained Decision Transformer in an environment.

Usage:
    python scripts/evaluate.py --checkpoint path/to/model.pt --config configs/hopper_medium.yaml
"""

import argparse
import yaml
import torch
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DecisionTransformer


def evaluate(model, env, target_return: float, context_len: int, device: str, state_mean, state_std):
    model.eval()
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Sliding context buffers
    states = torch.zeros(1, context_len, state_dim, device=device)
    actions = torch.zeros(1, context_len, act_dim, device=device)
    returns_to_go = torch.zeros(1, context_len, 1, device=device)
    timesteps = torch.zeros(1, context_len, dtype=torch.long, device=device)

    obs, _ = env.reset()
    obs = (obs - state_mean) / state_std
    episode_return = 0.0

    for t in range(env.spec.max_episode_steps or 1000):
        # Shift context and insert current step
        states = states.roll(-1, dims=1)
        actions = actions.roll(-1, dims=1)
        returns_to_go = returns_to_go.roll(-1, dims=1)
        timesteps = timesteps.roll(-1, dims=1)

        states[0, -1] = torch.tensor(obs, dtype=torch.float32)
        returns_to_go[0, -1] = target_return - episode_return
        timesteps[0, -1] = t

        with torch.no_grad():
            action_preds = model(states, actions, returns_to_go, timesteps)
        action = action_preds[0, -1].cpu().numpy()
        action = np.clip(action, env.action_space.low, env.action_space.high)

        actions[0, -1] = torch.tensor(action, dtype=torch.float32)

        obs, reward, terminated, truncated, _ = env.step(action)
        obs = (obs - state_mean) / state_std
        episode_return += reward

        if terminated or truncated:
            break

    return episode_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/hopper_medium.yaml")
    parser.add_argument("--target_return", type=float, default=None)
    parser.add_argument("--n_eval", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_return = args.target_return or cfg["target_return"]

    model = DecisionTransformer(
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
        hidden_size=cfg["hidden_size"],
        max_length=cfg["context_len"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # TODO: load state_mean, state_std from saved stats
    state_mean = np.zeros(cfg["state_dim"])
    state_std = np.ones(cfg["state_dim"])

    import gymnasium as gym
    env = gym.make(cfg["env_name"])

    returns = []
    for ep in range(args.n_eval):
        ret = evaluate(model, env, target_return, cfg["context_len"], device, state_mean, state_std)
        returns.append(ret)
        print(f"Episode {ep+1}: return = {ret:.2f}")

    print(f"\nMean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")


if __name__ == "__main__":
    main()
