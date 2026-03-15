"""
Basic sanity tests for DecisionTransformer.

Run with:
    pytest tests/
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DecisionTransformer


STATE_DIM = 11
ACT_DIM = 3
HIDDEN = 64
CONTEXT_LEN = 10
BATCH = 4


@pytest.fixture
def model():
    return DecisionTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        hidden_size=HIDDEN,
        max_length=CONTEXT_LEN,
        n_layer=2,
        n_head=2,
    )


def make_batch(B=BATCH, T=CONTEXT_LEN):
    states = torch.randn(B, T, STATE_DIM)
    actions = torch.randn(B, T, ACT_DIM)
    returns_to_go = torch.randn(B, T, 1)
    timesteps = torch.arange(T).unsqueeze(0).expand(B, -1)
    return states, actions, returns_to_go, timesteps


def test_output_shape(model):
    states, actions, rtg, timesteps = make_batch()
    out = model(states, actions, rtg, timesteps)
    assert out.shape == (BATCH, CONTEXT_LEN, ACT_DIM), f"Expected {(BATCH, CONTEXT_LEN, ACT_DIM)}, got {out.shape}"


def test_output_range(model):
    """Actions should be in [-1, 1] due to Tanh head."""
    states, actions, rtg, timesteps = make_batch()
    out = model(states, actions, rtg, timesteps)
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_different_batch_sizes(model):
    for B in [1, 2, 8]:
        states, actions, rtg, timesteps = make_batch(B=B)
        out = model(states, actions, rtg, timesteps)
        assert out.shape == (B, CONTEXT_LEN, ACT_DIM)


def test_different_context_lengths(model):
    for T in [1, 5, CONTEXT_LEN]:
        states, actions, rtg, timesteps = make_batch(T=T)
        out = model(states, actions, rtg, timesteps)
        assert out.shape == (BATCH, T, ACT_DIM)


def test_gradient_flows(model):
    states, actions, rtg, timesteps = make_batch()
    out = model(states, actions, rtg, timesteps)
    loss = out.mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_no_nan_in_output(model):
    states, actions, rtg, timesteps = make_batch()
    out = model(states, actions, rtg, timesteps)
    assert not torch.isnan(out).any(), "NaN detected in model output"
