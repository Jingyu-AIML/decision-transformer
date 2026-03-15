"""
Decision Transformer model.

Reference: Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)
"""

import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        max_length: int = 20,       # context length (K in the paper)
        max_ep_len: int = 1000,     # max episode length for timestep embedding
        n_layer: int = 3,
        n_head: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        # Input embeddings
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Causal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # Action prediction head
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        Args:
            states:         (B, T, state_dim)
            actions:        (B, T, act_dim)
            returns_to_go:  (B, T, 1)
            timesteps:      (B, T)
        Returns:
            action_preds:   (B, T, act_dim)
        """
        B, T, _ = states.shape

        # Embed each modality
        t_emb = self.embed_timestep(timesteps)                   # (B, T, H)
        r_emb = self.embed_return(returns_to_go) + t_emb         # (B, T, H)
        s_emb = self.embed_state(states) + t_emb                 # (B, T, H)
        a_emb = self.embed_action(actions) + t_emb               # (B, T, H)

        # Interleave: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Stack to (B, T, 3, H) then reshape to (B, 3T, H)
        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2)      # (B, T, 3, H)
        stacked = stacked.reshape(B, 3 * T, self.hidden_size)     # (B, 3T, H)
        stacked = self.embed_ln(stacked)

        # Causal mask: each token only attends to past tokens
        seq_len = stacked.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=states.device)

        x = self.transformer(stacked, mask=causal_mask, is_causal=True)  # (B, 3T, H)

        # Extract state token outputs (positions 1, 4, 7, ... i.e. index 1 mod 3)
        # These predict the action to take
        x = x.reshape(B, T, 3, self.hidden_size)   # (B, T, 3, H)
        state_repr = x[:, :, 1, :]                  # (B, T, H)  — state tokens

        action_preds = self.predict_action(state_repr)  # (B, T, act_dim)
        return action_preds
