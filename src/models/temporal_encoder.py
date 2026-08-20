from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):

    def __init__(
        self,
        input_features: int = 11,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_features,
            hidden_dim,
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # x:
        # [batch, time, features]

        x = self.input_projection(x)

        x = torch.relu(x)

        sequence, hidden = self.gru(x)

        # Last temporal state
        last_state = sequence[:, -1, :]

        embedding = self.output_projection(
            last_state
        )

        return embedding