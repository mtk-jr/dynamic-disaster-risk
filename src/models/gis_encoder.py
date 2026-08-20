from __future__ import annotations

import torch
import torch.nn as nn


class GISEncoder(nn.Module):

    def __init__(
        self,
        input_features: int = 8,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_features,
                64,
            ),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                embedding_dim,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)