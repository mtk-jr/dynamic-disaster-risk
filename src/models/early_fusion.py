from __future__ import annotations

import torch
import torch.nn as nn

from src.models.temporal_encoder import TemporalEncoder
from src.models.gis_encoder import GISEncoder


class EarlyFusionModel(nn.Module):

    def __init__(
        self,
        remote_features: int = 11,
        gis_features: int = 8,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.remote_encoder = TemporalEncoder(
            input_features=remote_features,
            embedding_dim=embedding_dim,
        )

        self.gis_encoder = GISEncoder(
            input_features=gis_features,
            embedding_dim=embedding_dim,
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                128,
            ),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        remote: torch.Tensor,
        gis: torch.Tensor,
    ):

        remote_embedding = self.remote_encoder(
            remote
        )

        gis_embedding = self.gis_encoder(
            gis
        )

        fused = torch.cat(
            [
                remote_embedding,
                gis_embedding,
            ],
            dim=1,
        )

        output = self.fusion(fused)

        # flood_ratio must be between 0 and 1
        output = torch.sigmoid(output)

        return output.squeeze(1)