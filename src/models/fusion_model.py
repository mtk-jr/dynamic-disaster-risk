import torch
from torch import nn

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        satellite_dim=8,
        iot_dim=8,
        gis_dim=8,
        social_dim=8,
        hidden_dim=128,
        fusion_dim=64,
        num_classes=4,
        dropout=0.2,
    ):
        super().__init__()

        self.satellite_encoder = ModalityEncoder(
            satellite_dim, hidden_dim, fusion_dim, dropout
        )
        self.iot_encoder = ModalityEncoder(
            iot_dim, hidden_dim, fusion_dim, dropout
        )
        self.gis_encoder = ModalityEncoder(
            gis_dim, hidden_dim, fusion_dim, dropout
        )
        self.social_encoder = ModalityEncoder(
            social_dim, hidden_dim, fusion_dim, dropout
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, satellite, iot, gis, social):
        z_sat = self.satellite_encoder(satellite)
        z_iot = self.iot_encoder(iot)
        z_gis = self.gis_encoder(gis)
        z_social = self.social_encoder(social)

        fused = torch.cat([z_sat, z_iot, z_gis, z_social], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)
