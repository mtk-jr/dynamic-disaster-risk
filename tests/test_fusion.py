import torch
from src.models.fusion_model import EarlyFusionModel

def test_fusion_shape():
    model = EarlyFusionModel()
    output = model(
        torch.randn(4, 8),
        torch.randn(4, 8),
        torch.randn(4, 8),
        torch.randn(4, 8),
    )
    assert output.shape == (4, 4)
