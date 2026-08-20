import torch

from src.data.dataset import create_splits
from src.models.early_fusion import EarlyFusionModel


def main():

    data = create_splits()

    model = EarlyFusionModel()

    train_idx = data["train_idx"][:8]

    remote = torch.tensor(
        data["remote"][train_idx]
    )

    gis = torch.tensor(
        data["gis"][train_idx]
    )

    target = torch.tensor(
        data["target"][train_idx]
    )

    output = model(
        remote,
        gis,
    )

    print("Remote input:", remote.shape)
    print("GIS input:", gis.shape)
    print("Target:", target.shape)
    print("Model output:", output.shape)

    print()
    print("Output:")
    print(output)


if __name__ == "__main__":
    main()