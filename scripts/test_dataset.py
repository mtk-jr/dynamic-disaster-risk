from src.data.dataset import create_splits


def main():

    data = create_splits()

    print("Dataset loaded successfully.")
    print()

    print(
        "Remote:",
        data["remote"].shape
    )

    print(
        "GIS:",
        data["gis"].shape
    )

    print(
        "Target:",
        data["target"].shape
    )

    print()

    print(
        "Training samples:",
        len(data["train_idx"])
    )

    print(
        "Validation samples:",
        len(data["val_idx"])
    )

    print(
        "Test samples:",
        len(data["test_idx"])
    )


if __name__ == "__main__":
    main()