import pandas as pd

from src.data.temporal_features import (
    build_temporal_tensor,
)


INPUT = "data/processed/impactmesh_remote.csv"


def main():

    print("Loading remote-sensing dataset...")

    df = pd.read_csv(INPUT)

    print("Input shape:")
    print(df.shape)

    tensor = build_temporal_tensor(df)

    print()
    print("Temporal tensor created:")
    print(tensor.shape)

    print()
    print("Expected:")
    print("(samples, time, features)")

    print()
    print("First sample:")
    print(tensor[0])


if __name__ == "__main__":
    main()