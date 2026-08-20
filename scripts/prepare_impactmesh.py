from pathlib import Path

import pandas as pd
import numpy as np

from src.data.impactmesh_schema import (
    REMOTE_FEATURES,
    GIS_FEATURES,
    TARGET,
    ID_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "impactmesh_features.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
)


def main():

    print("Loading ImpactMesh dataset...")

    df = pd.read_csv(INPUT)

    print(f"Original shape: {df.shape}")

    # ------------------------------------------------
    # Validate required columns
    # ------------------------------------------------

    required_columns = (
        ID_COLUMNS
        + REMOTE_FEATURES
        + GIS_FEATURES
        + [TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    # ------------------------------------------------
    # Replace invalid numerical values
    # ------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ------------------------------------------------
    # Remote sensing dataset
    # ------------------------------------------------

    remote_df = df[
        ID_COLUMNS + REMOTE_FEATURES
    ].copy()

    # ------------------------------------------------
    # GIS dataset
    # ------------------------------------------------

    gis_df = df[
        ID_COLUMNS + GIS_FEATURES
    ].copy()

    # ------------------------------------------------
    # Target dataset
    # ------------------------------------------------

    labels_df = df[
        ID_COLUMNS + [TARGET]
    ].copy()

    # ------------------------------------------------
    # Save datasets
    # ------------------------------------------------

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    remote_path = OUTPUT / "impactmesh_remote.csv"
    gis_path = OUTPUT / "impactmesh_gis.csv"
    labels_path = OUTPUT / "impactmesh_labels.csv"

    remote_df.to_csv(
        remote_path,
        index=False
    )

    gis_df.to_csv(
        gis_path,
        index=False
    )

    labels_df.to_csv(
        labels_path,
        index=False
    )

    # ------------------------------------------------
    # Report
    # ------------------------------------------------

    print()
    print("Datasets created successfully.")
    print()

    print(f"Remote sensing : {remote_df.shape}")
    print(f"GIS            : {gis_df.shape}")
    print(f"Labels         : {labels_df.shape}")

    print()
    print("Files:")

    print(remote_path)
    print(gis_path)
    print(labels_path)


if __name__ == "__main__":
    main()