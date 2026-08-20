from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.temporal_features import build_temporal_tensor


REMOTE_FILE = "data/processed/impactmesh_remote.csv"
GIS_FILE = "data/processed/impactmesh_gis.csv"
LABEL_FILE = "data/processed/impactmesh_labels.csv"


def load_impactmesh():

    remote = pd.read_csv(REMOTE_FILE)
    gis = pd.read_csv(GIS_FILE)
    labels = pd.read_csv(LABEL_FILE)

    # Make sure all three datasets have the same samples.
    if not (
        remote["sample_id"].equals(gis["sample_id"])
        and remote["sample_id"].equals(labels["sample_id"])
    ):
        raise ValueError(
            "sample_id mismatch between datasets."
        )

    remote_tensor = build_temporal_tensor(remote)

    gis_features = gis.drop(
        columns=["sample_id"]
    ).to_numpy(
        dtype=np.float32
    )

    target = labels["flood_ratio"].to_numpy(
        dtype=np.float32
    )

    sample_ids = remote["sample_id"].to_numpy()

    return (
        sample_ids,
        remote_tensor,
        gis_features,
        target,
    )


def create_splits(
    random_state: int = 42,
):

    (
        sample_ids,
        remote,
        gis,
        target,
    ) = load_impactmesh()

    indices = np.arange(
        len(sample_ids)
    )

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
    )

    # ------------------------------------------------
    # Fit scalers ONLY on training data
    # ------------------------------------------------

    remote_scaler = StandardScaler()

    train_remote_flat = remote[
        train_idx
    ].reshape(
        len(train_idx),
        -1,
    )

    remote_scaler.fit(
        train_remote_flat
    )

    remote_scaled = remote_scaler.transform(
        remote.reshape(
            len(remote),
            -1,
        )
    ).reshape(
        remote.shape
    )

    # ------------------------------------------------
    # GIS scaler
    # ------------------------------------------------

    gis_scaler = StandardScaler()

    gis_scaler.fit(
        gis[train_idx]
    )

    gis_scaled = gis_scaler.transform(
        gis
    ).astype(
        np.float32
    )

    return {
        "sample_ids": sample_ids,

        "remote": remote_scaled.astype(
            np.float32
        ),

        "gis": gis_scaled,

        "target": target,

        "train_idx": train_idx,

        "val_idx": val_idx,

        "test_idx": test_idx,

        "remote_scaler": remote_scaler,

        "gis_scaler": gis_scaler,
    }