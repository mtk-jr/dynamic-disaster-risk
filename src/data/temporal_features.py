from __future__ import annotations

import numpy as np
import pandas as pd


S1_FEATURES = [
    "vh_mean",
    "vh_std",
    "vv_mean",
    "vv_std",
    "vv_vh_difference",
]


S2_FEATURES = [
    "mndwi_mean",
    "mndwi_std",
    "ndvi_mean",
    "ndwi_mean",
    "ndwi_std",
    "water_ratio",
]


TEMPORAL_STEPS = ["t0", "t1", "t2", "t3"]


def build_temporal_tensor(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Convert flattened ImpactMesh remote-sensing
    features into a temporal tensor.

    Returns:

        shape = (samples, 4, 11)

    where:

        4  = temporal observations
        11 = S1 + S2 features
    """

    tensors = []

    for _, row in df.iterrows():

        temporal_steps = []

        for t in TEMPORAL_STEPS:

            features = []

            for feature in S1_FEATURES:
                features.append(
                    row[f"s1_{t}_{feature}"]
                )

            for feature in S2_FEATURES:
                features.append(
                    row[f"s2_{t}_{feature}"]
                )

            temporal_steps.append(features)

        tensors.append(temporal_steps)

    return np.asarray(
        tensors,
        dtype=np.float32,
    )