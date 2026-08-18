from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def normalized_difference(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Calculate a normalized difference index."""
    return (a - b) / (a + b + eps)


def _safe_mean(values: np.ndarray) -> float:
    """Return a finite mean, or 0.0 if no valid pixels exist."""
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    """Return a finite standard deviation."""
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.std(values))


def extract_sentinel2_features(
    path: str | Path,
) -> dict[str, float]:
    """
    Extract an 8-dimensional flood-oriented feature vector
    from a Sentinel-2 GeoTIFF.

    Expected bands:
        B03 - Green
        B04 - Red
        B08 - NIR
        B11 - SWIR

    Returns:
        Dictionary containing eight numerical features.
    """

    path = Path(path)

    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(
                "The raster must contain at least four bands "
                "(B03, B04, B08, B11)."
            )

        data = src.read().astype(np.float32)

        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = np.nan

    # Current baseline assumes the first four channels correspond
    # to B03, B04, B08 and B11.
    b03 = data[0]
    b04 = data[1]
    b08 = data[2]
    b11 = data[3]

    ndvi = normalized_difference(b08, b04)
    ndwi = normalized_difference(b03, b08)
    mndwi = normalized_difference(b03, b11)

    valid = (
        np.isfinite(b03)
        & np.isfinite(b04)
        & np.isfinite(b08)
        & np.isfinite(b11)
    )

    valid_pixels = int(np.sum(valid))

    if valid_pixels == 0:
        raise ValueError("Raster contains no valid pixels.")

    # Simple water/flood-oriented thresholds.
    water_mask = (mndwi > 0.0) & valid

    water_ratio = float(np.mean(water_mask[valid]))

    features = {
        "ndvi_mean": _safe_mean(ndvi),
        "ndwi_mean": _safe_mean(ndwi),
        "mndwi_mean": _safe_mean(mndwi),
        "ndwi_std": _safe_std(ndwi),
        "mndwi_std": _safe_std(mndwi),
        "water_ratio": water_ratio,
        "valid_pixel_ratio": float(
            valid_pixels / valid.size
        ),
        "mean_nir": _safe_mean(b08),
    }

    return features