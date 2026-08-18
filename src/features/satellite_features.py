from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


REQUIRED_BANDS = ("B03", "B04", "B08", "B11")


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
    """Return a finite standard deviation, or 0.0 if no valid pixels exist."""
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.std(values))


def _normalise_band_name(name: str) -> str:
    """Normalize common Sentinel-2 band naming variants."""
    name = name.strip().upper().replace("-", "").replace("_", "")

    if name.startswith("B"):
        return name

    return f"B{name}"


def _get_band_map(src: rasterio.DatasetReader) -> dict[str, int]:
    """
    Build a Sentinel-2 band-name → raster band-index mapping.

    Band descriptions are expected to contain names such as B03, B04,
    B08 and B11.
    """
    band_map: dict[str, int] = {}

    for index, description in enumerate(src.descriptions, start=1):
        if description is None:
            continue

        name = _normalise_band_name(description)

        if name in REQUIRED_BANDS:
            band_map[name] = index

    missing = [
        band for band in REQUIRED_BANDS
        if band not in band_map
    ]

    if missing:
        raise ValueError(
            "Missing required Sentinel-2 bands: "
            + ", ".join(missing)
            + ". Raster band descriptions must identify "
              "B03, B04, B08 and B11."
        )

    return band_map


def extract_sentinel2_features(
    path: str | Path,
) -> dict[str, float]:
    """
    Extract an 8-dimensional flood-oriented feature vector
    from a Sentinel-2 GeoTIFF.

    Required bands:
        B03 - Green
        B04 - Red
        B08 - NIR
        B11 - SWIR

    The raster must identify these bands through Rasterio band
    descriptions.
    """
    path = Path(path)

    with rasterio.open(path) as src:
        band_map = _get_band_map(src)

        b03 = src.read(band_map["B03"]).astype(np.float32)
        b04 = src.read(band_map["B04"]).astype(np.float32)
        b08 = src.read(band_map["B08"]).astype(np.float32)
        b11 = src.read(band_map["B11"]).astype(np.float32)

        nodata = src.nodata

    if nodata is not None:
        b03[b03 == nodata] = np.nan
        b04[b04 == nodata] = np.nan
        b08[b08 == nodata] = np.nan
        b11[b11 == nodata] = np.nan

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

    water_mask = (mndwi > 0.0) & valid

    water_ratio = float(np.mean(water_mask[valid]))

    return {
        "ndvi_mean": _safe_mean(ndvi),
        "ndwi_mean": _safe_mean(ndwi),
        "mndwi_mean": _safe_mean(mndwi),
        "ndwi_std": _safe_std(ndwi),
        "mndwi_std": _safe_std(mndwi),
        "water_ratio": water_ratio,
        "valid_pixel_ratio": float(valid_pixels / valid.size),
        "mean_nir": _safe_mean(b08),
    }