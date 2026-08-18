import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.features.satellite_features import extract_sentinel2_features


def test_extract_sentinel2_features(tmp_path):
    raster_path = tmp_path / "sentinel2_test.tif"

    # Four synthetic bands:
    # B03 = Green
    # B04 = Red
    # B08 = NIR
    # B11 = SWIR
    data = np.array(
        [
            np.full((10, 10), 0.6, dtype=np.float32),  # B03
            np.full((10, 10), 0.3, dtype=np.float32),  # B04
            np.full((10, 10), 0.7, dtype=np.float32),  # B08
            np.full((10, 10), 0.2, dtype=np.float32),  # B11
        ]
    )

    transform = from_origin(
        76.0,
        10.0,
        0.0001,
        0.0001,
    )

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    features = extract_sentinel2_features(raster_path)

    assert len(features) == 8

    for name, value in features.items():
        assert isinstance(value, float)
        assert np.isfinite(value), f"{name} is not finite"

    assert "ndvi_mean" in features
    assert "ndwi_mean" in features
    assert "mndwi_mean" in features
    assert "water_ratio" in features