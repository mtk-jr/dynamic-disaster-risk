from pathlib import Path

import numpy as np
import rasterio
import zarr

from src.ingestion.impactmesh import ImpactMeshReader


def create_fake_impactmesh(root: Path) -> None:
    for modality in ["S2L2A", "S1RTC", "DEM", "MASK"]:
        (root / modality).mkdir(parents=True)

    patch_id = "event_001"

    # Create S2 Zarr.
    s2_path = (
        root
        / "S2L2A"
        / f"{patch_id}_S2L2A.zarr.zip"
    )

    store = zarr.ZipStore(str(s2_path), mode="w")

    try:
        root_group = zarr.group(store=store)

        root_group.create_dataset(
            "bands",
            shape=(4, 12, 12, 12),
            dtype="float32",
            data=np.ones((4, 12, 12, 12), dtype=np.float32),
        )

        zarr.consolidate_metadata(store)
    finally:
        store.close()

    # Create S1 Zarr.
    s1_path = (
        root
        / "S1RTC"
        / f"{patch_id}_S1RTC.zarr.zip"
    )

    store = zarr.ZipStore(str(s1_path), mode="w")

    try:
        root_group = zarr.group(store=store)
        root_group.create_dataset(
            "bands",
            shape=(4, 12, 12, 2),
            dtype="float32",
            data=np.ones((4, 12, 12, 2), dtype=np.float32),
        )
        zarr.consolidate_metadata(store)
    finally:
        store.close()

    transform = rasterio.transform.from_origin(
        76.0,
        10.0,
        0.0001,
        0.0001,
    )

    # DEM.
    with rasterio.open(
        root / "DEM" / f"{patch_id}_DEM.tif",
        "w",
        driver="GTiff",
        height=12,
        width=12,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.ones((1, 12, 12), dtype=np.float32))

    # Flood mask.
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[:6, :] = 1

    with rasterio.open(
        root / "MASK" / f"{patch_id}_annotation_flood.tif",
        "w",
        driver="GTiff",
        height=12,
        width=12,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(mask, 1)


def test_impactmesh_reader(tmp_path):
    create_fake_impactmesh(tmp_path)

    reader = ImpactMeshReader(tmp_path)

    samples = reader.available_samples()

    assert samples == ["event_001"]

    sample = reader.read_sample("event_001")

    assert sample["S2L2A"].shape == (4, 12, 12, 12)
    assert sample["S1RTC"].shape == (4, 12, 12, 2)
    assert sample["DEM"].shape == (12, 12)
    assert sample["mask"].shape == (12, 12)

    ratio = reader.flood_ratio(sample["mask"])

    assert ratio == 0.5