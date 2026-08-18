from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import zarr


DEFAULT_MODALITIES = ("S2L2A", "S1RTC", "DEM")

DEFAULT_PATTERNS = {
    "S2L2A": "_S2L2A.zarr.zip",
    "S1RTC": "_S1RTC.zarr.zip",
    "DEM": "_DEM.tif",
}

DEFAULT_LABEL_PATTERN = "_annotation_flood.tif"


class ImpactMeshReader:
    """
    Lightweight reader for ImpactMesh flood samples.

    This adapter intentionally does not depend on TerraTorch.
    It allows our project to inspect and preprocess ImpactMesh
    data independently of the original training framework.
    """

    def __init__(
        self,
        data_root: str | Path,
        modalities: tuple[str, ...] = DEFAULT_MODALITIES,
    ) -> None:
        self.data_root = Path(data_root)
        self.modalities = modalities

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"ImpactMesh data root does not exist: {self.data_root}"
            )

    def sample_path(
        self,
        patch_id: str,
        modality: str,
    ) -> Path:
        if modality not in DEFAULT_PATTERNS:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                f"Expected one of {list(DEFAULT_PATTERNS)}."
            )

        return (
            self.data_root
            / modality
            / f"{patch_id}{DEFAULT_PATTERNS[modality]}"
        )

    def label_path(self, patch_id: str) -> Path:
        return (
            self.data_root
            / "MASK"
            / f"{patch_id}{DEFAULT_LABEL_PATTERN}"
        )

    def available_samples(self) -> list[str]:
        """
        Return sample IDs discovered from Sentinel-2 files.
        """
        s2_dir = self.data_root / "S2L2A"

        if not s2_dir.exists():
            raise FileNotFoundError(
                f"Sentinel-2 directory does not exist: {s2_dir}"
            )

        suffix = DEFAULT_PATTERNS["S2L2A"]

        samples = [
            path.name.removesuffix(suffix)
            for path in s2_dir.glob(f"*{suffix}")
        ]

        return sorted(samples)

    def read_zarr(
        self,
        path: str | Path,
    ) -> np.ndarray:
        """
        Read the ImpactMesh Zarr 'bands' array.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        data = zarr.open_consolidated(path, mode="r")

        if "bands" not in data:
            raise KeyError(
                f"'bands' array not found in {path}"
            )

        return np.asarray(data["bands"][...])

    def read_dem(
        self,
        path: str | Path,
    ) -> np.ndarray:
        """
        Read a DEM GeoTIFF.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)

    def read_label(
        self,
        patch_id: str,
    ) -> np.ndarray:
        """
        Read the binary flood annotation mask.
        """
        path = self.label_path(patch_id)

        if not path.exists():
            raise FileNotFoundError(path)

        with rasterio.open(path) as src:
            return src.read(1)

    def read_sample(
        self,
        patch_id: str,
    ) -> dict[str, Any]:
        """
        Read all available ImpactMesh modalities for one sample.
        """
        output: dict[str, Any] = {
            "patch_id": patch_id,
        }

        for modality in self.modalities:
            path = self.sample_path(patch_id, modality)

            if modality in {"S2L2A", "S1RTC"}:
                output[modality] = self.read_zarr(path)

            elif modality == "DEM":
                output[modality] = self.read_dem(path)

        label = self.label_path(patch_id)

        if label.exists():
            output["mask"] = self.read_label(patch_id)

        return output

    @staticmethod
    def flood_ratio(mask: np.ndarray) -> float:
        """
        Calculate the proportion of pixels marked as flooded.
        """
        valid = mask >= 0

        if not np.any(valid):
            return 0.0

        flooded = mask > 0

        return float(
            np.sum(flooded & valid) / np.sum(valid)
        )