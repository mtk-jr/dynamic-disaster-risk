from __future__ import annotations

import csv
import tarfile
import tempfile
from pathlib import Path

import h3
import rasterio
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "impactmesh_download"
)

INDEX_FILE = (
    ROOT
    / "data"
    / "processed"
    / "impactmesh_val_index.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "impactmesh_metadata.csv"
)

DEM_TAR = DATA_ROOT / "val" / "DEM.tar"

# H3 resolution.
# Resolution 8 gives relatively small spatial cells
# suitable for local disaster-risk modelling.
H3_RESOLUTION = 8


def extract_dem(
    archive: tarfile.TarFile,
    member_name: str,
) -> Path:

    member = archive.getmember(member_name)

    extracted = archive.extractfile(member)

    if extracted is None:
        raise RuntimeError(
            f"Could not extract {member_name}"
        )

    temp = tempfile.NamedTemporaryFile(
        suffix=".tif",
        delete=False,
    )

    temp_path = Path(temp.name)

    try:
        temp.write(extracted.read())
    finally:
        temp.close()

    return temp_path


def get_sample_location(
    dem_path: Path,
) -> tuple[float, float, float, float]:

    with rasterio.open(dem_path) as src:

        bounds = src.bounds

        # Center of the raster in the native CRS.
        x = (bounds.left + bounds.right) / 2.0
        y = (bounds.bottom + bounds.top) / 2.0

        source_crs = src.crs

        if source_crs is None:
            raise ValueError(
                f"DEM has no CRS: {dem_path}"
            )

        # Convert projected coordinates to WGS84.
        transformer = Transformer.from_crs(
            source_crs,
            "EPSG:4326",
            always_xy=True,
        )

        longitude, latitude = transformer.transform(
            x,
            y,
        )

        return (
            float(x),
            float(y),
            float(latitude),
            float(longitude),
        )


def main() -> None:

    print("Loading ImpactMesh index...")

    with INDEX_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    print(f"Samples found: {len(rows)}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_id",
        "event_id",
        "utm_x",
        "utm_y",
        "latitude",
        "longitude",
        "h3_cell",
        "flood_ratio",
    ]

    with tarfile.open(
        DEM_TAR,
        "r",
    ) as dem_archive:

        with OUTPUT_FILE.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for index, row in enumerate(
                rows,
                start=1,
            ):

                dem_path = None

                try:

                    dem_path = extract_dem(
                        dem_archive,
                        row["dem_member"],
                    )

                    (
                        utm_x,
                        utm_y,
                        latitude,
                        longitude,
                    ) = get_sample_location(
                        dem_path
                    )

                    h3_cell = h3.latlng_to_cell(
                        latitude,
                        longitude,
                        H3_RESOLUTION,
                    )

                    # Event identifier.
                    #
                    # ImpactMesh sample IDs contain the
                    # event prefix before the tile coordinates.
                    #
                    # Example:
                    # EMSR264_16_38KMD_x493425_y7815805
                    #
                    # We retain the complete sample ID as
                    # the unique sample identifier and use
                    # the EMSR portion as the event identifier.
                    sample_id = row["sample_id"]

                    event_id = sample_id.split(
                        "_x",
                        1,
                    )[0]

                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "event_id": event_id,
                            "utm_x": utm_x,
                            "utm_y": utm_y,
                            "latitude": latitude,
                            "longitude": longitude,
                            "h3_cell": h3_cell,
                            "flood_ratio": float(
                                row["flood_ratio"]
                            ),
                        }
                    )

                    if index % 100 == 0:
                        print(
                            f"Processed "
                            f"{index}/{len(rows)}"
                        )

                finally:

                    if dem_path is not None:

                        try:
                            dem_path.unlink(
                                missing_ok=True
                            )
                        except PermissionError:
                            pass

    print()
    print("ImpactMesh metadata created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()