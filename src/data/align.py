from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FEATURES = {
    "satellite": [
        "sat_ndvi", "sat_ndwi", "sat_water_fraction", "sat_flood_probability",
        "sat_b02", "sat_b03", "sat_b04", "sat_b08"
    ],
    "iot": [
        "water_level", "water_level_change", "rainfall", "rainfall_rate",
        "soil_moisture", "temperature", "humidity", "pressure"
    ],
    "gis": [
        "elevation", "slope", "river_distance", "road_density",
        "building_density", "population_density", "land_use_urban", "land_use_water"
    ],
    "social": [
        "post_count", "urgent_post_count", "disaster_probability",
        "negative_sentiment", "help_request_probability",
        "verified_report_fraction", "location_confidence", "text_disaster_score"
    ],
}

def build_fused_dataset():
    keys = ["event_id", "timestamp", "latitude", "longitude", "h3_cell"]
    result = None

    for modality, cols in FEATURES.items():
        df = pd.read_csv(
            ROOT / "data" / "raw" / f"{modality}.csv",
            parse_dates=["timestamp"]
        )
        selected = df[keys + cols].copy()

        if result is None:
            result = selected
        else:
            result = result.merge(
                selected,
                on=keys,
                how="inner",
                validate="one_to_one"
            )

    labels = pd.read_csv(ROOT / "data" / "raw" / "labels.csv")
    result = result.merge(
        labels, on="event_id", how="inner", validate="one_to_one"
    )

    out = ROOT / "data" / "processed" / "fused_dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(f"Aligned samples: {len(result)}")
    print(f"Wrote: {out}")
    return result
