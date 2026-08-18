from pathlib import Path
import numpy as np
import pandas as pd
import h3

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)
N = 2000

event_id = [f"event_{i:05d}" for i in range(N)]
lat = rng.uniform(8.3, 12.8, N)
lon = rng.uniform(74.8, 77.3, N)
h3_cells = [h3.latlng_to_cell(float(a), float(b), 7) for a, b in zip(lat, lon)]

base = rng.beta(2, 4, N)
rain = np.clip(20 + 150 * base + rng.normal(0, 10, N), 0, None)
water = np.clip(20 + 80 * base + rng.normal(0, 5, N), 0, None)
soil = np.clip(30 + 60 * base + rng.normal(0, 4, N), 0, 100)
elevation = np.clip(80 - 55 * base + rng.normal(0, 10, N), 1, None)
slope = np.clip(12 - 5 * base + rng.normal(0, 2, N), 0, None)
social = np.clip(1 + 20 * base + rng.normal(0, 3, N), 0, None)
flood_area = np.clip(100 * base + rng.normal(0, 5, N), 0, None)

risk_score = np.clip(
    0.30 * (rain / 180)
    + 0.25 * (water / 100)
    + 0.15 * (soil / 100)
    + 0.15 * (1 - elevation / 100)
    + 0.10 * (social / 20)
    + 0.05 * (flood_area / 100),
    0, 1
)

risk_label = pd.cut(
    risk_score,
    bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
    labels=[0, 1, 2, 3],
).astype(int)

timestamps = pd.date_range("2026-01-01", periods=N, freq="h")

common = {
    "event_id": event_id,
    "timestamp": timestamps,
    "latitude": lat,
    "longitude": lon,
    "h3_cell": h3_cells,
}

satellite = pd.DataFrame({
    **common,
    "sat_ndvi": rng.normal(0.4 - 0.15 * base, 0.05),
    "sat_ndwi": rng.normal(0.1 + 0.4 * base, 0.05),
    "sat_water_fraction": np.clip(flood_area / 100 + rng.normal(0, .03, N), 0, 1),
    "sat_flood_probability": np.clip(base + rng.normal(0, .05, N), 0, 1),
    "sat_b02": rng.normal(.15 + .02 * base, .03, N),
    "sat_b03": rng.normal(.18 + .03 * base, .03, N),
    "sat_b04": rng.normal(.16 + .02 * base, .03, N),
    "sat_b08": rng.normal(.30 - .03 * base, .04, N),
})

iot = pd.DataFrame({
    **common,
    "water_level": water,
    "water_level_change": np.gradient(water),
    "rainfall": rain,
    "rainfall_rate": np.gradient(rain),
    "soil_moisture": soil,
    "temperature": rng.normal(29, 2, N),
    "humidity": np.clip(60 + 30 * base + rng.normal(0, 4, N), 0, 100),
    "pressure": rng.normal(1008 - 8 * base, 2, N),
})

gis = pd.DataFrame({
    **common,
    "elevation": elevation,
    "slope": slope,
    "river_distance": np.clip(5000 - 3500 * base + rng.normal(0, 400, N), 20, None),
    "road_density": np.clip(0.2 + 1.5 * base + rng.normal(0, .2, N), 0, None),
    "building_density": np.clip(0.2 + 1.5 * base + rng.normal(0, .2, N), 0, None),
    "population_density": np.clip(100 + 1000 * base + rng.normal(0, 100, N), 0, None),
    "land_use_urban": np.clip(.2 + .5 * base + rng.normal(0, .05, N), 0, 1),
    "land_use_water": np.clip(.1 + .6 * base + rng.normal(0, .05, N), 0, 1),
})

social_df = pd.DataFrame({
    **common,
    "post_count": social,
    "urgent_post_count": np.clip(social * .4 + rng.normal(0, 1, N), 0, None),
    "disaster_probability": np.clip(base + rng.normal(0, .07, N), 0, 1),
    "negative_sentiment": np.clip(.3 + .5 * base + rng.normal(0, .08, N), 0, 1),
    "help_request_probability": np.clip(.1 + .6 * base + rng.normal(0, .08, N), 0, 1),
    "verified_report_fraction": np.clip(.2 + .2 * base + rng.normal(0, .05, N), 0, 1),
    "location_confidence": np.clip(.5 + .4 * base + rng.normal(0, .05, N), 0, 1),
    "text_disaster_score": np.clip(base + rng.normal(0, .05, N), 0, 1),
})

labels = pd.DataFrame({
    "event_id": event_id,
    "risk_score": risk_score,
    "risk_label": risk_label,
})

for name, df in {
    "satellite": satellite,
    "iot": iot,
    "gis": gis,
    "social": social_df,
    "labels": labels,
}.items():
    path = ROOT / "data" / "raw" / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path}")
