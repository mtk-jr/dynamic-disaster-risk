import pandas as pd

def load_iot_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = ["event_id", "timestamp", "latitude", "longitude", "water_level", "rainfall"]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"Missing IoT columns: {missing}")
    return df
