from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Dynamic Disaster Risk", layout="wide")
st.title("Dynamic Disaster Risk Assessment")
st.caption("Remote Sensing + IoT + GIS + Social Media — Early Fusion")

metrics_path = ROOT / "artifacts" / "metrics.json"

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    b.metric("Weighted F1", f"{metrics['f1_weighted']:.3f}")
    c.metric("Weighted Recall", f"{metrics['recall_weighted']:.3f}")
else:
    st.warning("Run: python scripts/run_pipeline.py")

data_path = ROOT / "data" / "processed" / "fused_dataset.csv"
if data_path.exists():
    df = pd.read_csv(data_path)
    st.subheader("Risk distribution")
    st.bar_chart(df["risk_label"].value_counts().sort_index())
    st.subheader("Aligned sample locations")
    st.map(df[["latitude", "longitude"]].rename(
        columns={"latitude": "lat", "longitude": "lon"}
    ).head(500))
