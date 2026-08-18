from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.models.fusion_model import EarlyFusionModel
from src.data.align import FEATURES
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]
cfg = load_config()
df = pd.read_csv(ROOT / "data" / "processed" / "fused_dataset.csv")

idx = np.arange(len(df))
_, test_idx = train_test_split(
    idx, test_size=cfg["training"]["test_size"],
    random_state=42, stratify=df["risk_label"]
)

arrays = {}
for modality, cols in FEATURES.items():
    scaler = joblib.load(ROOT / "artifacts" / f"{modality}_scaler.joblib")
    arrays[modality] = scaler.transform(df[cols].values)

model = EarlyFusionModel(
    satellite_dim=8, iot_dim=8, gis_dim=8, social_dim=8,
    hidden_dim=cfg["model"]["hidden_dim"],
    fusion_dim=cfg["model"]["fusion_dim"],
    num_classes=4,
    dropout=cfg["model"]["dropout"],
)
checkpoint = torch.load(ROOT / "artifacts" / "model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    logits = model(
        torch.tensor(arrays["satellite"][test_idx], dtype=torch.float32),
        torch.tensor(arrays["iot"][test_idx], dtype=torch.float32),
        torch.tensor(arrays["gis"][test_idx], dtype=torch.float32),
        torch.tensor(arrays["social"][test_idx], dtype=torch.float32),
    )
    pred = logits.argmax(1).numpy()

y = df["risk_label"].values[test_idx]
precision, recall, f1, _ = precision_recall_fscore_support(
    y, pred, average="weighted", zero_division=0
)

metrics = {
    "accuracy": float(accuracy_score(y, pred)),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "f1_weighted": float(f1),
    "confusion_matrix": confusion_matrix(y, pred).tolist(),
}

print(json.dumps(metrics, indent=2))
(ROOT / "artifacts" / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
