from pathlib import Path
import joblib
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.models.fusion_model import EarlyFusionModel

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="Dynamic Disaster Risk API", version="0.1.0")

MODEL = None
SCALERS = {}

class PredictionRequest(BaseModel):
    satellite: list[float] = Field(min_length=8, max_length=8)
    iot: list[float] = Field(min_length=8, max_length=8)
    gis: list[float] = Field(min_length=8, max_length=8)
    social: list[float] = Field(min_length=8, max_length=8)

@app.on_event("startup")
def load_model():
    global MODEL, SCALERS

    MODEL = EarlyFusionModel()
    checkpoint_path = ROOT / "artifacts" / "model.pt"

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        MODEL.eval()

    for name in ["satellite", "iot", "gis", "social"]:
        p = ROOT / "artifacts" / f"{name}_scaler.joblib"
        if p.exists():
            SCALERS[name] = joblib.load(p)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None and len(SCALERS) == 4}

@app.post("/predict")
def predict(req: PredictionRequest):
    if MODEL is None or len(SCALERS) != 4:
        return {"error": "Run training first."}

    raw = {
        "satellite": req.satellite,
        "iot": req.iot,
        "gis": req.gis,
        "social": req.social,
    }

    scaled = {
        name: SCALERS[name].transform(
            np.array(values, dtype=float).reshape(1, -1)
        )
        for name, values in raw.items()
    }

    with torch.no_grad():
        logits = MODEL(
            torch.tensor(scaled["satellite"], dtype=torch.float32),
            torch.tensor(scaled["iot"], dtype=torch.float32),
            torch.tensor(scaled["gis"], dtype=torch.float32),
            torch.tensor(scaled["social"], dtype=torch.float32),
        )
        probs = torch.softmax(logits, dim=1)[0].numpy()

    idx = int(np.argmax(probs))
    names = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    return {
        "risk_class": idx,
        "risk_level": names[idx],
        "risk_score": float(sum(i * float(p) for i, p in enumerate(probs)) / 3),
        "probabilities": probs.tolist(),
    }
