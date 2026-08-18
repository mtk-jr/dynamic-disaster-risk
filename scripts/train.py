from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.models.fusion_model import EarlyFusionModel
from src.data.align import FEATURES
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]
cfg = load_config()
df = pd.read_csv(ROOT / "data" / "processed" / "fused_dataset.csv")

arrays, scalers = {}, {}
for modality, cols in FEATURES.items():
    scaler = StandardScaler()
    arrays[modality] = scaler.fit_transform(df[cols].values)
    scalers[modality] = scaler

y = df["risk_label"].astype(int).values
idx = np.arange(len(df))

train_idx, test_idx = train_test_split(
    idx, test_size=cfg["training"]["test_size"],
    random_state=42, stratify=y
)
train_idx, val_idx = train_test_split(
    train_idx, test_size=cfg["training"]["validation_size"],
    random_state=42, stratify=y[train_idx]
)

def make_loader(indices, shuffle):
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(
        torch.tensor(arrays["satellite"][indices], dtype=torch.float32),
        torch.tensor(arrays["iot"][indices], dtype=torch.float32),
        torch.tensor(arrays["gis"][indices], dtype=torch.float32),
        torch.tensor(arrays["social"][indices], dtype=torch.float32),
        torch.tensor(y[indices], dtype=torch.long),
    )
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=shuffle)

train_loader = make_loader(train_idx, True)
val_loader = make_loader(val_idx, False)

model = EarlyFusionModel(**{
    "satellite_dim": 8,
    "iot_dim": 8,
    "gis_dim": 8,
    "social_dim": 8,
    "hidden_dim": cfg["model"]["hidden_dim"],
    "fusion_dim": cfg["model"]["fusion_dim"],
    "num_classes": cfg["model"]["num_classes"],
    "dropout": cfg["model"]["dropout"],
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=cfg["training"]["learning_rate"]
)

best_val = float("inf")
artifact_dir = ROOT / "artifacts"
artifact_dir.mkdir(exist_ok=True)

for epoch in range(cfg["training"]["epochs"]):
    model.train()
    train_loss = 0.0

    for sat, iot, gis, social, labels in train_loader:
        sat, iot, gis, social, labels = [
            x.to(device) for x in (sat, iot, gis, social, labels)
        ]
        optimizer.zero_grad()
        loss = criterion(model(sat, iot, gis, social), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    yt, yp = [], []

    with torch.no_grad():
        for sat, iot, gis, social, labels in val_loader:
            sat, iot, gis, social, labels = [
                x.to(device) for x in (sat, iot, gis, social, labels)
            ]
            logits = model(sat, iot, gis, social)
            val_loss += criterion(logits, labels).item()
            yt.extend(labels.cpu().numpy())
            yp.extend(logits.argmax(1).cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(yt, yp)

    print(
        f"Epoch {epoch+1:03d} | "
        f"train_loss={train_loss/len(train_loader):.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {"model_state_dict": model.state_dict(), "config": cfg},
            artifact_dir / "model.pt"
        )

for modality, scaler in scalers.items():
    joblib.dump(scaler, artifact_dir / f"{modality}_scaler.joblib")

print("Saved model and scalers.")
