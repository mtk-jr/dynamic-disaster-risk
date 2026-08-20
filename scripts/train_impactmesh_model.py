from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data.dataset import create_splits
from src.models.early_fusion import EarlyFusionModel


SEED = 42

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 15

CHECKPOINT_DIR = Path(
    "models/checkpoints"
)

BEST_MODEL = (
    CHECKPOINT_DIR
    / "impactmesh_early_fusion_best.pt"
)


def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_batches(
    remote,
    gis,
    target,
    indices,
    batch_size,
    shuffle=False,
):

    indices = np.asarray(indices)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        yield (
            torch.tensor(
                remote[batch_indices],
                dtype=torch.float32,
            ),
            torch.tensor(
                gis[batch_indices],
                dtype=torch.float32,
            ),
            torch.tensor(
                target[batch_indices],
                dtype=torch.float32,
            ),
        )


def evaluate(
    model,
    remote,
    gis,
    target,
    indices,
):

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for (
            remote_batch,
            gis_batch,
            target_batch,
        ) in create_batches(
            remote,
            gis,
            target,
            indices,
            BATCH_SIZE,
            shuffle=False,
        ):

            output = model(
                remote_batch,
                gis_batch,
            )

            predictions.extend(
                output.cpu().numpy()
            )

            actual.extend(
                target_batch.cpu().numpy()
            )

    predictions = np.asarray(
        predictions
    )

    actual = np.asarray(
        actual
    )

    mse = mean_squared_error(
        actual,
        predictions,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        actual,
        predictions,
    )

    r2 = r2_score(
        actual,
        predictions,
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    print()
    print("Loading dataset...")

    data = create_splits(
        random_state=SEED
    )

    remote = data["remote"]
    gis = data["gis"]
    target = data["target"]

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    print(
        "Training samples:",
        len(train_idx),
    )

    print(
        "Validation samples:",
        len(val_idx),
    )

    print(
        "Test samples:",
        len(test_idx),
    )

    model = EarlyFusionModel()

    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    print()
    print("Starting training...")
    print()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        train_losses = []

        for (
            remote_batch,
            gis_batch,
            target_batch,
        ) in create_batches(
            remote,
            gis,
            target,
            train_idx,
            BATCH_SIZE,
            shuffle=True,
        ):

            remote_batch = remote_batch.to(
                device
            )

            gis_batch = gis_batch.to(
                device
            )

            target_batch = target_batch.to(
                device
            )

            optimizer.zero_grad()

            predictions = model(
                remote_batch,
                gis_batch,
            )

            loss = criterion(
                predictions,
                target_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(train_losses)
        )

        # Validation
        model.eval()

        val_losses = []

        with torch.no_grad():

            for (
                remote_batch,
                gis_batch,
                target_batch,
            ) in create_batches(
                remote,
                gis,
                target,
                val_idx,
                BATCH_SIZE,
                shuffle=False,
            ):

                remote_batch = remote_batch.to(
                    device
                )

                gis_batch = gis_batch.to(
                    device
                )

                target_batch = target_batch.to(
                    device
                )

                predictions = model(
                    remote_batch,
                    gis_batch,
                )

                loss = criterion(
                    predictions,
                    target_batch,
                )

                val_losses.append(
                    loss.item()
                )

        val_loss = float(
            np.mean(val_losses)
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "val_loss":
                        val_loss,
                },
                BEST_MODEL,
            )

            print(
                "  ✓ Best model saved"
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping."
            )

            break

    # ------------------------------------------------
    # Load best model
    # ------------------------------------------------

    print()
    print(
        "Loading best checkpoint..."
    )

    checkpoint = torch.load(
        BEST_MODEL,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------

    metrics = evaluate(
        model,
        remote,
        gis,
        target,
        test_idx,
    )

    print()
    print("==============================")
    print("FINAL TEST RESULTS")
    print("==============================")

    print(
        f"MSE  : {metrics['mse']:.6f}"
    )

    print(
        f"RMSE : {metrics['rmse']:.6f}"
    )

    print(
        f"MAE  : {metrics['mae']:.6f}"
    )

    print(
        f"R²   : {metrics['r2']:.6f}"
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(BEST_MODEL)


if __name__ == "__main__":
    main()