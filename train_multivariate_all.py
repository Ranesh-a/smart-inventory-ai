"""
🚀 Multi-Category Multivariate LSTM Training Script
Trains separate multivariate LSTM models for Hobbies and Household.

Reuses the same architecture and hyperparameters as the existing Food model
(train_multivariate.py) but targets the new evaluation datasets.

Outputs:
    models/model_hobbies_multivariate.pth   + scalers/hobbies_scaler.pkl
    models/model_household_multivariate.pth + scalers/household_scaler.pkl
"""

import sys
import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import LSTMModel
from src.dataset import (
    load_dataset,
    generate_sequences_per_item,
    MultivariateSlidingWindowDataset,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
)

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.80

INPUT_SIZE = len(FEATURE_COLUMNS)  # 6
HIDDEN_SIZE = 50
NUM_LAYERS = 1
OUTPUT_SIZE = 1

# Max items to use per category (None = all). Reduce if OOM.
MAX_ITEMS = 200

CATEGORIES = [
    {
        "name": "hobbies",
        "csv": "hobbies_evaluation_dataset.csv",
        "model_path": "models/model_hobbies_multivariate.pth",
        "scaler_path": "scalers/hobbies_scaler.pkl",
        "history_path": "models/training_history_hobbies_mv.json",
    },
    {
        "name": "household",
        "csv": "household_evaluation_dataset.csv",
        "model_path": "models/model_household_multivariate.pth",
        "scaler_path": "scalers/household_scaler.pkl",
        "history_path": "models/training_history_household_mv.json",
    },
]


# ─────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────

def train_category(cfg: dict, device: torch.device):
    """Train a multivariate LSTM for a single category."""
    name = cfg["name"]
    csv_path = cfg["csv"]

    print(f"\n{'='*60}")
    print(f"  CATEGORY: {name.upper()}")
    print(f"{'='*60}")

    start = time.time()

    # ── Load dataset ─────────────────────────────────────────
    if not os.path.exists(csv_path):
        print(f"  ⚠️  {csv_path} not found — SKIPPING.")
        return None

    df = load_dataset(csv_path)
    total_items = df["item_id"].nunique()

    if MAX_ITEMS and MAX_ITEMS < total_items:
        sample_items = sorted(df["item_id"].unique()[:MAX_ITEMS])
        df = df[df["item_id"].isin(sample_items)].copy()
        print(f"  📋 Sampled {MAX_ITEMS}/{total_items} items")
    else:
        print(f"  📋 Using all {total_items} items")

    # ── Fit & save scaler on sales column only ───────────────
    os.makedirs("scalers", exist_ok=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales_vals = df["sales"].values.reshape(-1, 1)
    scaler.fit(sales_vals)
    joblib.dump(scaler, cfg["scaler_path"])
    print(f"  💾 Scaler saved → {cfg['scaler_path']}")
    print(f"  Sales range: [{sales_vals.min():.0f}, {sales_vals.max():.0f}]")

    # ── Generate sequences ───────────────────────────────────
    X, y = generate_sequences_per_item(df, sequence_length=SEQUENCE_LENGTH)
    dataset = MultivariateSlidingWindowDataset(X, y)
    del X, y, df
    gc.collect()

    # ── Train/Val split ──────────────────────────────────────
    n_total = len(dataset)
    n_train = int(n_total * TRAIN_RATIO)
    train_set = Subset(dataset, list(range(n_train)))
    val_set = Subset(dataset, list(range(n_train, n_total)))

    print(f"  Total: {n_total:,}  Train: {n_train:,}  Val: {n_total - n_train:,}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ────────────────────────────────────────────────
    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: LSTM({INPUT_SIZE}, {HIDDEN_SIZE}) → FC({HIDDEN_SIZE}, {OUTPUT_SIZE})")
    print(f"  Parameters:   {total_params:,}")

    # ── Training loop ────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            if torch.isnan(loss):
                print(f"  ❌ NaN at epoch {epoch}!")
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        history["train_loss"].append(float(avg_train))
        history["val_loss"].append(float(avg_val))

        marker = ""
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            marker = " ⭐ best"

            os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "output_size": OUTPUT_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "feature_columns": FEATURE_COLUMNS,
            }, cfg["model_path"])

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{EPOCHS} │ "
            f"Train: {avg_train:.6f} │ Val: {avg_val:.6f} │ "
            f"{elapsed:.1f}s{marker}"
        )

    # Save history
    with open(cfg["history_path"], "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start
    print(f"\n  ✅ {name.upper()} complete in {total_time / 60:.1f} min")
    print(f"     Best epoch: {best_epoch}  |  Best val loss: {best_val:.6f}")
    print(f"     Model: {cfg['model_path']}")
    print(f"     Scaler: {cfg['scaler_path']}")

    # Cleanup
    del model, dataset, train_set, val_set, train_loader, val_loader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return {"category": name, "best_val_loss": best_val, "best_epoch": best_epoch}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Multi-Category Multivariate LSTM Training")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    results = []
    for cfg in CATEGORIES:
        r = train_category(cfg, device)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  ✅ {r['category'].upper():12s} | Best epoch: {r['best_epoch']} | Val loss: {r['best_val_loss']:.6f}")
    print()


if __name__ == "__main__":
    main()
