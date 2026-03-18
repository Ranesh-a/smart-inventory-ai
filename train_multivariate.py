"""
🚀 Multivariate LSTM Training Script
Trains the LSTM model on food_evaluation_dataset.csv with 6 input features.

Config:
    Sequence Length  = 30
    Batch Size       = 64
    Epochs           = 20
    Optimizer        = Adam (lr=0.001)
    Loss             = MSELoss
    Split            = 80/20 sequential (time-series safe)
    GPU              = auto-detect

Output:
    models/model_food_multivariate.pth
"""

import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

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

CSV_PATH = "food_evaluation_dataset.csv"
CHECKPOINT_PATH = "models/model_food_multivariate.pth"
HISTORY_PATH = "models/training_history_multivariate.json"

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.80

INPUT_SIZE = len(FEATURE_COLUMNS)  # 6
HIDDEN_SIZE = 50
NUM_LAYERS = 1
OUTPUT_SIZE = 1

# Max items to use (None = all). Reduce if OOM.
MAX_ITEMS = 200


def main():
    start_time = time.time()

    # ── Device ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"🟢 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("🟡 Training on CPU")

    # ── Step 1: Load & Prepare Dataset ──────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 1: Loading Dataset")
    print(f"{'='*60}")
    df = load_dataset(CSV_PATH)
    total_items = df["item_id"].nunique()

    if MAX_ITEMS and MAX_ITEMS < total_items:
        sample_items = sorted(df["item_id"].unique()[:MAX_ITEMS])
        df = df[df["item_id"].isin(sample_items)].copy()
        print(f"📋 Sampled {MAX_ITEMS}/{total_items} items for training")
    else:
        print(f"📋 Using all {total_items} items")

    # ── Step 2: Generate Sequences ──────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 2: Generating Sliding Window Sequences")
    print(f"{'='*60}")
    X, y = generate_sequences_per_item(df, sequence_length=SEQUENCE_LENGTH)
    dataset = MultivariateSlidingWindowDataset(X, y)

    # Free raw arrays
    del X, y, df

    # ── Step 3: Sequential Train/Val Split ──────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 3: Sequential Train/Validation Split")
    print(f"{'='*60}")
    n_total = len(dataset)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = n_total - n_train

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(f"  Total sequences: {n_total:,}")
    print(f"  Train: {n_train:,} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {n_val:,} ({(1-TRAIN_RATIO)*100:.0f}%)")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # ── Step 4: Model, Optimizer, Loss ──────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 4: Model Setup")
    print(f"{'='*60}")
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
    print(f"  Optimizer:    Adam (lr={LEARNING_RATE})")
    print(f"  Loss:         MSELoss")

    # ── Step 5: Training Loop ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 5: Training ({EPOCHS} Epochs)")
    print(f"{'='*60}")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ───────────────────────────────────────────
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)

            # Check for NaN/exploding gradients
            if torch.isnan(loss):
                print(f"\n❌ NaN loss detected at epoch {epoch}! Stopping.")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ── Validate ────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        elapsed = time.time() - epoch_start
        marker = ""

        # ── Save best checkpoint ────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            marker = " ⭐ best"

            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "output_size": OUTPUT_SIZE,
                    "sequence_length": SEQUENCE_LENGTH,
                    "feature_columns": FEATURE_COLUMNS,
                },
                CHECKPOINT_PATH,
            )

        print(
            f"  Epoch {epoch:02d}/{EPOCHS} │ "
            f"Train Loss: {avg_train_loss:.6f} │ "
            f"Val Loss: {avg_val_loss:.6f} │ "
            f"{elapsed:.1f}s{marker}"
        )

    # ── Step 6: Summary ─────────────────────────────────────
    total_time = time.time() - start_time

    # Save training history
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:       {total_time/60:.1f} min")
    print(f"  Best epoch:       {best_epoch}")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final val loss:   {history['val_loss'][-1]:.6f}")
    print(f"  Best val loss:    {best_val_loss:.6f}")
    print(f"  Checkpoint:       {CHECKPOINT_PATH}")
    print(f"  History:          {HISTORY_PATH}")

    # Trend check
    if history["train_loss"][-1] < history["train_loss"][0]:
        print(f"\n  ✅ Loss decreased: {history['train_loss'][0]:.6f} → {history['train_loss'][-1]:.6f}")
    else:
        print(f"\n  ⚠️  Loss did NOT decrease — may need tuning")

    print()


if __name__ == "__main__":
    main()
