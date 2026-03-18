"""
Training script for Smart Inventory Demand Forecasting LSTM model.
Trains separate models for Food, Hobby, and Household categories.
Supports both long-format and wide-format CSV files.
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SlidingWindowDataset, create_train_test_split
from src.model import LSTMModel, get_device


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 50
NUM_LAYERS = 1

DATASETS = [
    {"name": "food",      "file": "project_data.csv",      "target_id": "FOODS_3_002"},
    {"name": "hobby",     "file": "subset_hobbies.csv",    "target_id": None},
    {"name": "household", "file": "subset_household.csv",  "target_id": None},
]


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def load_csv_to_long(file_path: str) -> pd.DataFrame:
    """
    Load a CSV and return it in long format with columns: [item_id, d, sales].
    Handles both long-format (food) and wide-format (hobby/household) CSVs.
    """
    df = pd.read_csv(file_path)

    if 'sales' in df.columns and 'd' in df.columns:
        # Already in long format (e.g., project_data.csv)
        print(f"  Format: LONG  |  Shape: {df.shape}")
        return df[['item_id', 'd', 'sales']].copy()

    # Wide format: columns like d_1, d_2, ..., d_1941
    day_cols = [c for c in df.columns if c.startswith('d_')]
    if not day_cols:
        raise ValueError(f"Cannot detect format for {file_path}. No 'sales' or 'd_*' columns found.")

    print(f"  Format: WIDE  |  Shape: {df.shape}  |  Day columns: {len(day_cols)}")

    id_cols = [c for c in df.columns if not c.startswith('d_')]
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name='d',
        value_name='sales',
    )
    # Convert d_123 → 123 (integer) for proper sorting
    long_df['d_num'] = long_df['d'].str.replace('d_', '', regex=False).astype(int)
    long_df = long_df.sort_values(['item_id', 'd_num']).reset_index(drop=True)

    return long_df[['item_id', 'd', 'sales', 'd_num']].copy()


def select_item(df: pd.DataFrame, target_id) -> tuple:
    """
    Select the target item from the dataframe.
    If target_id is None, pick the item with the highest total sales.
    Returns (item_id, item_sales_array).
    """
    if target_id is not None:
        item_df = df[df['item_id'] == target_id]
        if item_df.empty:
            raise ValueError(f"Item '{target_id}' not found in the dataset.")
        item_id = target_id
    else:
        # Auto-select: item with highest total sales
        totals = df.groupby('item_id')['sales'].sum()
        item_id = totals.idxmax()
        item_df = df[df['item_id'] == item_id]
        print(f"  Auto-selected item: {item_id}  (total sales: {totals[item_id]:.0f})")

    # Sort by day and extract sales
    if 'd_num' in item_df.columns:
        sales = item_df.sort_values('d_num')['sales'].values.astype(float)
    else:
        item_df_sorted = item_df.copy()
        # Handle both "d_1" style strings and plain integers
        item_df_sorted['d_num'] = (
            item_df_sorted['d']
            .astype(str)
            .str.replace('d_', '', regex=False)
            .astype(int)
        )
        sales = item_df_sorted.sort_values('d_num')['sales'].values.astype(float)

    return item_id, sales


# ─────────────────────────────────────────────────────────────
# Training Function (preserved from original)
# ─────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    learning_rate: float = 0.001,
    model_save_path: str = "models/lstm_model.pth"
):
    """
    Train the LSTM model.

    Args:
        model: The LSTMModel to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test/validation data.
        device: Device to train on (cuda/cpu).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        model_save_path: Path to save the best model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')

    print(f"\n{'─'*50}")
    print(f"  Device: {device}  |  Epochs: {epochs}  |  LR: {learning_rate}")
    print(f"  Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")
    print(f"{'─'*50}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation phase
        model.eval()
        epoch_test_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_test_loss += loss.item()

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss

            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }, model_save_path)

            print(f"  Epoch [{epoch+1:2d}/{epochs}] | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Test: {avg_test_loss:.6f} | "
                  f"✓ Best!")
        else:
            print(f"  Epoch [{epoch+1:2d}/{epochs}] | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Test: {avg_test_loss:.6f}")

    print(f"\n  ✅ Best Test Loss: {best_test_loss:.6f}")
    print(f"  📁 Model saved to: {model_save_path}\n")

    return train_losses, test_losses


def plot_training_history(
    train_losses: list,
    test_losses: list,
    title: str = "Training History",
    save_path: str = "training_history.png",
):
    """Plot and save training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(test_losses, label='Test Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Training plot saved to: {save_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    """Train one LSTM model per dataset category."""

    print(f"\n{'='*60}")
    print(f"  Smart Inventory — Multi-Category Model Training")
    print(f"{'='*60}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Categories: {[d['name'] for d in DATASETS]}\n")

    device = get_device()
    os.makedirs("models", exist_ok=True)

    results = []

    for idx, config in enumerate(DATASETS, 1):
        name = config["name"]
        file_path = config["file"]
        target_id = config["target_id"]

        model_path = f"models/model_{name}.pth"
        scaler_path = f"models/scaler_{name}.pkl"
        plot_path = f"models/training_history_{name}.png"

        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(DATASETS)}] Category: {name.upper()}")
        print(f"{'='*60}")

        # ── Step 1: Load CSV ────────────────────────────────
        if not os.path.exists(file_path):
            print(f"  ⚠️  File not found: {file_path} — SKIPPING.\n")
            continue

        try:
            print(f"  Loading: {file_path}")
            df = load_csv_to_long(file_path)
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e} — SKIPPING.\n")
            continue

        # ── Step 2: Select item ─────────────────────────────
        try:
            item_id, sales = select_item(df, target_id)
        except Exception as e:
            print(f"  ❌ Item selection failed: {e} — SKIPPING.\n")
            continue

        print(f"  Training {name.upper()} model for Item: {item_id}")
        print(f"  Sales history length: {len(sales)} days")

        if len(sales) < SEQUENCE_LENGTH + 2:
            print(f"  ⚠️  Not enough data ({len(sales)} < {SEQUENCE_LENGTH + 2}) — SKIPPING.\n")
            continue

        # ── Step 3: Normalize with a fresh scaler ───────────
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_sales = scaler.fit_transform(sales.reshape(-1, 1)).flatten()
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved to: {scaler_path}")
        print(f"  Sales range: [{sales.min():.0f}, {sales.max():.0f}]")

        # ── Step 4: Create datasets ─────────────────────────
        dataset = SlidingWindowDataset(
            sales_data=normalized_sales,
            sequence_length=SEQUENCE_LENGTH,
            prediction_horizon=1,
        )
        train_set, test_set = create_train_test_split(dataset, train_ratio=0.8)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

        # ── Step 5: Fresh model (no weight bleed) ───────────
        model = LSTMModel(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=1,
        )
        model = model.to(device)
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # ── Step 6: Train ───────────────────────────────────
        train_losses, test_losses = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            model_save_path=model_path,
        )

        # ── Step 7: Save training plot ──────────────────────
        plot_training_history(
            train_losses,
            test_losses,
            title=f"Training History — {name.upper()} ({item_id})",
            save_path=plot_path,
        )

        results.append({
            "category": name,
            "item_id": item_id,
            "best_test_loss": min(test_losses),
            "model_path": model_path,
            "scaler_path": scaler_path,
        })

        # ── Cleanup memory ──────────────────────────────────
        del model, dataset, train_set, test_set, train_loader, test_loader
        del df, normalized_sales, scaler
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Summary")
    print(f"{'='*60}")
    if results:
        for r in results:
            print(f"  ✅ {r['category'].upper():10s} | Item: {r['item_id']:20s} | "
                  f"Loss: {r['best_test_loss']:.6f} | Model: {r['model_path']}")
    else:
        print("  ⚠️  No models were trained.")

    print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
