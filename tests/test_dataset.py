"""
Tests for MultivariateSlidingWindowDataset.
Validates: shapes, no leakage, no cross-item contamination, NaN rejection.
"""

import sys, os
import numpy as np
import pandas as pd
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import (
    load_dataset,
    select_features,
    generate_sequences_per_item,
    MultivariateSlidingWindowDataset,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
)

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")


# ─────────────────────────────────────────────────────────────
# Create a small synthetic CSV for controlled testing
# ─────────────────────────────────────────────────────────────

def make_dummy_csv(path: str, n_items: int = 3, n_days: int = 40):
    """Create a tiny CSV where sales = day_num * 10 + item_index for traceability."""
    rows = []
    items = [f"ITEM_{i}" for i in range(n_items)]
    for i, item in enumerate(items):
        for d in range(1, n_days + 1):
            rows.append({
                "item_id": item,
                "store_id": "STORE_1",
                "d": f"d_{d}",
                "sales": float(d * 10 + i),  # unique per item+day
                "price": 2.0 + d * 0.01,
                "weekday": d % 7,
                "month": (d // 30) + 1,
                "is_weekend": 1 if d % 7 >= 5 else 0,
                "is_event_day": 1 if d % 10 == 0 else 0,
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return items, n_days


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy_data.csv")
    items, n_days = make_dummy_csv(dummy_path, n_items=3, n_days=40)
    n_features = len(FEATURE_COLUMNS)

    print("\n" + "=" * 60)
    print("  TEST SUITE: MultivariateSlidingWindowDataset")
    print("=" * 60)

    # ── Test 1: load_dataset ─────────────────────────────────
    print("\n[1] load_dataset")
    df = load_dataset(dummy_path)
    check("Row count = 3 items × 40 days = 120", len(df) == 120)
    check("Sorted by item_id + day_num", df.iloc[0]["day_num"] == 1)

    # ── Test 2: select_features ──────────────────────────────
    print("\n[2] select_features")
    feat_df = select_features(df)
    check(f"Feature columns = {n_features}", feat_df.shape[1] == n_features)
    check("No item_id column", "item_id" not in feat_df.columns)
    check("No store_id column", "store_id" not in feat_df.columns)

    # ── Test 3: generate_sequences_per_item ──────────────────
    print("\n[3] generate_sequences_per_item")
    # With 40 days per item: windows = 40 - 30 = 10 per item, 3 items → 30
    X, y = generate_sequences_per_item(df, sequence_length=SEQUENCE_LENGTH)
    expected_windows = 3 * (n_days - SEQUENCE_LENGTH)
    check(f"Total windows = {expected_windows}", X.shape[0] == expected_windows)
    check(f"X shape = (N, 30, {n_features})", X.shape[1:] == (SEQUENCE_LENGTH, n_features))
    check("y is 1D", y.ndim == 1)

    # ── Test 4: No Leakage ───────────────────────────────────
    print("\n[4] No sequence leakage")
    # For ITEM_0: sales = day_num * 10 + 0
    # First window: X[0] covers days 1..30, target = day 31 → sales = 310
    check("y[0] == sales at day 31 for ITEM_0 (310.0)", abs(y[0] - 310.0) < 1e-5)
    # X[0] should NOT contain the target day's sales (310)
    sales_in_window = X[0, :, 0]  # sales is feature index 0
    check("Target value 310 NOT in X[0] input window", 310.0 not in sales_in_window)
    # Last value in X[0] should be day 30 → sales = 300
    check("Last input value = sales at day 30 (300.0)", abs(sales_in_window[-1] - 300.0) < 1e-5)

    # ── Test 5: No Cross-Item Contamination ──────────────────
    print("\n[5] No cross-item contamination")
    # ITEM_0 has sales = day*10, ITEM_1 has sales = day*10+1
    # Windows 0-9 belong to ITEM_0, windows 10-19 to ITEM_1
    windows_per_item = n_days - SEQUENCE_LENGTH  # 10

    # Check ITEM_0's last window (index 9): target = day 40 → sales = 400
    check("ITEM_0 last target = day 40 sales (400.0)", abs(y[windows_per_item - 1] - 400.0) < 1e-5)

    # Check ITEM_1's first window (index 10): covers days 1..30, target day 31 → sales = 311
    check("ITEM_1 first target = day 31 sales (311.0)", abs(y[windows_per_item] - 311.0) < 1e-5)

    # ITEM_1 window should contain ITEM_1 sales (ending in 1), not ITEM_0 sales
    item1_first_sales = X[windows_per_item, 0, 0]  # first day sales for ITEM_1's first window
    check("ITEM_1 window starts with ITEM_1 data (11.0)", abs(item1_first_sales - 11.0) < 1e-5)

    # ── Test 6: Dataset Class ────────────────────────────────
    print("\n[6] MultivariateSlidingWindowDataset")
    dataset = MultivariateSlidingWindowDataset(X, y)
    check(f"len(dataset) == {expected_windows}", len(dataset) == expected_windows)
    sample_X, sample_y = dataset[0]
    check(f"sample X shape = (30, {n_features})", sample_X.shape == (SEQUENCE_LENGTH, n_features))
    check("sample y shape = (1,)", sample_y.shape == (1,))
    check("X is FloatTensor", sample_X.dtype == torch.float32)
    check("y is FloatTensor", sample_y.dtype == torch.float32)

    # ── Test 7: Data Quality Checks ──────────────────────────
    print("\n[7] Data quality checks")
    check("No NaN in X", not torch.isnan(dataset.X).any().item())
    check("No NaN in y", not torch.isnan(dataset.y).any().item())

    # Verify NaN rejection
    try:
        bad_X = np.full((5, 30, 6), np.nan, dtype=np.float32)
        bad_y = np.zeros(5, dtype=np.float32)
        MultivariateSlidingWindowDataset(bad_X, bad_y)
        check("NaN X raises ValueError", False)
    except ValueError:
        check("NaN X raises ValueError", True)

    # Verify shape rejection
    try:
        bad_X = np.zeros((5, 30), dtype=np.float32)  # 2D instead of 3D
        bad_y = np.zeros(5, dtype=np.float32)
        MultivariateSlidingWindowDataset(bad_X, bad_y)
        check("2D X raises ValueError", False)
    except ValueError:
        check("2D X raises ValueError", True)

    # ── Test 8: DataLoader Compatibility ─────────────────────
    print("\n[8] DataLoader compatibility")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    batch_X, batch_y = next(iter(loader))
    check(f"Batch X shape = (4, 30, {n_features})", batch_X.shape == (4, SEQUENCE_LENGTH, n_features))
    check("Batch y shape = (4, 1)", batch_y.shape == (4, 1))

    # ── Clean up ─────────────────────────────────────────────
    os.remove(dummy_path)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60 + "\n")

    if FAIL > 0:
        sys.exit(1)
