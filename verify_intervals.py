"""
🔍 Prediction Interval Verification — Read-Only Diagnostic

Performs STRICT statistical validation on LSTM prediction intervals.
Does NOT modify any data or logic. Only diagnoses and reports.

Usage:
    python verify_intervals.py
"""

import os
import sys
import numpy as np
import torch
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import LSTMModel, get_device
from src.dataset import SlidingWindowDataset

# ─────────────────────────────────────────────────────────────
# Constants (must match train.py / baseline_comparison.py)
# ─────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 50
NUM_LAYERS = 1
INPUT_SIZE = 1
OUTPUT_SIZE = 1
TRAIN_RATIO = 0.8
Z_SCORE = 1.96

DATASETS = [
    {"name": "food",      "file": "project_data.csv",     "target_id": "FOODS_3_002"},
    {"name": "hobby",     "file": "subset_hobbies.csv",   "target_id": None},
    {"name": "household", "file": "subset_household.csv", "target_id": None},
]


# ─────────────────────────────────────────────────────────────
# Data helpers (unchanged from baseline_comparison.py)
# ─────────────────────────────────────────────────────────────

def load_csv_to_long(file_path):
    df = pd.read_csv(file_path)
    if "sales" in df.columns and "d" in df.columns:
        return df[["item_id", "d", "sales"]].copy()
    day_cols = [c for c in df.columns if c.startswith("d_")]
    id_cols = [c for c in df.columns if not c.startswith("d_")]
    long_df = df.melt(id_vars=id_cols, value_vars=day_cols,
                      var_name="d", value_name="sales")
    long_df["d_num"] = long_df["d"].str.replace("d_", "", regex=False).astype(int)
    long_df = long_df.sort_values(["item_id", "d_num"]).reset_index(drop=True)
    return long_df[["item_id", "d", "sales", "d_num"]].copy()


def select_item(df, target_id):
    if target_id is not None:
        item_df = df[df["item_id"] == target_id]
        item_id = target_id
    else:
        totals = df.groupby("item_id")["sales"].sum()
        item_id = totals.idxmax()
        item_df = df[df["item_id"] == item_id]
    if "d_num" in item_df.columns:
        sales = item_df.sort_values("d_num")["sales"].values.astype(float)
    else:
        item_df = item_df.copy()
        item_df["d_num"] = item_df["d"].astype(str).str.replace("d_", "", regex=False).astype(int)
        sales = item_df.sort_values("d_num")["sales"].values.astype(float)
    return item_id, sales


def lstm_forecast(normalized_sales, test_indices, model, device, scaler):
    model.eval()
    preds_normed = []
    with torch.no_grad():
        for idx in test_indices:
            window = normalized_sales[idx: idx + SEQUENCE_LENGTH]
            x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x).cpu().item()
            preds_normed.append(max(pred, 0.0))
    preds_inv = scaler.inverse_transform(
        np.array(preds_normed).reshape(-1, 1)
    ).flatten()
    return np.maximum(preds_inv, 0)


# ─────────────────────────────────────────────────────────────
# Verification (read-only — NO modifications)
# ─────────────────────────────────────────────────────────────

def verify_category(name, targets, lstm_preds):
    """Run all 6 verification steps. Returns (verdict, errors_list)."""
    errors = []
    warnings = []

    n = len(targets)
    residuals = targets - lstm_preds
    sigma = float(np.std(residuals))

    lower = np.maximum(lstm_preds - Z_SCORE * sigma, 0)
    upper = lstm_preds + Z_SCORE * sigma
    width = upper - lower

    print(f"\n{'='*60}")
    print(f"  VERIFICATION: {name.upper()}")
    print(f"{'='*60}")

    # ── STEP 1: Residual Std Dev Check ──────────────────────
    print(f"\n  STEP 1 — Residual Std Dev (σ): {sigma:.6f}")
    if sigma == 0:
        errors.append("σ == 0 (degenerate)")
        print("    ❌ ERROR: σ == 0")
    elif np.isnan(sigma):
        errors.append("σ is NaN")
        print("    ❌ ERROR: σ is NaN")
    elif np.isinf(sigma):
        errors.append("σ is Inf")
        print("    ❌ ERROR: σ is Inf")
    else:
        print("    ✅ PASS")

    # ── STEP 2: Interval Logic Check ────────────────────────
    invalid_intervals = int(
        ((lower >= lstm_preds) | (upper <= lstm_preds)).sum()
    )
    # Special case: when prediction is 0 and lower is clipped to 0,
    # lower == prediction is acceptable (not a logic error).
    # Only flag cases where lower > prediction or upper < prediction.
    strict_invalid = int(
        ((lower > lstm_preds) | (upper < lstm_preds)).sum()
    )
    # Also check: lower == prediction can happen when pred - z*σ < 0
    # That's a valid clipping scenario, not a logic error.
    edge_cases = int((lower == lstm_preds).sum())

    print(f"\n  STEP 2 — Interval Logic Check:")
    print(f"    Invalid intervals (lower >= pred OR upper <= pred): {invalid_intervals}")
    print(f"      Of which: lower == pred (zero-clipping edge cases): {edge_cases}")
    print(f"      Strict errors (lower > pred OR upper < pred): {strict_invalid}")
    if strict_invalid > 0:
        errors.append(f"{strict_invalid} strict invalid intervals detected")
        print("    ❌ ERROR: Strict invalid intervals found")
    else:
        print("    ✅ PASS")

    # ── STEP 3: Negative Lower Bound Check ──────────────────
    negative_lower = int((lower < 0).sum())
    print(f"\n  STEP 3 — Negative Lower Bound Check:")
    print(f"    Negative lower bounds: {negative_lower}")
    if negative_lower > 0:
        warnings.append(f"{negative_lower} negative lower bounds")
        print("    ⚠️ WARNING: Negative lower bounds present")
    else:
        print("    ✅ PASS (all clipped at 0)")

    # ── STEP 4: Interval Width Sanity Check ─────────────────
    print(f"\n  STEP 4 — Interval Width Sanity Check:")
    width_series = pd.Series(width)
    desc = width_series.describe()
    print(f"    count  = {desc['count']:.0f}")
    print(f"    mean   = {desc['mean']:.4f}")
    print(f"    std    = {desc['std']:.4f}")
    print(f"    min    = {desc['min']:.4f}")
    print(f"    25%    = {desc['25%']:.4f}")
    print(f"    50%    = {desc['50%']:.4f}")
    print(f"    75%    = {desc['75%']:.4f}")
    print(f"    max    = {desc['max']:.4f}")

    collapsed = int((width < 1e-6).sum())
    if collapsed > 0:
        errors.append(f"{collapsed} collapsed intervals (width ≈ 0)")
        print(f"    ❌ ERROR: {collapsed} collapsed intervals")

    # "Extremely large" heuristic: width > 10× mean of targets
    target_mean = float(np.mean(targets)) if np.mean(targets) > 0 else 1.0
    suspicious_threshold = 10 * target_mean
    too_wide = int((width > suspicious_threshold).sum())
    if too_wide > 0:
        warnings.append(f"{too_wide} intervals wider than 10× target mean ({suspicious_threshold:.1f})")
        print(f"    ⚠️ WARNING: {too_wide} intervals wider than {suspicious_threshold:.1f}")
    else:
        print("    ✅ PASS")

    # ── STEP 5: Coverage Check ──────────────────────────────
    coverage = float(((targets >= lower) & (targets <= upper)).mean())
    print(f"\n  STEP 5 — Coverage Check:")
    print(f"    Coverage: {coverage:.4f} ({coverage*100:.1f}%)")

    if coverage < 0.70:
        errors.append(f"Coverage {coverage:.2%} — POOR intervals")
        print("    ❌ POOR: Coverage below 70%")
    elif 0.85 <= coverage <= 0.98:
        print("    ✅ GOOD: Coverage in expected 85–98% range")
    elif coverage == 1.00:
        warnings.append("Coverage 100% — intervals may be suspiciously wide")
        print("    ⚠️ WARNING: Coverage 100% — suspiciously wide intervals")
    elif coverage > 0.98:
        print(f"    ✅ ACCEPTABLE: Coverage slightly above expected range ({coverage:.1%})")
    else:
        print(f"    ℹ️  Coverage {coverage:.1%} — outside sweet spot but not flagged as error")

    # ── STEP 6: Sample Interval Output ──────────────────────
    print(f"\n  STEP 6 — Sample Intervals (first 10 test points):")
    sample_df = pd.DataFrame({
        "sales": targets[:10],
        "prediction": lstm_preds[:10],
        "lower": lower[:10],
        "upper": upper[:10],
    })
    print(sample_df.to_string(index=True, float_format="%.2f"))

    # ── VERDICT ─────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    if errors:
        verdict = "❌ INVALID"
        print(f"  Verification Verdict: {verdict}")
        for e in errors:
            print(f"    ERROR: {e}")
    else:
        verdict = "✅ VALID"
        print(f"  Verification Verdict: {verdict}")

    if warnings:
        for w in warnings:
            print(f"    WARNING: {w}")

    return verdict, errors, warnings


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  🔍  Prediction Interval Verification — Diagnostic Report")
    print(f"{'='*60}")

    verdicts = {}

    for config in DATASETS:
        name = config["name"]
        file_path = config["file"]
        target_id = config["target_id"]
        model_path = f"models/model_{name}.pth"
        scaler_path = f"models/scaler_{name}.pkl"

        if not os.path.exists(file_path) or not os.path.exists(model_path):
            print(f"\n  ⚠️  Skipping {name}: missing files.")
            continue

        # Load data
        df = load_csv_to_long(file_path)
        item_id, raw_sales = select_item(df, target_id)

        # Load saved scaler
        scaler = joblib.load(scaler_path)
        normalized = scaler.transform(raw_sales.reshape(-1, 1)).flatten()

        # Same sliding-window split
        dataset = SlidingWindowDataset(normalized, SEQUENCE_LENGTH, prediction_horizon=1)
        n_samples = len(dataset)
        train_size = int(n_samples * TRAIN_RATIO)
        test_indices = list(range(train_size, n_samples))

        # Ground truth
        targets = np.array([raw_sales[idx + SEQUENCE_LENGTH] for idx in test_indices])

        # LSTM predictions
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        lstm_preds = lstm_forecast(normalized, test_indices, model, device, scaler)

        # Run verification
        verdict, errs, warns = verify_category(name, targets, lstm_preds)
        verdicts[name.upper()] = verdict

        del model, df
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Final Summary ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL VERIFICATION SUMMARY")
    print(f"{'='*60}")
    for cat, v in verdicts.items():
        print(f"  {cat:<12} {v}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
