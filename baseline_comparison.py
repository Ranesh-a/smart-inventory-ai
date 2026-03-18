"""
📊 Baseline Comparison Script — Naïve, Moving Average, and LSTM

Evaluates all three models on the SAME test split (80/20 sequential)
using MAE and RMSE.  Includes residual-based 95% prediction intervals
for LSTM.  No retraining.  No data leakage.

Usage:
    python baseline_comparison.py
"""

import os
import sys
import numpy as np
import torch
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import LSTMModel, get_device
from src.dataset import SlidingWindowDataset, create_train_test_split

# ─────────────────────────────────────────────────────────────
# Constants (must match train.py)
# ─────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 50
NUM_LAYERS = 1
INPUT_SIZE = 1
OUTPUT_SIZE = 1
TRAIN_RATIO = 0.8
MA_WINDOW = 7  # Moving-average look-back
Z_SCORE = 1.96  # 95% confidence level

DATASETS = [
    {"name": "food",      "file": "project_data.csv",     "target_id": "FOODS_3_002"},
    {"name": "hobby",     "file": "subset_hobbies.csv",   "target_id": None},
    {"name": "household", "file": "subset_household.csv", "target_id": None},
]


# ─────────────────────────────────────────────────────────────
# Data Loading (reused from train.py)
# ─────────────────────────────────────────────────────────────

def load_csv_to_long(file_path: str) -> pd.DataFrame:
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


def select_item(df: pd.DataFrame, target_id):
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


# ─────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────

def naive_forecast(raw_sales: np.ndarray, test_indices: list) -> np.ndarray:
    """
    Naïve forecast: prediction = last value in the input window.
    For each test sample at index `idx`, the input window is
    raw_sales[idx : idx + SEQUENCE_LENGTH] and the target is
    raw_sales[idx + SEQUENCE_LENGTH].
    Prediction = raw_sales[idx + SEQUENCE_LENGTH - 1]  (previous day).
    """
    preds = []
    for idx in test_indices:
        preds.append(raw_sales[idx + SEQUENCE_LENGTH - 1])
    return np.array(preds)


def moving_average_forecast(raw_sales: np.ndarray, test_indices: list,
                            window: int = MA_WINDOW) -> np.ndarray:
    """
    Moving Average forecast: prediction = mean of the last `window` values
    in the input sequence.
    """
    preds = []
    for idx in test_indices:
        recent = raw_sales[idx + SEQUENCE_LENGTH - window: idx + SEQUENCE_LENGTH]
        preds.append(np.mean(recent))
    return np.array(preds)


# ─────────────────────────────────────────────────────────────
# LSTM Inference
# ─────────────────────────────────────────────────────────────

def lstm_forecast(normalized_sales: np.ndarray, test_indices: list,
                  model, device, scaler) -> np.ndarray:
    """
    Run the saved LSTM model on each test window (one-step prediction).
    Returns predictions in original scale.
    """
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
# Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def compute_prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray,
                                  z: float = Z_SCORE) -> dict:
    """
    Compute prediction intervals from residual variance.

    Method:
        residuals = actual - predicted
        σ = std(residuals)
        lower = prediction - z * σ   (clipped at 0)
        upper = prediction + z * σ

    Returns dict with: residual_std, lower_bound, upper_bound,
                       mean_width, coverage_pct.
    """
    residuals = y_true - y_pred
    sigma = float(np.std(residuals))

    lower = np.maximum(y_pred - z * sigma, 0)
    upper = y_pred + z * sigma

    # Coverage: fraction of actuals falling within [lower, upper]
    within = np.sum((y_true >= lower) & (y_true <= upper))
    coverage = float(within / len(y_true)) * 100

    mean_width = float(np.mean(upper - lower))

    return {
        "residual_std": sigma,
        "lower_bound": lower,
        "upper_bound": upper,
        "mean_width": mean_width,
        "coverage_pct": coverage,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    device = get_device()

    print(f"\n{'='*65}")
    print(f"  Baseline Comparison — Naïve vs Moving Average vs LSTM")
    print(f"{'='*65}\n")

    all_results = []

    for config in DATASETS:
        name = config["name"]
        file_path = config["file"]
        target_id = config["target_id"]
        model_path = f"models/model_{name}.pth"
        scaler_path = f"models/scaler_{name}.pkl"

        # ── Skip if resources missing ───────────────────────
        if not os.path.exists(file_path):
            print(f"  ⚠️  Data file missing: {file_path} — skipping.\n")
            continue
        if not os.path.exists(model_path):
            print(f"  ⚠️  Model file missing: {model_path} — skipping.\n")
            continue

        print(f"  ── {name.upper()} ──────────────────────────────")

        # ── 1. Load data ────────────────────────────────────
        df = load_csv_to_long(file_path)
        item_id, raw_sales = select_item(df, target_id)
        print(f"  Item: {item_id}  |  Sales length: {len(raw_sales)} days")

        # ── 2. Load saved scaler (no refitting!) ────────────
        scaler = joblib.load(scaler_path)
        normalized = scaler.transform(raw_sales.reshape(-1, 1)).flatten()

        # ── 3. Create same sliding-window split ─────────────
        dataset = SlidingWindowDataset(normalized, SEQUENCE_LENGTH, prediction_horizon=1)
        n_samples = len(dataset)
        train_size = int(n_samples * TRAIN_RATIO)
        test_indices = list(range(train_size, n_samples))
        print(f"  Total samples: {n_samples}  |  Test samples: {len(test_indices)}")

        # ── 4. Ground-truth targets (original scale) ────────
        targets = np.array([raw_sales[idx + SEQUENCE_LENGTH] for idx in test_indices])

        # ── 5. Naïve forecast ───────────────────────────────
        naive_preds = naive_forecast(raw_sales, test_indices)
        naive_mae, naive_rmse = compute_metrics(targets, naive_preds)

        # ── 6. Moving Average forecast (N=7) ────────────────
        ma_preds = moving_average_forecast(raw_sales, test_indices, window=MA_WINDOW)
        ma_mae, ma_rmse = compute_metrics(targets, ma_preds)

        # ── 7. LSTM forecast ────────────────────────────────
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

        lstm_preds = lstm_forecast(normalized, test_indices, model, device, scaler)
        lstm_mae, lstm_rmse = compute_metrics(targets, lstm_preds)

        # ── 8. Prediction intervals for LSTM ────────────────
        intervals = compute_prediction_intervals(targets, lstm_preds)

        # ── 9. Category results ─────────────────────────────
        cat_results = [
            {"Category": name.upper(), "Model": "Naïve",          "MAE": naive_mae, "RMSE": naive_rmse},
            {"Category": name.upper(), "Model": f"MA (N={MA_WINDOW})", "MAE": ma_mae,    "RMSE": ma_rmse},
            {"Category": name.upper(), "Model": "LSTM",           "MAE": lstm_mae,  "RMSE": lstm_rmse},
        ]
        all_results.extend(cat_results)

        # Print per-category metrics table
        print(f"\n  {'Model':<15} {'MAE':>10} {'RMSE':>10}")
        print(f"  {'─'*37}")
        for r in cat_results:
            print(f"  {r['Model']:<15} {r['MAE']:>10.4f} {r['RMSE']:>10.4f}")

        # Print LSTM prediction interval stats
        print(f"\n  LSTM Prediction Intervals (95%):")
        print(f"    Residual σ       = {intervals['residual_std']:.4f}")
        print(f"    Mean width       = {intervals['mean_width']:.4f}")
        print(f"    Coverage         = {intervals['coverage_pct']:.1f}%")

        # Sample interval values (first 5 test points)
        print(f"\n  Sample intervals (first 5 test points):")
        print(f"    {'Actual':>10} {'Predicted':>10} {'Lower':>10} {'Upper':>10}")
        print(f"    {'─'*42}")
        for i in range(min(5, len(targets))):
            print(f"    {targets[i]:>10.2f} {lstm_preds[i]:>10.2f} "
                  f"{intervals['lower_bound'][i]:>10.2f} {intervals['upper_bound'][i]:>10.2f}")
        print()

        # Store interval summary for combined output
        all_results[-1]["Residual_σ"] = intervals["residual_std"]
        all_results[-1]["Interval_Width"] = intervals["mean_width"]
        all_results[-1]["Coverage_%"] = intervals["coverage_pct"]

        # Cleanup
        del model, df
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Combined Results Table ──────────────────────────────
    if all_results:
        print(f"\n{'='*65}")
        print(f"  COMBINED RESULTS")
        print(f"{'='*65}")
        print(f"\n  {'Category':<12} {'Model':<15} {'MAE':>10} {'RMSE':>10}")
        print(f"  {'─'*49}")
        for r in all_results:
            print(f"  {r['Category']:<12} {r['Model']:<15} {r['MAE']:>10.4f} {r['RMSE']:>10.4f}")
        print(f"\n{'='*65}\n")

        # Save to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("baseline_comparison_results.csv", index=False)
        print("  📁 Results saved to: baseline_comparison_results.csv\n")


if __name__ == "__main__":
    main()
