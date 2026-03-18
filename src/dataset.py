"""
Dataset module for Smart Inventory Demand Forecasting.
Handles data loading, normalization, and PyTorch Dataset creation.

Supports both:
  - Univariate mode  (original SalesDataProcessor + SlidingWindowDataset)
  - Multivariate mode (MultivariateSlidingWindowDataset for M5 data)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from typing import Tuple, Optional, List


# ─────────────────────────────────────────────────────────────
# Feature Configuration
# ─────────────────────────────────────────────────────────────

FEATURE_COLUMNS = ["sales", "price", "weekday", "month", "is_weekend", "is_event_day"]
TARGET_COLUMN = "sales"
SEQUENCE_LENGTH = 30
PREDICTION_HORIZON = 1


# ─────────────────────────────────────────────────────────────
# Step 1 — Load Dataset
# ─────────────────────────────────────────────────────────────

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the multivariate CSV and sort by item_id + day number.

    MUST preserve temporal ordering per item.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    # Validate required columns
    required = ["item_id", "d", "sales"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract numeric day index from 'd_1', 'd_2', …
    df["day_num"] = df["d"].astype(str).str.replace("d_", "", regex=False).astype(int)

    # Sort by item_id THEN chronological day — MANDATORY
    df = df.sort_values(["item_id", "day_num"]).reset_index(drop=True)

    print(f"✅ Loaded {file_path}: {df.shape[0]:,} rows, {df['item_id'].nunique()} items")
    return df


# ─────────────────────────────────────────────────────────────
# Step 2 — Feature Selection
# ─────────────────────────────────────────────────────────────

def select_features(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Select ONLY the numeric ML features.
    Excludes: item_id, store_id, d, day_num, and any other non-feature columns.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataframe: {missing}")

    return df[feature_cols].copy()


# ─────────────────────────────────────────────────────────────
# Step 3 & 4 — Per-Item Sliding Window Generation
# ─────────────────────────────────────────────────────────────

def generate_sequences_per_item(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    sequence_length: int = SEQUENCE_LENGTH,
    prediction_horizon: int = PREDICTION_HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding-window sequences grouped BY ITEM.

    For each item:
        X[t] = features[t : t + sequence_length]          → shape (seq_len, N_features)
        y[t] = sales[t + sequence_length]                  → scalar

    Returns:
        all_X: np.ndarray of shape (N_total_windows, sequence_length, N_features)
        all_y: np.ndarray of shape (N_total_windows,)

    CRITICAL:
        - NEVER mixes items across window boundaries.
        - Target sales[t+30] is NEVER inside the input window X[t].
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Validate
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "item_id" not in df.columns:
        raise ValueError("DataFrame must contain 'item_id' for per-item grouping.")

    target_idx = feature_cols.index(TARGET_COLUMN)
    min_length = sequence_length + prediction_horizon

    all_X: List[np.ndarray] = []
    all_y: List[float] = []

    grouped = df.groupby("item_id", sort=False)
    skipped = 0

    for item_id, item_df in grouped:
        # Extract feature matrix for this item (already sorted by day_num)
        features = item_df[feature_cols].values.astype(np.float32)  # (T, N_features)

        if len(features) < min_length:
            skipped += 1
            continue

        # Sliding window — NO cross-item contamination
        for t in range(len(features) - sequence_length):
            X_window = features[t : t + sequence_length]        # (30, N_features)
            y_target = features[t + sequence_length, target_idx] # scalar: sales at t+30

            all_X.append(X_window)
            all_y.append(y_target)

    if not all_X:
        raise ValueError("No valid sequences generated. Check data length and sequence_length.")

    X_arr = np.array(all_X, dtype=np.float32)  # (N, 30, N_features)
    y_arr = np.array(all_y, dtype=np.float32)  # (N,)

    if skipped > 0:
        print(f"⚠️  Skipped {skipped} items with fewer than {min_length} days of history.")

    print(f"✅ Generated {X_arr.shape[0]:,} sequences | X: {X_arr.shape} | y: {y_arr.shape}")
    return X_arr, y_arr


# ─────────────────────────────────────────────────────────────
# Step 5 & 6 — PyTorch Dataset Class
# ─────────────────────────────────────────────────────────────

class MultivariateSlidingWindowDataset(Dataset):
    """
    PyTorch Dataset for multivariate sliding-window time series forecasting.

    Input:  (sequence_length, N_features)   e.g. (30, 6)
    Target: (1,)                             next-day sales
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Input sequences, shape (N, sequence_length, N_features).
            y: Target values, shape (N,).
        """
        # ── Step 7: Data Quality Checks ─────────────────────
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (N, seq_len, features), got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (N,), got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y sample count mismatch: {X.shape[0]} vs {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("Dataset is empty — no sequences provided.")
        if np.isnan(X).any():
            raise ValueError("X contains NaN values. Clean data before creating dataset.")
        if np.isnan(y).any():
            raise ValueError("y contains NaN values. Clean data before creating dataset.")

        self.X = torch.FloatTensor(X)  # (N, 30, N_features)
        self.y = torch.FloatTensor(y)  # (N,)

        print(
            f"✅ MultivariateSlidingWindowDataset created: "
            f"{len(self)} samples | X: {tuple(self.X.shape)} | y: {tuple(self.y.shape)}"
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: (sequence_length, N_features) e.g. (30, 6)
            y: (1,) next-day sales scalar
        """
        return self.X[idx], self.y[idx].unsqueeze(0)


# ─────────────────────────────────────────────────────────────
# Convenience: End-to-End Builder
# ─────────────────────────────────────────────────────────────

def build_multivariate_dataset(
    file_path: str,
    feature_cols: List[str] = None,
    sequence_length: int = SEQUENCE_LENGTH,
) -> MultivariateSlidingWindowDataset:
    """
    One-call pipeline:  CSV → sorted DataFrame → per-item windows → Dataset.

    Usage:
        dataset = build_multivariate_dataset("food_multivariate_dataset.csv")
        X, y = dataset[0]  # X: (30, 6), y: (1,)
    """
    df = load_dataset(file_path)
    X, y = generate_sequences_per_item(df, feature_cols, sequence_length)
    return MultivariateSlidingWindowDataset(X, y)


# ─────────────────────────────────────────────────────────────
# Legacy: Original Univariate Classes (kept for backward compat)
# ─────────────────────────────────────────────────────────────

class SalesDataProcessor:
    """
    Processes sales data: loads, normalizes, and prepares for training.
    (Original univariate processor — kept for backward compatibility.)
    """

    def __init__(self, data_path: str = "project_data.csv", scaler_path: str = "scaler.pkl"):
        self.data_path = data_path
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.normalized_sales = None

    def load_data(self) -> pd.DataFrame:
        print(f"Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"Data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        return self.data

    def normalize_sales(self, save_scaler: bool = True) -> np.ndarray:
        if self.data is None:
            self.load_data()
        sales_values = self.data['sales'].values.reshape(-1, 1)
        self.normalized_sales = self.scaler.fit_transform(sales_values).flatten()
        print(f"Sales range: [{self.data['sales'].min()}, {self.data['sales'].max()}]")
        print(f"Normalized range: [{self.normalized_sales.min():.4f}, {self.normalized_sales.max():.4f}]")
        if save_scaler:
            joblib.dump(self.scaler, self.scaler_path)
            print(f"Scaler saved to: {self.scaler_path}")
        return self.normalized_sales

    def load_scaler(self) -> MinMaxScaler:
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"Scaler loaded from: {self.scaler_path}")
            return self.scaler
        raise FileNotFoundError(f"Scaler not found at: {self.scaler_path}")

    def get_item_sales(self, item_id: str) -> np.ndarray:
        if self.data is None:
            self.load_data()
        item_data = self.data[self.data['item_id'] == item_id].sort_values('d')
        return item_data['sales'].values

    def get_unique_items(self) -> list:
        if self.data is None:
            self.load_data()
        return self.data['item_id'].unique().tolist()


class SlidingWindowDataset(Dataset):
    """
    PyTorch Dataset for univariate sliding window time series forecasting.
    (Original class — kept for backward compatibility with existing models.)
    """

    def __init__(self, sales_data: np.ndarray, sequence_length: int = 30, prediction_horizon: int = 1):
        self.sales_data = sales_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.n_samples = len(sales_data) - sequence_length - prediction_horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {sequence_length + prediction_horizon} "
                f"but got {len(sales_data)}"
            )
        print(f"Created dataset with {self.n_samples} samples")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.sales_data[idx:idx + self.sequence_length]
        y = self.sales_data[idx + self.sequence_length]
        x_tensor = torch.FloatTensor(x).unsqueeze(-1)  # (seq_len, 1)
        y_tensor = torch.FloatTensor([y])               # (1,)
        return x_tensor, y_tensor


def create_train_test_split(
    dataset: Dataset, train_ratio: float = 0.8
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Split dataset into training and test sets (sequential, not random)."""
    n_samples = len(dataset)
    train_size = int(n_samples * train_ratio)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, n_samples))
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    print(f"Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")
    return train_subset, test_subset


# ─────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Multivariate demo ---
    csv_path = "food_multivariate_dataset.csv"
    if os.path.exists(csv_path):
        dataset = build_multivariate_dataset(csv_path)
        X, y = dataset[0]
        print(f"\nSample X shape: {X.shape}")  # (30, 6)
        print(f"Sample y shape: {y.shape}")    # (1,)
    else:
        print(f"⚠️  {csv_path} not found — skipping multivariate demo.")

    # --- Legacy univariate demo ---
    if os.path.exists("project_data.csv"):
        processor = SalesDataProcessor()
        processor.load_data()
        normalized = processor.normalize_sales()
        ds = SlidingWindowDataset(normalized, sequence_length=30)
        x, y = ds[0]
        print(f"\n[Legacy] Sample input shape: {x.shape}")
        print(f"[Legacy] Sample target shape: {y.shape}")
        train_set, test_set = create_train_test_split(ds)
