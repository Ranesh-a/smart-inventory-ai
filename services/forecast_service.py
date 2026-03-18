"""
Forecast Service — centralised prediction layer for the
Multi-Category Supermarket Decision Intelligence System.

Loads multivariate LSTM models and scalers ONCE at initialisation,
then exposes a single `predict_product` method used by the routing engine.

🚨 This module NEVER retrains models — inference only.
"""

import os
import numpy as np
import torch
import joblib
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import LSTMModel

# ─────────────────────────────────────────────────────────────
# Model Registry — single source of truth for category → asset mapping
# ─────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 50
NUM_LAYERS = 1
INPUT_SIZE = 6     # multivariate: sales, price, weekday, month, is_weekend, is_event_day
OUTPUT_SIZE = 1

FEATURE_COLUMNS = ["sales", "price", "weekday", "month", "is_weekend", "is_event_day"]

CATEGORY_REGISTRY: Dict[str, dict] = {
    "Grocery": {
        "model": "models/model_food_multivariate.pth",
        "scaler": "models/scaler_food.pkl",
    },
    "Household": {
        "model": "models/model_household_multivariate.pth",
        "scaler": "scalers/household_scaler.pkl",
    },
    "Hobbies": {
        "model": "models/model_hobbies_multivariate.pth",
        "scaler": "scalers/hobbies_scaler.pkl",
    },
}

# User-facing aliases → canonical name
_CATEGORY_ALIAS = {
    "Food":    "Grocery",
    "Hobby":   "Hobbies",
    "grocery": "Grocery",
    "hobbies": "Hobbies",
    "household": "Household",
}


def resolve_category(name: str) -> str:
    """Map any user-facing category string to its canonical registry key."""
    cleaned = name.strip()
    return _CATEGORY_ALIAS.get(cleaned, _CATEGORY_ALIAS.get(cleaned.lower(), cleaned))


# ─────────────────────────────────────────────────────────────
# Forecast Service
# ─────────────────────────────────────────────────────────────

class ForecastService:
    """
    Centralised forecasting service.

    • Loads every registered model + scaler exactly once.
    • Provides `predict_product(category, feature_matrix, forecast_days)`
      returning predictions + prediction intervals.
    """

    def __init__(self, base_dir: str = "."):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = base_dir
        self.models: Dict[str, LSTMModel] = {}
        self.scalers: Dict[str, object] = {}

        self._load_models()
        self._load_scalers()

    # ── Private loaders ──────────────────────────────────────

    def _load_models(self):
        """Load all registered multivariate LSTM models."""
        for category, cfg in CATEGORY_REGISTRY.items():
            model_path = os.path.join(self.base_dir, cfg["model"])
            if not os.path.exists(model_path):
                print(f"  ⚠️  Model not found: {model_path} — {category} unavailable.")
                continue

            model = LSTMModel(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                output_size=OUTPUT_SIZE,
            )
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self.device)
            model.eval()
            self.models[category] = model
            print(f"  ✅ Loaded model: {category} → {model_path}")

    def _load_scalers(self):
        """Load all registered MinMaxScalers."""
        for category, cfg in CATEGORY_REGISTRY.items():
            scaler_path = os.path.join(self.base_dir, cfg["scaler"])
            if not os.path.exists(scaler_path):
                print(f"  ⚠️  Scaler not found: {scaler_path} — {category} unavailable.")
                continue

            self.scalers[category] = joblib.load(scaler_path)
            print(f"  ✅ Loaded scaler: {category} → {scaler_path}")

    # ── Public API ───────────────────────────────────────────

    @property
    def available_categories(self) -> List[str]:
        """Categories with both model AND scaler loaded."""
        return [c for c in CATEGORY_REGISTRY if c in self.models and c in self.scalers]

    def is_available(self, category: str) -> bool:
        cat = resolve_category(category)
        return cat in self.models and cat in self.scalers

    def predict_product(
        self,
        category: str,
        feature_matrix: np.ndarray,
        forecast_days: int = 7,
    ) -> Dict:
        """
        Run multi-step iterative LSTM forecast for a single product.

        Args:
            category:       One of the registered category names (or alias).
            feature_matrix: Shape (T, 6) — raw feature rows [sales, price,
                            weekday, month, is_weekend, is_event_day].
            forecast_days:  Number of days to forecast (1-30).

        Returns:
            dict with keys: predictions, lower_bound, upper_bound.
        """
        cat = resolve_category(category)

        if cat not in self.models:
            raise ValueError(f"No model loaded for category '{cat}'.")
        if cat not in self.scalers:
            raise ValueError(f"No scaler loaded for category '{cat}'.")

        model = self.models[cat]
        scaler = self.scalers[cat]

        mat = feature_matrix.copy().astype(np.float32)  # (T, 6)

        if mat.shape[0] < SEQUENCE_LENGTH:
            raise ValueError(
                f"Need ≥ {SEQUENCE_LENGTH} rows, got {mat.shape[0]}."
            )

        # ── Iterative forecast ───────────────────────────────
        predictions_raw: List[float] = []
        model.eval()
        with torch.no_grad():
            for _ in range(forecast_days):
                window = mat[-SEQUENCE_LENGTH:]                        # (30, 6)
                x = torch.FloatTensor(window).unsqueeze(0).to(self.device)  # (1, 30, 6)
                pred = model(x).cpu().item()
                pred = max(pred, 0.0)
                predictions_raw.append(pred)

                # Build next row: predicted sales + repeat last auxiliary cols
                next_row = mat[-1].copy()
                next_row[0] = pred
                mat = np.vstack([mat, next_row])

        # ── Inverse-transform through scaler ─────────────────
        preds_inv = scaler.inverse_transform(
            np.array(predictions_raw).reshape(-1, 1)
        ).flatten()
        preds_inv = np.maximum(preds_inv, 0)
        preds = [round(float(p), 2) for p in preds_inv]

        # ── Prediction intervals (residual-based) ────────────
        sales_col = feature_matrix[:, 0]  # original sales column
        lower, upper = self._compute_intervals(sales_col, preds)

        return {
            "predictions": preds,
            "lower_bound": lower,
            "upper_bound": upper,
        }

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _compute_intervals(
        sales: np.ndarray,
        predictions: List[float],
        z: float = 1.96,
    ) -> Tuple[List[float], List[float]]:
        """Residual-based prediction intervals (95% CI)."""
        if len(sales) < 2:
            return predictions, predictions

        diffs = np.diff(sales)
        std = float(np.std(diffs)) if len(diffs) > 0 else 0.0

        lower = [round(max(0, p - z * std), 2) for p in predictions]
        upper = [round(max(0, p + z * std), 2) for p in predictions]
        return lower, upper
