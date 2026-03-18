"""
🚀 Smart-Inventory API — FastAPI Backend for LSTM Inference
Runs on http://0.0.0.0:8000

Endpoints:
    POST /predict         → returns multi-day demand forecast for a given category + item
    POST /predict_custom  → forecast from user-supplied sales history (CSV upload)
    GET  /health          → liveness check
    GET  /items           → list available item_ids for a category
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# ── Ensure project root is on sys.path ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import LSTMModel

# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart-Inventory API",
    description="LSTM demand forecasting backend for Food, Hobby, and Household categories.",
    version="1.0.0",
)

# Routing & Forecast Service
from services.forecast_service import ForecastService, resolve_category

# Initialize ForecastService once at startup
forecast_service = ForecastService(base_dir=".")

# ─────────────────────────────────────────────────────────────
# Constants & Category Configuration
# ─────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 50
NUM_LAYERS = 1
OUTPUT_SIZE = 1

# Feature columns expected by multivariate models (must match training order)
MV_FEATURE_COLUMNS = ["sales", "price", "weekday", "month", "is_weekend", "is_event_day"]

# Allowed category names for routing (user-facing → internal match)
ALLOWED_CATEGORIES = ["Food", "Hobby", "Household",
                      "Grocery", "Hobbies"]  # aliases

def _resolve_category(cat: str) -> str:
    """Resolve user-facing category name to internal config key."""
    return resolve_category(cat)

CATEGORY_CONFIG = {
    "Grocery": {
        "model": "models/model_food.pth",
        "model_mv": "models/model_food_multivariate.pth",
        "scaler": "models/scaler_food.pkl",
        "scaler_mv": "models/scaler_food.pkl",
        "data": "project_data.csv",
        "input_size": 1,       # static dataset uses univariate model
        "input_size_mv": 6,    # uploaded data uses multivariate model
    },
    "Hobbies": {
        "model": "models/model_hobby.pth",
        "model_mv": "models/model_hobbies_multivariate.pth",
        "scaler": "models/scaler_hobby.pkl",
        "scaler_mv": "scalers/hobbies_scaler.pkl",
        "data": "subset_hobbies.csv",
        "input_size": 1,
        "input_size_mv": 6,
    },
    "Household": {
        "model": "models/model_household.pth",
        "model_mv": "models/model_household_multivariate.pth",
        "scaler": "models/scaler_household.pkl",
        "scaler_mv": "scalers/household_scaler.pkl",
        "data": "subset_household.csv",
        "input_size": 1,
        "input_size_mv": 6,
    },
}

# ─────────────────────────────────────────────────────────────
# In-Memory Cache (Global Dictionaries)
# ─────────────────────────────────────────────────────────────

models: dict = {}      # category → (model, device)         — univariate
models_mv: dict = {}   # category → (model, device)         — multivariate
scalers: dict = {}     # category → scaler (univariate)
scalers_mv: dict = {}  # category → scaler (multivariate)
data_cache: dict = {}  # category → DataFrame (long format)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_loaded(category: str):
    category = _resolve_category(category)
    if category not in CATEGORY_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Choose from: {list(CATEGORY_CONFIG.keys())}",
        )

    cfg = CATEGORY_CONFIG[category]
    if category not in data_cache:
        df = pd.read_csv(cfg["data"])
        if "sales" in df.columns and "d" in df.columns:
            data_cache[category] = df
        else:
            day_cols = [c for c in df.columns if c.startswith("d_")]
            id_cols = [c for c in df.columns if not c.startswith("d_")]
            long_df = df.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
            data_cache[category] = long_df

def _get_item_sales(category: str, item_id: str) -> np.ndarray:
    """Extract sorted sales array for a specific item."""
    category = _resolve_category(category)
    df = data_cache[category]
    item_df = df[df["item_id"] == item_id].copy()

    if item_df.empty:
        available = sorted(df["item_id"].unique().tolist())[:10]
        raise HTTPException(
            status_code=404,
            detail=f"Item '{item_id}' not found in {category}. First 10 available: {available}",
        )

    # Extract numeric day index from 'd_1', 'd_2', …
    item_df["day_num"] = item_df["d"].astype(str).str.replace("d_", "", regex=False).astype(int)
    item_df = item_df.sort_values("day_num")
    return item_df["sales"].values.astype(float)


# Pydantic Schemas
# ─────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    category: str = Field(..., description="Product category: Food, Hobby, or Household")
    item_id: str = Field(..., description="Item ID from the dataset (e.g. FOODS_3_090)")
    forecast_days: int = Field(default=7, ge=1, le=30, description="Number of days to forecast (1-30)")


class PredictionResponse(BaseModel):
    status: str
    category: str
    item_id: str
    forecast_days: int
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    device: str


class ItemsResponse(BaseModel):
    status: str
    category: str
    count: int
    items: List[str]


class CustomPredictionRequest(BaseModel):
    """Request schema for user-uploaded data (univariate or multivariate)."""
    sales_history: Optional[List[float]] = Field(
        default=None,
        min_length=30,
        description="At least 30 historical sales values (univariate mode).",
    )
    feature_rows: Optional[List[List[float]]] = Field(
        default=None,
        description=(
            "Full multivariate feature rows, each row = [sales, price, weekday, month, is_weekend, is_event_day]. "
            "At least 30 rows required."
        ),
    )
    category: str = Field(
        ...,
        description="Category whose model/scaler to use: Food/Grocery, Hobby/Hobbies, or Household.",
    )
    forecast_days: int = Field(
        default=7, ge=1, le=30,
        description="Number of days to forecast (1-30).",
    )


class CustomPredictionResponse(BaseModel):
    status: str
    category: str
    forecast_days: int
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    device: str


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Liveness check."""
    return {
        "status": "ok",
        "device": str(forecast_service.device),
        "categories_available": forecast_service.available_categories,
    }


@app.get("/items/{category}", response_model=ItemsResponse)
def list_items(category: str):
    """List all unique item IDs available in a category."""
    _ensure_loaded(category)
    items = sorted(data_cache[_resolve_category(category)]["item_id"].unique().tolist())
    return ItemsResponse(
        status="success",
        category=category,
        count=len(items),
        items=items,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """
    Run LSTM inference for a given category + item.
    Returns `forecast_days` predictions.
    """
    cat = _resolve_category(req.category)

    # 1. Get item sales history (we still use data_cache for static datasets)
    _ensure_loaded(cat)
    sales = _get_item_sales(cat, req.item_id)

    if len(sales) < SEQUENCE_LENGTH + 1:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough sales history for {req.item_id}. "
                   f"Need at least {SEQUENCE_LENGTH + 1} days, got {len(sales)}.",
        )

    # Convert pure univariate sales array into multivariate shape (T, 6)
    # Since static mode only uses sales, we pad the other 5 columns with 0.
    T = len(sales)
    mat = np.zeros((T, 6), dtype=np.float32)
    mat[:, 0] = sales

    try:
        res = forecast_service.predict_product(
            category=cat,
            feature_matrix=mat,
            forecast_days=req.forecast_days
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        status="success",
        category=req.category,
        item_id=req.item_id,
        forecast_days=req.forecast_days,
        predictions=res["predictions"],
        lower_bound=res["lower_bound"],
        upper_bound=res["upper_bound"],
        device=str(forecast_service.device),
    )


@app.post("/predict_custom", response_model=CustomPredictionResponse)
def predict_custom(req: CustomPredictionRequest):
    """
    Run LSTM inference on user-supplied data.
    """
    cat = _resolve_category(req.category)

    # ── Multivariate mode ────────────────────────────────────
    if req.feature_rows is not None:
        mat = np.array(req.feature_rows, dtype=np.float32)  # (T, 6)

        if mat.ndim != 2 or mat.shape[1] != len(MV_FEATURE_COLUMNS):
            raise HTTPException(
                status_code=400,
                detail=f"Each feature row must have {len(MV_FEATURE_COLUMNS)} values.",
            )
        if mat.shape[0] < SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {SEQUENCE_LENGTH} rows, got {mat.shape[0]}.",
            )

    # ── Univariate mode (backward compatible) ────────────────
    elif req.sales_history is not None:
        sales = np.array(req.sales_history, dtype=np.float32)
        if len(sales) < SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {SEQUENCE_LENGTH} sales values, got {len(sales)}.",
            )

        mat = np.zeros((len(sales), len(MV_FEATURE_COLUMNS)), dtype=np.float32)
        mat[:, 0] = sales

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'sales_history' or 'feature_rows'.",
        )

    try:
        res = forecast_service.predict_product(
            category=cat,
            feature_matrix=mat,
            forecast_days=req.forecast_days
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    return CustomPredictionResponse(
        status="success",
        category=cat,
        forecast_days=req.forecast_days,
        predictions=res["predictions"],
        lower_bound=res["lower_bound"],
        upper_bound=res["upper_bound"],
        device=str(forecast_service.device),
    )


# ─────────────────────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
