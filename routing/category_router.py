"""
Category Routing Engine — orchestrates the full multi-product
forecasting pipeline for uploaded datasets.

Pipeline:  Validate → Group → Route → Forecast → Cost → Risk → Decision

🚨  This module NEVER loads models directly.
    All predictions go through the ForecastService.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from services.forecast_service import (
    ForecastService,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    resolve_category,
    CATEGORY_REGISTRY,
)

from decision.reorder_engine import compute_reorder_quantity
from decision.procurement_engine import compute_procurement_plan
from decision.purchase_order_engine import generate_purchase_order


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "category", "product_name", "date",
    "sales", "price", "weekday", "month", "is_weekend", "is_event_day",
]

CATEGORY_NORMALIZATION = {
    "FOOD":      "Grocery",
    "FOODS":     "Grocery",
    "GROCERY":   "Grocery",
    "HOUSEHOLD": "Household",
    "HOBBY":     "Hobbies",
    "HOBBIES":   "Hobbies",
}

SUPPORTED_CATEGORIES = ["Grocery", "Household", "Hobbies"]


# ─────────────────────────────────────────────────────────────
# 1.  Dataset Validation
# ─────────────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Validate an uploaded multi-product dataset.

    Returns:
        {
            "valid": bool,
            "errors": list[str],       — fatal issues preventing processing
            "warnings": list[str],     — non-fatal notes
            "products": list[tuple],   — (category, product_name) pairs that passed
        }
    """
    result: Dict = {"valid": False, "errors": [], "warnings": [], "products": []}

    # ── 1a. Required columns ─────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result["errors"].append(
            f"Missing required columns: {', '.join(missing)}"
        )
        return result

    # ── 1b. Numeric types ────────────────────────────────────
    numeric_cols = ["sales", "price", "weekday", "month", "is_weekend", "is_event_day"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            result["errors"].append(f"Column '{col}' must be numeric.")

    if result["errors"]:
        return result

    # ── 1c. NaN check ────────────────────────────────────────
    nan_cols = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
    if nan_cols:
        result["errors"].append(
            f"NaN values found in: {', '.join(nan_cols)}"
        )
        return result

    # ── 1d. Category normalisation ──────────────────────────
    df["category"] = df["category"].str.upper().str.strip()
    df["category"] = df["category"].map(
        lambda x: CATEGORY_NORMALIZATION.get(x, x)
    )

    # Log mappings for debugging
    for raw in df["category"].unique():
        if raw not in SUPPORTED_CATEGORIES:
            print(f"  ⚠️  Unmapped category: '{raw}'")
        else:
            print(f"  ✅ Category '{raw}' ready")

    # Validate that every category is now supported
    invalid_cats = [c for c in df["category"].unique() if c not in SUPPORTED_CATEGORIES]
    if invalid_cats:
        result["errors"].append(
            f"Unsupported category after normalisation: {', '.join(invalid_cats)}. "
            f"Allowed: {', '.join(SUPPORTED_CATEGORIES)}."
        )
        return result

    # ── 1e. Per-product row count ────────────────────────────
    grouped = df.groupby(["category", "product_name"])
    valid_products = []
    skipped = []

    for (cat, prod), grp in grouped:
        if len(grp) < SEQUENCE_LENGTH:
            skipped.append(f"{prod} ({cat}, {len(grp)} rows)")
        else:
            valid_products.append((cat, prod))

    if skipped:
        result["warnings"].append(
            f"Skipped {len(skipped)} product(s) with < {SEQUENCE_LENGTH} rows: "
            + ", ".join(skipped[:5])
            + ("…" if len(skipped) > 5 else "")
        )

    if not valid_products:
        result["errors"].append(
            f"No product has ≥ {SEQUENCE_LENGTH} rows of data."
        )
        return result

    # ── All passed ───────────────────────────────────────────
    result["valid"] = True
    result["products"] = valid_products
    return result


# ─────────────────────────────────────────────────────────────
# 2.  Cost Engine (pure computation — no UI)
# ─────────────────────────────────────────────────────────────

def compute_economics(
    predictions: List[float],
    lower_bound: Optional[List[float]],
    upper_bound: Optional[List[float]],
    current_stock: int,
    unit_price: float,
    holding_rate: float,
    stockout_penalty: float,
) -> Dict:
    """
    Compute economic impact metrics for a single product.

    Returns dict with: revenue, holding_cost, shortage_cost, total_cost,
    best_case_cost, worst_case_cost, risk_exposure.
    """
    total_demand = float(np.sum(predictions))

    revenue = total_demand * unit_price
    holding_cost = current_stock * holding_rate
    shortage_units = max(0.0, total_demand - current_stock)
    shortage_cost = shortage_units * stockout_penalty
    total_cost = holding_cost + shortage_cost

    # Best / worst case from intervals
    best_case_cost = total_cost
    worst_case_cost = total_cost
    risk_exposure = 0.0

    if lower_bound is not None and len(lower_bound) > 0 and upper_bound is not None and len(upper_bound) > 0:
        lower_demand = float(np.sum(lower_bound))
        upper_demand = float(np.sum(upper_bound))

        best_case_shortage = max(0.0, lower_demand - current_stock)
        best_case_cost = holding_cost + (best_case_shortage * stockout_penalty)

        worst_case_shortage = max(0.0, upper_demand - current_stock)
        worst_case_cost = holding_cost + (worst_case_shortage * stockout_penalty)

        risk_exposure = worst_case_cost - total_cost

    return {
        "revenue": round(revenue, 2),
        "holding_cost": round(holding_cost, 2),
        "shortage_cost": round(shortage_cost, 2),
        "total_cost": round(total_cost, 2),
        "best_case_cost": round(best_case_cost, 2),
        "worst_case_cost": round(worst_case_cost, 2),
        "risk_exposure": round(risk_exposure, 2),
    }


# ─────────────────────────────────────────────────────────────
# 3.  Risk & Decision Logic
# ─────────────────────────────────────────────────────────────

def evaluate_risk(
    total_demand: float,
    current_stock: int,
    shortage_cost: float,
    holding_cost: float,
) -> Dict:
    """
    Determine risk level and recommended action.

    Returns: { risk_level: str, action: str }
    """
    if current_stock <= 0 or total_demand > current_stock * 1.5:
        risk_level = "High"
        action = "Reorder"
    elif total_demand > current_stock:
        risk_level = "Medium"
        action = "Monitor"
    elif shortage_cost > holding_cost:
        risk_level = "Medium"
        action = "Reorder"
    else:
        risk_level = "Low"
        action = "Hold"

    return {"risk_level": risk_level, "action": action}


# ─────────────────────────────────────────────────────────────
# 4.  Main Orchestrator
# ─────────────────────────────────────────────────────────────

def process_uploaded_dataset(
    df: pd.DataFrame,
    forecast_service: ForecastService,
    forecast_days: int = 7,
    current_stock: int = 10,
    holding_rate: float = 0.5,
    stockout_penalty: float = 5.0,
    stock_map: dict = None,
    demand_multiplier_percent: float = 0.0,
) -> List[Dict]:
    """
    Full pipeline: validate → group → route → forecast → cost → risk.

    Args:
        df:                Uploaded DataFrame (already normalised column names).
        forecast_service:  Pre-initialised ForecastService instance.
        forecast_days:     Number of days to forecast.
        current_stock:     Default stock level (used when stock_map is missing a product).
        unit_price:        Selling price per unit.
        holding_rate:      Holding cost per unit per day.
        stockout_penalty:  Penalty per unit of unmet demand.
        stock_map:         Optional dict mapping product_name → stock level.

    Returns:
        List of per-product result dicts.
    """
    results: List[Dict] = []

    # ── Validate ─────────────────────────────────────────────
    vr = validate_dataset(df)
    if not vr["valid"]:
        # Return a single error result
        results.append({
            "product_name": "—",
            "category": "—",
            "forecast": [],
            "lower_bound": [],
            "upper_bound": [],
            "total_demand": 0,
            "stock": current_stock,
            "unit_price": 0.0,
            "risk_level": "—",
            "action": "—",
            "revenue": 0,
            "holding_cost": 0,
            "shortage_cost": 0,
            "total_cost": 0,
            "sales_history": np.array([]),
            "error": "; ".join(vr["errors"]),
            "forecast_total": 0,
            "safety_stock": 0,
            "reorder_point": 0,
            "reorder_quantity": 0,
            "procurement_quantity": 0,
            "procurement_cost": 0,
            "urgency": "—",
            "recommended_action": "—",
            "purchase_order": None,
        })
        return results

    products = vr["products"]

    # ── Process each product ─────────────────────────────────
    for cat_raw, prod_name in products:
        cat = resolve_category(cat_raw)

        # Resolve per-product stock
        if stock_map and prod_name in stock_map:
            product_stock = int(stock_map[prod_name])
        else:
            product_stock = current_stock

        # --- Extract & sort product slice ---
        mask = (
            (df["category"] == cat_raw)
            & (df["product_name"] == prod_name)
        )
        prod_df = df.loc[mask].copy()
        if "date" in prod_df.columns:
            prod_df["date"] = pd.to_datetime(prod_df["date"], errors="coerce")
            prod_df = prod_df.dropna(subset=["date"])
            prod_df = prod_df.sort_values("date")

            # --- Detect & fill date gaps ---
            if len(prod_df) >= 2:
                full_range = pd.date_range(
                    start=prod_df["date"].min(),
                    end=prod_df["date"].max(),
                    freq="D",
                )
                n_before = len(prod_df)
                prod_df = prod_df.set_index("date").reindex(full_range)
                prod_df.index.name = "date"
                n_filled = len(prod_df) - n_before

                if n_filled > 0:
                    print(
                        f"  📅 Missing dates detected for '{prod_name}'. "
                        f"Filled {n_filled} missing day(s)."
                    )

                # Fill missing values
                prod_df["sales"] = prod_df["sales"].fillna(0)
                prod_df["price"] = prod_df["price"].ffill().bfill()
                prod_df["weekday"] = prod_df.index.weekday
                prod_df["month"] = prod_df.index.month
                prod_df["is_weekend"] = prod_df["weekday"].isin([5, 6]).astype(int)
                prod_df["is_event_day"] = prod_df["is_event_day"].ffill().fillna(0)
                prod_df["category"] = cat_raw
                prod_df["product_name"] = prod_name
                prod_df = prod_df.reset_index().rename(columns={"index": "date"})

        # --- Ensure minimum sequence length ---
        if len(prod_df) < SEQUENCE_LENGTH:
            print(
                f"  ⚠️  Skipping '{prod_name}' — only {len(prod_df)} rows "
                f"after date alignment (need {SEQUENCE_LENGTH})."
            )
            continue

        sales_history = prod_df["sales"].astype(float).values
        feature_matrix = prod_df[FEATURE_COLUMNS].values.astype(np.float32)

        # --- Forecast via service ---
        try:
            forecast_result = forecast_service.predict_product(
                category=cat,
                feature_matrix=feature_matrix,
                forecast_days=forecast_days,
            )
        except Exception as exc:
            results.append({
                "product_name": prod_name,
                "category": cat,
                "forecast": [],
                "lower_bound": [],
                "upper_bound": [],
                "total_demand": 0,
                "stock": product_stock,
                "unit_price": 0.0,
                "risk_level": "—",
                "action": "—",
                "revenue": 0,
                "holding_cost": 0,
                "shortage_cost": 0,
                "total_cost": 0,
                "sales_history": sales_history,
                "error": str(exc),
                "forecast_total": 0,
                "safety_stock": 0,
                "reorder_point": 0,
                "reorder_quantity": 0,
                "procurement_quantity": 0,
                "procurement_cost": 0,
                "urgency": "—",
                "recommended_action": "—",
                "purchase_order": None,
            })
            continue

        preds = forecast_result["predictions"]
        lower = forecast_result["lower_bound"]
        upper = forecast_result["upper_bound"]
        
        # --- What-If Demand Adjustment ---
        if demand_multiplier_percent != 0.0:
            multiplier = 1.0 + (demand_multiplier_percent / 100.0)
            preds = np.maximum(0, np.array(preds) * multiplier)
            if len(lower) > 0:
                lower = np.maximum(0, np.array(lower) * multiplier)
            if len(upper) > 0:
                upper = np.maximum(0, np.array(upper) * multiplier)

        total_demand = float(np.sum(preds))

        # --- Economics ---
        product_unit_price = float(prod_df["price"].iloc[-1]) if "price" in prod_df.columns and not prod_df.empty else 0.0
        econ = compute_economics(
            predictions=preds,
            lower_bound=lower,
            upper_bound=upper,
            current_stock=product_stock,
            unit_price=product_unit_price,
            holding_rate=holding_rate,
            stockout_penalty=stockout_penalty,
        )

        # --- Reorder Metrics ---
        reorder_metrics = compute_reorder_quantity(
            forecast=preds,
            lower_bound=lower,
            upper_bound=upper,
            current_stock=product_stock,
            category=cat
        )

        # --- Procurement Plan ---
        procurement = compute_procurement_plan(
            forecast_total=reorder_metrics["forecast_total"],
            safety_stock=reorder_metrics["safety_stock"],
            reorder_point=reorder_metrics["reorder_point"],
            reorder_quantity=reorder_metrics["reorder_quantity"],
            current_stock=product_stock,
            unit_price=product_unit_price,
            category=cat,
        )

        # --- Risk & action ---
        risk = evaluate_risk(
            total_demand=total_demand,
            current_stock=product_stock,
            shortage_cost=econ["shortage_cost"],
            holding_cost=econ["holding_cost"],
        )

        res_dict = {
            "product_name": prod_name,
            "category": cat,
            "forecast": preds,
            "lower_bound": lower,
            "upper_bound": upper,
            "total_demand": round(total_demand, 2),
            "stock": product_stock,
            "unit_price": product_unit_price,
            "risk_level": risk["risk_level"],
            "action": risk["action"],
            "revenue": econ["revenue"],
            "holding_cost": econ["holding_cost"],
            "shortage_cost": econ["shortage_cost"],
            "total_cost": econ["total_cost"],
            "sales_history": sales_history,
            "error": None,
        }
        res_dict.update(reorder_metrics)
        res_dict.update(procurement)

        # --- Purchase Order Draft ---
        po = generate_purchase_order(
            product_name=prod_name,
            category=cat,
            reorder_quantity=procurement.get("procurement_quantity", reorder_metrics.get("reorder_quantity", 0)),
            unit_price=product_unit_price,
        )
        res_dict["purchase_order"] = po

        results.append(res_dict)

    return results
