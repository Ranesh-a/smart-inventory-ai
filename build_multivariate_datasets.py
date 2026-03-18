"""
🚀 Build Multivariate Evaluation Datasets for Hobbies & Household
Adapts the Food pipeline (build_food_eval_dataset.py) for the other two M5 categories.

Inputs (from R:/visual studio code/d/):
    - sales_train_evaluation.csv
    - calendar.csv
    - sell_prices.csv

Outputs (into project inventory):
    - hobbies_evaluation_dataset.csv
    - household_evaluation_dataset.csv

Each output contains: item_id, store_id, d, sales, price, weekday, month,
                       is_weekend, is_event_day, wm_yr_wk
"""

import pandas as pd
import numpy as np
import os
import gc

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

RAW_DIR = r"R:\visual studio code\d"
SALES_PATH = os.path.join(RAW_DIR, "sales_train_evaluation.csv")
CALENDAR_PATH = os.path.join(RAW_DIR, "calendar.csv")
PRICES_PATH = os.path.join(RAW_DIR, "sell_prices.csv")

OUTPUT_DIR = r"R:\visual studio code\project\project inventory"

CATEGORIES = {
    "HOBBIES": {
        "cat_id": "HOBBIES",
        "output": os.path.join(OUTPUT_DIR, "hobbies_evaluation_dataset.csv"),
    },
    "HOUSEHOLD": {
        "cat_id": "HOUSEHOLD",
        "output": os.path.join(OUTPUT_DIR, "household_evaluation_dataset.csv"),
    },
}

FINAL_COLUMNS = [
    "item_id", "store_id", "d", "sales", "price",
    "weekday", "month", "is_weekend", "is_event_day", "wm_yr_wk",
]


# ─────────────────────────────────────────────────────────────
# Functions
# ─────────────────────────────────────────────────────────────

def load_and_filter_sales(sales_path: str, cat_id: str) -> pd.DataFrame:
    """Load the wide-format sales CSV and filter for a single M5 category."""
    print(f"\n  Loading sales data for {cat_id}…")
    df = pd.read_csv(sales_path)
    df_cat = df[df["cat_id"] == cat_id].copy()
    print(f"  Filtered: {df_cat.shape[0]:,} items × {df_cat.shape[1]} columns")
    del df
    gc.collect()
    return df_cat


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide d_1…d_N columns to long format."""
    print("  Melting to long format…")
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    id_vars = [c for c in id_vars if c in df.columns]
    long = pd.melt(df, id_vars=id_vars, var_name="d", value_name="sales")
    print(f"  Long shape: {long.shape[0]:,} rows")
    return long


def merge_calendar(df: pd.DataFrame, calendar_path: str) -> pd.DataFrame:
    """Merge with calendar to get weekday, month, event info, wm_yr_wk."""
    print("  Merging calendar…")
    cal = pd.read_csv(calendar_path)
    keep = ["d", "wm_yr_wk", "weekday", "month", "event_name_1"]
    merged = df.merge(cal[keep], on="d", how="left")
    print(f"  After calendar merge: {merged.shape[0]:,} rows")
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric weekday, is_weekend, is_event_day; drop text cols."""
    print("  Engineering features…")
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }
    df["weekday_num"] = df["weekday"].map(day_map)
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in ("Saturday", "Sunday") else 0)
    df["is_event_day"] = df["event_name_1"].notna().astype(int)
    df.drop(columns=["event_name_1", "weekday"], inplace=True)
    df.rename(columns={"weekday_num": "weekday"}, inplace=True)
    return df


def merge_prices(df: pd.DataFrame, prices_path: str) -> pd.DataFrame:
    """Inner-join with sell_prices on (item_id, store_id, wm_yr_wk)."""
    print("  Merging prices…")
    prices = pd.read_csv(prices_path)
    merged = df.merge(prices, on=["item_id", "store_id", "wm_yr_wk"], how="inner")
    merged.rename(columns={"sell_price": "price"}, inplace=True)
    print(f"  After price merge: {merged.shape[0]:,} rows")
    return merged


def validate(df: pd.DataFrame) -> None:
    """Quick quality checks."""
    print("  Validating…")
    assert "sales" in df.columns
    assert "price" in df.columns
    assert not df["sales"].isna().any(), "NaN in sales!"
    assert not df["price"].isna().any(), "NaN in price!"
    dupes = df.duplicated(subset=["item_id", "store_id", "d"]).sum()
    assert dupes == 0, f"{dupes} duplicate rows!"
    print(f"  ✅ Valid — {df.shape[0]:,} rows, {df['item_id'].nunique()} items")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    for label, cfg in CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"  Building dataset: {label}")
        print(f"{'='*60}")

        df = load_and_filter_sales(SALES_PATH, cfg["cat_id"])
        df = melt_to_long(df)
        gc.collect()

        df = merge_calendar(df, CALENDAR_PATH)
        gc.collect()

        df = engineer_features(df)
        df = merge_prices(df, PRICES_PATH)
        gc.collect()

        # Ensure final column order
        missing = [c for c in FINAL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        validate(df)

        out_path = cfg["output"]
        df[FINAL_COLUMNS].to_csv(out_path, index=False)
        print(f"  💾 Saved → {out_path}")

        del df
        gc.collect()

    print(f"\n{'='*60}")
    print("  ✅ All datasets built successfully.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
