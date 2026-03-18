import re

with open("api.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace variables
text = re.sub(r"models: dict = \{}.*?data_cache: dict = \{\}  # category  DataFrame \(long format\)", "data_cache: dict = {}  # category -> DataFrame (long format)", text, flags=re.DOTALL)

# Replace _ensure_loaded
text = re.sub(r"def _ensure_loaded.*?def _ensure_loaded_mv", """def _ensure_loaded(category: str):
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

def _ensure_loaded_mv""", text, flags=re.DOTALL)

# Remove unused functions
text = re.sub(r"def _ensure_loaded_mv.*?def _get_item_sales", "def _get_item_sales", text, flags=re.DOTALL)
text = re.sub(r"def _predict\(.*?# Pydantic Schemas", "# Pydantic Schemas", text, flags=re.DOTALL)

with open("api.py", "w", encoding="utf-8") as f:
    f.write(text)
print("api.py rewritten")
