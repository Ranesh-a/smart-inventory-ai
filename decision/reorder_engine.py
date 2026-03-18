import math

def compute_reorder_quantity(
    forecast: list[float],
    lower_bound: list[float],
    upper_bound: list[float],
    current_stock: float,
    category: str
) -> dict:
    """
    Computes inventory reorder metrics based on forecasted demand and uncertainty.

    Args:
        forecast: List of predicted sales for the given time window.
        lower_bound: Lower prediction intervals.
        upper_bound: Upper prediction intervals.
        current_stock: The current inventory level for the product.
        category: The canonical product category (Grocery, Household, Hobbies).

    Returns:
        dict: A dictionary containing forecast_total, safety_stock, reorder_point, and reorder_quantity.
    """
    # 1. Total forecasted demand
    F = sum(forecast)
    
    # 2. Maximum uncertainty
    # Handle missing intervals or empty bounds safely
    U = F
    if upper_bound is not None and len(upper_bound) > 0:
        U = max(upper_bound)
    
    # Ensure U is not less than F (negative uncertainty)
    U = max(U, F)

    # 3. Category factor mapping
    # Fallback to 1.0 if category is unrecognized
    category_factors = {
        "Grocery": 1.0,
        "Household": 1.2,
        "Hobbies": 1.4
    }
    
    # Map alias to canonical just in case, though it should be resolved
    canonical_cat = category
    if canonical_cat.lower() == "food":
        canonical_cat = "Grocery"
    elif canonical_cat.lower() == "hobby":
        canonical_cat = "Hobbies"

    factor = category_factors.get(canonical_cat, 1.0)

    # 4. Compute Safety Stock
    safety_stock = max(0.0, (U - F) * factor)

    # 5. Compute Reorder Point
    reorder_point = F + safety_stock

    # 6. Compute Recommended Reorder Quantity
    reorder_qty = max(0.0, reorder_point - current_stock)

    return {
        "forecast_total": round(F, 2),
        "safety_stock": math.ceil(safety_stock),
        "reorder_point": math.ceil(reorder_point),
        "reorder_quantity": math.ceil(reorder_qty)
    }
