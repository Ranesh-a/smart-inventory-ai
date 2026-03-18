"""
Retail Operations Command Dashboard
Aggregates insights from all processed products to provide top-level
supermarket operational metrics including inventory risk, procurement budget,
category demand, and operational recommendations.
"""

def compute_operations_summary(product_results: list) -> dict:
    """
    Compute aggregated operational metrics across all forecasted products.

    Args:
        product_results: List of dictionaries returned by process_uploaded_dataset.
                         Each dictionary represents one product.

    Returns:
        dict containing:
            total_products: int
            total_forecast_demand: float
            total_procurement_cost: float
            high_risk_products: list
            category_demand: dict
            category_revenue: dict
    """
    valid_results = [r for r in product_results if not r.get("error")]

    total_products = len(valid_results)
    total_forecast_demand = sum(r.get("forecast_total", 0) for r in valid_results)
    total_procurement_cost = sum(r.get("procurement_cost", 0) for r in valid_results)

    # Top 5 High-Risk Products
    high_risk_products = [
        r["product_name"] for r in valid_results if r.get("risk_level") == "High"
    ][:5]

    # Category aggregations
    category_demand = {"Grocery": 0.0, "Household": 0.0, "Hobbies": 0.0}
    category_revenue = {"Grocery": 0.0, "Household": 0.0, "Hobbies": 0.0}

    for r in valid_results:
        cat = r.get("category", "")
        if cat in category_demand:
            category_demand[cat] += r.get("forecast_total", 0.0)
            category_revenue[cat] += r.get("revenue", 0.0)
        else:
            # Handle aliases if needed, or fallback
            if cat.lower() in ("food", "grocery"):
                category_demand["Grocery"] += r.get("forecast_total", 0.0)
                category_revenue["Grocery"] += r.get("revenue", 0.0)
            elif cat.lower() in ("hobby", "hobbies"):
                category_demand["Hobbies"] += r.get("forecast_total", 0.0)
                category_revenue["Hobbies"] += r.get("revenue", 0.0)
            elif cat.lower() == "household":
                category_demand["Household"] += r.get("forecast_total", 0.0)
                category_revenue["Household"] += r.get("revenue", 0.0)

    return {
        "total_products": total_products,
        "total_forecast_demand": round(total_forecast_demand, 2),
        "total_procurement_cost": round(total_procurement_cost, 2),
        "high_risk_products": high_risk_products,
        "category_demand": category_demand,
        "category_revenue": category_revenue
    }
