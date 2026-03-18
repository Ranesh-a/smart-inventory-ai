"""
AI Decision Brief Generator
Produces a concise natural language summary of operational insights for supermarket managers.
"""

from typing import Dict, List


def generate_decision_brief(operations_summary: Dict, product_results: List[Dict]) -> str:
    """
    Generate a short, human-readable executive summary based on aggregated analytics results.

    Args:
        operations_summary: Dict containing total demand, costs, risks, and category metrics.
        product_results: List of dicts containing per-product analytics results.

    Returns:
        str: A concatenated narrative briefing.
    """
    # 1. Extract from operations_summary
    total_forecast_demand = operations_summary.get("total_forecast_demand", 0)
    total_procurement_cost = operations_summary.get("total_procurement_cost", 0)
    
    cat_demand = operations_summary.get("category_demand", {})
    top_category = ""
    if cat_demand:
        top_category = max(cat_demand, key=cat_demand.get)
    else:
        top_category = "General"

    # 2. Extract from product_results
    valid_results = [r for r in product_results if not r.get("error")]
    
    # Sort for highest demand products
    sorted_by_demand = sorted(valid_results, key=lambda x: x.get("forecast_total", 0), reverse=True)
    
    # Top 3 products
    top_3_products = [r["product_name"] for r in sorted_by_demand[:3]]
    
    # High risk products
    high_risk_list = [r["product_name"] for r in valid_results if r.get("risk_level") == "High"]
    
    if high_risk_list:
        if len(high_risk_list) <= 3:
             risk_products_str = ", ".join(high_risk_list)
        else:
             risk_products_str = f"{', '.join(high_risk_list[:3])} and {len(high_risk_list) - 3} others"
        risk_sentence = f"High-risk products identified: {risk_products_str}."
    else:
        risk_sentence = "No high-risk products identified."


    # 3. Build Template
    summary = f"""Total expected demand across all categories is {total_forecast_demand:,.0f} units.

The {top_category} category shows the strongest demand activity.

A procurement budget of ₹{total_procurement_cost:,.2f} is required to maintain safe stock levels.

{risk_sentence}

**Recommended Actions:**
• Reorder high-risk products immediately
• Monitor inventory for medium-risk items
• Maintain current stock levels for low-risk items"""

    return summary
