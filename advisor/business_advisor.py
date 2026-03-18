"""
AI Business Advisor — generates natural-language business insights
from forecasting, inventory, and procurement analytics.

Produces five insight strings per analysis run:
  • demand_insight
  • inventory_insight
  • procurement_insight
  • category_insight
  • executive_summary

🚨 Pure text generation — no model access, no UI.
"""

from typing import Dict, List


def generate_business_insights(
    product_results: List[Dict],
    operations_summary: Dict,
) -> Dict[str, str]:
    """
    Convert analytics results into clear business insights.

    Args:
        product_results:    Per-product result dicts from the routing pipeline.
        operations_summary: Aggregated metrics from compute_operations_summary.

    Returns:
        dict with demand_insight, inventory_insight, procurement_insight,
        category_insight, executive_summary.
    """
    valid = [r for r in product_results if not r.get("error")]
    total_demand = operations_summary.get("total_forecast_demand", 0)
    total_cost = operations_summary.get("total_procurement_cost", 0)
    high_risk = operations_summary.get("high_risk_products", [])
    cat_demand = operations_summary.get("category_demand", {})
    n_products = operations_summary.get("total_products", 0)

    # ── 1. Demand Insight ────────────────────────────────────
    avg_demand = total_demand / n_products if n_products else 0

    # Identify products with strongest demand
    top_demand = sorted(valid, key=lambda r: r.get("forecast_total", 0), reverse=True)[:3]
    top_names = [r["product_name"] for r in top_demand]

    if avg_demand > 50:
        demand_insight = (
            f"Demand is expected to be **strong** over the forecast window, "
            f"averaging **{avg_demand:.0f} units** per product. "
            f"Top movers include **{', '.join(top_names)}**, indicating robust purchasing momentum."
        )
    elif avg_demand > 20:
        demand_insight = (
            f"Demand levels are **moderate** across the portfolio, "
            f"averaging **{avg_demand:.0f} units** per product. "
            f"No unusual spikes detected."
        )
    else:
        demand_insight = (
            f"Demand remains **low to stable** across most products, "
            f"averaging **{avg_demand:.0f} units** per product. "
            f"Consider promotional activity to stimulate sales."
        )

    # ── 2. Inventory Insight ─────────────────────────────────
    n_high = len(high_risk)
    immediate_items = [
        r["product_name"] for r in valid
        if r.get("urgency") == "Immediate Reorder"
    ]

    if n_high >= 3:
        inventory_insight = (
            f"**{n_high} products** show elevated stockout risk. "
            f"Immediate inventory review is recommended to prevent supply disruptions. "
            f"Priority items: **{', '.join(high_risk[:5])}**."
        )
    elif n_high >= 1:
        inventory_insight = (
            f"**{n_high} product(s)** flagged as high risk: **{', '.join(high_risk)}**. "
            f"Monitor closely and consider expedited restocking."
        )
    else:
        inventory_insight = (
            "Inventory levels appear **stable** with low risk of stock shortages "
            "in the upcoming forecast window. No immediate action required."
        )

    # ── 3. Procurement Insight ───────────────────────────────
    n_orders = sum(1 for r in valid if r.get("procurement_quantity", 0) > 0)

    if total_cost > 10000:
        procurement_insight = (
            f"The system recommends procurement for **{n_orders} item(s)** "
            f"with an estimated total cost of **${total_cost:,.2f}**. "
            f"This represents a significant capital outlay — review budget allocation before placing orders."
        )
    elif total_cost > 2000:
        procurement_insight = (
            f"Procurement is recommended for **{n_orders} item(s)**, "
            f"totalling **${total_cost:,.2f}**. "
            f"This is within normal operational range."
        )
    elif total_cost > 0:
        procurement_insight = (
            f"Procurement requirements are **moderate** — **{n_orders} item(s)** "
            f"totalling **${total_cost:,.2f}**. "
            f"Current stock levels can sustain expected demand in the short term."
        )
    else:
        procurement_insight = (
            "No procurement is currently required. "
            "Stock levels are sufficient to cover forecast demand across all categories."
        )

    # ── 4. Category Insight ──────────────────────────────────
    if cat_demand:
        dominant_cat = max(cat_demand, key=cat_demand.get)
        dominant_pct = (
            (cat_demand[dominant_cat] / total_demand * 100) if total_demand > 0 else 0
        )
        category_insight = (
            f"**{dominant_cat}** accounts for **{dominant_pct:.0f}%** of total forecast demand, "
            f"reflecting strong turnover in this segment. "
        )
        # Check for volatility in other categories
        other_cats = {k: v for k, v in cat_demand.items() if k != dominant_cat and v > 0}
        if other_cats:
            smallest_cat = min(other_cats, key=other_cats.get)
            category_insight += (
                f"**{smallest_cat}** shows comparatively lower demand and may benefit from targeted promotions."
            )
    else:
        category_insight = "Category-level demand data is unavailable for this analysis."

    # ── 5. Executive Summary ─────────────────────────────────
    summary_parts = []

    if n_high == 0:
        summary_parts.append("Overall store operations appear **stable**.")
    else:
        summary_parts.append(
            f"**{n_high} product(s)** show rising demand and may require immediate replenishment."
        )

    if total_cost > 0:
        summary_parts.append(
            f"Procurement planning should prioritize high-risk items "
            f"(estimated budget: **${total_cost:,.2f}**) to avoid stockouts "
            f"while maintaining balanced inventory costs."
        )
    else:
        summary_parts.append(
            "No immediate procurement action is needed. Continue standard monitoring."
        )

    executive_summary = " ".join(summary_parts)

    return {
        "demand_insight": demand_insight,
        "inventory_insight": inventory_insight,
        "procurement_insight": procurement_insight,
        "category_insight": category_insight,
        "executive_summary": executive_summary,
    }
