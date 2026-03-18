"""
Smart Procurement Recommendation Engine — converts forecasting insights
into actionable procurement decisions for supermarket managers.

Produces per-product:
  • procurement_quantity   — how much to order
  • procurement_cost       — expected capital requirement
  • urgency                — when to order (Immediate / Soon / Not Needed)
  • recommended_action     — plain-English guidance

🚨 This module is pure computation — no model access, no UI.
"""


def compute_procurement_plan(
    forecast_total: float,
    safety_stock: float,
    reorder_point: float,
    reorder_quantity: float,
    current_stock: float,
    unit_price: float,
    category: str,
) -> dict:
    """
    Compute a procurement recommendation for a single product.

    Args:
        forecast_total:   Total forecasted demand over the horizon.
        safety_stock:     Calculated safety stock buffer.
        reorder_point:    Inventory level that triggers a reorder.
        reorder_quantity: Raw reorder quantity from the Reorder Engine.
        current_stock:    Current on-hand inventory.
        unit_price:       Unit selling / purchase price.
        category:         Canonical category name (Grocery, Household, Hobbies).

    Returns:
        dict with procurement_quantity, procurement_cost, urgency,
        recommended_action.
    """

    # ── 1. Procurement Quantity ───────────────────────────────
    # Ensure minimum economic order size (30 % of forecast)
    minimum_order = max(0.0, forecast_total * 0.3)
    procurement_quantity = max(reorder_quantity, minimum_order)

    # Never negative
    procurement_quantity = max(0.0, procurement_quantity)

    # Round up to whole units
    procurement_quantity = int(round(procurement_quantity + 0.4999))  # ceil-like

    # If stock already exceeds reorder point, no purchase needed
    if current_stock >= reorder_point and reorder_quantity <= 0:
        procurement_quantity = 0

    # ── 2. Procurement Cost ──────────────────────────────────
    safe_price = max(0.0, unit_price) if unit_price else 0.0
    procurement_cost = round(procurement_quantity * safe_price, 2)

    # ── 3. Urgency Classification ────────────────────────────
    if current_stock < safety_stock:
        urgency = "Immediate Reorder"
    elif current_stock < reorder_point:
        urgency = "Reorder Soon"
    else:
        urgency = "Stock Sufficient"

    # ── 4. Recommended Action Text ───────────────────────────
    if urgency == "Immediate Reorder":
        recommended_action = (
            f"Order {procurement_quantity} units immediately to prevent stockout. "
            f"Current stock ({int(current_stock)}) is below safety stock ({int(safety_stock)})."
        )
    elif urgency == "Reorder Soon":
        recommended_action = (
            f"Monitor inventory and reorder {procurement_quantity} units within 2–3 days. "
            f"Stock ({int(current_stock)}) is approaching the reorder point ({int(reorder_point)})."
        )
    else:
        recommended_action = (
            f"No purchase required currently. "
            f"Stock ({int(current_stock)}) comfortably covers forecast demand ({int(forecast_total)})."
        )

    return {
        "procurement_quantity": procurement_quantity,
        "procurement_cost": procurement_cost,
        "urgency": urgency,
        "recommended_action": recommended_action,
    }
