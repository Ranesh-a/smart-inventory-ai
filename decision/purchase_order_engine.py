"""
Supplier Purchase Order Engine — generates purchase order drafts
when inventory falls below the reorder point.

Maps each category to a default supplier with estimated delivery times.
"""

from typing import Dict

# ─────────────────────────────────────────────────────────────
# Supplier Configuration
# ─────────────────────────────────────────────────────────────

SUPPLIERS = {
    "Grocery":   {"name": "FreshMart Food Distributor",   "delivery_days": 2},
    "Household": {"name": "CleanHome Supplies Ltd",       "delivery_days": 3},
    "Hobbies":   {"name": "FunPlay Retail Supplier",      "delivery_days": 4},
}

DEFAULT_SUPPLIER = {"name": "General Wholesale Co.", "delivery_days": 5}


# ─────────────────────────────────────────────────────────────
# Purchase Order Generation
# ─────────────────────────────────────────────────────────────

def generate_purchase_order(
    product_name: str,
    category: str,
    reorder_quantity: int,
    unit_price: float,
) -> Dict:
    """
    Generate a supplier purchase order draft for a single product.

    Args:
        product_name:      Name of the product.
        category:          Product category (Grocery / Household / Hobbies).
        reorder_quantity:   Number of units to order.
        unit_price:        Cost per unit from supplier.

    Returns:
        Dict with supplier, product, quantity, unit_price, total_cost,
        delivery_days, and a formatted order summary.
    """
    supplier_info = SUPPLIERS.get(category, DEFAULT_SUPPLIER)
    total_cost = round(reorder_quantity * unit_price, 2)

    order = {
        "supplier":       supplier_info["name"],
        "product":        product_name,
        "category":       category,
        "quantity":        int(reorder_quantity),
        "unit_price":     round(unit_price, 2),
        "total_cost":     total_cost,
        "delivery_days":  supplier_info["delivery_days"],
    }

    # Human-readable summary
    order["summary"] = (
        f"Order {reorder_quantity} units of {product_name} "
        f"from {supplier_info['name']} "
        f"at ${unit_price:.2f}/unit (Total: ${total_cost:,.2f}). "
        f"Expected delivery: {supplier_info['delivery_days']} day(s)."
    )

    return order
