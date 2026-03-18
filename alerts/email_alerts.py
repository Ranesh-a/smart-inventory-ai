"""
Email Alert System — sends automated stockout risk alerts
when a product's current stock falls below its safety stock.

🚨 Uses Gmail SMTP. Requires sender email + app password.
"""

import os
import smtplib
from email.message import EmailMessage
from typing import List, Dict, Optional


def check_and_send_alerts(
    product_results: List[Dict],
    recipient_email: str,
    high_risk_list: List[str],
    force_test: bool = False,
) -> List[str]:
    """
    Check all product results for stockout risk and send a consolidated
    email alert if any product has current_stock < safety_stock.

    Args:
        product_results:  List of per-product result dicts from the pipeline.
        recipient_email:  Email address to receive alerts.

    Returns:
        List of alert messages generated (empty if no alerts).
    """
    alerts = []

    if force_test:
        print("🛠️ DEBUG: Force Test enabled. Bypassing stock evaluation.")
        alerts.append({
            "product": "FORCE_TEST_ITEM",
            "category": "TEST",
            "current_stock": 0,
            "safety_stock": 100,
            "forecast_demand": 50,
            "reorder_quantity": 100,
            "urgency": "Immediate Reorder",
        })
    else:
        for r in product_results:
            if r.get("error") or r.get("product_name") not in high_risk_list:
                continue

            stock = float(r.get("stock", 0))
            safety = float(r.get("safety_stock", 0))

            alerts.append({
                "product": r["product_name"],
                "category": r.get("category", "—"),
                "current_stock": stock,
                "safety_stock": safety,
                "forecast_demand": r.get("forecast_total", 0),
                "reorder_quantity": r.get("procurement_quantity", r.get("reorder_quantity", 0)),
                "urgency": r.get("urgency", "—"),
            })

    if not alerts:
        return []

    # Build email body
    subject = f"🚨 Stockout Alert — {len(alerts)} product(s) below safety stock"

    body_lines = [
        "STOCKOUT RISK ALERT",
        "=" * 40,
        f"{len(alerts)} product(s) have stock below safety level.\n",
    ]

    for a in alerts:
        body_lines.append(f"Product:          {a['product']} ({a['category']})")
        body_lines.append(f"Current Stock:    {a['current_stock']} units")
        body_lines.append(f"Safety Stock:     {a['safety_stock']} units")
        body_lines.append(f"Forecast Demand:  {a['forecast_demand']} units")
        body_lines.append(f"Recommended Order:{a['reorder_quantity']} units")
        body_lines.append(f"Urgency:          {a['urgency']}")
        body_lines.append("-" * 40)

    body_lines.append("\nThis is an automated alert from the AI Supermarket Decision Intelligence system.")
    body = "\n".join(body_lines)

    # Send email
    try:
        sender_email = os.getenv("GMAIL_SENDER")
        app_password = os.getenv("GMAIL_PASSWORD")
        
        if not sender_email or not app_password:
            return ["⚠️ Email send failed: Missing GMAIL_SENDER or GMAIL_PASSWORD in environment."]

        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
    except smtplib.SMTPAuthenticationError:
        return ["⚠️ Email send failed: SMTP Authentication Error. Check GMAIL_PASSWORD."]
    except Exception as e:
        return [f"⚠️ Email send failed: {str(e)}"]

    alert_messages = [
        f"🚨 {a['product']}: Stock ({a['current_stock']}) < Safety ({a['safety_stock']}) → Order {a['reorder_quantity']} units"
        for a in alerts
    ]
    return alert_messages
