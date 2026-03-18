"""
🛒 AI Supermarket Decision Intelligence — Model Testing Console
Clean decision support interface for the ML forecasting pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.ai_service import ai_service
from ai.conversation_assistant import assistant
from services.forecast_service import ForecastService, FEATURE_COLUMNS
from routing.category_router import process_uploaded_dataset
from analytics.operations_dashboard import compute_operations_summary
from analytics.decision_brief import generate_decision_brief
from advisor.business_advisor import generate_business_insights
from alerts.email_alerts import check_and_send_alerts

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Supermarket Decision Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEQUENCE_LENGTH = 30
MV_FEATURE_COLUMNS = ["sales", "price", "weekday", "month", "is_weekend", "is_event_day"]
ALLOWED_CATEGORIES_UPLOAD = ["grocery", "food", "foods", "hobbies", "hobby", "household"]


# ─────────────────────────────────────────────────────────────
# ForecastService — cached singleton
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def _get_forecast_service():
    return ForecastService(base_dir=".")

forecast_svc = _get_forecast_service()


# ─────────────────────────────────────────────────────────────
# Upload Validation
# ─────────────────────────────────────────────────────────────

def _validate_uploaded_data(df: pd.DataFrame) -> dict:
    """Validate an uploaded multi-product sales DataFrame."""
    result = {"valid": False, "level": "error", "title": "", "detail": "",
              "tips": [], "multivariate": False, "products": []}

    if "sales" not in df.columns:
        found = ", ".join(df.columns[:10])
        result.update(
            title="🔴 STRUCTURE ERROR",
            detail=f"Missing required **sales** column. Columns: {found}",
            tips=["Add a column named exactly **sales**."],
        )
        return result

    mv_cols_present = all(c in df.columns for c in MV_FEATURE_COLUMNS)
    has_category = "category" in df.columns
    has_product = "product_name" in df.columns
    is_multivariate = mv_cols_present and has_category and has_product

    if not is_multivariate:
        result.update(
            title="🔴 FORMAT ERROR",
            detail=(
                "Requires multivariate dataset with columns: `category`, `product_name`, "
                "`date`, `sales`, `price`, `stock`, `weekday`, `month`, `is_weekend`, `is_event_day`."
            ),
        )
        return result

    cats = df["category"].dropna().str.strip().str.lower().unique().tolist()
    invalid_cats = [c for c in cats if c not in ALLOWED_CATEGORIES_UPLOAD]
    if invalid_cats:
        result.update(
            title="🔴 CATEGORY ERROR",
            detail=f"Unknown category: **{', '.join(invalid_cats)}**. Allowed: Grocery, Hobbies, Household.",
        )
        return result

    if not pd.api.types.is_numeric_dtype(df["sales"]):
        try:
            pd.to_numeric(df["sales"], errors="raise")
        except (ValueError, TypeError):
            result.update(title="🔴 DATA QUALITY ERROR", detail="Non-numeric values in **sales**.")
            return result

    n_missing = int(df["sales"].isna().sum())
    if n_missing > 0:
        result.update(title="🔴 DATA QUALITY ERROR", detail=f"**{n_missing}** missing sales values.")
        return result

    sales_float = df["sales"].astype(float)
    if int((sales_float < 0).sum()) > 0:
        result.update(title="🔴 DATA QUALITY ERROR", detail="Negative sales values found.")
        return result

    products_list = []
    grouped = df.groupby(["category", "product_name"])
    short_products = []
    for (cat, prod), grp in grouped:
        if len(grp) < SEQUENCE_LENGTH:
            short_products.append(f"{prod} ({cat}, {len(grp)} rows)")
        else:
            products_list.append((cat, prod))

    if not products_list:
        result.update(title="🔴 INSUFFICIENT DATA", detail=f"No product has ≥ {SEQUENCE_LENGTH} rows.")
        return result
    if short_products:
        result["tips"].append(f"🟡 Skipped: {', '.join(short_products[:5])}")

    n_rows = len(df)
    n_products = len(products_list)
    n_cats = len(set(c for c, _ in products_list))
    result.update(
        valid=True, level="success", title="🟢 VALID DATASET",
        detail=f"**{n_rows} rows** • **{n_products} product(s)** across **{n_cats} category(ies)**",
        multivariate=True, products=products_list,
    )
    return result





# ─────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────

def main():

    # ══════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════
    st.sidebar.header("🛒 Decision Intelligence Panel")

    # System Status
    st.sidebar.subheader("📡 System Status")
    loaded_cats = list(forecast_svc.models.keys()) if hasattr(forecast_svc, "models") else []
    if loaded_cats:
        st.sidebar.success(f"✅ Models: {', '.join(loaded_cats)}")
    else:
        st.sidebar.warning("⚠️ No models loaded.")
    st.sidebar.caption("Pipeline: **Active**")

    st.sidebar.divider()

    # Upload Master Dataset
    st.sidebar.subheader("📂 Master Dataset")
    st.sidebar.caption(
        "Columns: `category`, `product_name`, `date`, `sales`, "
        "`price`, `stock`, `weekday`, `month`, `is_weekend`, `is_event_day`."
    )
    uploaded_file = st.sidebar.file_uploader("Upload Master CSV", type=["csv", "xlsx", "xls"], key="master_file")

    default_stock = st.sidebar.number_input(
        "Default Stock Level", min_value=0, value=100, step=1,
        help="Used if the dataset is missing a 'stock' column."
    )

    st.sidebar.divider()

    # Parameters
    st.sidebar.subheader("⚙️ Parameters")
    forecast_days = st.sidebar.slider("📅 Forecast Horizon (days)", 1, 30, 7)
    demand_multiplier_percent = st.sidebar.slider("📉 What-If Scenario: Demand (%)", -50, 50, 0)

    with st.sidebar.expander("💰 Economic Settings", expanded=False):
        holding_rate = st.number_input("Holding Cost (₹/unit/day)", value=0.5, step=0.1, min_value=0.0)
        stockout_penalty = st.number_input("Stockout Penalty (₹/unit)", value=5.0, step=0.5, min_value=0.0)

    st.sidebar.divider()

    # Email Alert Settings
    st.sidebar.subheader("📧 Email Alerts")
    email_enabled = st.sidebar.checkbox("Enable stockout alerts", value=False)
    recipient_email = ""
    if email_enabled:
        recipient_email = st.sidebar.text_input("Enter Manager / Seller Email", placeholder="manager@store.com")

    # ══════════════════════════════════════════════════════════
    # MAIN PAGE
    # ══════════════════════════════════════════════════════════

    st.title("🛒 AI Supermarket Decision Intelligence")
    st.caption("Upload a sales dataset and optional inventory file to run the full decision pipeline")
    st.divider()

    if uploaded_file is None:
        
        # Validation checks on startup
        import os
        if not os.getenv("GMAIL_SENDER") or not os.getenv("GMAIL_PASSWORD"):
            st.error("⚠️ Email credentials missing in .env file (GMAIL_SENDER, GMAIL_PASSWORD). Email alerts will fail.")
            
        st.info(
            "👈 **Upload a Master CSV** from the sidebar to begin.\n\n"
            "This file should contain both historical sales and current stock levels.\n\n"
            "**Supported categories:** Grocery, Household, Hobbies"
        )
        return

    # ── Read sales file ───────────────────────────────────────
    custom_df = None
    try:
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            custom_df = pd.read_excel(uploaded_file)
        else:
            custom_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"🔴 **File Parsing Error:** {e}")
        return

    if custom_df is None:
        return

    custom_df.columns = [c.strip().lower() for c in custom_df.columns]

    # ── Extract Stock Map ─────────────────────────────────────
    stock_map = None
    if "stock" in custom_df.columns:
        custom_df["stock"] = pd.to_numeric(custom_df["stock"], errors="coerce").fillna(0).astype(int)
        stock_map = custom_df.groupby("product_name")["stock"].last().to_dict()

    # ── Validate ──────────────────────────────────────────────
    vr = _validate_uploaded_data(custom_df)
    if not vr["valid"]:
        feedback = f"**{vr['title']}**\n\n{vr['detail']}"
        if vr["tips"]:
            feedback += "\n\n" + "\n".join(f"- {tip}" for tip in vr["tips"])
        st.error(feedback)
        return

    st.success(f"**{vr['title']}** — {vr['detail']}")
    for tip in vr.get("tips", []):
        st.caption(tip)

    products = vr["products"]
    inv_label = f"Inventory: **{len(stock_map)} products**" if stock_map else f"Default stock: **{default_stock}**"
    st.caption(
        f"📂 **{len(custom_df)} rows** | **{len(products)} product(s)** | "
        f"Forecast: **{forecast_days}d** | {inv_label}"
    )

    # ══════════════════════════════════════════════════════════
    # RUN PIPELINE
    # ══════════════════════════════════════════════════════════

    with st.spinner("🔮 Running forecasting pipeline across all products…"):
        pipeline_results = process_uploaded_dataset(
            df=custom_df,
            forecast_service=forecast_svc,
            forecast_days=forecast_days,
            current_stock=default_stock,
            holding_rate=holding_rate,
            stockout_penalty=stockout_penalty,
            stock_map=stock_map,
            demand_multiplier_percent=demand_multiplier_percent,
        )

    # ══════════════════════════════════════════════════════════
    # PRE-COMPUTE OPERATIONS SUMMARY (Used for UI & Email triggers)
    # ══════════════════════════════════════════════════════════

    ops_summary = compute_operations_summary(pipeline_results)
    advisor_insights = generate_business_insights(pipeline_results, ops_summary)

    # ══════════════════════════════════════════════════════════
    # EMAIL ALERTS (auto-triggered)
    # ══════════════════════════════════════════════════════════

    if email_enabled and recipient_email:
        force_test = st.checkbox("🛠️ Force Test (Bypass stock evaluation)")
        
        st.markdown("---")
        st.subheader("🛠️ Debug Tracer: Email Execution Path")
        
        # Identify trigger condition directly from UI's summary list
        high_risk_ui_list = ops_summary.get("high_risk_products", [])
        
        st.info(
            f"**UI Trigger State:**\n"
            f"- `email_enabled`: {email_enabled}\n"
            f"- `recipient_email`: '{recipient_email}'\n"
            f"- `Condition Met?`: {len(high_risk_ui_list) > 0}\n"
            f"- `High Risk Products Triggering Alert`: {high_risk_ui_list if high_risk_ui_list else 'None'}"
        )

        st.markdown("**📧 Manual Dispatch Ready**")
        if st.button("📤 Dispatch Email Alerts Now"):
            st.write("Debug: Button clicked, entering spinner...")
            with st.spinner("📧 Preparing and sending stockout alerts..."):
                st.write(f"Debug: Passing {len(pipeline_results)} pipeline results to backend (Force Test: {force_test})...")
                alert_messages = check_and_send_alerts(
                    product_results=pipeline_results,
                    recipient_email=recipient_email,
                    high_risk_list=high_risk_ui_list,
                    force_test=force_test,
                )
                st.write(f"Debug: Backend returned {len(alert_messages)} messages.")
                
                if alert_messages:
                    if any("⚠️" in msg for msg in alert_messages):
                        for msg in alert_messages:
                            if "⚠️" in msg:
                                st.error(msg)
                    else:
                        st.success(f"✅ **{len(alert_messages)} stockout alert(s) sent** to {recipient_email}")
                        with st.expander("📧 Sent Alert Details", expanded=True):
                            for msg in alert_messages:
                                st.caption(msg)
                else:
                    st.warning("⚠️ Debug: `alert_messages` list was empty. Alert conditions not met or backend aborted silently.")
    else:
        st.warning(f"Debug: Execution blocked. `email_enabled`: {email_enabled}, `recipient_email`: '{recipient_email}'")

    # ══════════════════════════════════════════════════════════
    # AI DECISION BRIEF
    # ══════════════════════════════════════════════════════════
    
    st.subheader("🧠 AI Retail Operations Brief")
    
    brief = generate_decision_brief(ops_summary, pipeline_results)
    st.info(brief)
    
    st.divider()

    # ══════════════════════════════════════════════════════════
    # 1. AI BUSINESS ADVISOR
    # ══════════════════════════════════════════════════════════

    st.header("🧠 AI Business Advisor")
    with st.expander("📋 Executive Summary & Insights", expanded=True):
        st.markdown("**📌 Executive Summary**")
        st.success(advisor_insights["executive_summary"])

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**📈 Demand Outlook**")
            st.info(advisor_insights["demand_insight"])
            st.markdown("**🛒 Procurement Guidance**")
            st.info(advisor_insights["procurement_insight"])
        with a2:
            st.markdown("**📦 Inventory Health**")
            st.info(advisor_insights["inventory_insight"])
            st.markdown("**📊 Category Performance**")
            st.info(advisor_insights["category_insight"])

    st.divider()

    # ══════════════════════════════════════════════════════════
    # 2. RETAIL OPERATIONS OVERVIEW
    # ══════════════════════════════════════════════════════════

    st.header("🏢 Retail Operations Overview")

    o1, o2, o3 = st.columns(3)
    with o1:
        st.metric("📦 Products Analyzed", ops_summary["total_products"])
    with o2:
        st.metric("📈 Forecast Demand", f"{ops_summary['total_forecast_demand']:,.1f} units")
    with o3:
        st.metric("💰 Procurement Budget", f"₹{ops_summary['total_procurement_cost']:,.2f}")

    st.subheader("📊 Category Analytics")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Demand by Category**")
        demand_df = pd.DataFrame(
            list(ops_summary["category_demand"].items()),
            columns=["Category", "Forecast Demand"],
        ).set_index("Category")
        st.bar_chart(demand_df)
    with ch2:
        st.markdown("**Projected Revenue by Category**")
        rev_df = pd.DataFrame(
            list(ops_summary["category_revenue"].items()),
            columns=["Category", "Revenue (₹)"],
        ).set_index("Category")
        st.bar_chart(rev_df)

    st.subheader("⚠️ Risk Alerts")
    if ops_summary["high_risk_products"]:
        st.error("**High Risk Products:** " + ", ".join(ops_summary["high_risk_products"]))
    else:
        st.success("✅ No high-risk products detected.")

    st.subheader("🧠 Operational Insights")
    op_insights = []
    if ops_summary["total_procurement_cost"] > 5000:
        op_insights.append("High procurement requirement expected this cycle.")
    if len(ops_summary["high_risk_products"]) >= 3:
        op_insights.append("Inventory shortage risk increasing. Prioritize High Risk items.")
    if not op_insights:
        op_insights.append("Operations are stable. Standard restocking recommended.")
    for oi in op_insights:
        st.info(f"💡 {oi}")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # 3. PRODUCT SUMMARY TABLE
    # ══════════════════════════════════════════════════════════

    st.header("📑 Decision Summary Table")

    summary_rows = []
    for r in pipeline_results:
        if r.get("error"):
            summary_rows.append({
                "Product": r["product_name"], "Category": r["category"],
                "Forecast": "❌ Error", "Stock": r["stock"],
                "Risk": "—", "Action": r["error"][:40],
                "Order Qty": "—", "Cost": "—", "Urgency": "—",
            })
        else:
            summary_rows.append({
                "Product": r["product_name"], "Category": r["category"],
                "Forecast": round(r["total_demand"], 1), "Stock": r["stock"],
                "Risk": r["risk_level"], "Action": r["action"],
                "Order Qty": r["procurement_quantity"],
                "Cost": f"₹{r['procurement_cost']:,.2f}",
                "Urgency": r["urgency"],
            })

    summary_df = pd.DataFrame(summary_rows)

    # Sort by risk level descending
    risk_order = {"High": 0, "Medium": 1, "Low": 2, "—": 3}
    summary_df["_sort"] = summary_df["Risk"].map(risk_order).fillna(3)
    summary_df = summary_df.sort_values("_sort").drop(columns=["_sort"])

    def _risk_colour(val):
        if val == "High":
            return "background-color: #ff4b4b22; color: #ff4b4b;"
        elif val == "Medium":
            return "background-color: #ffa50022; color: #ffa500;"
        elif val == "Low":
            return "background-color: #00cc6622; color: #00cc66;"
        return ""

    st.dataframe(
        summary_df.style.applymap(_risk_colour, subset=["Risk"]),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ══════════════════════════════════════════════════════════
    # 4. PRODUCT DETAIL VIEWS
    # ══════════════════════════════════════════════════════════

    st.header("📦 Product Detail Views")

    for idx, r in enumerate(pipeline_results):
        if r.get("error"):
            with st.expander(f"❌ {r['product_name']} ({r['category']}) — Error", expanded=False):
                st.error(f"**Processing failed:** {r['error']}")
            continue

        preds = np.array(r["forecast"])
        lower = r["lower_bound"]
        upper = r["upper_bound"]
        risk_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(r["risk_level"], "⚪")

        label = (
            f"{risk_icon} {r['product_name']} ({r['category']}) — "
            f"{r['risk_level']} Risk • {r['action']}"
        )
        with st.expander(label, expanded=(idx == 0)):

            # Key Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("📈 Forecast Demand", f"{r['forecast_total']} units")
            with m2:
                st.metric("📊 Safety Stock", f"{r['safety_stock']} units")
            with m3:
                st.metric("📦 Current Stock", f"{r['stock']} units")
            with m4:
                st.metric("📍 Reorder Point", f"{r['reorder_point']} units")

            # Forecast Chart
            chart_df = pd.DataFrame({
                "Day": list(range(1, len(preds) + 1)),
                "Forecast": preds,
            })
            if lower is not None and len(lower) > 0 and upper is not None and len(upper) > 0:
                chart_df["Lower Bound"] = lower
                chart_df["Upper Bound"] = upper
            st.line_chart(chart_df.set_index("Day"))

            # Cost Breakdown
            st.markdown("**💰 Cost Analysis**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("💵 Revenue", f"₹{r['revenue']:,.2f}")
            with c2:
                st.metric("📦 Holding Cost", f"₹{r['holding_cost']:,.2f}")
            with c3:
                st.metric("🚨 Shortage Cost", f"₹{r['shortage_cost']:,.2f}")
            with c4:
                st.metric("📉 Total Cost", f"₹{r['total_cost']:,.2f}")

            # Procurement Decision
            st.markdown("**🛒 Procurement Decision**")
            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric("🛒 Suggested Order", f"{r['procurement_quantity']} units")
            with p2:
                st.metric("💰 Procurement Cost", f"₹{r['procurement_cost']:,.2f}")
            with p3:
                urgency = r.get("urgency", "—")
                urgency_icon = {
                    "Immediate Reorder": "🔴",
                    "Reorder Soon": "🟡",
                    "Stock Sufficient": "🟢",
                }.get(urgency, "⚪")
                st.metric("⚠️ Urgency", f"{urgency_icon} {urgency}")

            st.info(f"📝 **{r.get('recommended_action', '')}**")

            # 🧠 AI Decision Advisor Component
            st.markdown("**🧠 AI Decision Advisor**")
            
            cache_key = ai_service.get_cache_key(r)
            if cache_key in ai_service.cache:
                st.info(ai_service.cache[cache_key])
            else:
                # Use unique key per product for the button state
                btn_key = f"ai_adv_btn_{idx}_{r['product_name']}"
                
                if st.button("Explain Decision with AI", key=btn_key):
                    with st.spinner("🧠 AI analyzing decision..."):
                        try:
                            advice = ai_service.generate_advice(r)
                            if "⚠️" in advice:
                                st.error(advice)
                            else:
                                st.success(advice)
                        except Exception as e:
                            st.error(f"⚠️ Unexpected Error during analysis: {str(e)}")
            # 💬 Conversational AI Assistant Component
            st.divider()
            st.subheader("💬 Ask the AI Inventory Advisor")
            
            # Setup session state for this product's chat history
            chat_history_key = f"chat_history_{r['product_name']}"
            if chat_history_key not in st.session_state:
                st.session_state[chat_history_key] = []
                
            # Quick Prompts
            # Chat Input Form
            user_question = st.chat_input("Ask a question about this product...", key=f"chat_input_{idx}")
            
            if user_question:
                # 1. Display user message 
                st.session_state[chat_history_key].append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                # 2. Call Assistant
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        response = assistant.ask_question(r['product_name'], r, user_question)
                        st.markdown(response)
                        
                # 3. Save to state
                st.session_state[chat_history_key].append({"role": "assistant", "content": response})

            # Supplier Order Draft
            po = r.get("purchase_order")
            if po and po.get("quantity", 0) > 0:
                st.markdown("**📋 Supplier Order Draft**")
                so1, so2, so3, so4 = st.columns(4)
                with so1:
                    st.metric("🏢 Supplier", po["supplier"])
                with so2:
                    st.metric("📦 Order Qty", f"{po['quantity']} units")
                with so3:
                    st.metric("💲 Unit Price", f"₹{po['unit_price']:.2f}")
                with so4:
                    st.metric("🚚 Delivery", f"{po['delivery_days']} day(s)")

                st.caption(f"💰 Total Order Cost: **₹{po['total_cost']:,.2f}**")

                # Download Purchase Order as CSV
                po_df = pd.DataFrame([{
                    "Supplier": po["supplier"],
                    "Product": po["product"],
                    "Category": po["category"],
                    "Quantity": po["quantity"],
                    "Unit Price (₹)": po["unit_price"],
                    "Total Cost (₹)": po["total_cost"],
                    "Delivery Days": po["delivery_days"],
                }])
                csv_data = po_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Purchase Order",
                    data=csv_data,
                    file_name=f"purchase_order_{po['product'].replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"po_download_{idx}",
                )


if __name__ == "__main__":
    main()
