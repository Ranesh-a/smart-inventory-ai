"""
LLM Decision Advisor
Generates natural language explanations and operational advice
for retail inventory recommendations.
"""

import google.generativeai as genai
import os
import streamlit as st
from typing import Dict

def generate_ai_advice(product_result: Dict) -> str:
    """
    Calls the Gemini LLM to generate a concise business 
    explanation of the supply-chain recommendation.

    Args:
        product_result: Dictionary containing:
            product_name, category, forecast_total, safety_stock,
            reorder_point, procurement_quantity, stock, risk_level, procurement_cost

    Returns:
        str: Generated advice or a safe fallback string on failure.
    """
    
    # 1. Structure the prompt
    prompt = f"""You are an AI retail inventory advisor helping a supermarket manager.
Analyze the following inventory analytics and explain the decision.

Product: {product_result.get('product_name', 'Unknown')}
Category: {product_result.get('category', 'Unknown')}

Forecast Demand: {product_result.get('forecast_total', 0)}
Current Stock: {product_result.get('stock', 0)}

Safety Stock: {product_result.get('safety_stock', 0)}
Reorder Point: {product_result.get('reorder_point', 0)}
Recommended Order Quantity: {product_result.get('procurement_quantity', 0)}

Risk Level: {product_result.get('risk_level', 'Unknown')}
Procurement Cost: ${product_result.get('procurement_cost', 0):.2f}

Explain:
1. Why the reorder decision was made
2. How the system calculated the recommendation
3. What operational action the store manager should take
4. Any risks or opportunities related to this inventory decision

Provide a clear business explanation in 4-6 sentences.
"""

    # 2. Extract API Key safely
    api_key = None
    
    # Try st.session_state first (from app.py sidebar)
    if "GEMINI_API_KEY" in st.session_state and st.session_state["GEMINI_API_KEY"]:
        api_key = st.session_state["GEMINI_API_KEY"]
    # Fallback to secrets.toml if available
    elif "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    # Fallback to environment variables
    elif os.environ.get("GEMINI_API_KEY"):
         api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return "⚠️ LLM Advisor Unavailable: Gemini API key not configured. Please enter your API key in the sidebar."

    # 3. Call Gemini
    try:
        genai.configure(api_key=api_key)
        
        # Use simple gemini-1.5-flash as the fast reliable default
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
            
    except Exception as e:
        return f"⚠️ LLM Generation Error: {str(e)}"
        
    return "⚠️ System error generating AI advice."
