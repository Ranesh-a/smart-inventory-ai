"""
Conversational Retail Analytics Assistant
Maintains context-aware conversation history per product for interactive inventory analysis.
"""

import google.generativeai as genai
import streamlit as st
import os
from typing import Dict, List

class RetailAnalyticsAssistant:
    def __init__(self):
        # Maps product_name -> list of message dicts
        self.conversations = {}

    def _get_api_key(self) -> str:
        if "GEMINI_API_KEY" in st.session_state and st.session_state["GEMINI_API_KEY"]:
            return st.session_state["GEMINI_API_KEY"]
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        return os.environ.get("GEMINI_API_KEY", "")

    def ask_question(self, product_id: str, analytics_data: Dict, user_question: str) -> str:
        """
        Builds a grounded prompt, appends it to the product's conversation history,
        and retrieves the LLM response.
        """
        api_key = self._get_api_key()
        if not api_key:
            return "⚠️ System Unavailable: Gemini API key not configured."

        # Initialize conversation history for this product if new
        if product_id not in self.conversations:
            system_prompt = f"""SYSTEM ROLE:
You are a senior retail inventory analyst advising supermarket managers.

GROUNDING RULES:
• Only use the analytics data provided below.
• Do not invent additional numbers.
• Do not modify the reorder calculation.
• If information is missing, state that it is unavailable.

SYSTEM DECISION LOGIC:
Reorder Point = Forecast Demand + Safety Stock
Reorder Quantity = max(0, Reorder Point - Current Stock)

ANALYTICS DATA:

Product: {analytics_data.get('product_name')}
Category: {analytics_data.get('category')}

Forecast Demand: {analytics_data.get('forecast_total')}
Current Stock: {analytics_data.get('stock')}
Safety Stock: {analytics_data.get('safety_stock')}
Reorder Point: {analytics_data.get('reorder_point')}
Recommended Order Quantity: {analytics_data.get('procurement_quantity')}

Procurement Cost: ₹{analytics_data.get('procurement_cost', 0):.2f}
Risk Level: {analytics_data.get('risk_level')}

Verify that all reasoning uses the analytics data provided.
Do not create new numerical values.
"""
            # Store the initial system state instructions (Using Gemini API roles: 'user' acts as system initializer here, followed by model ack)
            self.conversations[product_id] = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "Understood. I will strictly follow the grounding rules and base all my advice on the provided analytics data."}]}
            ]

        # Append the new user question to the history
        self.conversations[product_id].append(
            {"role": "user", "parts": [{"text": user_question}]}
        )

        # Enforce history limit (Keep the first 2 messages which establish the system role/data, then slice the last 4 user/model turns to make 6 total limit)
        if len(self.conversations[product_id]) > 6:
            # Reconstruct list keeping base context + recent history
            base_context = self.conversations[product_id][:2]
            recent_context = self.conversations[product_id][-4:]
            self.conversations[product_id] = base_context + recent_context

        # Call the LLM
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            response = model.generate_content(self.conversations[product_id])
            
            if response and response.text:
                answer = response.text.strip()
                # Append the model's response to the conversation history
                self.conversations[product_id].append(
                    {"role": "model", "parts": [{"text": answer}]}
                )
                return answer
        except Exception as e:
            # If call fails, remove the newly appended user query to prevent out-of-sync history
            self.conversations[product_id].pop()
            return f"⚠️ Chat processing error: {str(e)}"
            
        return "⚠️ Unexpected error generating response."

# Global singleton available for app.py
assistant = RetailAnalyticsAssistant()
