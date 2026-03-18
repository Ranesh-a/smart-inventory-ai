"""
AI Service
Handles background generation and caching of LLM advice.
"""

import json
import os
import hashlib
import threading
import google.generativeai as genai
import streamlit as st
from typing import Dict

class AIService:
    def __init__(self, cache_file="cache/ai_explanations.json"):
        self.cache_file = cache_file
        self.cache = {}
        self.lock = threading.Lock()
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
                
    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _get_api_key(self):
        if "GEMINI_API_KEY" in st.session_state and st.session_state["GEMINI_API_KEY"]:
            return st.session_state["GEMINI_API_KEY"]
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        return os.environ.get("GEMINI_API_KEY")

    def _create_summary(self, product_result: Dict) -> Dict:
        """Only send summary metrics to limit tokens."""
        return {
            "product_name": product_result.get("product_name"),
            "category": product_result.get("category"),
            "forecast_total": product_result.get("forecast_total"),
            "safety_stock": product_result.get("safety_stock"),
            "reorder_point": product_result.get("reorder_point"),
            "reorder_quantity": product_result.get("procurement_quantity"),
            "current_stock": product_result.get("stock"),
            "risk_level": product_result.get("risk_level"),
            "procurement_cost": product_result.get("procurement_cost")
        }
        
    def get_cache_key(self, product_result: Dict) -> str:
        """Returns stable cache key based on metrics."""
        summary = self._create_summary(product_result)
        return hashlib.md5(str(summary).encode()).hexdigest()

    def generate_advice(self, product_result: Dict) -> str:
        """Generates AI advice using LLM and caches the output."""
        summary = self._create_summary(product_result)
        key = self.get_cache_key(product_result)

        with self.lock:
            if key in self.cache:
                return self.cache[key]

        api_key = self._get_api_key()
        if not api_key:
            return "⚠️ LLM Advisor Unavailable: Gemini API key not configured."

        system_prompt = """You are an AI retail inventory advisor helping a supermarket manager.

GROUNDING RULES:
• Only use the numerical values provided in the data section.
• Do not invent additional numbers or calculations.
• Do not modify the reorder quantity or forecast values.
• If any value is missing, state that the information is unavailable.

SYSTEM DECISION LOGIC:
Reorder Point = Forecast Demand + Safety Stock
Recommended Order Quantity = max(0, Reorder Point - Current Stock)
Safety Stock is derived from forecast uncertainty.

TASK:
Using only the provided data, explain the inventory decision.
Your explanation must include:
1. Demand Interpretation: Explain the forecast demand and inventory situation.
2. Safety Stock Explanation: Explain how safety stock protects against uncertainty.
3. Reorder Logic: Explain how the reorder point and order quantity were determined.
4. Operational Advice: Provide practical guidance for the supermarket manager.

Limit the response to 4-6 concise sentences.
Use professional retail operations language.
Before generating advice, verify that the calculations in the data section are consistent.
Do not change any values."""

        user_prompt = f"""INVENTORY ANALYTICS DATA

Product: {summary.get('product_name')}
Category: {summary.get('category')}

Forecast Demand (next 7 days): {summary.get('forecast_total')}
Current Stock: {summary.get('current_stock')}

Safety Stock: {summary.get('safety_stock')}
Reorder Point: {summary.get('reorder_point')}
Recommended Order Quantity: {summary.get('reorder_quantity')}

Procurement Cost: ₹{summary.get('procurement_cost', 0):.2f}
Risk Level: {summary.get('risk_level')}
"""
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            messages = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "Understood. I will strictly follow the grounding rules and system logic while analyzing the provided data."}]},
                {"role": "user", "parts": [{"text": user_prompt}]}
            ]
            
            response = model.generate_content(messages)
            if response and response.text:
                explanation = response.text.strip()
                with self.lock:
                    self.cache[key] = explanation
                    self._save_cache()
                return explanation
        except Exception as e:
            return f"⚠️ LLM Generation Error: {str(e)}"
            
        return "⚠️ System error generating AI advice."

# Singleton instance exported for use in the app
ai_service = AIService()
