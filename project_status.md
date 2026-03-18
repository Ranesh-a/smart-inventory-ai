# Project Status: Smart-Inventory (AI Retail Inventory Decision Intelligence Platform)

> **Generated:** March 9, 2026
> **Purpose:** Comprehensive final-year AI&DS engineering project status report detailing workflow, architecture, and milestones.

---

## 1. Project Overview

**Smart-Inventory** is an AI-powered Demand Forecasting and Decision Intelligence platform designed for retail inventory management. The system leverages deep learning (LSTM) to predict future product demand, Explainable AI (XAI) to ensure predictions are interpretable by non-technical stakeholders, and Generative AI (Google Gemini) to provide a conversational, data-grounded business advisor.

**Core Objectives Solved:**
- Eliminates manual inventory guessing by predicting demand dynamically using trailing 30-day multivariate features.
- Converts black-box LSTM predictions into understandable business insights (feature importance).
- Employs an LLM to automatically generate actionable procurement advice, highlight stockout risks, and answer context-aware follow-up questions for individual product items.
- Provides a robust "What-If" scenario simulator for proactive supply-chain planning.

---

## 2. Current Tech Stack

The platform is structured using a decoupled, high-performance tech stack:

- **Machine Learning (Forecasting):** PyTorch (Multivariate LSTM networks, 1 layer, 50 hidden size)
- **Data Processing & Scaling:** Pandas, NumPy, Scikit-Learn (`MinMaxScaler`)
- **Backend Orchestration:** FastAPI, Uvicorn (ASGI)
- **Frontend / UI Layer:** Streamlit, Matplotlib, Plotly
- **Conversational AI & Reasoning:** Google Gemini 2.5 Flash SDK (`google-generativeai`)
- **Alerting & Notifications:** Gmail SMTP Automation (`python-dotenv` for credential injection)
- **Data Persistence:** SQLite 3 (Audit trail for predictions, chat logs, and SHAP-style features)
- **Environment Management:** `dotenv` (`.env` file) for securing API keys and SMTP credentials.

---

## 3. Architecture & Data Flow

The architecture operates as a modern 5-layer SaaS product:

1. **Presentation Layer (`app.py`, `ui/`)**: The Streamlit interface collects dataset uploads and user interactions (What-If sliders, Chat inputs). It operates independently of PyTorch.
2. **Application Layer (`api.py`, `routing/`, `services/`)**: FastAPI orchestrates category routing, validates payloads, and serves as the REST bridge.
3. **Model Layer (`models/`, `scalers/`)**: PyTorch LSTM networks and scalers are lazy-loaded dynamically based on retail category (Grocery, Household, Hobbies). Employs an iterative multi-step forecasting engine.
4. **Analytics Layer (`decision/`, `analytics/`)**: A deterministic engine that calculates safety stock, reorder points, required procurement quantities, and economic holding/penalty costs.
5. **AI Reasoning Layer (`ai/`)**: Generates an LLM-driven Decision Brief summarizing operations, identifying risks, and powering the **Conversational Analytics Assistant**, which grounds LLM outputs strictly to the analytical data to prevent hallucination.

**End-to-end Data Flow:**
`User Uploads CSV` → `Dataset Validation` → `Category Router` → `Forecast Service (LSTM)` → `Inventory Decision Engine` → `Procurement Recommendation` → `Operations Analytics` → `AI Decision Advisor (Gemini)` → `Streamlit Dashboard Rendering`

---

## 4. Directory Structure

A high-level view of the decoupled, service-oriented structure:

```text
project inventory/
├── .env                          # Secure config for API Keys & SMTP
├── api.py                        # FastAPI Backend (REST endpoints for inference)
├── app.py                        # Streamlit Frontend Menu & Dashboard Component
├── requirements.txt              # Core project dependencies (Updated with dotenv/fastapi)
├── project_logs.db               # SQLite database logging user sessions and predictions
├── ai/
│   ├── ai_service.py             # Asynchronous LLM connection caching & batching
│   ├── conversation_assistant.py # Stateful ChatBot history manager (truncates memory)
│   └── decision_advisor.py       # Base prompt grounding templates
├── alerts/
│   └── email_alerts.py           # Automated stockout email dispatch logic
├── analytics/
│   ├── decision_brief.py         # AI operations summary text builder
│   └── operations_dashboard.py   # High-level budget and risk mathematical aggregators
├── decision/
│   ├── procurement_engine.py     # Cost calculations (Holding/Stockout)
│   └── reorder_engine.py         # Safety stock and reorder point boundary math
├── docs/
│   └── system_architecture.md    # Master SaaS architectural documentation
├── models/                       # Stored *.pth PyTorch weights for multiple categories
├── scalers/                      # Stored *.pkl Scikit-Learn data normalizers
├── routing/                      
│   └── category_router.py        # Maps CSV generic products to specific ML pipelines
├── services/                     
│   └── forecast_service.py       # PyTorch tensor initialization and inference loops
└── ui/                           # Segregated interface components (Future expansion)
```

---

## 5. Completed Milestones

- **Core ML Pipeline:** Multi-category LSTM inference active for Grocery, Household, and Hobbies models.
- **REST Backend:** Fully functional decoupled FastAPI backend serving predictions without locking the UI.
- **Explainable AI (XAI):** Feature importance variance maps implemented and visualized in the UI.
- **AI Decision Advisor:** Integrated Gemini-based recommendations explaining *why* a reorder quantity was drafted based on mathematical boundaries.
- **Conversational Assistant:** Added an interactive, context-aware chatbot for each product to answer follow-up questions interactively.
- **Automated Alerts:** Gmail-based automated warning dispatches securely pulling from `.env`.
- **SaaS Architecture Migration:** Formally restructured codebase out of a monolith into strict analytical service layers.
- **Database Auditing:** SQLite integration successfully preserving historical states.

---

## 6. Current Status & Blockers

**Status:** The system is currently in a stable, functional V1 state with a fully decoupled AI front-end and a PyTorch backend. The conversational AI and automated alerts have been successfully implemented and tested locally.

**Known Blockers & Technical Debt:**
1. **Explainability Proxy Limitations:** The system currently relies on a variance-based proxy for Feature Importance (`compute_simple_feature_importance()`) rather than true Gradient/Deep SHAP values, limiting the absolute precision of complex interaction explainability.
2. **Single-Item Training Skew:** Current category models (`model_food.pth`, etc.) were trained on the highest-sales item as a proxy for the entire category. This limits predictive accuracy on low-volume, high-variance items.
3. **What-If Constraint:** The What-If simulation relies on a flat percentage multiplier array applied post-inference, rather than altering the core input sequence and re-running the tensor.

---

## 7. Immediate Next Steps

1. **Multivariate What-If Simulation:** Upgrade the What-If slider to support event-based simulation (e.g., dynamically altering pricing, boolean weekend logic, or sudden promotion triggers) within the 30-day tensor *before* forecasting.
2. **True SHAP Integration:** Implement `shap.DeepExplainer` on the LSTM to replace the variance proxy for more rigorous academic compliance in XAI metrics.
3. **Dynamic ML Accuracy Reporting:** Pipe actual Test MSE/MAE dynamic loss values generated during model training directly into the Streamlit UI, replacing current hardcoded placeholders.
4. **Deployment & Containerization:** Write `Dockerfile` configurations and separate `docker-compose.yml` specs for the FastAPI app and the Streamlit frontend to prepare the system for cloud hosting.
