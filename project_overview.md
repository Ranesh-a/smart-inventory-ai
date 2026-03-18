# Smart-Inventory: Project Overview

## 1. Core System Architecture
The Smart-Inventory system follows a decoupled, two-tier architecture tailored for decision intelligence:
- **Frontend (UI)**: A Streamlit-based web application (`app.py`) serves as the central hub for users (e.g., supermarket managers). It handles dataset uploads, parameter configuration (e.g., stockout penalties, holding costs, forecast horizons), and visualizing metrics. The frontend abstracts away machine learning execution; it relies entirely on the backend for inference.
- **Backend (API)**: A FastAPI service (`api.py`) exposing REST endpoints. It manages PyTorch LSTM neural network models and `scikit-learn` scalers. Models are lazy-loaded dynamically into memory based on the requested product category to process time-series predictions.
- **Data Persistence**: A local SQLite database (`project_logs.db`) operates as an audit log capturing prediction events, model feature importance maps, and AI chat histories.

## 2. Current Tech Stack
- **Frontend**: Streamlit, pandas (data manipulation), Plotly / Matplotlib (charting)
- **Backend (API)**: FastAPI, Uvicorn (ASGI server)
- **Machine Learning**: PyTorch (LSTM networks, size 50 hidden, 1 layer), scikit-learn (MinMaxScaler)
- **Database**: SQLite 3
- **Conversational AI Module**: Python dict/string formatting (Offline Business Advisor), Google Gemini 2.5 Flash SDK (legacy/optional fallback API)
- **Notifications**: Native SMTP via Gmail API (`alerts/email_alerts.py`)

## 3. Data Models & AI Outputs
- **Demand Forecasting Model**: Uses an iterative multi-step forecasting mechanism leveraging an LSTM. The input is a historical sales window (typically 30 days trailing). The output yields a sequence spanning `forecast_days` (1-30 days ahead) returning numerical predicted sales volumes, alongside uncertainty boundaries (`lower_bound` and `upper_bound`).
- **Explainable AI (XAI) Output**: The frontend computes feature importance reflecting the relative influence of preceding days/features on the forecast. These are represented as contiguous sequences of influence scores mapped back to the 30-day lookback window. (Variance-based proxy implemented, with a fallback available for true tree/gradient SHAP scores).
- **Data Partitions & Scalers**: Multi-categorical structuring (e.g., Food, Hobbies, Household). Each branch operates a dedicated neural network checkpoint (`.pth`) and scaler object (`.pkl`) reflecting differing domain distributions.

## 4. Conversational Interface (Business Advisor)
The conversational decision-support infrastructure has transitioned to emphasize speed and autonomy:
- While previously leveraging full prompt engineering via Google Gemini, the **Current State** utilizes a rule-based deterministic summary engine (`advisor/business_advisor.py`). 
- **Generation mechanism**: Post-inference, the backend aggregates analytical operations (such as projected total cost and risk thresholds across products) and pipes them into function `generate_business_insights()`. This yields natural-language summaries packaged into five categories: `executive_summary`, `demand_insight`, `category_insight`, `procurement_insight`, and `inventory_insight`.
- **Planned Evolution**: Future integration in the frontend may reinstate an interactive NLP chatbot (`ChatBox`) where users can filter the UI components by typing contextual questions to the model over the retrieved JSON.

## 5. API Endpoints
The frontend primarily orchestrates processes across these REST routes:
- `GET /health`
  - Validates liveness, checks if PyTorch leverages CUDA/CPU, and validates accessible model configurations.
- `GET /items/{category}`
  - Returns a list of all unique item IDs present in the memory-cached datasets for a distinct category classification.
- `POST /predict`
  - Payload: `{category, item_id, forecast_days}`
  - Action: Runs inference on integrated static datasets extracting arrays of predictions combined with upper/lower bounds.
- `POST /predict_custom`
  - Payload: Univariate (`sales_history` array) or Multivariate (`feature_rows` containing `sales, price, weekday, month, is_weekend, is_event_day`).
  - Action: Exposes the primary predictive engine to arbitrary user-supplied CSV files uploaded in the UI. Predicts the specified `forecast_days` based on contextual data sequences without relying on pre-cached dataset strings.
