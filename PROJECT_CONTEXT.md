# 🛒 Smart-Inventory: Project Context & Handover Document

> **Last updated:** 2026-02-13  
> **Purpose:** Enable another AI agent (or developer) to pick up this project with full context.

---

## 1. Project Goal

**Smart-Inventory** is an AI-powered **Demand Forecasting** system for retail inventory management, built with **Explainable AI (XAI)** principles. The core objective is:

- Predict future product demand using **LSTM neural networks** trained on historical sales data.
- Provide **transparent, explainable predictions** (feature importance / SHAP-style analysis) so non-technical store managers can understand *why* the model made a prediction.
- Enable **scenario planning** via a "What-If" demand simulator.
- Automate **reorder recommendations** with AI-drafted supplier emails.
- Offer an **AI chatbot** (Google Gemini-powered, with offline fallback) for conversational Q&A about forecasts.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)              │
│  - Category & Item Selection                        │
│  - Demand Forecast Visualization                    │
│  - What-If Simulator (% demand slider)              │
│  - XAI / Feature Importance Charts                  │
│  - Gemini AI Chatbot (with offline fallback)        │
│  - Automated Reorder Email Drafting (Gmail SMTP)    │
│  - Admin Panel (SQLite Audit Trail)                 │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (REST)
                       ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend (api.py)                │
│  - POST /predict → multi-day LSTM forecast          │
│  - GET  /health  → liveness check                   │
│  - GET  /items/{category} → list available items    │
│  - Lazy-loads models/scalers per category           │
│  Runs on http://localhost:8000                      │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   model_food.pth  model_hobby.pth  model_household.pth
   scaler_food.pkl scaler_hobby.pkl scaler_household.pkl
```

**Key design decision:** The Streamlit frontend contains **zero PyTorch code**. All model inference is handled by the FastAPI backend. The frontend calls the API and applies any What-If adjustments client-side.

---

## 3. Tech Stack

| Layer         | Technology                          | Notes                                     |
|---------------|-------------------------------------|-------------------------------------------|
| **ML Model**  | PyTorch LSTM (`src/model.py`)       | `hidden_size=50`, `num_layers=1`, univariate |
| **XAI**       | SHAP-style feature importance       | Variance-based proxy in frontend; gradient-based fallback available |
| **Training**  | `train.py`                         | Per-category training, MinMaxScaler, sliding-window dataset |
| **Backend**   | FastAPI + Uvicorn (`api.py`)        | REST API for inference; runs on port 8000 |
| **Frontend**  | Streamlit (`app.py`)                | Wide layout, sidebar config, Plotly/Matplotlib charts |
| **Database**  | SQLite (`project_logs.db`)          | Prediction logs, SHAP values (JSON), chat history |
| **AI Chat**   | Google Gemini 2.5 Flash             | Via `google-generativeai` SDK; offline `generate_narrative()` fallback |
| **Email**     | Gmail SMTP (SSL)                    | Auto-drafted reorder emails via Gemini |
| **Scaling**   | scikit-learn `MinMaxScaler`         | One scaler per category, saved as `.pkl` |
| **Language**  | Python 3.11                         | Windows development environment |

### Dependencies (`requirements.txt`)

```
pandas
numpy
torch
matplotlib
scikit-learn
shap
streamlit
plotly
joblib
```

> **Note:** `fastapi`, `uvicorn`, `requests`, and `google-generativeai` are also required but are not currently listed in `requirements.txt`. This should be fixed.

---

## 4. Dataset Details — M5 Walmart

The project uses the **M5 Forecasting Competition** dataset from Walmart, containing daily unit sales across ~3,000 products over ~1,941 days.

| File                    | Category    | Format     | Approx Size |
|-------------------------|-------------|------------|-------------|
| `project_data.csv`      | **Food**    | Long format (`item_id`, `d`, `sales`) | ~109 MB |
| `subset_hobbies.csv`    | **Hobbies** | Wide format (`d_1`, `d_2`, …, `d_1941`) | ~21 MB |
| `subset_household.csv`  | **Household** | Wide format (`d_1`, `d_2`, …, `d_1941`) | ~40 MB |

**Data handling:** Both formats are supported. Wide-format CSVs are melted into long format at load time (see `load_csv_to_long()` in `train.py` and `load_data_cached()` in `app.py`).

---

## 5. Model Details

### 5.1 LSTM Architecture (`src/model.py` → `LSTMModel`)

```
Input (batch, 30, 1)
    → LSTM(input=1, hidden=50, layers=1, batch_first=True)
    → Take last timestep output (batch, 50)
    → Linear(50 → 1)
Output (batch, 1)
```

- **Sequence length:** 30 days (look-back window)
- **Prediction horizon:** 1 day (iterative multi-step via sliding window in `api.py`)
- **Total parameters:** ~10,451

### 5.2 Training Configuration (`train.py`)

| Hyperparameter   | Value  |
|-------------------|--------|
| Sequence Length   | 30     |
| Batch Size        | 64     |
| Epochs            | 20     |
| Learning Rate     | 0.001  |
| Optimizer         | Adam   |
| Loss Function     | MSE    |
| Train/Test Split  | 80/20 (sequential, not random — correct for time series) |

### 5.3 Training Strategy

- **One model per category** — separate `model_food.pth`, `model_hobby.pth`, `model_household.pth`.
- **Item selection for training:**
  - Food: Hardcoded to `FOODS_3_002`.
  - Hobby & Household: Auto-selected as the item with highest total sales.
- **Scaler:** One `MinMaxScaler` per category, fitted on the selected item's sales data.
- **Checkpoint format:** Each `.pth` file is a dict with `model_state_dict`, `optimizer_state_dict`, `train_loss`, `test_loss`, and `epoch`.

### 5.4 Current Performance Metrics

| Category    | Best Test Loss (MSE) | MAE (displayed) | RMSE (displayed) |
|-------------|----------------------|------------------|-------------------|
| Food        | ~0.004               | 0.06520          | 0.08595           |
| Hobby       | ~0.003               | 0.05410          | 0.07595           |
| Household   | ~0.003               | 0.05410          | 0.07595           |

> **Note:** The MAE/RMSE values shown in the sidebar are currently **hardcoded** in `app.py` (lines 405–409), not dynamically computed. The Hobby and Household values are identical placeholders.

---

## 6. Current Progress & Feature Status

### ✅ Completed

| Feature | File(s) | Status |
|---------|---------|--------|
| LSTM model definition | `src/model.py` | ✅ Done |
| Data pipeline (SlidingWindowDataset, SalesDataProcessor) | `src/dataset.py` | ✅ Done |
| Multi-category training script | `train.py` | ✅ Done |
| 3 trained models (Food, Hobby, Household) | `models/*.pth` | ✅ Done |
| FastAPI inference backend | `api.py` | ✅ Done |
| Streamlit frontend (no local PyTorch) | `app.py` | ✅ Done |
| What-If demand simulator (±50% slider) | `app.py` (lines 355–361) | ✅ Done |
| XAI — feature importance bar charts | `app.py` (line 661+) | ✅ Done |
| AI narrative generation (SHAP-based) | `app.py` → `generate_narrative()` | ✅ Done |
| Gemini AI chatbot (with offline fallback) | `app.py` (line 697+) | ✅ Done |
| Automated reorder email (Gemini-drafted, Gmail SMTP) | `app.py` (lines 508–613) | ✅ Done |
| SQLite audit trail (predictions, SHAP, chat) | `db_manager.py` | ✅ Done |
| Admin panel (log viewer + CSV download) | `app.py` (lines 831–876) | ✅ Done |
| System health check script | `check_setup.py` | ✅ Done |
| Training history plots | `models/training_history_*.png` | ✅ Done |
| Category-specific scalers | `models/scaler_*.pkl` | ✅ Done |
| Stock level alerts (critical/low/healthy) | `app.py` (lines 476–506) | ✅ Done |
| Inventory days-remaining calculation | `app.py` (lines 484–487) | ✅ Done |

### ⚠️ Known Issues / Gaps

1. **`requirements.txt` is incomplete** — Missing `fastapi`, `uvicorn`, `requests`, `google-generativeai`.
2. **Hardcoded accuracy metrics** — MAE/RMSE in sidebar are hardcoded, not dynamically read from model checkpoints.
3. **Single-item training** — Each category model is only trained on one item (the highest-sales item), then used to predict for *all* items in that category. This limits accuracy for low-volume items.
4. **Hobby & Household metrics are identical** — Placeholder values, not actual test results.
5. **SHAP is listed in `requirements.txt`** but not actively used — replaced by a variance-based proxy (`compute_simple_feature_importance()`).
6. **Gmail credentials** are stored in `.streamlit/secrets.toml` (sensitive — should not be committed to version control).

---

## 7. File Map

```
project inventory/
├── app.py                          # Streamlit frontend (879 lines)
├── api.py                          # FastAPI backend for LSTM inference (265 lines)
├── train.py                        # Multi-category training script (375 lines)
├── check_setup.py                  # System readiness checker (65 lines)
├── db_manager.py                   # SQLite database manager (220 lines)
├── requirements.txt                # Python dependencies (incomplete)
├── project_data.csv                # M5 Food category data (~109 MB, long format)
├── subset_hobbies.csv              # M5 Hobbies subset (~21 MB, wide format)
├── subset_household.csv            # M5 Household subset (~40 MB, wide format)
├── project_logs.db                 # SQLite database — prediction & chat logs
├── scaler.pkl                      # Legacy scaler (pre-refactor, may be unused)
├── training_history.png            # Legacy training plot (pre-refactor)
├── .streamlit/
│   └── secrets.toml                # Gmail SMTP credentials (GMAIL_USER, GMAIL_PASS)
├── src/
│   ├── __init__.py                 # Package init
│   ├── model.py                    # LSTMModel class + get_device()
│   └── dataset.py                  # SalesDataProcessor + SlidingWindowDataset
└── models/
    ├── model_food.pth              # Trained LSTM — Food category
    ├── model_hobby.pth             # Trained LSTM — Hobby category
    ├── model_household.pth         # Trained LSTM — Household category
    ├── lstm_model.pth              # Legacy single model (pre-refactor)
    ├── scaler_food.pkl             # MinMaxScaler — Food
    ├── scaler_hobby.pkl            # MinMaxScaler — Hobby
    ├── scaler_household.pkl        # MinMaxScaler — Household
    ├── training_history_food.png   # Loss curve — Food
    ├── training_history_hobby.png  # Loss curve — Hobby
    └── training_history_household.png  # Loss curve — Household
```

---

## 8. How to Run

### Prerequisites
- Python 3.11+ with `venv`
- NVIDIA GPU optional (auto-detects CUDA, falls back to CPU)

### Steps

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn requests google-generativeai

# 3. (Optional) Train models from scratch
python train.py

# 4. Start the FastAPI backend (Terminal 1)
python api.py
# → Runs on http://localhost:8000

# 5. Start the Streamlit frontend (Terminal 2)
streamlit run app.py
# → Opens on http://localhost:8501
```

### Configuration
- **Gemini API key:** Enter in the Streamlit sidebar at runtime (starts with `AIza…`).
- **Gmail SMTP:** Configure in `.streamlit/secrets.toml`:
  ```toml
  GMAIL_USER = "you@gmail.com"
  GMAIL_PASS = "your-app-password"
  ```

---

## 9. Planned Features & Roadmap

### 🔲 What-If Simulator V2 (Enhanced)
The current slider applies a flat percentage adjustment to predictions. Planned enhancements:
- **Event-based simulation:** Model the impact of promotions, holidays, price changes on demand.
- **Multi-variable What-If:** Adjust price, weather, competitor activity independently.
- **Monte Carlo simulation:** Generate probabilistic demand ranges, not just point estimates.

### 🔲 Multi-Item Training
Currently each category model is trained on a single representative item. Future work:
- Train on **all items per category** (or a representative sample).
- Use **item embeddings** so one model generalises across products.

### 🔲 True SHAP Integration
Replace the variance-based proxy with actual SHAP KernelExplainer or DeepExplainer for genuine model-level explainability.

### 🔲 Dynamic Accuracy Metrics
Read test loss from model checkpoints and compute MAE/RMSE on test data at load time, instead of using hardcoded values.

### 🔲 Deployment
- Dockerize the two-service architecture (FastAPI + Streamlit).
- Add environment variable support for secrets.
- CI/CD pipeline for model retraining.

### 🔲 Additional Categories
Expand beyond Food, Hobbies, and Household to cover all M5 categories.

---

## 10. Key Design Decisions (for continuing agent)

1. **Frontend ↔ Backend separation:** `app.py` has zero PyTorch imports. All model logic is in `api.py`. This was a deliberate refactor to keep the Streamlit app lightweight and independently deployable.

2. **Lazy loading in API:** Models and scalers are loaded on first request per category (`_ensure_loaded()`), not at startup. They remain in RAM forever once loaded.

3. **Sequential time-series split:** The train/test split is chronological (first 80% train, last 20% test), not random — critical for time-series integrity.

4. **Iterative multi-step forecasting:** The API predicts one day ahead, appends the prediction to the input window, and repeats for `forecast_days` steps. This is simpler but can accumulate error over longer horizons.

5. **Offline-first chatbot:** The `generate_narrative()` function provides a fully functional, non-AI explanation even without a Gemini API key. Gemini is an enhancement, not a dependency.

6. **Category configuration is dict-driven:** Both `app.py` and `api.py` use `CATEGORY_CONFIG` / `CATEGORY_MAP` dicts. Adding a new category only requires adding an entry to these dicts plus the corresponding model/scaler/data files.

---

## 11. Database Schema (`project_logs.db`)

```sql
CREATE TABLE logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,      -- ISO 8601 timestamp
    item_id           TEXT,               -- e.g. "FOODS_3_002"
    prediction        REAL,               -- predicted sales value
    shap_values_json  TEXT,               -- JSON array of 30 importance scores
    user_query        TEXT                -- chat message + bot response (truncated to 500 chars)
);
```

---

*This document was auto-generated from the project source code to facilitate handover.*
